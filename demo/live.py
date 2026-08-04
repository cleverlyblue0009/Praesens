"""Milestone 6: live dashboard.

The demo runs on one laptop screen, projected to the panel, so this window
IS the emitter's fullscreen window: the light-pattern border keeps lighting
the subject's face while the dashboard renders into the centre "preview"
slot Milestone 2 built for exactly this. Everything -- capture, scoring,
drawing, and reading keyboard shortcuts -- runs in a single loop on one
thread, because OpenCV's GUI calls (imshow/waitKey) aren't safe to split
across threads for the same window, and this window needs waitKey() both to
actually display frames and to catch [E]/[R]/[S]/[Q]. That's why the
emitter is driven manually here (render_frame + log_redraw) instead of via
its own start()/_run() background thread, which session.py uses instead
where there's no competing waitKey() loop.

The score shown is NOT the batch score from a fixed 20 s session
(Milestone 4) -- it's recomputed every frame over a short rolling window
(default 6 s) so the verdict updates live as conditions change (a face
appearing, the emitter being toggled, a feed being swapped). This trades
some statistical power for immediacy, which is the right trade for a demo
where the audience needs to see the collapse happen, not wait 20 seconds
for a verdict.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from mediapipe.tasks.python import vision as mp_vision

from praesens.challenge import Challenge
from praesens.emit import Emitter, EmitterConfig
from praesens.optical import (
    OpticalConfig, create_landmarker, lock_camera, CadenceDetector,
    moving_average_detrend, cross_correlate_lag_search, estimate_snr_db,
    resample_emitted, _zscore,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

GREEN = (60, 180, 60)    # BGR
RED = (40, 40, 200)
GRAY = (100, 100, 100)
WHITE = (255, 255, 255)
LIGHT_GRAY = (210, 210, 210)
BLUE = (220, 150, 60)
ORANGE = (40, 130, 240)


def _draw_dashed_hline(img, y, x0, x1, color, thickness=2, dash_len=10, gap_len=8):
    x = x0
    while x < x1:
        x_end = min(x + dash_len, x1)
        cv2.line(img, (int(x), int(y)), (int(x_end), int(y)), color, thickness)
        x += dash_len + gap_len


def _plot_series(img, rect, xs, ys, y_range, color, thickness=2):
    x0, y0, x1, y1 = rect
    xs, ys = np.asarray(xs), np.asarray(ys)
    if len(xs) < 2:
        return
    t0, t1 = xs[0], xs[-1]
    if t1 <= t0:
        return
    ylo, yhi = y_range
    yspan = max(yhi - ylo, 1e-9)
    px = x0 + (xs - t0) / (t1 - t0) * (x1 - x0)
    py = y1 - (np.clip(ys, ylo, yhi) - ylo) / yspan * (y1 - y0)
    pts = np.stack([px, py], axis=1).astype(np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


@dataclass
class DemoConfig:
    rolling_window_s: float = 6.0
    score_history_s: float = 30.0
    score_threshold: float = 0.5
    min_valid_samples: int = 8
    dashboard_width: int = 1280
    dashboard_height: int = 800
    border_fraction: float = 0.14
    min_fps: float = 15.0
    screenshot_dir: str = "demo/screenshots"
    camera_display_name: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "DemoConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LiveDashboard:
    def __init__(self, raw_config: dict):
        self.dconfig = DemoConfig.from_dict(raw_config["demo"])

        # Loops indefinitely: unlike a scored Milestone 4 session, a demo
        # isn't a fixed 20s window.
        self.challenge = Challenge(chip_rate_hz=raw_config["challenge"]["chip_rate_hz"], loop=True)

        econfig = EmitterConfig.from_dict(raw_config["emitter"])
        econfig.border_fraction = self.dconfig.border_fraction
        self.emitter = Emitter(self.challenge, econfig)

        self.oconfig = OpticalConfig.from_dict(raw_config["optical"])
        model_path = Path(self.oconfig.model_path)
        self.oconfig.model_path = str(model_path if model_path.is_absolute() else REPO_ROOT / model_path)

        self.cap = cv2.VideoCapture(self.oconfig.camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera index {self.oconfig.camera_index}")

        self._camera_name = self.dconfig.camera_display_name or f"camera #{self.oconfig.camera_index}"

        self.landmarker = create_landmarker(self.oconfig.model_path, mp_vision.RunningMode.VIDEO,
                                             self.oconfig.min_face_confidence)
        self.detector = CadenceDetector(self.landmarker, self.oconfig.detect_every_n_frames,
                                         self.oconfig.detect_downscale_width)

        self.warn_list: list = []
        self.exposure_locked = lock_camera(self.cap, self.oconfig, self.warn_list)
        for w in self.warn_list:
            print(f"WARNING: {w}")

        self.timestamps: deque = deque()
        self.luminances: deque = deque()
        self.face_flags: deque = deque()
        self.score_history: deque = deque()
        self.frame_times: deque = deque(maxlen=30)

        self.last_ts_ms = -1
        self.current_score = 0.0
        self.current_lag_ms = 0.0
        self.current_snr_db = float("nan")
        self.face_recently_detected = False

        self.screenshot_dir = REPO_ROOT / self.dconfig.screenshot_dir
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self.start_time = time.perf_counter()
        self.emitter.begin_manual_drive(self.start_time)

    # -- data pipeline ----------------------------------------------------

    def _trim_buffers(self, now: float) -> None:
        window_start = now - self.dconfig.rolling_window_s
        while self.timestamps and self.timestamps[0] < window_start:
            self.timestamps.popleft()
            self.luminances.popleft()
            self.face_flags.popleft()
        hist_start = now - self.dconfig.score_history_s
        while self.score_history and self.score_history[0][0] < hist_start:
            self.score_history.popleft()

    def _process_frame(self) -> bool:
        grabbed = self.cap.grab()
        t = time.perf_counter()
        if not grabbed:
            return False
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return False

        elapsed = t - self.start_time
        ts_ms = max(self.last_ts_ms + 1, int(elapsed * 1000))
        self.last_ts_ms = ts_ms

        lum, face_ok = self.detector.process(frame, ts_ms, self.oconfig.roi_margin_frac)

        self.timestamps.append(t)
        self.luminances.append(lum)
        self.face_flags.append(face_ok)
        self._trim_buffers(t)

        self.face_recently_detected = any(list(self.face_flags)[-5:])

        ts_arr = np.array(self.timestamps)
        lum_arr = np.array(self.luminances)
        n_valid = int(np.sum(~np.isnan(lum_arr)))

        if n_valid >= self.dconfig.min_valid_samples:
            detrended = moving_average_detrend(ts_arr, lum_arr, self.oconfig.detrend_window_s)
            emitter_log = self.emitter.get_log()
            score, lag_ms, *_ = cross_correlate_lag_search(
                detrended, ts_arr, emitter_log, self.oconfig.lag_search_max_ms, self.oconfig.lag_step_ms
            )
            snr_db = estimate_snr_db(ts_arr, detrended, self.challenge.chip_rate_hz,
                                      self.oconfig.snr_low_cut_hz, self.oconfig.snr_resample_rate_hz)
            self.current_score, self.current_lag_ms, self.current_snr_db = score, lag_ms, snr_db
            self.score_history.append((t, score))
        else:
            self.current_score, self.current_lag_ms, self.current_snr_db = 0.0, 0.0, float("nan")

        return True

    def _fps(self) -> float:
        if len(self.frame_times) < 2:
            return 0.0
        return (len(self.frame_times) - 1) / (self.frame_times[-1] - self.frame_times[0])

    # -- rendering ----------------------------------------------------------

    def _render_dashboard(self):
        w, h = self.dconfig.dashboard_width, self.dconfig.dashboard_height
        canvas = np.full((h, w, 3), (15, 15, 15), dtype=np.uint8)

        banner_h = int(h * 0.25)
        trace_h = int(h * 0.35)
        strip_h = int(h * 0.25)
        status_h = h - banner_h - trace_h - strip_h

        y = 0
        y = self._draw_banner(canvas, 0, y, w, banner_h)
        y = self._draw_trace_panel(canvas, 0, y, w, trace_h)
        y = self._draw_score_strip(canvas, 0, y, w, strip_h)
        self._draw_status_row(canvas, 0, y, w, status_h)
        return canvas

    def _draw_banner(self, canvas, x, y, w, h):
        if not self.face_recently_detected:
            color, title, reason = GRAY, "NO FACE DETECTED", "waiting for a face in frame"
        elif self.current_score >= self.dconfig.score_threshold:
            color = GREEN
            title = "LIVE HUMAN PRESENT"
            reason = f"screen pattern reflected on face (r={self.current_score:.2f})"
        else:
            color = RED
            title = "INJECTED FEED DETECTED"
            if not np.isnan(self.current_snr_db) and self.current_snr_db < self.oconfig.snr_floor_db:
                reason = f"insufficient signal to measure (snr={self.current_snr_db:.1f}dB) -- move to better light"
            else:
                reason = f"screen pattern not reflected on face (r={self.current_score:.2f})"

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)

        title_scale = h / 130.0
        (tw, _th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, 4)
        cv2.putText(canvas, title, (x + (w - tw) // 2, y + int(h * 0.55)),
                    cv2.FONT_HERSHEY_DUPLEX, title_scale, WHITE, 4, cv2.LINE_AA)

        reason_scale = h / 400.0
        (rw, _rh), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, reason_scale, 2)
        cv2.putText(canvas, reason, (x + (w - rw) // 2, y + int(h * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, reason_scale, WHITE, 2, cv2.LINE_AA)
        return y + h

    def _draw_trace_panel(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (32, 32, 32), -1)
        pad = 30
        rect = (x + pad, y + pad, x + w - pad, y + h - pad)

        ts_arr = np.array(self.timestamps)
        lum_arr = np.array(self.luminances)
        if len(ts_arr) >= 2:
            detrended = moving_average_detrend(ts_arr, lum_arr, self.oconfig.detrend_window_s)
            emitter_log = self.emitter.get_log()
            emitted = resample_emitted(emitter_log, ts_arr, lag_s=self.current_lag_ms / 1000.0)
            valid = ~np.isnan(detrended)
            if valid.sum() >= 2:
                _plot_series(canvas, rect, ts_arr[valid], _zscore(emitted[valid]), (-3, 3), BLUE, 3)
                _plot_series(canvas, rect, ts_arr[valid], _zscore(detrended[valid]), (-3, 3), ORANGE, 3)

        lx, ly = rect[0] + 10, rect[1] + 20
        cv2.line(canvas, (lx, ly), (lx + 30, ly), BLUE, 4)
        cv2.putText(canvas, "emitted", (lx + 40, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        cv2.line(canvas, (lx + 170, ly), (lx + 200, ly), ORANGE, 4)
        cv2.putText(canvas, "measured", (lx + 210, ly + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA)
        return y + h

    def _draw_score_strip(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (26, 26, 26), -1)
        pad = 30
        rx0, ry0, rx1, ry1 = x + pad, y + pad, x + w - pad, y + h - pad
        thr = self.dconfig.score_threshold

        thr_y = int(ry1 - (thr - (-1)) / 2.0 * (ry1 - ry0))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (rx0, thr_y), (rx1, ry1), (40, 40, 150), -1)
        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, dst=canvas)
        _draw_dashed_hline(canvas, thr_y, rx0, rx1, LIGHT_GRAY, 2)

        if self.score_history:
            xs = [t for t, _ in self.score_history]
            ys = [s for _, s in self.score_history]
            _plot_series(canvas, (rx0, ry0, rx1, ry1), xs, ys, (-1, 1), WHITE, 3)

        cv2.putText(canvas, f"threshold={thr:.2f}", (rx0, ry0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, LIGHT_GRAY, 1, cv2.LINE_AA)
        return y + h

    def _draw_status_row(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (12, 12, 12), -1)
        text = (f"FPS: {self._fps():.1f}   lag: {self.current_lag_ms:.0f}ms   "
                f"{self._camera_name}   emitter: {'ON' if self.emitter.is_enabled() else 'OFF'}")
        cv2.putText(canvas, text, (x + 20, y + int(h * 0.65)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, LIGHT_GRAY, 1, cv2.LINE_AA)
        return y + h

    # -- controls -----------------------------------------------------------

    def reset(self) -> None:
        self.timestamps.clear()
        self.luminances.clear()
        self.face_flags.clear()
        self.score_history.clear()
        self.current_score, self.current_lag_ms, self.current_snr_db = 0.0, 0.0, float("nan")
        print("reset: rolling buffers cleared")

    def screenshot(self, frame) -> None:
        path = self.screenshot_dir / f"screenshot_{int(time.time())}.png"
        cv2.imwrite(str(path), frame)
        print(f"saved {path}")

    def _handle_key(self, key: int, frame) -> bool:
        """Returns False to stop the run loop. Subclasses (attack.py,
        panel.py) override this to add hotkeys while keeping [E]/[R]/[S]/[Q]
        by calling super()."""
        if key == ord('q'):
            return False
        elif key == ord('e'):
            self.emitter.set_enabled(not self.emitter.is_enabled())
        elif key == ord('r'):
            self.reset()
        elif key == ord('s'):
            self.screenshot(frame)
        return True

    # -- main loop -----------------------------------------------------------

    def run(self, max_seconds: float | None = None) -> None:
        cv2.namedWindow(Emitter.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(Emitter.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            while True:
                if max_seconds is not None and (time.perf_counter() - self.start_time) >= max_seconds:
                    print(f"max_seconds={max_seconds} reached, stopping (smoke-test mode). "
                          f"final FPS={self._fps():.1f}")
                    self.screenshot(self._render_dashboard())
                    break
                try:
                    self._process_frame()
                except Exception as e:
                    # never crash the demo -- a face-lost or transient
                    # decode error should degrade the display, not kill it.
                    print(f"WARNING: frame processing error (continuing): {e}")
                    self.face_recently_detected = False

                dashboard = self._render_dashboard()
                self.emitter.set_preview(dashboard)

                elapsed = time.perf_counter() - self.start_time
                frame, chip_value, luminance, enabled = self.emitter.render_frame(elapsed)
                cv2.imshow(Emitter.WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                self.emitter.log_redraw(chip_value, luminance, enabled)
                self.frame_times.append(time.perf_counter())

                if not self._handle_key(key, frame):
                    break
        finally:
            self.cap.release()
            self.landmarker.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 6 live dashboard")
    parser.add_argument("--max-seconds", type=float, default=None,
                         help="auto-stop after N seconds (smoke-test use; omit for real demo use)")
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)
    dashboard = LiveDashboard(raw)
    print("Live dashboard running. Keys: [E] toggle emitter, [R] reset, [S] screenshot, [Q] quit.")
    dashboard.run(max_seconds=args.max_seconds)
