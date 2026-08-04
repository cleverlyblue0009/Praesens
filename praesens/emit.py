"""Milestone 2: emitter.

Turns the Challenge sequence into actual light. A fullscreen window renders
the current chip value as a luminance step on a wide screen-edge border,
leaving the centre free for a live camera preview (used later by the demo) --
the border still lights the user's face from the sides even when the centre
is occupied. Every redraw logs time.perf_counter() at the moment the frame
was actually pushed to the window, not the moment it was scheduled for,
because the optical lane must align against what was truly on screen, not
an idealised schedule. `emitter_enabled=False` is the experiment's control
condition: it renders a constant mid-grey at exactly the same mean luminance
the modulated pattern would average to (since chips are a zero-mean +/-1
sequence, the ON-case time-average equals base_luminance already), so
disabling it changes only the light pattern and nothing else about the
window, loop timing, or logging -- otherwise it wouldn't isolate the one
variable the control is meant to isolate.
"""
from __future__ import annotations

import ctypes
import platform
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from praesens.challenge import Challenge


def _screen_resolution(fallback_w: int, fallback_h: int) -> tuple[int, int]:
    if platform.system() == "Windows":
        try:
            user32 = ctypes.windll.user32
            user32.SetProcessDPIAware()
            w = user32.GetSystemMetrics(0)
            h = user32.GetSystemMetrics(1)
            if w > 0 and h > 0:
                return w, h
        except Exception:
            pass
    return fallback_w, fallback_h


@dataclass
class EmitterConfig:
    emitter_enabled: bool = True
    base_luminance: float = 127.0
    modulation_depth: float = 60.0
    max_modulation_depth: float = 110.0
    border_fraction: float = 0.25
    fallback_window_width: int = 1920
    fallback_window_height: int = 1080

    @classmethod
    def from_dict(cls, d: dict) -> "EmitterConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Emitter:
    """Drives the fullscreen light-pattern window on a background thread."""

    WINDOW_NAME = "praesens_emitter"

    def __init__(self, challenge: Challenge, config: EmitterConfig):
        self.challenge = challenge
        self.config = config

        self._lock = threading.Lock()
        self._enabled = bool(config.emitter_enabled)
        self._modulation_depth = float(config.modulation_depth)
        self._adaptive_boost_applied = False

        self._log: list[dict] = []
        self._preview = None  # optional BGR np.ndarray blitted into the centre

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._start_time: float | None = None

        self._width, self._height = _screen_resolution(
            config.fallback_window_width, config.fallback_window_height
        )

    # -- runtime controls, safe to call from another thread ----------------

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_modulation_depth(self, depth: float) -> None:
        """Used by the optical lane's adaptive-depth compensation (Milestone
        3) when session SNR is too low in low-reflectance conditions."""
        depth = float(np.clip(depth, 0.0, self.config.max_modulation_depth))
        with self._lock:
            self._modulation_depth = depth
            self._adaptive_boost_applied = self._adaptive_boost_applied or (
                depth > self.config.modulation_depth
            )

    def get_modulation_depth(self) -> float:
        with self._lock:
            return self._modulation_depth

    def adaptive_boost_applied(self) -> bool:
        with self._lock:
            return self._adaptive_boost_applied

    def set_preview(self, frame) -> None:
        with self._lock:
            self._preview = frame

    # -- rendering -----------------------------------------------------------

    def render_frame(self, elapsed_s: float):
        """Pure function from elapsed session time -> BGR frame + the chip
        value/luminance that frame represents. Exposed standalone so it can
        be tested without opening a window."""
        with self._lock:
            enabled = self._enabled
            depth = self._modulation_depth
            preview = self._preview
        base = self.config.base_luminance

        if enabled:
            chip_value = int(self.challenge.value_at(elapsed_s))
            luminance = float(np.clip(base + depth * chip_value, 0, 255))
        else:
            chip_value = 0
            luminance = float(np.clip(base, 0, 255))

        frame = np.full((self._height, self._width, 3), luminance, dtype=np.uint8)

        bf = self.config.border_fraction
        bw = int(self._width * bf)
        bh = int(self._height * bf)
        if preview is not None and bw > 0 and bh > 0 and self._width - 2 * bw > 0 and self._height - 2 * bh > 0:
            ph, pw = preview.shape[:2]
            target_w, target_h = self._width - 2 * bw, self._height - 2 * bh
            scale = min(target_w / pw, target_h / ph)
            new_w, new_h = max(1, int(pw * scale)), max(1, int(ph * scale))
            resized = cv2.resize(preview, (new_w, new_h))
            y0 = bh + (target_h - new_h) // 2
            x0 = bw + (target_w - new_w) // 2
            frame[y0:y0 + new_h, x0:x0 + new_w] = resized

        return frame, chip_value, luminance, enabled

    # -- loop lifecycle --------------------------------------------------

    def start(self, start_time: float, duration_s: float) -> None:
        """start_time is a shared time.perf_counter() reference (usually set
        by session.py) so the emitter and optical capture agree on t=0."""
        self._start_time = start_time
        self._stop_event.clear()
        self._log = []
        self._thread = threading.Thread(
            target=self._run, args=(duration_s,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self, duration_s: float) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
        try:
            while not self._stop_event.is_set():
                elapsed = time.perf_counter() - self._start_time
                if elapsed >= duration_s:
                    break
                frame, chip_value, luminance, enabled = self.render_frame(elapsed)
                cv2.imshow(self.WINDOW_NAME, frame)
                cv2.waitKey(1)
                actual_t = time.perf_counter()  # the ACTUAL redraw moment
                self._log.append({
                    "t": actual_t,
                    "elapsed_s": actual_t - self._start_time,
                    "chip_value": chip_value,
                    "luminance": luminance,
                    "enabled": enabled,
                })
        finally:
            cv2.destroyWindow(self.WINDOW_NAME)
            cv2.waitKey(1)

    def get_log(self) -> list[dict]:
        return list(self._log)


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Milestone 2 emitter smoke test")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--disabled", action="store_true", help="run the emitter_enabled=false control")
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)

    challenge = Challenge(**raw["challenge"])
    econfig = EmitterConfig.from_dict(raw["emitter"])
    if args.disabled:
        econfig.emitter_enabled = False

    emitter = Emitter(challenge, econfig)
    start = time.perf_counter()
    emitter.start(start, args.seconds)
    time.sleep(args.seconds + 0.5)
    emitter.stop()

    log = emitter.get_log()
    if len(log) < 2:
        print("No redraws logged -- something is wrong.")
    else:
        intervals = np.diff([e["t"] for e in log])
        lum = np.array([e["luminance"] for e in log])
        print(f"redraws={len(log)} measured_fps={1/np.mean(intervals):.1f} "
              f"jitter_std_ms={np.std(intervals)*1000:.2f}")
        print(f"luminance mean={lum.mean():.1f} std={lum.std():.1f} "
              f"min={lum.min():.1f} max={lum.max():.1f}")
        print(f"emitter_enabled={econfig.emitter_enabled} "
              f"(std should be ~0 when disabled, large when enabled)")
