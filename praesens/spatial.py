"""Milestone 9: spatially-differential optical challenge.

The single-zone emitter (Milestones 1-8) modulates the WHOLE border with one
m-sequence, so the face's reflected signal is a single global luminance
waveform. An adversary who can see the victim's screen (a webcam pointed at
the screen, a screen-recording, a compromised second app) can estimate that
one waveform and multiply their injected video's overall brightness by it,
and the existing global correlation score would pass -- the score only ever
checked "does SOME brightness modulation track the pattern," not "does the
right PART of the face track the right PART of the screen." This is the
first attack a competent reviewer names, and before this milestone it works.

The fix: divide the border into >=3 independently-driven zones, each
carrying a DIFFERENT (cyclically shifted, near-orthogonal -- see
praesens.challenge.derive_zone_challenges) sequence from the same session
seed. A genuine reflection has each face region lit PREDOMINANTLY by its
nearest zone (forehead under the top strip, each cheek nearer one side), so
the correlation matrix R[roi, zone] should be diagonal-dominant: each ROI
peaks specifically on ITS OWN zone, not just on "some zone or other."

A single global-brightness multiply cannot reproduce that structure: the
attacker has one estimated waveform, so every ROI ends up correlating
similarly with every zone (whichever zone that waveform happens to resemble
most, if any) -- the GLOBAL correlation (computed the old way, treating the
whole border as one sequence, kept here as a separate sub-score) can still
be high, but the spatial ASSIGNMENT score (diagonal minus best off-diagonal)
collapses toward zero, because there is no genuine per-region correspondence
to find. That collapse, not the global score, is what this milestone adds.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from praesens.challenge import Challenge, ZoneChallenge, derive_zone_challenges
from praesens.emit import EmitterConfig, _screen_resolution
from praesens.optical import (
    PER_ROI_NAMES, per_roi_luminance, moving_average_detrend,
    cross_correlate_lag_search, resample_emitted, _zscore,
)


# ---------------------------------------------------------------------------
# Spatial emitter: same physical window as Emitter, but the border is
# divided into independently-driven zones instead of one uniform ring.
# ---------------------------------------------------------------------------

class SpatialEmitter:
    """Drives a fullscreen window whose border is split into left/right/top
    (or however many zones are configured) regions, each showing its own
    zone challenge's current chip value. API mirrors praesens.emit.Emitter
    (render_frame/log_redraw/begin_manual_drive) deliberately, so callers
    that already know how to drive Emitter (demo/live.py's manual-drive
    pattern) need minimal changes to drive this instead -- but this is kept
    as a SEPARATE class rather than modifying Emitter itself, so the
    existing Milestone 0-8 single-zone pipeline (session.py, demo/*, eval/*)
    is completely unaffected by this addition (rule 4: conditions must be
    measured identically to how they always have been)."""

    WINDOW_NAME = "praesens_spatial_emitter"

    def __init__(self, zones: list[ZoneChallenge], config: EmitterConfig):
        self.zones = {z.zone_name: z for z in zones}
        self.config = config

        self._lock = threading.Lock()
        self._enabled = bool(config.emitter_enabled)
        self._modulation_depth = float(config.modulation_depth)

        self._log: dict[str, list] = {name: [] for name in self.zones}
        self._preview = None

        self._start_time: float | None = None
        self._width, self._height = _screen_resolution(
            config.fallback_window_width, config.fallback_window_height
        )

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def set_preview(self, frame) -> None:
        with self._lock:
            self._preview = frame

    def begin_manual_drive(self, start_time: float) -> None:
        self._start_time = start_time
        self._log = {name: [] for name in self.zones}

    def render_frame(self, elapsed_s: float):
        with self._lock:
            enabled = self._enabled
            depth = self._modulation_depth
            preview = self._preview
        base = self.config.base_luminance

        frame = np.full((self._height, self._width, 3), base, dtype=np.uint8)
        bf = self.config.border_fraction
        bw = int(self._width * bf)
        bh = int(self._height * bf)

        chip_values = {}
        # Paint top first, then left/right, so the side strips own their
        # corners rather than the top strip -- cosmetic only, doesn't
        # affect the ROI regions the optical lane actually reads.
        paint_order = [n for n in ("top", "left", "right") if n in self.zones]
        paint_order += [n for n in self.zones if n not in paint_order]

        for name in paint_order:
            zone = self.zones[name]
            if enabled:
                chip_value = int(zone.challenge.value_at(elapsed_s))
            else:
                chip_value = 0
            chip_values[name] = chip_value
            lum = float(np.clip(base + depth * chip_value, 0, 255))

            if name == "left" and bw > 0:
                frame[:, :bw] = lum
            elif name == "right" and bw > 0:
                frame[:, self._width - bw:] = lum
            elif name == "top" and bh > 0:
                frame[:bh, :] = lum
            elif name == "bottom" and bh > 0:
                frame[self._height - bh:, :] = lum

        if preview is not None and bw > 0 and bh > 0 and self._width - 2 * bw > 0 and self._height - 2 * bh > 0:
            ph, pw = preview.shape[:2]
            target_w, target_h = self._width - 2 * bw, self._height - 2 * bh
            scale = min(target_w / pw, target_h / ph)
            new_w, new_h = max(1, int(pw * scale)), max(1, int(ph * scale))
            resized = cv2.resize(preview, (new_w, new_h))
            y0 = bh + (target_h - new_h) // 2
            x0 = bw + (target_w - new_w) // 2
            frame[y0:y0 + new_h, x0:x0 + new_w] = resized

        return frame, chip_values, enabled

    def log_redraw(self, chip_values: dict, enabled: bool) -> None:
        actual_t = time.perf_counter()
        for name, chip_value in chip_values.items():
            self._log[name].append({"t": actual_t, "chip_value": chip_value, "enabled": enabled})

    def drive_and_log(self, elapsed_s: float):
        """Milestone 14: same one-call interface as praesens.emit.Emitter's
        drive_and_log -- see that method's docstring. Lets demo/live.py's
        run loop drive whichever emitter is currently assigned to
        self.emitter without a type check, even though this class's own
        render_frame/log_redraw shapes (a per-zone chip_values dict) differ
        from Emitter's (one scalar chip_value/luminance pair)."""
        frame, chip_values, enabled = self.render_frame(elapsed_s)
        self.log_redraw(chip_values, enabled)
        return frame

    def get_zone_log(self, zone_name: str) -> list:
        return list(self._log.get(zone_name, []))

    def get_global_log(self) -> list:
        """Union of all zones' redraws, chip_value = mean across zones at
        that instant -- a stand-in "whole border" sequence for computing
        the GLOBAL sub-score the old single-zone pipeline would have
        produced, so it can be reported alongside the new spatial score
        rather than losing that comparison point."""
        if not self._log or not next(iter(self._log.values()), []):
            return []
        any_zone_log = next(iter(self._log.values()))
        n = len(any_zone_log)
        out = []
        for i in range(n):
            t = any_zone_log[i]["t"]
            enabled = any_zone_log[i]["enabled"]
            mean_chip = float(np.mean([self._log[name][i]["chip_value"] for name in self._log
                                        if i < len(self._log[name])]))
            out.append({"t": t, "chip_value": mean_chip, "enabled": enabled})
        return out


# ---------------------------------------------------------------------------
# Correlation matrix + spatial assignment score
# ---------------------------------------------------------------------------

@dataclass
class SpatialResult:
    global_score: float
    spatial_score: float
    correlation_matrix: list  # roi x zone, as nested lists (JSON-friendly)
    roi_names: list
    zone_names: list
    lag_matrix_ms: list


def compute_correlation_matrix(per_roi_traces: dict, zone_logs: dict, detrend_window_s: float,
                                lag_max_ms: float, lag_step_ms: float):
    """R[i,j] = peak correlation of ROI i's detrended luminance against
    zone j's actual redraw log. per_roi_traces: {roi_name: (timestamps,
    luminances)}. zone_logs: {zone_name: emitter_log}. Returns
    (R, lag_matrix_ms, roi_names, zone_names)."""
    roi_names = [n for n in PER_ROI_NAMES if n in per_roi_traces]
    zone_names = list(zone_logs.keys())
    R = np.zeros((len(roi_names), len(zone_names)))
    lag_matrix = np.zeros((len(roi_names), len(zone_names)))

    for i, roi in enumerate(roi_names):
        ts, lum = per_roi_traces[roi]
        ts = np.asarray(ts)
        lum = np.asarray(lum)
        detrended = moving_average_detrend(ts, lum, detrend_window_s)
        for j, zone in enumerate(zone_names):
            score, lag_ms, *_ = cross_correlate_lag_search(detrended, ts, zone_logs[zone], lag_max_ms, lag_step_ms)
            R[i, j] = score
            lag_matrix[i, j] = lag_ms

    return R, lag_matrix, roi_names, zone_names


def spatial_assignment_score(R: np.ndarray, roi_names: list, zone_names: list, expected_assignment: dict) -> float:
    """Mean diagonal (expected ROI-zone pairing) minus mean best-off-
    diagonal (best WRONG pairing) per ROI. Positive and large: each ROI
    clearly prefers its own zone. Near zero: no genuine per-region
    correspondence -- exactly what a single global-brightness-multiply
    attack produces (a roughly uniform matrix, since one estimated
    waveform cannot distinguish which zone's code belongs where)."""
    diag_scores, offdiag_scores = [], []
    for i, roi in enumerate(roi_names):
        expected_zone = expected_assignment.get(roi)
        if expected_zone not in zone_names:
            continue
        j_expected = zone_names.index(expected_zone)
        diag_scores.append(R[i, j_expected])
        others = [R[i, j] for j in range(len(zone_names)) if j != j_expected]
        if others:
            offdiag_scores.append(max(others))
    if not diag_scores:
        return float("nan")
    return float(np.mean(diag_scores) - (np.mean(offdiag_scores) if offdiag_scores else 0.0))


def global_score_from_logs(per_roi_traces: dict, global_log: list, detrend_window_s: float,
                            lag_max_ms: float, lag_step_ms: float) -> float:
    """The OLD (Milestone 3) style score: treat the whole border as one
    sequence (here, the mean across zones) and correlate the COMBINED
    ROI trace (mean across the per-roi traces) against it -- kept as an
    explicit separate sub-score, not replaced, per the milestone's own
    instruction not to discard the existing global correlation."""
    all_ts = None
    lum_stack = []
    for roi, (ts, lum) in per_roi_traces.items():
        ts = np.asarray(ts)
        if all_ts is None:
            all_ts = ts
        lum_stack.append(np.asarray(lum))
    if all_ts is None or not lum_stack:
        return 0.0
    combined_lum = np.nanmean(np.stack(lum_stack), axis=0)
    detrended = moving_average_detrend(all_ts, combined_lum, detrend_window_s)
    score, *_ = cross_correlate_lag_search(detrended, all_ts, global_log, lag_max_ms, lag_step_ms)
    return float(score)


@dataclass
class SpatialConfig:
    zone_names: tuple = ("left", "right", "top")
    min_shift_chips: int = 0
    detrend_window_s: float = 1.5
    lag_search_max_ms: float = 300
    lag_step_ms: float = 5
    expected_assignment: dict = field(default_factory=lambda: {
        "forehead": "top", "left_cheek": "left", "right_cheek": "right"
    })

    @classmethod
    def from_dict(cls, d: dict) -> "SpatialConfig":
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "zone_names" in kwargs:
            kwargs["zone_names"] = tuple(kwargs["zone_names"])
        return cls(**kwargs)


if __name__ == "__main__":
    import argparse
    import yaml
    from mediapipe.tasks.python import vision as mp_vision

    from praesens.optical import OpticalConfig, create_landmarker, lock_camera, measure_capture_fps
    from praesens.challenge import pick_auto_chip_rate
    from praesens.capture import CaptureConfig, configure_capture_format

    parser = argparse.ArgumentParser(description="Milestone 9 spatial challenge smoke test")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "config.yaml") as f:
        raw = yaml.safe_load(f)

    sconfig = SpatialConfig.from_dict(raw["spatial"])
    econfig = EmitterConfig.from_dict(raw["emitter"])
    oconfig = OpticalConfig.from_dict(raw["optical"])
    oconfig.model_path = str(repo_root / oconfig.model_path)

    # Camera opens first, same reasoning as session.py: auto_chip_rate needs
    # a real FPS measurement from THIS camera before the challenge (and here,
    # its zone derivatives) can be built with a chip rate that's actually
    # samplable -- each zone is sampled at the SAME per-frame rate the
    # single-zone pipeline is, so the same Nyquist floor applies per zone.
    cap = cv2.VideoCapture(oconfig.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {oconfig.camera_index}")
    warn_list: list = []
    cconfig = CaptureConfig.from_dict(raw.get("capture", {}))
    configure_capture_format(cap, cconfig, warn_list)
    lock_camera(cap, oconfig, warn_list)
    for w in warn_list:
        print(f"WARNING: {w}")

    chip_rate_hz = raw["challenge"]["chip_rate_hz"]
    duration_s = args.seconds
    if raw["optical"].get("auto_chip_rate", False):
        preflight_fps = measure_capture_fps(cap)
        chip_rate_hz, min_duration_s = pick_auto_chip_rate(
            preflight_fps,
            divisor=raw["optical"].get("auto_chip_rate_divisor", 6.0),
            min_hz=0.5, max_hz=5.0,
            min_chips=raw["optical"].get("auto_chip_rate_min_chips", 60),
            base_duration_s=args.seconds,
        )
        duration_s = max(duration_s, min_duration_s)
        print(f"auto_chip_rate: measured preflight FPS={preflight_fps:.1f} -> "
              f"chip_rate_hz={chip_rate_hz:.2f}, duration_s={duration_s:.1f}")

    master = Challenge(chip_rate_hz=chip_rate_hz, duration_s=duration_s)
    zones = derive_zone_challenges(master, sconfig.zone_names, sconfig.min_shift_chips)
    print(f"master seed={master.seed} order={master.order} n_chips={master.n_chips}")
    for z in zones:
        print(f"  zone={z.zone_name} shift_chips={z.shift_chips}")

    emitter = SpatialEmitter(zones, econfig)
    args.seconds = duration_s  # the capture loop below reads args.seconds as its stop condition

    landmarker = create_landmarker(oconfig.model_path, mp_vision.RunningMode.VIDEO, oconfig.min_face_confidence)
    detect_every_n = max(1, oconfig.detect_every_n_frames)

    cv2.namedWindow(SpatialEmitter.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(SpatialEmitter.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    start_time = time.perf_counter()
    emitter.begin_manual_drive(start_time)

    per_roi_ts = {name: [] for name in PER_ROI_NAMES}
    per_roi_lum = {name: [] for name in PER_ROI_NAMES}
    last_ts_ms = -1
    frame_i = 0
    last_result = None

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.seconds:
                break

            grabbed = cap.grab()
            t = time.perf_counter()
            if grabbed:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    ts_ms = max(last_ts_ms + 1, int((t - start_time) * 1000))
                    last_ts_ms = ts_ms

                    # Per-ROI luminance needs the individual boxes, not
                    # CadenceDetector's merged mask -- same detect-every-Nth-
                    # frame cadence, tracked locally instead.
                    if frame_i % detect_every_n == 0:
                        import mediapipe as mp
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        last_result = landmarker.detect_for_video(mp_image, ts_ms)
                    frame_i += 1

                    lum_dict = per_roi_luminance(frame, last_result, oconfig.roi_margin_frac) if last_result else \
                        {name: float("nan") for name in PER_ROI_NAMES}
                    for name in PER_ROI_NAMES:
                        per_roi_ts[name].append(t)
                        per_roi_lum[name].append(lum_dict[name])

                    emitter.set_preview(frame)

            frame_img, chip_values, enabled = emitter.render_frame(elapsed)
            cv2.imshow(SpatialEmitter.WINDOW_NAME, frame_img)
            cv2.waitKey(1)
            emitter.log_redraw(chip_values, enabled)
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()

    per_roi_traces = {name: (per_roi_ts[name], per_roi_lum[name]) for name in PER_ROI_NAMES}
    zone_logs = {name: emitter.get_zone_log(name) for name in emitter.zones}

    R, lag_matrix, roi_names, zone_names = compute_correlation_matrix(
        per_roi_traces, zone_logs, sconfig.detrend_window_s, sconfig.lag_search_max_ms, sconfig.lag_step_ms
    )
    spatial_score = spatial_assignment_score(R, roi_names, zone_names, sconfig.expected_assignment)
    global_score = global_score_from_logs(per_roi_traces, emitter.get_global_log(),
                                           sconfig.detrend_window_s, sconfig.lag_search_max_ms, sconfig.lag_step_ms)

    print(f"\nglobal_score={global_score:.3f}  spatial_assignment_score={spatial_score:.3f}")
    print(f"\ncorrelation matrix R[roi, zone]:")
    print(f"{'':14s}" + "".join(f"{z:>10s}" for z in zone_names))
    for i, roi in enumerate(roi_names):
        print(f"{roi:14s}" + "".join(f"{R[i,j]:10.3f}" for j in range(len(zone_names))))
