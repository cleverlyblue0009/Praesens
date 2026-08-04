"""Milestone 3: optical lane.

Turns raw webcam frames into a liveness score. Captures with a
perf_counter() timestamp taken at grab time (before the slower JPEG/YUV
decode step, so the timestamp reflects when light actually hit the sensor,
not when decoding finished); locks exposure and white balance so the
camera's own auto-exposure doesn't cancel out the very brightness changes
we're trying to measure; finds the face via MediaPipe and averages
luminance over the forehead and both cheeks (flat, largely shadow-free skin,
away from eyes/eyebrows/nose/mouth where geometry or specular highlights
would add noise unrelated to the light pattern); detrends against a moving
average to remove slow lighting drift unrelated to the 5 Hz challenge; and
cross-correlates that against what the emitter actually displayed,
resampled onto the camera's real (jittery) frame timestamps rather than an
idealised schedule. The peak correlation is the liveness score, and the lag
at that peak is a diagnostic (should sit near the camera's own capture
latency, a few tens of ms, not near-zero or wildly off).

SNR handling exists because "low correlation" is ambiguous between two very
different situations that must not be conflated: an attack (no physical
coupling to the pattern exists at all) and a measurement failure (the
coupling exists but is too weak to see, e.g. dark skin under dim light, or
makeup increasing surface scatter). We estimate signal-to-noise in the 5 Hz
band; if it's too low early in the session we increase the emitter's
modulation depth to compensate (compensate the light, don't penalise the
subject); if it's still too low at maximum depth we say so explicitly via
insufficient_signal rather than silently folding it into a low score.
"""
from __future__ import annotations

import time
import warnings as _warnings_module
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import scipy.signal

import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

from praesens.challenge import Challenge
from praesens.emit import Emitter


# ---------------------------------------------------------------------------
# ROI landmark groups, derived from MediaPipe's own connection sets rather
# than a hand-memorised list of landmark indices -- the m-sequence tap table
# bug earlier came from exactly that kind of memorised-and-wrong table, so
# here we pull the indices straight out of the installed library instead.
# ---------------------------------------------------------------------------

def _unique_sorted_indices(connections) -> list:
    idx = set()
    for c in connections:
        idx.add(c.start)
        idx.add(c.end)
    return sorted(idx)


_FLC = mp_vision.FaceLandmarksConnections
_LEFT_EYE_IDX = _unique_sorted_indices(_FLC.FACE_LANDMARKS_LEFT_EYE)
_RIGHT_EYE_IDX = _unique_sorted_indices(_FLC.FACE_LANDMARKS_RIGHT_EYE)
_LEFT_EYEBROW_IDX = _unique_sorted_indices(_FLC.FACE_LANDMARKS_LEFT_EYEBROW)
_RIGHT_EYEBROW_IDX = _unique_sorted_indices(_FLC.FACE_LANDMARKS_RIGHT_EYEBROW)
_LIPS_IDX = _unique_sorted_indices(_FLC.FACE_LANDMARKS_LIPS)
_FACE_OVAL_IDX = _unique_sorted_indices(_FLC.FACE_LANDMARKS_FACE_OVAL)
_NOSE_IDX = _unique_sorted_indices(_FLC.FACE_LANDMARKS_NOSE)
_EYES_IDX = _LEFT_EYE_IDX + _RIGHT_EYE_IDX
_EYEBROWS_IDX = _LEFT_EYEBROW_IDX + _RIGHT_EYEBROW_IDX


def _landmark_pixels(landmarks, idx_list, width, height) -> np.ndarray:
    return np.array([[landmarks[i].x * width, landmarks[i].y * height] for i in idx_list])


def compute_roi_boxes(frame_shape, landmarks, margin_frac: float):
    """Forehead + both-cheeks boxes, positioned relative to THIS frame's
    detected face geometry (not fixed pixel coordinates), excluding eyes,
    eyebrows, nose and mouth. Returns a list of (x0, y0, x1, y1) boxes in
    pixel coordinates, possibly empty if the face geometry is degenerate
    (e.g. extreme profile turn)."""
    h, w = frame_shape[:2]
    oval = _landmark_pixels(landmarks, _FACE_OVAL_IDX, w, h)
    eyes = _landmark_pixels(landmarks, _EYES_IDX, w, h)
    eyebrows = _landmark_pixels(landmarks, _EYEBROWS_IDX, w, h)
    lips = _landmark_pixels(landmarks, _LIPS_IDX, w, h)
    nose = _landmark_pixels(landmarks, _NOSE_IDX, w, h)

    face_top, face_bottom = oval[:, 1].min(), oval[:, 1].max()
    face_left, face_right = oval[:, 0].min(), oval[:, 0].max()
    eyebrow_top = eyebrows[:, 1].min()
    eye_bottom = eyes[:, 1].max()
    lips_top = lips[:, 1].min()
    nose_left, nose_right = nose[:, 0].min(), nose[:, 0].max()
    center_x = (face_left + face_right) / 2.0
    face_w = face_right - face_left
    face_h = face_bottom - face_top
    margin = margin_frac * max(face_h, 1.0)

    boxes = []

    fy0, fy1 = face_top + margin, eyebrow_top - margin
    fx0, fx1 = center_x - 0.30 * face_w, center_x + 0.30 * face_w
    if fy1 > fy0 and fx1 > fx0:
        boxes.append((fx0, fy0, fx1, fy1))

    cy0, cy1 = eye_bottom + margin, lips_top - margin
    if cy1 > cy0:
        lx0, lx1 = face_left + margin, nose_left - margin
        if lx1 > lx0:
            boxes.append((lx0, cy0, lx1, cy1))
        rx0, rx1 = nose_right + margin, face_right - margin
        if rx1 > rx0:
            boxes.append((rx0, cy0, rx1, cy1))

    return boxes


def boxes_to_mask(frame_shape, boxes) -> np.ndarray:
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        xi0, yi0 = max(0, int(x0)), max(0, int(y0))
        xi1, yi1 = min(w, int(x1)), min(h, int(y1))
        if xi1 > xi0 and yi1 > yi0:
            mask[yi0:yi1, xi0:xi1] = True
    return mask


def roi_luminance(frame, detection_result, margin_frac: float):
    """Mean grayscale luminance over the forehead+cheeks ROI for one frame.
    Returns (luminance_or_nan, face_detected)."""
    if not detection_result.face_landmarks:
        return float("nan"), False
    landmarks = detection_result.face_landmarks[0]
    boxes = compute_roi_boxes(frame.shape, landmarks, margin_frac)
    mask = boxes_to_mask(frame.shape, boxes)
    if not mask.any():
        return float("nan"), False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray[mask].mean()), True


def draw_roi_debug(frame, detection_result, margin_frac: float):
    """Annotated copy of frame with ROI boxes drawn -- for visually
    confirming the boxes land on forehead/cheeks (scripts/optical.py can be
    run standalone with --debug-roi to save one of these)."""
    out = frame.copy()
    if detection_result.face_landmarks:
        landmarks = detection_result.face_landmarks[0]
        boxes = compute_roi_boxes(frame.shape, landmarks, margin_frac)
        for x0, y0, x1, y1 in boxes:
            cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)
    else:
        cv2.putText(out, "NO FACE DETECTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
    return out


# ---------------------------------------------------------------------------
# Camera exposure/WB lock
# ---------------------------------------------------------------------------

def _mean_luminance(frame) -> float:
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)))


def lock_camera(cap, config, warn_list: list) -> bool:
    """Best-effort exposure/WB lock, verified BEHAVIORALLY (luminance drift
    over a burst of frames) rather than by reading properties back --
    Milestone 0 found this driver's CAP_PROP_AUTO_EXPOSURE readback is a
    broken -1 sentinel while the underlying set() call still works, so
    trusting readback here would produce a false "not locked" verdict."""
    for candidate in (0.25, 1, 0):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, candidate)
    cap.set(cv2.CAP_PROP_EXPOSURE, config.exposure_value)
    if config.disable_auto_wb:
        cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    time.sleep(0.5)

    for _ in range(5):
        cap.read()
    samples = []
    for _ in range(config.lock_verification_frames):
        ok, frame = cap.read()
        if ok:
            samples.append(_mean_luminance(frame))
    samples = np.array(samples)
    drift = float(np.std(samples)) if len(samples) else float("nan")
    locked = len(samples) > 0 and drift <= config.lock_verification_max_std

    if not locked:
        msg = (
            f"exposure/WB lock could not be verified behaviorally "
            f"(luminance std={drift:.2f} over {len(samples)} frames, "
            f"threshold={config.lock_verification_max_std}). Auto-exposure may "
            f"still be compensating for the light pattern; scores from this "
            f"session may be unreliable."
        )
        print("!" * 70)
        print(f"WARNING: {msg}")
        print("!" * 70)
        warn_list.append(msg)
    return locked


# ---------------------------------------------------------------------------
# Signal processing: detrend, resample, cross-correlate, SNR
# ---------------------------------------------------------------------------

def moving_average_detrend(timestamps: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    """Subtract a centred, time-windowed moving average (not sample-count
    windowed, since frame spacing isn't perfectly uniform) to remove slow
    lighting drift unrelated to the challenge frequency. NaN in -> NaN out."""
    ts = np.asarray(timestamps, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(vals)
    out = np.full_like(vals, np.nan)
    half = window_s / 2.0
    for i in range(len(ts)):
        if not valid[i]:
            continue
        lo = np.searchsorted(ts, ts[i] - half, side="left")
        hi = np.searchsorted(ts, ts[i] + half, side="right")
        window_vals = vals[lo:hi][valid[lo:hi]]
        if len(window_vals) == 0:
            continue
        out[i] = vals[i] - np.mean(window_vals)
    return out


def resample_emitted(emitter_log: list, query_timestamps: np.ndarray, lag_s: float = 0.0) -> np.ndarray:
    """Zero-order-hold lookup of the emitter's ACTUAL logged chip value at
    (query_timestamps - lag_s) -- ground truth for what was on screen, not
    an idealised schedule, per Milestone 2's actual-redraw logging."""
    if not emitter_log:
        return np.zeros(len(query_timestamps))
    log_t = np.array([e["t"] for e in emitter_log])
    log_val = np.array([e["chip_value"] for e in emitter_log], dtype=np.float64)
    query_t = np.asarray(query_timestamps) - lag_s
    idx = np.searchsorted(log_t, query_t, side="right") - 1
    idx = np.clip(idx, 0, len(log_t) - 1)
    return log_val[idx]


def cross_correlate_lag_search(measured_detrended: np.ndarray, frame_timestamps: np.ndarray,
                                emitter_log: list, lag_max_ms: float, lag_step_ms: float):
    """Normalised (Pearson) cross-correlation over a lag grid, because frame
    timestamps are non-uniform: rather than shifting sample indices, we
    re-resample the emitted signal at (t - lag) for each lag hypothesis.
    Returns (score, lag_ms, emitted_at_best_lag, valid_timestamps, valid_measured)."""
    valid = ~np.isnan(measured_detrended)
    ts = np.asarray(frame_timestamps)[valid]
    m = np.asarray(measured_detrended)[valid]

    if len(m) < 5 or np.std(m) < 1e-9:
        return 0.0, 0.0, np.zeros_like(m), ts, m

    m_centered = m - m.mean()
    m_std = np.std(m_centered)

    lags_ms = np.arange(0, lag_max_ms + 1e-9, lag_step_ms)
    best_score, best_lag, best_emitted = -np.inf, 0.0, np.zeros_like(m)

    for lag_ms in lags_ms:
        emitted = resample_emitted(emitter_log, ts, lag_s=lag_ms / 1000.0)
        e_centered = emitted - emitted.mean()
        e_std = np.std(e_centered)
        if e_std < 1e-9:
            corr = 0.0
        else:
            corr = float(np.mean(m_centered * e_centered) / (m_std * e_std))
        if corr > best_score:
            best_score, best_lag, best_emitted = corr, float(lag_ms), emitted

    return best_score, best_lag, best_emitted, ts, m


def estimate_snr_db(timestamps: np.ndarray, detrended_values: np.ndarray, chip_rate_hz: float,
                     low_cut_hz: float, resample_rate_hz: float) -> float:
    """In-band power vs. out-of-band power, on a uniformly-resampled copy of
    the (non-uniformly sampled) detrended trace -- scipy's filters assume
    even spacing.

    The "signal band" is [low_cut_hz, chip_rate_hz], not a window centred on
    chip_rate_hz: a random +/-1 chip sequence has a sinc^2-shaped power
    spectrum with its main lobe running from DC to the chip rate (first
    null at chip_rate_hz), peaking near DC, not a tone AT the chip rate.
    Verified empirically for this LFSR's actual output -- 90%+ of its power
    sits below chip_rate_hz. A window centred on chip_rate_hz (e.g.
    [3.5, 6.5] Hz for a 5 Hz chip rate) catches only a sliver of the true
    signal and mislabels the rest as noise, which was caught here by a
    synthetic-signal test returning an impossible negative SNR for a
    strong, clean signal. low_cut_hz keeps the passband off DC/near-DC
    where detrending already suppresses slow drift and a filter singularity
    would sit at f=0."""
    valid = ~np.isnan(detrended_values)
    ts = np.asarray(timestamps)[valid]
    vals = np.asarray(detrended_values)[valid]
    if len(ts) < 16 or ts[-1] <= ts[0]:
        return float("nan")

    uniform_t = np.arange(ts[0], ts[-1], 1.0 / resample_rate_hz)
    if len(uniform_t) < 16:
        return float("nan")
    uniform_vals = np.interp(uniform_t, ts, vals)
    uniform_vals = uniform_vals - np.mean(uniform_vals)

    nyq = resample_rate_hz / 2.0
    low = max(0.05, low_cut_hz) / nyq
    high = min(nyq * 0.99, chip_rate_hz) / nyq
    if not (0 < low < high < 1):
        return float("nan")

    try:
        b, a = scipy.signal.butter(4, [low, high], btype="band")
        if len(uniform_vals) <= 3 * max(len(a), len(b)):
            return float("nan")
        with _warnings_module.catch_warnings():
            _warnings_module.simplefilter("ignore")
            in_band = scipy.signal.filtfilt(b, a, uniform_vals)
    except (ValueError, ZeroDivisionError):
        return float("nan")

    signal_power = float(np.mean(in_band ** 2))
    residual = uniform_vals - in_band
    noise_power = float(np.mean(residual ** 2))
    if noise_power < 1e-12:
        return float("inf") if signal_power > 1e-12 else float("nan")
    return float(10 * np.log10(signal_power / noise_power))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    std = np.std(x)
    if std < 1e-9:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


# ---------------------------------------------------------------------------
# FaceLandmarker lifecycle
# ---------------------------------------------------------------------------

def create_landmarker(model_path: str, running_mode=mp_vision.RunningMode.VIDEO,
                       min_face_confidence: float = 0.5):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"FaceLandmarker model not found at {model_path}. Run "
            f"'python scripts/fetch_model.py' once to download it."
        )
    options = mp_vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=running_mode,
        num_faces=1,
        min_face_detection_confidence=min_face_confidence,
        min_face_presence_confidence=min_face_confidence,
        min_tracking_confidence=min_face_confidence,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


# ---------------------------------------------------------------------------
# Config + main entry point
# ---------------------------------------------------------------------------

@dataclass
class OpticalConfig:
    camera_index: int = 0
    model_path: str = "models/face_landmarker.task"
    exposure_value: float = -6
    disable_auto_wb: bool = True
    lock_verification_frames: int = 15
    lock_verification_max_std: float = 3.0
    roi_margin_frac: float = 0.06
    detrend_window_s: float = 1.5
    lag_search_max_ms: float = 300
    lag_step_ms: float = 5
    snr_low_cut_hz: float = 0.3
    snr_resample_rate_hz: float = 60.0
    snr_floor_db: float = 3.0
    adaptive_window_s: float = 3.0
    min_face_confidence: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "OpticalConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class OpticalResult:
    score: float
    lag_ms: float
    snr_db: float
    insufficient_signal: bool
    adaptive_boost_applied: bool
    exposure_locked: bool
    trace_emitted: list
    trace_measured: list
    timestamps: list
    n_frames: int
    n_face_detected: int
    warnings: list = field(default_factory=list)


def run_session(cap, challenge: Challenge, config: OpticalConfig, start_time: float,
                 duration_s: float, emitter: Emitter | None = None, landmarker=None) -> OpticalResult:
    """Runs the capture+measurement loop for one session and returns the
    liveness score, lag, SNR, and both aligned traces. If `emitter` is
    given, its modulation depth is boosted mid-session when early SNR is
    too low (Milestone 3's SNR addendum)."""
    warn_list: list = []
    exposure_locked = lock_camera(cap, config, warn_list)

    own_landmarker = landmarker is None
    if own_landmarker:
        landmarker = create_landmarker(config.model_path, mp_vision.RunningMode.VIDEO,
                                        config.min_face_confidence)

    timestamps, luminances, face_flags = [], [], []
    adaptive_triggered = False
    last_ts_ms = -1

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration_s:
                break

            grabbed = cap.grab()
            t = time.perf_counter()  # timestamp at GRAB, before decode
            if not grabbed:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            ts_ms = max(last_ts_ms + 1, int((t - start_time) * 1000))
            last_ts_ms = ts_ms

            if emitter is not None:
                emitter.set_preview(frame)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            lum, face_ok = roi_luminance(frame, result, config.roi_margin_frac)

            timestamps.append(t)
            luminances.append(lum)
            face_flags.append(face_ok)

            if emitter is not None and not adaptive_triggered and (t - start_time) >= config.adaptive_window_s:
                adaptive_triggered = True
                ts_arr = np.array(timestamps)
                lum_arr = np.array(luminances)
                early_mask = ts_arr <= start_time + config.adaptive_window_s
                if early_mask.sum() >= 8:
                    early_detrended = moving_average_detrend(
                        ts_arr[early_mask], lum_arr[early_mask], config.detrend_window_s
                    )
                    early_snr = estimate_snr_db(
                        ts_arr[early_mask], early_detrended, challenge.chip_rate_hz,
                        config.snr_low_cut_hz, config.snr_resample_rate_hz
                    )
                    if not np.isnan(early_snr) and early_snr < config.snr_floor_db:
                        emitter.set_modulation_depth(emitter.config.max_modulation_depth)
                        msg = (f"early SNR {early_snr:.1f} dB below floor "
                               f"{config.snr_floor_db} dB -- boosted modulation depth to "
                               f"{emitter.config.max_modulation_depth}")
                        warn_list.append(msg)
    finally:
        if own_landmarker:
            landmarker.close()

    ts_arr = np.array(timestamps)
    lum_arr = np.array(luminances)
    n_face = int(np.sum(face_flags))

    if len(ts_arr) == 0:
        return OpticalResult(
            score=0.0, lag_ms=0.0, snr_db=float("nan"), insufficient_signal=True,
            adaptive_boost_applied=(emitter.adaptive_boost_applied() if emitter else False),
            exposure_locked=exposure_locked, trace_emitted=[], trace_measured=[],
            timestamps=[], n_frames=0, n_face_detected=0,
            warnings=warn_list + ["no frames captured"],
        )

    detrended = moving_average_detrend(ts_arr, lum_arr, config.detrend_window_s)
    emitter_log = emitter.get_log() if emitter is not None else []

    score, lag_ms, emitted_at_lag, valid_ts, valid_measured = cross_correlate_lag_search(
        detrended, ts_arr, emitter_log, config.lag_search_max_ms, config.lag_step_ms
    )

    final_snr_db = estimate_snr_db(
        ts_arr, detrended, challenge.chip_rate_hz, config.snr_low_cut_hz, config.snr_resample_rate_hz
    )
    insufficient_signal = bool(np.isnan(final_snr_db) or final_snr_db < config.snr_floor_db)

    return OpticalResult(
        score=float(score),
        lag_ms=float(lag_ms),
        snr_db=float(final_snr_db) if not np.isnan(final_snr_db) else float("nan"),
        insufficient_signal=insufficient_signal,
        adaptive_boost_applied=(emitter.adaptive_boost_applied() if emitter else False),
        exposure_locked=exposure_locked,
        trace_emitted=_zscore(emitted_at_lag).tolist(),
        trace_measured=_zscore(valid_measured).tolist(),
        timestamps=valid_ts.tolist(),
        n_frames=len(ts_arr),
        n_face_detected=n_face,
        warnings=warn_list,
    )


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Milestone 3 optical lane smoke test")
    parser.add_argument("--debug-roi", action="store_true",
                         help="save one annotated frame showing the ROI boxes and exit")
    parser.add_argument("--seconds", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "config.yaml") as f:
        raw = yaml.safe_load(f)

    oconfig = OpticalConfig.from_dict(raw["optical"])
    oconfig.model_path = str(repo_root / oconfig.model_path)

    cap = cv2.VideoCapture(oconfig.camera_index, cv2.CAP_DSHOW)
    warn_list: list = []
    lock_camera(cap, oconfig, warn_list)

    if args.debug_roi:
        landmarker = create_landmarker(oconfig.model_path, mp_vision.RunningMode.IMAGE,
                                        oconfig.min_face_confidence)
        for _ in range(10):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        annotated = draw_roi_debug(frame, result, oconfig.roi_margin_frac)
        out_path = repo_root / "logs" / "roi_debug.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"faces_detected={len(result.face_landmarks)} -> saved {out_path}")
    else:
        challenge = Challenge(**raw["challenge"])
        from praesens.emit import Emitter, EmitterConfig
        econfig = EmitterConfig.from_dict(raw["emitter"])
        emitter = Emitter(challenge, econfig)
        start = time.perf_counter()
        emitter.start(start, args.seconds)
        result = run_session(cap, challenge, oconfig, start, args.seconds, emitter=emitter)
        emitter.stop()
        cap.release()
        print(f"score={result.score:.3f} lag_ms={result.lag_ms:.1f} snr_db={result.snr_db:.2f} "
              f"insufficient_signal={result.insufficient_signal} "
              f"n_frames={result.n_frames} n_face_detected={result.n_face_detected} "
              f"exposure_locked={result.exposure_locked}")
        for w in result.warnings:
            print(f"  warning: {w}")
