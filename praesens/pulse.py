"""rPPG pulse lane (advisory by default).

Blood volume in facial skin rises and falls with every heartbeat, which
shifts the skin's colour by a fraction of a percent -- far below what an eye
notices, well within what a webcam's averaged ROI pixels resolve. A live face
therefore carries a quasi-periodic colour signal at the heart rate
(~0.7-3 Hz); a printed photo, a phone/monitor replay of a STILL face, a mask
or a 3D model carries none, and most face-swap/reenactment generators do not
synthesise a coherent one.

Method: POS ("Plane-Orthogonal-to-Skin", Wang, den Brinker, Stuijk & de Haan,
IEEE TBME 2017) on the mean RGB of the SAME forehead/cheek ROI boxes the
optical lane already computes (praesens.optical.compute_roi_boxes) -- no
second detector. POS projects the temporally-normalised RGB onto a plane
orthogonal to the skin-tone direction, which removes pure INTENSITY changes
(every channel scaled by the same factor).

Why that matters here: the optical lane's challenge modulates the SCREEN
brightness at ~1-2 Hz -- inside the heart-rate band. That light is
achromatic, so it is mostly an intensity change POS rejects; what leaks
through (camera gamma, white-balance residue) is removed explicitly by
regressing the KNOWN emitted chip waveform (at its best lag) out of the
pulse signal. The challenge is our own signal -- we know exactly what to
subtract.

Advisory, not gated, unless config pulse.gate is true: at ~16 fps under a
locked, deliberately-dim exposure (exposure_value -4, see NOTES 2026-09-01)
rPPG SNR has NOT been measured on this project's corpus yet. The lane reports
BPM + SNR honestly; turning it into a gate needs a measured threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.signal

from praesens.optical import boxes_to_mask, resample_emitted


@dataclass
class PulseConfig:
    resample_hz: float = 30.0
    window_s: float = 10.0          # analysis window (live HUD) / whole session (final)
    pos_window_s: float = 1.6       # POS overlap-add window (Wang et al. use ~1.6 s)
    min_duration_s: float = 6.0
    min_coverage: float = 0.6       # fraction of the window with a valid skin sample
    band_hz: tuple = (0.7, 3.0)     # 42-180 bpm
    peak_halfwidth_hz: float = 0.12
    snr_lo_db: float = -6.0         # subscore 0 at/below this SNR
    snr_hi_db: float = 4.0          # subscore 1 at/above this SNR
    regress_challenge: bool = True
    challenge_lag_max_ms: float = 300.0
    challenge_lag_step_ms: float = 20.0
    gate: bool = False              # include in the ACCEPT gate (unmeasured -> off by default)
    pass_threshold: float = 0.3

    @classmethod
    def from_dict(cls, d: dict) -> "PulseConfig":
        kw = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        if "band_hz" in kw:
            kw["band_hz"] = tuple(float(f) for f in kw["band_hz"])
        return cls(**kw)


def skin_rgb(frame_bgr: np.ndarray, boxes: list) -> tuple:
    """Mean (R, G, B) over the ROI boxes, or NaNs when there are none."""
    if not boxes:
        return (float("nan"),) * 3
    mask = boxes_to_mask(frame_bgr.shape, boxes)
    if not mask.any():
        return (float("nan"),) * 3
    b, g, r = (float(frame_bgr[..., c][mask].mean()) for c in range(3))
    return r, g, b


def pos_signal(rgb: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """POS overlap-add. rgb: (N, 3) uniformly sampled, no NaNs."""
    n = len(rgb)
    w = max(3, int(round(window_s * fs)))
    h = np.zeros(n)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    for start in range(0, n - w + 1):
        c = rgb[start:start + w]
        mu = c.mean(axis=0)
        if np.any(mu <= 1e-6):
            continue
        cn = c / mu
        s = cn @ proj.T                    # (w, 2)
        s1, s2 = s[:, 0], s[:, 1]
        sd2 = s2.std()
        alpha = (s1.std() / sd2) if sd2 > 1e-12 else 0.0
        p = s1 + alpha * s2
        h[start:start + w] += p - p.mean()
    return h


@dataclass
class PulseEstimate:
    status: str                    # "ok" | "no_evidence"
    bpm: float = float("nan")
    snr_db: float = float("nan")
    subscore: float | None = None
    coverage: float = 0.0
    challenge_leak: float = 0.0    # |corr| of the pulse signal with the emitted chips before removal
    wave_t: np.ndarray | None = None
    wave: np.ndarray | None = None
    spectrum_hz: np.ndarray | None = None
    spectrum: np.ndarray | None = None
    diagnostics: str = ""


def _bandpass(x: np.ndarray, fs: float, band: tuple) -> np.ndarray:
    sos = scipy.signal.butter(3, band, btype="band", fs=fs, output="sos")
    padlen = min(len(x) - 1, 3 * 7)
    return scipy.signal.sosfiltfilt(sos, x, padlen=padlen)


def estimate_pulse(timestamps, rgb, config: PulseConfig, emitter_log: list | None = None,
                   t_end: float | None = None) -> PulseEstimate:
    """timestamps: perf_counter seconds; rgb: (N, 3) with NaN rows for frames
    without a face. Uses the last config.window_s seconds before t_end (all
    of it when window_s <= 0)."""
    ts = np.asarray(timestamps, dtype=np.float64)
    c = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if len(ts) < 2:
        return PulseEstimate(status="no_evidence", diagnostics="no frames")
    t_end = float(ts[-1]) if t_end is None else float(t_end)
    if config.window_s > 0:
        keep = ts >= t_end - config.window_s
        ts, c = ts[keep], c[keep]
    valid = np.all(np.isfinite(c), axis=1)
    if valid.sum() < 8:
        return PulseEstimate(status="no_evidence", diagnostics=f"only {int(valid.sum())} skin samples")
    span = float(ts[valid][-1] - ts[valid][0])
    coverage = float(valid.mean())
    if span < config.min_duration_s or coverage < config.min_coverage:
        return PulseEstimate(status="no_evidence", coverage=coverage,
                             diagnostics=f"face tracked {span:.1f}s / coverage {coverage:.0%} "
                                         f"(need {config.min_duration_s:.0f}s, {config.min_coverage:.0%})")

    fs = config.resample_hz
    grid = np.arange(ts[valid][0], ts[valid][-1], 1.0 / fs)
    u = np.stack([np.interp(grid, ts[valid], c[valid, k]) for k in range(3)], axis=1)
    h = pos_signal(u, fs, config.pos_window_s)
    h = _bandpass(h - h.mean(), fs, config.band_hz)

    leak = 0.0
    if config.regress_challenge and emitter_log:
        best = None
        for lag_ms in np.arange(0.0, config.challenge_lag_max_ms + 1e-9, config.challenge_lag_step_ms):
            e = resample_emitted(emitter_log, grid, lag_s=lag_ms / 1000.0).astype(np.float64)
            if np.std(e) < 1e-9:
                continue
            e = _bandpass(e - e.mean(), fs, config.band_hz)
            r = abs(float(np.corrcoef(h, e)[0, 1])) if np.std(e) > 1e-12 else 0.0
            if best is None or r > best[0]:
                best = (r, e)
        if best is not None:
            leak, e = best
            beta = float(np.dot(h, e) / np.dot(e, e))
            h = h - beta * e

    n_fft = max(4096, 1 << int(np.ceil(np.log2(len(h) * 4))))
    win = np.hanning(len(h))
    spec = np.abs(np.fft.rfft(h * win, n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    lo, hi = config.band_hz
    band = (freqs >= lo) & (freqs <= hi)
    f0 = float(freqs[band][int(np.argmax(spec[band]))])
    hw = config.peak_halfwidth_hz
    sig_mask = (np.abs(freqs - f0) <= hw) | (np.abs(freqs - 2 * f0) <= 2 * hw)
    ref_mask = (freqs >= lo) & (freqs <= min(2 * hi, fs / 2 - 0.1))
    p_sig = float(spec[sig_mask & ref_mask].sum())
    p_noise = float(spec[ref_mask & ~sig_mask].sum())
    snr_db = float(10 * np.log10(p_sig / p_noise)) if p_sig > 0 and p_noise > 0 else float("nan")
    sub = float(np.clip((snr_db - config.snr_lo_db) / (config.snr_hi_db - config.snr_lo_db), 0.0, 1.0)) \
        if np.isfinite(snr_db) else 0.0
    sd = float(np.std(h))
    wave = h / sd if sd > 1e-12 else h
    disp = (freqs >= 0.5) & (freqs <= 4.0)
    spec_disp = spec[disp] / (spec[disp].max() or 1.0)
    return PulseEstimate(
        status="ok", bpm=f0 * 60.0, snr_db=snr_db, subscore=sub, coverage=coverage, challenge_leak=leak,
        wave_t=grid, wave=wave, spectrum_hz=freqs[disp], spectrum=spec_disp,
        diagnostics=f"bpm={f0 * 60.0:.0f}, snr_db={snr_db:.1f}, coverage={coverage:.0%}"
                    + (f", challenge_leak={leak:.2f} (regressed out)" if leak else ""),
    )


def pulse_record(est: PulseEstimate | None) -> dict | None:
    if est is None:
        return None
    f = lambda v: None if v is None or not np.isfinite(v) else float(v)  # noqa: E731
    return {"status": est.status, "bpm": f(est.bpm), "snr_db": f(est.snr_db), "subscore": est.subscore,
            "coverage": est.coverage, "challenge_leak": est.challenge_leak, "diagnostics": est.diagnostics}


if __name__ == "__main__":
    import argparse
    import time
    from pathlib import Path

    import yaml

    from praesens.camera import open_camera
    from praesens.capture import CaptureConfig, configure_capture_format
    from praesens.optical import OpticalConfig, OpticalFrameProcessor, lock_camera

    parser = argparse.ArgumentParser(
        description="rPPG pulse lane smoke test: face the camera and keep still for N seconds")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--window", type=float, default=0.0,
                        help="analysis window in seconds (0 = the whole capture, like a scored session)")
    args = parser.parse_args()

    REPO_ROOT = Path(__file__).resolve().parent.parent
    raw = yaml.safe_load(open(REPO_ROOT / "config.yaml"))

    oconfig = OpticalConfig.from_dict(raw["optical"])
    model_path = Path(oconfig.model_path)
    oconfig.model_path = str(model_path if model_path.is_absolute() else REPO_ROOT / model_path)
    if args.camera_index is not None:
        oconfig.camera_index = args.camera_index

    puconfig = PulseConfig.from_dict(raw.get("pulse", {}))
    puconfig.window_s = args.window        # 0 -> estimate over everything captured

    cap = open_camera(oconfig.camera_index)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {oconfig.camera_index}")

    warns: list = []
    configure_capture_format(cap, CaptureConfig.from_dict(raw.get("capture", {})), warns)
    # same exposure lock the scored pipeline runs under: rPPG at a locked, dim
    # exposure is the honest condition, not auto-exposure's brighter frames
    exposure_locked = lock_camera(cap, oconfig, warns)
    for w in warns:
        print(f"WARNING: {w}")
    for _ in range(int(raw["optical"].get("camera_warmup_frames", 30))):
        cap.grab()

    proc = OpticalFrameProcessor(oconfig)   # emitter=None: no challenge is emitted for this check
    skin: list = []
    start = time.perf_counter()
    print(f"capturing {args.seconds:.0f}s at exposure_value={oconfig.exposure_value} "
          f"(locked={exposure_locked}) -- hold still, face the camera", flush=True)
    try:
        while time.perf_counter() - start < args.seconds:
            if not cap.grab():
                continue
            t = time.perf_counter()
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            msg = proc.process_frame(frame, t, start, None, None)
            if msg:
                print(f"WARNING: {msg}")
            skin.append((t, skin_rgb(frame, proc.detector.last_boxes)))
    finally:
        proc.close()
        cap.release()

    n = len(skin)
    span = (skin[-1][0] - skin[0][0]) if n > 1 else 0.0
    face = sum(1 for _, c in skin if np.all(np.isfinite(c)))
    print(f"frames={n} span={span:.1f}s fps={(n - 1) / span if span else float('nan'):.1f} "
          f"face_detected={face}/{n} ({face / n:.0%})" if n else "no frames captured")

    est = estimate_pulse([t for t, _ in skin], [c for _, c in skin], puconfig)
    print(f"status={est.status} bpm={est.bpm:.1f} snr_db={est.snr_db:.2f} "
          f"subscore={est.subscore} coverage={est.coverage:.0%}")
    print(f"  {est.diagnostics}")
    print(f"  (advisory lane: pulse.gate={puconfig.gate}, pass_threshold={puconfig.pass_threshold})")
