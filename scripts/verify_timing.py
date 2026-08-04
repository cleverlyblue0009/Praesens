"""FIX 3 diagnostic: verify the emitter and capture timestamps share a
common clock with a sane relative offset, and show the FULL correlation-
vs-lag curve for a short real session -- not just its peak.

A lag consistently pinned at the grid's smallest step (e.g. always exactly
5ms) has two very different possible explanations that look identical if
you only ever print the argmax: a genuine clock/rebasing bug (timestamps
from two different clocks, or rebased on one side but not the other), or a
flat, noise-dominated correlation curve where no lag has a real peak and
the argmax is just picking out whichever tiny bit of noise happened to be
highest. This script checks the first directly (raw timestamps, side by
side) and makes the second impossible to miss (the whole curve, not the
winner).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from mediapipe.tasks.python import vision as mp_vision

from praesens.challenge import Challenge
from praesens.emit import Emitter, EmitterConfig
from praesens.optical import (
    OpticalConfig, CadenceDetector, create_landmarker, lock_camera,
    moving_average_detrend, compute_correlation_curve,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="FIX 3 timing/correlation-curve diagnostic")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--n-print", type=int, default=20,
                         help="how many frames to print raw timestamp comparisons for")
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)

    challenge = Challenge(**raw["challenge"])
    econfig = EmitterConfig.from_dict(raw["emitter"])
    oconfig = OpticalConfig.from_dict(raw["optical"])
    model_path = Path(oconfig.model_path)
    oconfig.model_path = str(model_path if model_path.is_absolute() else REPO_ROOT / model_path)

    cap = cv2.VideoCapture(oconfig.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {oconfig.camera_index}")

    warn_list: list = []
    lock_camera(cap, oconfig, warn_list)
    for w in warn_list:
        print(f"WARNING: {w}")

    landmarker = create_landmarker(oconfig.model_path, mp_vision.RunningMode.VIDEO, oconfig.min_face_confidence)
    detector = CadenceDetector(landmarker, oconfig.detect_every_n_frames, oconfig.detect_downscale_width)

    emitter = Emitter(challenge, econfig)
    start_time = time.perf_counter()
    emitter.start(start_time, args.seconds)

    print(f"Running a {args.seconds}s session -- sit in front of the camera for a meaningful "
          f"correlation curve (the timestamp comparison below works either way).")

    timestamps, luminances = [], []
    last_ts_ms = -1
    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= args.seconds:
                break
            grabbed = cap.grab()
            t = time.perf_counter()
            if not grabbed:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            ts_ms = max(last_ts_ms + 1, int((t - start_time) * 1000))
            last_ts_ms = ts_ms
            lum, face_ok = detector.process(frame, ts_ms, oconfig.roi_margin_frac)
            timestamps.append(t)
            luminances.append(lum)
    finally:
        emitter.stop()
        cap.release()
        landmarker.close()

    ts_arr = np.array(timestamps)
    lum_arr = np.array(luminances)
    emitter_log = emitter.get_log()
    log_t = np.array([e["t"] for e in emitter_log])

    if len(ts_arr) == 0 or len(log_t) == 0:
        print("No frames captured or no emitter redraws logged -- can't diagnose timing "
              "with zero data on one side. Check the camera and display are both working "
              "(see scripts/probe.py).")
        return

    print(f"\ncaptured {len(ts_arr)} frames, emitter logged {len(emitter_log)} redraws")
    print(f"frame timestamp range:   [{ts_arr[0]:.4f}, {ts_arr[-1]:.4f}]  (perf_counter seconds)")
    print(f"emitter timestamp range: [{log_t[0]:.4f}, {log_t[-1]:.4f}]  (perf_counter seconds)")
    sane = log_t[0] <= ts_arr[0] + 0.5 and ts_arr[-1] <= log_t[-1] + 0.5
    print(f"ranges overlap as expected: {sane}  "
          f"{'(same clock, sane relative offset)' if sane else '(SUSPICIOUS -- check for a rebasing bug)'}")

    print(f"\nFirst {args.n_print} frames: raw frame timestamp vs nearest PRIOR emitter "
          f"redraw, and their difference (eyeball whether these look like the same timebase):")
    print(f"{'frame_t':>14s} {'nearest_emit_t':>16s} {'diff_ms':>10s}")
    for t in ts_arr[:args.n_print]:
        idx = np.searchsorted(log_t, t, side="right") - 1
        idx = int(np.clip(idx, 0, len(log_t) - 1))
        diff_ms = (t - log_t[idx]) * 1000
        print(f"{t:14.4f} {log_t[idx]:16.4f} {diff_ms:10.2f}")

    detrended = moving_average_detrend(ts_arr, lum_arr, oconfig.detrend_window_s)
    n_valid = int(np.sum(~np.isnan(detrended)))
    print(f"\n{n_valid}/{len(ts_arr)} frames had a valid (face-detected) luminance sample")
    if n_valid < 5:
        print("Not enough valid samples for a correlation curve -- sit in front of the "
              "camera and re-run to see the curve.")
        return

    lags_ms, scores = compute_correlation_curve(detrended, ts_arr, emitter_log,
                                                  oconfig.lag_search_max_ms, oconfig.lag_step_ms)
    peak_idx = int(np.argmax(scores))
    print(f"\nFull correlation-vs-lag curve ({len(lags_ms)} points, peak marked):")
    for l, s in zip(lags_ms, scores):
        marker = " <-- PEAK" if l == lags_ms[peak_idx] else ""
        print(f"  {l:6.1f} ms: {s:+.4f}{marker}")

    curve_range = float(scores.max() - scores.min())
    print(f"\npeak={scores[peak_idx]:.4f} at {lags_ms[peak_idx]:.1f}ms; "
          f"curve range={curve_range:.4f}, std={scores.std():.4f}")
    if curve_range < 0.15:
        print("\nFLAT CURVE: no lag stands out from the noise floor. This is a SIGNAL "
              "problem (see FIX 2 -- undersampling, weak modulation depth, distance from "
              "the screen), not a clock/timestamp alignment bug. The argmax lag reported "
              "above is essentially arbitrary noise and should not be trusted as a real "
              "latency measurement.")
    else:
        print("\nCurve shows real structure around the peak. If the peak lag still looks "
              "physically implausible (e.g. exactly 0ms, or pinned at the same value every "
              "run), that's still worth a second look, but the timing/alignment mechanism "
              "itself is not obviously broken.")


if __name__ == "__main__":
    main()
