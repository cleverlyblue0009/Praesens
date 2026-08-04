"""DIAGNOSTIC: emitter-off control test -- the single most informative test
in this repo.

Runs three back-to-back sessions with a real person present, changing only
the light pattern between them:
  1. emitter ON, normal modulation depth
  2. emitter OFF (the control)
  3. emitter ON, maximum modulation depth

Session 2 is the whole point. If it does NOT collapse toward zero
correlation, the pipeline is measuring something other than the emitted
pattern -- ambient flicker near the chip rate, the subject's own
micro-movements, auto-exposure fighting a now-constant screen and
producing a coincidentally-correlated drift -- and every other number in
this repo (including tonight's other fixes) is moot: a high score in
session 1 would mean "looks like session 1 used to look," not liveness.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from praesens.challenge import Challenge
from praesens.emit import Emitter, EmitterConfig
from praesens.optical import OpticalConfig, run_session

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_one_diagnostic(label: str, cap, oconfig, raw_config: dict, emitter_enabled: bool,
                        modulation_depth: float, duration_s: float | None = None) -> dict:
    """Each call lets run_session() create and close its OWN FaceLandmarker
    (the normal, already-tested pattern -- see session.py) rather than
    sharing one across all three diagnostic sessions: a shared landmarker
    was tried first here and broke, because MediaPipe's VIDEO-mode API
    requires timestamps to keep increasing for the landmarker's ENTIRE
    lifetime, but each session computes its own ts_ms relative to its own
    fresh start_time -- session 2's timestamps restart near zero, lower
    than what session 1 already fed the same landmarker instance, and it
    correctly rejects that as non-monotonic. Reloading the model three
    times costs about a second total; not worth the cross-session
    timestamp bookkeeping to avoid."""
    challenge_cfg = dict(raw_config["challenge"])
    if duration_s is not None:
        challenge_cfg["duration_s"] = duration_s
    challenge = Challenge(**challenge_cfg)

    econfig = EmitterConfig.from_dict(raw_config["emitter"])
    econfig.emitter_enabled = emitter_enabled
    econfig.modulation_depth = modulation_depth

    emitter = Emitter(challenge, econfig)
    start_time = time.perf_counter()
    emitter.start(start_time, challenge.duration_s)
    try:
        result = run_session(cap, challenge, oconfig, start_time, challenge.duration_s,
                              emitter=emitter)
    finally:
        emitter.stop()

    samples_per_chip = (result.measured_fps / challenge.chip_rate_hz
                         if not np.isnan(result.measured_fps) else float("nan"))

    return {
        "label": label,
        "emitter_enabled": emitter_enabled,
        "modulation_depth": modulation_depth,
        "measured_fps": result.measured_fps,
        "chips_emitted": challenge.n_chips,
        "samples_per_chip": samples_per_chip,
        "snr_db": result.snr_db,
        "score": result.score,
        "lag_ms": result.lag_ms,
        "insufficient_signal": result.insufficient_signal,
        "n_face_detected": result.n_face_detected,
        "n_frames": result.n_frames,
        "warnings": result.warnings,
    }


def print_table(rows: list) -> None:
    print("\n" + "=" * 112)
    print("DIAGNOSTIC: emitter-off control comparison")
    print("=" * 112)
    fmt = "{:<26s} {:>10s} {:>8s} {:>10s} {:>9s} {:>8s} {:>9s} {:>14s}"
    print(fmt.format("session", "fps", "chips", "smpl/chip", "snr_db", "score", "lag_ms", "face/frames"))
    for r in rows:
        print(fmt.format(
            r["label"], f"{r['measured_fps']:.1f}", str(r["chips_emitted"]),
            f"{r['samples_per_chip']:.1f}",
            "nan" if np.isnan(r["snr_db"]) else f"{r['snr_db']:.1f}",
            f"{r['score']:.3f}", f"{r['lag_ms']:.1f}",
            f"{r['n_face_detected']}/{r['n_frames']}",
        ))
    print("=" * 112)

    if len(rows) < 3:
        return

    on_normal, off_control, on_max = rows[0], rows[1], rows[2]

    print(f"\nControl check: emitter OFF score = {off_control['score']:.3f} "
          f"(should be near zero, clearly below the ON sessions)")
    collapsed = off_control["score"] < 0.3 and off_control["score"] < 0.5 * max(on_normal["score"], 1e-9)
    if not collapsed:
        print("!" * 70)
        print("WARNING: the OFF control did NOT clearly collapse relative to the ON")
        print("sessions. The pipeline may be measuring something other than the emitted")
        print("pattern -- do not trust ON-session scores until this is understood. Check:")
        print("  - ambient light flicker (mains, other displays) near the chip rate")
        print("  - auto-exposure fighting the now-constant screen, producing a luminance")
        print("    drift that coincidentally correlates with part of the LFSR schedule")
        print("  - the emitted signal actually being the ACTUAL logged redraws (not a")
        print("    stale/idealised schedule) -- see scripts/verify_timing.py")
        print("!" * 70)
    else:
        print("OK: OFF control collapsed relative to the ON sessions -- the score is "
              "tracking the light pattern, not just the subject's presence.")

    print(f"\nModulation depth check: normal={on_normal['score']:.3f} vs "
          f"max_depth={on_max['score']:.3f} (max depth should be >= normal; a big jump "
          f"suggests the normal session was reflectance/distance-limited, not detection-limited)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Emitter-off control diagnostic")
    parser.add_argument("--seconds", type=float, default=None,
                         help="override session duration (smoke-test use; omit for the real 20s test)")
    parser.add_argument("--no-prompt", action="store_true",
                         help="skip the Enter-to-continue prompts between sessions (smoke-test use)")
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw_config = yaml.safe_load(f)

    oconfig = OpticalConfig.from_dict(raw_config["optical"])
    model_path = Path(oconfig.model_path)
    oconfig.model_path = str(model_path if model_path.is_absolute() else REPO_ROOT / model_path)

    cap = cv2.VideoCapture(oconfig.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {oconfig.camera_index}")

    base_depth = raw_config["emitter"]["modulation_depth"]
    max_depth = raw_config["emitter"]["max_modulation_depth"]

    print("This runs THREE back-to-back sessions. Sit in front of the camera and stay")
    print("there for all three -- only the light pattern changes between them.\n")

    def prompt(msg: str) -> None:
        if not args.no_prompt:
            input(msg)
        else:
            print(msg)

    rows = []
    try:
        prompt("Press Enter to start session 1/3 (emitter ON, normal depth)...")
        rows.append(run_one_diagnostic("1: emitter ON, normal", cap, oconfig, raw_config,
                                        emitter_enabled=True, modulation_depth=base_depth,
                                        duration_s=args.seconds))

        prompt("\nPress Enter to start session 2/3 (emitter OFF -- the control)...")
        rows.append(run_one_diagnostic("2: emitter OFF (control)", cap, oconfig, raw_config,
                                        emitter_enabled=False, modulation_depth=base_depth,
                                        duration_s=args.seconds))

        prompt("\nPress Enter to start session 3/3 (emitter ON, MAX depth)...")
        rows.append(run_one_diagnostic("3: emitter ON, max depth", cap, oconfig, raw_config,
                                        emitter_enabled=True, modulation_depth=max_depth,
                                        duration_s=args.seconds))
    finally:
        cap.release()

    print_table(rows)


if __name__ == "__main__":
    main()
