"""Milestone 4: session runner.

Ties challenge + emitter + optical lane together for one 20 s session and
writes a single JSON log record. `condition` is recorded but never
influences how the session is measured -- bonafide/replay/swap/emitter_off
all run through the identical capture and scoring path, and only differ in
what's physically in front of the camera or whether the emitter is on. That
is what makes the emitter_off condition a real scientific control rather
than a special-cased "off" mode: the pipeline can't tell it apart from any
other session except by the light pattern itself. Metadata (lighting,
distance, makeup, glasses, subject, skin tone) is recorded per session
because Milestone 5's analysis needs it to check the system doesn't fail
quietly for some conditions more than others.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import cv2
import yaml

from praesens.challenge import Challenge
from praesens.emit import Emitter, EmitterConfig
from praesens.optical import OpticalConfig, run_session

VALID_CONDITIONS = {"bonafide", "replay", "swap", "emitter_off"}

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_session_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


def run_one_session(condition: str, meta: dict, raw_config: dict | None = None,
                     camera_index_override: int | None = None,
                     output_dir: str | Path = "logs") -> tuple[dict, Path]:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}, got {condition!r}")

    if raw_config is None:
        raw_config = load_config()

    challenge = Challenge(**raw_config["challenge"])

    econfig = EmitterConfig.from_dict(raw_config["emitter"])
    if condition == "emitter_off":
        econfig.emitter_enabled = False

    oconfig = OpticalConfig.from_dict(raw_config["optical"])
    model_path = Path(oconfig.model_path)
    oconfig.model_path = str(model_path if model_path.is_absolute() else REPO_ROOT / model_path)
    if camera_index_override is not None:
        oconfig.camera_index = camera_index_override

    cap = cv2.VideoCapture(oconfig.camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {oconfig.camera_index}")

    emitter = Emitter(challenge, econfig)
    session_id = generate_session_id()

    start_time = time.perf_counter()
    emitter.start(start_time, challenge.duration_s)
    try:
        result = run_session(cap, challenge, oconfig, start_time, challenge.duration_s, emitter=emitter)
    finally:
        emitter.stop()
        cap.release()

    record = {
        "session": session_id,
        "condition": condition,
        "score": result.score,
        "lag_ms": result.lag_ms,
        "seed": challenge.seed,
        "emitter_enabled": econfig.emitter_enabled,
        "snr_db": result.snr_db,
        "insufficient_signal": result.insufficient_signal,
        "adaptive_boost_applied": result.adaptive_boost_applied,
        "exposure_locked": result.exposure_locked,
        "n_frames": result.n_frames,
        "n_face_detected": result.n_face_detected,
        "warnings": result.warnings,
        "meta": meta,
        "trace_emitted": result.trace_emitted,
        "trace_measured": result.trace_measured,
        "timestamps": result.timestamps,
    }

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return record, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 4 session runner")
    parser.add_argument("--condition", required=True, choices=sorted(VALID_CONDITIONS))
    parser.add_argument("--lighting", default="normal",
                         choices=["normal", "dim", "backlit", "side-lit"])
    parser.add_argument("--distance-cm", type=float, default=60.0)
    parser.add_argument("--makeup", default="none", choices=["none", "foundation_powder"])
    parser.add_argument("--glasses", default="without", choices=["with", "without"])
    parser.add_argument("--subject", default="unknown")
    parser.add_argument("--skin-tone", type=int, default=None,
                         help="Fitzpatrick scale 1-6, self-reported")
    parser.add_argument("--camera-index", type=int, default=None)
    args = parser.parse_args()

    meta = {
        "lighting": args.lighting,
        "distance_cm": args.distance_cm,
        "makeup": args.makeup,
        "glasses": args.glasses,
        "subject": args.subject,
        "skin_tone": args.skin_tone,
    }

    raw_config = load_config()
    print(f"Starting {args.condition} session ({raw_config['challenge']['duration_s']}s)... look at the screen.")
    record, out_path = run_one_session(args.condition, meta, raw_config=raw_config,
                                        camera_index_override=args.camera_index)

    print(f"\nsession={record['session']} condition={record['condition']}")
    print(f"score={record['score']:.3f} lag_ms={record['lag_ms']:.1f} "
          f"snr_db={record['snr_db']:.2f} insufficient_signal={record['insufficient_signal']}")
    print(f"n_frames={record['n_frames']} n_face_detected={record['n_face_detected']} "
          f"exposure_locked={record['exposure_locked']} adaptive_boost={record['adaptive_boost_applied']}")
    for w in record["warnings"]:
        print(f"  warning: {w}")
    print(f"saved to {out_path}")
