"""Milestone 12: records a clean N-second video of the operator's face
under a chosen lighting/makeup condition, for later use as attack source
material (fed into attacks/adaptive_injector.py --source, or used
directly as an OBS Virtual Camera source).

Saved under data/ (gitignored, hard rule 2 -- attack videos, recorded
sessions, and face imagery never go in git). The sidecar JSON records only
RECORDING metadata (duration, condition/lighting/makeup/subject labels,
timestamp, frame count/resolution) -- no different in kind from what
session logs already record about a session, just alongside a video file
instead of numeric traces, and that sidecar itself also lives under data/,
never committed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from praesens.capture import CaptureConfig, configure_capture_format

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"


def record_source(camera_index: int, seconds: float, fps: float, out_path: Path,
                   width: int = 1280, height: int = 720, fourcc: str | None = "MJPG") -> dict:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    cconfig = CaptureConfig(fourcc=fourcc, width=width, height=height, requested_fps=fps)
    format_info = configure_capture_format(cap, cconfig)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output codec for the recorded FILE -- unrelated to (and named
    # differently from) the INPUT capture's fourcc set just above via
    # configure_capture_format(), which is the negotiated pixel format
    # coming OFF the camera, not the codec this file gets encoded with.
    output_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), output_fourcc, fps, (actual_w, actual_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open VideoWriter for {out_path} (codec 'mp4v' unavailable?)")

    print(f"Recording {seconds}s to {out_path} ...")
    start = time.perf_counter()
    n_frames = 0
    try:
        while time.perf_counter() - start < seconds:
            ok, frame = cap.read()
            if not ok:
                continue
            writer.write(frame)
            n_frames += 1
    finally:
        writer.release()
        cap.release()

    elapsed = time.perf_counter() - start
    print(f"Done: {n_frames} frames in {elapsed:.1f}s ({n_frames / max(elapsed, 1e-9):.1f} fps)")
    return {"n_frames": n_frames, "elapsed_s": elapsed, "width": actual_w, "height": actual_h,
            "fps_requested": fps, "input_capture_format": format_info}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 12: record clean attack source material")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--condition-label", default="clean",
                         help="free-text label for what this recording is (e.g. 'normal', 'foundation_powder')")
    parser.add_argument("--lighting", default="normal")
    parser.add_argument("--makeup", default="none")
    parser.add_argument("--subject", default="unknown")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG",
                         help="input capture pixel format, e.g. MJPG; 'none' leaves the driver default")
    parser.add_argument("--no-prompt", action="store_true", help="skip the Enter-to-start prompt")
    args = parser.parse_args()
    fourcc = None if args.fourcc.lower() == "none" else args.fourcc

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    session_id = time.strftime("%Y%m%dT%H%M%S")
    out_path = DATA_DIR / f"source_{session_id}.mp4"

    print(f"\nRECORDING SOURCE MATERIAL -- condition={args.condition_label} lighting={args.lighting} "
          f"makeup={args.makeup}")
    if not args.no_prompt:
        input("Press ENTER when ready...")

    stats = record_source(args.camera_index, args.seconds, args.fps, out_path, fourcc=fourcc)

    meta = {
        "session": session_id, "condition_label": args.condition_label,
        "lighting": args.lighting, "makeup": args.makeup, "subject": args.subject,
        "camera_index": args.camera_index, **stats,
        "video_path": str(out_path.relative_to(REPO_ROOT)),
    }
    meta_path = DATA_DIR / f"source_{session_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {out_path}")
    print(f"saved {meta_path}")
