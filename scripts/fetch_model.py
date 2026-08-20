"""One-time setup: download the MediaPipe FaceLandmarker and (Milestone 10)
HandLandmarker model bundles.

mediapipe 1.0's Tasks API (used by praesens/optical.py and praesens/typing.py)
needs these assets to run at all -- neither is bundled with the pip package.
Both are official Google-hosted model assets, documented at
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker and
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker.
"""
from pathlib import Path
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parent.parent

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_PATH = REPO_ROOT / "models" / "face_landmarker.task"

HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = REPO_ROOT / "models" / "hand_landmarker.task"


def _ensure(url: str, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {path} ...")
    urlretrieve(url, path)
    print(f"Done ({path.stat().st_size} bytes).")
    return path


def ensure_model() -> Path:
    """Back-compat name -- face model only."""
    return _ensure(FACE_MODEL_URL, FACE_MODEL_PATH)


def ensure_hand_model() -> Path:
    return _ensure(HAND_MODEL_URL, HAND_MODEL_PATH)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch MediaPipe model bundles")
    parser.add_argument("--hand", action="store_true", help="fetch only the hand landmarker model")
    parser.add_argument("--face", action="store_true", help="fetch only the face landmarker model")
    args = parser.parse_args()

    if not args.hand and not args.face:
        ensure_model()
        ensure_hand_model()
    else:
        if args.face:
            ensure_model()
        if args.hand:
            ensure_hand_model()
