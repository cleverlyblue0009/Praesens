"""One-time setup: download the MediaPipe FaceLandmarker model bundle.

mediapipe 1.0's Tasks API (used by praesens/optical.py) needs this asset to
run at all -- it's not bundled with the pip package. Official Google-hosted
model asset, documented at https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker.
"""
from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"


def ensure_model() -> Path:
    if MODEL_PATH.exists():
        return MODEL_PATH
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_URL} -> {MODEL_PATH} ...")
    urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Done ({MODEL_PATH.stat().st_size} bytes).")
    return MODEL_PATH


if __name__ == "__main__":
    ensure_model()
