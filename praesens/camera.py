"""Cross-platform camera backend selection.

OpenCV's VideoCapture backend flag is platform-specific: CAP_DSHOW/CAP_MSMF
are Windows-only, CAP_AVFOUNDATION is macOS-only, CAP_V4L2 is Linux-only.
Passing the wrong one for the current OS doesn't raise an error -- the
capture just silently fails to open, which looks identical to a camera
permissions problem. This picks the right backend for the current OS
automatically, falling back to CAP_ANY if the preferred backend can't
open the device.
"""
from __future__ import annotations

import platform

import cv2


def default_backend() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    elif system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a camera using the right backend for this OS, with a CAP_ANY fallback."""
    cap = cv2.VideoCapture(index, default_backend())
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
    return cap