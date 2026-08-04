"""Milestone 7: attack harness.

Extends the live dashboard (Milestone 6) with instant camera-source
switching -- the moment the demo actually turns on. "Instant" ruled out the
obvious approach: closing and reopening a cv2.VideoCapture for the new
source on keypress, since opening a capture device (especially over DSHOW
on Windows) routinely takes several hundred milliseconds, which would show
up as a visible stall right at the moment meant to look effortless. Instead
every available camera source (real webcam, OBS Virtual Camera, whatever
else is enumerated) is opened and exposure-locked once at startup, and a
keypress just swaps which already-open capture handle the main loop reads
from -- a pointer swap, not a device open.

Just as important: switching does NOT clear the rolling score/trace
buffers. The whole point is that the audience watches the SAME rolling
window's score visibly collapse as frames from the new source arrive,
which reads as "the physics stopped matching" -- a hard reset before
showing the new source would hide that transition and look like a scripted
cut instead of a live measurement.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import yaml

from demo.live import LiveDashboard, REPO_ROOT
from praesens.optical import lock_camera


def probe_camera(index: int, oconfig, warn_prefix: str = "") -> cv2.VideoCapture | None:
    """Opens, verifies a frame is actually deliverable, and exposure-locks
    a camera index. Returns None (and releases) if the index doesn't work,
    rather than raising -- enumeration expects most indices to be empty."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None
    warn_list: list = []
    lock_camera(cap, oconfig, warn_list)
    for w in warn_list:
        print(f"{warn_prefix}WARNING: {w}")
    return cap


class AttackDashboard(LiveDashboard):
    def __init__(self, raw_config: dict, max_camera_probe: int = 6):
        super().__init__(raw_config)  # opens+locks the configured primary camera as self.cap

        primary_index = self.oconfig.camera_index
        self._captures: dict[int, cv2.VideoCapture] = {primary_index: self.cap}

        print(f"Probing camera indices 0-{max_camera_probe - 1} for additional sources "
              f"(e.g. OBS Virtual Camera)...")
        for idx in range(max_camera_probe):
            if idx == primary_index:
                continue
            cap = probe_camera(idx, self.oconfig, warn_prefix=f"[camera {idx}] ")
            if cap is not None:
                self._captures[idx] = cap

        self.camera_slots = sorted(self._captures.keys())
        print("Available camera sources (press the number key shown to switch instantly; "
              "rolling score history is preserved across the switch):")
        for i, idx in enumerate(self.camera_slots):
            marker = " (current)" if idx == primary_index else ""
            print(f"  [{i}] camera index {idx}{marker}")

    def switch_camera(self, new_index: int) -> None:
        if new_index not in self._captures or new_index == self.oconfig.camera_index:
            return
        self.cap = self._captures[new_index]
        self.oconfig.camera_index = new_index
        self._camera_name = self.dconfig.camera_display_name or f"camera #{new_index}"
        self.last_ts_ms = -1  # each capture's frame clock restarts; keep grab timestamps monotonic per-source
        print(f"-> switched to camera index {new_index} (instant; rolling history preserved)")
        # Deliberately NOT calling self.reset(): the score collapsing in the
        # SAME rolling window is the point of the demo.

    def _handle_key(self, key: int, frame) -> bool:
        for i, idx in enumerate(self.camera_slots):
            if key == ord(str(i)) and i < 10:
                self.switch_camera(idx)
                return True
        return super()._handle_key(key, frame)

    def run(self, max_seconds: float | None = None) -> None:
        try:
            super().run(max_seconds=max_seconds)  # releases self.cap (the active one) in its own finally
        finally:
            active = self.cap
            for idx, cap in self._captures.items():
                if cap is not active:
                    cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 7 attack harness")
    parser.add_argument("--max-seconds", type=float, default=None,
                         help="auto-stop after N seconds (smoke-test use; omit for real demo use)")
    parser.add_argument("--max-camera-probe", type=int, default=6,
                         help="how many camera indices to probe at startup")
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)

    dashboard = AttackDashboard(raw, max_camera_probe=args.max_camera_probe)
    print("Attack harness running. Keys: [0-9] switch camera source, [E] toggle emitter, "
          "[R] reset, [S] screenshot, [Q] quit.")
    dashboard.run(max_seconds=args.max_seconds)
