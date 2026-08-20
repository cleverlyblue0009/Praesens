"""Milestone 12: adaptive injector -- the strongest attack this repo
builds against itself, so Milestone 9's spatial defence is tested against
a REAL implementation of the attack it exists to catch, not only the
synthetic unit test standing in for one.

Screen-captures the emitter's border region(s) via mss, estimates the
currently-emitted waveform(s) in real time from what a real adversary
could actually see (this code is never given the session seed, the true
chip values, or any other privileged channel -- only pixel brightness of
a screen region, exactly what someone pointing a second camera at their
own monitor would have), and multiplies a source video's luminance by
that estimate -- with a configurable processing lag -- before outputting
via a virtual camera. --mode global applies ONE estimate uniformly (the
attack Milestone 9's tests already prove the spatial score catches);
--mode per-zone estimates left/right/top separately and modulates
matching THIRDS of the source frame, the strongest plausible version of
this attack without also running live face detection inside the attacker
tool itself.

This tool captures NOTHING from a stranger's webcam and has no access to
any real identity beyond whatever video is explicitly supplied to it as
--source: either a pre-recorded file (see scripts/record_source.py,
stored under the gitignored data/) or a live secondary camera the
operator points at attack material they already have. It only re-renders
material already in the operator's possession.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from praesens.emit import EmitterConfig, _screen_resolution
from praesens.optical import resample_emitted  # reused, not reimplemented -- same zero-order-hold lookup


def border_regions(width: int, height: int, border_fraction: float, zone_names: tuple) -> dict:
    """Screen-pixel regions matching Emitter/SpatialEmitter's OWN border
    geometry, so the estimate is sampled from exactly where the real
    pattern is actually drawn -- not a guess at where the border might be."""
    bw = int(width * border_fraction)
    bh = int(height * border_fraction)
    regions = {}
    if "left" in zone_names:
        regions["left"] = (0, 0, max(1, bw), height)
    if "right" in zone_names:
        regions["right"] = (width - bw, 0, max(1, bw), height)
    if "top" in zone_names:
        regions["top"] = (0, 0, width, max(1, bh))
    if "global" in zone_names:
        # A thin strip along all four edges combined -- a crude but
        # plausible single-region estimate for an attacker not attempting
        # per-zone separation at all.
        regions["global"] = (0, 0, width, max(1, bh))
    return regions


def region_luminance(screen_grabber, region: tuple) -> float:
    """region = (left, top, width, height) in screen pixel coordinates.
    screen_grabber is an mss.mss() instance, passed in rather than
    constructed here so callers control its lifecycle."""
    x, y, w, h = region
    shot = screen_grabber.grab({"left": int(x), "top": int(y), "width": int(w), "height": int(h)})
    frame = np.array(shot)  # BGRA
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    return float(np.mean(gray))


def luminance_multiply(frame: np.ndarray, estimate: float, depth: float, baseline: float = 127.0) -> np.ndarray:
    """Applies the attack to one frame region: scales ONLY the luminance
    (Y) channel by a multiplier derived from the estimated brightness,
    colour preserved -- what a real attacker doing this would actually
    do, not a crude all-channel scale that would also shift colour."""
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb).astype(np.float64)
    multiplier = 1.0 + float(np.clip(depth * (estimate - baseline) / baseline, -0.6, 0.6))
    ycrcb[:, :, 0] = np.clip(ycrcb[:, :, 0] * multiplier, 0, 255)
    return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def apply_global_multiply(frame: np.ndarray, estimate: float, depth: float, baseline: float = 127.0) -> np.ndarray:
    return luminance_multiply(frame, estimate, depth, baseline)


def apply_zone_multiply(frame: np.ndarray, zone_estimates: dict, depth: float, baseline: float = 127.0) -> np.ndarray:
    """--mode per-zone: splits the frame into top/left/right thirds (the
    best a screen-capture-only adversary can plausibly do without ALSO
    running face detection on its own source video) and modulates each
    region by its own zone's estimate independently."""
    h, w = frame.shape[:2]
    out = frame.copy()
    if "top" in zone_estimates:
        y1 = int(h * 0.4)
        out[:y1, :] = luminance_multiply(out[:y1, :], zone_estimates["top"], depth, baseline)
    if "left" in zone_estimates:
        x1 = int(w * 0.35)
        out[:, :x1] = luminance_multiply(out[:, :x1], zone_estimates["left"], depth, baseline)
    if "right" in zone_estimates:
        x0 = int(w * 0.65)
        out[:, x0:] = luminance_multiply(out[:, x0:], zone_estimates["right"], depth, baseline)
    return out


@dataclass
class InjectorConfig:
    mode: str = "global"                # "global" | "per-zone"
    lag_ms: float = 50.0                # attacker's own estimation/processing delay, applied deliberately
    depth: float = 1.0                  # attack modulation strength, relative to the estimate's own swing
    baseline_luminance: float = 127.0
    fps: float = 30.0
    border_fraction: float = 0.25
    zone_names: tuple = ("left", "right", "top")
    history_seconds: float = 2.0        # how much screen-sample history to keep for the lag lookup

    @classmethod
    def from_dict(cls, d: dict) -> "InjectorConfig":
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "zone_names" in kwargs:
            kwargs["zone_names"] = tuple(kwargs["zone_names"])
        return cls(**kwargs)


class AdaptiveInjector:
    """Wires screen-capture -> waveform estimate -> luminance-multiply ->
    virtual-camera output together. mss/pyvirtualcam are only imported and
    constructed inside run() (not __init__), so the pure estimate/multiply
    logic above can be unit-tested with zero screen or virtual-camera
    hardware -- and so this class can be constructed and its regions
    inspected without side effects."""

    def __init__(self, source_path_or_index, config: InjectorConfig, econfig: EmitterConfig):
        self.source_path_or_index = source_path_or_index
        self.config = config
        self.width, self.height = _screen_resolution(econfig.fallback_window_width, econfig.fallback_window_height)
        zone_names = config.zone_names if config.mode == "per-zone" else ("global",)
        self.regions = border_regions(self.width, self.height, config.border_fraction, zone_names)
        self._logs = {name: [] for name in self.regions}  # {"t":..., "chip_value":...} shape, reused via resample_emitted

    def record_sample(self, zone: str, t: float, luminance: float) -> None:
        self._logs[zone].append({"t": t, "chip_value": luminance})
        cutoff = t - self.config.history_seconds
        while self._logs[zone] and self._logs[zone][0]["t"] < cutoff:
            self._logs[zone].pop(0)

    def estimate_at(self, zone: str, now: float) -> float:
        """Looks up the zone's estimate exactly lag_ms in the past --
        resample_emitted's zero-order-hold logic reused directly, not
        reimplemented, since this is exactly the same "look up the most
        recent logged value at query_t - lag" operation."""
        log = self._logs.get(zone, [])
        if not log:
            return self.config.baseline_luminance
        value = resample_emitted(log, np.array([now]), lag_s=self.config.lag_ms / 1000.0)
        return float(value[0])

    def process_frame(self, source_frame: np.ndarray, now: float) -> np.ndarray:
        resized = cv2.resize(source_frame, (self.width, self.height))
        if self.config.mode == "global":
            estimate = self.estimate_at("global", now)
            return apply_global_multiply(resized, estimate, self.config.depth, self.config.baseline_luminance)
        zone_estimates = {name: self.estimate_at(name, now) for name in self.regions}
        return apply_zone_multiply(resized, zone_estimates, self.config.depth, self.config.baseline_luminance)

    def run(self, seconds: float | None = None) -> None:
        import mss
        import pyvirtualcam

        cap = cv2.VideoCapture(self.source_path_or_index)
        if not cap.isOpened():
            raise RuntimeError(f"could not open source {self.source_path_or_index!r}")
        is_file = isinstance(self.source_path_or_index, str)

        with mss.mss() as sct, pyvirtualcam.Camera(width=self.width, height=self.height,
                                                     fps=self.config.fps) as vcam:
            print(f"AdaptiveInjector running: mode={self.config.mode} lag_ms={self.config.lag_ms} "
                  f"depth={self.config.depth} -> virtual camera {vcam.device}")
            start = time.perf_counter()
            frame_period = 1.0 / self.config.fps
            next_frame_t = start
            try:
                while seconds is None or (time.perf_counter() - start) < seconds:
                    now = time.perf_counter()
                    for zone, region in self.regions.items():
                        lum = region_luminance(sct, region)
                        self.record_sample(zone, now, lum)

                    ok, frame = cap.read()
                    if not ok:
                        if is_file:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the source file
                            ok, frame = cap.read()
                        if not ok:
                            break

                    out_frame = self.process_frame(frame, now)
                    out_rgb = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
                    vcam.send(out_rgb)

                    next_frame_t += frame_period
                    sleep_s = next_frame_t - time.perf_counter()
                    if sleep_s > 0:
                        vcam.sleep_until_next_frame()
            finally:
                cap.release()


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Milestone 12 adaptive injector")
    parser.add_argument("--source", required=True,
                         help="path to a video file (see scripts/record_source.py) or an integer camera index")
    parser.add_argument("--mode", choices=["global", "per-zone"], default="global")
    parser.add_argument("--lag-ms", type=float, default=50.0)
    parser.add_argument("--depth", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    with open(repo_root / "config.yaml") as f:
        raw = yaml.safe_load(f)

    econfig = EmitterConfig.from_dict(raw["emitter"])
    icfg_dict = dict(raw.get("attacks", {}).get("adaptive_injector", {}))
    icfg_dict["mode"] = args.mode
    icfg_dict["lag_ms"] = args.lag_ms
    icfg_dict["depth"] = args.depth
    iconfig = InjectorConfig.from_dict(icfg_dict)

    source = args.source
    if source.isdigit():
        source = int(source)

    injector = AdaptiveInjector(source, iconfig, econfig)
    injector.run(seconds=args.seconds)
