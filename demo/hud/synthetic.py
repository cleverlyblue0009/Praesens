"""Synthetic sensor world for the HUD -- UI development and dry runs ONLY.

`python -m demo.hud --synthetic` replaces the webcam, microphone and keyboard
with simulated ones so the dashboard can be built, styled and rehearsed on a
machine with no camera/mic (or no MediaPipe model). Every screen in this mode
carries a permanent "SYNTHETIC SOURCE" banner and every log it writes is
tagged meta.synthetic=true: these numbers are NOT measurements and must never
be shown to anyone as results.

The simulation is physically shaped, not hard-coded scores: the "face"
reflects whatever the browser's emitter border ACTUALLY displayed (the same
log the optical lane scores against), the skin carries a pulse, and each
simulated key strike puts an impact click into the simulated mic -- so the
real lane code runs end to end. The sim attack toggles break exactly one of
those couplings each.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np


MESH = json.loads((Path(__file__).resolve().parent / "static" / "facemesh.json").read_text())
_V = np.array(MESH["vertices"], dtype=np.float64)          # canonical face, cm, y up
_V -= _V.mean(axis=0)
_IRIS_SRC = {468: (33, 133, 159, 145), 473: (362, 263, 386, 374)}
PBV = np.array([0.33, 0.77, 0.53])


def _rot(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


class SyntheticWorld:
    W, H = 640, 480

    def __init__(self, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.t0 = time.perf_counter()
        self.attack_video = False     # "injected video": face no longer reflects the screen, no pulse
        self.attack_keys = False      # "injected keystrokes": key events without physical impacts
        self.face_present = True
        self.emitter_fn = lambda: None   # -> the BrowserEmitter currently on screen
        self.display_latency_s = 0.045
        self._drift = 0.0
        self._phase = 0.0
        self._last_t = None

    def landmarks(self, t: float) -> np.ndarray:
        """(478, 3) MediaPipe-style normalised landmarks (x right, y down, z toward camera negative)."""
        s = t - self.t0
        R = _rot(np.radians(16 * np.sin(0.23 * s)), np.radians(7 * np.sin(0.31 * s + 1)),
                 np.radians(3 * np.sin(0.17 * s)))
        v = _V @ R.T
        scale = 1 / 38.0
        x = 0.5 + v[:, 0] * scale * (self.H / self.W) + 0.01 * np.sin(0.13 * s)
        y = 0.47 - v[:, 1] * scale + 0.008 * np.sin(0.19 * s)
        z = -v[:, 2] * scale
        pts = np.stack([x, y, z], axis=1)
        iris = []
        for base, src in _IRIS_SRC.items():
            c = pts[list(src)].mean(axis=0)
            iris.append(c)
            for k in range(4):
                a = k * np.pi / 2
                iris.append(c + np.array([0.006 * np.cos(a), 0.006 * np.sin(a), 0]))
        return np.vstack([pts, np.array(iris)])

    def observe(self, t: float):
        """One camera frame's worth: (frame_bgr, lum, rgb, landmarks|None, face_ok)."""
        if not self.face_present:
            frame = np.full((self.H, self.W, 3), 12, np.uint8)
            return frame, float("nan"), (float("nan"),) * 3, None, False
        lm = self.landmarks(t)
        self._drift += self.rng.normal(0, 0.03)
        # what the screen is physically showing right now (the browser paints the challenge
        # schedule), seen by the "skin" one display latency later
        em = self.emitter_fn()
        chip = float(em.challenge.value_at(t - self.display_latency_s - em.start_perf)) if em is not None else 0.0
        if self.attack_video:
            chip = 0.0  # injected video: whatever the screen does, the "face" never responds
        s = t - self.t0
        bpm = 74 + 3 * np.sin(0.05 * s)
        if self._last_t is not None:
            self._phase += 2 * np.pi * (bpm / 60.0) * (t - self._last_t)
        self._last_t = t
        phase = self._phase
        ppg = 0.0 if self.attack_video else 0.004 * (np.sin(phase) + 0.3 * np.sin(2 * phase + 0.5))
        lum = 118.0 + self._drift + 4.5 * chip + self.rng.normal(0, 0.9)
        base = np.array([172.0, 126.0, 106.0]) * (lum / 118.0)
        rgb = tuple(base * (1 + ppg * PBV) + self.rng.normal(0, 0.25, 3))

        frame = np.zeros((self.H, self.W, 3), np.uint8)
        frame[:] = (18, 14, 10)
        hull = cv2.convexHull((lm[:468, :2] * [self.W, self.H]).astype(np.int32))
        g = int(np.clip(lum * 0.55, 0, 255))
        cv2.fillConvexPoly(frame, hull, (int(g * 0.62), int(g * 0.72), g))
        for p in lm[::3]:
            cv2.circle(frame, (int(p[0] * self.W), int(p[1] * self.H)), 1, (230, 220, 120), -1)
        cv2.putText(frame, "SYNTHETIC SOURCE", (12, self.H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (60, 60, 255), 2, cv2.LINE_AA)
        return frame, float(lum), rgb, lm, True


class SyntheticMic:
    """Same read API as praesens.pressure.MicRecorder; audio is synthesised on
    demand: room noise + an impact transient at every registered strike."""

    def __init__(self, sample_rate_hz: int = 44100, seed: int = 3):
        self.fs = sample_rate_hz
        self.rng = np.random.default_rng(seed)
        self._clicks: list = []  # (perf_t, amplitude)
        self._lock = threading.Lock()
        self.t_start = time.perf_counter()

    def start(self):
        pass

    def stop(self):
        pass

    def strike(self, t_perf: float, amp: float):
        with self._lock:
            self._clicks.append((t_perf, amp))
            cutoff = time.perf_counter() - 180
            self._clicks = [c for c in self._clicks if c[0] > cutoff]

    def audio_between(self, t0: float, t1: float):
        t1 = min(t1, time.perf_counter())
        n = max(0, int((t1 - t0) * self.fs))
        with self._lock:
            x = self.rng.normal(0, 0.002, n)
            clicks = list(self._clicks)
            noise = self.rng.normal(0, 1, (len(clicks), int(0.012 * self.fs)))
        L = int(0.012 * self.fs)
        decay = np.exp(-np.arange(L) / (0.002 * self.fs))
        for (tc, amp), nz in zip(clicks, noise):
            i = int((tc - t0) * self.fs)
            if -L < i < n:
                burst = amp * nz * decay
                lo, hi = max(0, i), min(n, i + L)
                x[lo:hi] += burst[lo - i:hi - i]
        return x.astype(np.float32), t0

    def get_audio(self):
        now = time.perf_counter()
        return self.audio_between(max(self.t_start, now - 60), now)


class SyntheticTyper:
    """Same read API as praesens.typing.KeystrokeCapture (character-free
    events). In ACTIVE mode it 'types' the phrase with human-like rhythm;
    in PASSIVE mode, occasional bursts. Each key strike also hits the mic,
    unless the world's attack_keys toggle says the keys are injected."""

    def __init__(self, world: SyntheticWorld, mic: SyntheticMic, expected_phrase: str | None = None,
                 audio_lag_s: float = 0.12, seed: int = 5):
        self.world, self.mic = world, mic
        self.expected_phrase = expected_phrase
        self.audio_lag_s = audio_lag_s
        self.rng = np.random.default_rng(seed)
        self.events: list = []
        self._cursor = 0
        self._stop = threading.Event()
        self._thread = None

    @property
    def chars_matched(self) -> int:
        return self._cursor

    @property
    def phrase_complete(self) -> bool:
        return self.expected_phrase is not None and self._cursor >= len(self.expected_phrase)

    def _press(self, matched):
        t = time.perf_counter()
        ev = {"t_down": t, "t_up": None, "matched_expected": matched}
        self.events.append(ev)
        if not self.world.attack_keys:
            self.mic.strike(t + self.audio_lag_s + self.rng.normal(0, 0.004), float(self.rng.uniform(0.05, 0.2)))
        time.sleep(float(self.rng.uniform(0.06, 0.11)))
        ev["t_up"] = time.perf_counter()

    def _run(self):
        if self.expected_phrase is not None:
            if self._stop.wait(1.6):  # reading the phrase first
                return
            while not self._stop.is_set() and not self.phrase_complete:
                self._press(True)
                self._cursor += 1
                if self._stop.wait(float(self.rng.lognormal(np.log(0.12), 0.4))):
                    return
            return
        while not self._stop.is_set():
            if self._stop.wait(float(self.rng.uniform(2.0, 5.0))):
                return
            for _ in range(int(self.rng.integers(6, 16))):
                if self._stop.is_set():
                    return
                self._press(None)
                self._stop.wait(float(self.rng.lognormal(np.log(0.13), 0.4)))

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_events(self):
        return [dict(e) for e in self.events]
