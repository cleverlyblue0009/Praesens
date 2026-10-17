"""HUD engine: sensors -> live lane telemetry -> scored SCAN sessions.

The browser page IS the emitter. Its four screen-edge strips show the
challenge luminance, and every animation frame it reports (time, chip value)
back over the websocket, converted onto THIS process's time.perf_counter()
clock with an NTP-style min-RTT offset estimate (localhost RTT is ~1 ms, so
the conversion error is well under the camera's 60 ms frame period).
BrowserEmitter turns those reports into exactly the redraw log that
praesens.emit.Emitter writes for the OpenCV window -- so the optical lane is
scored by the SAME OpticalFrameProcessor.finalize() code the paper numbers
came from, against what the screen actually showed, not against a schedule.
The fixed browser->panel display latency lands in the lag the optical lane
already searches for (0-300 ms).

Modes:
  IDLE   -- looping live challenge; rolling optical score, live pulse,
            live keystroke-impact matching on whatever the operator types.
  SCAN   -- a fresh, non-looping challenge + challenge phrase; every lane in
            hud.lanes runs exactly as praesens.session runs it.
  RESULT -- verdict from praesens.fusion.adjudicate_two_lane, JSON log
            written to logs/ (same schema as praesens.session, add-only
            "source": "hud"), then back to IDLE.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from praesens.challenge import Challenge, generate_challenge_phrase, pick_auto_chip_rate
from praesens.emit import EmitterConfig
from praesens.fusion import AdjudicatorConfig, adjudicate_two_lane, acoustic_lane_stub, lane_passes, \
    lane_contributes, LANE_LAG_BOUNDS_MS
from praesens.optical import (
    OpticalConfig, OpticalFrameProcessor, moving_average_detrend, cross_correlate_lag_search,
    estimate_snr_db, resample_emitted, lock_camera,
)
from praesens.pressure import PressureConfig, analyse_keystroke_impacts, detect_impacts, pressure_record, \
    _probe_notches, PressureResult
from praesens.pulse import PulseConfig, estimate_pulse, skin_rgb, pulse_record
from praesens.session import (
    REPO_ROOT, generate_session_id, optical_lane_result, typing_lane_result, _typing_record_dict,
    acoustic_lane_result, pressure_lane_result, pulse_lane_result,
)
from praesens.typing import TypingConfig

N_LANDMARKS = 478


# ---------------------------------------------------------------------------
# Browser-driven emitter
# ---------------------------------------------------------------------------

class BrowserEmitter:
    """Drop-in for praesens.emit.Emitter as far as OpticalFrameProcessor is
    concerned (set_preview / set_modulation_depth / adaptive_boost_applied /
    config / get_log), but the redraw log is filled from the browser's own
    per-frame reports instead of an OpenCV window."""

    _next_id = 1

    def __init__(self, challenge: Challenge, config: EmitterConfig, start_perf: float, on_change=None,
                 max_log_s: float | None = None):
        self.challenge = challenge
        self.config = config
        self.start_perf = start_perf
        self.id = BrowserEmitter._next_id
        BrowserEmitter._next_id += 1
        self._depth = float(config.modulation_depth)
        self._boosted = False
        self._log: list = []
        self._lock = threading.Lock()
        self._on_change = on_change
        self.max_log_s = max_log_s

    # -- Emitter-compatible surface ------------------------------------
    def set_preview(self, frame) -> None:
        pass

    def set_enabled(self, enabled: bool) -> None:
        pass

    def is_enabled(self) -> bool:
        return True

    def set_modulation_depth(self, depth: float) -> None:
        depth = float(np.clip(depth, 0.0, self.config.max_modulation_depth))
        with self._lock:
            self._boosted = self._boosted or depth > self.config.modulation_depth
            self._depth = depth
        if self._on_change:
            self._on_change(self)

    def get_modulation_depth(self) -> float:
        return self._depth

    def adaptive_boost_applied(self) -> bool:
        return self._boosted

    def get_log(self) -> list:
        with self._lock:
            return list(self._log)

    # -- browser side ----------------------------------------------------
    def describe(self) -> dict:
        return {"type": "challenge", "id": self.id, "chips": self.challenge.chips.astype(int).tolist(),
                "chip_rate_hz": self.challenge.chip_rate_hz, "loop": bool(self.challenge.loop),
                "start_perf": self.start_perf, "duration_s": self.challenge.duration_s,
                "base": self.config.base_luminance, "depth": self._depth}

    def ingest(self, entries: list) -> int:
        """entries: [[t_perf, chip_value, luminance], ...] in display order."""
        rows = []
        for e in entries:
            try:
                t, chip, lum = float(e[0]), int(e[1]), float(e[2])
            except (TypeError, ValueError, IndexError):
                continue
            if not (math.isfinite(t) and math.isfinite(lum)) or chip not in (-1, 0, 1):
                continue
            rows.append({"t": t, "elapsed_s": t - self.start_perf, "chip_value": chip,
                         "luminance": lum, "enabled": True})
        with self._lock:
            if rows and self._log and rows[0]["t"] < self._log[-1]["t"]:
                rows = [r for r in rows if r["t"] >= self._log[-1]["t"]]
            self._log.extend(rows)
            if self.max_log_s and self._log:
                cutoff = self._log[-1]["t"] - self.max_log_s
                if self._log[0]["t"] < cutoff:
                    i = int(np.searchsorted([r["t"] for r in self._log], cutoff))
                    self._log = self._log[i:]
        return len(rows)


class SyntheticOpticalProcessor(OpticalFrameProcessor):
    """OpticalFrameProcessor with the landmarker replaced by the synthetic
    world's own observations (synthetic mode only). finalize() -- the actual
    scoring -- is the inherited, unmodified method."""

    def __init__(self, config: OpticalConfig):
        self.config = config
        self.own_landmarker = False
        self.landmarker = None
        self.detector = SimpleNamespace(last_landmarks=None, last_boxes=[])
        self.timestamps, self.luminances, self.face_flags = [], [], []
        self._adaptive_triggered = True
        self._last_ts_ms = -1

    def ingest(self, t: float, lum: float, face_ok: bool, landmarks) -> None:
        self.timestamps.append(t)
        self.luminances.append(lum)
        self.face_flags.append(face_ok)
        self.detector.last_landmarks = landmarks

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _downsample(x: np.ndarray, n: int, mode: str = "mean") -> list:
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return []
    if len(x) <= n:
        return [round(float(v), 4) if math.isfinite(v) else None for v in x]
    edges = np.linspace(0, len(x), n + 1).astype(int)
    f = np.nanmax if mode == "max" else np.nanmean
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = x[a:max(b, a + 1)]
        v = f(seg) if np.any(np.isfinite(seg)) else float("nan")
        out.append(round(float(v), 4) if math.isfinite(v) else None)
    return out


def _f(v, nd=3):
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def _landmarks_array(lm) -> np.ndarray | None:
    if lm is None:
        return None
    if isinstance(lm, np.ndarray):
        return lm
    return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)


@dataclass
class LiveOptical:
    score: float | None = None
    lag_ms: float | None = None
    snr_db: float | None = None
    measured: list = field(default_factory=list)
    emitted: list = field(default_factory=list)
    face_frac: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HudEngine:
    def __init__(self, raw_config: dict, synthetic: bool = False, camera_index: int | None = None,
                 lanes: str | None = None):
        self.raw = raw_config
        self.synthetic = synthetic
        hud = raw_config.get("hud", {})
        self.hud = hud
        self.lanes = (lanes or hud.get("lanes", "optical+typing+pressure")).split("+")
        self.stream_fps = float(hud.get("stream_fps", 15))
        self.jpeg_quality = int(hud.get("jpeg_quality", 70))
        self.live_window_s = float(hud.get("live_window_s", 8.0))
        self.verdict_hold_s = float(hud.get("verdict_hold_s", 12.0))

        self.oconfig = OpticalConfig.from_dict(raw_config.get("optical", {}))
        mp = Path(self.oconfig.model_path)
        self.oconfig.model_path = str(mp if mp.is_absolute() else REPO_ROOT / mp)
        if camera_index is not None:
            self.oconfig.camera_index = camera_index
        self.econfig = EmitterConfig.from_dict(raw_config.get("emitter", {}))
        self.tconfig = TypingConfig.from_dict(raw_config.get("typing", {}))
        self.pconfig = PressureConfig.from_dict(raw_config.get("pressure", {}))
        self.puconfig = PulseConfig.from_dict(raw_config.get("pulse", {}))
        self.fusion_config = AdjudicatorConfig.from_dict(raw_config.get("fusion", {}))
        # the live (between-scans) impact view runs every second on the same GIL as the camera
        # loop: a coarser lag grid and fewer decoys keep it cheap. Scans use the full pconfig.
        self.pconfig_live = PressureConfig(**{**self.pconfig.__dict__, "n_decoys": 6, "lag_step_ms": 4.0})

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.listeners = []          # callables(msg_dict) -- the web layer registers one
        self.mode = "idle"
        self.status_line = "initialising sensors"
        self.warnings: list = []
        self.frame_jpeg: bytes | None = None
        self._last_jpeg_t = 0.0
        self.frame_times: list = []
        self.landmarks: np.ndarray | None = None
        self.face_ok = False
        self.skin: list = []          # (t, rgb) -- current processor's trace
        self.live_optical = LiveOptical()
        self.live_pulse = None
        self.live_pressure: dict = {}
        self.scan: dict | None = None
        self.result: dict | None = None
        self.exposure_locked = False
        self.cap = None
        self.world = None
        self.mic = None
        self.mic_error = None
        self.keys = None
        self.keys_error = None
        self.emitter: BrowserEmitter | None = None
        self.processor = None
        self.chip_rate_hz = float(raw_config.get("challenge", {}).get("chip_rate_hz", 2.0))
        self.out_dir = REPO_ROOT / "logs"

    # -- pub/sub ------------------------------------------------------------
    def publish(self, msg: dict) -> None:
        for fn in list(self.listeners):
            try:
                fn(msg)
            except Exception:
                pass

    def _emitter_changed(self, em: BrowserEmitter) -> None:
        if em is self.emitter or (self.scan and self.scan.get("emitter") is em):
            self.publish(em.describe())

    # -- sensors ------------------------------------------------------------
    def _new_processor(self):
        if self.synthetic:
            return SyntheticOpticalProcessor(self.oconfig)
        return OpticalFrameProcessor(self.oconfig)

    def _new_keys(self, phrase: str | None):
        if self.synthetic:
            from demo.hud.synthetic import SyntheticTyper
            return SyntheticTyper(self.world, self.mic, expected_phrase=phrase)
        from praesens.typing import KeystrokeCapture
        return KeystrokeCapture(expected_phrase=phrase)

    def _start_keys(self, phrase: str | None):
        try:
            k = self._new_keys(phrase)
            k.start()
            self.keys, self.keys_error = k, None
        except Exception as e:  # no display / permission: typing + pressure degrade to no_evidence
            self.keys, self.keys_error = None, str(e)
            self.warnings.append(f"keystroke listener unavailable: {e}")

    def _stop_keys(self):
        if self.keys is not None:
            try:
                self.keys.stop()
            except Exception:
                pass

    def _start_mic(self):
        if self.synthetic:
            return
        try:
            from praesens.pressure import MicRecorder
            m = MicRecorder(self.pconfig, max_keep_s=120.0)
            m.start()
            self.mic, self.mic_error = m, None
        except Exception as e:
            self.mic, self.mic_error = None, str(e)
            self.warnings.append(f"microphone unavailable: {e}")

    def _stop_mic(self):
        if self.mic is not None and not self.synthetic:
            try:
                self.mic.stop()
            except Exception:
                pass
            self.mic = None

    def open_sensors(self) -> None:
        if self.synthetic:
            from demo.hud.synthetic import SyntheticWorld, SyntheticMic
            self.world = SyntheticWorld()
            self.world.emitter_fn = self._current_emitter
            self.mic = SyntheticMic(self.pconfig.sample_rate_hz)
            self.exposure_locked = True
        else:
            from praesens.camera import open_camera
            from praesens.capture import CaptureConfig, configure_capture_format
            self.cap = open_camera(self.oconfig.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError(f"could not open camera index {self.oconfig.camera_index}")
            configure_capture_format(self.cap, CaptureConfig.from_dict(self.raw.get("capture", {})), self.warnings)
            self.exposure_locked = lock_camera(self.cap, self.oconfig, self.warnings)
            for _ in range(int(self.raw.get("optical", {}).get("camera_warmup_frames", 30))):
                self.cap.grab()
                self.cap.retrieve()
            self._start_mic()
        self._start_keys(None)
        self._enter_idle()

    def _prime_camera(self, cap, tries: int = 25) -> bool:
        """isOpened() is not proof of life: on Windows a second handle to a
        camera that is already open reports True and then never delivers a
        frame. Only an actual retrieved frame counts."""
        for _ in range(tries):
            if cap.grab():
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    return True
            time.sleep(0.04)
        return False

    def _configure_camera(self, cap) -> None:
        from praesens.capture import CaptureConfig, configure_capture_format
        configure_capture_format(cap, CaptureConfig.from_dict(self.raw.get("capture", {})), self.warnings)
        self.exposure_locked = lock_camera(cap, self.oconfig, self.warnings)
        # DSHOW needs several frames after an exposure change before fps and brightness settle;
        # open_sensors() discards the same number, and lock verification is meaningless without them
        for _ in range(int(self.raw.get("optical", {}).get("camera_warmup_frames", 30))):
            cap.grab()
            cap.retrieve()

    def switch_camera(self, index: int) -> str:
        """Release the current camera BEFORE opening the next one. DirectShow
        hands out exclusive access, so opening while the old handle is still
        alive -- switching to the camera already in use, in particular --
        returns a capture that reports isOpened() but never grabs a frame, and
        the preview silently freezes on its last image (seen on the demo
        laptop, 2026-10-17). The previous camera is restored if the new one
        turns out to be dead."""
        if self.synthetic:
            return "synthetic mode has no cameras"
        from praesens.camera import open_camera
        index = int(index)
        previous = self.oconfig.camera_index

        with self.lock:
            old, self.cap = self.cap, None      # capture_loop idles while cap is None
        if old is not None:
            old.release()

        new = open_camera(index)
        if new.isOpened() and self._prime_camera(new):
            self._configure_camera(new)
            with self.lock:
                self.cap = new
                self.oconfig.camera_index = index
            if self.mode == "idle":
                self._enter_idle()              # fresh processor: no history from the old camera
            return f"switched to camera {index}"

        new.release()
        back = open_camera(previous)
        if back.isOpened() and self._prime_camera(back):
            self._configure_camera(back)
            with self.lock:
                self.cap = back
                self.oconfig.camera_index = previous
            return f"camera {index} delivered no frames -- still on camera {previous}"
        back.release()
        self.warnings.append(f"camera {index} failed and camera {previous} could not be reopened")
        return f"camera {index} did not open, and camera {previous} could not be reopened"

    def set_sim(self, key: str, value: bool) -> None:
        if self.world is not None and key in ("attack_video", "attack_keys", "face_present"):
            setattr(self.world, key, bool(value))

    # -- modes --------------------------------------------------------------
    def measured_fps(self) -> float:
        ts = self.frame_times[-40:]
        return float((len(ts) - 1) / (ts[-1] - ts[0])) if len(ts) >= 5 and ts[-1] > ts[0] else float("nan")

    def _pick_chip_rate(self, base_duration_s: float) -> tuple[float, float]:
        o = self.raw.get("optical", {})
        if not o.get("auto_chip_rate", False):
            return self.chip_rate_hz, float(self.raw.get("challenge", {}).get("duration_s", base_duration_s))
        fps = self.measured_fps()
        return pick_auto_chip_rate(fps, divisor=o.get("auto_chip_rate_divisor", 6.0), min_hz=0.5, max_hz=5.0,
                                   min_chips=o.get("auto_chip_rate_min_chips", 60),
                                   base_duration_s=base_duration_s)

    def _current_emitter(self):
        return self.scan["emitter"] if self.scan and self.scan.get("emitter") and \
            time.perf_counter() >= self.scan["start_perf"] else self.emitter

    def _enter_idle(self) -> None:
        chip_rate, _ = self._pick_chip_rate(20.0)
        live = Challenge(chip_rate_hz=chip_rate, duration_s=127 / chip_rate, loop=True)
        em = BrowserEmitter(live, self.econfig, start_perf=time.perf_counter() + 0.3,
                            on_change=self._emitter_changed, max_log_s=90.0)
        proc = self._new_processor()
        with self.lock:
            old = self.processor
            self.emitter, self.processor = em, proc
            self.mode = "idle"
            self.scan = None
            self.status_line = "live -- press SCAN to verify"
        if old is not None:
            old.close()
        self.publish(em.describe())

    def start_scan(self) -> str:
        with self.lock:
            if self.mode != "idle":
                return f"busy ({self.mode})"
            self.mode = "arming"
        threading.Thread(target=self._run_scan, daemon=True).start()
        return "scan armed"

    def _run_scan(self) -> None:
        try:
            self._scan_body()
        except Exception as e:  # never wedge the HUD on a scan failure
            import traceback
            traceback.print_exc()
            self.warnings.append(f"scan failed: {e}")
            self.result = {"verdict": "ERROR", "reason": str(e), "lanes": [], "t": time.perf_counter()}
            self.mode = "result"
            self.publish({"type": "result", "result": self.result})
            time.sleep(4.0)
            self._enter_idle()

    def _scan_body(self) -> None:
        chip_rate, duration = self._pick_chip_rate(float(self.raw.get("challenge", {}).get("duration_s", 20.0)))
        challenge = Challenge(chip_rate_hz=chip_rate, duration_s=duration)
        phrase = generate_challenge_phrase(challenge.seed, n_words=self.tconfig.n_words,
                                           generative=self.raw.get("typing", {}).get("generative", False))
        proc = self._new_processor()           # its own landmarker: VIDEO-mode timestamps restart at 0
        lead_s = 1.2
        start = time.perf_counter() + lead_s
        em = BrowserEmitter(challenge, self.econfig, start_perf=start, on_change=self._emitter_changed)
        use_acoustic = "acoustic" in self.lanes
        use_pressure = "pressure" in self.lanes
        acoustic_box: dict = {}
        scan = {"challenge": challenge, "phrase": phrase, "emitter": em, "processor": proc,
                "start_perf": start, "end_perf": start + duration, "skin": [], "acoustic": acoustic_box}
        with self.lock:
            self.scan = scan
            self.mode = "scan"
            self.result = None
            self.status_line = "verification scan running"
        self.publish(em.describe())
        self.publish({"type": "scan", "phrase": phrase, "start_perf": start, "duration_s": duration,
                      "chip_rate_hz": chip_rate, "lanes": self.lanes})

        # keyboard: passive listener out, the phrase-checking one in
        self._stop_keys()
        self._start_keys(phrase)
        if use_acoustic:
            self._stop_mic()   # one microphone stream: the acoustic lane's duplex one
            from praesens.acoustic import AcousticConfig, run_acoustic_session
            aconfig = AcousticConfig.from_dict(self.raw.get("acoustic", {}))
            scan["aconfig"] = aconfig

            def _acoustic():
                time.sleep(max(0.0, start - time.perf_counter()))
                try:
                    acoustic_box["result"] = run_acoustic_session(challenge, aconfig)
                except Exception as exc:
                    acoustic_box["error"] = exc
            at = threading.Thread(target=_acoustic, daemon=True)
            at.start()

        while time.perf_counter() < start + duration and not self.stop_event.is_set():
            time.sleep(0.05)
        end = time.perf_counter()
        with self.lock:
            self.mode = "scoring"
            self.status_line = "scoring lanes"
        keys = self.keys
        self._stop_keys()
        events = keys.get_events() if keys is not None else []
        events = [e for e in events if e["t_down"] >= start - 0.5]

        # -- optical: the unmodified finalize() against the browser's redraw log
        opt = proc.finalize(challenge, em, self.exposure_locked, list(self.warnings))
        optical_lane = optical_lane_result(opt)
        typing_lane = typing_lane_result(events, phrase, self.tconfig)

        acoustic_result = None
        if use_acoustic:
            at.join(timeout=10.0)
            acoustic_result = acoustic_box.get("result")
        acoustic_lane = acoustic_lane_result(acoustic_result) if acoustic_result is not None else None

        pressure_result = None
        if use_pressure:
            if acoustic_result is not None and acoustic_result.recording is not None:
                aconfig = scan["aconfig"]
                pressure_result = analyse_keystroke_impacts(
                    acoustic_result.recording, acoustic_result.sample0_perf, aconfig.sample_rate_hz, events,
                    self.pconfig, notch_freqs_hz=_probe_notches(aconfig.carrier_hz, aconfig.sample_rate_hz))
            elif self.mic is not None:
                audio, a0 = self.mic.audio_between(start - 0.5, end + 0.6)
                pressure_result = analyse_keystroke_impacts(audio, a0, self.pconfig.sample_rate_hz, events,
                                                            self.pconfig)
            else:
                pressure_result = PressureResult(status="no_evidence",
                                                 diagnostics=f"no microphone ({self.mic_error or 'unavailable'})")
        pressure_lane = pressure_lane_result(pressure_result) if pressure_result is not None else None

        skin = scan["skin"]
        pulse_est = estimate_pulse([t for t, _ in skin], [c for _, c in skin],
                                   PulseConfig(**{**self.puconfig.__dict__, "window_s": 0.0}),
                                   emitter_log=em.get_log()) if skin else None
        pulse_lane = pulse_lane_result(pulse_est) if pulse_est is not None else None

        typing_pass = self.tconfig.pass_match_fraction
        thr = self.fusion_config.reject_threshold
        joint = adjudicate_two_lane(optical_lane, typing_lane, optical_pass_threshold=thr,
                                    typing_pass_threshold=typing_pass, acoustic=acoustic_lane,
                                    acoustic_pass_threshold=thr, pressure=pressure_lane,
                                    pressure_pass_threshold=self.pconfig.pass_threshold,
                                    pulse=(pulse_lane if self.puconfig.gate else None),
                                    pulse_pass_threshold=self.puconfig.pass_threshold)
        lane_results = [optical_lane, typing_lane,
                        acoustic_lane if acoustic_lane is not None else acoustic_lane_stub()]
        lane_results += [l for l in (pressure_lane, pulse_lane) if l is not None]

        def _passes(l):
            if l.is_stub:
                return None
            if l.lane_name == "typing":
                return lane_passes(l, typing_pass)
            if l.lane_name == "pulse":
                return lane_passes(l, self.puconfig.pass_threshold)
            t = self.pconfig.pass_threshold if l.lane_name == "pressure" else thr
            return lane_passes(l, t) and lane_contributes(l, LANE_LAG_BOUNDS_MS)

        session_id = generate_session_id()
        record = {
            "session": session_id, "source": "hud", "condition": "bonafide",
            "score": opt.score, "lag_ms": opt.lag_ms, "seed": challenge.seed,
            "chip_rate_hz": challenge.chip_rate_hz, "duration_s": challenge.duration_s,
            "start_time_perf_counter": start, "auto_chip_rate_used": bool(self.raw.get("optical", {}).get("auto_chip_rate")),
            "emitter_enabled": True, "emitter": "browser", "snr_db": opt.snr_db,
            "insufficient_signal": opt.insufficient_signal, "adaptive_boost_applied": opt.adaptive_boost_applied,
            "exposure_locked": opt.exposure_locked, "n_frames": opt.n_frames, "n_face_detected": opt.n_face_detected,
            "measured_fps": opt.measured_fps, "warnings": opt.warnings,
            "meta": {"synthetic": self.synthetic, **({"sim_attack_video": self.world.attack_video,
                                                       "sim_attack_keys": self.world.attack_keys}
                                                      if self.world is not None else {})},
            "trace_emitted": opt.trace_emitted, "trace_measured": opt.trace_measured, "timestamps": opt.timestamps,
            "lanes": "+".join(self.lanes), "typing_phrase": phrase,
            "typing": _typing_record_dict(events, phrase, self.tconfig, typing_lane, typing_pass, start, duration),
            "acoustic": ({"status": acoustic_result.status, "score": acoustic_result.score,
                          "lag_ms": acoustic_result.lag_ms, "snr_db": acoustic_result.snr_db,
                          "diagnostics": acoustic_result.diagnostics} if acoustic_result is not None else None),
            "pressure": pressure_record(pressure_result),
            "pulse": pulse_record(pulse_est),
            "fusion": {
                "verdict": joint.verdict,
                "joint_score": None if math.isnan(joint.joint_score) else joint.joint_score,
                "reason_text": joint.reason_text, "per_lane_reasons": joint.per_lane_reasons,
                "accept_threshold": self.fusion_config.accept_threshold, "reject_threshold": thr,
                "lanes": [{"lane_name": l.lane_name, "subscore": l.subscore, "status": l.status,
                           "lag_ms": l.lag_ms, "confidence": l.confidence, "diagnostics": l.diagnostics,
                           "is_stub": l.is_stub} for l in lane_results],
                "policy": "adjudicate_two_lane", "optical_pass_threshold": thr,
                "typing_pass_threshold": typing_pass, "acoustic_pass_threshold": thr,
                "pressure_pass_threshold": self.pconfig.pass_threshold, "pulse_gated": self.puconfig.gate,
            },
        }
        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{session_id}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2, default=lambda o: None)

        self.result = {
            "t": time.perf_counter(), "session": session_id, "log": (str(out_path.relative_to(REPO_ROOT)) if out_path.is_relative_to(REPO_ROOT) else str(out_path)),
            "verdict": joint.verdict, "joint_score": _f(joint.joint_score), "reason": joint.reason_text,
            "synthetic": self.synthetic,
            "lanes": [{"name": l.lane_name, "status": l.status, "subscore": _f(l.subscore), "lag_ms": _f(l.lag_ms, 1),
                       "passes": _passes(l), "diagnostics": l.diagnostics,
                       "gated": not (l.lane_name == "pulse" and not self.puconfig.gate)}
                      for l in lane_results if not l.is_stub],
            "optical_trace": {"emitted": _downsample(opt.trace_emitted, 400),
                              "measured": _downsample(opt.trace_measured, 400)},
            "pulse_bpm": _f(pulse_est.bpm, 1) if pulse_est is not None else None,
            "pressure_keys": (pressure_result.per_key if pressure_result is not None else []),
        }
        with self.lock:
            self.mode = "result"
            self.status_line = f"{joint.verdict}: {joint.reason_text}"
        self.publish({"type": "result", "result": self.result})
        proc.close()

        if use_acoustic:
            self._start_mic()
        self._start_keys(None)
        t_end = time.perf_counter() + self.verdict_hold_s
        while time.perf_counter() < t_end and not self.stop_event.is_set():
            if self.mode != "result":  # dismissed early
                break
            time.sleep(0.1)
        self._enter_idle()

    def dismiss(self) -> None:
        if self.mode == "result":
            self.mode = "dismissing"

    # -- per-frame loop -------------------------------------------------------
    def capture_loop(self) -> None:
        while not self.stop_event.is_set():
            if self.synthetic:
                time.sleep(max(0.0, 1 / 16.0 - 0.002))
                t = time.perf_counter()
                frame, lum, rgb, lm, face_ok = self.world.observe(t)
            else:
                with self.lock:
                    cap = self.cap
                if cap is None or not cap.grab():
                    time.sleep(0.01)
                    continue
                t = time.perf_counter()
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue

            with self.lock:
                scan = self.scan
                in_scan = scan is not None and scan["start_perf"] <= t <= scan["end_perf"]
                proc = scan["processor"] if in_scan else self.processor
                em = scan["emitter"] if in_scan else self.emitter
                start_perf = scan["start_perf"] if in_scan else (em.start_perf if em else t)
                challenge = em.challenge if em else None
            if self.mode in ("scoring",) or proc is None:
                lm = None
            elif self.synthetic:
                proc.ingest(t, lum, face_ok, lm)
            else:
                try:
                    msg = proc.process_frame(frame, t, start_perf, challenge, em)
                except Exception as e:  # e.g. a processor closed mid-swap
                    self.warnings.append(f"frame processing: {e}")
                    continue
                if msg:
                    self.warnings.append(msg)
                lm = proc.detector.last_landmarks
                face_ok = proc.face_flags[-1] if proc.face_flags else False
                rgb = skin_rgb(frame, proc.detector.last_boxes)

            with self.lock:
                self.frame_times.append(t)
                if len(self.frame_times) > 120:
                    del self.frame_times[:-120]
                self.landmarks = _landmarks_array(lm)
                self.face_ok = bool(face_ok)
                if in_scan:
                    scan["skin"].append((t, rgb))       # scored trace: this scan only
                self.skin.append((t, rgb))              # live display trace: continuous
                if len(self.skin) > 2000:
                    del self.skin[:-1500]
            if t - self._last_jpeg_t >= 1.0 / self.stream_fps:
                self._last_jpeg_t = t
                small = frame if frame.shape[1] <= 640 else cv2.resize(
                    frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
                ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if ok:
                    self.frame_jpeg = buf.tobytes()

    # -- live analysis ---------------------------------------------------------
    def _retune_live_challenge(self) -> None:
        """The live challenge is created before the camera's real frame rate is
        known (auto_chip_rate falls back to its 0.5 Hz floor). Once the rate is
        measured, re-issue it at the rate a scan would use, so the live score
        means the same thing a scan score does."""
        if self.mode != "idle" or self.emitter is None or not math.isfinite(self.measured_fps()):
            return
        want, _ = self._pick_chip_rate(20.0)
        have = self.emitter.challenge.chip_rate_hz
        if abs(want - have) / have > 0.2:
            self._enter_idle()

    def analysis_loop(self) -> None:
        last_heavy = 0.0
        while not self.stop_event.is_set():
            time.sleep(0.33)
            try:
                if time.perf_counter() - last_heavy >= 1.0:
                    self._retune_live_challenge()
                self._live_optical()
                now = time.perf_counter()
                if now - last_heavy >= 1.0:
                    last_heavy = now
                    self._live_pulse()
                    self._live_pressure(heavy=True)
                else:
                    self._live_pressure(heavy=False)
            except Exception as e:
                self.warnings.append(f"live analysis: {e}")

    def _active(self):
        with self.lock:
            scan = self.scan
            now = time.perf_counter()
            if scan is not None and scan["start_perf"] <= now:
                return scan["processor"], scan["emitter"], self.skin
            return self.processor, self.emitter, self.skin

    def _live_optical(self) -> None:
        proc, em, _ = self._active()
        if proc is None or em is None:
            return
        n = len(proc.timestamps)
        ts = np.array(proc.timestamps[max(0, n - 400):n], dtype=np.float64)
        lum = np.array(proc.luminances[max(0, n - 400):n], dtype=np.float64)
        faces = np.array(proc.face_flags[max(0, n - 400):n], dtype=bool)
        if len(ts) < 10:
            self.live_optical = LiveOptical()
            return
        keep = ts >= ts[-1] - self.live_window_s
        ts, lum, faces = ts[keep], lum[keep], faces[keep]
        log = em.get_log()
        lo = LiveOptical(face_frac=float(faces.mean()) if len(faces) else 0.0)
        if np.isfinite(lum).sum() >= 10 and log:
            det = moving_average_detrend(ts, lum, self.oconfig.detrend_window_s)
            score, lag, emitted, vts, vm = cross_correlate_lag_search(
                det, ts, log, self.oconfig.lag_search_max_ms, self.oconfig.lag_step_ms)
            snr = estimate_snr_db(ts, det, em.challenge.chip_rate_hz, self.oconfig.snr_low_cut_hz,
                                  self.oconfig.snr_resample_rate_hz)
            z = lambda a: (a - np.mean(a)) / (np.std(a) or 1.0)  # noqa: E731
            lo.score, lo.lag_ms, lo.snr_db = float(score), float(lag), float(snr) if np.isfinite(snr) else None
            if len(vm) >= 5:
                lo.measured = _downsample(z(vm), 160)
                lo.emitted = _downsample(z(emitted), 160)
        self.live_optical = lo

    def _live_pulse(self) -> None:
        _, em, skin = self._active()
        with self.lock:
            trace = list(skin[-400:])
        if len(trace) < 10:
            self.live_pulse = None
            return
        est = estimate_pulse([t for t, _ in trace], [c for _, c in trace], self.puconfig,
                             emitter_log=em.get_log() if em else None, t_end=time.perf_counter())
        self.live_pulse = est

    def _live_pressure(self, heavy: bool) -> None:
        mic = self.mic
        now = time.perf_counter()
        out = dict(self.live_pressure)
        if mic is None:
            self.live_pressure = {"status": "no_evidence", "reason": self.mic_error or "microphone busy"}
            return
        audio, a0 = mic.audio_between(now - 3.0, now)
        if a0 is not None and len(audio) > 2000:
            on = detect_impacts(audio, self.pconfig.sample_rate_hz, self.pconfig)
            env = on.envelope_db
            out["env"] = _downsample(env - np.median(env), 200, mode="max")
            out["env_t0"] = a0
            out["onsets"] = [round(float(x), 3) for x in on.times_s]
            out["onsets_level"] = [round(float(x), 1) for x in on.level_db]
        keys = self.keys
        events = keys.get_events() if keys is not None else []
        recent = [e for e in events if e["t_down"] >= now - 8.5]
        out["recent_keys"] = [round(e["t_down"] - now, 3) for e in recent if e["t_down"] >= now - 6.0]
        if heavy:
            if len(recent) >= self.pconfig.min_keystrokes:
                audio8, b0 = mic.audio_between(now - 9.0, now)
                res = analyse_keystroke_impacts(audio8, b0, self.pconfig.sample_rate_hz, recent, self.pconfig_live,
                                                rng=np.random.default_rng(1))
                out.pop("reason", None)
                out.update({"status": res.status, "score": _f(res.score), "lag_ms": _f(res.lag_ms, 1),
                            "hit": _f(res.hit_fraction), "chance": _f(res.chance_fraction),
                            "jitter_ms": _f(res.lag_jitter_ms, 1),
                            "keys": [{"dt": round(k["t_down"] - now, 2), "matched": k["matched"],
                                      "force_db": _f(k["force_db"], 1), "dwell_ms": _f((k["dwell_s"] or 0) * 1000, 0)}
                                     for k in res.per_key[-24:]]})
            elif out.get("status") in ("ok", "stale"):
                out["status"] = "stale"   # keep the last burst's numbers on screen
            else:
                out.update({"status": "no_evidence", "score": None, "keys": [],
                            "reason": "type anything to see keystroke impacts"})
        self.live_pressure = out

    # -- telemetry -------------------------------------------------------------
    def telemetry(self) -> dict:
        now = time.perf_counter()
        with self.lock:
            lm = self.landmarks
            scan = self.scan
            mode = self.mode
        lo = self.live_optical
        pu = self.live_pulse
        keys = self.keys
        scan_info = None
        if scan is not None:
            el = now - scan["start_perf"]
            scan_info = {"phrase": scan["phrase"], "elapsed": round(el, 2),
                         "duration": scan["challenge"].duration_s,
                         "chip_rate_hz": scan["challenge"].chip_rate_hz,
                         "matched": getattr(keys, "chars_matched", 0) if keys is not None else 0,
                         "n_keys": len(keys.get_events()) if keys is not None else 0}
        msg = {
            "type": "telemetry", "now": now, "mode": mode, "synthetic": self.synthetic,
            "status": self.status_line, "fps": _f(self.measured_fps(), 1), "face": self.face_ok,
            "lanes": self.lanes, "camera": None if self.synthetic else self.oconfig.camera_index,
            "landmarks": (np.round(lm[:N_LANDMARKS, :3].astype(np.float64), 4).ravel().tolist()
                          if lm is not None else None),
            "optical": {"score": _f(lo.score), "lag_ms": _f(lo.lag_ms, 0), "snr_db": _f(lo.snr_db, 1),
                        "face_frac": _f(lo.face_frac, 2), "measured": lo.measured, "emitted": lo.emitted,
                        "pass": _f(self.fusion_config.reject_threshold)},
            "pulse": ({"status": pu.status, "bpm": _f(pu.bpm, 1), "snr_db": _f(pu.snr_db, 1),
                       "subscore": _f(pu.subscore), "reason": pu.diagnostics,
                       "wave": _downsample(pu.wave[-int(6 * self.puconfig.resample_hz):], 180) if pu.wave is not None else [],
                       "spectrum": _downsample(pu.spectrum, 70) if pu.spectrum is not None else []}
                      if pu is not None else {"status": "no_evidence", "reason": "acquiring skin signal"}),
            "pressure": {**self.live_pressure, "now": now},
            "scan": scan_info,
            "warnings": self.warnings[-4:],
            "sim": ({"attack_video": self.world.attack_video, "attack_keys": self.world.attack_keys,
                     "face_present": self.world.face_present} if self.world is not None else None),
        }
        return msg

    def shutdown(self) -> None:
        self.stop_event.set()
        self._stop_keys()
        self._stop_mic()
        if self.cap is not None:
            self.cap.release()
        if self.processor is not None:
            self.processor.close()
