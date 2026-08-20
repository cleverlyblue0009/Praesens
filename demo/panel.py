"""Milestone 8: panel mode.

A guided demo a sceptical panelist can drive themselves, entirely by
keypress: [1] live verification, [2] switch to an injected feed, [3]
toggle the light pattern off (the control, proving the score depends on
the pattern rather than the person), [4] a stored side-by-side genuine-vs-
injected comparison, [5] the typing lane (Milestone 14). Nothing is typed
and nothing can hang -- every transition is a state flip or an instant
capture-handle swap (Milestone 7), never a blocking call.

Panel mode uses a much shorter rolling score window than the general live
dashboard (Milestone 6's default 6s vs. this module's ~1.5s) so switching
to an injected feed visibly collapses the score within about a second
instead of taking up to the full window to fully evict the old genuine
samples. That's a real trade: a 1.5s window at a 5Hz chip rate is only
~7-8 chips, noisier than the 20s sessions Milestone 4/5 use for the
paper's numbers. It's the right trade here -- a demo needs the audience to
SEE the collapse happen, not wait it out.

Milestone 14 adds two things. First, the panel now drives Milestone 9's
SpatialEmitter instead of the single-zone Emitter, and continuously
computes BOTH the (old-style) global score and the spatial assignment
score from the same rolling capture -- so a global-brightness-only attack
(demo/attack.py's camera switch, or attacks/adaptive_injector.py --mode
global) can be shown raising the global score back up while the spatial
score stays dead, live, not just in tests/test_adaptive_injector.py. This
is additive to Milestone 9: praesens/spatial.py's own scoring functions
are reused unchanged, only WHICH emitter this demo drives is new (see
praesens.emit.Emitter.drive_and_log / praesens.spatial.SpatialEmitter.
drive_and_log for how demo/live.py's run loop stays emitter-type-agnostic).
Second, a `--rehearse` flag runs the whole panel sequence on a timer
instead of waiting for keypresses, so it's testable without a live panel
setup -- and per the hard rule that a demo must never crash on a missing
device, every scripted step (and every hardware-dependent lane: the
alternate camera, the typing lane's keyboard listener, its hand model)
degrades to a visible on-screen/console message instead of raising.
"""
from __future__ import annotations

import json
import time
import warnings
from collections import deque
from enum import Enum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml
from mediapipe.tasks.python import vision as mp_vision

from demo.attack import AttackDashboard
from demo.live import LiveDashboard, REPO_ROOT, GREEN, RED, GRAY, WHITE, LIGHT_GRAY, BLUE, ORANGE, _plot_series
from eval.analyse import load_sessions
from praesens.challenge import derive_zone_challenges
from praesens.emit import EmitterConfig
from praesens.optical import PER_ROI_NAMES, per_roi_luminance
from praesens.spatial import (
    SpatialEmitter, SpatialConfig, compute_correlation_matrix,
    spatial_assignment_score, global_score_from_logs,
)
from praesens.typing import (
    KeystrokeCapture, HandActivityTracker, TypingConfig, create_hand_landmarker,
    keystroke_hand_coherence, compute_typing_status, passive_confidence,
)


def collapse_latency_stats(latencies: list) -> dict:
    """Milestone 12: summary stats over however many source-switch ->
    first-REJECT deltas have been recorded. A single anecdote ("it took
    800ms once") isn't the claim; the distribution over >=20 real switches
    is -- this function is the reporting half of that, kept separate from
    collection so it's testable without needing 20 real switches to run it."""
    if not latencies:
        return {"n": 0}
    arr = np.array(latencies, dtype=np.float64)
    return {
        "n": int(len(arr)), "mean_s": float(arr.mean()), "median_s": float(np.median(arr)),
        "std_s": float(arr.std()), "min_s": float(arr.min()), "max_s": float(arr.max()),
        "p90_s": float(np.percentile(arr, 90)),
    }


class PanelState(str, Enum):
    LIVE = "live"
    ATTACK = "attack"
    REPLAY = "replay"
    TYPING = "typing"


def load_reference_traces(logs_dir: Path):
    """Best available stored genuine and injected sessions for the
    side-by-side comparison screen, or (None, None) if no data has been
    collected yet -- must not crash the demo before Milestone 4/5 data
    collection has happened."""
    records = load_sessions(logs_dir)
    bonafide = [r for r in records if r["condition"] == "bonafide" and r.get("timestamps")]
    attacks = [r for r in records if r["condition"] in ("replay", "swap") and r.get("timestamps")]
    genuine_ref = max(bonafide, key=lambda r: r["score"], default=None)
    attack_ref = min(attacks, key=lambda r: r["score"], default=None)
    return genuine_ref, attack_ref


class PanelDashboard(AttackDashboard):
    def __init__(self, raw_config: dict, max_camera_probe: int = 6, rehearse: bool = False):
        super().__init__(raw_config, max_camera_probe=max_camera_probe)

        # Trade statistical power for a fast, visible collapse -- see module docstring.
        self.dconfig.rolling_window_s = raw_config["demo"].get(
            "panel_rolling_window_s", self.dconfig.rolling_window_s
        )
        self.spatial_score_threshold = raw_config["demo"].get("spatial_score_threshold", 0.15)
        self.spatial_update_interval_s = raw_config["demo"].get("spatial_update_interval_s", 0.5)

        self.primary_index = self.oconfig.camera_index
        self.state = PanelState.LIVE
        self.status_message = ""

        # Milestone 12: source-switch -> first-REJECT collapse latency.
        # _pending_switch_time is armed by enter_attack() and disarmed by
        # _check_collapse_latency() the first time the verdict is observed
        # to cross to REJECT afterward.
        self.collapse_latencies: list = []
        self._pending_switch_time: float | None = None

        logs_dir = REPO_ROOT / raw_config["eval"]["logs_dir"]
        self.genuine_ref, self.attack_ref = load_reference_traces(logs_dir)
        if self.genuine_ref is None or self.attack_ref is None:
            print("NOTE: no stored bona fide/attack session logs found yet -- the [4] "
                  "side-by-side comparison screen will show a placeholder until "
                  "praesens/session.py has been run to collect real sessions.")

        # -- Milestone 14: spatial emitter/scoring -----------------------
        # Replaces self.emitter (built by LiveDashboard.__init__ above as a
        # single-zone Emitter) with Milestone 9's SpatialEmitter, so the
        # panel can show the global AND spatial scores side by side
        # continuously. self.challenge (the looping master sequence) and
        # self.start_time already exist from super().__init__() -- reused
        # as-is, not rebuilt, so the zone challenges stay phase-locked to
        # the same underlying m-sequence the rest of the demo assumes.
        self.sconfig = SpatialConfig.from_dict(raw_config["spatial"])
        zones = derive_zone_challenges(self.challenge, self.sconfig.zone_names, self.sconfig.min_shift_chips)
        spatial_econfig = EmitterConfig.from_dict(raw_config["emitter"])
        spatial_econfig.border_fraction = self.dconfig.border_fraction
        self.emitter = SpatialEmitter(zones, spatial_econfig)
        self.emitter.begin_manual_drive(self.start_time)

        self.per_roi_ts = {name: deque() for name in PER_ROI_NAMES}
        self.per_roi_lum = {name: deque() for name in PER_ROI_NAMES}
        self._recent_face_flags: deque = deque(maxlen=5)
        self._spatial_frame_i = 0
        self._spatial_last_ts_ms = -1
        self._spatial_last_result = None
        self._spatial_last_update_t = 0.0
        self.current_global_score = 0.0
        self.current_spatial_score = float("nan")
        # self.current_score/current_lag_ms/current_snr_db already exist
        # from LiveDashboard.__init__ -- current_score is kept as an ALIAS
        # for current_global_score below, so the existing banner/collapse-
        # latency logic (which reads current_score/dconfig.score_threshold)
        # needs no changes: the global score is still "the" verdict score,
        # spatial is the new secondary readout.

        # -- Milestone 14: typing lane (PASSIVE, shares self.cap) --------
        self.tconfig = TypingConfig.from_dict(raw_config.get("typing", {}))
        self._typing_hand_model_path = str(REPO_ROOT / self.tconfig.hand_model_path)
        self._typing_capture: KeystrokeCapture | None = None
        self._typing_hand_landmarker = None
        self._typing_hand_tracker = HandActivityTracker()
        self._typing_hand_ts: deque = deque()
        self._typing_hand_activity: deque = deque()
        self._typing_hand_ts_ms = -1
        self._typing_unavailable_reason: str | None = None
        self.current_typing_status = "no_evidence"
        self.current_typing_reason = "typing lane not started yet"
        self.current_typing_coherence = float("nan")
        self.current_typing_lag_ms = float("nan")
        self.current_typing_confidence = 0.0
        self.current_typing_n_keystrokes = 0

        # -- Milestone 14: --rehearse, a scripted timed sequence ---------
        self._rehearse = bool(rehearse)
        self._rehearse_script = list(raw_config["demo"].get("rehearse_script", [])) if self._rehearse else []
        self._rehearse_index = -1
        self._rehearse_next_switch_t: float | None = None
        self._rehearse_done = False
        if self._rehearse:
            if not self._rehearse_script:
                print("WARNING: --rehearse given but demo.rehearse_script is empty in config.yaml -- "
                      "nothing to run, exiting immediately.")
                self._rehearse_done = True
            else:
                self._advance_rehearse()

    # -- scenario transitions, each a single keypress, none of which block --

    def enter_live(self) -> None:
        self.switch_camera(self.primary_index)
        self.state = PanelState.LIVE
        self.status_message = ""
        self._pending_switch_time = None  # returning to LIVE cancels any pending collapse measurement

    def enter_attack(self) -> None:
        alternates = [idx for idx in self.camera_slots if idx != self.oconfig.camera_index]
        if not alternates:
            self.status_message = "no injected feed source found -- start OBS Virtual Camera and relaunch"
            print(f"WARNING: {self.status_message}")
            return
        self.switch_camera(alternates[0])
        self.state = PanelState.ATTACK
        self.status_message = ""
        self._pending_switch_time = time.perf_counter()  # arm collapse-latency measurement

    def toggle_pattern(self) -> None:
        self.emitter.set_enabled(not self.emitter.is_enabled())
        self.status_message = ""

    def enter_replay(self) -> None:
        self.state = PanelState.REPLAY
        self.status_message = ""

    def enter_typing(self) -> None:
        """PASSIVE mode only (no challenge phrase to display/verify in a
        panel context) -- see praesens/typing.py's module docstring for the
        privacy contract this lane operates under. Both the keystroke
        listener and the hand landmarker are started LAZILY and ONCE (kept
        running across repeated visits to this state rather than
        restarted), and each is individually wrapped so a missing/failed
        piece of hardware degrades this state to a visible no_evidence
        message instead of crashing the whole panel (hard rule: never
        crash on a missing device)."""
        self.state = PanelState.TYPING
        self.status_message = ""
        if self._typing_capture is None:
            try:
                self._typing_capture = KeystrokeCapture(expected_phrase=None)
                self._typing_capture.start()
            except Exception as e:
                print(f"WARNING: could not start keystroke capture -- typing lane will show "
                      f"no_evidence: {e}")
                self._typing_unavailable_reason = f"keystroke capture unavailable: {e}"
        if self._typing_hand_landmarker is None:
            try:
                self._typing_hand_landmarker = create_hand_landmarker(
                    self._typing_hand_model_path, mp_vision.RunningMode.VIDEO, self.tconfig.min_hand_confidence
                )
            except Exception as e:
                print(f"WARNING: could not load hand landmarker -- typing lane will show "
                      f"no_evidence: {e}")
                self._typing_unavailable_reason = f"hand landmarker unavailable: {e}"

    # -- Milestone 12: collapse-latency instrumentation ----------------------

    def _check_collapse_latency(self) -> None:
        """Called every frame after the score updates: if a switch is
        pending and the verdict has now genuinely crossed to REJECT (score
        below threshold, with a face actually visible so a "no face" gap
        can't masquerade as a fast collapse), records the delta and
        disarms. One measurement per switch -- see collapse_latency_stats
        for the distribution over however many have accumulated."""
        if self._pending_switch_time is None:
            return
        if not self.face_recently_detected:
            return
        if self.current_score < self.dconfig.score_threshold:
            delta = time.perf_counter() - self._pending_switch_time
            self.collapse_latencies.append(delta)
            print(f"collapse latency: {delta * 1000:.0f}ms (switch -> first REJECT), "
                  f"n={len(self.collapse_latencies)} recorded this session")
            self._pending_switch_time = None

    def _save_collapse_latencies(self) -> None:
        """Milestone 12: persist whatever collapse-latency deltas this
        session recorded, plus the summary distribution, so
        'report it as a distribution over >=20 switches, not one anecdote'
        is something that can actually be checked afterward instead of only
        being visible in the console scrollback while the demo was running.
        Writes even with 0 recordings (n=0) so a session that never
        triggered a genuine collapse is a visible, honest empty log, not a
        silently-missing file that looks the same as 'never ran'."""
        stats = collapse_latency_stats(self.collapse_latencies)
        record = {
            "session": time.strftime("%Y%m%dT%H%M%S"),
            "latencies_s": self.collapse_latencies,
            "stats": stats,
        }
        logs_dir = REPO_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        out_path = logs_dir / f"collapse_latency_{record['session']}.json"
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"collapse latency log saved to {out_path} (n={stats['n']})")

    # -- Milestone 14: spatial scoring (replaces the inherited single-zone
    # scoring in _process_frame -- see class docstring) ---------------------

    def _trim_spatial_buffers(self, now: float) -> None:
        window_start = now - self.dconfig.rolling_window_s
        for name in PER_ROI_NAMES:
            ts_dq, lum_dq = self.per_roi_ts[name], self.per_roi_lum[name]
            while ts_dq and ts_dq[0] < window_start:
                ts_dq.popleft()
                lum_dq.popleft()

    def _update_spatial_scores(self, now: float) -> None:
        """Throttled (spatial_update_interval_s, not every frame): a real
        cross-correlation lag search over 3 ROIs x however-many zones is
        not free, the same reasoning praesens/optical.py's
        detect_every_n_frames already applies to face detection."""
        per_roi_traces = {
            name: (list(self.per_roi_ts[name]), list(self.per_roi_lum[name]))
            for name in PER_ROI_NAMES if len(self.per_roi_ts[name]) > 0
        }
        if len(per_roi_traces) < len(PER_ROI_NAMES):
            return  # need all three ROIs represented at least once in the window

        n_valid = sum(int(np.sum(~np.isnan(np.array(lum)))) for _, lum in per_roi_traces.values())
        if n_valid < self.dconfig.min_valid_samples:
            self.current_global_score, self.current_spatial_score = 0.0, float("nan")
            self.current_score = self.current_global_score
            return

        window_start = now - self.dconfig.rolling_window_s
        zone_logs = {name: [e for e in self.emitter.get_zone_log(name) if e["t"] >= window_start]
                     for name in self.emitter.zones}
        global_log = [e for e in self.emitter.get_global_log() if e["t"] >= window_start]
        if not global_log or any(len(v) == 0 for v in zone_logs.values()):
            return

        try:
            # A live rolling window can legitimately contain an individual
            # timestamp where EVERY ROI is momentarily NaN (a single frame
            # with no face detected, sitting between two good ones) --
            # np.nanmean over that one all-NaN row is correct (its result
            # is NaN, handled fine downstream by detrend/cross-correlation)
            # but numpy warns about it every such frame; the fixed 20s
            # session fixtures in tests/test_spatial.py never hit this
            # because they're fully populated, so this is new exposure
            # from live use, not a change to spatial.py's own logic.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                global_score = global_score_from_logs(per_roi_traces, global_log, self.sconfig.detrend_window_s,
                                                        self.sconfig.lag_search_max_ms, self.sconfig.lag_step_ms)
                R, lag_matrix, roi_names, zone_names = compute_correlation_matrix(
                    per_roi_traces, zone_logs, self.sconfig.detrend_window_s,
                    self.sconfig.lag_search_max_ms, self.sconfig.lag_step_ms
                )
            spatial_score = spatial_assignment_score(R, roi_names, zone_names, self.sconfig.expected_assignment)
        except Exception as e:
            # Never let a scoring hiccup (e.g. a transient degenerate
            # window) crash the panel -- keep the previous readout and
            # try again next update.
            print(f"WARNING: spatial score computation error (continuing): {e}")
            return

        self.current_global_score = global_score
        self.current_spatial_score = spatial_score
        self.current_score = self.current_global_score
        self.current_lag_ms = float(np.mean(lag_matrix)) if lag_matrix.size else 0.0
        self.current_snr_db = float("nan")  # not computed on the spatial path -- honestly absent, not fabricated

        mismatch = (not np.isnan(spatial_score) and global_score >= self.dconfig.score_threshold
                    and spatial_score < self.spatial_score_threshold)
        if mismatch:
            self.status_message = ("GLOBAL PASSES BUT SPATIAL SCORE IS DEAD -- likely a "
                                    "global-brightness-only attack (Milestone 9's defence)")
        elif self.status_message.startswith("GLOBAL PASSES BUT SPATIAL"):
            self.status_message = ""

    # -- Milestone 14: typing lane processing --------------------------------

    def _process_typing_frame(self, frame, t: float) -> None:
        if self._typing_hand_landmarker is None:
            return
        ts_ms = max(self._typing_hand_ts_ms + 1, int((t - self.start_time) * 1000))
        self._typing_hand_ts_ms = ts_ms
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._typing_hand_landmarker.detect_for_video(mp_image, ts_ms)
            activity = self._typing_hand_tracker.process(result)
        except Exception as e:
            print(f"WARNING: hand-tracking error (continuing): {e}")
            activity = float("nan")
        self._typing_hand_ts.append(t)
        self._typing_hand_activity.append(activity)
        window_start = t - self.dconfig.rolling_window_s
        while self._typing_hand_ts and self._typing_hand_ts[0] < window_start:
            self._typing_hand_ts.popleft()
            self._typing_hand_activity.popleft()

    def _update_typing_status(self, now: float) -> None:
        if self._typing_capture is None or self._typing_hand_landmarker is None:
            self.current_typing_status = "no_evidence"
            self.current_typing_reason = self._typing_unavailable_reason or "typing lane unavailable"
            self.current_typing_coherence, self.current_typing_lag_ms = float("nan"), float("nan")
            self.current_typing_confidence = 0.0
            return

        events = self._typing_capture.get_events()
        last_kt = self._typing_capture.last_keystroke_time()
        silence_s = (now - last_kt) if last_kt is not None else float("inf")
        hand_ts_arr = np.array(self._typing_hand_ts)
        hand_activity_arr = np.array(self._typing_hand_activity)
        n_hand_valid = int(np.sum(~np.isnan(hand_activity_arr))) if len(hand_activity_arr) else 0

        status = compute_typing_status("passive", silence_s, n_hand_valid, self.tconfig.min_valid_hand_samples,
                                        self.tconfig.passive_no_evidence_after_s, len(events))
        self.current_typing_status = status
        self.current_typing_n_keystrokes = len(events)
        self.current_typing_confidence = passive_confidence(last_kt, now, self.tconfig.passive_decay_half_life_s)

        if status == "ok":
            score, lag = keystroke_hand_coherence(
                events, hand_ts_arr, hand_activity_arr, self.tconfig.coherence_sigma_s,
                self.tconfig.coherence_lag_max_ms, self.tconfig.coherence_lag_step_ms,
                self.tconfig.min_valid_hand_samples,
            )
            self.current_typing_coherence, self.current_typing_lag_ms = score, lag
            self.current_typing_reason = ""
        else:
            self.current_typing_coherence, self.current_typing_lag_ms = float("nan"), float("nan")
            self.current_typing_reason = ("no hand visible" if n_hand_valid < self.tconfig.min_valid_hand_samples
                                           else "no recent keystrokes")

    # -- Milestone 14: --rehearse driver --------------------------------------

    def _advance_rehearse(self) -> None:
        self._rehearse_index += 1
        if self._rehearse_index >= len(self._rehearse_script):
            print("REHEARSE: sequence complete.")
            self._rehearse_done = True
            return
        step = self._rehearse_script[self._rehearse_index]
        state_name, duration_s = step["state"], float(step["duration_s"])
        desc = step.get("description", "")
        print(f"REHEARSE [{self._rehearse_index + 1}/{len(self._rehearse_script)}] "
              f"-> {state_name} ({duration_s:.0f}s): {desc}")
        transitions = {"live": self.enter_live, "attack": self.enter_attack,
                        "replay": self.enter_replay, "typing": self.enter_typing}
        try:
            transitions[state_name]()
        except Exception as e:
            # Hard rule: a missing device must degrade the display, never
            # crash the rehearsal -- log it and move on to the next step.
            print(f"WARNING: rehearse step {state_name!r} failed (continuing): {e}")
            self.status_message = f"rehearse: {state_name} unavailable ({e})"
        self._rehearse_next_switch_t = time.perf_counter() + duration_s

    def _check_rehearse(self) -> None:
        if not self._rehearse or self._rehearse_done:
            return
        if self._rehearse_next_switch_t is not None and time.perf_counter() >= self._rehearse_next_switch_t:
            self._advance_rehearse()

    # -- main per-frame pipeline (replaces LiveDashboard's single-zone one) --

    def _process_frame(self) -> bool:
        grabbed = self.cap.grab()
        t = time.perf_counter()
        if not grabbed:
            return False
        ok, frame = self.cap.retrieve()
        if not ok or frame is None:
            return False

        elapsed = t - self.start_time
        ts_ms = max(self._spatial_last_ts_ms + 1, int(elapsed * 1000))
        self._spatial_last_ts_ms = ts_ms

        # Face (spatial) tracking is skipped while showing the typing
        # screen -- it isn't displayed there, and running both a face and
        # a hand landmarker every frame on this hardware's already-modest
        # FPS would cost real budget for no visible benefit.
        if self.state != PanelState.TYPING:
            do_detect = (self._spatial_frame_i % max(1, self.oconfig.detect_every_n_frames)) == 0
            self._spatial_frame_i += 1
            if do_detect:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    self._spatial_last_result = self.landmarker.detect_for_video(mp_image, ts_ms)
                except Exception as e:
                    print(f"WARNING: face detection error (continuing): {e}")
                    self._spatial_last_result = None

            if self._spatial_last_result is not None and self._spatial_last_result.face_landmarks:
                lum_dict = per_roi_luminance(frame, self._spatial_last_result, self.oconfig.roi_margin_frac)
                face_ok = True
            else:
                lum_dict = {name: float("nan") for name in PER_ROI_NAMES}
                face_ok = False

            for name in PER_ROI_NAMES:
                self.per_roi_ts[name].append(t)
                self.per_roi_lum[name].append(lum_dict[name])
            self._trim_spatial_buffers(t)

            self._recent_face_flags.append(face_ok)
            self.face_recently_detected = any(self._recent_face_flags)

            if t - self._spatial_last_update_t >= self.spatial_update_interval_s:
                self._spatial_last_update_t = t
                self._update_spatial_scores(t)
        else:
            self._process_typing_frame(frame, t)
            if t - self._spatial_last_update_t >= self.spatial_update_interval_s:
                self._spatial_last_update_t = t
                self._update_typing_status(t)

        self._check_collapse_latency()
        self._check_rehearse()
        return True

    def run(self, max_seconds: float | None = None) -> None:
        try:
            super().run(max_seconds=max_seconds)
        finally:
            self._save_collapse_latencies()
            if self._typing_capture is not None:
                self._typing_capture.stop()
            if self._typing_hand_landmarker is not None:
                self._typing_hand_landmarker.close()

    # -- key handling --------------------------------------------------------

    def _handle_key(self, key: int, frame) -> bool:
        if self._rehearse and self._rehearse_done:
            return False  # rehearsal sequence finished -- exit run() cleanly, same path as pressing Q
        if key == ord('1'):
            self.enter_live()
            return True
        elif key == ord('2'):
            self.enter_attack()
            return True
        elif key == ord('3'):
            self.toggle_pattern()
            return True
        elif key == ord('4'):
            self.enter_replay()
            return True
        elif key == ord('5'):
            self.enter_typing()
            return True
        # Deliberately bypass AttackDashboard's raw digit->camera-slot
        # mapping (it would collide with 1-5 above); keep only q/e/r/s.
        return LiveDashboard._handle_key(self, key, frame)

    # -- rendering ------------------------------------------------------------

    def _render_dashboard(self):
        if self.state == PanelState.REPLAY:
            return self._render_replay_screen()
        if self.state == PanelState.TYPING:
            return self._render_typing_screen()
        return super()._render_dashboard()

    def _draw_banner(self, canvas, x, y, w, h):
        """Same physics, panel-friendly wording: ACCEPT/REJECT instead of
        Milestone 6's LIVE HUMAN PRESENT/INJECTED FEED DETECTED. Milestone
        14: the reason line now also carries the spatial assignment score
        alongside the global one -- the verdict itself is still driven by
        current_score (== current_global_score), unchanged from Milestone
        8, so a global-multiply attack that fools the banner still fools
        it exactly as before; the spatial number is a visible SECOND
        opinion, not a second gate."""
        spatial_str = "n/a" if np.isnan(self.current_spatial_score) else f"{self.current_spatial_score:.2f}"
        if not self.face_recently_detected:
            color, title, reason = GRAY, "NO FACE DETECTED", "waiting for a face in frame"
        elif self.current_score >= self.dconfig.score_threshold:
            color, title = GREEN, "ACCEPT"
            reason = f"screen pattern reflected on face (global r={self.current_score:.2f}, spatial={spatial_str})"
        else:
            color, title = RED, "REJECT"
            if not np.isnan(self.current_snr_db) and self.current_snr_db < self.oconfig.snr_floor_db:
                reason = f"insufficient signal to measure (snr={self.current_snr_db:.1f}dB) -- move to better light"
            else:
                reason = f"screen pattern not reflected on face (global r={self.current_score:.2f}, spatial={spatial_str})"

        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        title_scale = h / 110.0
        (tw, _th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, title_scale, 5)
        cv2.putText(canvas, title, (x + (w - tw) // 2, y + int(h * 0.55)),
                    cv2.FONT_HERSHEY_DUPLEX, title_scale, WHITE, 5, cv2.LINE_AA)
        reason_scale = h / 400.0
        (rw, _rh), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, reason_scale, 2)
        cv2.putText(canvas, reason, (x + (w - rw) // 2, y + int(h * 0.80)),
                    cv2.FONT_HERSHEY_SIMPLEX, reason_scale, WHITE, 2, cv2.LINE_AA)
        return y + h

    def _draw_status_row(self, canvas, x, y, w, h):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (12, 12, 12), -1)
        keys = ("[1] LIVE   [2] SWITCH TO INJECTED FEED   [3] TOGGLE LIGHT PATTERN   "
                "[4] COMPARE STORED   [5] TYPING LANE   [Q] QUIT")
        cv2.putText(canvas, keys, (x + 20, y + int(h * 0.42)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
        state_text = (f"state={self.state.value}   FPS: {self._fps():.1f}   "
                      f"lag: {self.current_lag_ms:.0f}ms   {self._camera_name}   "
                      f"pattern: {'ON' if self.emitter.is_enabled() else 'OFF'}")
        if self._rehearse:
            state_text += f"   REHEARSE step {self._rehearse_index + 1}/{len(self._rehearse_script)}"
        if self.status_message:
            state_text += f"   -- {self.status_message}"
        cv2.putText(canvas, state_text, (x + 20, y + int(h * 0.85)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, LIGHT_GRAY, 1, cv2.LINE_AA)
        return y + h

    def _render_replay_screen(self):
        w, h = self.dconfig.dashboard_width, self.dconfig.dashboard_height
        canvas = np.full((h, w, 3), (15, 15, 15), dtype=np.uint8)

        header_h = 70
        cv2.putText(canvas, "STORED COMPARISON: genuine vs. injected", (30, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, WHITE, 2, cv2.LINE_AA)

        status_h = int(h * 0.10)
        body_h = h - header_h - status_h

        if self.genuine_ref is None or self.attack_ref is None:
            msg = "No stored session data yet -- run praesens/session.py to collect sessions first"
            (mw, _mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.putText(canvas, msg, ((w - mw) // 2, header_h + body_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, LIGHT_GRAY, 2, cv2.LINE_AA)
        else:
            half_w = w // 2
            self._draw_reference_panel(canvas, 0, header_h, half_w, body_h,
                                        self.genuine_ref, "GENUINE (bona fide)", GREEN)
            self._draw_reference_panel(canvas, half_w, header_h, w - half_w, body_h,
                                        self.attack_ref, "INJECTED (attack)", RED)
            cv2.line(canvas, (half_w, header_h), (half_w, header_h + body_h), (60, 60, 60), 2)

        self._draw_status_row(canvas, 0, header_h + body_h, w, status_h)
        return canvas

    def _draw_reference_panel(self, canvas, x, y, w, h, record, label, color):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (28, 28, 28), -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + 40), color, -1)
        cv2.putText(canvas, f"{label}  (score={record['score']:.2f})", (x + 15, y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

        pad = 30
        rect = (x + pad, y + 60, x + w - pad, y + h - pad)
        ts = np.array(record["timestamps"])
        if len(ts) >= 2:
            ts = ts - ts[0]
            _plot_series(canvas, rect, ts, record["trace_emitted"], (-3, 3), BLUE, 3)
            _plot_series(canvas, rect, ts, record["trace_measured"], (-3, 3), ORANGE, 3)

        lx, ly = rect[0] + 10, rect[1] + 20
        cv2.line(canvas, (lx, ly), (lx + 25, ly), BLUE, 4)
        cv2.putText(canvas, "emitted", (lx + 32, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.line(canvas, (lx + 140, ly), (lx + 165, ly), ORANGE, 4)
        cv2.putText(canvas, "measured", (lx + 172, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    def _render_typing_screen(self):
        """Milestone 14: live keystroke<->hand coherence, PASSIVE mode --
        type naturally, no phrase shown. Degrades to a plain no_evidence
        message (never a crash, never a fabricated score) if the keystroke
        listener or hand landmarker didn't start -- see enter_typing()."""
        w, h = self.dconfig.dashboard_width, self.dconfig.dashboard_height
        canvas = np.full((h, w, 3), (15, 15, 15), dtype=np.uint8)

        header_h = 70
        cv2.putText(canvas, "TYPING LANE (passive) -- type naturally, no phrase to match", (30, 45),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, WHITE, 2, cv2.LINE_AA)

        status_h = int(h * 0.10)
        body_h = h - header_h - status_h

        if self.current_typing_status == "ok":
            color, title = GREEN, "OK"
        elif self._typing_capture is None or self._typing_hand_landmarker is None:
            color, title = GRAY, "UNAVAILABLE"
        else:
            color, title = GRAY, "NO EVIDENCE"

        banner_h = int(body_h * 0.35)
        cv2.rectangle(canvas, (0, header_h), (w, header_h + banner_h), color, -1)
        (tw, _th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, 1.6, 4)
        cv2.putText(canvas, title, ((w - tw) // 2, header_h + int(banner_h * 0.55)),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, WHITE, 4, cv2.LINE_AA)
        reason = self.current_typing_reason or "keystroke timing tracks hand motion"
        (rw, _rh), _ = cv2.getTextSize(reason, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(canvas, reason, ((w - rw) // 2, header_h + int(banner_h * 0.8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        info_y = header_h + banner_h + 40
        coh_str = "n/a" if np.isnan(self.current_typing_coherence) else f"{self.current_typing_coherence:.2f}"
        lag_str = "n/a" if np.isnan(self.current_typing_lag_ms) else f"{self.current_typing_lag_ms:.0f}ms"
        info_lines = [
            f"keystrokes this window: {self.current_typing_n_keystrokes}",
            f"keystroke <-> hand-motion coherence: {coh_str}   lag: {lag_str}",
            f"confidence (decays with silence): {self.current_typing_confidence:.2f}",
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(canvas, line, (40, info_y + i * 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, LIGHT_GRAY, 1, cv2.LINE_AA)

        plot_y0 = info_y + len(info_lines) * 32 + 20
        plot_y1 = header_h + body_h - 20
        if plot_y1 > plot_y0 + 20 and len(self._typing_hand_ts) >= 2:
            rect = (40, plot_y0, w - 40, plot_y1)
            ts = np.array(self._typing_hand_ts)
            ts = ts - ts[0]
            activity = np.nan_to_num(np.array(self._typing_hand_activity), nan=0.0)
            span = max(float(np.max(activity)), 1e-6) if len(activity) else 1.0
            _plot_series(canvas, rect, ts, activity, (0, span), ORANGE, 2)
            cv2.putText(canvas, "hand activity (rolling window)", (rect[0] + 5, rect[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LIGHT_GRAY, 1, cv2.LINE_AA)

        self._draw_status_row(canvas, 0, header_h + body_h, w, status_h)
        return canvas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Milestone 8/14 panel mode")
    parser.add_argument("--max-seconds", type=float, default=None,
                         help="auto-stop after N seconds (smoke-test use; omit for real demo use)")
    parser.add_argument("--max-camera-probe", type=int, default=6)
    parser.add_argument("--rehearse", action="store_true",
                         help="Milestone 14: run the scripted demo.rehearse_script sequence on a "
                              "timer instead of waiting for keypresses, then exit automatically")
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)

    dashboard = PanelDashboard(raw, max_camera_probe=args.max_camera_probe, rehearse=args.rehearse)
    if args.rehearse:
        print(f"Panel mode running in --rehearse: {len(dashboard._rehearse_script)} scripted steps, "
              f"no keypresses needed. [Q] quits early.")
    else:
        print("Panel mode running. Keys: [1] live, [2] injected feed, [3] toggle pattern, "
              "[4] stored comparison, [5] typing lane, [S] screenshot, [Q] quit.")
    dashboard.run(max_seconds=args.max_seconds)
