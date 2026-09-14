"""Milestone 4/2026-09-02: session runner.

Ties challenge + emitter + optical lane (+ typing lane, since 2026-09-02)
together for one session and writes a single JSON log record. `condition`
is recorded but never influences how the session is measured --
bonafide/replay/swap/emitter_off all run through the identical capture
and scoring path, and only differ in what's physically in front of the
camera or whether the emitter is on. That is what makes the emitter_off
condition a real scientific control rather than a special-cased "off"
mode: the pipeline can't tell it apart from any other session except by
the light pattern itself. Metadata (lighting, distance, makeup, glasses,
subject, skin tone) is recorded per session because Milestone 5's
analysis needs it to check the system doesn't fail quietly for some
conditions more than others.

`--lanes optical+typing` (the default) runs the optical and typing lanes
CONCURRENTLY off ONE shared capture loop -- each frame is grabbed exactly
once and fed to the optical lane's per-frame processor, answering the
SAME session challenge (including a displayed typing phrase) in the SAME
window. `--lanes optical` skips typing entirely and keeps the exact
single-lane path this module has always used (praesens.optical.
run_session, which owns its own loop).

2026-09-03: the typing lane's verdict is keystroke-TIMING only (see
praesens.typing.keystroke_normality_score) -- it no longer depends on the
camera seeing the operator's hands at all, since a webcam framed on the
face for the optical lane physically cannot also frame the hands during
normal typing. The keystroke listener thread still runs for the same
start_time/duration_s window as the shared capture loop (so both lanes
still answer the SAME session challenge concurrently), it just no longer
needs any frame handed to it.

Both lanes' raw numbers feed praesens.fusion.adjudicate_two_lane for one
joint verdict -- an explicit, asymmetric binary gate (not the generic
confidence-weighted Adjudicator used elsewhere in this repo, e.g.
eval/ablate.py): optical is the PRIMARY signal (defeats video injection)
and typing is SECONDARY confirmation of live engagement with this
session's unpredictable challenge, so
  optical PASS + typing PASS  -> ACCEPT
  optical PASS + typing FAIL  -> RE-CHALLENGE (ask them to type again)
  optical FAIL (either way)   -> REJECT (typing can never rescue a
                                  failing optical lane)
2026-09-14: `--lanes optical+typing+acoustic` adds praesens/acoustic.py's
speaker->microphone probe (same session m-sequence, its own thread, same
start_time) as a second SECONDARY lane under the same rule -- ACCEPT
needs all three to pass; optical passing with typing or acoustic failing
is RE-CHALLENGE; optical failing is REJECT.
This prints as a short clean block on the terminal (scores/lag/SNR/every
raw field still go to the JSON log in full -- `fusion`/`typing` are ADDED
fields, nothing existing is removed or renamed).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import platform
import sys
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml

from praesens.camera import open_camera
from praesens.capture import CaptureConfig, configure_capture_format, measure_steady_state_fps
from praesens.challenge import Challenge, generate_challenge_phrase, pick_auto_chip_rate
from praesens.emit import Emitter, EmitterConfig
from praesens.fusion import (
    LaneResult, AdjudicatorConfig, acoustic_lane_stub,
    adjudicate_two_lane, lane_passes, lane_contributes, LANE_LAG_BOUNDS_MS,
)
from praesens.optical import (
    OpticalConfig, OpticalFrameProcessor, OpticalResult, run_session,
    lock_camera, measure_capture_fps, check_nyquist,
)
from praesens.typing import TypingConfig, KeystrokeCapture, keystroke_normality_score
from praesens.acoustic import AcousticConfig, run_acoustic_session, acoustic_confidence

VALID_CONDITIONS = {
    "bonafide", "emitter_off",
    "replay", "swap",  # Milestones 4/5's original attack labels -- kept for the existing corpus
    # Milestone 12's expanded attack vocabulary, matching eval/corpus_plan.yaml:
    "inject_static", "inject_swap", "inject_reenact", "inject_adaptive",
}

VALID_LANES = ("optical", "optical+typing", "optical+typing+acoustic")

REPO_ROOT = Path(__file__).resolve().parent.parent


def generate_session_id() -> str:
    return time.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]


def load_config(config_path: str | Path = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# LaneResult wrapping -- confidence formulas match established precedent,
# not invented fresh here. Optical: identical to eval/ablate.py's
# optical_lane_result_from_log() (SNR margin above zero, clipped to
# [0.05, 1.0]) -- same formula, reused, so a live session and an offline-
# analysed log agree on what "confidence" means for this lane.
# ---------------------------------------------------------------------------

def optical_lane_result(result: OpticalResult) -> LaneResult:
    if result.n_frames == 0:
        return LaneResult(lane_name="optical", subscore=None, status="no_evidence",
                           lag_ms=None, confidence=0.0, diagnostics="no frames captured")
    if result.insufficient_signal:
        return LaneResult(lane_name="optical", subscore=None, status="insufficient_signal",
                           lag_ms=None, confidence=0.0, diagnostics=f"snr_db={result.snr_db}")
    snr_db = result.snr_db
    confidence = 0.3 if math.isnan(snr_db) else float(np.clip(snr_db / 15.0, 0.05, 1.0))
    return LaneResult(lane_name="optical", subscore=result.score, status="ok",
                       lag_ms=result.lag_ms, confidence=confidence,
                       diagnostics=f"snr_db={snr_db:.1f}" if not math.isnan(snr_db) else "snr_db=nan")


def typing_lane_result(events: list, expected_phrase: str | None, tconfig: TypingConfig) -> LaneResult:
    """2026-09-03: keystroke-TIMING only (see
    praesens.typing.keystroke_normality_score) -- replaced the earlier
    hand-coherence-based version (Milestone 10/2026-09-02) because a
    webcam framed on the face for the optical lane cannot also see the
    hands during normal typing, which made this lane read as no_evidence
    in practice regardless of whether genuine typing happened (see the
    module docstring's 2026-09-03 note). confidence is fixed at 1.0 when
    status=='ok' -- there's no per-session signal-quality measure for
    pure keystroke timing the way SNR is for optical; the subscore itself
    already reflects both match correctness and timing plausibility, so
    a confident-but-low subscore is exactly how a badly-typed or
    suspiciously-timed attempt is expressed. lag_ms is always None: this
    lane no longer measures anything against a camera-observed stimulus,
    so there is no lag to report (see LANE_LAG_BOUNDS_MS's "typing" entry,
    which now only applies to praesens.fusion.Adjudicator's generic path,
    not this lane's own binary-gate result)."""
    status, subscore, diagnostics = keystroke_normality_score(events, expected_phrase, tconfig)
    if status != "ok":
        return LaneResult(lane_name="typing", subscore=None, status=status,
                           lag_ms=None, confidence=0.0, diagnostics=diagnostics)
    return LaneResult(lane_name="typing", subscore=subscore, status="ok",
                       lag_ms=None, confidence=1.0, diagnostics=diagnostics)


def _typing_record_dict(events: list, expected_phrase: str | None, tconfig: TypingConfig,
                         typing_lane: LaneResult, typing_pass_threshold: float,
                         start_time: float, duration_s: float) -> dict:
    """Builds the JSON log's "typing" field. Keeps every field NAME the
    2026-09-02 schema already used (add-only, see module docstring) --
    "coherence_score"/"coherence_lag_ms" are now always None because
    hand-tracking no longer runs in this path (2026-09-03), not because
    the fields were removed; "subscore"/"diagnostics"/"pass_threshold"
    are the new fields this milestone adds."""
    n_matched = sum(1 for e in events if e.get("matched_expected") is True)
    n_expected = len(expected_phrase) if expected_phrase else None

    t_downs = sorted(e["t_down"] for e in events)
    intervals = [b - a for a, b in zip(t_downs, t_downs[1:])]
    inter_key_mean_s = (sum(intervals) / len(intervals)) if intervals else float("nan")
    inter_key_std_s = float(np.std(intervals)) if len(intervals) >= 2 else float("nan")

    session_end = start_time + duration_s
    last_kt = max(t_downs) if t_downs else None
    seconds_since_last_keystroke = (session_end - last_kt) if last_kt is not None else float("inf")

    return {
        "mode": "active", "status": typing_lane.status,
        "n_keystrokes": len(events), "n_matched": n_matched, "n_expected": n_expected,
        "inter_key_mean_s": inter_key_mean_s, "inter_key_std_s": inter_key_std_s,
        "coherence_score": None, "coherence_lag_ms": None,
        "seconds_since_last_keystroke": seconds_since_last_keystroke,
        "events": events,  # t_down/t_up/matched_expected ONLY -- see praesens/typing.py's privacy contract
        # -- added 2026-09-03, ADD-ONLY: keystroke-timing-only scoring --
        "subscore": typing_lane.subscore,
        "diagnostics": typing_lane.diagnostics,
        "pass_threshold": typing_pass_threshold,
    }

def acoustic_lane_result(result) -> LaneResult:
    if result.status != "ok":
        return LaneResult(lane_name="acoustic", subscore=None, status=result.status,
                           lag_ms=None, confidence=0.0, diagnostics=result.diagnostics)
    confidence = acoustic_confidence(result.snr_db)
    return LaneResult(lane_name="acoustic", subscore=result.score, status="ok",
                       lag_ms=result.lag_ms, confidence=confidence, diagnostics=result.diagnostics)

def _draw_phrase_overlay(frame: np.ndarray, phrase: str, chars_matched: int) -> np.ndarray:
    """A COPY of frame with the challenge phrase drawn across the top --
    the caller already fed the UNMODIFIED frame to optical's ROI sampling
    before this runs, so this cannot contaminate the luminance
    measurement. The matched prefix (a COUNT only -- see KeystrokeCapture.
    chars_matched) is highlighted; the phrase text itself is not secret,
    it's generated from the session seed and displayed for the operator
    to read, the same way the light challenge is fully reconstructable
    from the seed alone."""
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    banner_h = max(60, h // 8)
    cv2.rectangle(canvas, (0, 0), (w, banner_h), (20, 20, 20), -1)
    cv2.putText(canvas, "TYPE:", (10, max(18, banner_h // 2 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (200, 200, 200), 1, cv2.LINE_AA)

    matched = phrase[:chars_matched]
    remaining = phrase[chars_matched:]
    scale = float(np.clip((w - 20) / max(8 * len(phrase), 1), 0.5, 1.2))
    y = banner_h - 12
    (mw, _mh), _ = cv2.getTextSize(matched, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(canvas, matched, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (80, 220, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, remaining, (10 + mw, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _run_concurrent_lanes(cap, challenge: Challenge, oconfig: OpticalConfig,
                           emitter: Emitter, start_time: float, duration_s: float,
                           expected_phrase: str) -> tuple[OpticalResult, list]:
    """ONE shared capture loop: grabs each frame exactly once and feeds it
    to the optical ROI processor -- see module docstring. The typing lane
    (2026-09-03: keystroke-timing only, see
    praesens.typing.keystroke_normality_score) needs no frames at all any
    more -- its keystroke listener runs independently on its own thread
    for this same start_time/duration_s window, so it's started/stopped
    around this loop but never touches a frame. This still satisfies the
    "exactly one cv2.VideoCapture consumer, one grab per frame" rule this
    function has always followed: typing was never a second CAMERA
    consumer to begin with, only its former hand-tracking piece was, and
    that piece is what's removed here. Mirrors run_one_session()'s own
    existing platform branch (macOS needs the emitter's HighGUI window on
    the main thread; Windows/Linux the reverse) since this replaces the
    single-lane run_session() call in that branch, not the branch itself."""
    warn_list: list = []
    exposure_locked = lock_camera(cap, oconfig, warn_list)

    optical_processor = OpticalFrameProcessor(oconfig)  # owns its own FaceLandmarker

    keystroke_capture = KeystrokeCapture(expected_phrase=expected_phrase)
    keystroke_unavailable_reason = None
    try:
        keystroke_capture.start()
    except Exception as e:
        keystroke_unavailable_reason = str(e)
        print(f"WARNING: could not start keystroke listener -- typing lane will show no_evidence: {e}")

    preflight_fps = measure_capture_fps(cap)
    preflight_warning = check_nyquist(preflight_fps, challenge.chip_rate_hz, oconfig.min_fps_multiple_of_chip_rate)
    if preflight_warning:
        msg = f"PRE-FLIGHT (optimistic, no face required to trigger this): {preflight_warning}"
        print("!" * 70)
        print(f"WARNING: {msg}")
        print("!" * 70)
        warn_list.append(msg)

    def _loop() -> None:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= duration_s:
                break

            grabbed = cap.grab()
            t = time.perf_counter()
            if not grabbed:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue

            boost_msg = optical_processor.process_frame(frame, t, start_time, challenge, emitter)
            if boost_msg:
                warn_list.append(boost_msg)

            preview = _draw_phrase_overlay(frame, expected_phrase, keystroke_capture.chars_matched)
            emitter.set_preview(preview)

    try:
        if platform.system() == "Darwin":
            # See run_one_session()'s single-lane branch for why: Cocoa
            # requires HighGUI calls on the main thread, so capture/scoring
            # moves to a background thread here and the emitter blocks
            # the main thread instead.
            capture_thread = threading.Thread(target=_loop, daemon=True)
            capture_thread.start()
            try:
                emitter.run_blocking(start_time, duration_s)
            finally:
                capture_thread.join(timeout=duration_s + 5.0)
        else:
            emitter.start(start_time, duration_s)
            try:
                _loop()
            finally:
                emitter.stop()
    finally:
        keystroke_capture.stop()
        optical_processor.close()

    optical_result = optical_processor.finalize(challenge, emitter, exposure_locked, warn_list)

    events = keystroke_capture.get_events()
    if keystroke_unavailable_reason:
        # keystroke_normality_score already decides "no_evidence" from
        # zero events on its own; this just makes WHY visible in the
        # log's warnings, without touching optical_result otherwise.
        optical_result.warnings = optical_result.warnings + [
            f"keystroke listener unavailable: {keystroke_unavailable_reason}"
        ]

    return optical_result, events


def run_one_session(condition: str, meta: dict, raw_config: dict | None = None,
                     camera_index_override: int | None = None,
                     output_dir: str | Path = "logs", lanes: str = "optical") -> tuple[dict, Path]:
    """lanes defaults to "optical" here (this function's own default,
    called directly by scripts/collect.py and scripts/run_corpus.py,
    neither of which asked for typing) -- the `python -m praesens.session`
    CLI below defaults its OWN --lanes flag to "optical+typing" instead,
    per the brief. Existing callers that don't pass lanes= are completely
    unaffected by this milestone."""
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"condition must be one of {sorted(VALID_CONDITIONS)}, got {condition!r}")
    if lanes not in VALID_LANES:
        raise ValueError(f"lanes must be one of {VALID_LANES}, got {lanes!r}")

    if raw_config is None:
        raw_config = load_config()

    oconfig = OpticalConfig.from_dict(raw_config["optical"])
    model_path = Path(oconfig.model_path)
    oconfig.model_path = str(model_path if model_path.is_absolute() else REPO_ROOT / model_path)
    if camera_index_override is not None:
        oconfig.camera_index = camera_index_override

    # Camera must be open BEFORE the challenge/emitter are built when
    # auto_chip_rate is on, since the chip rate depends on a quick FPS
    # measurement of this specific camera -- deciding it from a fixed
    # config value first (the old order) would defeat the point of FIX 2c.
    cap = open_camera(oconfig.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {oconfig.camera_index}")

    try:
        # Format negotiated BEFORE the auto_chip_rate preflight measurement
        # below, so that measurement reflects the true achievable throughput
        # (MJPG vs whatever the driver defaulted to) -- not touching chip-rate
        # selection itself, only what fps it's given to work with.
        cconfig = CaptureConfig.from_dict(raw_config.get("capture", {}))
        capture_warn_list: list = []
        configure_capture_format(cap, cconfig, capture_warn_list)
        for w in capture_warn_list:
            print(f"WARNING: {w}")

        challenge_cfg = dict(raw_config["challenge"])
        auto_chip_rate_used = bool(raw_config["optical"].get("auto_chip_rate", False))
        if auto_chip_rate_used:
            # Bug fixed 2026-09-01: this measurement used to run BEFORE exposure
            # was ever locked (lock_camera only happened later, inside
            # run_session() below) -- see praesens.capture.measure_steady_state_fps's
            # docstring for the full story. Locking exposure here is safe even
            # though run_session() will lock it again later (idempotent).
            preflight_warn_list: list = []
            warmup_frames = raw_config["optical"].get("camera_warmup_frames", 30)
            preflight_fps = measure_steady_state_fps(cap, oconfig, warmup_frames, preflight_warn_list)
            for w in preflight_warn_list:
                print(f"WARNING: {w}")

            chip_rate_hz, duration_s = pick_auto_chip_rate(
                preflight_fps,
                divisor=raw_config["optical"].get("auto_chip_rate_divisor", 6.0),
                min_hz=0.5, max_hz=5.0,
                min_chips=raw_config["optical"].get("auto_chip_rate_min_chips", 60),
                base_duration_s=challenge_cfg.get("duration_s", 20.0),
            )
            print(f"auto_chip_rate: measured preflight FPS={preflight_fps:.1f} -> "
                  f"chip_rate_hz={chip_rate_hz:.2f}, duration_s={duration_s:.1f}")
            challenge_cfg["chip_rate_hz"] = chip_rate_hz
            challenge_cfg["duration_s"] = duration_s

        challenge = Challenge(**challenge_cfg)

        econfig = EmitterConfig.from_dict(raw_config["emitter"])
        if condition == "emitter_off":
            econfig.emitter_enabled = False

        emitter = Emitter(challenge, econfig)
        session_id = generate_session_id()

        typing_enabled = lanes in ("optical+typing", "optical+typing+acoustic")
        acoustic_enabled = lanes == "optical+typing+acoustic"
        tconfig = None
        expected_phrase = None
        if typing_enabled:
            tconfig = TypingConfig.from_dict(raw_config.get("typing", {}))
            tconfig.hand_model_path = str(REPO_ROOT / tconfig.hand_model_path)
            expected_phrase = generate_challenge_phrase(
                challenge.seed, n_words=tconfig.n_words,
                generative=raw_config.get("typing", {}).get("generative", False),
            )

        start_time = time.perf_counter()
        typing_events: list | None = None

        # Acoustic doesn't touch the camera, so it can't slot into
        # _run_concurrent_lanes()'s shared grab/retrieve loop the way
        # optical+typing do -- but for the joint-temporal-coherence claim
        # to mean anything, it must still answer the SAME challenge in
        # the SAME window as whichever camera path runs below, not
        # before or after it. Started on its own thread at the same
        # shared start_time; run_acoustic_session() blocks internally
        # for its own duration via sd.sleep(), so this thread finishes
        # on its own without the camera path needing to wait for it.
        acoustic_thread: threading.Thread | None = None
        acoustic_box: dict = {}
        if acoustic_enabled:
            aconfig = AcousticConfig.from_dict(raw_config.get("acoustic", {}))

            def _acoustic_worker():
                try:
                    acoustic_box["result"] = run_acoustic_session(challenge, aconfig)
                except Exception as exc:
                    acoustic_box["error"] = exc

            acoustic_thread = threading.Thread(target=_acoustic_worker, daemon=True)
            acoustic_thread.start()

        if typing_enabled:
            optical_result, typing_events = _run_concurrent_lanes(
                cap, challenge, oconfig, emitter, start_time, challenge.duration_s, expected_phrase
            )
        elif platform.system() == "Darwin":
            # macOS requires the emitter's OpenCV window to run on the main
            # thread (Cocoa raises an unrecoverable cv2.error otherwise), so we
            # flip which piece owns the main thread here: capture/scoring moves
            # to a background thread, and the emitter's window loop blocks the
            # main thread instead -- the reverse of the Windows/Linux path below.
            result_box: dict = {}
            error_box: dict = {}

            def _capture_worker():
                try:
                    result_box["result"] = run_session(
                        cap, challenge, oconfig, start_time, challenge.duration_s, emitter=emitter
                    )
                except Exception as exc:
                    error_box["error"] = exc

            capture_thread = threading.Thread(target=_capture_worker, daemon=True)
            capture_thread.start()
            try:
                emitter.run_blocking(start_time, challenge.duration_s)
            finally:
                capture_thread.join(timeout=challenge.duration_s + 5.0)

            if "error" in error_box:
                raise error_box["error"]
            optical_result = result_box["result"]
        else:
            emitter.start(start_time, challenge.duration_s)
            try:
                optical_result = run_session(cap, challenge, oconfig, start_time, challenge.duration_s,
                                              emitter=emitter)
            finally:
                emitter.stop()

        if acoustic_thread is not None:
            # + lag-search tail recording, stream start-up and detection time
            acoustic_thread.join(timeout=challenge.duration_s + 10.0)
            if "error" in acoustic_box:
                # A crashed acoustic thread must not silently look like
                # "no_evidence" (that's a different, honest claim -- see
                # praesens/acoustic.py's own module docstring on this
                # point) or take down a session whose optical/typing
                # results are already valid; log it and fall through to
                # acoustic_lane_stub() below via acoustic_result staying None.
                print(f"WARNING: acoustic lane thread raised: {acoustic_box['error']}")
    finally:
        cap.release()

    # -- fusion: build LaneResults and adjudicate. 2026-09-03: the verdict
    # comes from adjudicate_two_lane(), an explicit binary gate (see module
    # docstring), not the generic confidence-weighted Adjudicator used
    # elsewhere in this repo (e.g. eval/ablate.py, still available and
    # unaffected). --lanes optical passes typing=None through, which
    # collapses to a plain optical-only ACCEPT/REJECT (no RE-CHALLENGE --
    # there's no secondary lane to ask the operator to retry on).
    optical_lane = optical_lane_result(optical_result)
    typing_lane = None
    typing_pass_threshold = (tconfig.pass_match_fraction if tconfig is not None
                              else raw_config.get("typing", {}).get("pass_match_fraction", 0.7))
    if typing_events is not None:
        typing_lane = typing_lane_result(typing_events, expected_phrase, tconfig)

    # 2026-09-14: acoustic is a real secondary lane when --lanes includes
    # it. A crashed acoustic thread (requested, but no result) never ran,
    # so it doesn't gate the verdict -- it's logged as the is_stub
    # fallback below, and the WARNING printed above says why.
    acoustic_result = acoustic_box.get("result") if acoustic_enabled else None
    acoustic_lane = acoustic_lane_result(acoustic_result) if acoustic_result is not None else None

    fusion_config = AdjudicatorConfig.from_dict(raw_config.get("fusion", {}))
    # optical's own pass bar reuses reject_threshold -- the same bar this
    # module's terminal/banner display has always required an optical
    # subscore to clear (see _lane_display_rows, below). Acoustic's subscore
    # is on the same [0, 1] correlation scale (noise-floor corrected, see
    # praesens.acoustic.detect_probe), so it reuses that bar.
    optical_pass_threshold = fusion_config.reject_threshold
    acoustic_pass_threshold = fusion_config.reject_threshold
    joint_result = adjudicate_two_lane(
        optical_lane, typing_lane,
        optical_pass_threshold=optical_pass_threshold,
        typing_pass_threshold=typing_pass_threshold,
        acoustic=acoustic_lane,
        acoustic_pass_threshold=acoustic_pass_threshold,
    )

    # lane_results feeds the JSON log's "fusion.lanes" list below (unchanged
    # shape); acoustic falls back to the is_stub entry whenever it didn't run.
    lane_results = [optical_lane]
    if typing_lane is not None:
        lane_results.append(typing_lane)
    lane_results.append(acoustic_lane if acoustic_lane is not None else acoustic_lane_stub())

    record = {
        "session": session_id,
        "condition": condition,
        "score": optical_result.score,
        "lag_ms": optical_result.lag_ms,
        "seed": challenge.seed,
        "chip_rate_hz": challenge.chip_rate_hz,
        "duration_s": challenge.duration_s,
        "start_time_perf_counter": start_time,
        "auto_chip_rate_used": auto_chip_rate_used,
        "emitter_enabled": econfig.emitter_enabled,
        "snr_db": optical_result.snr_db,
        "insufficient_signal": optical_result.insufficient_signal,
        "adaptive_boost_applied": optical_result.adaptive_boost_applied,
        "exposure_locked": optical_result.exposure_locked,
        "n_frames": optical_result.n_frames,
        "n_face_detected": optical_result.n_face_detected,
        "measured_fps": optical_result.measured_fps,
        "warnings": optical_result.warnings,
        "meta": meta,
        "trace_emitted": optical_result.trace_emitted,
        "trace_measured": optical_result.trace_measured,
        "timestamps": optical_result.timestamps,
        # -- added 2026-09-02, ADD-ONLY: concurrent typing lane + fusion verdict --
        "lanes": lanes,
        "typing_phrase": expected_phrase,  # not secret -- seed-derived and shown on screen; keystrokes are what's never logged
        "typing": (_typing_record_dict(typing_events, expected_phrase, tconfig, typing_lane,
                                        typing_pass_threshold, start_time, challenge.duration_s)
                   if typing_events is not None else None),
        "acoustic": ({
            "status": acoustic_result.status, "score": acoustic_result.score,
            "lag_ms": acoustic_result.lag_ms, "snr_db": acoustic_result.snr_db,
            "diagnostics": acoustic_result.diagnostics,
        } if acoustic_result is not None else None),
        "fusion": {
            "verdict": joint_result.verdict,
            "joint_score": (None if math.isnan(joint_result.joint_score) else joint_result.joint_score),
            "reason_text": joint_result.reason_text,
            "per_lane_reasons": joint_result.per_lane_reasons,
            "accept_threshold": fusion_config.accept_threshold,
            "reject_threshold": fusion_config.reject_threshold,
            "lanes": [
                {"lane_name": l.lane_name, "subscore": l.subscore, "status": l.status,
                 "lag_ms": l.lag_ms, "confidence": l.confidence, "diagnostics": l.diagnostics,
                 "is_stub": l.is_stub}
                for l in lane_results
            ],
            # -- added 2026-09-03, ADD-ONLY: the actual binary-gate policy --
            "policy": "adjudicate_two_lane",
            "optical_pass_threshold": optical_pass_threshold,
            "typing_pass_threshold": typing_pass_threshold,
            "acoustic_pass_threshold": acoustic_pass_threshold,
        },
    }

    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.json"
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    return record, out_path


# ---------------------------------------------------------------------------
# Terminal presentation -- a short, clean block, not the raw numbers (those
# are always in the JSON log, see run_one_session's record dict above).
# ---------------------------------------------------------------------------

_LANE_DISPLAY_NAMES = {"optical": "Optical lane", "typing": "Typing lane ", "acoustic": "Acoustic lane"}


def _lane_display_rows(record: dict) -> tuple[list, list]:
    """Shared by print_clean_block() and the on-screen verdict banner, so
    the terminal and the screen never disagree. Returns
    ([(label, status_word, detail), ...], passing_lane_names). 2026-09-03:
    PASS is exactly what adjudicate_two_lane() itself requires for each
    lane (see praesens/fusion.py) -- for optical, lane_passes() (subscore
    clears optical_pass_threshold) AND lane_contributes() (measured lag
    against the emitted light stimulus falls in its plausible window,
    LANE_LAG_BOUNDS_MS); for typing, lane_passes() alone (subscore clears
    typing_pass_threshold) -- typing has no lag left to check now that
    its scoring is keystroke-timing-only (see praesens/typing.py's
    keystroke_normality_score)."""
    fusion = record["fusion"]
    lane_dicts = fusion["lanes"]
    optical_pass_threshold = fusion.get("optical_pass_threshold", fusion["reject_threshold"])
    typing_pass_threshold = fusion.get("typing_pass_threshold", 0.7)
    acoustic_pass_threshold = fusion.get("acoustic_pass_threshold", optical_pass_threshold)

    rows, passing = [], []
    for l in lane_dicts:
        # is_stub (default False) is what actually gates display now, not
        # merely appearing in _LANE_DISPLAY_NAMES -- acoustic is IN that
        # dict unconditionally (see above), because it needs a label for
        # the sessions where it DID run for real. Defaulting missing
        # is_stub to False assumes "real lane" for any dict that predates
        # this field (correct for optical/typing, which were never
        # stubs); the one case this doesn't cover -- replaying a
        # pre-patch acoustic-stub log through this function -- isn't how
        # print_clean_block is actually used (always called on a
        # freshly-built same-run record, never a reloaded historical one).
        if l["lane_name"] not in _LANE_DISPLAY_NAMES or l.get("is_stub", False):
            continue
        label = _LANE_DISPLAY_NAMES[l["lane_name"]]
        lane = LaneResult(lane_name=l["lane_name"], subscore=l["subscore"], status=l["status"],
                           lag_ms=l["lag_ms"], confidence=l["confidence"], diagnostics=l["diagnostics"])

        if l["lane_name"] == "typing":
            lag_plausible = True  # not applicable -- typing has no lag concept any more
            passes = lane_passes(lane, typing_pass_threshold)
        else:
            # optical and acoustic both measure a physical lag against the emitted challenge
            lag_plausible = lane_contributes(lane, LANE_LAG_BOUNDS_MS)
            threshold = optical_pass_threshold if l["lane_name"] == "optical" else acoustic_pass_threshold
            passes = lane_passes(lane, threshold) and lag_plausible

        if l["status"] != "ok":
            status_word, detail = "NO EVIDENCE", f"({l['diagnostics']})" if l["diagnostics"] else ""
        elif passes:
            status_word = "PASS"
            metric = "match" if l["lane_name"] == "typing" else "score"
            detail = f"({metric} {l['subscore']:.2f})"
            passing.append(l["lane_name"])
        elif not lag_plausible:
            status_word, detail = "FAIL", f"(subscore {l['subscore']:.2f}, implausible lag)"
        else:
            status_word, detail = "FAIL", f"(subscore {l['subscore']:.2f}, below pass threshold)"
        rows.append((label, status_word, detail))
    return rows, passing


def _friendly_reason(record: dict) -> str:
    """adjudicate_two_lane()'s own reason_text, verbatim -- nothing
    invented here (see praesens/fusion.py: it already produces a clean
    sentence for every case, ACCEPT included, e.g. "both lanes agree:
    optical 0.80, typing 0.90" or "optical lane passes (score 0.80)")."""
    return record["fusion"]["reason_text"]


def print_clean_block(record: dict) -> None:
    fusion = record["fusion"]
    rows, passing = _lane_display_rows(record)
    verdict = fusion["verdict"]
    joint_score = fusion["joint_score"]

    header = f"PRAESENS  --  session {record['session']}   [{record['condition']}]"
    width = max(46, len(header) + 4)
    print("\n" + "=" * width)
    print(f"  {header}")
    print("=" * width)

    for label, status_word, detail in rows:
        print(f"  {label} : {status_word:<12s} {detail}")

    print("  " + "-" * (width - 4))
    js = f"{joint_score:.2f}" if joint_score is not None else "n/a"
    print(f"  VERDICT      : {verdict:<12s} (joint {js})")
    print(f"  Reason       : {_friendly_reason(record)}")
    print("=" * width)


# Verdict -> (background, text) colour, BGR. For an audience the colour IS
# the message, and it follows praesens.fusion.adjudicate_two_lane exactly:
#   green  ACCEPT        every lane that ran passed (optical + typing + acoustic)
#   yellow RE-CHALLENGE  optical passed but a secondary lane (typing or
#                        acoustic) didn't -- retry, not a rejection
#   red    REJECT        optical failed; no other lane can rescue that
# Dark text on yellow: white on yellow is unreadable from across a room.
_BANNER_STYLES = {
    "ACCEPT": ((60, 180, 60), (255, 255, 255)),
    "RE-CHALLENGE": ((0, 215, 255), (20, 20, 20)),
    "REJECT": ((40, 40, 200), (255, 255, 255)),
}


def _wrap_text(text: str, font: int, scale: float, thickness: int, max_width: int,
               max_lines: int = 3) -> list:
    """Greedy word wrap to max_width pixels -- a RE-CHALLENGE reason can
    carry a lane's full diagnostics (acoustic's run long) and would
    otherwise run off both edges of the screen."""
    lines, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and cv2.getTextSize(candidate, font, scale, thickness)[0][0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;") + " ..."
    return lines


def render_verdict_banner(record: dict, width: int, height: int) -> np.ndarray:
    """The banner image itself, with no window -- so tests can check
    exactly what the audience sees. Verdict/rows/reason come from the SAME
    helpers print_clean_block() uses, so the screen and the terminal never
    disagree."""
    verdict = record["fusion"]["verdict"]
    rows, _passing = _lane_display_rows(record)
    reason = _friendly_reason(record)
    background, text_color = _BANNER_STYLES.get(verdict, ((90, 90, 90), (255, 255, 255)))
    frame = np.full((height, width, 3), background, dtype=np.uint8)

    title_scale = height / 220.0
    (tw, _th), _ = cv2.getTextSize(verdict, cv2.FONT_HERSHEY_DUPLEX, title_scale, 8)
    cv2.putText(frame, verdict, ((width - tw) // 2, int(height * 0.40)), cv2.FONT_HERSHEY_DUPLEX,
                title_scale, text_color, 8, cv2.LINE_AA)

    row_step = int(height * 0.06)
    for i, (label, status_word, _detail) in enumerate(rows):
        line = f"{label.strip()}: {status_word}"
        scale = height / 700.0
        (lw, _lh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        cv2.putText(frame, line, ((width - lw) // 2, int(height * 0.53) + i * row_step),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 3, cv2.LINE_AA)

    reason_scale = height / 900.0
    reason_y = int(height * 0.53) + len(rows) * row_step + int(height * 0.05)
    for j, line in enumerate(_wrap_text(reason, cv2.FONT_HERSHEY_SIMPLEX, reason_scale, 2, int(width * 0.9))):
        (rw, _rh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, reason_scale, 2)
        cv2.putText(frame, line, ((width - rw) // 2, reason_y + j * int(height * 0.045)),
                    cv2.FONT_HERSHEY_SIMPLEX, reason_scale, text_color, 2, cv2.LINE_AA)
    return frame


def show_verdict_banner(record: dict, hold_seconds: float) -> None:
    """A big, fullscreen, colour-coded ACCEPT/RE-CHALLENGE/REJECT banner --
    green/yellow/red, see _BANNER_STYLES -- for demo purposes: an audience
    reads a fullscreen colour instantly, they don't read a terminal. This
    is presentation only, no new pass/fail logic (render_verdict_banner).
    Opens its own short-lived window (the emitter's own challenge window
    has already closed by the time a verdict exists, since fusion runs
    after the capture loop) using the same fullscreen technique as
    praesens.emit.Emitter._run(); never raises on a windowless/headless
    environment -- a demo failing to SHOW the verdict must not crash the
    session that already computed and saved it."""
    try:
        from praesens.emit import _screen_resolution
        width, height = _screen_resolution(1920, 1080)
        frame = render_verdict_banner(record, width, height)

        window_name = "praesens_verdict"
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        start = time.perf_counter()
        try:
            while time.perf_counter() - start < hold_seconds:
                cv2.imshow(window_name, frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break
        finally:
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
    except Exception as e:
        print(f"WARNING: could not show the on-screen verdict banner (session result is unaffected, "
              f"already saved): {e}")


@contextlib.contextmanager
def _quiet_console(log_path: Path):
    """Redirects BOTH Python's own print() output AND the raw OS file
    descriptors (fd 1/2) to log_path -- MediaPipe's C++ backend (glog/
    TensorFlow Lite) writes those "W0000 ..."/"INFO: Created TensorFlow
    Lite..." lines directly to the underlying file descriptor, bypassing
    anything that only reassigns sys.stdout/sys.stderr at the Python
    level (contextlib.redirect_stdout alone does NOT catch this). Nothing
    is lost -- it's all still in log_path -- this only keeps it off the
    terminal so the clean block (printed AFTER this context exits) is the
    only thing the operator sees, per the brief's --verbose requirement."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout_fd, 1)
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)


if __name__ == "__main__":
    # Best-effort reduction at the source, in addition to the fd-level
    # redirect below -- must be set before mediapipe's C++ backend
    # initialises (i.e. before any landmarker is created), so this has to
    # happen at true module-import time, ahead of anything that could
    # trigger it.
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

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
    parser.add_argument("--lanes", choices=VALID_LANES, default="optical+typing",
                         help="which lanes run concurrently off the shared capture loop")
    parser.add_argument("--verbose", action="store_true",
                         help="show full diagnostic output (capture format, preflight, MediaPipe "
                              "logging) live on the terminal instead of routing it to a log file")
    parser.add_argument("--no-banner", action="store_true",
                         help="skip the fullscreen ACCEPT/REJECT/RE-CHALLENGE banner shown on screen "
                              "after the session (the clean terminal block always still prints)")
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
    typing_note = " -- type the on-screen phrase while it plays" if "typing" in args.lanes else ""
    print(f"Starting {args.condition} session, lanes={args.lanes} "
          f"({raw_config['challenge']['duration_s']}s)... look at the screen{typing_note}.")

    console_log_path = None
    redirect_ctx = contextlib.nullcontext()
    if not args.verbose:
        console_log_path = REPO_ROOT / "logs" / f"console_{int(time.time())}.log"
        redirect_ctx = _quiet_console(console_log_path)

    with redirect_ctx:
        record, out_path = run_one_session(args.condition, meta, raw_config=raw_config,
                                            camera_index_override=args.camera_index, lanes=args.lanes)

    print_clean_block(record)
    if console_log_path is not None:
        print(f"(full diagnostic output: {console_log_path})")
    print(f"saved to {out_path}")

    if not args.no_banner:
        show_verdict_banner(record, raw_config["demo"].get("verdict_banner_hold_s", 4.0))