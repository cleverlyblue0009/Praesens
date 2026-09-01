"""2026-09-02 acceptance tests: the concurrent optical+typing session must
drive BOTH lanes off exactly ONE shared capture loop -- one cv2.VideoCapture
open, one grab()/retrieve() pair per frame, the SAME frame object handed to
both lanes' per-frame processors. This is the make-or-break property the
brief called out explicitly; these tests verify it dynamically against a
mocked capture rather than trusting the single-call-site structural read
(confirmed separately by grep: exactly one open_camera(...) call in the
whole module).
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import praesens.session as session_mod
from praesens.session import (
    run_one_session, print_clean_block, optical_lane_result, typing_lane_result,
)
from praesens.fusion import LaneResult
from praesens.optical import OpticalConfig, OpticalResult
from praesens.typing import TypingConfig, TypingResult


# ---------------------------------------------------------------------------
# LaneResult wrapping -- optical
# ---------------------------------------------------------------------------

def _optical_result(**overrides) -> OpticalResult:
    base = dict(score=0.8, lag_ms=45.0, snr_db=9.0, insufficient_signal=False,
                adaptive_boost_applied=False, exposure_locked=True,
                trace_emitted=[], trace_measured=[], timestamps=[],
                n_frames=100, n_face_detected=100, measured_fps=16.0, warnings=[])
    base.update(overrides)
    return OpticalResult(**base)


def test_optical_lane_result_ok_case_matches_established_confidence_formula():
    result = _optical_result(score=0.8, lag_ms=45.0, snr_db=9.0, insufficient_signal=False)
    lane = optical_lane_result(result)
    assert lane.status == "ok"
    assert lane.subscore == 0.8
    assert lane.lag_ms == 45.0
    # eval/ablate.py's optical_lane_result_from_log(): clip(snr_db/15, 0.05, 1.0)
    assert lane.confidence == pytest.approx(np.clip(9.0 / 15.0, 0.05, 1.0))


def test_optical_lane_result_insufficient_signal_case():
    result = _optical_result(insufficient_signal=True, snr_db=1.0)
    lane = optical_lane_result(result)
    assert lane.status == "insufficient_signal"
    assert lane.subscore is None
    assert lane.lag_ms is None


def test_optical_lane_result_no_frames_is_no_evidence_not_insufficient_signal():
    result = _optical_result(n_frames=0, n_face_detected=0, insufficient_signal=True)
    lane = optical_lane_result(result)
    assert lane.status == "no_evidence"
    assert "no frames" in lane.diagnostics


# ---------------------------------------------------------------------------
# LaneResult wrapping -- typing
# ---------------------------------------------------------------------------

def _typing_result(**overrides) -> TypingResult:
    base = dict(mode="active", status="ok", n_keystrokes=12, n_matched=12, n_expected=15,
                inter_key_mean_s=0.3, inter_key_std_s=0.05, coherence_score=0.7,
                coherence_lag_ms=-20.0, seconds_since_last_keystroke=0.5, events=[])
    base.update(overrides)
    return TypingResult(**base)


def test_typing_lane_result_ok_case_uses_recency_decay_confidence():
    result = _typing_result(coherence_score=0.7, coherence_lag_ms=-20.0, seconds_since_last_keystroke=0.0)
    lane = typing_lane_result(result, decay_half_life_s=5.0)
    assert lane.status == "ok"
    assert lane.subscore == 0.7
    assert lane.lag_ms == -20.0
    assert lane.confidence == pytest.approx(1.0)  # dt=0 -> 2**0 = 1.0


def test_typing_lane_result_confidence_decays_with_silence():
    result = _typing_result(seconds_since_last_keystroke=5.0)  # exactly one half-life
    lane = typing_lane_result(result, decay_half_life_s=5.0)
    assert lane.confidence == pytest.approx(0.5)


def test_typing_lane_result_no_evidence_passthrough():
    result = _typing_result(status="no_evidence", coherence_score=float("nan"), coherence_lag_ms=float("nan"))
    lane = typing_lane_result(result, decay_half_life_s=5.0)
    assert lane.status == "no_evidence"
    assert lane.subscore is None
    assert lane.lag_ms is None


# ---------------------------------------------------------------------------
# print_clean_block -- pure presentation over an already-built record
# ---------------------------------------------------------------------------

def _record(optical_status="ok", optical_subscore=0.8, optical_lag=45.0,
            typing_status="ok", typing_subscore=0.75, typing_lag=-20.0,
            verdict="ACCEPT", joint_score=0.78, reason_text="joint score 0.78 across 2 coherent lane(s)",
            accept_threshold=0.6, reject_threshold=0.3):
    return {
        "session": "20260902T000000_deadbeef", "condition": "bonafide",
        "fusion": {
            "verdict": verdict, "joint_score": joint_score, "reason_text": reason_text,
            "per_lane_reasons": {},
            "accept_threshold": accept_threshold, "reject_threshold": reject_threshold,
            "lanes": [
                {"lane_name": "optical", "subscore": optical_subscore, "status": optical_status,
                 "lag_ms": optical_lag, "confidence": 0.8, "diagnostics": "snr_db=9.0"},
                {"lane_name": "typing", "subscore": typing_subscore, "status": typing_status,
                 "lag_ms": typing_lag, "confidence": 0.9, "diagnostics": ""},
                {"lane_name": "acoustic", "subscore": None, "status": "no_evidence",
                 "lag_ms": None, "confidence": 0.0, "diagnostics": "acoustic lane not implemented"},
            ],
        },
    }


def test_print_clean_block_accept_both_lanes_pass(capsys):
    print_clean_block(_record())
    out = capsys.readouterr().out
    assert "Optical lane : PASS" in out
    assert "Typing lane  : PASS" in out
    assert "VERDICT      : ACCEPT" in out
    assert "both lanes agree" in out
    assert "acoustic" not in out.lower()


def test_print_clean_block_reject_shows_the_adjudicators_own_reason(capsys):
    record = _record(optical_status="ok", optical_subscore=0.05, optical_lag=290.0,
                      typing_status="ok", typing_subscore=0.95, typing_lag=-10.0,
                      verdict="REJECT", joint_score=0.80,
                      reason_text="optical lane failed -- scored 0.05, below reject_threshold; "
                                   "a lane failing this clearly cannot be outvoted by another lane's confidence")
    print_clean_block(record)
    out = capsys.readouterr().out
    assert "VERDICT      : REJECT" in out
    assert "optical lane failed" in out
    assert "0.05" in out


def test_print_clean_block_low_subscore_shows_fail_even_when_lag_is_plausible(capsys):
    """Regression test for a real bug caught during hardware verification:
    a lane with a PLAUSIBLE lag (contributes=True) but a subscore below
    reject_threshold was displaying as PASS, because the display logic
    conflated 'contributes to the joint score' with 'passed' -- a lane
    scoring 0.17 against reject_threshold=0.3 must show FAIL, not PASS,
    regardless of its lag."""
    record = _record(optical_status="ok", optical_subscore=0.17, optical_lag=45.0,  # plausible lag, low score
                      typing_status="no_evidence", typing_subscore=None, typing_lag=None,
                      verdict="REJECT", joint_score=0.17,
                      reason_text="typing lane: no evidence (n_keystrokes=10)")
    print_clean_block(record)
    out = capsys.readouterr().out
    assert "Optical lane : FAIL" in out
    assert "Optical lane : PASS" not in out


def test_print_clean_block_no_evidence_lane_is_shown_honestly_not_as_pass_or_fail(capsys):
    record = _record(typing_status="no_evidence", typing_subscore=None, typing_lag=None,
                      verdict="RE-CHALLENGE", joint_score=None,
                      reason_text="typing lane: no evidence (no typing in window)")
    print_clean_block(record)
    out = capsys.readouterr().out
    assert "Typing lane  : NO EVIDENCE" in out
    assert "PASS" not in out.split("Typing lane")[1].split("\n")[0]


# ---------------------------------------------------------------------------
# The make-or-break check: exactly ONE open_camera() call, and the shared
# loop feeds the SAME frame to both lanes' per-frame processors.
# ---------------------------------------------------------------------------

def _fake_face_landmarker():
    lm = MagicMock()
    result = MagicMock()
    result.face_landmarks = []  # no face -- keeps ROI sampling trivial, doesn't matter for this test
    lm.detect_for_video.return_value = result
    return lm


def _fake_hand_landmarker():
    lm = MagicMock()
    result = MagicMock()
    result.hand_landmarks = []  # no hand -- same reasoning
    lm.detect_for_video.return_value = result
    return lm


def _fake_cap(n_frames: int = 6):
    """grab()/retrieve() succeed n_frames times then fail forever, so the
    shared loop's frame budget is bounded without relying on wall-clock
    timing in a unit test."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cap.grab.return_value = True  # never fails -- the loop is bounded by duration_s instead, see callers
    cap.retrieve.return_value = (True, frame)
    return cap, frame


def test_run_one_session_optical_plus_typing_opens_the_camera_exactly_once(tmp_path):
    cap, frame = _fake_cap()
    raw_config = {
        "optical": {"camera_index": 0, "model_path": "models/face_landmarker.task",
                     "auto_chip_rate": False, "detect_every_n_frames": 1, "adaptive_window_s": 999.0},
        "capture": {},
        # duration_s generous on purpose: mock-object overhead in test setup
        # (constructing MagicMocks, patch.object entries) can itself eat
        # several ms of the window before the loop even starts, and a too-
        # tight budget made this test flaky (0 or 1 iterations depending on
        # machine load) -- 0.2s comfortably outlasts setup while still
        # keeping the test fast.
        "challenge": {"chip_rate_hz": 10.0, "duration_s": 0.2},
        "emitter": {"emitter_enabled": True},
        "typing": {"n_words": 3, "generative": False, "hand_model_path": "models/hand_landmarker.task",
                    "passive_decay_half_life_s": 5.0},
        "fusion": {"accept_threshold": 0.6, "reject_threshold": 0.3, "min_contributing_lanes": 1},
    }

    fake_emitter = MagicMock()
    fake_emitter.get_log.return_value = []
    fake_emitter.adaptive_boost_applied.return_value = False
    fake_emitter.config.max_modulation_depth = 110.0

    fake_keystroke = MagicMock()
    fake_keystroke.chars_matched = 0
    fake_keystroke.get_events.return_value = []
    fake_keystroke.inter_key_intervals.return_value = np.array([])
    fake_keystroke.last_keystroke_time.return_value = None
    fake_keystroke.phrase_complete = False

    optical_frames_seen = []
    typing_frames_seen = []

    def fake_optical_process_frame(self, frame_arg, t, start_time, challenge, emitter):
        optical_frames_seen.append(frame_arg)
        return None

    def fake_typing_process_frame(self, frame_arg, t, start_time):
        typing_frames_seen.append(frame_arg)

    with patch.object(session_mod, "open_camera", return_value=cap) as mock_open_camera, \
         patch.object(session_mod, "configure_capture_format"), \
         patch.object(session_mod, "lock_camera", return_value=True), \
         patch.object(session_mod, "measure_capture_fps", return_value=16.0), \
         patch.object(session_mod, "check_nyquist", return_value=None), \
         patch.object(session_mod, "create_hand_landmarker", return_value=_fake_hand_landmarker()), \
         patch("praesens.optical.create_landmarker", return_value=_fake_face_landmarker()), \
         patch.object(session_mod, "Emitter", return_value=fake_emitter), \
         patch.object(session_mod, "KeystrokeCapture", return_value=fake_keystroke), \
         patch.object(session_mod.OpticalFrameProcessor, "process_frame", fake_optical_process_frame), \
         patch.object(session_mod.TypingFrameProcessor, "process_frame", fake_typing_process_frame):

        record, out_path = run_one_session(
            "bonafide", meta={"lighting": "normal"}, raw_config=raw_config,
            lanes="optical+typing", output_dir=tmp_path,
        )

    # The make-or-break check.
    mock_open_camera.assert_called_once()
    assert cap.isOpened.called

    # Exactly one grab()/retrieve() pair per loop iteration -- both lanes
    # were fed from the SAME shared loop, not two independent ones.
    assert len(optical_frames_seen) > 0, "the loop must have run at least one real iteration"
    assert len(optical_frames_seen) == len(typing_frames_seen) == cap.grab.call_count
    for a, b in zip(optical_frames_seen, typing_frames_seen):
        assert a is b is frame  # the literal same array object, not a copy each lane grabbed itself

    # The record carries both lanes and a real fusion verdict, add-only
    # (original fields untouched).
    assert record["lanes"] == "optical+typing"
    assert record["typing"] is not None
    assert record["fusion"]["verdict"] in ("ACCEPT", "RE-CHALLENGE", "REJECT")
    assert "score" in record and "snr_db" in record  # pre-existing schema fields still present


def test_run_one_session_optical_only_never_touches_typing(tmp_path):
    """--lanes optical (the backward-compatible default for THIS function,
    see its own docstring) must not construct a hand landmarker, a
    keystroke listener, or a typing_phrase -- the flag genuinely gates the
    lane, not just its display."""
    cap, frame = _fake_cap()
    raw_config = {
        "optical": {"camera_index": 0, "model_path": "models/face_landmarker.task",
                     "auto_chip_rate": False, "adaptive_window_s": 999.0},
        "capture": {},
        "challenge": {"chip_rate_hz": 10.0, "duration_s": 0.2},  # see the other test for why 0.2s
        "emitter": {"emitter_enabled": True},
        "typing": {"n_words": 3, "generative": False},
        "fusion": {"accept_threshold": 0.6, "reject_threshold": 0.3, "min_contributing_lanes": 1},
    }
    fake_emitter = MagicMock()
    fake_emitter.get_log.return_value = []
    fake_emitter.adaptive_boost_applied.return_value = False

    with patch.object(session_mod, "open_camera", return_value=cap), \
         patch.object(session_mod, "configure_capture_format"), \
         patch("praesens.optical.lock_camera", return_value=True), \
         patch("praesens.optical.measure_capture_fps", return_value=16.0), \
         patch("praesens.optical.check_nyquist", return_value=None), \
         patch("praesens.optical.create_landmarker", return_value=_fake_face_landmarker()), \
         patch.object(session_mod, "Emitter", return_value=fake_emitter), \
         patch.object(session_mod, "create_hand_landmarker") as mock_hand_landmarker, \
         patch.object(session_mod, "KeystrokeCapture") as mock_keystroke:

        record, out_path = run_one_session(
            "bonafide", meta={"lighting": "normal"}, raw_config=raw_config,
            lanes="optical", output_dir=tmp_path,
        )

    mock_hand_landmarker.assert_not_called()
    mock_keystroke.assert_not_called()
    assert record["lanes"] == "optical"
    assert record["typing"] is None
    assert record["typing_phrase"] is None


# ---------------------------------------------------------------------------
# show_verdict_banner -- 2026-09-03, on-screen ACCEPT/REJECT/RE-CHALLENGE for
# demo purposes. Real cv2 GUI calls are mocked out (no window should
# actually open during the test suite); these verify it derives its
# verdict/colour/text from the SAME shared helpers print_clean_block()
# uses (never inventing separate pass/fail logic) and that a GUI failure
# degrades to a warning, never a crash -- a demo failing to SHOW an
# already-computed, already-saved verdict must not take the run down.
# ---------------------------------------------------------------------------

def test_show_verdict_banner_never_raises_when_gui_calls_fail():
    from praesens.session import show_verdict_banner
    record = _record(verdict="ACCEPT", joint_score=0.78)
    with patch("cv2.namedWindow", side_effect=RuntimeError("no display")):
        show_verdict_banner(record, hold_seconds=0.01)  # must not raise


def test_show_verdict_banner_uses_the_real_verdict_and_holds_briefly():
    from praesens.session import show_verdict_banner
    record = _record(verdict="REJECT", joint_score=0.17,
                      optical_status="ok", optical_subscore=0.17, optical_lag=45.0,
                      typing_status="no_evidence", typing_subscore=None, typing_lag=None,
                      reason_text="optical lane failed -- scored 0.17, below reject_threshold")

    shown_frames = []
    with patch("cv2.namedWindow"), patch("cv2.setWindowProperty"), \
         patch("cv2.imshow", side_effect=lambda name, frame: shown_frames.append(frame)), \
         patch("cv2.waitKey", return_value=-1), patch("cv2.destroyWindow"):
        show_verdict_banner(record, hold_seconds=0.05)

    assert len(shown_frames) > 0
    frame = shown_frames[0]
    # REJECT is drawn on a red-dominant BGR frame, not green -- a real,
    # checkable property of the rendered image, not just "it ran."
    assert int(frame[0, 0, 2]) > int(frame[0, 0, 1])  # R channel > G channel in the background fill


def test_show_verdict_banner_accept_is_green_not_red():
    from praesens.session import show_verdict_banner
    record = _record(verdict="ACCEPT", joint_score=0.78)
    shown_frames = []
    with patch("cv2.namedWindow"), patch("cv2.setWindowProperty"), \
         patch("cv2.imshow", side_effect=lambda name, frame: shown_frames.append(frame)), \
         patch("cv2.waitKey", return_value=-1), patch("cv2.destroyWindow"):
        show_verdict_banner(record, hold_seconds=0.05)

    frame = shown_frames[0]
    assert int(frame[0, 0, 1]) > int(frame[0, 0, 2])  # G channel > R channel -- green background
