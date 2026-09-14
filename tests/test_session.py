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
from praesens.typing import TypingConfig


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
# LaneResult wrapping -- typing (2026-09-03: keystroke-timing only, no
# camera/hand dependency -- see praesens.typing.keystroke_normality_score
# and typing_lane_result's own docstring for why the hand-coherence-based
# version this replaced is gone from this path)
# ---------------------------------------------------------------------------

def _typing_config(**overrides) -> TypingConfig:
    base = dict(min_keystrokes=3, min_inter_key_s=0.03, max_inter_key_s=3.0,
                implausible_timing_penalty=0.3, pass_match_fraction=0.7)
    base.update(overrides)
    return TypingConfig(**base)


def _keystroke_events(n=5, dt=0.2, matched=True):
    return [{"t_down": i * dt, "t_up": i * dt + 0.05, "matched_expected": matched} for i in range(n)]


def test_typing_lane_result_ok_case_scores_from_timing_alone():
    events = _keystroke_events(n=5, dt=0.2, matched=True)
    lane = typing_lane_result(events, expected_phrase="abcde", tconfig=_typing_config())
    assert lane.status == "ok"
    assert lane.subscore == pytest.approx(1.0)  # all 5 matched, all 5 expected
    assert lane.lag_ms is None  # no camera/hand signal left to measure a lag against
    assert lane.confidence == pytest.approx(1.0)


def test_typing_lane_result_below_min_keystrokes_is_no_evidence():
    events = _keystroke_events(n=2, dt=0.2, matched=True)
    lane = typing_lane_result(events, expected_phrase="ab", tconfig=_typing_config(min_keystrokes=3))
    assert lane.status == "no_evidence"
    assert lane.subscore is None
    assert lane.confidence == 0.0


def test_typing_lane_result_wrong_phrase_is_ok_status_with_low_subscore():
    """Typing the WRONG thing is a real, scored attempt (status='ok'), not
    no_evidence -- a low subscore is how it fails, exactly matching
    keystroke_normality_score's own contract."""
    events = _keystroke_events(n=5, dt=0.2, matched=False)
    lane = typing_lane_result(events, expected_phrase="abcde", tconfig=_typing_config())
    assert lane.status == "ok"
    assert lane.subscore == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# print_clean_block -- pure presentation over an already-built record
# ---------------------------------------------------------------------------

def _record(optical_status="ok", optical_subscore=0.8, optical_lag=45.0,
            typing_status="ok", typing_subscore=0.75, typing_lag=-20.0,
            verdict="ACCEPT", joint_score=0.78,
            reason_text="both lanes agree: optical 0.80, typing 0.75",
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
                 "lag_ms": None, "confidence": 0.0, "diagnostics": "acoustic lane not implemented",
                 "is_stub": True},
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
    """2026-09-03: the typing lane no longer processes any frame at all
    (see praesens.typing.keystroke_normality_score) -- what this test now
    verifies is narrower but still the make-or-break property: exactly
    ONE cv2.VideoCapture opened, and the shared loop's grab()/retrieve()
    pairs are consumed by exactly one frame-based consumer (optical). The
    keystroke listener runs off its own independent thread for the same
    window, started/stopped around the loop, never handed a frame."""
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
        "typing": {"n_words": 3, "generative": False, "min_keystrokes": 3,
                    "min_inter_key_s": 0.03, "max_inter_key_s": 3.0,
                    "implausible_timing_penalty": 0.3, "pass_match_fraction": 0.7},
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

    def fake_optical_process_frame(self, frame_arg, t, start_time, challenge, emitter):
        optical_frames_seen.append(frame_arg)
        return None

    with patch.object(session_mod, "open_camera", return_value=cap) as mock_open_camera, \
         patch.object(session_mod, "configure_capture_format"), \
         patch.object(session_mod, "lock_camera", return_value=True), \
         patch.object(session_mod, "measure_capture_fps", return_value=16.0), \
         patch.object(session_mod, "check_nyquist", return_value=None), \
         patch("praesens.optical.create_landmarker", return_value=_fake_face_landmarker()), \
         patch.object(session_mod, "Emitter", return_value=fake_emitter), \
         patch.object(session_mod, "KeystrokeCapture", return_value=fake_keystroke), \
         patch.object(session_mod.OpticalFrameProcessor, "process_frame", fake_optical_process_frame):

        record, out_path = run_one_session(
            "bonafide", meta={"lighting": "normal"}, raw_config=raw_config,
            lanes="optical+typing", output_dir=tmp_path,
        )

    # The make-or-break check: exactly one camera open, exactly one
    # frame-based consumer of the shared grab()/retrieve() loop.
    mock_open_camera.assert_called_once()
    assert cap.isOpened.called
    assert len(optical_frames_seen) > 0, "the loop must have run at least one real iteration"
    assert len(optical_frames_seen) == cap.grab.call_count
    for f in optical_frames_seen:
        assert f is frame  # the literal same array object, not a copy

    # Typing's keystroke listener runs independently, for the same window,
    # never touching a frame.
    fake_keystroke.start.assert_called_once()
    fake_keystroke.stop.assert_called_once()

    # The record carries both lanes and a real fusion verdict, add-only
    # (original fields untouched).
    assert record["lanes"] == "optical+typing"
    assert record["typing"] is not None
    assert record["fusion"]["verdict"] in ("ACCEPT", "RE-CHALLENGE", "REJECT")
    assert "score" in record and "snr_db" in record  # pre-existing schema fields still present

    # acoustic wasn't requested -- must fall back to the stub, and the
    # stub's is_stub flag must survive into the saved record (this is
    # what keeps it correctly hidden in print_clean_block, see
    # test_print_clean_block_accept_both_lanes_pass).
    assert record["acoustic"] is None
    acoustic_entry = next(l for l in record["fusion"]["lanes"] if l["lane_name"] == "acoustic")
    assert acoustic_entry["is_stub"] is True


# ---------------------------------------------------------------------------
# Acoustic lane wiring -- --lanes optical+typing+acoustic. run_acoustic_session
# itself is mocked (it does real sounddevice I/O, out of scope for a unit
# test); what's under test here is session.py's OWN wiring: the exact bug
# fixed after code review (typing_enabled == "optical+typing" excluded the
# 3-lane string, silently dropping typing whenever acoustic was requested
# alongside it), the acoustic thread reaching the shared start_time, and a
# crashed acoustic thread degrading to the stub rather than the session.
# ---------------------------------------------------------------------------

def _base_three_lane_config(extra_fusion=None):
    cfg = {
        "optical": {"camera_index": 0, "model_path": "models/face_landmarker.task",
                     "auto_chip_rate": False, "detect_every_n_frames": 1, "adaptive_window_s": 999.0},
        "capture": {},
        "challenge": {"chip_rate_hz": 10.0, "duration_s": 0.2},
        "emitter": {"emitter_enabled": True},
        "typing": {"n_words": 3, "generative": False, "min_keystrokes": 3,
                    "min_inter_key_s": 0.03, "max_inter_key_s": 3.0,
                    "implausible_timing_penalty": 0.3, "pass_match_fraction": 0.7},
        "acoustic": {"sample_rate_hz": 44100, "carrier_hz": 18000.0},
        "fusion": {"accept_threshold": 0.6, "reject_threshold": 0.3, "min_contributing_lanes": 1},
    }
    if extra_fusion:
        cfg["fusion"].update(extra_fusion)
    return cfg


def _patched_three_lane_session(cap, fake_acoustic_session):
    """Shared patch context for the tests below -- same mocks as the
    2-lane test above, plus run_acoustic_session swapped for a fake that
    never touches real audio hardware."""
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

    return patch.object(session_mod, "open_camera", return_value=cap), \
        patch.object(session_mod, "configure_capture_format"), \
        patch.object(session_mod, "lock_camera", return_value=True), \
        patch.object(session_mod, "measure_capture_fps", return_value=16.0), \
        patch.object(session_mod, "check_nyquist", return_value=None), \
        patch("praesens.optical.create_landmarker", return_value=_fake_face_landmarker()), \
        patch.object(session_mod, "Emitter", return_value=fake_emitter), \
        patch.object(session_mod, "KeystrokeCapture", return_value=fake_keystroke), \
        patch.object(session_mod, "run_acoustic_session", side_effect=fake_acoustic_session)


def test_three_lane_mode_still_runs_typing_not_just_optical_and_acoustic(tmp_path):
    """Regression test for the exact bug found in code review: lanes ==
    "optical+typing+acoustic" != "optical+typing", so the ORIGINAL
    typing_enabled check silently disabled typing whenever acoustic was
    also requested. This asserts typing actually ran (its keystroke
    listener started and stopped around the shared loop -- since
    2026-09-03 typing is keystroke-timing only and never sees a frame),
    not just that the session completed."""
    from praesens.acoustic import AcousticResult
    cap, frame = _fake_cap()

    def fake_acoustic_session(challenge, aconfig):
        return AcousticResult(status="ok", score=0.9, lag_ms=8.0, snr_db=20.0, diagnostics="snr_db=20.00")

    patches = _patched_three_lane_session(cap, fake_acoustic_session)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], \
         patches[7] as mock_keystroke_cls, patches[8], \
         patch.object(session_mod.OpticalFrameProcessor, "process_frame", lambda *a, **k: None):

        record, out_path = run_one_session(
            "bonafide", meta={"lighting": "normal"}, raw_config=_base_three_lane_config(),
            lanes="optical+typing+acoustic", output_dir=tmp_path,
        )

    mock_keystroke_cls.return_value.start.assert_called_once()
    mock_keystroke_cls.return_value.stop.assert_called_once()
    assert record["typing"] is not None


def test_three_lane_mode_wires_a_real_acoustic_result_into_record_and_fusion(tmp_path):
    from praesens.acoustic import AcousticResult
    cap, frame = _fake_cap()
    seen_challenges = []

    def fake_acoustic_session(challenge, aconfig):
        seen_challenges.append(challenge)
        return AcousticResult(status="ok", score=0.93, lag_ms=7.5, snr_db=22.0, diagnostics="snr_db=22.00")

    patches = _patched_three_lane_session(cap, fake_acoustic_session)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], \
         patches[7], patches[8], \
         patch.object(session_mod.OpticalFrameProcessor, "process_frame", lambda *a, **k: None):

        record, out_path = run_one_session(
            "bonafide", meta={"lighting": "normal"}, raw_config=_base_three_lane_config(),
            lanes="optical+typing+acoustic", output_dir=tmp_path,
        )

    # Same challenge object optical/typing already answer -- see
    # praesens/acoustic.py's module docstring on this being the point of
    # reusing Challenge's own chip sequence rather than a second seed.
    assert len(seen_challenges) == 1

    assert record["acoustic"] == {
        "status": "ok", "score": 0.93, "lag_ms": 7.5, "snr_db": 22.0, "diagnostics": "snr_db=22.00",
    }
    acoustic_entry = next(l for l in record["fusion"]["lanes"] if l["lane_name"] == "acoustic")
    assert acoustic_entry["is_stub"] is False
    assert acoustic_entry["status"] == "ok"
    assert acoustic_entry["subscore"] == 0.93
    assert acoustic_entry["lag_ms"] == 7.5
    assert acoustic_entry["confidence"] == pytest.approx(np.clip(22.0 / 20.0, 0.05, 1.0))  # acoustic_lane_result's own formula

    # acoustic now gates the verdict as a secondary lane: it passed here, but
    # the mocked keystroke listener recorded nothing, so typing is what asks
    # for a retry -- and the reason must say so, not blame acoustic.
    assert record["fusion"]["verdict"] in ("RE-CHALLENGE", "REJECT")
    assert record["fusion"]["acoustic_pass_threshold"] == 0.3
    assert "acoustic" not in record["fusion"]["reason_text"]


def test_three_lane_mode_acoustic_thread_crash_degrades_to_stub_not_a_session_failure(tmp_path):
    """An exception inside the acoustic worker thread (e.g. a real
    sounddevice/PortAudio error on unfamiliar hardware) must not take the
    whole session down -- optical/typing results are already valid and
    must still be saved. See session.py's acoustic_box error handling."""
    cap, frame = _fake_cap()

    def fake_acoustic_session(challenge, aconfig):
        raise RuntimeError("PortAudio device unavailable")

    patches = _patched_three_lane_session(cap, fake_acoustic_session)
    with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], \
         patches[7], patches[8], \
         patch.object(session_mod.OpticalFrameProcessor, "process_frame", lambda *a, **k: None):

        record, out_path = run_one_session(
            "bonafide", meta={"lighting": "normal"}, raw_config=_base_three_lane_config(),
            lanes="optical+typing+acoustic", output_dir=tmp_path,
        )

    assert record["acoustic"] is None
    acoustic_entry = next(l for l in record["fusion"]["lanes"] if l["lane_name"] == "acoustic")
    assert acoustic_entry["is_stub"] is True
    assert record["fusion"]["verdict"] in ("ACCEPT", "RE-CHALLENGE", "REJECT")  # session still produced a verdict


def test_print_clean_block_shows_acoustic_lane_when_it_really_ran(capsys):
    """Mirror of test_print_clean_block_accept_both_lanes_pass, which
    checks acoustic stays HIDDEN when it's a stub -- this checks the
    opposite case, that it's shown once is_stub is False."""
    record = _record()
    record["fusion"]["lanes"][2] = {
        "lane_name": "acoustic", "subscore": 0.9, "status": "ok",
        "lag_ms": 8.0, "confidence": 0.9, "diagnostics": "snr_db=20.00", "is_stub": False,
    }
    print_clean_block(record)
    out = capsys.readouterr().out
    assert "Acoustic lane : PASS" in out


def test_run_one_session_optical_only_never_touches_typing(tmp_path):
    """--lanes optical (the backward-compatible default for THIS function,
    see its own docstring) must not construct a keystroke listener or a
    typing_phrase -- the flag genuinely gates the lane, not just its
    display."""
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
         patch.object(session_mod, "KeystrokeCapture") as mock_keystroke:

        record, out_path = run_one_session(
            "bonafide", meta={"lighting": "normal"}, raw_config=raw_config,
            lanes="optical", output_dir=tmp_path,
        )

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


def _gate_record(optical, typing, acoustic):
    """A record whose verdict comes from the REAL gate
    (fusion.adjudicate_two_lane), not a hand-written verdict string -- so
    the banner tests below check lane outcomes -> colour end to end."""
    from praesens.fusion import adjudicate_two_lane
    joint = adjudicate_two_lane(optical, typing, 0.3, 0.7, acoustic=acoustic, acoustic_pass_threshold=0.3)
    return {
        "session": "20260914T000000_deadbeef", "condition": "bonafide",
        "fusion": {
            "verdict": joint.verdict, "joint_score": joint.joint_score, "reason_text": joint.reason_text,
            "per_lane_reasons": joint.per_lane_reasons, "accept_threshold": 0.6, "reject_threshold": 0.3,
            "optical_pass_threshold": 0.3, "typing_pass_threshold": 0.7, "acoustic_pass_threshold": 0.3,
            "lanes": [{"lane_name": l.lane_name, "subscore": l.subscore, "status": l.status, "lag_ms": l.lag_ms,
                       "confidence": l.confidence, "diagnostics": l.diagnostics, "is_stub": l.is_stub}
                      for l in joint.lane_results],
        },
    }


_OPTICAL_PASS = LaneResult(lane_name="optical", subscore=0.75, status="ok", lag_ms=25.0, confidence=0.47)
_OPTICAL_FAIL = LaneResult(lane_name="optical", subscore=0.05, status="ok", lag_ms=25.0, confidence=0.2)
_TYPING_PASS = LaneResult(lane_name="typing", subscore=1.0, status="ok", lag_ms=None, confidence=1.0,
                          diagnostics="matched 27/27, mean_inter_key=0.146s")
_TYPING_FAIL = LaneResult(lane_name="typing", subscore=None, status="no_evidence", lag_ms=None, confidence=0.0,
                          diagnostics="n_keystrokes=0 (need >= 3)")
_ACOUSTIC_PASS = LaneResult(lane_name="acoustic", subscore=0.93, status="ok", lag_ms=332.0, confidence=0.54,
                            diagnostics="snr_db=10.84")
_ACOUSTIC_FAIL = LaneResult(lane_name="acoustic", subscore=None, status="insufficient_signal", lag_ms=None,
                            confidence=0.0,
                            diagnostics="snr_db=0.32, tone_db=-0.1, peak=0.19, off_carrier_floor=0.18, "
                                        "decoy_floor=0.16, out='Speakers (Realtek(R) Audio)', "
                                        "in='Microphone Array (Realtek(R) Au', tone not heard -- check the "
                                        "output device isn't muted/at 0% volume and isn't headphones")

_GREEN, _YELLOW, _RED = (60, 180, 60), (0, 215, 255), (40, 40, 200)  # BGR


@pytest.mark.parametrize("optical, typing, acoustic, verdict, background", [
    (_OPTICAL_PASS, _TYPING_PASS, _ACOUSTIC_PASS, "ACCEPT", _GREEN),
    (_OPTICAL_FAIL, _TYPING_PASS, _ACOUSTIC_PASS, "REJECT", _RED),
    (_OPTICAL_FAIL, _TYPING_FAIL, _ACOUSTIC_FAIL, "REJECT", _RED),
    (_OPTICAL_PASS, _TYPING_FAIL, _ACOUSTIC_PASS, "RE-CHALLENGE", _YELLOW),
    (_OPTICAL_PASS, _TYPING_PASS, _ACOUSTIC_FAIL, "RE-CHALLENGE", _YELLOW),
], ids=["all_three_pass_green", "optical_fails_red", "everything_fails_red",
        "typing_fails_yellow", "acoustic_fails_yellow"])
def test_three_lane_banner_colour_follows_the_gate(optical, typing, acoustic, verdict, background):
    from praesens.session import render_verdict_banner
    record = _gate_record(optical, typing, acoustic)
    assert record["fusion"]["verdict"] == verdict

    frame = render_verdict_banner(record, 1280, 720)
    assert tuple(int(c) for c in frame[0, 0]) == background
    rows, _ = session_mod._lane_display_rows(record)
    assert [label.strip() for label, _, _ in rows] == ["Optical lane", "Typing lane", "Acoustic lane"]


def test_yellow_banner_uses_dark_text_and_long_reasons_stay_on_screen():
    from praesens.session import render_verdict_banner
    record = _gate_record(_OPTICAL_PASS, _TYPING_FAIL, _ACOUSTIC_FAIL)  # longest reason: both secondaries failing
    frame = render_verdict_banner(record, 1280, 720)

    assert np.any(np.all(frame == (20, 20, 20), axis=2))       # dark text is drawn...
    assert not np.any(np.all(frame == (255, 255, 255), axis=2))  # ...and no white-on-yellow
    margin = int(1280 * 0.03)
    assert np.all(frame[:, :margin] == _YELLOW) and np.all(frame[:, -margin:] == _YELLOW)  # nothing runs off the edges


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