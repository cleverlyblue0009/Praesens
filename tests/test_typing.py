"""Milestone 10 acceptance tests: the keystroke-hand coherence detector
must distinguish genuine, timing-correlated typing from unrelated hand
motion, PASSIVE-mode confidence must decay and recover correctly, and --
the one non-negotiable check -- KeystrokeCapture must never store a
pressed character anywhere, structurally verified, not just claimed in a
docstring.
"""
import numpy as np
import pytest
from pynput.keyboard import KeyCode, Key

from praesens.typing import (
    KeystrokeCapture, keystroke_hand_coherence, keystroke_impulse_density,
    passive_confidence, compute_typing_status, keystroke_normality_score, TypingConfig,
)


def test_keystroke_capture_never_stores_a_character():
    """Structural check, not a docstring claim: press a very identifiable
    phrase and verify NONE of the stored event dicts contain the pressed
    character in any form, and the only keys present are the three the
    module docstring promises."""
    capture = KeystrokeCapture(expected_phrase="secret")
    for ch in "secret":
        capture._on_press(KeyCode.from_char(ch))
        capture._on_release(KeyCode.from_char(ch))

    events = capture.get_events()
    assert len(events) == 6
    for e in events:
        assert set(e.keys()) == {"t_down", "t_up", "matched_expected"}, f"unexpected keys: {e.keys()}"
        for v in e.values():
            assert not isinstance(v, str), f"a string value leaked into a stored event: {v!r}"
    assert all(e["matched_expected"] is True for e in events)


def test_keystroke_capture_flags_mismatches_without_recording_what_was_typed():
    capture = KeystrokeCapture(expected_phrase="cat")
    capture._on_press(KeyCode.from_char("c"))  # matches 'c', cursor -> 1
    capture._on_press(KeyCode.from_char("x"))  # expected 'a', got 'x' -- mismatch, cursor stays
    capture._on_press(KeyCode.from_char("t"))  # expected 'a', got 't' -- still mismatch
    events = capture.get_events()
    assert [e["matched_expected"] for e in events] == [True, False, False]
    for e in events:
        assert not isinstance(e["matched_expected"], str)


def test_chars_matched_is_a_count_not_a_character():
    """chars_matched is the one extra thing an on-screen prompt is allowed
    to read (2026-09-02, for the concurrent session's live phrase overlay)
    -- must be an int count that advances only on a correct match, never
    anything that leaks the pressed character."""
    capture = KeystrokeCapture(expected_phrase="cat")
    assert capture.chars_matched == 0
    capture._on_press(KeyCode.from_char("c"))  # matches -> advances
    assert capture.chars_matched == 1
    capture._on_press(KeyCode.from_char("x"))  # mismatch -> does not advance
    assert capture.chars_matched == 1
    capture._on_press(KeyCode.from_char("a"))  # matches -> advances
    assert capture.chars_matched == 2
    assert isinstance(capture.chars_matched, int)


def test_passive_mode_never_sets_matched_expected():
    capture = KeystrokeCapture(expected_phrase=None)
    for ch in "hello":
        capture._on_press(KeyCode.from_char(ch))
    events = capture.get_events()
    assert all(e["matched_expected"] is None for e in events)


def test_space_key_handled_without_a_char_attribute():
    capture = KeystrokeCapture(expected_phrase="a b")
    capture._on_press(KeyCode.from_char("a"))
    capture._on_press(Key.space)
    capture._on_press(KeyCode.from_char("b"))
    events = capture.get_events()
    assert [e["matched_expected"] for e in events] == [True, True, True]


def test_release_pairs_with_press_fifo():
    capture = KeystrokeCapture(expected_phrase=None)
    capture._on_press(KeyCode.from_char("a"))
    capture._on_release(KeyCode.from_char("a"))
    events = capture.get_events()
    assert events[0]["t_up"] is not None and events[0]["t_up"] >= events[0]["t_down"]


def test_keystroke_hand_coherence_detects_genuine_timing_correlation():
    rng = np.random.default_rng(0)
    keystroke_times = np.arange(0.5, 20.0, 0.8)
    events = [{"t_down": t, "t_up": t + 0.05, "matched_expected": None} for t in keystroke_times]

    hand_ts = np.arange(0, 20.0, 1 / 30.0)
    TRUE_LAG_S = -0.05  # hand motion slightly leads the keystroke registration
    density = keystroke_impulse_density(events, hand_ts - TRUE_LAG_S, sigma_s=0.15)
    hand_activity = 1.0 * density + rng.normal(0, 0.05, len(hand_ts))

    score, lag_ms = keystroke_hand_coherence(events, hand_ts, hand_activity,
                                              sigma_s=0.15, lag_max_ms=300, lag_step_ms=10)
    assert score > 0.7, f"expected strong coherence for genuine timing, got {score:.3f}"
    assert abs(lag_ms - (TRUE_LAG_S * 1000)) <= 15, f"recovered lag {lag_ms}ms far from true {TRUE_LAG_S*1000}ms"


def test_keystroke_hand_coherence_collapses_for_unrelated_hand_motion():
    rng = np.random.default_rng(1)
    keystroke_times = np.arange(0.5, 20.0, 0.8)
    events = [{"t_down": t, "t_up": t + 0.05, "matched_expected": None} for t in keystroke_times]

    hand_ts = np.arange(0, 20.0, 1 / 30.0)
    hand_activity = 0.3 + 0.1 * np.sin(2 * np.pi * 0.05 * hand_ts) + rng.normal(0, 0.05, len(hand_ts))

    score, lag_ms = keystroke_hand_coherence(events, hand_ts, hand_activity,
                                              sigma_s=0.15, lag_max_ms=300, lag_step_ms=10)
    assert score < 0.4, f"expected low coherence for unrelated hand motion, got {score:.3f}"


def test_keystroke_hand_coherence_reports_nan_for_insufficient_hand_data():
    events = [{"t_down": 1.0, "t_up": 1.05, "matched_expected": None}]
    hand_ts = np.array([0.5, 0.6])
    hand_activity = np.array([np.nan, np.nan])
    score, lag_ms = keystroke_hand_coherence(events, hand_ts, hand_activity, sigma_s=0.15,
                                              lag_max_ms=300, lag_step_ms=10, min_valid_samples=5)
    assert np.isnan(score) and np.isnan(lag_ms), "expected (nan, nan), which callers map to no_evidence"


def test_passive_status_no_evidence_when_operator_stops_typing_for_10s():
    """The acceptance criterion, verified deterministically: a PASSIVE
    session where typing happened early but stopped >=10s before the
    session ended must report no_evidence by the end -- not "ok" just
    because SOME typing occurred somewhere in the window."""
    status_kept_typing = compute_typing_status(
        "passive", silence_s=2.0, n_hand_valid=20, min_valid_hand_samples=5,
        passive_no_evidence_after_s=10.0, n_events=15,
    )
    assert status_kept_typing == "ok"

    status_stopped = compute_typing_status(
        "passive", silence_s=12.0, n_hand_valid=20, min_valid_hand_samples=5,
        passive_no_evidence_after_s=10.0, n_events=15,
    )
    assert status_stopped == "no_evidence"

    status_never_typed = compute_typing_status(
        "passive", silence_s=float("inf"), n_hand_valid=20, min_valid_hand_samples=5,
        passive_no_evidence_after_s=10.0, n_events=0,
    )
    assert status_never_typed == "no_evidence"


def test_active_status_no_evidence_when_zero_keystrokes():
    status = compute_typing_status("active", silence_s=float("inf"), n_hand_valid=20,
                                    min_valid_hand_samples=5, passive_no_evidence_after_s=10.0, n_events=0)
    assert status == "no_evidence"


def test_status_no_evidence_when_no_hand_data_regardless_of_typing():
    status = compute_typing_status("active", silence_s=0.0, n_hand_valid=1,
                                    min_valid_hand_samples=5, passive_no_evidence_after_s=10.0, n_events=20)
    assert status == "no_evidence"


def test_passive_confidence_decays_and_absent_gives_zero():
    assert passive_confidence(None, now=10.0, decay_half_life_s=5.0) == 0.0
    c0 = passive_confidence(0.0, now=0.0, decay_half_life_s=5.0)
    c_half = passive_confidence(0.0, now=5.0, decay_half_life_s=5.0)
    c_decayed = passive_confidence(0.0, now=50.0, decay_half_life_s=5.0)
    assert c0 == pytest.approx(1.0)
    assert c_half == pytest.approx(0.5, abs=0.01)
    assert c_decayed < 0.01


# ---------------------------------------------------------------------------
# keystroke_normality_score -- keystroke-TIMING-only scoring (2026-09-03),
# no camera/hand dependency. See the function's own docstring and
# praesens/session.py's module docstring for why this exists: a webcam
# framed on the face for the optical lane cannot also see the hands during
# normal typing, so the earlier hand-coherence-based typing lane read as
# no_evidence in practice regardless of whether genuine typing happened.
# ---------------------------------------------------------------------------

def _tconfig(**overrides) -> TypingConfig:
    base = dict(min_keystrokes=3, min_inter_key_s=0.03, max_inter_key_s=3.0,
                implausible_timing_penalty=0.3, pass_match_fraction=0.7)
    base.update(overrides)
    return TypingConfig(**base)


def _events(n=5, dt=0.2, matched=True):
    return [{"t_down": i * dt, "t_up": i * dt + 0.05, "matched_expected": matched} for i in range(n)]


def test_keystroke_normality_score_genuine_typing_scores_high():
    events = _events(n=20, dt=0.2, matched=True)
    status, subscore, diagnostics = keystroke_normality_score(events, "x" * 20, _tconfig())
    assert status == "ok"
    assert subscore == pytest.approx(1.0)
    assert "20/20" in diagnostics


def test_keystroke_normality_score_too_few_keystrokes_is_no_evidence():
    events = _events(n=2, dt=0.2, matched=True)
    status, subscore, diagnostics = keystroke_normality_score(events, "ab", _tconfig(min_keystrokes=3))
    assert status == "no_evidence"
    assert subscore is None
    assert "n_keystrokes=2" in diagnostics


def test_keystroke_normality_score_penalizes_implausibly_fast_robotic_timing():
    """A naive scripted/replayed keystroke injection types with unnaturally
    uniform, fast timing -- caught even when every character matches."""
    events = _events(n=20, dt=0.005, matched=True)  # 5ms/key, well under min_inter_key_s
    status, subscore, diagnostics = keystroke_normality_score(events, "x" * 20, _tconfig())
    assert status == "ok"
    assert subscore == pytest.approx(1.0 * 0.3)  # match_fraction * implausible_timing_penalty
    assert "implausible" in diagnostics or "outside plausible" in diagnostics


def test_keystroke_normality_score_wrong_phrase_scores_low_but_still_ok():
    events = _events(n=20, dt=0.2, matched=False)
    status, subscore, diagnostics = keystroke_normality_score(events, "x" * 20, _tconfig())
    assert status == "ok"
    assert subscore == pytest.approx(0.0)
    assert "0/20" in diagnostics


def test_keystroke_normality_score_partial_match_scales_linearly():
    events = _events(n=10, dt=0.2, matched=True) + _events(n=10, dt=0.2, matched=False)
    status, subscore, diagnostics = keystroke_normality_score(events, "x" * 20, _tconfig())
    assert status == "ok"
    assert subscore == pytest.approx(0.5)  # 10/20 matched, plausible timing


def test_keystroke_normality_score_never_sees_a_character():
    """Structural privacy check: the function's only inputs are timing +
    a boolean match flag -- confirm no event dict passed in needs (or is
    given) a pressed-character field for this to work at all."""
    events = _events(n=5, dt=0.2, matched=True)
    assert all(set(e.keys()) == {"t_down", "t_up", "matched_expected"} for e in events)
    status, subscore, diagnostics = keystroke_normality_score(events, "abcde", _tconfig())
    assert status == "ok"
