"""Milestone 11 acceptance tests: joint coherence must require BOTH a
strong subscore AND a plausible lag from each contributing lane, not
average in whatever numbers arrive; verdicts must carry an explanation
that names the actually-informative lane, not the permanently-stubbed
acoustic lane; and hysteresis must actually prevent flicker at a
threshold boundary.
"""
import pytest

from praesens.fusion import (
    LaneResult, Adjudicator, AdjudicatorConfig, acoustic_lane_stub,
    compute_joint_score, lane_contributes, LANE_LAG_BOUNDS_MS,
)


def _ok(name, subscore, lag_ms, confidence=1.0):
    return LaneResult(lane_name=name, subscore=subscore, status="ok", lag_ms=lag_ms, confidence=confidence)


def _no_evidence(name, diagnostics=""):
    return LaneResult(lane_name=name, subscore=None, status="no_evidence", lag_ms=None,
                       confidence=0.0, diagnostics=diagnostics)


def test_lane_result_rejects_subscore_with_non_ok_status():
    with pytest.raises(ValueError):
        LaneResult(lane_name="optical", subscore=0.8, status="no_evidence", lag_ms=None, confidence=0.0)


def test_acoustic_stub_is_always_no_evidence_and_does_no_io():
    r = acoustic_lane_stub()
    assert r.status == "no_evidence"
    assert r.subscore is None
    assert r.lane_name == "acoustic"


def test_high_score_at_implausible_lag_does_not_contribute():
    """The core claim of this milestone: a lane agreeing on THAT (subscore)
    without agreeing on WHEN (lag within its window) contributes nothing,
    however high the subscore looks."""
    optical_bad_lag = _ok("optical", subscore=0.95, lag_ms=500.0)  # bound is (0, 300)
    assert not lane_contributes(optical_bad_lag, LANE_LAG_BOUNDS_MS)

    joint_score, contributions = compute_joint_score([optical_bad_lag], LANE_LAG_BOUNDS_MS)
    assert contributions["optical"]["contributes"] is False
    assert contributions["optical"]["weight"] == 0.0
    import math
    assert math.isnan(joint_score), "a single non-contributing lane must leave the joint score undefined, not 0"


def test_typing_lag_window_is_bounded_both_sides():
    typing_leads = _ok("typing", subscore=0.8, lag_ms=-150.0)
    typing_lags = _ok("typing", subscore=0.8, lag_ms=150.0)
    typing_too_early = _ok("typing", subscore=0.8, lag_ms=-350.0)
    assert lane_contributes(typing_leads, LANE_LAG_BOUNDS_MS)
    assert lane_contributes(typing_lags, LANE_LAG_BOUNDS_MS)
    assert not lane_contributes(typing_too_early, LANE_LAG_BOUNDS_MS)


def test_joint_score_is_confidence_weighted_mean_of_contributing_lanes_only():
    optical = _ok("optical", subscore=0.8, lag_ms=45.0, confidence=1.0)
    typing = _ok("typing", subscore=0.75, lag_ms=-20.0, confidence=0.8)
    joint_score, _ = compute_joint_score([optical, typing], LANE_LAG_BOUNDS_MS)
    expected = (0.8 * 1.0 + 0.75 * 0.8) / (1.0 + 0.8)
    assert joint_score == pytest.approx(expected, abs=1e-9)


def test_adjudicate_accepts_when_lanes_agree_on_score_and_timing():
    adj = Adjudicator(AdjudicatorConfig())
    result = adj.adjudicate([
        _ok("optical", subscore=0.8, lag_ms=45.0),
        _ok("typing", subscore=0.75, lag_ms=-20.0, confidence=0.8),
        acoustic_lane_stub(),
    ])
    assert result.verdict == "ACCEPT"
    assert "coherent lane" in result.reason_text


def test_adjudicate_names_the_implausible_lag_lane_not_the_acoustic_stub():
    """Regression test for the exact bug found while building this: with
    a real lane excluded for lag AND the always-present acoustic stub
    both "not contributing," the reason must name the real, dynamic
    anomaly (optical's implausible lag), not the permanent, uninformative
    fact that acoustic is unimplemented."""
    adj = Adjudicator(AdjudicatorConfig())
    result = adj.adjudicate([_ok("optical", subscore=0.9, lag_ms=500.0), acoustic_lane_stub()])
    assert result.verdict != "ACCEPT"
    assert "optical" in result.reason_text
    assert "acoustic" not in result.reason_text


def test_adjudicate_names_no_evidence_lane_when_that_is_the_reason():
    adj = Adjudicator(AdjudicatorConfig())
    result = adj.adjudicate([
        _ok("optical", subscore=0.1, lag_ms=40.0),
        _no_evidence("typing", diagnostics="no typing in window"),
    ])
    assert result.verdict == "REJECT"
    assert "typing" in result.reason_text
    assert "no evidence" in result.reason_text


def test_ablating_the_only_contributing_lane_leaves_evidence_insufficient():
    """Same property the ablation harness checks over stored logs,
    verified directly here: with only one real lane and it held out
    (removed from the list entirely, as an ablation would), the verdict
    must be RE-CHALLENGE (insufficient evidence), never a confident REJECT
    or ACCEPT manufactured from nothing."""
    adj_full = Adjudicator(AdjudicatorConfig())
    full = adj_full.adjudicate([_ok("optical", subscore=0.85, lag_ms=45.0), acoustic_lane_stub()])
    assert full.verdict == "ACCEPT"

    adj_ablated = Adjudicator(AdjudicatorConfig())
    ablated = adj_ablated.adjudicate([acoustic_lane_stub()])  # optical held out
    assert ablated.verdict == "RE-CHALLENGE"


def test_hysteresis_prevents_flicker_at_the_accept_boundary():
    config = AdjudicatorConfig(accept_threshold=0.6, reject_threshold=0.3, hysteresis_margin=0.05)
    adj = Adjudicator(config)

    # Cross into ACCEPT first.
    r1 = adj.adjudicate([_ok("optical", subscore=0.62, lag_ms=40.0)])
    assert r1.verdict == "ACCEPT"

    # A score that would fail a FRESH 0.60 threshold (0.58 < 0.60) should
    # still read as ACCEPT immediately after, because hysteresis lowers
    # the effective threshold to 0.55 while already ACCEPTed.
    r2 = adj.adjudicate([_ok("optical", subscore=0.58, lag_ms=40.0)])
    assert r2.verdict == "ACCEPT", "hysteresis should have kept this ACCEPT, not flickered to RE-CHALLENGE"

    # But a real drop below the lowered threshold does leave ACCEPT.
    r3 = adj.adjudicate([_ok("optical", subscore=0.50, lag_ms=40.0)])
    assert r3.verdict != "ACCEPT"


def test_min_contributing_lanes_forces_re_challenge_even_with_a_high_score():
    config = AdjudicatorConfig(min_contributing_lanes=2)
    adj = Adjudicator(config)
    result = adj.adjudicate([_ok("optical", subscore=0.95, lag_ms=40.0), acoustic_lane_stub()])
    assert result.verdict == "RE-CHALLENGE", "one contributing lane should not satisfy a 2-lane minimum"


# ---------------------------------------------------------------------------
# 2026-09-02 bug fix: a confidence-weighted mean must not let one confident,
# high-scoring lane rescue a session where another CONTRIBUTING lane's own
# subscore already reads as an attack (below reject_threshold). Confirmed
# reproducible before this fix with realistic values (optical subscore=0.05
# confidence=0.20 -- the lowest confidence reachable while status is still
# "ok" at the default snr_floor_db=3.0/15.0 scaling -- plus typing
# subscore=0.95 confidence=1.0 -> joint=0.80, ACCEPT).
# ---------------------------------------------------------------------------

def test_a_passing_lane_cannot_rescue_a_contributing_lane_that_scored_below_reject_threshold():
    config = AdjudicatorConfig(accept_threshold=0.6, reject_threshold=0.3)
    adj = Adjudicator(config)
    optical_attack = _ok("optical", subscore=0.05, lag_ms=290.0, confidence=0.20)
    typing_strong = _ok("typing", subscore=0.95, lag_ms=-10.0, confidence=1.0)

    result = adj.adjudicate([optical_attack, typing_strong, acoustic_lane_stub()])

    # The unguarded weighted mean here is (0.05*0.20 + 0.95*1.0)/1.20 = 0.80,
    # which clears accept_threshold -- confirming this test actually
    # exercises the rescue scenario, not a case that would fail anyway.
    assert result.joint_score == pytest.approx(0.80, abs=0.01)
    assert result.verdict == "REJECT"
    assert "optical" in result.reason_text
    assert "0.05" in result.reason_text


def test_downgrade_only_applies_when_the_failing_lane_actually_contributes():
    """A lane with a low subscore that does NOT contribute (implausible
    lag, or non-ok status) must not trigger the downgrade -- only a lane
    that genuinely counted toward the joint score and still scored below
    reject_threshold should pull an ACCEPT back."""
    config = AdjudicatorConfig(accept_threshold=0.6, reject_threshold=0.3)
    adj = Adjudicator(config)
    optical_excluded = _ok("optical", subscore=0.05, lag_ms=500.0, confidence=0.20)  # lag outside (0,300)
    typing_strong = _ok("typing", subscore=0.95, lag_ms=-10.0, confidence=1.0)

    result = adj.adjudicate([optical_excluded, typing_strong])

    assert result.verdict == "ACCEPT"  # typing alone, uncontested, legitimately accepts


def test_downgrade_does_not_fire_for_a_verdict_that_was_already_below_accept():
    """The downgrade is scoped to pulling back an ACCEPT -- a verdict that
    was already REJECT/RE-CHALLENGE through the normal threshold branches
    must keep ITS OWN reason (e.g. a no_evidence lane), not have the
    failing-lane message override it."""
    config = AdjudicatorConfig(accept_threshold=0.6, reject_threshold=0.3)
    adj = Adjudicator(config)
    result = adj.adjudicate([
        _ok("optical", subscore=0.1, lag_ms=40.0),  # contributes, below reject_thr, but verdict is
        _no_evidence("typing", diagnostics="no typing in window"),  # already REJECT via the normal branch
    ])
    assert result.verdict == "REJECT"
    assert "typing" in result.reason_text
    assert "no evidence" in result.reason_text


def test_re_challenge_names_the_real_joint_score_not_the_acoustic_stub():
    """Bug fix (2026-09-02): a RE-CHALLENGE where two REAL lanes both
    contributed but the weighted mean fell in the gap between thresholds
    used to name the permanently-stubbed acoustic lane instead of the
    actual number -- the acoustic lane is uninformative here, the real
    story is 'joint score below acceptance.'"""
    config = AdjudicatorConfig(accept_threshold=0.6, reject_threshold=0.3)
    adj = Adjudicator(config)
    optical = _ok("optical", subscore=0.046, lag_ms=295.0, confidence=0.37)
    typing = _ok("typing", subscore=0.75, lag_ms=-30.0, confidence=0.8)

    result = adj.adjudicate([optical, typing, acoustic_lane_stub()])

    assert result.verdict == "RE-CHALLENGE"
    assert "joint score" in result.reason_text
    assert "acoustic" not in result.reason_text
