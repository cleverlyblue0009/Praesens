"""Milestone 11: fusion and adjudication.

Combines LaneResults from however many lanes actually ran (optical,
typing, and -- when requested -- acoustic, see praesens/acoustic.py;
acoustic_lane_stub stands in whenever it didn't run) into a single
three-way verdict: ACCEPT, RE-CHALLENGE, or REJECT.

The claim being made is JOINT TEMPORAL COHERENCE, not an averaged
classifier score. A lane's subscore only counts toward the joint score if
that lane's OWN measured lag falls within its configured, physically
plausible window (see LANE_LAG_BOUNDS_MS) -- a lane reporting a high
correlation at an implausible lag (coincidence, a bug, or an adversary who
got lucky on magnitude but not on timing) contributes ZERO to the joint
score regardless of how high that number looks. This matters because
averaging in a spurious high subscore would let one lucky/broken lane prop
up a verdict the other lanes don't support; requiring each contributing
lane to also agree on WHEN is what makes "joint" mean something more than
"whichever lane scored highest."

Explainability is a deliverable, not a nicety (rule from the brief): every
JointResult carries a plain-sentence reason naming the lane, its subscore,
and its lag, not just a verdict enum -- eval/ablate.py and (later)
demo/panel.py both read this same reason_text/per_lane_reasons structure
rather than reconstructing an explanation from raw numbers.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

VALID_STATUSES = ("ok", "no_evidence", "insufficient_signal")
VERDICTS = ("ACCEPT", "RE-CHALLENGE", "REJECT")


@dataclass
class LaneResult:
    """Uniform result shape every lane emits, on a common rolling clock
    aligned to the same challenge start_time (each lane's lag_ms is
    relative to that shared t=0, the same convention praesens.optical and
    praesens.spatial already use for their own lag searches)."""
    lane_name: str
    subscore: float | None          # None unless status == "ok"
    status: str                     # "ok" | "no_evidence" | "insufficient_signal"
    lag_ms: float | None            # None unless status == "ok"
    confidence: float               # 0..1, this lane's own confidence in subscore (e.g. sample count, SNR)
    diagnostics: str = ""
    is_stub: bool = False           # True ONLY for a lane that never ran at all this session (e.g.
                                     # acoustic_lane_stub() below when --lanes excluded acoustic). A
                                     # REAL lane that ran and genuinely found nothing (mic too quiet,
                                     # no face detected) is status="no_evidence" with is_stub=False --
                                     # that's a real, informative measurement gap, not the same thing
                                     # as "this lane doesn't exist yet." status alone can't
                                     # distinguish these two once a lane sometimes runs for real and
                                     # sometimes doesn't (depending on --lanes), which is why this is
                                     # a separate field rather than inferred from lane_name/status.
    timestamp: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {self.status!r}")
        if self.status != "ok" and self.subscore is not None:
            raise ValueError(f"{self.lane_name}: subscore must be None when status={self.status!r}")


def acoustic_lane_stub() -> LaneResult:
    """Fallback LaneResult for a session where --lanes did not include
    acoustic (or a real acoustic run raised an exception -- see
    praesens/session.py's acoustic_box error handling). is_stub=True is
    what tells Adjudicator._summarize() to treat this as "acoustic never
    ran," not as a real measurement gap -- see LaneResult.is_stub's
    docstring for why that distinction can't be made from lane_name or
    status alone anymore, now that praesens.acoustic.run_acoustic_session
    is a real implementation that sometimes DOES run and legitimately
    reports no_evidence/insufficient_signal itself."""
    return LaneResult(lane_name="acoustic", subscore=None, status="no_evidence",
                       lag_ms=None, confidence=0.0, diagnostics="acoustic lane not implemented",
                       is_stub=True)


# Per-lane plausible-lag windows, in ms, relative to the shared challenge
# clock. Optical/spatial: response cannot precede stimulus (light travels
# screen -> face -> camera in that order), so [0, bound]. Typing: hand
# motion and key-down registration are both downstream of the SAME
# physical press, not a one-directional stimulus-response pair, so the
# window is bounded on both sides of zero. Acoustic: not implemented, but
# a window is defined now so wiring it in later doesn't also require a
# fusion-layer change here.
LANE_LAG_BOUNDS_MS = {
    "optical": (0.0, 300.0),
    "spatial": (0.0, 300.0),
    "typing": (-300.0, 300.0),
    "acoustic": (-500.0, 500.0),
}


def lane_contributes(lane: LaneResult, lag_bounds: dict) -> bool:
    """A lane contributes to the joint score only if status=='ok' AND its
    measured lag falls within ITS OWN configured window. This is the
    single check that turns "joint coherence" into something stricter
    than "average whatever numbers came in.\""""
    if lane.status != "ok" or lane.subscore is None or lane.lag_ms is None:
        return False
    lo, hi = lag_bounds.get(lane.lane_name, (0.0, 300.0))
    return lo <= lane.lag_ms <= hi


def compute_joint_score(lane_results: list, lag_bounds: dict) -> tuple:
    """Confidence-weighted mean of CONTRIBUTING lanes' subscores only;
    non-contributing lanes (no_evidence, insufficient_signal, or lag
    outside their window) get weight zero and are excluded from the mean
    entirely, not averaged in at zero. Returns (joint_score,
    per_lane_contribution) -- nan joint_score if nothing contributed."""
    contributions = {}
    weighted_sum = 0.0
    weight_total = 0.0
    for lane in lane_results:
        contributes = lane_contributes(lane, lag_bounds)
        weight = lane.confidence if contributes else 0.0
        contributions[lane.lane_name] = {
            "contributes": contributes, "weight": weight,
            "subscore": lane.subscore, "status": lane.status, "lag_ms": lane.lag_ms,
        }
        if contributes:
            weighted_sum += weight * lane.subscore
            weight_total += weight
    joint_score = (weighted_sum / weight_total) if weight_total > 1e-9 else float("nan")
    return joint_score, contributions


def _lane_reason(lane: LaneResult, contributes: bool) -> str:
    if lane.status == "no_evidence":
        return f"{lane.lane_name} lane: no evidence ({lane.diagnostics or 'nothing to measure'})"
    if lane.status == "insufficient_signal":
        return f"{lane.lane_name} lane: insufficient signal ({lane.diagnostics or 'signal too weak to measure'})"
    if not contributes:
        return (f"{lane.lane_name} lane: reflected pattern did not match this session's code -- "
                f"subscore {lane.subscore:.2f} at lag {lane.lag_ms:.0f}ms, outside the plausible "
                f"window, excluded from the joint score")
    return f"{lane.lane_name} lane: subscore {lane.subscore:.2f} at lag {lane.lag_ms:.0f}ms"


@dataclass
class JointResult:
    verdict: str
    joint_score: float
    lane_results: list
    reason_text: str
    per_lane_reasons: dict
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class AdjudicatorConfig:
    accept_threshold: float = 0.6
    reject_threshold: float = 0.3     # below this -> REJECT; between reject and accept -> RE-CHALLENGE
    hysteresis_margin: float = 0.05   # once a verdict is reached, needs to clear the threshold by
                                       # this much more to leave it -- stops the demo flickering
                                       # when the joint score sits right at a boundary
    min_contributing_lanes: int = 1

    @classmethod
    def from_dict(cls, d: dict) -> "AdjudicatorConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class Adjudicator:
    """Stateful ONLY to apply hysteresis (holds the previous verdict) --
    everything else is a pure function of the current lane_results."""

    def __init__(self, config: AdjudicatorConfig, lag_bounds: dict | None = None):
        self.config = config
        self.lag_bounds = lag_bounds or LANE_LAG_BOUNDS_MS
        self._last_verdict = "RE-CHALLENGE"

    def reset(self) -> None:
        self._last_verdict = "RE-CHALLENGE"

    def adjudicate(self, lane_results: list) -> JointResult:
        joint_score, contributions = compute_joint_score(lane_results, self.lag_bounds)
        n_contributing = sum(1 for c in contributions.values() if c["contributes"])

        accept_thr = self.config.accept_threshold
        reject_thr = self.config.reject_threshold
        if self._last_verdict == "ACCEPT":
            accept_thr -= self.config.hysteresis_margin
        elif self._last_verdict == "REJECT":
            reject_thr += self.config.hysteresis_margin

        if n_contributing < self.config.min_contributing_lanes or np.isnan(joint_score):
            verdict = "RE-CHALLENGE"
        elif joint_score >= accept_thr:
            verdict = "ACCEPT"
        elif joint_score < reject_thr:
            verdict = "REJECT"
        else:
            verdict = "RE-CHALLENGE"

        # Bug fix (2026-09-02): a confidence-weighted MEAN can let one
        # high-confidence, high-subscore lane numerically outvote another
        # CONTRIBUTING lane that, on its own terms, already looks like an
        # attack (subscore below reject_threshold) -- confirmed reachable
        # with realistic values (optical subscore=0.05 confidence=0.20 +
        # typing subscore=0.95 confidence=1.0 -> joint=0.80, ACCEPT, before
        # this fix). Averaging is not the same claim as joint coherence:
        # "both lanes agree" must mean no contributing lane is independently
        # screaming REJECT. Scoped narrowly to the ACCEPT case only -- a
        # verdict already RE-CHALLENGE/REJECT is not made MORE lenient by
        # this check, only an about-to-be-lenient ACCEPT is pulled back.
        failing_contributors = [
            lane for lane in lane_results
            if contributions[lane.lane_name]["contributes"] and lane.subscore < reject_thr
        ]
        downgraded_from_accept = verdict == "ACCEPT" and bool(failing_contributors)
        if downgraded_from_accept:
            verdict = "REJECT"

        self._last_verdict = verdict

        per_lane_reasons = {
            lane.lane_name: _lane_reason(lane, contributions[lane.lane_name]["contributes"])
            for lane in lane_results
        }

        reason_text = self._summarize(
            verdict, lane_results, contributions, joint_score,
            failing_contributors if downgraded_from_accept else None,
        )
        return JointResult(verdict=verdict, joint_score=joint_score, lane_results=lane_results,
                            reason_text=reason_text, per_lane_reasons=per_lane_reasons)

    @classmethod
    def _summarize(cls, verdict: str, lane_results: list, contributions: dict, joint_score: float,
                    failing_contributors: list | None = None) -> str:
        n_contributing = sum(1 for c in contributions.values() if c["contributes"])

        # Most specific reason of all: this verdict is REJECT specifically
        # BECAUSE a contributing lane's own subscore said so, overriding
        # what would otherwise have been an ACCEPT -- see adjudicate()'s
        # downgraded_from_accept. Only ever non-empty in that exact case,
        # never for a REJECT/RE-CHALLENGE reached through the normal
        # threshold branches (those keep their own, different reasons below).
        if failing_contributors:
            worst = min(failing_contributors, key=lambda l: l.subscore)
            return (f"{worst.lane_name} lane failed -- scored {worst.subscore:.2f}, below "
                    f"reject_threshold; a lane failing this clearly cannot be outvoted by "
                    f"another lane's confidence")

        if verdict == "ACCEPT":
            return f"joint score {joint_score:.2f} across {n_contributing} coherent lane(s)"

        # A lane that never ran this session (is_stub=True, e.g. acoustic
        # when --lanes excluded it) is never picked as the PRIMARY reason
        # unless nothing else explains the verdict -- otherwise a
        # permanent, unchanging non-fact would perpetually drown out
        # whatever a real lane actually did this window. A lane that DID
        # run and genuinely found nothing (is_stub=False, status=
        # no_evidence/insufficient_signal) is a real, dynamic diagnosis
        # and belongs in real_lanes like any other measured lane.
        real_lanes = [l for l in lane_results if not l.is_stub]

        # Most informative first: a real lane that measured something (ok)
        # but got excluded for landing at an implausible lag -- a dynamic,
        # actionable anomaly, not a permanent fact about this deployment.
        for lane in real_lanes:
            if not contributions[lane.lane_name]["contributes"] and lane.status == "ok":
                return _lane_reason(lane, False)
        # Then: a real lane with a genuine measurement gap this window.
        for lane in real_lanes:
            if lane.status in ("no_evidence", "insufficient_signal"):
                return _lane_reason(lane, False)
        # Then: a real, numeric explanation -- something DID contribute,
        # it just didn't clear the threshold -- BEFORE falling back to
        # naming a permanently-stubbed lane (bug fixed 2026-09-02: this
        # used to be checked AFTER the stubbed-lane loop below, so it was
        # dead code whenever acoustic was present -- every RE-CHALLENGE
        # with two real contributing-but-insufficient lanes named the inert
        # acoustic stub instead of the actual joint score).
        if not np.isnan(joint_score):
            return f"joint score {joint_score:.2f} below the acceptance threshold"
        # Then: any never-ran stub, only if nothing real explains the verdict.
        for lane in lane_results:
            if lane.is_stub:
                return _lane_reason(lane, False)
        return "no lane produced usable evidence this window"


# ---------------------------------------------------------------------------
# Binary gate (2026-09-03; acoustic added as a second secondary lane
# 2026-09-14) -- the actual policy the concurrent session
# (praesens/session.py) verdicts against. This is
# a SEPARATE, additional adjudication path, not a replacement for
# Adjudicator above: Adjudicator's confidence-weighted-mean-with-lag-
# coherence remains what it always was (used by eval/ablate.py, its own
# tests, and any future N-lane/joint-coherence work); this function
# instead implements an explicit, asymmetric decision table for the
# specific two real lanes this project has today, requested directly:
#
#   optical PASS + typing PASS  -> ACCEPT
#   optical PASS + typing FAIL  -> RE-CHALLENGE ("try typing again" --
#       typing not clearing its bar is treated as INCONCLUSIVE, not a
#       hard failure, when the PRIMARY liveness signal (optical, which
#       defeats video injection) is solid on its own)
#   optical FAIL (either way)   -> REJECT (a passing typing lane must
#       NEVER rescue a failing optical lane -- this project's core
#       "faking one lane doesn't help" claim, the same property
#       Adjudicator's own downgrade rule above enforces for the general
#       weighted-mean case)
#
# The asymmetry (typing failing is soft, optical failing is hard) is
# deliberate: optical is what the whole system exists to measure against
# video injection; typing is a secondary confirmation that the operator
# is presently, actively engaged with THIS session's unpredictable
# challenge, not the primary defence.
#
# With --lanes optical+typing+acoustic, acoustic joins typing as a
# SECONDARY lane under the same rule: ACCEPT needs optical AND every
# secondary lane to pass; optical passing with any secondary failing is
# RE-CHALLENGE; optical failing is REJECT whatever the others say.
# ---------------------------------------------------------------------------

def lane_passes(lane: LaneResult, pass_threshold: float) -> bool:
    """A lane 'passes' on its own terms: status=='ok' and its subscore
    clears pass_threshold. Deliberately simpler than lane_contributes()
    above (no lag-window check) -- lag-based joint coherence isn't the
    applicable concept for a lane with no camera-correlated timing to
    check against (e.g. keystroke_normality_score has no lag at all)."""
    return lane.status == "ok" and lane.subscore is not None and lane.subscore >= pass_threshold


def _short_diagnostics(lane: LaneResult) -> str:
    """The headline of a lane's diagnostics for a one-sentence reason (it's
    shown fullscreen on the verdict banner) -- the full string stays in
    the log's fusion.lanes[].diagnostics. Acoustic diagnostics run to a
    dozen fields; "tone not heard" is the one an operator can act on."""
    diag = lane.diagnostics or ""
    if "tone not heard" in diag:
        return "tone not heard -- speaker muted or too quiet, or output is headphones"
    return diag.split(", ")[0]


def _secondary_fail_reason(lane: LaneResult, retry_hint: str) -> str:
    if lane.status != "ok":
        label = "no evidence" if lane.status == "no_evidence" else "insufficient signal"
        return f"{lane.lane_name} lane: {label} ({_short_diagnostics(lane) or 'nothing to measure'}) -- {retry_hint}"
    if lane.lag_ms is not None and not lane_contributes(lane, LANE_LAG_BOUNDS_MS):
        return (f"{lane.lane_name} lane: subscore {lane.subscore:.2f} but implausible lag "
                f"({lane.lag_ms:.0f}ms) -- {retry_hint}")
    return (f"{lane.lane_name} lane: subscore {lane.subscore:.2f} below pass threshold "
            f"({_short_diagnostics(lane)}) -- {retry_hint}")


def _per_lane_reason(lane: LaneResult) -> str:
    if lane.status != "ok":
        return f"{lane.lane_name} lane: {lane.diagnostics or lane.status}"
    if lane.lane_name == "optical":
        return f"optical lane: subscore {lane.subscore:.2f}"
    if lane.lag_ms is None:
        return f"{lane.lane_name} lane: subscore {lane.subscore:.2f} ({lane.diagnostics})"
    return f"{lane.lane_name} lane: subscore {lane.subscore:.2f} at lag {lane.lag_ms:.0f}ms"


def adjudicate_two_lane(optical: LaneResult, typing: LaneResult | None,
                         optical_pass_threshold: float, typing_pass_threshold: float,
                         acoustic: LaneResult | None = None,
                         acoustic_pass_threshold: float = 0.3) -> JointResult:
    """typing=None (--lanes optical) collapses to a plain two-way gate:
    ACCEPT if optical passes, REJECT if not -- there is no secondary
    signal to ask the operator to retry, so RE-CHALLENGE doesn't apply.

    optical "passing" additionally requires lane_contributes() (its
    measured lag against the emitted light stimulus falls in its
    plausible window, LANE_LAG_BOUNDS_MS) on top of clearing
    optical_pass_threshold -- unlike typing (which has no equivalent
    stimulus-response lag left to check now that keystroke_normality_score
    is timing-only, see praesens/typing.py), optical still measures a
    real physical lag every session, and a high subscore at an
    implausible lag is exactly the "lucky magnitude, wrong timing"
    scenario the joint-temporal-coherence claim exists to catch (see
    module docstring) -- worth keeping even in this simpler binary gate.

    acoustic (optional, despite this function's name): a secondary lane
    like typing, but it DOES measure a physical lag against the emitted
    probe, so passing needs lane_contributes() as well as clearing
    acoustic_pass_threshold."""
    def _optical_fail_reason() -> str:
        if optical.status != "ok":
            return f"optical lane: {optical.diagnostics or 'insufficient signal'}"
        if not lane_contributes(optical, LANE_LAG_BOUNDS_MS):
            return f"optical lane: subscore {optical.subscore:.2f} but implausible lag ({optical.lag_ms}ms)"
        return f"optical lane failed -- scored {optical.subscore:.2f}, below pass threshold"

    optical_ok = lane_passes(optical, optical_pass_threshold) and lane_contributes(optical, LANE_LAG_BOUNDS_MS)

    if typing is None and acoustic is None:
        verdict = "ACCEPT" if optical_ok else "REJECT"
        reason = (f"optical lane passes (score {optical.subscore:.2f})" if optical_ok
                   else _optical_fail_reason())
        lane_results = [optical]
        per_lane_reasons = {"optical": reason}
        joint_score = optical.subscore if optical.subscore is not None else float("nan")
        return JointResult(verdict=verdict, joint_score=joint_score, lane_results=lane_results,
                            reason_text=reason, per_lane_reasons=per_lane_reasons)

    # (lane, passes, retry hint) for each secondary lane that ran
    secondaries = []
    if typing is not None:
        secondaries.append((typing, lane_passes(typing, typing_pass_threshold), "try typing again"))
    if acoustic is not None:
        acoustic_ok = (lane_passes(acoustic, acoustic_pass_threshold)
                       and lane_contributes(acoustic, LANE_LAG_BOUNDS_MS))
        secondaries.append((acoustic, acoustic_ok, "check the speaker/microphone and try again"))
    lane_results = [optical] + [lane for lane, _, _ in secondaries]
    failing = [(lane, hint) for lane, ok, hint in secondaries if not ok]

    if optical_ok and not failing:
        verdict = "ACCEPT"
        scores = ", ".join(f"{l.lane_name} {l.subscore:.2f}" for l in lane_results)
        reason = (f"both lanes agree: {scores}" if len(lane_results) == 2
                  else f"all {len(lane_results)} lanes agree: {scores}")
        joint_score = float(np.mean([l.subscore for l in lane_results]))
    elif optical_ok:
        verdict = "RE-CHALLENGE"
        reason = "; ".join(_secondary_fail_reason(lane, hint) for lane, hint in failing)
        joint_score = optical.subscore  # what actually carries this verdict is optical's own number
    else:
        verdict = "REJECT"
        reason = _optical_fail_reason()
        joint_score = optical.subscore if optical.subscore is not None else float("nan")

    per_lane_reasons = {lane.lane_name: _per_lane_reason(lane) for lane in lane_results}
    return JointResult(verdict=verdict, joint_score=joint_score, lane_results=lane_results,
                        reason_text=reason, per_lane_reasons=per_lane_reasons)
