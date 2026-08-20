"""Milestone 11: fusion and adjudication.

Combines LaneResults from however many lanes actually ran (today: optical,
typing; acoustic is a registered-but-permanently-no_evidence stub, see
acoustic_lane_stub -- rule: no acoustic capture or audio I/O exists
anywhere in this repo) into a single three-way verdict: ACCEPT,
RE-CHALLENGE, or REJECT.

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
    timestamp: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {VALID_STATUSES}, got {self.status!r}")
        if self.status != "ok" and self.subscore is not None:
            raise ValueError(f"{self.lane_name}: subscore must be None when status={self.status!r}")


def acoustic_lane_stub() -> LaneResult:
    """Registers the acoustic lane's LaneResult interface now, permanently
    no_evidence, so a real acoustic lane later is a new implementation
    behind this same interface, not a fusion-layer rewrite. No audio
    capture, no microphone access, no acoustic signal processing exists
    anywhere in this repo -- this function does no I/O of any kind."""
    return LaneResult(lane_name="acoustic", subscore=None, status="no_evidence",
                       lag_ms=None, confidence=0.0, diagnostics="acoustic lane not implemented")


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
        self._last_verdict = verdict

        per_lane_reasons = {
            lane.lane_name: _lane_reason(lane, contributions[lane.lane_name]["contributes"])
            for lane in lane_results
        }

        reason_text = self._summarize(verdict, lane_results, contributions, joint_score)
        return JointResult(verdict=verdict, joint_score=joint_score, lane_results=lane_results,
                            reason_text=reason_text, per_lane_reasons=per_lane_reasons)

    # Lanes that are ALWAYS no_evidence today (registered-but-stubbed, e.g.
    # acoustic) are never picked as the PRIMARY reason unless they're the
    # only lane present -- otherwise a permanent, unchanging fact would
    # perpetually drown out whatever a real lane (optical/typing) actually
    # did this window, which is the useful, dynamic diagnosis an operator
    # or panel needs to see.
    _PERMANENTLY_STUBBED_LANES = frozenset({"acoustic"})

    @classmethod
    def _summarize(cls, verdict: str, lane_results: list, contributions: dict, joint_score: float) -> str:
        n_contributing = sum(1 for c in contributions.values() if c["contributes"])
        if verdict == "ACCEPT":
            return f"joint score {joint_score:.2f} across {n_contributing} coherent lane(s)"

        real_lanes = [l for l in lane_results if l.lane_name not in cls._PERMANENTLY_STUBBED_LANES]

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
        # Then: any stubbed lane, only if nothing real explains the verdict.
        for lane in lane_results:
            if lane.lane_name in cls._PERMANENTLY_STUBBED_LANES:
                return _lane_reason(lane, False)
        if np.isnan(joint_score):
            return "no lane produced usable evidence this window"
        return f"joint score {joint_score:.2f} below the acceptance threshold"
