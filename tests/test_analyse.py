"""Milestone 13 acceptance tests for eval/analyse.py's evaluation
extensions: per-attack-type APCER/BPCER/ACER (not pooled), the
cross-tabulated robustness table, the collapse-latency CDF's data loading,
and the per-lane ablation summary table -- each a pure function over
already-loaded data, so these are tested directly with small synthetic
inputs rather than needing a live camera or a real multi-hundred-session
corpus."""
import json

import numpy as np

from eval.analyse import (
    per_attack_type_metrics,
    robustness_table,
    collapse_latency_summary,
    load_collapse_latencies,
    ablation_summary_table,
)


def _record(condition, score, meta=None, insufficient_signal=False, snr_db=5.0):
    return {
        "session": f"s_{condition}_{score}", "condition": condition, "score": score,
        "insufficient_signal": insufficient_signal, "snr_db": snr_db,
        "meta": meta or {"lighting": "normal", "distance_cm": 60.0, "makeup": "none",
                          "glasses": "without", "skin_tone": None},
    }


# ---------------------------------------------------------------------------
# per_attack_type_metrics
# ---------------------------------------------------------------------------

def test_per_attack_type_metrics_breaks_out_easy_and_hard_types_separately():
    records = (
        [_record("bonafide", 0.8) for _ in range(5)]
        + [_record("inject_static", 0.1) for _ in range(5)]      # easy to catch: low scores
        + [_record("inject_adaptive", 0.75) for _ in range(5)]   # hard: scores near bonafide
    )
    out = per_attack_type_metrics(records, ["inject_static", "inject_adaptive", "inject_swap"], threshold=0.5)

    assert out["inject_static"]["n"] == 5
    assert out["inject_static"]["apcer"] == 0.0   # none of the easy attack's scores clear 0.5
    assert out["inject_adaptive"]["n"] == 5
    assert out["inject_adaptive"]["apcer"] == 1.0  # all of the hard attack's scores clear 0.5
    # a pooled APCER (5 easy + 5 hard) would have reported 0.5 for both types --
    # exactly the averaging-away this milestone exists to avoid.
    assert out["inject_adaptive"]["apcer"] != out["inject_static"]["apcer"]
    # BPCER is identical across rows -- it only depends on bonafide_scores + threshold.
    assert out["inject_static"]["bpcer"] == out["inject_adaptive"]["bpcer"]


def test_per_attack_type_metrics_reports_honest_zero_n_for_uncollected_type():
    records = [_record("bonafide", 0.8), _record("inject_static", 0.1)]
    out = per_attack_type_metrics(records, ["inject_static", "inject_reenact"], threshold=0.5)
    assert out["inject_reenact"] == {"n": 0}
    assert "apcer" not in out["inject_reenact"]  # no fabricated metric for a type with zero data


def test_per_attack_type_metrics_excludes_insufficient_signal():
    records = [
        _record("bonafide", 0.8),
        _record("inject_static", 0.9, insufficient_signal=True),  # measurement failure, not a real high score
        _record("inject_static", 0.1),
    ]
    out = per_attack_type_metrics(records, ["inject_static"], threshold=0.5)
    assert out["inject_static"]["n"] == 1  # the insufficient_signal one is excluded


# ---------------------------------------------------------------------------
# robustness_table
# ---------------------------------------------------------------------------

def test_robustness_table_groups_by_full_combination_not_one_field():
    records = [
        _record("bonafide", 0.8, meta={"lighting": "normal", "distance_cm": 60.0, "makeup": "none",
                                        "glasses": "without", "skin_tone": None}),
        _record("bonafide", 0.2, meta={"lighting": "dim", "distance_cm": 60.0, "makeup": "none",
                                        "glasses": "without", "skin_tone": None}),
        _record("bonafide", 0.9, meta={"lighting": "normal", "distance_cm": 60.0, "makeup": "none",
                                        "glasses": "without", "skin_tone": None}),
    ]
    out = robustness_table(records)
    # two distinct combinations present -- normal/60/none/without (n=2) and dim/60/none/without (n=1)
    assert len(out) == 2
    normal_row = next(r for r in out if r["lighting"] == "normal")
    assert normal_row["n"] == 2
    assert normal_row["score_mean"] == np.mean([0.8, 0.9])
    dim_row = next(r for r in out if r["lighting"] == "dim")
    assert dim_row["n"] == 1
    assert dim_row["score_mean"] == 0.2


def test_robustness_table_ignores_non_bonafide_records():
    records = [_record("bonafide", 0.8), _record("inject_static", 0.1)]
    out = robustness_table(records)
    assert len(out) == 1
    assert out[0]["n"] == 1


# ---------------------------------------------------------------------------
# collapse latency
# ---------------------------------------------------------------------------

def test_collapse_latency_summary_matches_hand_computed_values():
    stats = collapse_latency_summary([0.1, 0.2, 0.3, 0.4])
    assert stats["n"] == 4
    assert stats["mean_s"] == 0.25
    assert stats["min_s"] == 0.1
    assert stats["max_s"] == 0.4


def test_collapse_latency_summary_empty():
    assert collapse_latency_summary([]) == {"n": 0}


def test_load_collapse_latencies_pools_across_files(tmp_path):
    (tmp_path / "collapse_latency_20260101T000000.json").write_text(
        json.dumps({"latencies_s": [0.1, 0.2], "stats": {}}))
    (tmp_path / "collapse_latency_20260101T000100.json").write_text(
        json.dumps({"latencies_s": [0.3], "stats": {}}))
    (tmp_path / "some_session.json").write_text(json.dumps({"session": "x", "condition": "bonafide"}))

    latencies, n_files = load_collapse_latencies(tmp_path, "collapse_latency_*.json")

    assert n_files == 2
    assert sorted(latencies) == [0.1, 0.2, 0.3]


def test_load_collapse_latencies_empty_dir_returns_empty(tmp_path):
    latencies, n_files = load_collapse_latencies(tmp_path, "collapse_latency_*.json")
    assert latencies == []
    assert n_files == 0


# ---------------------------------------------------------------------------
# ablation_summary_table
# ---------------------------------------------------------------------------

def test_ablation_summary_table_reports_fraction_degraded_per_lane(tmp_path):
    rows = [
        {"session": "a", "full": {"verdict": "ACCEPT"}, "optical": {"verdict": "RE-CHALLENGE"},
         "typing": {"verdict": "ACCEPT"}, "acoustic": {"verdict": "ACCEPT"}},
        {"session": "b", "full": {"verdict": "ACCEPT"}, "optical": {"verdict": "RE-CHALLENGE"},
         "typing": {"verdict": "RE-CHALLENGE"}, "acoustic": {"verdict": "ACCEPT"}},
        {"session": "c", "full": {"verdict": "RE-CHALLENGE"}, "optical": {"verdict": "RE-CHALLENGE"},
         "typing": {"verdict": "RE-CHALLENGE"}, "acoustic": {"verdict": "RE-CHALLENGE"}},
    ]
    path = tmp_path / "ablation.json"
    path.write_text(json.dumps(rows))

    out = ablation_summary_table(path)

    assert out["n_sessions"] == 3
    assert out["full_accept_count"] == 2  # sessions a and b only -- c never accepted with all lanes
    assert out["lanes"]["optical"]["n_degraded_when_ablated"] == 2  # both a and b degrade
    assert out["lanes"]["optical"]["fraction_degraded"] == 1.0
    assert out["lanes"]["typing"]["n_degraded_when_ablated"] == 1  # only b degrades
    assert out["lanes"]["typing"]["fraction_degraded"] == 0.5
    assert out["lanes"]["acoustic"]["n_degraded_when_ablated"] == 0  # stub never changes the verdict


def test_ablation_summary_table_missing_file_returns_none(tmp_path):
    assert ablation_summary_table(tmp_path / "does_not_exist.json") is None
