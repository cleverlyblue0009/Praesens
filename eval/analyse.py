"""Milestone 5: analysis.

Turns the session logs written by praesens/session.py into the numbers and
figures the paper needs. The one methodological decision worth spelling
out: emitter_off sessions are a scientific control, not an attack, and
insufficient_signal sessions are a measurement failure, not a liveness
failure -- both are excluded from the headline ROC/EER/APCER/BPCER/ACER
table and reported separately instead. Folding either into "attack" would
misrepresent what the system actually detected: emitter_off proves the
score depends on the physical light pattern (that's the point of the
control), and insufficient_signal means the sensor couldn't get a reading
at all (e.g. dark skin under dim light, heavy makeup), which is exactly the
failure mode the SNR/adaptive-depth logic in Milestone 3 exists to
distinguish from an actual spoof.

Milestone 13 extends this with: (1) per-attack-type APCER/BPCER/ACER, since
a pooled ACER across every attack condition can hide one easy type behind
several hard ones (or vice versa); (2) the cross-session offline condition
(eval/cross_session.py) surfaced as its own clearly-labeled PROXY section,
never merged into the real-attack ROC/EER -- it is a stand-in for an
injection attack, built from bonafide recordings, not a recording of one;
(3) a collapse-latency CDF from demo/panel.py's Milestone 12 instrumentation;
(4) a per-lane ablation table read back from eval/ablate.py's output; (5) a
robustness table cross-tabulated over the full bona fide metadata
combination (lighting x distance x makeup x glasses x skin_tone) rather
than one field at a time, so a combination that's actually hard (or never
tested) is visible instead of averaged away.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# np.trapz was removed in numpy 2.x (deprecated since 2.0); np.trapezoid is
# its replacement but doesn't exist on older numpy. Try the new name first,
# fall back to the old one, so this runs on either.
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_sessions(logs_dir: Path) -> list:
    records = []
    for path in sorted(logs_dir.glob("*.json")):
        if path.name.startswith("probe_"):
            continue
        with open(path) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                continue
        if "session" in d and "condition" in d and "score" in d:
            records.append(d)
    return records


# ---------------------------------------------------------------------------
# (a) score distribution per condition
# ---------------------------------------------------------------------------

def score_distribution_by_condition(records: list) -> dict:
    out = {}
    conditions = sorted(set(r["condition"] for r in records))
    for cond in conditions:
        scores = np.array([r["score"] for r in records if r["condition"] == cond])
        out[cond] = {"n": int(len(scores)),
                     "mean": float(np.mean(scores)) if len(scores) else float("nan"),
                     "std": float(np.std(scores)) if len(scores) else float("nan")}
    return out


# ---------------------------------------------------------------------------
# (b)/(c) ROC, AUC, EER, APCER/BPCER/ACER
# bona fide = positive class. emitter_off and insufficient_signal sessions
# are excluded (see module docstring).
# ---------------------------------------------------------------------------

def compute_roc(bonafide_scores: np.ndarray, attack_scores: np.ndarray):
    """Manual ROC/AUC (no sklearn in the stated stack). Positive class is
    bonafide: higher score => predicted bonafide."""
    scores = np.concatenate([bonafide_scores, attack_scores])
    labels = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(attack_scores))])

    thresholds = np.unique(scores)
    thresholds = np.concatenate(([thresholds[0] - 1e-6], thresholds, [thresholds[-1] + 1e-6]))[::-1]

    tpr, fpr = [], []
    for thr in thresholds:
        pred_pos = scores >= thr
        tp = np.sum(pred_pos & (labels == 1))
        fn = np.sum(~pred_pos & (labels == 1))
        fp = np.sum(pred_pos & (labels == 0))
        tn = np.sum(~pred_pos & (labels == 0))
        tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    tpr, fpr = np.array(tpr), np.array(fpr)

    order = np.argsort(fpr)
    auc = float(_trapezoid(tpr[order], fpr[order]))
    return fpr, tpr, thresholds, auc


def compute_eer(bonafide_scores: np.ndarray, attack_scores: np.ndarray):
    fpr, tpr, thresholds, auc = compute_roc(bonafide_scores, attack_scores)
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fpr - fnr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    eer_threshold = float(thresholds[idx])
    return eer, eer_threshold, auc, fpr, tpr


def apcer_bpcer_acer(bonafide_scores: np.ndarray, attack_scores: np.ndarray, threshold: float):
    apcer = float(np.mean(attack_scores >= threshold)) if len(attack_scores) else float("nan")
    bpcer = float(np.mean(bonafide_scores < threshold)) if len(bonafide_scores) else float("nan")
    acer = float((apcer + bpcer) / 2.0) if not (np.isnan(apcer) or np.isnan(bpcer)) else float("nan")
    return apcer, bpcer, acer


# ---------------------------------------------------------------------------
# Milestone 13 (b)/(c) extension: per-attack-type APCER/BPCER/ACER, at the
# SAME threshold the pooled ROC/EER picked -- so this is a breakdown of one
# operating point, not a re-optimised threshold per attack type (which
# would flatter every type individually and misrepresent a single deployed
# system). BPCER is identical across rows by construction (it only depends
# on bonafide_scores and the shared threshold) -- included per row anyway
# so each row is a complete, standalone ACER computation.
# ---------------------------------------------------------------------------

def per_attack_type_metrics(records: list, attack_conditions: list, threshold: float) -> dict:
    clean = [r for r in records if not r.get("insufficient_signal")]
    bonafide_scores = np.array([r["score"] for r in clean if r["condition"] == "bonafide"])
    out = {}
    for cond in attack_conditions:
        attack_scores = np.array([r["score"] for r in clean if r["condition"] == cond])
        if len(attack_scores) == 0:
            out[cond] = {"n": 0}
            continue
        apcer, bpcer, acer = apcer_bpcer_acer(bonafide_scores, attack_scores, threshold)
        out[cond] = {"n": int(len(attack_scores)), "apcer": apcer, "bpcer": bpcer, "acer": acer}
    return out


# ---------------------------------------------------------------------------
# Milestone 13: robustness table, cross-tabulated over the FULL bona fide
# metadata combination rather than one field at a time (breakdown_by()
# above). A single-field breakdown can hide an interaction -- e.g.
# foundation_powder might only actually hurt under dim lighting -- and
# would silently average that away across the other lighting conditions
# it's paired with. Grouping by the exact combination present in the
# corpus also makes sparse coverage honestly visible (most combinations
# will have n=1 or n=0 in a small corpus) instead of implying more support
# than exists.
# ---------------------------------------------------------------------------

ROBUSTNESS_FIELDS = ("lighting", "distance_cm", "makeup", "glasses", "skin_tone")


def robustness_table(records: list) -> list:
    bonafide = [r for r in records if r["condition"] == "bonafide"]
    combos: dict = {}
    for r in bonafide:
        m = r["meta"]
        key = tuple(str(m.get(f)) for f in ROBUSTNESS_FIELDS)
        combos.setdefault(key, []).append(r)

    out = []
    for key, rs in sorted(combos.items()):
        scores = np.array([r["score"] for r in rs])
        snrs = np.array([r["snr_db"] for r in rs
                          if r.get("snr_db") is not None and not np.isnan(r["snr_db"])])
        row = dict(zip(ROBUSTNESS_FIELDS, key))
        row.update({
            "n": len(rs),
            "score_mean": float(scores.mean()), "score_std": float(scores.std()),
            "snr_db_mean": float(snrs.mean()) if len(snrs) else float("nan"),
        })
        out.append(row)
    return out


# ---------------------------------------------------------------------------
# Milestone 13: collapse-latency CDF, reading demo/panel.py's Milestone 12
# logs/collapse_latency_*.json files back in. Pools every recorded switch
# across however many panel sessions have been run -- "a distribution over
# >=20 switches, not one anecdote" is the claim; this is the reporting half
# of it, same split as collapse_latency_stats() itself in demo/panel.py.
# ---------------------------------------------------------------------------

def collapse_latency_summary(latencies: list) -> dict:
    """Same summary shape as demo/panel.py's own collapse_latency_stats()
    -- duplicated rather than imported, since demo/panel.py already
    imports FROM this module (load_sessions), and importing it back here
    would be a circular import as well as pulling cv2/mediapipe into a
    plain evaluation script that otherwise needs neither."""
    if not latencies:
        return {"n": 0}
    arr = np.array(latencies, dtype=np.float64)
    return {
        "n": int(len(arr)), "mean_s": float(arr.mean()), "median_s": float(np.median(arr)),
        "std_s": float(arr.std()), "min_s": float(arr.min()), "max_s": float(arr.max()),
        "p90_s": float(np.percentile(arr, 90)),
    }


def load_collapse_latencies(logs_dir: Path, glob_pattern: str) -> tuple:
    latencies: list = []
    n_files = 0
    for path in sorted(logs_dir.glob(glob_pattern)):
        try:
            with open(path) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue
        n_files += 1
        latencies.extend(d.get("latencies_s", []))
    return latencies, n_files


def plot_collapse_latency_cdf(latencies: list, out_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if not latencies:
        ax.text(0.5, 0.5, "no collapse-latency data recorded yet\n(see demo/panel.py, Milestone 12)",
                 ha="center", va="center", transform=ax.transAxes)
    else:
        arr = np.sort(np.asarray(latencies, dtype=np.float64)) * 1000.0
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.step(arr, cdf, where="post")
        ax.axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90th percentile")
        ax.set_xlabel("collapse latency (ms): source switch -> first genuine REJECT")
        ax.set_ylabel("cumulative fraction of switches")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right")
    ax.set_title(f"Collapse latency CDF (n={len(latencies)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Milestone 13: per-lane ablation table, read back from eval/ablate.py's
# (Milestone 11) per-session output. Reports, among sessions that ACCEPTed
# with every lane present, what fraction stop doing so when a given lane
# is held out -- i.e. how load-bearing each lane actually is, not just
# whether the joint score moves.
# ---------------------------------------------------------------------------

def ablation_summary_table(ablation_path: Path) -> dict | None:
    if not ablation_path.exists():
        return None
    with open(ablation_path) as f:
        rows = json.load(f)
    if not rows:
        return {"n_sessions": 0, "full_accept_count": 0, "lanes": {}}

    lane_names = sorted({k for row in rows for k in row if k not in ("session", "full")})
    full_accepts = [row for row in rows if row["full"]["verdict"] == "ACCEPT"]
    lanes = {}
    for lane in lane_names:
        degraded = sum(1 for row in full_accepts if lane in row and row[lane]["verdict"] != "ACCEPT")
        lanes[lane] = {
            "n_full_accept_sessions": len(full_accepts),
            "n_degraded_when_ablated": degraded,
            "fraction_degraded": float(degraded / len(full_accepts)) if full_accepts else float("nan"),
        }
    return {"n_sessions": len(rows), "full_accept_count": len(full_accepts), "lanes": lanes}


# ---------------------------------------------------------------------------
# Milestone 13: cross-session offline condition (eval/cross_session.py),
# surfaced here as its own clearly-labeled PROXY section -- constructed
# from bonafide recordings scored against a mismatched challenge, standing
# in for an injection attack, never merged into the real-attack ROC/EER
# above so the two are never conflated in the reported numbers.
# ---------------------------------------------------------------------------

def load_cross_session_proxy(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    return d.get("summary")


# ---------------------------------------------------------------------------
# (d) bona fide breakdown by lighting/distance/makeup/glasses, extended with
# snr_db per the Milestone 3 addendum, generalised to also cover skin_tone.
# ---------------------------------------------------------------------------

def breakdown_by(records: list, meta_field: str, value_fields=("score", "snr_db")) -> dict:
    out = {}
    values = sorted({str(r["meta"].get(meta_field)) for r in records}, key=lambda x: (x == "None", x))
    for v in values:
        subset = [r for r in records if str(r["meta"].get(meta_field)) == v]
        row = {"n": len(subset)}
        for field in value_fields:
            arr = np.array([r[field] for r in subset if r.get(field) is not None
                             and not (isinstance(r[field], float) and np.isnan(r[field]))])
            row[f"{field}_mean"] = float(np.mean(arr)) if len(arr) else float("nan")
            row[f"{field}_std"] = float(np.std(arr)) if len(arr) else float("nan")
        out[v] = row
    return out


# ---------------------------------------------------------------------------
# (e) lag at peak, mean/std per condition
# ---------------------------------------------------------------------------

def lag_by_condition(records: list) -> dict:
    out = {}
    for cond in sorted(set(r["condition"] for r in records)):
        lags = np.array([r["lag_ms"] for r in records if r["condition"] == cond])
        out[cond] = {"n": int(len(lags)),
                     "mean_ms": float(np.mean(lags)) if len(lags) else float("nan"),
                     "std_ms": float(np.std(lags)) if len(lags) else float("nan")}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_score_histogram(records: list, out_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    conditions = sorted(set(r["condition"] for r in records))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(conditions), 1)))
    for cond, color in zip(conditions, colors):
        scores = [r["score"] for r in records if r["condition"] == cond]
        if not scores:
            continue
        ax.hist(scores, bins=20, range=(-1, 1), alpha=0.6, label=f"{cond} (n={len(scores)})", color=color)
    ax.set_xlabel("score (peak normalised cross-correlation)")
    ax.set_ylabel("count")
    ax.set_title("Score distribution by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_roc(fpr, tpr, auc, eer, eer_threshold, out_path: Path, dpi: int, attack_conditions: list = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    order = np.argsort(fpr)
    ax.plot(fpr[order], tpr[order], label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.plot(eer, 1 - eer, "o", color="red", label=f"EER={eer:.3f} @ thr={eer_threshold:.3f}")
    ax.set_xlabel("APCER / False Positive Rate (attack accepted)")
    ax.set_ylabel("True Positive Rate (bona fide accepted)")
    label = "+".join(attack_conditions) if attack_conditions else "attack"
    ax.set_title(f"ROC -- bona fide vs. attack ({label})")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_trace_pair(genuine_record: dict | None, attack_record: dict | None, out_path: Path, dpi: int):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    for ax, record, label in [(axes[0], genuine_record, "genuine (bona fide)"),
                               (axes[1], attack_record, "injected (attack)")]:
        if record is None or not record.get("timestamps"):
            ax.text(0.5, 0.5, f"no {label} session with trace data available",
                     ha="center", va="center", transform=ax.transAxes)
            ax.set_ylabel(label)
            continue
        ts = np.array(record["timestamps"])
        ts = ts - ts[0]
        ax.plot(ts, record["trace_emitted"], label="emitted", linewidth=1.5)
        ax.plot(ts, record["trace_measured"], label="measured", linewidth=1.5)
        ax.set_ylabel(label)
        ax.legend(loc="upper right")
        ax.set_title(f"{record['session']} (score={record['score']:.3f})")

    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Emitted vs. measured luminance (z-scored)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_analysis(logs_dir: Path, figures_dir: Path, dpi: int, attack_conditions: list,
                  ablation_json: Path | None = None, collapse_latency_glob: str = "collapse_latency_*.json"
                  ) -> dict:
    records = load_sessions(logs_dir)
    summary: dict = {"n_sessions_total": len(records)}

    # -- Milestone 13: collapse-latency CDF and ablation table don't need
    # any session logs at all (different data sources), so run them even
    # when records is empty rather than bailing out below -- consistent
    # with this function's job being "everything eval/analyse.py reports",
    # not only the parts that need session.py logs.
    figures_dir.mkdir(parents=True, exist_ok=True)
    latencies, n_latency_files = load_collapse_latencies(logs_dir, collapse_latency_glob)
    summary["collapse_latency"] = {"n_files": n_latency_files, "n_switches": len(latencies),
                                    "stats": collapse_latency_summary(latencies)}
    plot_collapse_latency_cdf(latencies, figures_dir / "collapse_latency_cdf.png", dpi)

    summary["ablation"] = ablation_summary_table(ablation_json) if ablation_json else None

    summary["cross_session_proxy"] = load_cross_session_proxy(REPO_ROOT / "eval" / "cross_session.json")

    if not records:
        print(f"No session logs found in {logs_dir}. Nothing more to analyse yet.")
        return summary

    # -- (a) --
    summary["score_by_condition"] = score_distribution_by_condition(records)

    # -- coverage: insufficient_signal, reported separately, never as "attack" --
    insuff = [r for r in records if r.get("insufficient_signal")]
    summary["insufficient_signal_count"] = len(insuff)
    summary["insufficient_signal_by_condition"] = {
        cond: sum(1 for r in insuff if r["condition"] == cond)
        for cond in sorted(set(r["condition"] for r in records))
    }

    # -- (b)/(c): ROC/EER/APCER/BPCER/ACER, bonafide vs replay+swap,
    # excluding emitter_off (control) and insufficient_signal (measurement
    # failure, not a liveness verdict) from both classes --
    clean = [r for r in records if not r.get("insufficient_signal")]
    bonafide_scores = np.array([r["score"] for r in clean if r["condition"] == "bonafide"])
    attack_scores = np.array([r["score"] for r in clean if r["condition"] in attack_conditions])

    if len(bonafide_scores) > 0 and len(attack_scores) > 0:
        eer, eer_thr, auc, fpr, tpr = compute_eer(bonafide_scores, attack_scores)
        apcer, bpcer, acer = apcer_bpcer_acer(bonafide_scores, attack_scores, eer_thr)
        summary["roc_eer"] = {
            "n_bonafide": len(bonafide_scores), "n_attack": len(attack_scores),
            "auc": auc, "eer": eer, "eer_threshold": eer_thr,
            "apcer": apcer, "bpcer": bpcer, "acer": acer,
        }
        plot_roc(fpr, tpr, auc, eer, eer_thr, figures_dir / "roc_curve.png", dpi, attack_conditions)
        # Milestone 13: same eer_thr, broken out per attack TYPE instead of pooled.
        summary["per_attack_type"] = per_attack_type_metrics(records, attack_conditions, eer_thr)
    else:
        summary["roc_eer"] = None
        summary["per_attack_type"] = None
        print(f"Skipping ROC/EER: need >=1 bonafide and >=1 attack score "
              f"(have {len(bonafide_scores)} bonafide, {len(attack_scores)} attack, "
              f"after excluding insufficient_signal sessions).")

    # -- (d) bona fide breakdown, + snr_db (Milestone 3 addendum), also by skin_tone --
    bonafide_records = [r for r in records if r["condition"] == "bonafide"]
    summary["bonafide_breakdown"] = {
        field: breakdown_by(bonafide_records, field)
        for field in ("lighting", "distance_cm", "makeup", "glasses", "skin_tone")
    }
    # Milestone 13: the same bona fide data, cross-tabulated by full combination.
    summary["robustness_table"] = robustness_table(records)

    # -- (e) lag at peak per condition --
    summary["lag_by_condition"] = lag_by_condition(records)

    # -- figures --
    plot_score_histogram(records, figures_dir / "score_histogram.png", dpi)

    genuine = max((r for r in bonafide_records if r.get("timestamps")),
                  key=lambda r: r["score"], default=None)
    attack_recs = [r for r in records if r["condition"] in attack_conditions and r.get("timestamps")]
    attack = min(attack_recs, key=lambda r: r["score"], default=None)
    plot_trace_pair(genuine, attack, figures_dir / "trace_pair.png", dpi)

    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 70)
    print("PRAESENS ANALYSIS SUMMARY")
    print("=" * 70)
    print(f"\nTotal sessions: {summary.get('n_sessions_total', 0)}")

    print("\n(a) Score distribution by condition:")
    for cond, s in summary.get("score_by_condition", {}).items():
        print(f"  {cond:12s} n={s['n']:3d}  mean={s['mean']:.3f}  std={s['std']:.3f}")

    print(f"\nInsufficient-signal sessions (measurement failure, NOT reported as attack): "
          f"{summary.get('insufficient_signal_count', 0)}")
    for cond, n in summary.get("insufficient_signal_by_condition", {}).items():
        print(f"  {cond:12s} {n}")

    roc = summary.get("roc_eer")
    print("\n(b)/(c) ROC / EER / APCER / BPCER / ACER (bona fide vs. pooled real attack conditions, "
          "excludes emitter_off control and insufficient_signal sessions):")
    if roc:
        print(f"  n_bonafide={roc['n_bonafide']} n_attack={roc['n_attack']}")
        print(f"  AUC={roc['auc']:.4f}  EER={roc['eer']:.4f} @ threshold={roc['eer_threshold']:.4f}")
        print(f"  APCER={roc['apcer']:.4f}  BPCER={roc['bpcer']:.4f}  ACER={roc['acer']:.4f}")
    else:
        print("  (skipped -- insufficient data)")

    print("\n(b')  Per-attack-type APCER/BPCER/ACER at the SAME pooled threshold above "
          "(Milestone 13 -- a pooled ACER can hide one easy or hard type):")
    per_type = summary.get("per_attack_type")
    if per_type:
        for cond, row in per_type.items():
            if row["n"] == 0:
                print(f"  {cond:16s} n=0  (no real attack sessions collected for this type yet)")
            else:
                print(f"  {cond:16s} n={row['n']:3d}  APCER={row['apcer']:.4f}  "
                      f"BPCER={row['bpcer']:.4f}  ACER={row['acer']:.4f}")
    else:
        print("  (skipped -- no pooled threshold available)")

    xproxy = summary.get("cross_session_proxy")
    print("\n(b'') Cross-session PROXY (eval/cross_session.py) -- constructed from bona fide "
          "recordings scored against a MISMATCHED challenge, standing in for an injection attack. "
          "NOT a real attack recording; kept separate from (b)/(c)/(b') above by construction:")
    if xproxy and xproxy.get("table2"):
        t2 = xproxy["table2"]
        print(f"  n_positive(bonafide)={t2['n_positive']}  n_negative(xsession+emitter_off)={t2['n_negative']}")
        print(f"  AUC={t2['auc']:.4f}  EER={t2['eer']:.4f}  "
              f"APCER={t2['apcer']:.4f}  BPCER={t2['bpcer']:.4f}  ACER={t2['acer']:.4f}")
    else:
        print("  (not available -- run `python -m eval.cross_session` first)")

    print("\n(d) Bona fide score & snr_db breakdown (operating envelope, single field at a time):")
    for field, table in summary.get("bonafide_breakdown", {}).items():
        print(f"  by {field}:")
        for value, row in table.items():
            print(f"    {value:20s} n={row['n']:3d}  "
                  f"score={row['score_mean']:.3f}+/-{row['score_std']:.3f}  "
                  f"snr_db={row['snr_db_mean']:.2f}+/-{row['snr_db_std']:.2f}")

    print("\n(d') Robustness table (Milestone 13) -- full lighting x distance x makeup x glasses x "
          "skin_tone combination as actually present in the corpus:")
    rtable = summary.get("robustness_table") or []
    if rtable:
        print(f"  {'lighting':10s} {'dist_cm':8s} {'makeup':18s} {'glasses':8s} {'skin':6s} "
              f"{'n':>4s} {'score':>14s} {'snr_db':>8s}")
        for row in rtable:
            print(f"  {row['lighting']:10s} {row['distance_cm']:8s} {row['makeup']:18s} "
                  f"{row['glasses']:8s} {row['skin_tone']:6s} {row['n']:4d} "
                  f"{row['score_mean']:6.3f}+/-{row['score_std']:5.3f} {row['snr_db_mean']:8.2f}")
    else:
        print("  (no bona fide sessions)")

    print("\n(e) Lag at peak (ms) by condition:")
    for cond, s in summary.get("lag_by_condition", {}).items():
        print(f"  {cond:12s} n={s['n']:3d}  mean={s['mean_ms']:.1f}ms  std={s['std_ms']:.1f}ms")

    print("\n(f) Collapse latency (Milestone 12/13) -- source switch -> first genuine REJECT, "
          "pooled across every demo/panel.py session that recorded any:")
    cl = summary.get("collapse_latency", {})
    stats = cl.get("stats", {"n": 0})
    if stats["n"] > 0:
        print(f"  n_files={cl['n_files']}  n_switches={stats['n']}  "
              f"mean={stats['mean_s']*1000:.0f}ms  median={stats['median_s']*1000:.0f}ms  "
              f"p90={stats['p90_s']*1000:.0f}ms  min={stats['min_s']*1000:.0f}ms  max={stats['max_s']*1000:.0f}ms")
    else:
        print("  (no collapse-latency data recorded yet -- run demo/panel.py with a real "
              "source switch and let a REJECT fire)")

    print("\n(g) Per-lane ablation table (Milestone 11/13) -- among sessions that ACCEPTed with "
          "every lane present, fraction that stop doing so when a given lane is held out:")
    abl = summary.get("ablation")
    if abl and abl.get("full_accept_count"):
        print(f"  n_sessions={abl['n_sessions']}  full_accept_count={abl['full_accept_count']}")
        for lane, row in abl["lanes"].items():
            print(f"  {lane:10s} degraded={row['n_degraded_when_ablated']:3d}/"
                  f"{row['n_full_accept_sessions']:<3d}  fraction={row['fraction_degraded']:.2f}")
    else:
        print("  (not available -- run `python -m eval.ablate` first, or no session ACCEPTed "
              "with every lane present)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 5 analysis")
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)["eval"]

    logs_dir = Path(args.logs_dir) if args.logs_dir else REPO_ROOT / raw["logs_dir"]
    figures_dir = Path(args.figures_dir) if args.figures_dir else REPO_ROOT / raw["figures_dir"]
    ablation_json = REPO_ROOT / raw.get("ablation_json", "eval/ablation.json")
    collapse_latency_glob = raw.get("collapse_latency_glob", "collapse_latency_*.json")

    summary = run_analysis(logs_dir, figures_dir, raw["figure_dpi"], raw["attack_conditions"],
                            ablation_json=ablation_json, collapse_latency_glob=collapse_latency_glob)
    print_summary(summary)

    summary_path = REPO_ROOT / "eval" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary written to {summary_path}")
    print(f"Figures written to {figures_dir}")
