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


def plot_roc(fpr, tpr, auc, eer, eer_threshold, out_path: Path, dpi: int):
    fig, ax = plt.subplots(figsize=(6, 6))
    order = np.argsort(fpr)
    ax.plot(fpr[order], tpr[order], label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="chance")
    ax.plot(eer, 1 - eer, "o", color="red", label=f"EER={eer:.3f} @ thr={eer_threshold:.3f}")
    ax.set_xlabel("APCER / False Positive Rate (attack accepted)")
    ax.set_ylabel("True Positive Rate (bona fide accepted)")
    ax.set_title("ROC -- bona fide vs. attack (replay+swap)")
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

def run_analysis(logs_dir: Path, figures_dir: Path, dpi: int, attack_conditions: list) -> dict:
    records = load_sessions(logs_dir)
    summary: dict = {"n_sessions_total": len(records)}

    if not records:
        print(f"No session logs found in {logs_dir}. Nothing to analyse yet.")
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
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_roc(fpr, tpr, auc, eer, eer_thr, figures_dir / "roc_curve.png", dpi)
    else:
        summary["roc_eer"] = None
        print(f"Skipping ROC/EER: need >=1 bonafide and >=1 attack score "
              f"(have {len(bonafide_scores)} bonafide, {len(attack_scores)} attack, "
              f"after excluding insufficient_signal sessions).")

    # -- (d) bona fide breakdown, + snr_db (Milestone 3 addendum), also by skin_tone --
    bonafide_records = [r for r in records if r["condition"] == "bonafide"]
    summary["bonafide_breakdown"] = {
        field: breakdown_by(bonafide_records, field)
        for field in ("lighting", "distance_cm", "makeup", "glasses", "skin_tone")
    }

    # -- (e) lag at peak per condition --
    summary["lag_by_condition"] = lag_by_condition(records)

    # -- figures --
    figures_dir.mkdir(parents=True, exist_ok=True)
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
    print("\n(b)/(c) ROC / EER / APCER / BPCER / ACER (bona fide vs. replay+swap, "
          "excludes emitter_off control and insufficient_signal sessions):")
    if roc:
        print(f"  n_bonafide={roc['n_bonafide']} n_attack={roc['n_attack']}")
        print(f"  AUC={roc['auc']:.4f}  EER={roc['eer']:.4f} @ threshold={roc['eer_threshold']:.4f}")
        print(f"  APCER={roc['apcer']:.4f}  BPCER={roc['bpcer']:.4f}  ACER={roc['acer']:.4f}")
    else:
        print("  (skipped -- insufficient data)")

    print("\n(d) Bona fide score & snr_db breakdown (operating envelope):")
    for field, table in summary.get("bonafide_breakdown", {}).items():
        print(f"  by {field}:")
        for value, row in table.items():
            print(f"    {value:20s} n={row['n']:3d}  "
                  f"score={row['score_mean']:.3f}+/-{row['score_std']:.3f}  "
                  f"snr_db={row['snr_db_mean']:.2f}+/-{row['snr_db_std']:.2f}")

    print("\n(e) Lag at peak (ms) by condition:")
    for cond, s in summary.get("lag_by_condition", {}).items():
        print(f"  {cond:12s} n={s['n']:3d}  mean={s['mean_ms']:.1f}ms  std={s['std_ms']:.1f}ms")

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

    summary = run_analysis(logs_dir, figures_dir, raw["figure_dpi"], raw["attack_conditions"])
    print_summary(summary)

    summary_path = REPO_ROOT / "eval" / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull summary written to {summary_path}")
    print(f"Figures written to {figures_dir}")
