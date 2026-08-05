"""Cross-session offline scoring.

Constructs an injection-attack condition without any attack recordings: a
genuine session's measured trace, scored against a DIFFERENT session's
challenge. Physically, that is exactly what a video-injection attack looks
like to the verifier -- a real face's light response, evaluated against a
pseudo-random pattern it never actually saw, because the video was captured
(or, here, borrowed) under conditions that had nothing to do with the
challenge currently being asked. If the correlation collapses in this
condition the way it should, that's direct evidence the score is keyed to
"this session's specific challenge," not to "a real face was present" in
general -- which is the property an injection attack needs to defeat and
can't, whether the mismatched challenge came from a recording made an hour
ago or from another session already sitting in logs/.

No real emitter redraw log is persisted in session JSON logs (only
seed/chip_rate_hz/duration_s are, so the exact sequence is reproducible,
but not the true per-redraw timing jitter) -- so the challenge being scored
against is reconstructed as an idealised zero-order-hold schedule, sampled
finely enough to stand in for a real emitter log. The actual decision rule
(normalised cross-correlation, peak over non-negative lags only) is
imported directly from praesens.optical, not reimplemented, so every number
here comes from the exact same maths as the online path.
"""
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import numpy as np
import yaml

from praesens.challenge import Challenge
from praesens.optical import cross_correlate_lag_search
from eval.analyse import load_sessions, compute_eer, apcer_bpcer_acer, breakdown_by

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Reconstruction: a session log stores seed/chip_rate_hz/duration_s, which
# fully and deterministically reproduce its challenge (Challenge.__post_init__
# recomputes the LFSR order from those alone -- it isn't stored separately).
# No real emitter log survives, so one is synthesised at a fine, fixed
# sample rate matching the ~90-130 Hz redraw rates measured live -- well
# above the chip rate, so this idealised schedule is a faithful stand-in.
# ---------------------------------------------------------------------------

def reconstruct_challenge(record: dict) -> Challenge | None:
    try:
        return Challenge(chip_rate_hz=record["chip_rate_hz"], duration_s=record["duration_s"],
                          seed=record["seed"])
    except KeyError:
        return None  # older log, collected before chip_rate_hz/duration_s were added to the schema


def build_synthetic_emitter_log(challenge: Challenge, sample_rate_hz: float) -> list:
    """Formatted exactly like Emitter.get_log() so cross_correlate_lag_search
    / resample_emitted need no changes at all -- imported, not reimplemented."""
    t = np.arange(0, challenge.duration_s, 1.0 / sample_rate_hz)
    vals = challenge.value_at(t)
    return [{"t": float(tt), "chip_value": int(v)} for tt, v in zip(t, vals)]


def score_pair(measured: list, timestamps: list, challenge: Challenge,
               lag_max_ms: float, lag_step_ms: float, sample_rate_hz: float,
               session_start: float | None = None):
    """Same decision rule as the online path: praesens.optical.
    cross_correlate_lag_search, imported directly. It already only searches
    lags in [0, lag_max_ms] -- peak over non-negative lags only, a response
    cannot precede its stimulus -- so no change was needed there.

    Timestamps are rebased to "elapsed time since the challenge started"
    before scoring, since each session's stored timestamps are absolute
    time.perf_counter() values from an independent process run with no
    shared origin across sessions. If session_start (the record's
    start_time_perf_counter) is available, rebasing is exact: ts -
    session_start is precisely elapsed-since-challenge-began, matching what
    the online path itself uses. Older logs collected before that field
    existed fall back to ts - ts[0] (elapsed since the first VALID,
    face-detected frame) -- an approximation that's only exact if the
    subject was already being tracked at the very start of the session; any
    warm-up gap before the first detected face shows up as a corresponding
    UNDER-estimate of the true lag. Scores are essentially unaffected either
    way (Pearson correlation only cares about relative alignment, and the
    lag search still finds the same relative peak), but reported lag_ms
    values from the fallback path should not be treated as exactly
    comparable to the online path's lag_ms for that session.

    Traces are truncated to the overlap with the scored-against challenge's
    own duration ("truncate paired traces to the shorter length"), so the
    comparison never runs into the challenge's clamped, degenerate tail
    beyond its own recorded length.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    if len(ts) == 0:
        return None
    ts = ts - (session_start if session_start is not None else ts[0])
    m = np.asarray(measured, dtype=np.float64)

    mask = ts < challenge.duration_s
    ts_trunc, m_trunc = ts[mask], m[mask]
    if len(m_trunc) < 5:
        return None

    emitter_log = build_synthetic_emitter_log(challenge, sample_rate_hz)
    score, lag_ms, *_ = cross_correlate_lag_search(m_trunc, ts_trunc, emitter_log, lag_max_ms, lag_step_ms)
    return float(score), float(lag_ms), int(len(m_trunc))


# ---------------------------------------------------------------------------
# Trial construction
# ---------------------------------------------------------------------------

def build_trials(records: list, lag_max_ms: float, lag_step_ms: float, sample_rate_hz: float) -> list:
    bonafide = [r for r in records if r["condition"] == "bonafide" and r.get("timestamps")]
    emitter_off = [r for r in records if r["condition"] == "emitter_off" and r.get("timestamps")]

    trials = []
    skipped = []

    def add_self_trial(r: dict, condition_label: str):
        challenge = reconstruct_challenge(r)
        if challenge is None:
            skipped.append(r["session"])
            return
        result = score_pair(r["trace_measured"], r["timestamps"], challenge, lag_max_ms, lag_step_ms,
                             sample_rate_hz, session_start=r.get("start_time_perf_counter"))
        if result is None:
            return
        score, lag_ms, n = result
        trials.append({
            "trial": f"{r['session']}_vs_self", "condition": condition_label,
            "session_measured": r["session"], "session_challenge": r["session"],
            "score": score, "lag_ms": lag_ms, "n_samples": n, "meta": r["meta"],
        })

    for r in bonafide:
        add_self_trial(r, "bonafide")
    for r in emitter_off:
        add_self_trial(r, "emitter_off")

    # xsession: every ORDERED pair of DISTINCT bonafide sessions -- "genuine
    # sessions" means real-person-and-pattern-on sessions specifically;
    # emitter_off sessions have no real challenge-following relationship to
    # cross-pair in the same sense and are handled as their own condition above.
    for a, b in permutations(bonafide, 2):
        challenge = reconstruct_challenge(b)
        if challenge is None:
            continue
        # Rebasing uses A's own start_time -- A's timestamps are what's
        # being rebased, regardless of whose challenge (B's) they're scored
        # against.
        result = score_pair(a["trace_measured"], a["timestamps"], challenge, lag_max_ms, lag_step_ms,
                             sample_rate_hz, session_start=a.get("start_time_perf_counter"))
        if result is None:
            continue
        score, lag_ms, n = result
        trials.append({
            "trial": f"{a['session']}_vs_{b['session']}", "condition": "xsession",
            "session_measured": a["session"], "session_challenge": b["session"],
            "score": score, "lag_ms": lag_ms, "n_samples": n, "meta": a["meta"],
        })

    if skipped:
        print(f"NOTE: skipped {len(skipped)} session(s) missing chip_rate_hz/duration_s/seed "
              f"(collected before those fields existed in the schema): {skipped}")

    return trials


def reconciliation_check(trials: list, records: list) -> None:
    """Sanity check: the offline-recomputed bonafide score (own reconstructed
    idealised challenge) vs. the score praesens/session.py actually logged
    online (real emitter redraw log). They approximate the same quantity
    through different signal sources, so small differences are expected --
    a LARGE divergence would mean the reconstruction here doesn't actually
    match the online decision rule despite reusing the same function, which
    would undermine every table below."""
    by_session = {r["session"]: r for r in records}
    diffs = []
    for t in trials:
        if t["condition"] != "bonafide":
            continue
        online_score = by_session.get(t["session_measured"], {}).get("score")
        if online_score is None:
            continue
        diffs.append(abs(t["score"] - online_score))
    if not diffs:
        return
    diffs = np.array(diffs)
    print(f"\nReconciliation check (offline-recomputed vs. online-logged bonafide score): "
          f"mean|diff|={diffs.mean():.3f}, max|diff|={diffs.max():.3f}")
    if diffs.max() > 0.15:
        print("WARNING: some sessions diverge notably between online and offline scoring -- "
              "inspect before trusting Table I/II (see build_synthetic_emitter_log's "
              "sample_rate_hz, or an unusually low/variable capture FPS for that session).")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

CONDITIONS = ("bonafide", "xsession", "emitter_off")


def table1(trials: list) -> dict:
    print("\nTABLE I -- score distribution per condition")
    print(f"{'condition':<12s} {'n':>5s} {'mean':>8s} {'SD':>8s} {'min':>8s} {'max':>8s} {'median':>8s}")
    out = {}
    for cond in CONDITIONS:
        scores = np.array([t["score"] for t in trials if t["condition"] == cond])
        if len(scores) == 0:
            print(f"{cond:<12s} {0:5d}  (no trials)")
            out[cond] = {"n": 0}
            continue
        row = {"n": int(len(scores)), "mean": float(scores.mean()), "std": float(scores.std()),
               "min": float(scores.min()), "max": float(scores.max()), "median": float(np.median(scores))}
        print(f"{cond:<12s} {row['n']:5d} {row['mean']:8.3f} {row['std']:8.3f} "
              f"{row['min']:8.3f} {row['max']:8.3f} {row['median']:8.3f}")
        out[cond] = row
    return out


def table2(trials: list) -> dict | None:
    print("\nTABLE II -- ROC: bonafide (positive) vs xsession+emitter_off (negative)")
    positive = np.array([t["score"] for t in trials if t["condition"] == "bonafide"])
    negative = np.array([t["score"] for t in trials if t["condition"] in ("xsession", "emitter_off")])
    if len(positive) == 0 or len(negative) == 0:
        print("  (insufficient data -- need at least one bonafide and one xsession/emitter_off trial)")
        return None
    eer, eer_thr, auc, fpr, tpr = compute_eer(positive, negative)
    apcer, bpcer, acer = apcer_bpcer_acer(positive, negative, eer_thr)
    print(f"  n_positive(bonafide)={len(positive)}  n_negative(xsession+emitter_off)={len(negative)}")
    print(f"  AUC    = {auc:.4f}")
    print(f"  EER    = {eer:.4f}  @ threshold={eer_thr:.4f}")
    print(f"  APCER  = {apcer:.4f}")
    print(f"  BPCER  = {bpcer:.4f}")
    print(f"  ACER   = {acer:.4f}")
    return {"n_positive": int(len(positive)), "n_negative": int(len(negative)), "auc": auc,
            "eer": eer, "eer_threshold": eer_thr, "apcer": apcer, "bpcer": bpcer, "acer": acer}


META_FIELDS = ("lighting", "distance_cm", "makeup", "glasses", "skin_tone", "subject")


def table3(trials: list) -> dict:
    print("\nTABLE III -- genuine (bonafide) score by meta field")
    bonafide_as_records = [{"score": t["score"], "meta": t["meta"]}
                            for t in trials if t["condition"] == "bonafide"]
    out = {}
    for field in META_FIELDS:
        breakdown = breakdown_by(bonafide_as_records, field, value_fields=("score",))
        out[field] = breakdown
        print(f"  by {field}:")
        for value, row in breakdown.items():
            print(f"    {value:20s} n={row['n']:3d}  score={row['score_mean']:.3f} +/- {row['score_std']:.3f}")
    return out


def lag_table(trials: list) -> dict:
    print("\nLag at peak (ms), mean/SD per condition")
    out = {}
    for cond in CONDITIONS:
        lags = np.array([t["lag_ms"] for t in trials if t["condition"] == cond])
        if len(lags) == 0:
            continue
        row = {"n": int(len(lags)), "mean_ms": float(lags.mean()), "std_ms": float(lags.std())}
        out[cond] = row
        print(f"  {cond:<12s} n={row['n']:4d}  mean={row['mean_ms']:.1f}ms  SD={row['std_ms']:.1f}ms")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-session offline injection-attack scoring")
    parser.add_argument("--logs-dir", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    with open(REPO_ROOT / "config.yaml") as f:
        raw = yaml.safe_load(f)

    logs_dir = Path(args.logs_dir) if args.logs_dir else REPO_ROOT / raw["eval"]["logs_dir"]
    xcfg = raw.get("cross_session", {})
    output_json = Path(args.output_json) if args.output_json else REPO_ROOT / xcfg.get(
        "output_json", "eval/cross_session.json")

    records = load_sessions(logs_dir)
    print(f"Loaded {len(records)} session logs from {logs_dir}")
    n_bonafide = sum(1 for r in records if r["condition"] == "bonafide")
    n_off = sum(1 for r in records if r["condition"] == "emitter_off")
    print(f"  bonafide={n_bonafide}  emitter_off={n_off}")
    if n_bonafide < 2:
        print("WARNING: fewer than 2 bonafide sessions -- xsession needs at least 2 to form any pairs.")

    trials = build_trials(
        records,
        lag_max_ms=raw["optical"]["lag_search_max_ms"],
        lag_step_ms=raw["optical"]["lag_step_ms"],
        sample_rate_hz=xcfg.get("synthetic_emitter_sample_rate_hz", 100.0),
    )
    print(f"\nBuilt {len(trials)} trials "
          f"(bonafide={sum(1 for t in trials if t['condition']=='bonafide')}, "
          f"xsession={sum(1 for t in trials if t['condition']=='xsession')}, "
          f"emitter_off={sum(1 for t in trials if t['condition']=='emitter_off')})")

    reconciliation_check(trials, records)

    t1 = table1(trials)
    t2 = table2(trials)
    t3 = table3(trials)
    lags = lag_table(trials)

    summary = {"n_sessions_loaded": len(records), "n_trials": len(trials),
               "table1": t1, "table2": t2, "table3": t3, "lag_by_condition": lags}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"summary": summary, "trials": trials}, f, indent=2)
    print(f"\nFull per-trial results written to {output_json}")


if __name__ == "__main__":
    main()
