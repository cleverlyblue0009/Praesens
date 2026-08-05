"""Single hands-off batch runner: sanity gate, then the full scripted
corpus, then analysis -- one command, spoken-style prompts, paper tables
at the end.

The sanity gate exists because 15 sessions of garbage data (camera
misconfigured, subject out of frame, exposure not locked) costs real time
to collect and is worse than useless -- it has to be diagnosed AFTER the
fact instead of before. One bonafide + one emitter_off session, scored
against the same 0.7/0.3 thresholds as the rest of this repo's controls,
catches that in under a minute instead of twenty.

Every session (gate and corpus) goes through the exact same
praesens.session.run_one_session() used everywhere else in this repo --
no new scoring path, no shortcuts, just orchestration around it.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from praesens.session import run_one_session, load_config
from scripts.collect import prompt_int
from eval.analyse import load_sessions
from eval.cross_session import build_trials, reconciliation_check, table1, table2, table3, lag_table

REPO_ROOT = Path(__file__).resolve().parent.parent

GATE_META = {"lighting": "normal", "distance_cm": 60.0, "makeup": "none", "glasses": "without"}

# Grouped so setup changes as few times as possible -- each block is one
# physical setup, run to completion before moving to the next.
CORPUS_PLAN = [
    {
        "instruction": "SIT AT NORMAL DISTANCE (~60cm), NORMAL ROOM LIGHTING",
        "condition": "bonafide",
        "meta": {"lighting": "normal", "distance_cm": 60.0, "makeup": "none", "glasses": "without"},
        "count": 4,
    },
    {
        "instruction": "TURN THE ROOM LIGHTS OFF (or dim them right down)",
        "condition": "bonafide",
        "meta": {"lighting": "dim", "distance_cm": 60.0, "makeup": "none", "glasses": "without"},
        "count": 2,
    },
    {
        "instruction": "LIGHTS BACK TO NORMAL. SIT CLOSER -- ARM'S LENGTH, ~40cm",
        "condition": "bonafide",
        "meta": {"lighting": "normal", "distance_cm": 40.0, "makeup": "none", "glasses": "without"},
        "count": 1,
    },
    {
        "instruction": "SIT FARTHER BACK, ~80cm",
        "condition": "bonafide",
        "meta": {"lighting": "normal", "distance_cm": 80.0, "makeup": "none", "glasses": "without"},
        "count": 1,
    },
    {
        "instruction": "BACK TO ~60cm, NORMAL LIGHTING. PUT ON FOUNDATION + POWDER -- take your time",
        "condition": "bonafide",
        "meta": {"lighting": "normal", "distance_cm": 60.0, "makeup": "foundation_powder", "glasses": "without"},
        "count": 2,
    },
    {
        "instruction": "PUT ON GLASSES (makeup optional from here on)",
        "condition": "bonafide",
        "meta": {"lighting": "normal", "distance_cm": 60.0, "makeup": "none", "glasses": "with"},
        "count": 2,
    },
    {
        "instruction": "REMOVE GLASSES. Sit normally -- the light pattern will be OFF for "
                        "these, that's expected",
        "condition": "emitter_off",
        "meta": {"lighting": "normal", "distance_cm": 60.0, "makeup": "none", "glasses": "without"},
        "count": 3,
    },
]

BONAFIDE_LOW_SCORE_FLOOR = 0.4
GATE_GENUINE_FLOOR = 0.7
GATE_OFF_CEILING = 0.3


def flatten_plan() -> list:
    """One entry per actual session, each carrying a stable corpus_slot id
    (plan position, e.g. "corpus_003") used by --resume to detect what's
    already been collected -- independent of the random session id/uuid
    each run_one_session() call generates."""
    slots = []
    n = 0
    for block in CORPUS_PLAN:
        for i in range(block["count"]):
            n += 1
            slots.append({
                "corpus_slot": f"corpus_{n:03d}",
                "instruction": block["instruction"],
                "condition": block["condition"],
                "meta_overrides": block["meta"],
                "block_run": i + 1,
                "block_count": block["count"],
            })
    return slots


def print_big(text: str) -> None:
    print("\n" + "#" * 74)
    for line in text.split("\n"):
        print(f"#  {line}")
    print("#" * 74)


def countdown() -> None:
    for n in (3, 2, 1):
        print(n, flush=True)
        time.sleep(1)
    print("GO", flush=True)


def already_done_slots(logs_dir: Path) -> set:
    done = set()
    if not logs_dir.exists():
        return done
    for path in logs_dir.glob("*.json"):
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        slot = d.get("meta", {}).get("corpus_slot")
        if slot:
            done.add(slot)
    return done


def run_session_safe(condition: str, meta: dict, raw_config: dict, label: str) -> dict | None:
    """Never lets one session's crash abort the batch: catches, reports,
    returns None so the caller can record the failure and move on."""
    try:
        record, out_path = run_one_session(condition, meta, raw_config=raw_config)
        return record
    except Exception:
        print(f"\nCRASHED during {label}:")
        traceback.print_exc()
        print("Continuing with the next session.\n")
        return None


def print_score_line(record: dict) -> None:
    fps = record.get("measured_fps", float("nan"))
    chip_rate = record.get("chip_rate_hz", float("nan"))
    samples_per_chip = fps / chip_rate if chip_rate else float("nan")
    print(f"score={record['score']:.3f}  lag_ms={record['lag_ms']:.1f}  "
          f"snr_db={record['snr_db']:.2f}  fps={fps:.1f}  samples/chip={samples_per_chip:.1f}  "
          f"insufficient_signal={record['insufficient_signal']}")


# ---------------------------------------------------------------------------
# Sanity gate
# ---------------------------------------------------------------------------

def run_gate(subject: str, skin_tone: int | None, raw_config: dict) -> bool:
    print_big("SANITY GATE\nSIT AT NORMAL DISTANCE (~60cm), NORMAL ROOM LIGHTING")
    input("Press ENTER when ready...")
    print("\n--- gate session 1/2: bonafide ---")
    countdown()
    meta = dict(GATE_META, subject=subject, skin_tone=skin_tone, corpus_slot="gate_bonafide")
    genuine = run_session_safe("bonafide", meta, raw_config, "gate/bonafide")
    if genuine is not None:
        print_score_line(genuine)

    print_big("SAME SETUP. This one has the light pattern OFF -- just sit normally")
    input("Press ENTER when ready...")
    print("\n--- gate session 2/2: emitter_off ---")
    countdown()
    meta = dict(GATE_META, subject=subject, skin_tone=skin_tone, corpus_slot="gate_emitter_off")
    off = run_session_safe("emitter_off", meta, raw_config, "gate/emitter_off")
    if off is not None:
        print_score_line(off)

    print("\n" + "=" * 74)
    print("SANITY GATE RESULT")
    print("=" * 74)
    if genuine is None or off is None:
        print("FAIL - one or both gate sessions crashed. Do not collect more.")
        return False

    print(f"{'condition':<14s} {'score':>8s} {'fps':>8s} {'smpl/chip':>10s} {'snr_db':>8s}")
    for label, r in (("bonafide", genuine), ("emitter_off", off)):
        fps = r.get("measured_fps", float("nan"))
        chip_rate = r.get("chip_rate_hz", float("nan"))
        spc = fps / chip_rate if chip_rate else float("nan")
        print(f"{label:<14s} {r['score']:8.3f} {fps:8.1f} {spc:10.1f} {r['snr_db']:8.2f}")

    passed = genuine["score"] >= GATE_GENUINE_FLOOR and off["score"] <= GATE_OFF_CEILING
    if passed:
        print(f"\nPASS - genuine={genuine['score']:.3f} >= {GATE_GENUINE_FLOOR} and "
              f"emitter_off={off['score']:.3f} <= {GATE_OFF_CEILING} - continuing to full corpus")
    else:
        print(f"\nFAIL - genuine={genuine['score']:.3f} (need >= {GATE_GENUINE_FLOOR}), "
              f"emitter_off={off['score']:.3f} (need <= {GATE_OFF_CEILING}) - stopping. "
              f"Do not collect more.")
        for label, r in (("bonafide", genuine), ("emitter_off", off)):
            for w in r["warnings"]:
                print(f"  [{label}] warning: {w}")
    return passed


# ---------------------------------------------------------------------------
# Scripted corpus
# ---------------------------------------------------------------------------

def run_corpus(subject: str, skin_tone: int | None, raw_config: dict, resume: bool) -> dict:
    slots = flatten_plan()
    total = len(slots)

    skip = already_done_slots(REPO_ROOT / raw_config["eval"]["logs_dir"]) if resume else set()
    if skip:
        print(f"\n--resume: {len(skip)} session(s) already present in logs/, skipping those.")

    flagged_low_score = []
    crashed = []
    completed = 0
    last_instruction = None

    for slot in slots:
        if slot["corpus_slot"] in skip:
            completed += 1
            continue

        if slot["instruction"] != last_instruction:
            print_big(slot["instruction"])
            last_instruction = slot["instruction"]
        else:
            print(f"\n(same setup)")

        print(f"Session {slot['corpus_slot'][-3:].lstrip('0') or '0'}/{total} overall -- "
              f"block run {slot['block_run']}/{slot['block_count']} -- condition={slot['condition']}")
        input("Press ENTER when ready...")
        countdown()

        meta = dict(slot["meta_overrides"], subject=subject, skin_tone=skin_tone,
                    corpus_slot=slot["corpus_slot"])
        record = run_session_safe(slot["condition"], meta, raw_config, slot["corpus_slot"])

        if record is None:
            crashed.append(slot["corpus_slot"])
            continue

        print_score_line(record)
        completed += 1

        if slot["condition"] == "bonafide" and record["score"] < BONAFIDE_LOW_SCORE_FLOOR:
            print(f"  ^ WARNING: low bonafide score ({record['score']:.3f} < "
                  f"{BONAFIDE_LOW_SCORE_FLOOR}) -- flagged, continuing.")
            flagged_low_score.append({"corpus_slot": slot["corpus_slot"], "score": record["score"],
                                       "meta": slot["meta_overrides"]})

    print(f"\nCorpus collection done: {completed}/{total} sessions present "
          f"({len(crashed)} crashed this run, {len(skip)} skipped via --resume).")
    if crashed:
        print(f"  crashed: {crashed}")
    if flagged_low_score:
        print(f"  flagged low-score bonafide sessions:")
        for f in flagged_low_score:
            print(f"    {f['corpus_slot']}: score={f['score']:.3f} meta={f['meta']}")

    return {"total": total, "completed": completed, "crashed": crashed, "flagged_low_score": flagged_low_score}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis_and_write_results(raw_config: dict) -> tuple[Path, dict]:
    logs_dir = REPO_ROOT / raw_config["eval"]["logs_dir"]
    xcfg = raw_config.get("cross_session", {})
    output_json = REPO_ROOT / xcfg.get("output_json", "eval/cross_session.json")
    results_path = REPO_ROOT / "eval" / "RESULTS.txt"

    buf = io.StringIO()
    with redirect_stdout(buf):
        records = load_sessions(logs_dir)
        print(f"Loaded {len(records)} session logs from {logs_dir}")
        n_bonafide = sum(1 for r in records if r["condition"] == "bonafide")
        n_off = sum(1 for r in records if r["condition"] == "emitter_off")
        print(f"  bonafide={n_bonafide}  emitter_off={n_off}")

        trials = build_trials(
            records,
            lag_max_ms=raw_config["optical"]["lag_search_max_ms"],
            lag_step_ms=raw_config["optical"]["lag_step_ms"],
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

    captured = buf.getvalue()
    print(captured)
    with open(results_path, "w") as f:
        f.write(captured)

    return results_path, summary


def one_line_summary(summary: dict) -> str:
    n_parts = []
    for cond, row in summary.get("table1", {}).items():
        n_parts.append(f"{cond} n={row.get('n', 0)}")
    n_str = ", ".join(n_parts)
    t2 = summary.get("table2")
    if t2:
        return f"SUMMARY: {n_str} | EER={t2['eer']:.4f} AUC={t2['auc']:.4f}"
    return f"SUMMARY: {n_str} | EER/AUC unavailable (insufficient data)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hands-off batch corpus collection + analysis")
    parser.add_argument("--skip-gate", action="store_true", help="bypass the sanity gate")
    parser.add_argument("--resume", action="store_true",
                         help="skip corpus sessions already present in logs/ (matched by corpus_slot)")
    args = parser.parse_args()

    raw_config = load_config()

    print("PRAESENS hands-off corpus collection")
    subject = input("subject id: ").strip() or "unknown"
    skin_tone = prompt_int("skin tone (Fitzpatrick scale)", 1, 6)

    if not args.skip_gate:
        passed = run_gate(subject, skin_tone, raw_config)
        if not passed:
            print("\nStopping here per the sanity gate result. Fix the setup and rerun "
                  "(use --skip-gate once you've passed to avoid repeating it).")
            sys.exit(1)
    else:
        print("\n--skip-gate: bypassing the sanity gate.")

    run_corpus(subject, skin_tone, raw_config, resume=args.resume)

    print("\nRunning analysis (eval/cross_session.py)...")
    results_path, summary = run_analysis_and_write_results(raw_config)

    print(f"\nResults written to: {results_path}")
    print(one_line_summary(summary))


if __name__ == "__main__":
    main()
