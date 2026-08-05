"""Interactive batch session collector.

Typing --lighting/--distance-cm/--makeup/--glasses/--subject/--skin-tone by
hand on PowerShell for every one of dozens of sessions is exactly the kind
of repetitive, typo-prone task worth automating away. This prompts once for
the things that don't change within a sitting (subject id, skin tone), then
loops asking only what changes between sessions, running one real 20s
session each time via praesens.session.run_one_session and printing the
score immediately -- so a bad setup (wrong distance, bad lighting) shows up
after session 1, not after session 15.

emitter_off is handled entirely in memory: run_one_session() builds a fresh
EmitterConfig from the loaded config dict on every call and only ever sets
emitter_enabled=False on that new, local object when condition=="emitter_off"
-- it never writes to config.yaml and never mutates the dict it was given.
Calling it repeatedly with different --condition values in the same process
is already safe; there is nothing to restore because nothing was changed.
"""
from __future__ import annotations

import argparse

from praesens.session import run_one_session, load_config, VALID_CONDITIONS

LIGHTING_OPTIONS = ["normal", "dim", "backlit", "side-lit"]
MAKEUP_OPTIONS = ["none", "foundation_powder"]
GLASSES_OPTIONS = ["with", "without"]


def prompt_choice(label: str, options: list, default: str) -> str:
    opts_str = "/".join(options)
    while True:
        raw = input(f"{label} [{opts_str}] (default {default}): ").strip()
        if not raw:
            return default
        if raw in options:
            return raw
        print(f"  please enter one of: {opts_str}")


def prompt_float(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} (default {default}): ").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print("  please enter a number")


def prompt_int(label: str, lo: int, hi: int) -> int | None:
    while True:
        raw = input(f"{label} [{lo}-{hi}, blank to skip]: ").strip()
        if not raw:
            return None
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        print(f"  please enter an integer between {lo} and {hi}, or leave blank")


def collect_one(condition: str, subject: str, skin_tone: int | None, raw_config: dict) -> dict:
    lighting = prompt_choice("lighting", LIGHTING_OPTIONS, default="normal")
    distance_cm = prompt_float("distance_cm", default=60.0)
    makeup = prompt_choice("makeup", MAKEUP_OPTIONS, default="none")
    glasses = prompt_choice("glasses", GLASSES_OPTIONS, default="without")

    meta = {
        "lighting": lighting, "distance_cm": distance_cm, "makeup": makeup,
        "glasses": glasses, "subject": subject, "skin_tone": skin_tone,
    }

    duration_s = raw_config["challenge"]["duration_s"]
    print(f"\nRunning {condition} session ({duration_s}s)... look at the screen.")
    record, out_path = run_one_session(condition, meta, raw_config=raw_config)

    print(f"score={record['score']:.3f}  lag_ms={record['lag_ms']:.1f}  "
          f"snr_db={record['snr_db']:.2f}  insufficient_signal={record['insufficient_signal']}")
    if condition == "bonafide" and record["score"] < 0.2:
        print("  ^ LOW score for a bonafide session -- check face is in frame, lighting, "
              "and distance before running more (see scripts/diagnose.py if this persists).")
    for w in record["warnings"]:
        print(f"  warning: {w}")
    print(f"saved to {out_path}")
    return record


def run_quick(subject: str, skin_tone: int | None, raw_config: dict) -> None:
    print("\n--quick: running 1 bonafide + 1 emitter_off back to back as a sanity check.\n")
    print("=== bonafide ===")
    bonafide_record = collect_one("bonafide", subject, skin_tone, raw_config)
    print("\n=== emitter_off (control) ===")
    off_record = collect_one("emitter_off", subject, skin_tone, raw_config)

    print("\n--- quick sanity check ---")
    print(f"bonafide score:    {bonafide_record['score']:.3f}")
    print(f"emitter_off score: {off_record['score']:.3f}")
    collapsed = off_record["score"] < 0.3 and off_record["score"] < 0.5 * max(bonafide_record["score"], 1e-9)
    if collapsed:
        print("OK: emitter_off collapsed relative to bonafide.")
    else:
        print("WARNING: emitter_off did not clearly collapse relative to bonafide -- "
              "run scripts/diagnose.py before trusting the rest of the corpus.")


def main():
    parser = argparse.ArgumentParser(description="Interactive batch session collector")
    parser.add_argument("--condition", choices=sorted(VALID_CONDITIONS - {"replay", "swap"}),
                         default="bonafide", help="condition for the looped collection run")
    parser.add_argument("--quick", action="store_true",
                         help="run 1 bonafide + 1 emitter_off back to back and print both scores")
    args = parser.parse_args()

    raw_config = load_config()

    print("PRAESENS interactive session collector")
    subject = input("subject id: ").strip() or "unknown"
    skin_tone = prompt_int("skin tone (Fitzpatrick scale)", 1, 6)

    if args.quick:
        run_quick(subject, skin_tone, raw_config)
        return

    n = 0
    while True:
        n += 1
        print(f"\n=== session {n} ({args.condition}) ===")
        collect_one(args.condition, subject, skin_tone, raw_config)
        cont = input("\nContinue with another session? [Y/n]: ").strip().lower()
        if cont == "n":
            break

    print(f"\nDone. {n} session(s) collected for subject={subject}.")


if __name__ == "__main__":
    main()
