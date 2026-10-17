"""Keystroke injection, for attack testing ONLY.

Types a string with pynput's keyboard Controller -- the OS sees ordinary
key-down/key-up events, so the typing lane (praesens/typing.py, which scores
keystroke TIMING and phrase match) can be satisfied by a script. What a script
cannot fake is the physical strike: no finger hits the keyboard, so the
microphone hears nothing at the host's key-down-to-impact lag and the pressure
lane (praesens/pressure.py) collapses. That contrast is the point of this tool.

Timing is deliberately human-like -- lognormal inter-key intervals around a
target WPM, a longer pause after a space, and a realistic dwell (hold) time --
so that a pass here cannot be dismissed as "the bot typed at a robotic
constant rate, which typing.keystroke_normality_score would have caught
anyway" (it checks the mean inter-key interval against
typing.min_inter_key_s / max_inter_key_s).

Usage (the target window must have keyboard focus -- e.g. Notepad):
    python scripts/inject_keys.py --text "quick brown fox" --wpm 55 --delay 2
    python scripts/inject_keys.py --text-file phrase.txt --wpm 40
"""
from __future__ import annotations

import argparse
import random
import time


def human_intervals(text: str, wpm: float, rng: random.Random, sigma: float = 0.35) -> list:
    """Per-character inter-key intervals (seconds), lognormal around the mean
    implied by `wpm` (5 characters per word is the standard definition), with a
    longer pause after spaces, the way real typing bursts between words."""
    mean_interval = 60.0 / (wpm * 5.0)
    out = []
    for ch in text:
        iv = rng.lognormvariate(0.0, sigma) * mean_interval
        if ch == " ":
            iv *= 1.6
        out.append(min(max(iv, 0.02), 2.0))
    return out


def inject(text: str, wpm: float = 55.0, delay_s: float = 2.0, seed: int | None = None,
           dwell_ms: tuple = (55.0, 95.0), controller=None) -> list:
    """Types `text` into the focused window. Returns the perf_counter time of
    every key-down, so a caller can line them up against what a lane saw."""
    from pynput.keyboard import Controller

    rng = random.Random(seed)
    kb = controller if controller is not None else Controller()
    intervals = human_intervals(text, wpm, rng)

    if delay_s > 0:
        time.sleep(delay_s)

    downs = []
    for ch, iv in zip(text, intervals):
        downs.append(time.perf_counter())
        kb.press(ch)
        time.sleep(rng.uniform(*dwell_ms) / 1000.0)
        kb.release(ch)
        time.sleep(max(0.0, iv - rng.uniform(*dwell_ms) / 1000.0))
    return downs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Type a string into the focused window (attack tool)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="text to type")
    src.add_argument("--text-file", help="file whose contents to type")
    parser.add_argument("--wpm", type=float, default=55.0, help="target words per minute")
    parser.add_argument("--delay", type=float, default=2.0, help="seconds to wait before typing")
    parser.add_argument("--repeat", type=int, default=1, help="type the text this many times")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else open(args.text_file).read().strip()
    print(f"injecting {len(text) * args.repeat} chars at ~{args.wpm:.0f} wpm "
          f"after {args.delay:.1f}s -- focus the target window now")
    n = 0
    for r in range(args.repeat):
        downs = inject(text, wpm=args.wpm, delay_s=args.delay if r == 0 else 0.3, seed=args.seed)
        n += len(downs)
    print(f"injected {n} key-downs")


if __name__ == "__main__":
    main()
