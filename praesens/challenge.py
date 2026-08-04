"""Milestone 1: per-session challenge synthesiser.

Generates the pseudo-random +/-1 light pattern the emitter displays. It is a
maximum-length sequence (m-sequence) produced by a Fibonacci LFSR with a
primitive feedback polynomial: m-sequences have near-ideal (impulse-like)
autocorrelation, so when the optical lane later cross-correlates the emitted
pattern against measured facial luminance, a genuine reflection produces a
sharp, unambiguous peak instead of a broad, ambiguous one, and periodic
interference such as 50/60 Hz mains flicker does not accidentally line up
with the pattern and create a false peak. Each session draws a fresh random
seed so an attacker cannot pre-record a response to a pattern they've seen
before; the seed alone is sufficient to reproduce the exact sequence, which
is what makes a session's log independently checkable.
"""
from __future__ import annotations

import json
import math
import secrets
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

def _step(order: int, taps: tuple, state: int) -> int:
    feedback = 0
    for tap in taps:
        feedback ^= (state >> (tap - 1)) & 1
    return (state >> 1) | (feedback << (order - 1))


def _measured_period(order: int, taps: tuple) -> int:
    """Actual cycle length of this (order, taps) pair under our specific
    shift/feedback convention, found by simulation rather than assumed from
    a published table -- tap tables are convention-dependent (which end
    shifts, where feedback is inserted), and a mismatched convention
    silently produces a short, non-maximal cycle instead of an error."""
    period_full = (1 << order) - 1
    state = 1
    start = state
    for steps in range(1, period_full + 1):
        state = _step(order, taps, state)
        if state == start:
            return steps
    return -1


_TAPS_CACHE: dict[int, tuple] = {}


def _find_maximal_taps(order: int) -> tuple:
    """Search for a tap set giving the full period 2**order - 1, verified
    by simulation. Tries two-tap combinations first (covers any primitive
    trinomial under our convention), then falls back to four-tap searches."""
    if order in _TAPS_CACHE:
        return _TAPS_CACHE[order]

    period_full = (1 << order) - 1
    for k in range(1, order):
        taps = (order, k)
        if _measured_period(order, taps) == period_full:
            _TAPS_CACHE[order] = taps
            return taps

    from itertools import combinations
    for combo in combinations(range(1, order), 3):
        taps = (order,) + combo
        if _measured_period(order, taps) == period_full:
            _TAPS_CACHE[order] = taps
            return taps

    raise ValueError(f"no maximal-length tap set found for order={order}")


def _choose_order(n_chips: int) -> int:
    """Smallest LFSR order whose full period covers n_chips chips."""
    order = 2
    while (1 << order) - 1 < n_chips:
        order += 1
        if order > 32:
            raise ValueError(f"n_chips={n_chips} is impractically large for an LFSR order")
    return order


def _lfsr_bits(order: int, seed: int, n_bits: int) -> np.ndarray:
    """Fibonacci LFSR bit sequence, seed as the nonzero initial register state."""
    taps = _find_maximal_taps(order)
    period = (1 << order) - 1
    state = seed & period
    if state == 0:
        raise ValueError("LFSR seed must be nonzero (mod 2**order - 1)")

    bits = np.empty(n_bits, dtype=np.int8)
    for i in range(n_bits):
        bits[i] = state & 1
        state = _step(order, taps, state)
    return bits


@dataclass
class Challenge:
    """One session's emitted light pattern: a +/-1 m-sequence at a fixed
    chip rate, fully determined by (order, seed, chip_rate_hz, duration_s)
    so it can be exactly regenerated from a saved seed."""

    chip_rate_hz: float = 5.0
    duration_s: float = 20.0
    order: int = field(default=None)  # type: ignore[assignment]
    seed: int = field(default=None)   # type: ignore[assignment]
    chips: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    loop: bool = False  # if True, value_at() wraps past duration_s instead of clamping
                         # (the live demo runs indefinitely, unlike a fixed 20s scored session)

    def __post_init__(self):
        n_chips = round(self.duration_s * self.chip_rate_hz)
        if n_chips <= 0:
            raise ValueError("duration_s * chip_rate_hz must be >= 1 chip")

        if self.order is None:
            self.order = _choose_order(n_chips)
        elif (1 << self.order) - 1 < n_chips:
            raise ValueError(
                f"order={self.order} (period {(1 << self.order) - 1}) is too short "
                f"for {n_chips} chips; need order >= {_choose_order(n_chips)}"
            )

        if self.seed is None:
            period = (1 << self.order) - 1
            self.seed = secrets.randbelow(period) + 1  # nonzero

        if self.chips is None:
            bits = _lfsr_bits(self.order, self.seed, n_chips)
            self.chips = (2 * bits.astype(np.int8) - 1)  # {0,1} -> {-1,+1}

    @property
    def n_chips(self) -> int:
        return len(self.chips)

    @property
    def chip_duration_s(self) -> float:
        return 1.0 / self.chip_rate_hz

    def value_at(self, t) -> np.ndarray:
        """Zero-order-hold lookup: the chip value in effect at elapsed
        time(s) t (seconds since session start). t may be scalar or array.
        Times outside [0, duration_s) clamp to the nearest edge chip."""
        t = np.asarray(t, dtype=np.float64)
        idx = np.floor(t * self.chip_rate_hz).astype(np.int64)
        if self.loop:
            idx = np.mod(idx, self.n_chips)
        else:
            idx = np.clip(idx, 0, self.n_chips - 1)
        return self.chips[idx]

    def to_dict(self) -> dict:
        return {
            "chip_rate_hz": self.chip_rate_hz,
            "duration_s": self.duration_s,
            "order": self.order,
            "seed": self.seed,
            "n_chips": self.n_chips,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Challenge":
        return cls(
            chip_rate_hz=d["chip_rate_hz"],
            duration_s=d["duration_s"],
            order=d["order"],
            seed=d["seed"],
        )

    @classmethod
    def load(cls, path: str | Path) -> "Challenge":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def autocorrelation(chips: np.ndarray) -> np.ndarray:
    """Normalised circular autocorrelation, lag 0..len-1. For a true
    m-sequence this is 1.0 at lag 0 and approx -1/N elsewhere -- the
    "near-ideal" property the module docstring refers to. Provided so the
    LFSR construction can be sanity-checked (e.g. in tests) without
    duplicating cross-correlation logic that belongs to the optical lane."""
    n = len(chips)
    x = chips.astype(np.float64)
    energy = float(np.dot(x, x))
    return np.array([np.dot(x, np.roll(x, -lag)) / energy for lag in range(n)])


if __name__ == "__main__":
    c = Challenge()
    print(f"order={c.order} seed={c.seed} n_chips={c.n_chips} "
          f"chip_rate_hz={c.chip_rate_hz} duration_s={c.duration_s}")

    # Circular autocorrelation is only guaranteed near-ideal over the LFSR's
    # FULL period (2**order - 1 chips); our session truncates to n_chips,
    # so check both: the full-period sequence (proves the LFSR/taps are
    # correct) and the truncated one actually used in a session (what the
    # optical lane's narrow 0-300ms lag search actually sees).
    full = Challenge(chip_rate_hz=c.chip_rate_hz, order=c.order, seed=c.seed,
                      duration_s=((1 << c.order) - 1) / c.chip_rate_hz)
    ac_full = autocorrelation(full.chips)
    ac_used = autocorrelation(c.chips)
    print(f"full-period ({full.n_chips} chips) circular autocorr: "
          f"peak={ac_full[0]:.3f}, max off-peak={np.max(np.abs(ac_full[1:])):.4f} "
          f"(ideal: 1/{full.n_chips}={1/full.n_chips:.4f})")
    print(f"session-truncated ({c.n_chips} chips) circular autocorr: "
          f"peak={ac_used[0]:.3f}, max off-peak={np.max(np.abs(ac_used[1:])):.4f} "
          f"-- higher sidelobes are expected here; the optical lane's lag "
          f"search only spans 0-300ms (<=1.5 chips), where the peak stays sharp")
