"""Keystroke-impact ("pressure") lane.

The host this project is measured on (ASUS TUF A16) has no force-sensing
touchpad and no IMU reachable from Python (see praesens/inertial.py), so
key PRESSURE cannot be read from a dedicated sensor. What every laptop does
have is a microphone a few centimetres from the keyboard -- and a key that
is physically struck makes a broadband impact transient ("thock") whose
loudness grows with strike force. That gives a physically independent
witness for every keystroke the OS reports:

  * a genuine key press = an OS key-down event AND an acoustic impact a
    near-constant latency later (keyboard scan + USB/i8042 + OS hook on one
    side, ADC + audio buffering on the other -- both fixed per host);
  * an injected key press (pynput/SendInput/AutoHotkey, a remote-desktop
    or RDP/VNC session, a replayed keystroke trace, a "typing bot" driving
    the challenge phrase) = an OS key-down event with NO impact behind it.

This is the keystroke-vibration claim of the PRAESENS disclosure realised
with the sensor every laptop actually ships: the inertial lane needs an
accelerometer that isn't there; the impact itself is audible.

Scoring, in the same shape as the other lanes (subscore, lag_ms):
  1. Impact onsets are detected in the microphone signal (high-passed, with
     the acoustic lane's probe carrier and harmonics notched out when the
     recording is shared with praesens.acoustic -- the BPSK probe is a
     narrowband tone, an impact is broadband, so a few notches remove the
     probe without removing the impact).
  2. Every audio sample is mapped onto time.perf_counter() (the clock the
     keystroke listener stamps with) from callback-time anchors -- see
     fit_sample_clock().
  3. A single host lag L is searched: the fraction of key-downs that have an
     impact onset within +/- match_tolerance_ms of (t_down + L). Genuine
     typing peaks sharply at one L; injected keys never do.
  4. The raw hit fraction is corrected for chance with DECOY keystroke
     trains -- the same keystrokes with each key independently jittered by
     hundreds of ms, run through the SAME full lag search -- so a noisy
     room (speech, a second person typing to cover an injection) raises
     the chance floor instead of raising the score:
        subscore = clip((hit - chance) / (1 - chance), 0, 1).

Per-key strike force (impact level in dB above the room's envelope floor)
and dwell (t_up - t_down) are reported for the dashboard and the log as the
"pressure" profile; they are descriptive, not part of the pass decision.

PRIVACY: this lane only consumes the typing lane's character-free event
dicts ({t_down, t_up, matched_expected}); it never sees what was typed.
The raw audio is analysed in memory and NOT written to the session log --
only onset times/levels derived from it are.

What this does NOT defend against: an attacker who injects keystrokes AND
plays a synthesised click through a speaker at the matching instant for
every key. That requires real-time control of an audio output in the
victim's room; it's in the threat model as an explicit residual risk, not
silently assumed away.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.signal

try:
    import sounddevice as sd
except (ImportError, OSError) as e:  # pragma: no cover - no PortAudio build
    sd = None
    _SD_IMPORT_ERROR = e
else:
    _SD_IMPORT_ERROR = None


@dataclass
class PressureConfig:
    sample_rate_hz: int = 44100
    input_device: int | str | None = None
    highpass_hz: float = 1500.0          # keyboard impacts are broadband; voiced speech mostly isn't
    notch_bandwidth_hz: float = 120.0    # width of each probe-carrier notch (shared-recording case)
    envelope_rate_hz: float = 1000.0     # onset-detection resolution (1 ms)
    envelope_smooth_ms: float = 2.0
    rise_window_ms: float = 15.0         # an impact rises within this long
    onset_threshold_mad: float = 4.0     # onset strength must exceed median + k*MAD ...
    onset_min_rise_db: float = 4.0       # ... and at least this many dB of rise. 2026-10-17, real
                                          # typing on the ASUS TUF A16: at 5.0 / 6 dB the detector
                                          # reported NO onset within 60 ms of 8 of 17 unmatched keys
                                          # -- real strikes it never saw. 4.0 / 4 dB recovered them
                                          # (78 -> 90 onsets, genuine hit fraction 0.754 -> 0.838)
                                          # while an injected capture stayed at score 0.000.
    min_onset_separation_ms: float = 35.0
    edge_guard_ms: float = 50.0          # onsets this close to the START or END of the capture are
                                          # discarded: the recording begins and ends abruptly, and the
                                          # high-pass/notch filters plus the envelope's rise window turn
                                          # those discontinuities into a rise that looks like an impact
                                          # (found 2026-10-17 when the sensitivity above was raised --
                                          # a steady 4 kHz tone produced "impacts" at t=0.03 s and at
                                          # the final sample, where no strike exists). 50 ms covers
                                          # the rise window (15 ms) plus the high-pass/notch settling
                                          # measured on that tone (artefact peaked at 32 ms).
    match_tolerance_ms: float = 25.0
    lag_search_min_ms: float = -100.0    # key-down can be stamped slightly AFTER the sound
    lag_search_max_ms: float = 450.0     # Windows MME input buffering alone can be ~100-300 ms
    lag_step_ms: float = 2.0
    min_keystrokes: int = 5
    n_decoys: int = 12
    decoy_jitter_min_s: float = 0.3
    decoy_jitter_max_s: float = 1.5
    min_audio_std: float = 1e-6          # below this the capture is silent -> no_evidence
    pass_threshold: float = 0.5

    @classmethod
    def from_dict(cls, d: dict) -> "PressureConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Audio clock: map sample index -> time.perf_counter()
# ---------------------------------------------------------------------------

def fit_sample_clock(anchors: list, fs: float) -> float | None:
    """anchors: (perf_counter_at_callback, n_samples_recorded_after_callback).
    A callback can only run AFTER its block was captured, never before, so
    each anchor gives an upper bound on the perf time of sample 0:
    perf_k - n_k/fs. The least-delayed callback is the tightest bound;
    the remaining constant (ADC + driver buffering) is a fixed per-host
    offset, absorbed by the lag search exactly like the acoustic lane's
    loopback floor. Returns perf time of sample 0, or None."""
    if not anchors:
        return None
    arr = np.asarray(anchors, dtype=np.float64)
    return float(np.min(arr[:, 0] - arr[:, 1] / float(fs)))


class MicRecorder:
    """Input-only microphone stream for sessions where the acoustic lane is
    NOT running (when it is, its duplex recording is reused instead -- one
    microphone stream per session, same rule as one camera capture loop).

    max_keep_s bounds memory for long-running callers (the HUD keeps one
    stream open for hours): older audio is dropped, and the sample clock
    keeps counting so perf_counter mapping stays exact."""

    def __init__(self, config: PressureConfig, max_keep_s: float | None = None):
        if sd is None:
            raise RuntimeError(f"sounddevice/PortAudio unavailable: {_SD_IMPORT_ERROR}")
        self.config = config
        self.max_keep_s = max_keep_s
        self._chunks: list = []
        self._kept = 0          # samples currently held in _chunks
        self._dropped = 0       # samples discarded from the front
        self._sample0: float | None = None  # perf time of sample 0 (min over anchors, kept incrementally)
        self._lock = threading.Lock()
        self._stream = None
        self.status_flags: list = []

    @property
    def anchors(self) -> list:  # kept for API compatibility (tests / debugging)
        return [] if self._sample0 is None else [(self._sample0, 0)]

    def _callback(self, indata, frames, time_info, status):
        t = time.perf_counter()
        if status:
            self.status_flags.append(str(status))
        with self._lock:
            self._chunks.append(indata[:, 0].copy())
            self._kept += frames
            n_total = self._dropped + self._kept
            cand = t - n_total / float(self.config.sample_rate_hz)
            self._sample0 = cand if self._sample0 is None else min(self._sample0, cand)
            if self.max_keep_s is not None:
                limit = int(self.max_keep_s * self.config.sample_rate_hz)
                while self._kept - len(self._chunks[0]) >= limit and len(self._chunks) > 1:
                    n0 = len(self._chunks.pop(0))
                    self._kept -= n0
                    self._dropped += n0

    def start(self) -> None:
        self._stream = sd.InputStream(samplerate=self.config.sample_rate_hz, channels=1, dtype="float32",
                                      device=self.config.input_device, callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_audio(self) -> tuple[np.ndarray, float | None]:
        """Everything still held + perf time of its first sample."""
        with self._lock:
            audio = np.concatenate(self._chunks) if self._chunks else np.zeros(0, dtype=np.float32)
            dropped, s0 = self._dropped, self._sample0
        if s0 is None:
            return audio, None
        return audio, s0 + dropped / float(self.config.sample_rate_hz)

    def audio_between(self, t0: float, t1: float) -> tuple[np.ndarray, float | None]:
        """Audio covering perf-time [t0, t1] (clipped to what's held)."""
        audio, a0 = self.get_audio()
        if a0 is None or len(audio) == 0:
            return audio, a0
        fs = float(self.config.sample_rate_hz)
        i0 = int(np.clip(np.floor((t0 - a0) * fs), 0, len(audio)))
        i1 = int(np.clip(np.ceil((t1 - a0) * fs), i0, len(audio)))
        return audio[i0:i1], a0 + i0 / fs


# ---------------------------------------------------------------------------
# Impact onset detection
# ---------------------------------------------------------------------------

@dataclass
class Onsets:
    times_s: np.ndarray      # seconds from the first audio sample
    rise_db: np.ndarray      # onset strength (dB rise within rise_window_ms)
    level_db: np.ndarray     # envelope peak above the recording's median floor -> strike force proxy
    envelope_db: np.ndarray  # the 1 kHz envelope itself (for plotting)
    envelope_rate_hz: float


def _probe_notches(carrier_hz: float | None, fs: float, max_harmonic: int = 4) -> list:
    if not carrier_hz:
        return []
    return [k * carrier_hz for k in range(1, max_harmonic + 1) if k * carrier_hz < fs / 2 - 500]


def detect_impacts(audio: np.ndarray, fs: float, config: PressureConfig,
                   notch_freqs_hz: list | None = None) -> Onsets:
    x = np.asarray(audio, dtype=np.float64)
    empty = Onsets(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), config.envelope_rate_hz)
    if len(x) < int(0.2 * fs):
        return empty
    x = x - x.mean()
    sos = scipy.signal.butter(4, config.highpass_hz, btype="high", fs=fs, output="sos")
    for f0 in (notch_freqs_hz or []):
        lo, hi = f0 - config.notch_bandwidth_hz / 2, f0 + config.notch_bandwidth_hz / 2
        if lo > config.highpass_hz * 0.5 and hi < fs / 2:
            sos = np.vstack([sos, scipy.signal.butter(2, [lo, hi], btype="bandstop", fs=fs, output="sos")])
    y = scipy.signal.sosfiltfilt(sos, x)

    q = max(1, int(round(fs / config.envelope_rate_hz)))
    power = y * y
    n_blocks = len(power) // q
    env = power[:n_blocks * q].reshape(n_blocks, q).mean(axis=1)
    smooth = max(1, int(round(config.envelope_smooth_ms * config.envelope_rate_hz / 1000.0)))
    if smooth > 1:
        env = np.convolve(env, np.ones(smooth) / smooth, mode="same")
    env_db = 10.0 * np.log10(env + 1e-12)

    rise = max(2, int(round(config.rise_window_ms * config.envelope_rate_hz / 1000.0)))
    # onset strength: how far the envelope has climbed above its minimum over the preceding rise window
    padded = np.concatenate([np.full(rise, env_db[0]), env_db])
    windows = np.lib.stride_tricks.sliding_window_view(padded, rise)[:len(env_db)]
    strength = env_db - windows.min(axis=1)

    med = float(np.median(strength))
    mad = float(np.median(np.abs(strength - med))) + 1e-9
    height = max(med + config.onset_threshold_mad * 1.4826 * mad, config.onset_min_rise_db)
    distance = max(1, int(round(config.min_onset_separation_ms * config.envelope_rate_hz / 1000.0)))
    peaks, props = scipy.signal.find_peaks(strength, height=height, distance=distance)

    guard = int(round(config.edge_guard_ms * config.envelope_rate_hz / 1000.0))
    if guard > 0 and len(peaks):
        keep = (peaks >= guard) & (peaks < len(strength) - guard)
        peaks, props = peaks[keep], {k: v[keep] for k, v in props.items()}

    floor_db = float(np.median(env_db))
    # strike force proxy: the envelope's peak in the 10 ms following the onset, above the room floor
    look = max(1, int(round(10 * config.envelope_rate_hz / 1000.0)))
    levels = np.array([env_db[p:p + look].max() - floor_db for p in peaks]) if len(peaks) else np.zeros(0)
    return Onsets(times_s=peaks / config.envelope_rate_hz, rise_db=props["peak_heights"].astype(float),
                  level_db=levels, envelope_db=env_db, envelope_rate_hz=config.envelope_rate_hz)


# ---------------------------------------------------------------------------
# Keystroke <-> impact matching
# ---------------------------------------------------------------------------

def _hits_at_lags(key_t: np.ndarray, onset_t: np.ndarray, lags_s: np.ndarray, tol_s: float):
    """For each lag: number of keys with an onset within tol of (key + lag),
    plus the sum of |residual| over hits (tie-break: tighter alignment wins)."""
    if len(onset_t) == 0 or len(key_t) == 0:
        return np.zeros(len(lags_s), dtype=int), np.zeros(len(lags_s))
    hits = np.zeros(len(lags_s), dtype=int)
    resid = np.zeros(len(lags_s))
    for i, lag in enumerate(lags_s):
        target = key_t + lag
        idx = np.clip(np.searchsorted(onset_t, target), 1, len(onset_t) - 1) if len(onset_t) > 1 else \
            np.zeros(len(target), dtype=int)
        if len(onset_t) > 1:
            d = np.minimum(np.abs(onset_t[idx] - target), np.abs(onset_t[idx - 1] - target))
        else:
            d = np.abs(onset_t[0] - target)
        ok = d <= tol_s
        hits[i] = int(ok.sum())
        resid[i] = float(d[ok].sum())
    return hits, resid


def best_lag(key_t: np.ndarray, onset_t: np.ndarray, config: PressureConfig) -> tuple[int, float, int]:
    """Returns (index of best lag, best lag in s, hits at it)."""
    lags_s = np.arange(config.lag_search_min_ms, config.lag_search_max_ms + 1e-9, config.lag_step_ms) / 1000.0
    hits, resid = _hits_at_lags(key_t, onset_t, lags_s, config.match_tolerance_ms / 1000.0)
    order = np.lexsort((resid, -hits))  # most hits, then smallest residual
    i = int(order[0])
    return i, float(lags_s[i]), int(hits[i])


@dataclass
class PressureResult:
    status: str                       # "ok" | "no_evidence" | "insufficient_signal"
    score: float | None = None
    lag_ms: float | None = None
    hit_fraction: float = float("nan")
    chance_fraction: float = float("nan")
    lag_jitter_ms: float = float("nan")
    n_keys: int = 0
    n_onsets: int = 0
    per_key: list = field(default_factory=list)   # {t_down, matched, force_db, dwell_s}
    force_cv: float = float("nan")
    diagnostics: str = ""


def analyse_keystroke_impacts(audio: np.ndarray, sample0_perf: float | None, fs: float, events: list,
                              config: PressureConfig, notch_freqs_hz: list | None = None,
                              rng: np.random.Generator | None = None) -> PressureResult:
    """events: the typing lane's character-free dicts. Key times are taken
    on time.perf_counter(); sample0_perf maps the audio onto that clock."""
    key_t_perf = np.array(sorted(e["t_down"] for e in events if e.get("t_down") is not None), dtype=np.float64)
    n_keys = len(key_t_perf)
    if n_keys < config.min_keystrokes:
        return PressureResult(status="no_evidence", n_keys=n_keys,
                              diagnostics=f"n_keystrokes={n_keys} (need >= {config.min_keystrokes})")
    if audio is None or len(audio) == 0 or sample0_perf is None:
        return PressureResult(status="no_evidence", n_keys=n_keys, diagnostics="no microphone audio captured")
    if float(np.std(audio)) < config.min_audio_std:
        return PressureResult(status="no_evidence", n_keys=n_keys,
                              diagnostics="silent capture -- check microphone permission / input device")

    key_t = key_t_perf - sample0_perf  # onto the audio's own time axis
    dur = len(audio) / fs
    inside = (key_t >= -config.lag_search_min_ms / 1000.0) & (key_t <= dur - config.lag_search_max_ms / 1000.0)
    key_t_in = key_t[inside]
    if len(key_t_in) < config.min_keystrokes:
        return PressureResult(status="no_evidence", n_keys=n_keys,
                              diagnostics=f"only {len(key_t_in)}/{n_keys} keystrokes fall inside the audio capture")

    onsets = detect_impacts(audio, fs, config, notch_freqs_hz)
    onset_t = onsets.times_s
    _i, lag_s, hits = best_lag(key_t_in, onset_t, config)
    hit_frac = hits / len(key_t_in)

    rng = rng if rng is not None else np.random.default_rng(12345)
    decoy_fracs = []
    for _ in range(config.n_decoys):
        jitter = rng.uniform(config.decoy_jitter_min_s, config.decoy_jitter_max_s, len(key_t_in))
        jitter *= rng.choice([-1.0, 1.0], len(key_t_in))
        decoy = np.sort(np.clip(key_t_in + jitter, 0.0, dur))
        _j, _l, h = best_lag(decoy, onset_t, config)
        decoy_fracs.append(h / len(decoy))
    chance = float(np.mean(decoy_fracs)) if decoy_fracs else 0.0
    score = float(np.clip((hit_frac - chance) / (1.0 - chance), 0.0, 1.0)) if chance < 1.0 else 0.0

    # per-key profile at the chosen lag
    tol = config.match_tolerance_ms / 1000.0
    per_key, residuals, forces = [], [], []
    by_down = {e["t_down"]: e for e in events if e.get("t_down") is not None}
    for kp, kt in zip(key_t_perf, key_t):
        ev = by_down.get(kp, {})
        dwell = (ev["t_up"] - ev["t_down"]) if ev.get("t_up") is not None else None
        matched, force = False, None
        if len(onset_t):
            j = int(np.argmin(np.abs(onset_t - (kt + lag_s))))
            r = float(onset_t[j] - (kt + lag_s))
            if abs(r) <= tol:
                matched, force = True, float(onsets.level_db[j])
                residuals.append(r)
                forces.append(force)
        per_key.append({"t_down": float(kp), "matched": matched, "force_db": force,
                        "dwell_s": (float(dwell) if dwell is not None else None)})
    jitter_ms = float(np.std(residuals) * 1000.0) if len(residuals) >= 2 else float("nan")
    force_cv = (float(np.std(forces) / np.mean(forces)) if len(forces) >= 3 and np.mean(forces) > 0
                else float("nan"))

    lag_ms = lag_s * 1000.0
    diag = (f"impacts matched {hits}/{len(key_t_in)} keys (chance {chance:.2f}), "
            f"lag={lag_ms:.0f}ms jitter={jitter_ms:.1f}ms, onsets={len(onset_t)}")
    if np.isfinite(force_cv):
        diag += f", force_cv={force_cv:.2f}"
    if len(onset_t) == 0:
        diag += ", NO key impacts heard (injected keystrokes, or keyboard not audible to this mic)"

    edge = (abs(lag_ms - config.lag_search_min_ms) < config.lag_step_ms * 1.5 or
            abs(lag_ms - config.lag_search_max_ms) < config.lag_step_ms * 1.5)
    if edge and score >= config.pass_threshold:
        return PressureResult(status="insufficient_signal", hit_fraction=hit_frac, chance_fraction=chance,
                              n_keys=n_keys, n_onsets=len(onset_t), per_key=per_key,
                              diagnostics=f"best lag on search edge ({lag_ms:.0f}ms), {diag}")

    return PressureResult(status="ok", score=score, lag_ms=lag_ms, hit_fraction=hit_frac,
                          chance_fraction=chance, lag_jitter_ms=jitter_ms, n_keys=n_keys,
                          n_onsets=len(onset_t), per_key=per_key, force_cv=force_cv, diagnostics=diag)


def pressure_confidence(result: PressureResult) -> float:
    """More matched keystrokes -> more confidence; 20 keys is 'full'."""
    if result.status != "ok":
        return 0.0
    return float(np.clip(result.n_keys / 20.0, 0.2, 1.0))


def pressure_record(result: PressureResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "status": result.status, "score": result.score, "lag_ms": result.lag_ms,
        "hit_fraction": None if not np.isfinite(result.hit_fraction) else result.hit_fraction,
        "chance_fraction": None if not np.isfinite(result.chance_fraction) else result.chance_fraction,
        "lag_jitter_ms": None if not np.isfinite(result.lag_jitter_ms) else result.lag_jitter_ms,
        "force_cv": None if not np.isfinite(result.force_cv) else result.force_cv,
        "n_keys": result.n_keys, "n_onsets": result.n_onsets,
        "per_key": result.per_key, "diagnostics": result.diagnostics,
    }


if __name__ == "__main__":
    import argparse
    import yaml
    from pathlib import Path

    from praesens.typing import KeystrokeCapture

    parser = argparse.ArgumentParser(description="Keystroke-impact (pressure) lane smoke test: "
                                                 "type normally for N seconds")
    parser.add_argument("--seconds", type=float, default=12.0)
    args = parser.parse_args()

    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    raw = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
    pconfig = PressureConfig.from_dict(raw.get("pressure", {}))

    rec = MicRecorder(pconfig)
    keys = KeystrokeCapture(expected_phrase=None)
    rec.start()
    keys.start()
    print(f"recording {args.seconds:.0f}s -- type anything (characters are never stored)...")
    time.sleep(args.seconds)
    keys.stop()
    rec.stop()
    audio, t0 = rec.get_audio()
    res = analyse_keystroke_impacts(audio, t0, pconfig.sample_rate_hz, keys.get_events(), pconfig)
    print(f"status={res.status} score={res.score} lag_ms={res.lag_ms}\n  {res.diagnostics}")
