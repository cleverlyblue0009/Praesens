"""Milestone 13 (proposed): acoustic lane.

Replaces praesens.fusion.acoustic_lane_stub() with a real third
physically-independent lane, per the same architecture as optical
(praesens/optical.py) and typing (praesens/typing.py): a per-session
challenge is emitted through host hardware, a sensor observes the
environment's response to it, and a cross-correlation against the KNOWN
emitted signal yields (subscore, lag_ms, confidence) for fusion.py's
joint-temporal-coherence gate.

Design choice -- reuse the SAME m-sequence, don't mint a second one:
the optical lane already gets near-ideal (impulse-like) autocorrelation
from praesens.challenge.Challenge's LFSR sequence (see that module's
docstring). Rather than generating an independent acoustic code, this
module BPSK-modulates the SAME per-session chip sequence onto an audio
carrier. That keeps "one freshly generated challenge, multiple
independent lanes" literally true -- not three separately-seeded probes
that happen to run concurrently -- and it means a session's single saved
seed is still sufficient to regenerate every lane's expected signal
exactly, same as Challenge.save()/load() already promises for optical.

Duplex, not two streams: playback and capture run on ONE
sounddevice.Stream from a single callback, sharing one hardware clock --
the audio analogue of session.py's rule that optical+typing share one
camera capture loop rather than opening two. Two independent
play/record streams drift against each other on separate OS audio
clocks and the lag measurement becomes meaningless.

What this module does NOT yet solve (be honest about it, same spirit as
praesens/inertial.py's stub docstring): the fixed electrical/OS-buffer
round-trip latency between writing a sample to the output buffer and
that same instant appearing in the input buffer is hardware- and
driver-dependent and NOT the same thing as the acoustic travel time to a
face and back. LANE_LAG_BOUNDS_MS["acoustic"] in fusion.py is
deliberately wide (-500..500ms) to accommodate this, but a real
deployment should run calibrate_loopback_latency_ms() once per host
(speaker piped as directly as possible into the mic, no face in the
way) and subtract that fixed offset before trusting lag_ms as a
physical-distance measurement. Until that calibration step is wired
into session.py, treat lag_ms as "relative to this host's own loopback
floor," not an absolute time-of-flight -- report it, don't over-claim it.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from praesens.challenge import Challenge

try:
    import sounddevice as sd
except OSError as e:  # pragma: no cover - environment without a usable PortAudio build
    sd = None
    _SD_IMPORT_ERROR = e
else:
    _SD_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Probe synthesis -- BPSK-modulate the session's own Challenge chips
# ---------------------------------------------------------------------------

@dataclass
class AcousticConfig:
    sample_rate_hz: int = 44100
    carrier_hz: float = 18000.0        # near-ultrasonic: mostly inaudible on a laptop
                                        # speaker/mic pair sampling at 44.1kHz (Nyquist
                                        # allows up to ~22kHz); drop to ~3000-6000Hz if
                                        # this host's mic rolls off before 18kHz (check
                                        # with scripts/probe.py-style frequency sweep
                                        # before trusting a session run)
    amplitude: float = 0.6             # peak amplitude, [0,1] of full scale -- keep
                                        # headroom, this is BPSK not silence-vs-tone
    lag_search_max_ms: float = 60.0    # generous vs. optical's 300ms: sound covers a
                                        # laptop-to-face-and-back round trip (~1-2m) in
                                        # a few ms: this bound is dominated by
                                        # loopback/buffer latency, not physics
    snr_floor_db: float = 6.0          # acoustic noise floor is typically worse than
                                        # optical's; start conservative, tune from real logs
    ramp_ms: float = 5.0               # raised-cosine on/off ramp to avoid a click that
                                        # would itself correlate as a false sharp peak

    @classmethod
    def from_dict(cls, d: dict) -> "AcousticConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _raised_cosine_ramp(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))


def synthesize_probe(challenge: Challenge, config: AcousticConfig) -> np.ndarray:
    """BPSK-modulates challenge.chips (the SAME +/-1 m-sequence the optical
    lane emits as light) onto a carrier tone, one chip held for
    sample_rate_hz/chip_rate_hz samples (zero-order hold, matching
    Challenge.value_at()'s own convention). Returns a float32 mono buffer
    in [-amplitude, +amplitude], ready to hand to a sounddevice stream."""
    fs = config.sample_rate_hz
    samples_per_chip = int(round(fs / challenge.chip_rate_hz))
    if samples_per_chip < 2:
        raise ValueError(
            f"sample_rate_hz={fs} too low for chip_rate_hz={challenge.chip_rate_hz} "
            f"(need >= 2 samples/chip); this is unrelated to the optical lane's own "
            f"Nyquist check in optical.check_nyquist(), which governs CAMERA fps instead"
        )

    chip_signal = np.repeat(challenge.chips.astype(np.float64), samples_per_chip)
    n = len(chip_signal)
    t = np.arange(n) / fs
    carrier = np.cos(2 * np.pi * config.carrier_hz * t)
    probe = config.amplitude * chip_signal * carrier

    ramp_n = int(round(config.ramp_ms / 1000.0 * fs))
    ramp_n = min(ramp_n, n // 2)
    if ramp_n > 0:
        ramp = _raised_cosine_ramp(ramp_n)
        probe[:ramp_n] *= ramp
        probe[-ramp_n:] *= ramp[::-1]

    return probe.astype(np.float32)


# ---------------------------------------------------------------------------
# Duplex playback + capture -- one stream, one clock
# ---------------------------------------------------------------------------

class DuplexProbePlayer:
    """Plays `probe` once through the default output device while
    recording from the default input device on the SAME callback-driven
    stream, so both share one hardware sample clock (see module
    docstring for why two separate streams are the wrong design)."""

    def __init__(self, probe: np.ndarray, config: AcousticConfig, record_extra_s: float = 0.5):
        if sd is None:
            raise RuntimeError(
                f"sounddevice/PortAudio is not available in this environment "
                f"({_SD_IMPORT_ERROR}); acoustic lane cannot run headless/without audio "
                f"hardware -- this is a hard requirement, not a fallback-to-no_evidence "
                f"case, since a REAL host running a REAL session always has both"
            )
        self.probe = probe
        self.config = config
        self.record_extra_s = record_extra_s
        n_record = len(probe) + int(round(record_extra_s * config.sample_rate_hz))
        self._recorded = np.zeros(n_record, dtype=np.float32)
        self._play_idx = 0
        self._rec_idx = 0
        self._start_perf_counter: float | None = None

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            # xruns show up here -- don't silently drop them, they invalidate
            # the shared-clock assumption this whole lane depends on
            self._xrun_flags = getattr(self, "_xrun_flags", [])
            self._xrun_flags.append(str(status))

        if self._start_perf_counter is None:
            self._start_perf_counter = time.perf_counter()

        remaining_play = len(self.probe) - self._play_idx
        n_play = max(0, min(frames, remaining_play))
        if n_play > 0:
            outdata[:n_play, 0] = self.probe[self._play_idx:self._play_idx + n_play]
        if n_play < frames:
            outdata[n_play:, 0] = 0.0
        self._play_idx += n_play

        n_rec = min(frames, len(self._recorded) - self._rec_idx)
        if n_rec > 0:
            self._recorded[self._rec_idx:self._rec_idx + n_rec] = indata[:n_rec, 0]
        self._rec_idx += n_rec

    def run(self) -> tuple[np.ndarray, float, list[str]]:
        """Blocks until record_extra_s past the end of playback. Returns
        (recorded_mono_float32, stream_reported_latency_s, xrun_warnings)."""
        fs = self.config.sample_rate_hz
        with sd.Stream(samplerate=fs, channels=1, dtype="float32",
                        callback=self._callback) as stream:
            total_s = len(self._recorded) / fs
            sd.sleep(int((total_s + 0.1) * 1000))
            latency_s = float(stream.latency[0]) + float(stream.latency[1]) \
                if isinstance(stream.latency, (tuple, list)) else float(stream.latency)
        return self._recorded[:self._rec_idx], latency_s, getattr(self, "_xrun_flags", [])


def calibrate_loopback_latency_ms(config: AcousticConfig, n_trials: int = 3) -> float:
    """Run-once-per-host calibration: with the speaker piped as directly
    as possible into the mic (or physically adjacent, no face in the
    beam), the measured lag IS the fixed electrical/buffer round trip,
    with no acoustic travel-time component worth distinguishing from
    hardware latency. Median of n_trials short probes. See module
    docstring -- this offset should be subtracted from real-session
    lag_ms before treating it as a physical-distance measurement; this
    function does not yet get called automatically anywhere."""
    challenge = Challenge(chip_rate_hz=20.0, duration_s=1.0)  # short, fast probe is enough
    probe = synthesize_probe(challenge, config)
    lags = []
    for _ in range(n_trials):
        player = DuplexProbePlayer(probe, config, record_extra_s=0.2)
        recorded, _latency_s, xruns = player.run()
        if xruns:
            warnings.warn(f"xrun during calibration trial: {xruns}")
        score, lag_ms, *_ = cross_correlate_probe(recorded, probe, config)
        lags.append(lag_ms)
    return float(np.median(lags))


# ---------------------------------------------------------------------------
# Matched-filter cross-correlation + SNR (same shape as
# optical.cross_correlate_lag_search / optical.estimate_snr_db, adapted
# from a non-uniform-frame-timestamp problem to a uniformly-sampled-audio
# one, which is actually the simpler case -- no resampling needed)
# ---------------------------------------------------------------------------

def cross_correlate_probe(recorded: np.ndarray, probe: np.ndarray, config: AcousticConfig):
    """Normalised cross-correlation of `recorded` against the known
    `probe`, searched over [0, lag_search_max_ms]. Returns
    (score, lag_ms, peak_sample_idx, correlation_curve). A genuine
    acoustic reflection of a real 3-D face should show a single sharp,
    isolated peak (m-sequence autocorrelation property, see
    praesens.challenge's module docstring) at a lag consistent with this
    host's calibrated loopback floor; a flat or broad curve is the audio
    analogue of optical.py's FIX-3 "no detectable coupling" diagnosis."""
    fs = config.sample_rate_hz
    max_lag_samples = int(round(config.lag_search_max_ms / 1000.0 * fs))

    if len(recorded) < len(probe) or np.std(recorded) < 1e-9:
        return 0.0, 0.0, 0, np.zeros(max_lag_samples + 1)

    probe_centered = probe - probe.mean()
    probe_norm = np.linalg.norm(probe_centered)
    if probe_norm < 1e-12:
        return 0.0, 0.0, 0, np.zeros(max_lag_samples + 1)

    scores = np.zeros(max_lag_samples + 1)
    for lag in range(max_lag_samples + 1):
        window = recorded[lag:lag + len(probe)]
        if len(window) < len(probe):
            break
        w_centered = window - window.mean()
        w_norm = np.linalg.norm(w_centered)
        if w_norm < 1e-12:
            continue
        scores[lag] = float(np.dot(probe_centered, w_centered) / (probe_norm * w_norm))

    best_idx = int(np.argmax(scores))
    return float(scores[best_idx]), float(best_idx / fs * 1000.0), best_idx, scores


def estimate_snr_db(recorded: np.ndarray, probe_start_idx: int, probe_len: int) -> float:
    """Signal window (where the matched probe response landed) vs. a
    noise-floor window drawn from BEFORE the probe started playing --
    the acoustic equivalent of optical.estimate_snr_db's in-band/
    out-of-band split, but simpler here since we know exactly where in
    time the "should be quiet" region is (pre-playback), rather than
    needing a bandpass filter."""
    pre_roll = recorded[:max(0, probe_start_idx)]
    signal_window = recorded[probe_start_idx:probe_start_idx + probe_len]
    if len(pre_roll) < 100 or len(signal_window) < 100:
        return float("nan")
    signal_power = float(np.mean(signal_window.astype(np.float64) ** 2))
    noise_power = float(np.mean(pre_roll.astype(np.float64) ** 2))
    if noise_power < 1e-12:
        return float("inf") if signal_power > 1e-12 else float("nan")
    return float(10 * np.log10(signal_power / noise_power))


# ---------------------------------------------------------------------------
# Lane result -- same shape session.py's optical_lane_result/
# typing_lane_result already produce, so wiring this in is a new function
# alongside those two, not a change to fusion.py's LaneResult contract.
# ---------------------------------------------------------------------------

@dataclass
class AcousticResult:
    status: str                 # "ok" | "no_evidence" | "insufficient_signal"
    score: float | None = None
    lag_ms: float | None = None
    snr_db: float = float("nan")
    diagnostics: str = ""


def run_acoustic_session(challenge: Challenge, config: AcousticConfig) -> AcousticResult:
    if sd is None:
        return AcousticResult(status="no_evidence",
                               diagnostics=f"sounddevice unavailable: {_SD_IMPORT_ERROR}")

    probe = synthesize_probe(challenge, config)
    player = DuplexProbePlayer(probe, config)
    try:
        recorded, _latency_s, xruns = player.run()
    except Exception as e:
        return AcousticResult(status="no_evidence", diagnostics=f"stream error: {e}")

    if len(recorded) < len(probe):
        return AcousticResult(status="no_evidence",
                               diagnostics=f"short capture: {len(recorded)}/{len(probe)} samples")

    score, lag_ms, peak_idx, _curve = cross_correlate_probe(recorded, probe, config)
    snr_db = estimate_snr_db(recorded, peak_idx, len(probe))

    diag_suffix = f", xruns={len(xruns)}" if xruns else ""
    if np.isnan(snr_db) or snr_db < config.snr_floor_db:
        return AcousticResult(status="insufficient_signal", snr_db=snr_db,
                               diagnostics=f"snr_db={snr_db:.2f}{diag_suffix}")

    return AcousticResult(status="ok", score=score, lag_ms=lag_ms, snr_db=snr_db,
                           diagnostics=f"snr_db={snr_db:.2f}{diag_suffix}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import yaml

    parser = argparse.ArgumentParser(description="Acoustic lane smoke test")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--calibrate", action="store_true",
                         help="run loopback calibration instead of a normal probe")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "config.yaml"
    raw = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
    aconfig = AcousticConfig.from_dict(raw.get("acoustic", {}))

    if args.calibrate:
        offset_ms = calibrate_loopback_latency_ms(aconfig)
        print(f"loopback floor: {offset_ms:.2f} ms -- subtract this from real-session "
              f"lag_ms before treating it as physical time-of-flight (see module docstring)")
    else:
        challenge = Challenge(chip_rate_hz=5.0, duration_s=args.seconds, seed=args.seed)
        print(f"emitting acoustic probe: carrier={aconfig.carrier_hz}Hz "
              f"chip_rate={challenge.chip_rate_hz}Hz duration={challenge.duration_s}s "
              f"seed={challenge.seed}")
        result = run_acoustic_session(challenge, aconfig)
        print(f"status={result.status} score={result.score} lag_ms={result.lag_ms} "
              f"snr_db={result.snr_db:.2f} diagnostics={result.diagnostics!r}")