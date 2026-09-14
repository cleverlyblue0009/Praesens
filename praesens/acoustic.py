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
sounddevice.Stream from a single callback -- the audio analogue of
session.py's rule that optical+typing share one camera capture loop
rather than opening two.

Detection (rewritten 2026-09-14 after the first real 3-lane sessions,
logs/20260913T20*.json, all scored ~0.00005 with snr_db=inf):
  1. The OLD detector correlated the raw 44.1kHz recording against the
     raw carrier-modulated probe over the WHOLE session. That needs the
     speaker and mic clocks to agree to ~1 carrier cycle (250us at 4kHz)
     for 20s straight: ~50ppm of drift between the input and output
     devices (separate CoreAudio/WASAPI devices, bridged by PortAudio's
     ring buffer) or a single dropped buffer rotates the carrier phase
     and the coherent sum cancels to ~0 even when the tone is loud.
     Now: the recording is mixed down to complex baseband around
     carrier_hz, narrowly low-passed, and correlated against the chip
     envelope in segment_s-long segments that are combined
     differentially (each against the previous one, see
     _segmented_statistic) -- coherent within a segment, insensitive to
     a slowly rotating phase across segments -- so drift and xruns cost
     a little score instead of all of it.
  2. The OLD SNR estimate treated "recording before the correlation
     peak" as noise. On real hardware that region is the input stream's
     start-up zeros, so noise_power ~0 -> snr_db=inf -> status "ok" and
     confidence 1.0 for a lane that heard nothing, which then dragged
     genuine sessions to REJECT in fusion. Now: the noise floor is the
     SAME segmented statistic computed at off-carrier frequencies
     (noise_offsets_hz) where no probe energy exists -- the audio
     analogue of optical.estimate_snr_db's in-band/out-of-band split --
     so "no probe heard" lands at ~0dB, not inf.
  3. lag_search_max_ms was 60ms; a real duplex round trip (output +
     input buffers, ~1m of air) is often more than that on macOS, and
     Bluetooth output alone is 150ms+. Now 0..lag_search_max_ms (500ms
     by default, matching fusion.LANE_LAG_BOUNDS_MS["acoustic"]), and a
     peak sitting on the search edge is reported as insufficient_signal
     rather than scored.

What this module does NOT yet solve: the fixed electrical/OS-buffer
round-trip latency between writing a sample to the output buffer and
that same instant appearing in the input buffer is hardware- and
driver-dependent and NOT the same thing as the acoustic travel time to a
face and back. The first callback's PortAudio DAC/ADC timestamps are
logged as stream_offset_ms where the host API provides them, and
calibrate_loopback_latency_ms() measures it directly, but neither is
subtracted automatically -- treat lag_ms as "relative to this host's own
loopback floor," not an absolute time-of-flight.
"""
from __future__ import annotations

import threading
import time
import warnings
from dataclasses import dataclass

import numpy as np
import scipy.signal

from praesens.challenge import Challenge

try:
    import sounddevice as sd
except (ImportError, OSError) as e:  # pragma: no cover - no sounddevice / no usable PortAudio build
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
    carrier_hz: float = 4000.0         # laptop speakers and mics are both flat-ish here; 18kHz
                                        # (the first draft's default) rolls off on many mics
    amplitude: float = 0.6             # peak amplitude, [0,1] of full scale -- keep
                                        # headroom, this is BPSK not silence-vs-tone
    lag_search_max_ms: float = 500.0   # dominated by output+input buffer latency, not physics
                                        # (sound covers a laptop-to-face round trip in a few ms)
    lag_step_ms: float = 1.0           # baseband decimation step == lag resolution
    snr_floor_db: float = 6.0          # peak statistic vs. off-carrier noise statistic
    ramp_ms: float = 5.0               # raised-cosine on/off ramp to avoid a click that
                                        # would itself correlate as a false sharp peak
    transition_ms: float = 20.0        # Hann-smoothed chip transitions: instantaneous BPSK phase
                                        # flips splatter probe energy +/-hundreds of Hz around the
                                        # carrier, straight into the off-carrier noise channels
                                        # (measured 2026-09-14: +/-300-600Hz channels scored 0.37
                                        # vs 0.05-0.11 at +/-1kHz, capping snr_db near 9dB)
    baseband_bandwidth_hz: float = 10.0  # low-pass after mixing down; widened automatically
                                          # to 2x chip_rate_hz for fast-chipping probes
    segment_s: float = 1.0             # coherent-integration length; shorter tolerates more
                                        # clock drift, longer lowers the noise floor
    noise_offsets_hz: tuple = (-600.0, -300.0, 300.0, 600.0)  # off-carrier noise channels
    n_decoy_codes: int = 3             # cyclic shifts of the session m-sequence correlated on the
                                        # carrier channel -- the "right frequency, wrong code" floor
    input_device: int | str | None = None   # None = PortAudio default; see --list-devices
    output_device: int | str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "AcousticConfig":
        kwargs = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "noise_offsets_hz" in kwargs:
            kwargs["noise_offsets_hz"] = tuple(float(f) for f in kwargs["noise_offsets_hz"])
        return cls(**kwargs)


def _raised_cosine_ramp(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    return 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))


def _chip_envelope(challenge: Challenge, config: AcousticConfig) -> np.ndarray:
    """The +/-1 chip sequence zero-order-held at sample_rate_hz, with the
    on/off ramp applied. Shared by synthesize_probe() and the detector's
    reference so the two can never disagree on chip timing."""
    fs = config.sample_rate_hz
    samples_per_chip = int(round(fs / challenge.chip_rate_hz))
    if samples_per_chip < 2:
        raise ValueError(
            f"sample_rate_hz={fs} too low for chip_rate_hz={challenge.chip_rate_hz} "
            f"(need >= 2 samples/chip); this is unrelated to the optical lane's own "
            f"Nyquist check in optical.check_nyquist(), which governs CAMERA fps instead"
        )
    env = np.repeat(challenge.chips.astype(np.float64), samples_per_chip)
    transition_n = min(int(round(config.transition_ms / 1000.0 * fs)), samples_per_chip)
    if transition_n > 2:
        window = np.hanning(transition_n)
        env = scipy.signal.oaconvolve(env, window / window.sum(), mode="same")
    ramp_n = min(int(round(config.ramp_ms / 1000.0 * fs)), len(env) // 2)
    if ramp_n > 0:
        ramp = _raised_cosine_ramp(ramp_n)
        env[:ramp_n] *= ramp
        env[-ramp_n:] *= ramp[::-1]
    return env


def synthesize_probe(challenge: Challenge, config: AcousticConfig) -> np.ndarray:
    """BPSK-modulates challenge.chips (the SAME +/-1 m-sequence the optical
    lane emits as light) onto a carrier tone, one chip held for
    sample_rate_hz/chip_rate_hz samples (zero-order hold, matching
    Challenge.value_at()'s own convention, with transition_ms-smoothed
    chip edges). Returns a float32 mono buffer
    in [-amplitude, +amplitude], ready to hand to a sounddevice stream."""
    env = _chip_envelope(challenge, config)
    t = np.arange(len(env)) / config.sample_rate_hz
    return (config.amplitude * env * np.cos(2 * np.pi * config.carrier_hz * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Duplex playback + capture -- one stream, one callback
# ---------------------------------------------------------------------------

_HEADPHONE_HINTS = ("buds", "airpods", "headphone", "headset", "hands-free", "bluetooth")


def _device_name(device, kind: str) -> str:
    if sd is None:
        return "?"
    try:
        return str(sd.query_devices(device=device, kind=kind)["name"])
    except Exception:
        return "?"


class DuplexProbePlayer:
    """Plays `probe` once through the output device while recording from
    the input device on the SAME callback-driven stream (see module
    docstring for why two separate streams are the wrong design)."""

    def __init__(self, probe: np.ndarray, config: AcousticConfig, record_extra_s: float | None = None):
        if sd is None:
            raise RuntimeError(
                f"sounddevice/PortAudio is not available in this environment "
                f"({_SD_IMPORT_ERROR}); acoustic lane cannot run headless/without audio "
                f"hardware -- this is a hard requirement, not a fallback-to-no_evidence "
                f"case, since a REAL host running a REAL session always has both"
            )
        if record_extra_s is None:
            # keep recording long enough for a probe arriving at the far end of the lag search
            record_extra_s = config.lag_search_max_ms / 1000.0 + 0.25
        self.probe = probe
        self.config = config
        n_record = len(probe) + int(round(record_extra_s * config.sample_rate_hz))
        self._recorded = np.zeros(n_record, dtype=np.float32)
        self._play_idx = 0
        self._rec_idx = 0
        self._done = threading.Event()
        self._xrun_flags: list[str] = []
        self.stream_offset_ms: float | None = None

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            # xruns show up here -- don't silently drop them; the segmented
            # detector tolerates a few, but they belong in the diagnostics
            self._xrun_flags.append(str(status))

        if self.stream_offset_ms is None:
            dac, adc = float(time_info.outputBufferDacTime), float(time_info.inputBufferAdcTime)
            # some host APIs (e.g. Windows MME) report zeros here -- record nan then
            self.stream_offset_ms = (dac - adc) * 1000.0 if dac > 0 and adc > 0 else float("nan")

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
        if self._rec_idx >= len(self._recorded):
            self._done.set()

    def run(self) -> tuple[np.ndarray, float, list[str]]:
        """Blocks until the record buffer is full (or a generous timeout).
        Returns (recorded_mono_float32, stream_reported_latency_s, xrun_warnings)."""
        fs = self.config.sample_rate_hz
        device = (self.config.input_device, self.config.output_device)
        with sd.Stream(samplerate=fs, channels=1, dtype="float32", device=device,
                        callback=self._callback) as stream:
            total_s = len(self._recorded) / fs
            self._done.wait(timeout=total_s + 3.0)
            latency = stream.latency
            latency_s = (float(latency[0]) + float(latency[1])
                         if isinstance(latency, (tuple, list)) else float(latency))
        return self._recorded[:self._rec_idx], latency_s, list(self._xrun_flags)


# ---------------------------------------------------------------------------
# Detection: complex-baseband, segmented matched filter + off-carrier noise
# floor (see module docstring for why the raw-carrier version failed)
# ---------------------------------------------------------------------------

@dataclass
class ProbeDetection:
    score: float          # noise-floor-corrected peak statistic, [0, 1]
    peak: float           # raw statistic at the best lag on the carrier channel, [0, 1]
    noise_floor: float    # max(off_carrier_floor, decoy_floor) -- what peak is judged against
    off_carrier_floor: float  # mean best statistic at off-carrier frequencies (pure noise)
    decoy_floor: float    # mean best statistic for decoy codes on the carrier (noise + code sidelobes)
    lag_ms: float
    snr_db: float         # 20*log10(peak / noise_floor)
    tone_db: float        # carrier-band power vs off-carrier-band power: was the tone heard at all
    peak_at_edge: bool    # best lag sits on the lag-search boundary -> true lag probably outside it


def _baseband(x: np.ndarray, freq_hz: float, fs: float, sos: np.ndarray, q: int) -> np.ndarray:
    mixed = x * np.exp(-2j * np.pi * freq_hz * np.arange(len(x)) / fs)
    return (scipy.signal.sosfiltfilt(sos, mixed.real) + 1j * scipy.signal.sosfiltfilt(sos, mixed.imag))[::q]


def _segmented_statistic(bb: np.ndarray, ref: np.ndarray, seg_len: int, max_lag: int) -> np.ndarray:
    """For each lag in [0, max_lag], a [0, 1] statistic built from
    per-segment complex correlations z_k = <bb shifted by lag, ref segment k>.

    Differential across segments: |sum_k z_k * conj(z_{k-1})|, normalised
    by sum_k |z_k|max * |z_{k-1}|max (Cauchy-Schwarz bounds), square-rooted
    back to correlation scale. At the true lag every z_k carries the same
    chip sign and a carrier phase that only rotates slowly (a constant
    clock-drift frequency offset is a CONSTANT phase step between segments,
    which the magnitude of the sum ignores), so the products add up. At a
    wrong lag each segment's chip agreement is a random sign, so the
    products cancel -- summing plain |z_k| instead leaves wrong lags with a
    large positive bias when segments hold only a few chips, and a
    sidelobe two chips away can then pass the SNR gate."""
    need = len(ref) + max_lag
    if len(bb) < need:
        bb = np.concatenate([bb, np.zeros(need - len(bb), dtype=bb.dtype)])
    power_cs = np.concatenate([[0.0], np.cumsum(np.abs(bb) ** 2)])
    lags = np.arange(max_lag + 1)
    zs, bounds = [], []
    for s in range(0, len(ref), seg_len):
        seg = ref[s:s + seg_len]
        if len(seg) < max(1, seg_len // 2):
            break  # drop a short tail segment rather than let it add mostly noise
        zs.append(np.correlate(bb[s:s + len(seg) + max_lag], seg, mode="valid"))
        bb_energy = power_cs[s + lags + len(seg)] - power_cs[s + lags]
        bounds.append(np.sqrt(float(np.dot(seg, seg)) * bb_energy))

    if len(zs) == 1:  # too short to difference -- plain coherent correlation
        num, den = np.abs(zs[0]), bounds[0]
    else:
        num = np.abs(sum(zs[k] * np.conj(zs[k - 1]) for k in range(1, len(zs))))
        den = sum(bounds[k] * bounds[k - 1] for k in range(1, len(zs)))
        num, den = np.sqrt(num), np.sqrt(den)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 1e-20)


def detect_probe(recorded: np.ndarray, challenge: Challenge, config: AcousticConfig) -> ProbeDetection:
    fs = float(config.sample_rate_hz)
    q = max(1, int(round(fs * config.lag_step_ms / 1000.0)))
    bb_rate = fs / q
    bandwidth = max(config.baseband_bandwidth_hz, 2.0 * challenge.chip_rate_hz)
    sos = scipy.signal.butter(4, bandwidth, btype="low", fs=fs, output="sos")

    ref = scipy.signal.sosfiltfilt(sos, _chip_envelope(challenge, config))[::q]
    max_lag = int(round(config.lag_search_max_ms / 1000.0 * bb_rate))
    seg_len = max(1, int(round(config.segment_s * bb_rate)))

    x = recorded.astype(np.float64)
    x = x - x.mean()

    carrier_bb = _baseband(x, config.carrier_hz, fs, sos, q)
    stat = _segmented_statistic(carrier_bb, ref, seg_len, max_lag)
    best = int(np.argmax(stat))
    peak = float(stat[best])

    noise_stats, noise_powers = [], []
    for offset in config.noise_offsets_hz:
        bb = _baseband(x, config.carrier_hz + offset, fs, sos, q)
        noise_stats.append(float(np.max(_segmented_statistic(bb, ref, seg_len, max_lag))))
        noise_powers.append(float(np.mean(np.abs(bb) ** 2)))
    off_carrier_floor = float(np.mean(noise_stats)) if noise_stats else 0.0
    noise_power = float(np.mean(noise_powers)) if noise_powers else 0.0

    # Off-carrier channels contain no probe, so they can't see the code's
    # own sidelobes: when the probe IS heard but its true lag lies outside
    # the search window, a wrong alignment a couple of chips away still
    # correlates partially. Decoy codes on the carrier channel measure
    # exactly that "heard, but wrong code/alignment" level.
    decoy_stats = []
    for decoy in _decoy_challenges(challenge, config.n_decoy_codes):
        decoy_ref = scipy.signal.sosfiltfilt(sos, _chip_envelope(decoy, config))[::q]
        decoy_stats.append(float(np.max(_segmented_statistic(carrier_bb, decoy_ref, seg_len, max_lag))))
    decoy_floor = float(np.mean(decoy_stats)) if decoy_stats else 0.0
    noise_floor = max(off_carrier_floor, decoy_floor)
    carrier_power = float(np.mean(np.abs(carrier_bb) ** 2))

    snr_db = float(20 * np.log10(peak / noise_floor)) if noise_floor > 0 and peak > 0 else float("nan")
    tone_db = (float(10 * np.log10(carrier_power / noise_power))
               if noise_power > 0 and carrier_power > 0 else float("nan"))
    score = float(np.clip((peak - noise_floor) / (1.0 - noise_floor), 0.0, 1.0)) if noise_floor < 1.0 else 0.0

    return ProbeDetection(score=score, peak=peak, noise_floor=noise_floor,
                          off_carrier_floor=off_carrier_floor, decoy_floor=decoy_floor,
                          lag_ms=best * q / fs * 1000.0, snr_db=snr_db, tone_db=tone_db,
                          peak_at_edge=best >= max_lag - 1)


def _decoy_challenges(challenge: Challenge, n: int) -> list:
    """n cyclic shifts of the session's own m-sequence, evenly spread over
    its full period (same construction as challenge.derive_zone_challenges,
    whose docstring covers why shifts of one m-sequence are near-orthogonal)."""
    period = (1 << challenge.order) - 1
    if n <= 0 or period < 2 * (n + 1):
        return []
    full = Challenge(chip_rate_hz=challenge.chip_rate_hz, order=challenge.order, seed=challenge.seed,
                     duration_s=period / challenge.chip_rate_hz)
    decoys = []
    for i in range(1, n + 1):
        chips = np.roll(full.chips, -((i * period) // (n + 1)))[:challenge.n_chips].copy()
        decoys.append(Challenge(chip_rate_hz=challenge.chip_rate_hz, duration_s=challenge.duration_s,
                                order=challenge.order, seed=challenge.seed, chips=chips))
    return decoys


def acoustic_confidence(snr_db: float) -> float:
    """Shared by session.acoustic_lane_result and eval/ablate.py so a live
    session and an offline-analysed log agree on what confidence means.
    Non-finite SNR gets the floor, never the ceiling -- an inf here is
    exactly the old start-up-zeros artefact, not a clean detection."""
    if snr_db is None or not np.isfinite(snr_db):
        return 0.05
    return float(np.clip(snr_db / 20.0, 0.05, 1.0))


def calibrate_loopback_latency_ms(config: AcousticConfig, n_trials: int = 3) -> float:
    """Run-once-per-host calibration: with the speaker piped as directly
    as possible into the mic (or physically adjacent, no face in the
    beam), the measured lag IS the fixed electrical/buffer round trip.
    Median of the trials that actually detected the probe. See module
    docstring -- this offset is not yet subtracted automatically."""
    challenge = Challenge(chip_rate_hz=20.0, duration_s=2.0)  # short, fast probe is enough
    probe = synthesize_probe(challenge, config)
    lags = []
    for _ in range(n_trials):
        recorded, _latency_s, xruns = DuplexProbePlayer(probe, config).run()
        if xruns:
            warnings.warn(f"xrun during calibration trial: {xruns}")
        det = detect_probe(recorded, challenge, config)
        if np.isfinite(det.snr_db) and det.snr_db >= config.snr_floor_db and not det.peak_at_edge:
            lags.append(det.lag_ms)
        else:
            warnings.warn(f"calibration trial did not detect the probe (snr_db={det.snr_db:.1f}, "
                          f"tone_db={det.tone_db:.1f})")
    return float(np.median(lags)) if lags else float("nan")


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

    out_name = _device_name(config.output_device, "output")
    in_name = _device_name(config.input_device, "input")
    device_warning = ""
    if any(h in out_name.lower() for h in _HEADPHONE_HINTS):
        device_warning = (f", WARNING output '{out_name}' looks like headphones -- the mic can't hear "
                          f"the probe; set acoustic.output_device")
        print(f"WARNING: acoustic lane output device '{out_name}' looks like headphones/earbuds; "
              f"the microphone cannot hear the probe. Set acoustic.output_device in config.yaml "
              f"(python -m praesens.acoustic --list-devices).")

    probe = synthesize_probe(challenge, config)
    player = DuplexProbePlayer(probe, config)
    try:
        recorded, _latency_s, xruns = player.run()
    except Exception as e:
        return AcousticResult(status="no_evidence", diagnostics=f"stream error: {e}")

    if len(recorded) < len(probe):
        return AcousticResult(status="no_evidence",
                               diagnostics=f"short capture: {len(recorded)}/{len(probe)} samples")
    if float(np.std(recorded)) < 1e-9:
        return AcousticResult(status="no_evidence",
                               diagnostics=f"silent capture from input '{in_name}' (all zeros) -- check "
                                           f"microphone permission / input device")

    det = detect_probe(recorded, challenge, config)
    offset = player.stream_offset_ms
    diagnostics = (f"snr_db={det.snr_db:.2f}, tone_db={det.tone_db:.1f}, peak={det.peak:.2f}, "
                   f"off_carrier_floor={det.off_carrier_floor:.2f}, decoy_floor={det.decoy_floor:.2f}, "
                   f"out='{out_name}', in='{in_name}'"
                   + (f", stream_offset_ms={offset:.1f}" if offset is not None and np.isfinite(offset) else "")
                   + (f", xruns={len(xruns)}" if xruns else "")
                   + device_warning)

    if not np.isfinite(det.tone_db) or det.tone_db < 3.0:
        # no more energy at the carrier than beside it: the mic never heard the tone, which is a
        # speaker/routing problem (muted or 0% output, audio going to earbuds), not a detection one
        diagnostics += (", tone not heard -- check the output device isn't muted/at 0% volume "
                        "and isn't headphones")
    if det.peak_at_edge:
        return AcousticResult(status="insufficient_signal", snr_db=det.snr_db,
                               diagnostics=f"peak at lag-search edge ({det.lag_ms:.0f}ms), {diagnostics}")
    if not np.isfinite(det.snr_db) or det.snr_db < config.snr_floor_db:
        return AcousticResult(status="insufficient_signal", snr_db=det.snr_db, diagnostics=diagnostics)

    return AcousticResult(status="ok", score=det.score, lag_ms=det.lag_ms, snr_db=det.snr_db,
                           diagnostics=diagnostics)


if __name__ == "__main__":
    import argparse
    from pathlib import Path
    import yaml

    parser = argparse.ArgumentParser(description="Acoustic lane smoke test")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--chip-rate", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--input-device", default=None, help="overrides acoustic.input_device")
    parser.add_argument("--output-device", default=None, help="overrides acoustic.output_device")
    parser.add_argument("--list-devices", action="store_true", help="print PortAudio devices and exit")
    parser.add_argument("--calibrate", action="store_true",
                         help="run loopback calibration instead of a normal probe")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices() if sd is not None else f"sounddevice unavailable: {_SD_IMPORT_ERROR}")
        raise SystemExit(0)

    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "config.yaml"
    raw = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            raw = yaml.safe_load(f) or {}
    aconfig = AcousticConfig.from_dict(raw.get("acoustic", {}))
    for attr, value in (("input_device", args.input_device), ("output_device", args.output_device)):
        if value is not None:
            setattr(aconfig, attr, int(value) if value.lstrip("-").isdigit() else value)

    if args.calibrate:
        offset_ms = calibrate_loopback_latency_ms(aconfig)
        print(f"loopback floor: {offset_ms:.2f} ms -- subtract this from real-session "
              f"lag_ms before treating it as physical time-of-flight (see module docstring)")
    else:
        challenge = Challenge(chip_rate_hz=args.chip_rate, duration_s=args.seconds, seed=args.seed)
        print(f"emitting acoustic probe: carrier={aconfig.carrier_hz}Hz "
              f"chip_rate={challenge.chip_rate_hz}Hz duration={challenge.duration_s}s "
              f"seed={challenge.seed}")
        result = run_acoustic_session(challenge, aconfig)
        print(f"status={result.status} score={result.score} lag_ms={result.lag_ms} "
              f"snr_db={result.snr_db:.2f} diagnostics={result.diagnostics!r}")
