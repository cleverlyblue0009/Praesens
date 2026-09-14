"""Acoustic lane detector, exercised on synthetic recordings (no audio
hardware). The first real 3-lane sessions (logs/20260913T20*.json) failed
two ways these tests pin down: a heard probe scored ~0 once speaker/mic
clocks drifted or a buffer dropped, and an UNheard probe was reported as
status=ok with snr_db=inf because the recording's start-up zeros were
taken as the noise floor."""
from unittest.mock import patch

import numpy as np
import pytest

import praesens.acoustic as acoustic_mod
from praesens.acoustic import (
    AcousticConfig, AcousticResult, acoustic_confidence, detect_probe, run_acoustic_session,
    synthesize_probe,
)
from praesens.challenge import Challenge

FS = 44100
STARTUP_ZEROS = int(0.025 * FS)  # input streams deliver silence for their first buffers


@pytest.fixture(scope="module")
def setup():
    config = AcousticConfig(carrier_hz=4000.0)
    challenge = Challenge(chip_rate_hz=3.0, duration_s=10.0, seed=18)
    probe = synthesize_probe(challenge, config).astype(np.float64)
    return config, challenge, probe


def _recording(probe, config, delay_ms=10.0, gain=0.05, noise_rms=0.01, drift_ppm=0.0,
               drop_block_at=None, include_probe=True, clicks=False, seed=0):
    rng = np.random.default_rng(seed)
    n = len(probe) + int((config.lag_search_max_ms / 1000.0 + 0.25) * FS)
    rec = rng.standard_normal(n) * noise_rms
    if include_probe:
        src = probe
        if drift_ppm:
            src = np.interp(np.arange(len(probe)) * (1 + drift_ppm * 1e-6), np.arange(len(probe)), probe,
                            right=0.0)
        if drop_block_at is not None:
            src = np.concatenate([src[:drop_block_at], src[drop_block_at + 512:]])
        d = int(delay_ms / 1000.0 * FS)
        m = min(len(src), n - d)
        rec[d:d + m] += gain * src[:m]
    if clicks:  # loud broadband keystroke transients, no probe content
        for pos in rng.integers(STARTUP_ZEROS, n - 2000, size=40):
            rec[pos:pos + 400] += 0.3 * rng.standard_normal(400) * np.exp(-np.arange(400) / 60)
    rec[:STARTUP_ZEROS] = 0.0
    return rec.astype(np.float32)


def _passes(det, config):
    return np.isfinite(det.snr_db) and det.snr_db >= config.snr_floor_db and not det.peak_at_edge


def test_heard_probe_is_detected_at_the_right_lag(setup):
    config, challenge, probe = setup
    det = detect_probe(_recording(probe, config, delay_ms=40.0), challenge, config)
    assert _passes(det, config)
    assert det.score > 0.6
    assert det.lag_ms == pytest.approx(40.0, abs=3.0)


def test_inverted_polarity_is_still_detected(setup):
    config, challenge, probe = setup
    det = detect_probe(_recording(probe, config, gain=-0.05), challenge, config)
    assert _passes(det, config)


@pytest.mark.parametrize("kwargs", [dict(drift_ppm=50.0), dict(drop_block_at=200_000)],
                         ids=["50ppm_clock_drift", "dropped_buffer"])
def test_clock_drift_and_xruns_do_not_cancel_the_detection(setup, kwargs):
    """The old whole-session coherent correlation scored 0.013 at 50ppm
    and 0.12 for one dropped 512-sample block."""
    config, challenge, probe = setup
    det = detect_probe(_recording(probe, config, **kwargs), challenge, config)
    assert _passes(det, config)
    assert det.score > 0.5


def test_round_trip_longer_than_the_old_60ms_window_is_found(setup):
    config, challenge, probe = setup
    det = detect_probe(_recording(probe, config, delay_ms=250.0), challenge, config)
    assert _passes(det, config)
    assert det.lag_ms == pytest.approx(250.0, abs=5.0)


def test_lag_beyond_the_search_window_is_not_scored_at_a_sidelobe(setup):
    """The probe IS heard but arrives past lag_search_max_ms: the best
    in-window lag is then a code sidelobe (or the window edge), and must
    not pass as a detection at a wrong lag."""
    config, challenge, probe = setup
    det = detect_probe(_recording(probe, config, delay_ms=700.0), challenge, config)
    assert not _passes(det, config)


def test_unheard_probe_with_startup_zeros_and_typing_is_not_ok(setup):
    """Regression for snr_db=inf: start-up zeros + keystroke clicks and no
    probe at all must NOT pass the SNR gate."""
    config, challenge, probe = setup
    det = detect_probe(_recording(probe, config, include_probe=False, clicks=True), challenge, config)
    assert np.isfinite(det.snr_db)
    assert not _passes(det, config)
    assert det.score < 0.2


def test_smoothed_chip_edges_keep_probe_energy_out_of_the_noise_channels(setup):
    """Abrupt BPSK flips put probe energy ~-32dB into the +/-300-600Hz
    noise channels, which on real laptop speakers inflated the noise floor
    to 0.37 and capped snr_db near 9dB; 20ms smoothing takes it to ~-77dB."""
    config, challenge, probe = setup
    freqs = np.fft.rfftfreq(len(probe), 1 / FS)
    power = np.abs(np.fft.rfft(probe)) ** 2
    offset = np.abs(freqs - config.carrier_hz)
    main_band = power[offset < 20].sum()
    noise_band = power[(offset >= 280) & (offset < 620)].sum()
    assert 10 * np.log10(noise_band / main_band) < -60.0


def test_acoustic_confidence_never_rewards_non_finite_snr():
    assert acoustic_confidence(float("inf")) == 0.05
    assert acoustic_confidence(float("nan")) == 0.05
    assert acoustic_confidence(10.0) == pytest.approx(0.5)


class _FakePlayer:
    recording = None
    last_config = None

    def __init__(self, probe, config, record_extra_s=None):
        _FakePlayer.last_config = config
        self.stream_offset_ms = float("nan")

    def run(self):
        return _FakePlayer.recording, 0.05, []


def _run_with_recording(recording, challenge, config):
    _FakePlayer.recording = recording
    with patch.object(acoustic_mod, "DuplexProbePlayer", _FakePlayer), \
         patch.object(acoustic_mod, "_device_name", return_value="Speakers"):
        return run_acoustic_session(challenge, config)


def test_run_acoustic_session_ok_for_heard_probe(setup):
    config, challenge, probe = setup
    result = _run_with_recording(_recording(probe, config, delay_ms=30.0), challenge, config)
    assert isinstance(result, AcousticResult)
    assert result.status == "ok", result.diagnostics
    assert result.lag_ms == pytest.approx(30.0, abs=3.0)
    assert np.isfinite(result.snr_db)


def test_run_acoustic_session_insufficient_for_unheard_probe(setup):
    config, challenge, probe = setup
    result = _run_with_recording(_recording(probe, config, include_probe=False, clicks=True), challenge, config)
    assert result.status == "insufficient_signal"
    assert result.score is None
    assert "tone not heard" in result.diagnostics


def test_configured_device_missing_on_this_host_falls_back_to_default(setup):
    """config.yaml names one host's devices; on another host (Windows
    Realtek names on a Mac) the lane must fall back to the OS default and
    say so, not fail the stream."""
    config, challenge, probe = setup
    import dataclasses
    other_host = dataclasses.replace(config, output_device="Speakers Realtek MME",
                                     input_device="Microphone Array Realtek MME")
    _FakePlayer.recording = _recording(probe, config, delay_ms=30.0)
    with patch.object(acoustic_mod, "DuplexProbePlayer", _FakePlayer), \
         patch.object(acoustic_mod, "_device_name", return_value="MacBook Speakers"), \
         patch.object(acoustic_mod.sd, "query_devices", side_effect=ValueError("No device matching")):
        result = run_acoustic_session(challenge, other_host)
    assert result.status == "ok", result.diagnostics
    assert _FakePlayer.last_config.output_device is None
    assert _FakePlayer.last_config.input_device is None
    assert "not usable on this host" in result.diagnostics


def test_run_acoustic_session_all_zero_capture_is_no_evidence(setup):
    config, challenge, probe = setup
    silent = np.zeros(len(probe) + FS, dtype=np.float32)
    result = _run_with_recording(silent, challenge, config)
    assert result.status == "no_evidence"
    assert "microphone" in result.diagnostics
