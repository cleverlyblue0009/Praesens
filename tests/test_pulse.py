"""rPPG pulse lane -- synthetic skin-colour traces.

A genuine face: skin RGB carries a small pulsatile component along the
blood-volume direction, PLUS a much larger achromatic brightness change
driven by the optical challenge (the screen flashing), PLUS noise. The lane
must recover the heart rate through that, and must NOT find a pulse in a
"photo" that only carries the challenge modulation."""
import numpy as np
import pytest

from praesens.challenge import Challenge
from praesens.pulse import PulseConfig, estimate_pulse, pos_signal, pulse_record, skin_rgb

FPS = 16.0
PBV = np.array([0.33, 0.77, 0.53])  # blood-volume pulse signature in normalised RGB (Wang et al.)


def _trace(bpm=72.0, pulse=True, dur=20.0, seed=0, chip_depth=0.04, emitter=True, frame_jitter=0.004):
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.full(int(dur * FPS), 1 / FPS) + rng.normal(0, frame_jitter, int(dur * FPS))) + 50.0
    ch = Challenge(chip_rate_hz=1.33, duration_s=dur, seed=11)
    log = [{"t": 50.0 + k / 60.0, "chip_value": int(ch.value_at(k / 60.0)), "enabled": True}
           for k in range(int(dur * 60))]
    base = np.array([175.0, 128.0, 108.0])
    phase = 2 * np.pi * bpm / 60.0 * (t - t[0])
    ppg = 0.004 * (np.sin(phase) + 0.3 * np.sin(2 * phase + 0.5)) if pulse else 0.0 * t
    chips = np.array([ch.value_at(x - 50.0 - 0.06) for x in t], dtype=float) if emitter else 0 * t
    rgb = base * (1 + np.outer(ppg, PBV)) * (1 + chip_depth * chips)[:, None]
    rgb += rng.normal(0, 0.25, rgb.shape)
    return t, rgb, log


def test_recovers_heart_rate_through_the_challenge_flashing():
    t, rgb, log = _trace(bpm=72.0)
    est = estimate_pulse(t, rgb, PulseConfig(window_s=0), emitter_log=log)
    assert est.status == "ok"
    assert est.bpm == pytest.approx(72.0, abs=4.0)
    assert est.snr_db > 0
    assert est.subscore > 0.5


@pytest.mark.parametrize("bpm", [55.0, 95.0, 130.0])
def test_tracks_different_rates(bpm):
    t, rgb, log = _trace(bpm=bpm, seed=int(bpm))
    est = estimate_pulse(t, rgb, PulseConfig(window_s=0), emitter_log=log)
    assert est.bpm == pytest.approx(bpm, abs=5.0)


def test_photo_has_no_pulse():
    t, rgb, log = _trace(pulse=False)
    est = estimate_pulse(t, rgb, PulseConfig(window_s=0), emitter_log=log)
    assert est.status == "ok"
    assert est.subscore < 0.35


def test_live_window_uses_only_recent_samples():
    t, rgb, log = _trace(dur=30.0)
    est = estimate_pulse(t, rgb, PulseConfig(window_s=10.0), emitter_log=log)
    assert est.status == "ok"
    assert est.wave_t[0] >= t[-1] - 10.0 - 1e-6


def test_no_face_is_no_evidence():
    t, rgb, log = _trace()
    rgb[:] = np.nan
    est = estimate_pulse(t, rgb, PulseConfig(window_s=0), emitter_log=log)
    assert est.status == "no_evidence" and est.subscore is None
    assert pulse_record(est)["status"] == "no_evidence"


def test_too_short_is_no_evidence():
    t, rgb, log = _trace(dur=3.0)
    assert estimate_pulse(t, rgb, PulseConfig(window_s=0)).status == "no_evidence"


def test_pos_rejects_pure_intensity_change():
    n = 300
    intensity = 1 + 0.05 * np.sign(np.sin(np.linspace(0, 40, n)))
    rgb = np.array([170.0, 125.0, 105.0])[None, :] * intensity[:, None]
    h = pos_signal(rgb, 30.0, 1.6)
    assert np.std(h) < 1e-9


def test_skin_rgb_reads_rgb_order():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[..., 0], frame[..., 1], frame[..., 2] = 10, 20, 30  # BGR
    assert skin_rgb(frame, [(5, 5, 20, 20)]) == (30.0, 20.0, 10.0)
    assert all(np.isnan(v) for v in skin_rgb(frame, []))
