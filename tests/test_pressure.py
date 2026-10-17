"""Keystroke-impact (pressure) lane -- synthetic microphone recordings.

Each test builds a recording with impact transients (exponentially decaying
broadband bursts) at known instants relative to a list of character-free
keystroke events, then checks the lane separates genuine typing from
keystrokes that have no physical impact behind them (injection)."""
import numpy as np
import pytest
import scipy.signal

from praesens.fusion import LaneResult, LANE_LAG_BOUNDS_MS, adjudicate_two_lane
from praesens.pressure import (
    PressureConfig, analyse_keystroke_impacts, detect_impacts, fit_sample_clock, pressure_record,
)

FS = 44100
T0 = 1000.0  # perf_counter time of audio sample 0


def _recording(seed=0, n_keys=30, lag_s=0.12, clicks=True, probe=False, speech=False,
               cover_typing=False, dur=12.0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.002, int(dur * FS))
    t = 1.0 + np.cumsum(rng.lognormal(np.log(0.18), 0.45, n_keys))
    t = t[t < dur - 1.0]

    def click(at, amp):
        i, n = int(at * FS), int(0.012 * FS)
        if i + n < len(x):
            x[i:i + n] += amp * rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.002 * FS))

    if clicks:
        for tk in t:
            click(tk + lag_s + rng.normal(0, 0.004), rng.uniform(0.05, 0.2))
    if cover_typing:  # a person typing for real, but NOT the keys the OS reports
        for tk in 0.5 + np.cumsum(rng.lognormal(np.log(0.18), 0.45, 60)):
            if tk < dur - 0.5:
                click(tk, rng.uniform(0.05, 0.2))
    if probe:  # the acoustic lane's 4 kHz BPSK probe in the same recording
        chips = np.sign(rng.normal(size=100))
        env = np.repeat(chips, int(FS / 1.3))[:len(x)]
        env = np.pad(env, (0, len(x) - len(env)))
        x += 0.3 * env * np.sin(2 * np.pi * 4000 * np.arange(len(x)) / FS)
    if speech:
        sos = scipy.signal.butter(4, [150, 3000], btype="band", fs=FS, output="sos")
        gate = (np.sin(2 * np.pi * 4 * np.arange(len(x)) / FS) > 0.3) * 0.1
        x += scipy.signal.sosfilt(sos, rng.normal(0, 1, len(x))) * gate
    events = [{"t_down": T0 + tk, "t_up": T0 + tk + 0.09, "matched_expected": None} for tk in t]
    return x.astype(np.float32), events


NOTCHES = [4000.0, 8000.0, 12000.0, 16000.0]


@pytest.mark.parametrize("kw", [{}, {"probe": True}, {"lag_s": 0.35}])
def test_genuine_typing_passes(kw):
    audio, events = _recording(**kw)
    res = analyse_keystroke_impacts(audio, T0, FS, events, PressureConfig(), notch_freqs_hz=NOTCHES)
    assert res.status == "ok"
    assert res.score > 0.8
    assert abs(res.lag_ms - kw.get("lag_s", 0.12) * 1000) < 25
    assert res.hit_fraction > 0.9


def test_genuine_typing_survives_speech_in_the_room():
    audio, events = _recording(speech=True)
    res = analyse_keystroke_impacts(audio, T0, FS, events, PressureConfig())
    assert res.status == "ok" and res.score > 0.6


@pytest.mark.parametrize("kw", [{}, {"speech": True}, {"cover_typing": True}, {"probe": True}])
def test_injected_keystrokes_fail(kw):
    audio, events = _recording(clicks=False, **kw)
    res = analyse_keystroke_impacts(audio, T0, FS, events, PressureConfig(), notch_freqs_hz=NOTCHES)
    assert res.status == "ok"  # the mic worked and heard the room: absence of impacts IS the measurement
    assert res.score < 0.2


def test_too_few_keystrokes_is_no_evidence():
    audio, events = _recording(n_keys=3)
    res = analyse_keystroke_impacts(audio, T0, FS, events, PressureConfig())
    assert res.status == "no_evidence" and res.score is None


def test_silent_capture_is_no_evidence():
    _audio, events = _recording()
    res = analyse_keystroke_impacts(np.zeros(FS * 12, dtype=np.float32), T0, FS, events, PressureConfig())
    assert res.status == "no_evidence"


def test_per_key_profile_never_carries_characters():
    audio, events = _recording()
    res = analyse_keystroke_impacts(audio, T0, FS, events, PressureConfig())
    for k in res.per_key:
        assert set(k) == {"t_down", "matched", "force_db", "dwell_s"}
        assert not any(isinstance(v, str) for v in k.values())
    rec = pressure_record(res)
    assert "audio" not in rec and rec["n_keys"] == len(events)


def test_stronger_strikes_read_as_more_force():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.002, FS * 6)
    events = []
    for i, amp in enumerate([0.03, 0.03, 0.03, 0.3, 0.3, 0.3]):
        at = 0.5 + i * 0.6
        n = int(0.012 * FS)
        x[int(at * FS):int(at * FS) + n] += amp * rng.normal(0, 1, n) * np.exp(-np.arange(n) / (0.002 * FS))
        events.append({"t_down": T0 + at - 0.05, "t_up": T0 + at + 0.03, "matched_expected": None})
    res = analyse_keystroke_impacts(x.astype(np.float32), T0, FS, events, PressureConfig())
    forces = [k["force_db"] for k in res.per_key]
    assert all(f is not None for f in forces)
    assert np.mean(forces[3:]) > np.mean(forces[:3]) + 10  # 10x amplitude = +20 dB


def test_detect_impacts_ignores_a_steady_tone():
    t = np.arange(FS * 3) / FS
    onsets = detect_impacts((0.5 * np.sin(2 * np.pi * 4000 * t)).astype(np.float32), FS, PressureConfig(),
                            notch_freqs_hz=[4000.0])
    assert len(onsets.times_s) <= 1


def test_fit_sample_clock_uses_least_delayed_callback():
    # three 512-sample blocks; callbacks arrive 2 ms, 9 ms and 3 ms after their block ended
    anchors = [(10.0 + 512 / FS + 0.002, 512), (10.0 + 1024 / FS + 0.009, 1024), (10.0 + 1536 / FS + 0.003, 1536)]
    assert fit_sample_clock(anchors, FS) == pytest.approx(10.002, abs=1e-6)
    assert fit_sample_clock([], FS) is None


def test_pressure_gates_accept_and_rechallenges_on_injection():
    optical = LaneResult("optical", 0.8, "ok", 30.0, 0.9)
    typing = LaneResult("typing", 1.0, "ok", None, 1.0)
    good = LaneResult("pressure", 0.95, "ok", 130.0, 1.0)
    bad = LaneResult("pressure", 0.02, "ok", 50.0, 1.0)
    assert "pressure" in LANE_LAG_BOUNDS_MS
    assert adjudicate_two_lane(optical, typing, 0.3, 0.7, pressure=good).verdict == "ACCEPT"
    rc = adjudicate_two_lane(optical, typing, 0.3, 0.7, pressure=bad)
    assert rc.verdict == "RE-CHALLENGE" and "pressure" in rc.reason_text
    weak_optical = LaneResult("optical", 0.05, "ok", 30.0, 0.9)
    assert adjudicate_two_lane(weak_optical, typing, 0.3, 0.7, pressure=good).verdict == "REJECT"
    out_of_window = LaneResult("pressure", 0.95, "ok", 900.0, 1.0)
    assert adjudicate_two_lane(optical, typing, 0.3, 0.7, pressure=out_of_window).verdict == "RE-CHALLENGE"
