"""Milestone 12 acceptance tests for demo/panel.py's collapse-latency
instrumentation: timestamp a source switch, timestamp the first moment the
rolling verdict genuinely crosses to REJECT afterward, and log the delta --
reported as a distribution, not one anecdote.

PanelDashboard.__init__ opens real cameras and a real MediaPipe landmarker,
so it can't be constructed here. Instead these tests call the unbound
_check_collapse_latency/_save_collapse_latencies methods against a minimal
stand-in object carrying just the attributes those methods touch -- the
same "test the pure logic, not the hardware" split used throughout this
repo (e.g. attacks/adaptive_injector.py's luminance_multiply, praesens/
typing.py's compute_typing_status)."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from demo.panel import PanelDashboard, PanelState, collapse_latency_stats
from praesens.challenge import Challenge, derive_zone_challenges


def _fake_dashboard(score_threshold: float = 0.6, tmp_path=None):
    fake = SimpleNamespace(
        collapse_latencies=[],
        _pending_switch_time=None,
        face_recently_detected=True,
        current_score=1.0,
        dconfig=SimpleNamespace(score_threshold=score_threshold),
    )
    return fake


def test_collapse_latency_stats_empty():
    assert collapse_latency_stats([]) == {"n": 0}


def test_collapse_latency_stats_summarizes_distribution():
    stats = collapse_latency_stats([0.1, 0.2, 0.3, 0.4])
    assert stats["n"] == 4
    assert stats["mean_s"] == pytest.approx(0.25)
    assert stats["min_s"] == pytest.approx(0.1)
    assert stats["max_s"] == pytest.approx(0.4)
    assert stats["median_s"] == pytest.approx(0.25)


def test_check_collapse_latency_records_delta_on_first_reject_after_switch():
    fake = _fake_dashboard()
    fake._pending_switch_time = time.perf_counter() - 0.5  # switch happened 0.5s ago
    fake.current_score = 0.1  # below threshold -> REJECT
    fake.face_recently_detected = True

    PanelDashboard._check_collapse_latency(fake)

    assert len(fake.collapse_latencies) == 1
    assert fake.collapse_latencies[0] == pytest.approx(0.5, abs=0.05)
    assert fake._pending_switch_time is None  # disarmed after recording


def test_check_collapse_latency_does_nothing_when_no_switch_pending():
    fake = _fake_dashboard()
    fake._pending_switch_time = None
    fake.current_score = 0.0

    PanelDashboard._check_collapse_latency(fake)

    assert fake.collapse_latencies == []


def test_check_collapse_latency_ignores_no_face_gap():
    """A dropped/no-face frame must not masquerade as a fast collapse --
    the delta should only be recorded once a face is genuinely visible and
    scoring low, not merely absent."""
    fake = _fake_dashboard()
    fake._pending_switch_time = time.perf_counter() - 0.2
    fake.current_score = 0.0
    fake.face_recently_detected = False

    PanelDashboard._check_collapse_latency(fake)

    assert fake.collapse_latencies == []
    assert fake._pending_switch_time is not None  # still armed -- not consumed by a no-face gap


def test_check_collapse_latency_waits_for_score_to_actually_cross_threshold():
    fake = _fake_dashboard()
    fake._pending_switch_time = time.perf_counter()
    fake.current_score = 0.9  # still above threshold -- attack hasn't collapsed the score yet

    PanelDashboard._check_collapse_latency(fake)

    assert fake.collapse_latencies == []
    assert fake._pending_switch_time is not None


def test_check_collapse_latency_records_only_once_per_switch():
    fake = _fake_dashboard()
    fake._pending_switch_time = time.perf_counter() - 0.1
    fake.current_score = 0.1

    PanelDashboard._check_collapse_latency(fake)
    first_len = len(fake.collapse_latencies)
    # a second low-score frame with no new switch armed must not add another entry
    PanelDashboard._check_collapse_latency(fake)

    assert len(fake.collapse_latencies) == first_len == 1


def test_save_collapse_latencies_writes_json_with_stats(tmp_path, monkeypatch):
    import demo.panel as panel_mod

    monkeypatch.setattr(panel_mod, "REPO_ROOT", tmp_path)
    fake = SimpleNamespace(collapse_latencies=[0.2, 0.3])

    PanelDashboard._save_collapse_latencies(fake)

    out_files = list((tmp_path / "logs").glob("collapse_latency_*.json"))
    assert len(out_files) == 1
    record = json.loads(out_files[0].read_text())
    assert record["latencies_s"] == [0.2, 0.3]
    assert record["stats"]["n"] == 2


def test_save_collapse_latencies_writes_honest_empty_log(tmp_path, monkeypatch):
    """n=0 must still be written -- a session with no genuine collapse is a
    real (empty) result, not indistinguishable from 'the demo never ran'."""
    import demo.panel as panel_mod

    monkeypatch.setattr(panel_mod, "REPO_ROOT", tmp_path)
    fake = SimpleNamespace(collapse_latencies=[])

    PanelDashboard._save_collapse_latencies(fake)

    out_files = list((tmp_path / "logs").glob("collapse_latency_*.json"))
    assert len(out_files) == 1
    record = json.loads(out_files[0].read_text())
    assert record["stats"] == {"n": 0}


# ---------------------------------------------------------------------------
# Milestone 14: spatial score computation (_update_spatial_scores) --
# reuses tests/test_spatial.py's own fixture pattern (real Challenge/
# derive_zone_challenges/compute_correlation_matrix, no camera or display
# needed) to build a fake dashboard carrying just what the method touches.
# ---------------------------------------------------------------------------

from praesens.spatial import SpatialConfig  # noqa: E402

_ZONE_NAMES = ("left", "right", "top")
_ROI_TO_ZONE = {"forehead": "top", "left_cheek": "left", "right_cheek": "right"}


def _make_zone_log(challenge, sample_rate_hz=100.0):
    t = np.arange(0, challenge.duration_s, 1.0 / sample_rate_hz)
    vals = challenge.value_at(t)
    return [{"t": float(tt), "chip_value": int(v)} for tt, v in zip(t, vals)]


def _make_global_log(zone_logs):
    names = list(zone_logs.keys())
    n = len(zone_logs[names[0]])
    out = []
    for i in range(n):
        t = zone_logs[names[0]][i]["t"]
        mean_chip = float(np.mean([zone_logs[name][i]["chip_value"] for name in names]))
        out.append({"t": t, "chip_value": mean_chip})
    return out


class _FakeSpatialEmitter:
    def __init__(self, zone_logs, global_log, zone_names):
        self._zone_logs = zone_logs
        self._global_log = global_log
        self.zones = {name: None for name in zone_names}

    def get_zone_log(self, name):
        return self._zone_logs[name]

    def get_global_log(self):
        return self._global_log


def _fake_spatial_dashboard(per_roi_traces, zone_logs, global_log):
    return SimpleNamespace(
        per_roi_ts={roi: list(ts) for roi, (ts, lum) in per_roi_traces.items()},
        per_roi_lum={roi: list(lum) for roi, (ts, lum) in per_roi_traces.items()},
        dconfig=SimpleNamespace(min_valid_samples=8, rolling_window_s=25.0, score_threshold=0.5),
        emitter=_FakeSpatialEmitter(zone_logs, global_log, _ZONE_NAMES),
        sconfig=SpatialConfig(zone_names=_ZONE_NAMES, detrend_window_s=1.5,
                               lag_search_max_ms=300, lag_step_ms=5,
                               expected_assignment=_ROI_TO_ZONE),
        spatial_score_threshold=0.15,
        status_message="",
        current_global_score=0.0, current_spatial_score=float("nan"),
        current_score=0.0, current_lag_ms=0.0, current_snr_db=5.0,
    )


@pytest.fixture
def spatial_fixture():
    master = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=17)
    zones = derive_zone_challenges(master, _ZONE_NAMES)
    zone_by_name = {z.zone_name: z.challenge for z in zones}
    zone_logs = {z.zone_name: _make_zone_log(z.challenge) for z in zones}
    global_log = _make_global_log(zone_logs)
    frame_times = np.arange(0.2, 19.5, 1 / 29.97)
    return zone_by_name, zone_logs, global_log, frame_times


def test_update_spatial_scores_genuine_reflectance_no_mismatch(spatial_fixture):
    zone_by_name, zone_logs, global_log, frame_times = spatial_fixture
    rng = np.random.default_rng(0)
    per_roi_traces = {}
    for roi, zone_name in _ROI_TO_ZONE.items():
        true_vals = zone_by_name[zone_name].value_at(frame_times - 0.04)
        measured = 120.0 + 8.0 * true_vals + rng.normal(0, 0.5, len(frame_times))
        per_roi_traces[roi] = (frame_times, measured)
    fake = _fake_spatial_dashboard(per_roi_traces, zone_logs, global_log)

    PanelDashboard._update_spatial_scores(fake, now=19.5)

    assert fake.current_global_score > 0.4
    assert fake.current_spatial_score > 0.4
    assert fake.current_score == fake.current_global_score  # alias kept in sync
    assert fake.status_message == ""  # no mismatch -- both scores agree


def test_update_spatial_scores_flags_mismatch_under_global_multiply_attack(spatial_fixture):
    """The live version of tests/test_spatial.py's core Milestone 9 claim:
    a global-brightness-only attack raises the global score but leaves the
    spatial score dead, and the panel must visibly say so."""
    zone_by_name, zone_logs, global_log, frame_times = spatial_fixture
    rng = np.random.default_rng(1)
    global_vals = np.mean([zone_by_name[n].value_at(frame_times) for n in zone_by_name], axis=0)
    per_roi_traces = {}
    for roi in _ROI_TO_ZONE:
        measured = 120.0 + 8.0 * global_vals + rng.normal(0, 0.5, len(frame_times))
        per_roi_traces[roi] = (frame_times, measured)
    fake = _fake_spatial_dashboard(per_roi_traces, zone_logs, global_log)

    PanelDashboard._update_spatial_scores(fake, now=19.5)

    assert fake.current_global_score >= fake.dconfig.score_threshold
    assert fake.current_spatial_score < fake.spatial_score_threshold
    assert fake.status_message.startswith("GLOBAL PASSES BUT SPATIAL")


def test_update_spatial_scores_clears_a_stale_mismatch_message(spatial_fixture):
    """A mismatch flagged on a previous update must not stick around once
    the scores agree again (e.g. the camera was switched back)."""
    zone_by_name, zone_logs, global_log, frame_times = spatial_fixture
    rng = np.random.default_rng(0)
    per_roi_traces = {}
    for roi, zone_name in _ROI_TO_ZONE.items():
        true_vals = zone_by_name[zone_name].value_at(frame_times - 0.04)
        measured = 120.0 + 8.0 * true_vals + rng.normal(0, 0.5, len(frame_times))
        per_roi_traces[roi] = (frame_times, measured)
    fake = _fake_spatial_dashboard(per_roi_traces, zone_logs, global_log)
    fake.status_message = "GLOBAL PASSES BUT SPATIAL SCORE IS DEAD -- stale from a previous attack step"

    PanelDashboard._update_spatial_scores(fake, now=19.5)

    assert fake.status_message == ""


def test_update_spatial_scores_insufficient_samples_degrades_honestly(spatial_fixture):
    zone_by_name, zone_logs, global_log, frame_times = spatial_fixture
    per_roi_traces = {roi: ([frame_times[0], frame_times[1]], [float("nan"), float("nan")])
                       for roi in _ROI_TO_ZONE}
    fake = _fake_spatial_dashboard(per_roi_traces, zone_logs, global_log)

    PanelDashboard._update_spatial_scores(fake, now=19.5)

    assert fake.current_global_score == 0.0
    assert np.isnan(fake.current_spatial_score)  # no_evidence, not a fabricated zero


def test_update_spatial_scores_missing_an_roi_entirely_does_not_crash(spatial_fixture):
    """In the real PanelDashboard, self.per_roi_ts/per_roi_lum are always
    initialised with ALL of PER_ROI_NAMES as (possibly-empty) deques in
    __init__ -- a key is never actually absent. This test reproduces that
    same invariant (present but empty), the realistic version of 'an ROI
    hasn't been detected yet this window' rather than a dict missing a key
    outright."""
    zone_by_name, zone_logs, global_log, frame_times = spatial_fixture
    per_roi_traces = {"forehead": (list(frame_times), [120.0] * len(frame_times)),
                       "left_cheek": ([], []), "right_cheek": ([], [])}
    fake = _fake_spatial_dashboard(per_roi_traces, zone_logs, global_log)

    PanelDashboard._update_spatial_scores(fake, now=19.5)  # must not raise

    assert fake.current_global_score == 0.0  # untouched -- returned early


# ---------------------------------------------------------------------------
# Milestone 14: typing lane status (_update_typing_status) -- degrades to
# no_evidence honestly when the keystroke listener or hand landmarker never
# started (hard rule: never crash on a missing device), and recovers a real
# coherence score from genuine synthetic timing data otherwise.
# ---------------------------------------------------------------------------

from praesens.typing import keystroke_impulse_density  # noqa: E402


def _fake_typing_dashboard(capture=None, hand_landmarker=None, hand_ts=None, hand_activity=None,
                            unavailable_reason=None):
    return SimpleNamespace(
        _typing_capture=capture,
        _typing_hand_landmarker=hand_landmarker,
        _typing_hand_ts=hand_ts if hand_ts is not None else [],
        _typing_hand_activity=hand_activity if hand_activity is not None else [],
        _typing_unavailable_reason=unavailable_reason,
        tconfig=SimpleNamespace(min_valid_hand_samples=5, passive_no_evidence_after_s=10.0,
                                 coherence_sigma_s=0.15, coherence_lag_max_ms=300.0,
                                 coherence_lag_step_ms=10.0, passive_decay_half_life_s=5.0),
        current_typing_status="no_evidence", current_typing_reason="",
        current_typing_coherence=float("nan"), current_typing_lag_ms=float("nan"),
        current_typing_confidence=0.0, current_typing_n_keystrokes=0,
    )


def test_update_typing_status_no_evidence_when_capture_never_started():
    fake = _fake_typing_dashboard(capture=None, hand_landmarker=object(),
                                   unavailable_reason="keystroke capture unavailable: permission denied")

    PanelDashboard._update_typing_status(fake, now=10.0)

    assert fake.current_typing_status == "no_evidence"
    assert "permission denied" in fake.current_typing_reason
    assert np.isnan(fake.current_typing_coherence)


def test_update_typing_status_no_evidence_when_hand_landmarker_never_loaded():
    fake_capture = SimpleNamespace(get_events=lambda: [], last_keystroke_time=lambda: None)
    fake = _fake_typing_dashboard(capture=fake_capture, hand_landmarker=None,
                                   unavailable_reason="hand landmarker unavailable: model file not found")

    PanelDashboard._update_typing_status(fake, now=10.0)

    assert fake.current_typing_status == "no_evidence"
    assert "model file not found" in fake.current_typing_reason


def test_update_typing_status_recovers_genuine_coherence_from_synthetic_data():
    rng = np.random.default_rng(0)
    keystroke_times = np.arange(0.5, 20.0, 0.8)
    events = [{"t_down": t, "t_up": t + 0.05, "matched_expected": None} for t in keystroke_times]
    fake_capture = SimpleNamespace(get_events=lambda: events,
                                    last_keystroke_time=lambda: keystroke_times[-1])

    hand_ts = list(np.arange(0, 20.0, 1 / 30.0))
    true_lag_s = -0.05
    density = keystroke_impulse_density(events, np.array(hand_ts) - true_lag_s, sigma_s=0.15)
    hand_activity = list(1.0 * density + rng.normal(0, 0.05, len(hand_ts)))

    fake = _fake_typing_dashboard(capture=fake_capture, hand_landmarker=object(),
                                   hand_ts=hand_ts, hand_activity=hand_activity)

    PanelDashboard._update_typing_status(fake, now=20.0)

    assert fake.current_typing_status == "ok"
    assert fake.current_typing_coherence > 0.7
    assert fake.current_typing_n_keystrokes == len(events)


# ---------------------------------------------------------------------------
# Milestone 14: --rehearse scripted driver -- advances through the
# configured sequence on a timer with no keypresses, degrades a failing
# step to a status message instead of crashing, and signals completion via
# _handle_key so run()'s existing quit path (Q) is reused unchanged.
# ---------------------------------------------------------------------------

def _fake_rehearse_dashboard(script, transitions=None):
    calls = []
    default_transitions = {name: (lambda n=name: calls.append(n)) for name in ("live", "attack", "replay", "typing")}
    if transitions:
        default_transitions.update(transitions)
    fake = SimpleNamespace(
        _rehearse=True, _rehearse_script=script, _rehearse_index=-1,
        _rehearse_next_switch_t=None, _rehearse_done=False,
        status_message="",
        enter_live=default_transitions["live"], enter_attack=default_transitions["attack"],
        enter_replay=default_transitions["replay"], enter_typing=default_transitions["typing"],
    )
    # _check_rehearse calls self._advance_rehearse() -- bind the real
    # unbound method to this fake so that call resolves, same as it would
    # on an actual PanelDashboard instance.
    fake._advance_rehearse = lambda: PanelDashboard._advance_rehearse(fake)
    return fake, calls


def test_advance_rehearse_steps_through_script_in_order():
    script = [{"state": "live", "duration_s": 0.01, "description": "a"},
              {"state": "attack", "duration_s": 0.01, "description": "b"}]
    fake, calls = _fake_rehearse_dashboard(script)

    PanelDashboard._advance_rehearse(fake)
    assert calls == ["live"]
    assert fake._rehearse_index == 0
    assert not fake._rehearse_done

    PanelDashboard._advance_rehearse(fake)
    assert calls == ["live", "attack"]
    assert fake._rehearse_index == 1

    PanelDashboard._advance_rehearse(fake)  # past the end of the script
    assert fake._rehearse_done is True


def test_advance_rehearse_survives_a_failing_step():
    """Hard rule: a missing device must degrade the display, never crash
    the rehearsal -- a step whose transition raises must be logged to
    status_message and the rehearsal must keep going."""
    def _boom():
        raise RuntimeError("no injected feed source found")

    script = [{"state": "attack", "duration_s": 0.01, "description": "x"}]
    fake, calls = _fake_rehearse_dashboard(script, transitions={"attack": _boom})

    PanelDashboard._advance_rehearse(fake)  # must not raise

    assert "unavailable" in fake.status_message
    assert fake._rehearse_index == 0


def test_check_rehearse_advances_only_after_the_scheduled_time():
    script = [{"state": "live", "duration_s": 0.01, "description": "a"},
              {"state": "attack", "duration_s": 10.0, "description": "b"}]
    fake, calls = _fake_rehearse_dashboard(script)
    PanelDashboard._advance_rehearse(fake)  # arms step 0, schedules the switch ~0.01s out
    assert calls == ["live"]

    fake._rehearse_next_switch_t = time.perf_counter() + 10.0  # not due yet
    PanelDashboard._check_rehearse(fake)
    assert calls == ["live"]  # unchanged -- too early

    fake._rehearse_next_switch_t = time.perf_counter() - 0.01  # now due
    PanelDashboard._check_rehearse(fake)
    assert calls == ["live", "attack"]


def test_check_rehearse_noop_when_not_rehearsing():
    fake, calls = _fake_rehearse_dashboard([])
    fake._rehearse = False
    PanelDashboard._check_rehearse(fake)  # must not raise despite no script
    assert calls == []


def test_handle_key_exits_when_rehearse_sequence_is_done():
    fake = SimpleNamespace(_rehearse=True, _rehearse_done=True)
    assert PanelDashboard._handle_key(fake, ord('1'), frame=None) is False


def test_handle_key_normal_path_unaffected_when_not_rehearsing():
    fake, calls = _fake_rehearse_dashboard([])
    fake._rehearse = False
    fake._rehearse_done = False
    # enter_live is set on fake -- '1' should route there, not fall through
    result = PanelDashboard._handle_key(fake, ord('1'), frame=None)
    assert result is True
    assert calls == ["live"]
