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

from demo.panel import PanelDashboard, collapse_latency_stats


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
