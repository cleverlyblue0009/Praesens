"""HUD engine: browser-driven emitter log + a full synthetic SCAN end to end.

The synthetic world couples the simulated face to whatever the emitter
schedule shows, puts a pulse in the skin and an impact behind each key, so
these tests run the REAL lane code (OpticalFrameProcessor.finalize,
analyse_keystroke_impacts, estimate_pulse, adjudicate_two_lane); each
attack toggle breaks exactly one physical coupling."""
import json
import threading
import time

import pytest

from praesens.challenge import Challenge
from praesens.emit import EmitterConfig
from praesens.session import load_config
from demo.hud.engine import BrowserEmitter, HudEngine


def test_browser_emitter_log_matches_emitter_contract():
    em = BrowserEmitter(Challenge(chip_rate_hz=2.0, duration_s=10.0, seed=5), EmitterConfig(), start_perf=100.0,
                        max_log_s=5.0)
    assert em.ingest([[100.0, 1, 237], [100.016, 1, 237], ["bad"], [100.03, 7, 1], [100.033, -1, 17]]) == 3
    log = em.get_log()
    assert [r["chip_value"] for r in log] == [1, 1, -1]
    assert set(log[0]) >= {"t", "chip_value", "luminance", "enabled"}       # what resample_emitted reads
    em.ingest([[99.0, 1, 237]])                                            # out-of-order batch is dropped
    assert len(em.get_log()) == 3
    em.ingest([[110.0 + k / 60, 1, 237] for k in range(60)])              # trimmed to the last 5 s
    assert em.get_log()[0]["t"] >= 111.0 - 5.0 - 1e-9
    d = em.describe()
    assert d["type"] == "challenge" and len(d["chips"]) == 20 and d["id"] == em.id


def test_adaptive_depth_boost_is_published_to_the_page():
    seen = []
    em = BrowserEmitter(Challenge(chip_rate_hz=2.0, duration_s=5.0), EmitterConfig(modulation_depth=60.0),
                        start_perf=0.0, on_change=seen.append)
    em.set_modulation_depth(500.0)
    assert em.get_modulation_depth() == em.config.max_modulation_depth and em.adaptive_boost_applied()
    assert seen == [em] and em.describe()["depth"] == em.config.max_modulation_depth


def _run_scan(tmp_path, **sim):
    raw = load_config()
    raw["optical"]["auto_chip_rate"] = False
    raw["challenge"] = {"chip_rate_hz": 3.0, "duration_s": 9.0}
    raw["hud"] = {**raw.get("hud", {}), "verdict_hold_s": 0.5, "lanes": "optical+typing+pressure"}
    e = HudEngine(raw, synthetic=True)
    e.out_dir = tmp_path
    e.open_sensors()
    for k, v in sim.items():
        e.set_sim(k, v)
    results = []
    e.listeners.append(lambda m: results.append(m["result"]) if m.get("type") == "result" else None)
    stop = threading.Event()

    def browser():  # what hud.js does every animation frame: paint the schedule, report it
        while not stop.is_set():
            t = time.perf_counter()
            for em in {e.emitter, (e.scan or {}).get("emitter")} - {None}:
                if t >= em.start_perf - 0.5:
                    chip = int(em.challenge.value_at(t - em.start_perf))
                    em.ingest([[t, chip, 127 + 110 * chip]])
            time.sleep(1 / 60)

    threads = [threading.Thread(target=f, daemon=True) for f in (e.capture_loop, e.analysis_loop, browser)]
    for th in threads:
        th.start()
    try:
        time.sleep(1.0)
        assert e.start_scan() == "scan armed"
        t_end = time.time() + 30
        while not results and time.time() < t_end:
            time.sleep(0.1)
        tele = e.telemetry()
        json.dumps(tele)  # the websocket payload must always serialise
    finally:
        stop.set()
        e.shutdown()
    assert results, "scan never produced a result"
    return results[0], tele


def test_synthetic_scan_genuine_accepts_and_writes_a_log(tmp_path):
    r, tele = _run_scan(tmp_path)
    lanes = {l["name"]: l for l in r["lanes"]}
    assert r["verdict"] == "ACCEPT", r["reason"]
    assert lanes["optical"]["passes"] and lanes["pressure"]["passes"] and lanes["typing"]["passes"]
    assert lanes["pulse"]["gated"] is False
    log = json.loads((tmp_path / f"{r['session']}.json").read_text())
    assert log["source"] == "hud" and log["meta"]["synthetic"] is True
    assert log["fusion"]["verdict"] == "ACCEPT"
    assert all(set(ev) == {"t_down", "t_up", "matched_expected"} for ev in log["typing"]["events"])
    assert tele["landmarks"] is not None and len(tele["landmarks"]) == 478 * 3


def test_synthetic_injected_keystrokes_rechallenge(tmp_path):
    r, _ = _run_scan(tmp_path, attack_keys=True)
    assert r["verdict"] == "RE-CHALLENGE"
    assert "pressure" in r["reason"]


def test_synthetic_injected_video_rejects(tmp_path):
    r, _ = _run_scan(tmp_path, attack_video=True)
    assert r["verdict"] == "REJECT"
