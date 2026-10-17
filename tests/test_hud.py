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
        samples = []
        while not results and time.time() < t_end:
            samples.append(e.telemetry())
            time.sleep(0.1)
        json.dumps(e.telemetry())  # the websocket payload must always serialise
        # Sample from DURING the scan: once the scan ends the engine passes through
        # mode "scoring", which deliberately blanks landmarks (engine.capture_loop),
        # so a snapshot taken after the result lands reports None about half the time.
        tele = next((x for x in reversed(samples) if x.get("landmarks")), e.telemetry())
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


# ---------------------------------------------------------------------------
# switch_camera: found on the demo laptop (2026-10-17). Clicking the camera
# buttons froze the preview -- the engine opened the NEW capture before
# releasing the old one, and DirectShow hands out exclusive access, so the
# second handle to a camera that is still open reports isOpened() and then
# never delivers a frame. grab() failed forever, the capture loop idled, and
# the server kept re-broadcasting the last JPEG (77 identical frames measured
# over the websocket) while measured_fps stayed frozen at its last value.
# ---------------------------------------------------------------------------

class _FakeCap:
    """Mimics DirectShow exclusivity: a handle only delivers frames while it is
    the only live handle on that index."""
    live: dict = {}

    def __init__(self, index, dead=False):
        self.index, self.dead, self.released = index, dead, False
        self.exclusive = _FakeCap.live.get(index, 0) == 0
        _FakeCap.live[index] = _FakeCap.live.get(index, 0) + 1

    def isOpened(self):
        return not self.dead                    # dead cameras do not open at all
    def grab(self):
        return bool(self.exclusive and not self.dead and not self.released)
    def retrieve(self):
        import numpy as np
        return (True, np.zeros((48, 64, 3), dtype=np.uint8)) if self.grab() else (False, None)
    def set(self, *a, **k):
        return True
    def get(self, *a, **k):
        return 0.0
    def release(self):
        if not self.released:
            self.released = True
            _FakeCap.live[self.index] = max(0, _FakeCap.live.get(self.index, 1) - 1)


@pytest.fixture
def camera_engine(monkeypatch):
    import praesens.camera
    import praesens.capture
    import demo.hud.engine as eng

    _FakeCap.live = {}
    opened = []

    def fake_open(index):
        cap = _FakeCap(index, dead=(index >= 2))     # 0 and 1 exist, 2 does not
        opened.append(cap)
        return cap

    monkeypatch.setattr(praesens.camera, "open_camera", fake_open)
    monkeypatch.setattr(praesens.capture, "configure_capture_format", lambda *a, **k: None)
    monkeypatch.setattr(eng, "lock_camera", lambda *a, **k: True)

    raw = load_config()
    raw["optical"]["camera_warmup_frames"] = 2
    e = HudEngine(raw, synthetic=True)
    e.synthetic = False                          # exercise the real camera path, no hardware
    e.mode = "idle"
    monkeypatch.setattr(e, "_enter_idle", lambda: None)   # isolate camera handling
    e.cap = fake_open(0)
    e.oconfig.camera_index = 0
    return e, opened


def test_switching_to_the_camera_already_in_use_keeps_delivering_frames(camera_engine):
    """The exact click that froze the preview: camera 0 -> camera 0."""
    e, _ = camera_engine
    msg = e.switch_camera(0)
    assert msg == "switched to camera 0"
    assert e.cap.grab() is True, "preview frozen: the new handle never delivers a frame"
    assert e.oconfig.camera_index == 0


def test_switch_camera_releases_the_old_handle_before_opening_the_new_one(camera_engine):
    e, opened = camera_engine
    old = e.cap
    e.switch_camera(1)
    assert old.released is True
    assert e.cap is not old and e.cap.grab() is True
    assert e.oconfig.camera_index == 1


def test_switch_to_a_missing_camera_falls_back_to_the_working_one(camera_engine):
    e, _ = camera_engine
    msg = e.switch_camera(2)                     # no such device
    assert "camera 2" in msg and "2" in msg
    assert e.cap is not None and e.cap.grab() is True, "fell back to a dead capture"
    assert e.oconfig.camera_index == 0


# ---------------------------------------------------------------------------
# A page can be refreshed or closed at any instant -- including mid-scan, or
# while a toast is already in flight. On Windows that surfaces as
# ClientConnectionResetError("Cannot write to closing transport"); unguarded,
# it escaped the websocket handler and took the whole server down with it
# (2026-10-17, while reloading during a camera switch).
# ---------------------------------------------------------------------------

def test_send_to_a_closing_socket_does_not_raise():
    import asyncio
    from demo.hud.server import HudServer

    class DeadWS:
        async def send_str(self, *_a):
            raise ConnectionResetError("Cannot write to closing transport")
        async def send_bytes(self, *_a):
            raise ConnectionResetError("Cannot write to closing transport")

    async def go():
        ok_text = await HudServer._send(DeadWS(), {"type": "toast", "text": "switched to camera 0"})
        ok_bytes = await HudServer._send(DeadWS(), b"\xff\xd8jpeg")
        return ok_text, ok_bytes

    assert asyncio.run(go()) == (False, False)


def test_send_reports_success_on_a_live_socket():
    import asyncio
    from demo.hud.server import HudServer

    sent = []

    class LiveWS:
        async def send_str(self, s):
            sent.append(("str", s))
        async def send_bytes(self, b):
            sent.append(("bytes", b))

    async def go():
        return (await HudServer._send(LiveWS(), {"a": 1}), await HudServer._send(LiveWS(), b"xy"))

    assert asyncio.run(go()) == (True, True)
    assert sent[0][0] == "str" and json.loads(sent[0][1]) == {"a": 1}
    assert sent[1] == ("bytes", b"xy")
