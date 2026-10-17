"""PRAESENS HUD -- browser dashboard + emitter.

    python -m demo.hud                 # real camera / mic / keyboard
    python -m demo.hud --synthetic     # no hardware: simulated sensors, clearly labelled
    python -m demo.hud --lanes optical+typing+acoustic+pressure

Opens http://127.0.0.1:8765 . Press F11 for fullscreen -- the page's edge
strips are the optical challenge emitter, so the browser window must be the
thing lighting your face (same as the OpenCV emitter window was).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import webbrowser
from pathlib import Path

from aiohttp import WSMsgType, web

from praesens.session import load_config
from demo.hud.engine import HudEngine

STATIC = Path(__file__).resolve().parent / "static"


class HudServer:
    def __init__(self, engine: HudEngine):
        self.engine = engine
        self.clients: dict = {}       # ws -> {"role": str}
        self.master = None            # the client whose page is the emitter
        self.loop: asyncio.AbstractEventLoop | None = None
        engine.listeners.append(self._from_engine)

    # engine threads -> all clients
    def _from_engine(self, msg: dict) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(asyncio.ensure_future, self._broadcast_json(msg))

    async def _broadcast_json(self, msg: dict) -> None:
        data = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                pass

    async def index(self, request):
        return web.FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    async def ws_handler(self, request):
        ws = web.WebSocketResponse(heartbeat=20, max_msg_size=8 * 1024 * 1024)
        await ws.prepare(request)
        self.clients[ws] = {"role": "viewer"}
        e = self.engine
        await ws.send_str(json.dumps({
            "type": "hello", "synthetic": e.synthetic, "lanes": e.lanes,
            "border_fraction": float(e.hud.get("border_fraction", 0.13)),
            "base_luminance": e.econfig.base_luminance,
        }))
        for em in (e.emitter, e.scan["emitter"] if e.scan else None):
            if em is not None:
                await ws.send_str(json.dumps(em.describe()))
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    data = json.loads(msg.data)
                except ValueError:
                    continue
                await self._handle(ws, data)
        finally:
            self.clients.pop(ws, None)
            if self.master is ws:
                self.master = None
        return ws

    async def _handle(self, ws, data: dict) -> None:
        kind = data.get("type")
        e = self.engine
        if kind == "ping":
            await ws.send_str(json.dumps({"type": "pong", "c": data.get("c"), "s": time.perf_counter()}))
        elif kind == "role" and data.get("role") == "emitter":
            self.master = ws  # latest page to claim it drives the log
            self.clients[ws]["role"] = "emitter"
        elif kind == "emitlog" and ws is self.master:
            cid = data.get("id")
            for em in (e.emitter, e.scan["emitter"] if e.scan else None):
                if em is not None and em.id == cid:
                    em.ingest(data.get("entries", []))
        elif kind == "scan":
            await ws.send_str(json.dumps({"type": "toast", "text": e.start_scan()}))
        elif kind == "dismiss":
            e.dismiss()
        elif kind == "camera":
            text = await asyncio.get_running_loop().run_in_executor(None, e.switch_camera, int(data.get("index", 0)))
            await ws.send_str(json.dumps({"type": "toast", "text": text}))
        elif kind == "sim":
            e.set_sim(str(data.get("key")), bool(data.get("value")))

    async def telemetry_task(self):
        period = 1.0 / self.engine.stream_fps
        while True:
            await asyncio.sleep(period)
            if not self.clients:
                continue
            try:
                msg = json.dumps(self.engine.telemetry())
            except Exception as exc:  # never kill the stream over one bad frame
                msg = json.dumps({"type": "toast", "text": f"telemetry error: {exc}"})
            jpeg = self.engine.frame_jpeg
            for ws in list(self.clients):
                try:
                    await ws.send_str(msg)
                    if jpeg:
                        await ws.send_bytes(jpeg)
                except Exception:
                    pass

    async def on_startup(self, app):
        self.loop = asyncio.get_running_loop()
        app["telemetry"] = asyncio.ensure_future(self.telemetry_task())

    async def on_cleanup(self, app):
        app["telemetry"].cancel()
        self.engine.shutdown()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/", self.index)
        app.router.add_get("/ws", self.ws_handler)
        app.router.add_static("/static/", STATIC, show_index=False)
        app.on_startup.append(self.on_startup)
        app.on_cleanup.append(self.on_cleanup)
        return app


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="PRAESENS sci-fi HUD (browser dashboard + emitter)")
    ap.add_argument("--synthetic", action="store_true",
                    help="simulated camera/mic/keyboard for UI work -- every screen is labelled SYNTHETIC")
    ap.add_argument("--camera", type=int, default=None, help="camera index (overrides optical.camera_index)")
    ap.add_argument("--lanes", default=None, help="e.g. optical+typing+pressure (default: hud.lanes)")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--host", default=None)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args(argv)

    raw = load_config()
    hud = raw.get("hud", {})
    engine = HudEngine(raw, synthetic=args.synthetic, camera_index=args.camera, lanes=args.lanes)
    engine.open_sensors()
    threading.Thread(target=engine.capture_loop, daemon=True).start()
    threading.Thread(target=engine.analysis_loop, daemon=True).start()

    host = args.host or hud.get("host", "127.0.0.1")
    port = args.port or int(hud.get("port", 8765))
    url = f"http://{host}:{port}/"
    print(f"PRAESENS HUD on {url}  ({'SYNTHETIC sensors' if args.synthetic else 'real sensors'}) -- "
          f"F11 for fullscreen; the page border is the emitter")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    web.run_app(HudServer(engine).app(), host=host, port=port, print=None)


if __name__ == "__main__":
    main()
