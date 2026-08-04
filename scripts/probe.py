"""Milestone 0: hardware probe.

Answers the one question that decides whether the optical liveness approach
is viable on this machine at all: can we get a stable frame rate, can we
lock exposure/white balance so auto-exposure doesn't cancel out the light
pattern we're about to emit, and can the display redraw fast enough to carry
that pattern. Everything downstream (Milestones 1-8) assumes the answers
here are yes. If they're no, the architecture needs to change before any
more code is written.

Usage: python scripts/probe.py [--camera-index 0] [--skip-display]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    cfg_path = REPO_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)["probe"]


# ---------------------------------------------------------------------------
# Camera open / identify
# ---------------------------------------------------------------------------

def backend_name(backend_id: int) -> str:
    names = {
        cv2.CAP_DSHOW: "DSHOW",
        cv2.CAP_MSMF: "MSMF",
        cv2.CAP_ANY: "ANY",
    }
    return names.get(backend_id, str(backend_id))


def open_capture(index: int):
    """Try backends in order until one actually delivers a frame.

    OpenCV on Windows silently opens a capture object even when the backend
    can't talk to the device; isOpened() lies, so we require a real frame.
    """
    if platform.system() == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, backend
        cap.release()
    return None, None


# ---------------------------------------------------------------------------
# Resolution / FPS / jitter
# ---------------------------------------------------------------------------

def probe_resolution_and_fps(cap, n_frames: int) -> dict:
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    for _ in range(10):  # warm up, first frames are often stale/slow
        cap.read()

    timestamps = []
    for _ in range(n_frames):
        ok, _ = cap.read()
        t = time.perf_counter()
        if ok:
            timestamps.append(t)

    ts = np.array(timestamps)
    intervals = np.diff(ts)
    measured_fps = float(1.0 / np.mean(intervals)) if len(intervals) else float("nan")
    jitter_ms = float(np.std(intervals) * 1000) if len(intervals) else float("nan")
    max_gap_ms = float(np.max(intervals) * 1000) if len(intervals) else float("nan")

    return {
        "width": w,
        "height": h,
        "reported_fps": reported_fps,
        "measured_fps": measured_fps,
        "jitter_std_ms": jitter_ms,
        "max_frame_gap_ms": max_gap_ms,
        "n_frames_captured": len(timestamps),
        "n_frames_requested": n_frames,
    }


# ---------------------------------------------------------------------------
# Exposure / white-balance lock probe
# ---------------------------------------------------------------------------

def mean_luminance(frame) -> float:
    if frame is None:
        return float("nan")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def color_ratios(frame) -> tuple:
    """Blue/Green and Red/Green channel ratios -- these move with white
    balance / color temperature even when overall luminance doesn't, so
    they're the right metric for AUTO_WB (luminance is the wrong metric,
    it's what CAP_PROP_EXPOSURE affects, not what CAP_PROP_AUTO_WB affects)."""
    if frame is None:
        return float("nan"), float("nan")
    b, g, r = (frame[:, :, i].astype(np.float64) for i in range(3))
    mean_g = float(np.mean(g))
    if mean_g < 1e-6:
        return float("nan"), float("nan")
    return float(np.mean(b) / mean_g), float(np.mean(r) / mean_g)


def settle_and_sample(cap, n_discard: int = 5):
    for _ in range(n_discard):
        cap.read()
    ok, frame = cap.read()
    if not ok:
        return float("nan"), (float("nan"), float("nan"))
    return mean_luminance(frame), color_ratios(frame)


def probe_control(cap, prop_id: int, prop_name: str, candidate_values: list,
                   restore_value, luminance_threshold: float,
                   metric: str = "luminance", color_ratio_threshold: float = 0.02) -> dict:
    """Try each candidate value for a property and check for a real effect.

    Readback on some DSHOW drivers is broken independently of whether the
    underlying set() call worked -- CAP_PROP_AUTO_EXPOSURE on this class of
    driver returns -1.0 (OpenCV's "unsupported" sentinel) on every read
    regardless of what was set. So "stuck" is necessary-but-not-sufficient
    evidence; a measured physical effect is the real test, and is treated as
    authoritative when readback is the -1 sentinel.

    `metric` selects what counts as a "visible change": luminance for
    exposure-type properties, color-channel ratios (B/G, R/G) for white
    balance, which changes color temperature, not brightness.
    """
    original = cap.get(prop_id)
    baseline_lum, baseline_ratios = settle_and_sample(cap)

    attempts = []
    for value in candidate_values:
        set_ok = cap.set(prop_id, value)
        time.sleep(0.4)  # let the sensor/driver settle
        readback = cap.get(prop_id)
        lum, ratios = settle_and_sample(cap)
        lum_delta = lum - baseline_lum
        ratio_delta = max(abs(ratios[0] - baseline_ratios[0]), abs(ratios[1] - baseline_ratios[1]))

        readback_unsupported = not np.isnan(readback) and abs(readback - (-1.0)) < 1e-6
        stuck = not np.isnan(readback) and abs(readback - value) < 1e-2

        if metric == "color_ratio":
            visible = bool(not np.isnan(ratio_delta) and ratio_delta > color_ratio_threshold)
        else:
            visible = bool(abs(lum_delta) > luminance_threshold)

        attempt = {
            "tried_value": value,
            "set_call_returned_true": bool(set_ok),
            "readback_value": readback,
            "readback_unsupported_sentinel": bool(readback_unsupported),
            "value_stuck": stuck,
            "luminance_after": lum,
            "luminance_delta_from_baseline": lum_delta,
            "color_ratio_delta_from_baseline": ratio_delta,
            "visible_change": visible,
        }
        attempts.append(attempt)
        baseline_lum, baseline_ratios = lum, ratios

    cap.set(prop_id, restore_value)

    any_stuck = any(a["value_stuck"] for a in attempts)
    any_readback_unsupported = any(a["readback_unsupported_sentinel"] for a in attempts)
    any_visible = any(a["visible_change"] for a in attempts)

    # Usable if we have real evidence the property changes device behavior:
    # either a clean set+readback+effect, or (when readback itself is broken)
    # a real measured effect from set() alone.
    usable = any(a["value_stuck"] and a["visible_change"] for a in attempts) or (
        any_readback_unsupported and any_visible
    )

    return {
        "property": prop_name,
        "original_value": original,
        "metric": metric,
        "attempts": attempts,
        "any_value_stuck": any_stuck,
        "any_readback_unsupported_sentinel": any_readback_unsupported,
        "any_visible_change": any_visible,
        "usable": usable,
    }


def probe_exposure_lock(cap, cfg: dict) -> dict:
    thresh = cfg["luminance_change_threshold"]
    exposure_val = cfg["exposure_test_value"]

    # Auto-exposure mode is the least standardised property in OpenCV/UVC.
    # DSHOW/MSMF on Windows commonly use 0.25=manual / 0.75=auto, but some
    # drivers use 1=manual / 0=auto (V4L2 convention leaking through). Try both.
    auto_exposure = probe_control(
        cap, cv2.CAP_PROP_AUTO_EXPOSURE, "CAP_PROP_AUTO_EXPOSURE",
        candidate_values=[0.25, 1, 0], restore_value=0.75,
        luminance_threshold=thresh,
    )

    exposure = probe_control(
        cap, cv2.CAP_PROP_EXPOSURE, "CAP_PROP_EXPOSURE",
        candidate_values=[exposure_val, exposure_val - 3], restore_value=exposure_val,
        luminance_threshold=thresh,
    )

    auto_wb = probe_control(
        cap, cv2.CAP_PROP_AUTO_WB, "CAP_PROP_AUTO_WB",
        candidate_values=[0, 1], restore_value=1,
        luminance_threshold=thresh, metric="color_ratio",
    )

    return {
        "auto_exposure": auto_exposure,
        "exposure": exposure,
        "auto_wb": auto_wb,
    }


# ---------------------------------------------------------------------------
# Display refresh probe
# ---------------------------------------------------------------------------

def query_windows_refresh_rate() -> float:
    """Best-effort OS-reported refresh rate via EnumDisplaySettingsW. Returns
    NaN off Windows or if the call fails -- this is advisory, the redraw
    loop measurement below is what actually matters."""
    if platform.system() != "Windows":
        return float("nan")
    try:
        user32 = ctypes.windll.user32

        class DEVMODE(ctypes.Structure):
            _fields_ = [
                ("dmDeviceName", ctypes.c_wchar * 32),
                ("dmSpecVersion", ctypes.c_uint16),
                ("dmDriverVersion", ctypes.c_uint16),
                ("dmSize", ctypes.c_uint16),
                ("dmDriverExtra", ctypes.c_uint16),
                ("dmFields", ctypes.c_uint32),
                ("dmOrientation", ctypes.c_int16),
                ("dmPaperSize", ctypes.c_int16),
                ("dmPaperLength", ctypes.c_int16),
                ("dmPaperWidth", ctypes.c_int16),
                ("dmScale", ctypes.c_int16),
                ("dmCopies", ctypes.c_int16),
                ("dmDefaultSource", ctypes.c_int16),
                ("dmPrintQuality", ctypes.c_int16),
                ("dmColor", ctypes.c_int16),
                ("dmDuplex", ctypes.c_int16),
                ("dmYResolution", ctypes.c_int16),
                ("dmTTOption", ctypes.c_int16),
                ("dmCollate", ctypes.c_int16),
                ("dmFormName", ctypes.c_wchar * 32),
                ("dmLogPixels", ctypes.c_uint16),
                ("dmBitsPerPel", ctypes.c_uint32),
                ("dmPelsWidth", ctypes.c_uint32),
                ("dmPelsHeight", ctypes.c_uint32),
                ("dmDisplayFlags", ctypes.c_uint32),
                ("dmDisplayFrequency", ctypes.c_uint32),
            ]

        devmode = DEVMODE()
        devmode.dmSize = ctypes.sizeof(DEVMODE)
        ENUM_CURRENT_SETTINGS = -1
        if user32.EnumDisplaySettingsW(None, ENUM_CURRENT_SETTINGS, ctypes.byref(devmode)):
            return float(devmode.dmDisplayFrequency)
    except Exception:
        pass
    return float("nan")


def probe_display_redraw(seconds: float) -> dict:
    """Drive a fullscreen window with alternating full-white/full-black
    frames as fast as possible and measure actual redraw intervals via
    perf_counter -- this is what the emitter (Milestone 2) will depend on,
    so we test the exact mechanism, not just ask Windows what it thinks
    the monitor is capable of.
    """
    window = "praesens_probe"
    try:
        cv2.namedWindow(window, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    except Exception as e:
        return {"error": f"could not create fullscreen window: {e}"}

    w, h = 800, 600
    white = np.full((h, w, 3), 255, dtype=np.uint8)
    black = np.zeros((h, w, 3), dtype=np.uint8)

    timestamps = []
    start = time.perf_counter()
    frame_toggle = True
    try:
        while time.perf_counter() - start < seconds:
            cv2.imshow(window, white if frame_toggle else black)
            cv2.waitKey(1)
            timestamps.append(time.perf_counter())
            frame_toggle = not frame_toggle
    finally:
        cv2.destroyWindow(window)
        cv2.waitKey(1)

    ts = np.array(timestamps)
    intervals = np.diff(ts)
    measured_fps = float(1.0 / np.mean(intervals)) if len(intervals) else float("nan")
    jitter_ms = float(np.std(intervals) * 1000) if len(intervals) else float("nan")

    return {
        "os_reported_refresh_hz": query_windows_refresh_rate(),
        "measured_redraw_fps": measured_fps,
        "redraw_jitter_std_ms": jitter_ms,
        "n_redraws": len(timestamps),
        "seconds_run": seconds,
    }


# ---------------------------------------------------------------------------
# Multi-camera enumeration -- lets you positively identify which index is
# which physical/virtual device (e.g. OBS Virtual Camera) BEFORE relying on
# it in demo/attack.py. cv2.VideoCapture has no cross-backend way to read a
# device's friendly name back from an open capture, so on Windows this uses
# pygrabber to query DirectShow's own device list directly -- the same
# enumerator CAP_DSHOW uses, so the name order matches the index order.
# ---------------------------------------------------------------------------

def _dshow_device_names() -> list:
    """Friendly DirectShow device names in enumeration order (index i here
    corresponds to cv2.VideoCapture(i, cv2.CAP_DSHOW)), or [] if pygrabber
    isn't installed / not on Windows / the query fails for any reason --
    enumeration should degrade to index-only listing, never crash."""
    if platform.system() != "Windows":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception as e:
        print(f"NOTE: could not query DirectShow device names ({e}). "
              f"Install pygrabber for named enumeration: pip install pygrabber")
        return []


def enumerate_cameras(max_index: int) -> list:
    """Probes indices 0..max_index-1, pairing each with its DirectShow name
    (if available) and reporting resolution + which backend actually
    delivered a frame. This is the positive-identification step: run it,
    read the name column, and that's the index to hand to
    demo/attack.py / praesens/session.py --camera-index."""
    names = _dshow_device_names()
    results = []
    for idx in range(max_index):
        cap, backend = open_capture(idx)
        if cap is None:
            results.append({"index": idx, "opened": False, "name": None,
                             "backend": None, "width": None, "height": None})
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        name = names[idx] if idx < len(names) else None
        results.append({"index": idx, "opened": True, "name": name,
                         "backend": backend_name(backend), "width": w, "height": h})
    return results


def print_camera_list(results: list) -> None:
    print("\n" + "=" * 70)
    print("CAMERA ENUMERATION")
    print("=" * 70)
    any_named = any(r["name"] for r in results)
    if not any_named:
        print("(no DirectShow names available -- pip install pygrabber for named "
              "enumeration on Windows; showing index-only results)")
    for r in results:
        if not r["opened"]:
            print(f"  [{r['index']}] -- (could not open / no device)")
            continue
        name = r["name"] or "(name unavailable)"
        print(f"  [{r['index']}] {name:35s} {r['width']}x{r['height']}  backend={r['backend']}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_summary(results: dict, min_fps: float) -> None:
    print("\n" + "=" * 70)
    print("PRAESENS HARDWARE PROBE SUMMARY")
    print("=" * 70)

    cam = results["camera"]
    print(f"\nCamera: index={results['camera_index']} backend={cam['backend']}")
    print(f"  Resolution: {cam['fps']['width']}x{cam['fps']['height']}")
    print(f"  Reported FPS: {cam['fps']['reported_fps']:.1f}")
    print(f"  Measured FPS: {cam['fps']['measured_fps']:.1f} "
          f"(over {cam['fps']['n_frames_captured']} frames)")
    print(f"  Frame-interval jitter (std): {cam['fps']['jitter_std_ms']:.2f} ms")
    print(f"  Max single frame gap: {cam['fps']['max_frame_gap_ms']:.2f} ms")

    fps_ok = cam["fps"]["measured_fps"] >= min_fps
    print(f"  -> {'OK' if fps_ok else 'PROBLEM'}: measured FPS "
          f"{'meets' if fps_ok else 'is below'} {min_fps} Hz floor")

    print("\nExposure / white-balance lock:")
    for key, label in [("auto_exposure", "AUTO_EXPOSURE"),
                        ("exposure", "EXPOSURE"),
                        ("auto_wb", "AUTO_WB")]:
        c = cam["controls"][key]
        status = "USABLE" if c["usable"] else "NOT USABLE"
        note = " (readback unsupported by driver, trusting measured effect)" \
            if c["any_readback_unsupported_sentinel"] else ""
        print(f"  {label}: sticks={c['any_value_stuck']} "
              f"visible_change={c['any_visible_change']} -> {status}{note}")

    all_usable = all(cam["controls"][k]["usable"] for k in ("auto_exposure", "exposure"))
    print(f"\n  -> {'OK' if all_usable else 'PROBLEM'}: exposure lock "
          f"{'is' if all_usable else 'is NOT'} controllable on this camera.")
    if not all_usable:
        print("     Auto-exposure will fight the light pattern. The optical lane")
        print("     will need to either compensate (e.g. differential/high-freq")
        print("     signal design) or this hardware is not viable as-is.")

    disp = results["display"]
    if "error" in disp:
        print(f"\nDisplay: ERROR - {disp['error']}")
    else:
        print(f"\nDisplay:")
        print(f"  OS-reported refresh rate: {disp['os_reported_refresh_hz']:.1f} Hz")
        print(f"  Measured fullscreen redraw rate: {disp['measured_redraw_fps']:.1f} Hz "
              f"(jitter std {disp['redraw_jitter_std_ms']:.2f} ms)")
        disp_ok = disp["measured_redraw_fps"] >= min_fps
        print(f"  -> {'OK' if disp_ok else 'PROBLEM'}: redraw rate "
              f"{'meets' if disp_ok else 'is below'} {min_fps} Hz floor")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="PRAESENS Milestone 0 hardware probe")
    parser.add_argument("--camera-index", type=int, default=None,
                         help="override config.yaml probe.camera_index")
    parser.add_argument("--skip-display", action="store_true",
                         help="skip the fullscreen redraw test (e.g. headless run)")
    parser.add_argument("--output", type=str, default=None,
                         help="output JSON path (default: logs/probe_<timestamp>.json)")
    parser.add_argument("--list-cameras", action="store_true",
                         help="enumerate all camera indices with DirectShow names (if available) and exit "
                              "-- use this to positively identify which index is OBS Virtual Camera")
    parser.add_argument("--max-camera-index", type=int, default=None,
                         help="override config.yaml probe.max_camera_index for --list-cameras")
    args = parser.parse_args()

    cfg = load_config()

    if args.list_cameras:
        max_index = args.max_camera_index if args.max_camera_index is not None else cfg["max_camera_index"]
        print(f"Probing camera indices 0-{max_index - 1} ...")
        results = enumerate_cameras(max_index)
        print_camera_list(results)
        return

    camera_index = args.camera_index if args.camera_index is not None else cfg["camera_index"]

    print(f"Opening camera index {camera_index} ...")
    cap, backend = open_capture(camera_index)
    if cap is None:
        print(f"ERROR: could not open any camera at index {camera_index} with any backend.")
        sys.exit(1)

    print("Measuring resolution / FPS / jitter ...")
    fps_info = probe_resolution_and_fps(cap, cfg["n_frames"])

    print("Probing exposure / white-balance lock (camera will visibly change "
          "brightness/color during this step, that's expected) ...")
    controls = probe_exposure_lock(cap, cfg)

    cap.release()

    display_info = {"error": "skipped via --skip-display"}
    if not args.skip_display:
        print(f"Probing display redraw rate for {cfg['display_test_seconds']}s "
              f"(a fullscreen flashing window will appear) ...")
        display_info = probe_display_redraw(cfg["display_test_seconds"])

    results = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "camera_index": camera_index,
        "camera": {
            "backend": backend_name(backend),
            "fps": fps_info,
            "controls": controls,
        },
        "display": display_info,
    }

    out_path = Path(args.output) if args.output else REPO_ROOT / "logs" / f"probe_{int(time.time())}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")

    print_summary(results, cfg["min_acceptable_fps"])


if __name__ == "__main__":
    main()
