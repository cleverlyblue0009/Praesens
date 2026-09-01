"""Camera-capture format setup, shared by every capture site.

scripts/probe.py found the capture camera pinned at 8fps @ 640x480 over
DSHOW, with `Reported FPS: -1.0` and a 137ms max frame gap -- the
signature of an uncompressed pixel format (YUY2, the DSHOW default on
most UVC webcams) saturating the USB link, not a genuine sensor ceiling.
Forcing MJPG (the camera does the JPEG compression on-device, so far
less raw bandwidth crosses the bus) commonly unlocks much higher
throughput on the same hardware. That 8fps ceiling forced auto_chip_rate
down to ~0.8Hz, which is the direct cause of the project's chronic
insufficient_signal / negative-SNR readings -- this module exists to fix
the capture layer, nothing downstream of it (no scoring/SNR/chip-rate/
lag logic here or touched by it).

configure_capture_format() is the ONE place FOURCC/resolution/fps get
negotiated, in DSHOW-safe order: FOURCC and resolution are set (and, on
this backend, effectively negotiated together) BEFORE fps, and all of it
happens BEFORE any exposure/white-balance work -- that stays entirely in
praesens.optical.lock_camera, called separately by each site exactly as
it already was; configure_capture_format() itself never touches
exposure/WB, so every existing capture site's exposure/WB behaviour is
unchanged, byte-for-byte.

measure_steady_state_fps() (added 2026-09-01) fixes a related, separate
bug: praesens/session.py's auto_chip_rate preflight FPS measurement used
to run BEFORE exposure was ever locked (lock_camera only happened later,
inside praesens.optical.run_session), so it measured whatever fps the
driver defaulted to on open -- not the fps achievable at the CONFIGURED
optical.exposure_value. At exposure_value=-4 that meant a stale ~8fps
reading got baked into chip_rate_hz for the whole session while the real,
exposure-locked session then ran at ~16fps, badly undersampled relative
to that rate. This function locks exposure (via lock_camera, unchanged)
THEN discards a configurable warm-up burst (optical.camera_warmup_frames)
THEN measures -- DSHOW cameras take several frames to settle fps and
brightness after an exposure change, so measuring immediately after the
lock() call still risks reading a transient.

Every capture-creation site in the codebase now calls
configure_capture_format() immediately after opening the device and
confirming cap.isOpened(), before anything else touches the capture --
see demo/live.py, demo/attack.py, praesens/session.py, praesens/
spatial.py, praesens/typing.py, scripts/diagnose.py, scripts/
verify_timing.py, scripts/record_source.py for the call sites. Two
deliberate exceptions, both already using a different capture pattern
for reasons unrelated to this fix:
  - scripts/probe.py is Milestone 0's pre-flight sanity check and
    deliberately imports nothing from praesens/ (it must work standalone
    before the package is trusted); it applies the SAME config.yaml
    `capture:` block itself, independently, so its own "Measured FPS"
    output reflects the fix without taking the dependency.
  - attacks/adaptive_injector.py opens either a pre-recorded video FILE
    or a secondary camera as ATTACK SOURCE material (not the optical
    lane's own capture) -- forcing a live-camera FOURCC/exposure regime
    onto a file capture would be wrong, and it isn't part of the
    liveness measurement this fix targets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2

from praesens.optical import measure_capture_fps, lock_camera


@dataclass
class CaptureConfig:
    fourcc: str | None = "MJPG"   # None/null -> leave the driver's default pixel format alone
    width: int = 640
    height: int = 480
    requested_fps: float = 30.0
    warn_below_fps: float = 20.0

    @classmethod
    def from_dict(cls, d: dict) -> "CaptureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _fourcc_int_to_str(code) -> str:
    """cv2.CAP_PROP_FOURCC readback is a packed 32-bit int, four ASCII
    bytes little-endian -- e.g. MJPG round-trips as int('MJPG' packed).
    Decoded the same way cv2.VideoWriter_fourcc encodes it."""
    code = int(code)
    chars = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")
    return chars if chars else "?"


def configure_capture_format(cap, config: CaptureConfig, warn_list: list | None = None) -> dict:
    """Sets FOURCC -> width -> height -> requested_fps, in that order, on
    an ALREADY-OPEN capture (caller has already checked cap.isOpened()).
    fourcc=None skips the FOURCC call entirely, leaving the driver
    default untouched -- for a camera/driver combination where forcing a
    format makes things worse, this is the one-line way back out.

    Reads back what the driver ACTUALLY accepted (requested and actual
    routinely differ -- that's the whole reason this function exists
    instead of trusting `cap.get(CAP_PROP_FPS)` at face value) and prints
    a summary line unconditionally. Also runs a quick real 30-frame
    grab+retrieve measurement (praesens.optical.measure_capture_fps,
    reused rather than reimplemented -- it already accounts for this
    DSHOW driver's grab()-without-retrieve() not blocking for a new
    frame) and, if warn_list is given, appends a warning when that
    measured rate falls under config.warn_below_fps -- matching
    praesens.optical.lock_camera's own convention of mutating a
    caller-owned warn_list rather than raising.
    """
    if config.fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc))
    if config.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    if config.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    if config.requested_fps:
        cap.set(cv2.CAP_PROP_FPS, config.requested_fps)

    actual_fourcc = _fourcc_int_to_str(cap.get(cv2.CAP_PROP_FOURCC))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = float(cap.get(cv2.CAP_PROP_FPS))
    measured_fps = measure_capture_fps(cap, n_frames=30)

    print(f"capture format: requested fourcc={config.fourcc!r} {config.width}x{config.height} "
          f"@{config.requested_fps}fps -> actual fourcc={actual_fourcc!r} {actual_w}x{actual_h} "
          f"reported_fps={reported_fps:.1f} measured_fps={measured_fps:.1f}")

    if warn_list is not None and not math.isnan(measured_fps) and measured_fps < config.warn_below_fps:
        warn_list.append(
            f"measured capture FPS ({measured_fps:.1f}) is below warn_below_fps "
            f"({config.warn_below_fps}) -- fourcc negotiated as {actual_fourcc!r} "
            f"(requested {config.fourcc!r}). This camera/format is under-delivering; "
            f"auto_chip_rate will pick a low chip rate, or measurement may fall to "
            f"insufficient_signal."
        )

    return {
        "fourcc_requested": config.fourcc, "fourcc_actual": actual_fourcc,
        "width_requested": config.width, "width_actual": actual_w,
        "height_requested": config.height, "height_actual": actual_h,
        "fps_requested": config.requested_fps, "fps_reported": reported_fps,
        "fps_measured": measured_fps,
    }


def measure_steady_state_fps(cap, optical_config, warmup_frames: int, warn_list: list | None = None) -> float:
    """Locks exposure (praesens.optical.lock_camera, unchanged -- same
    candidate sweep, same exposure_value, same disable_auto_wb logic),
    discards `warmup_frames` frames (grab+retrieve pairs, matching
    measure_capture_fps's own DSHOW-safe pattern -- grab() alone doesn't
    block for a new frame on this driver), THEN measures fps. Returns the
    measured rate; warn_list, if given, is mutated in place by
    lock_camera exactly as it already was at every other call site.

    Callers: praesens/session.py's and praesens/spatial.py's
    auto_chip_rate preflight -- see module docstring for the bug this
    fixes. lock_camera() is safe to call here even though it will be
    called AGAIN later inside praesens.optical.run_session() for
    session.py's callers specifically: it's idempotent (re-locking the
    same exposure_value does nothing harmful), and moving it here is
    deliberately the ONLY ordering change -- run_session() itself is
    untouched.
    """
    lock_camera(cap, optical_config, warn_list if warn_list is not None else [])
    for _ in range(warmup_frames):
        cap.grab()
        cap.retrieve()
    return measure_capture_fps(cap)
