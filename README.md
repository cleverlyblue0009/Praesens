# Praesens

Multimodal proof of human presence on a stock laptop. A fresh per-session challenge is pushed out through
the laptop's own hardware and the physical response is read back on independent sensors; a presence claim
is accepted only when the lanes agree in time.

| Lane | Emits | Senses | Module |
|---|---|---|---|
| Optical | m-sequence on the screen edges | reflected light on forehead/cheeks (webcam) | `praesens/optical.py`, `spatial.py` |
| Typing | seed-derived challenge phrase | keystroke timing only, never characters | `praesens/typing.py` |
| Acoustic | BPSK probe on a 4 kHz carrier | the probe at the microphone | `praesens/acoustic.py` |
| Key impact ("pressure") | — | the physical strike of every key, heard by the mic, at a fixed lag behind each OS key-down | `praesens/pressure.py` |
| Pulse (advisory) | — | rPPG heart rate from skin colour, POS method | `praesens/pulse.py` |
| Inertial | — | no IMU on the target laptop; honest `no_evidence` stub | `praesens/inertial.py` |

Fusion (`praesens/fusion.py`, `adjudicate_two_lane`): optical must pass (score and plausible lag) or the
session is REJECTED; any other gated lane failing gives RE-CHALLENGE; everything passing gives ACCEPT.
Pulse is logged and shown but not gated until a threshold is measured (`pulse.gate` in `config.yaml`).

## Setup

```
pip install -r requirements.txt
python scripts/fetch_model.py          # MediaPipe face/hand models into models/
```

## Run

Browser HUD (holographic face mesh, live lanes, one-button scan):

```
python -m demo.hud                                   # real camera, mic, keyboard
python -m demo.hud --lanes optical+typing+acoustic+pressure
python -m demo.hud --synthetic                       # no hardware; every screen says SYNTHETIC
```

It opens `http://127.0.0.1:8765`. Press **F11**: the page's grey edge strips are the optical emitter, so the
browser has to be the thing lighting your face. Click **Start scan**, keep your face in frame and type the
phrase. Each scan writes the same JSON log as the CLI (`logs/<session>.json`, `"source": "hud"`).

CLI session (OpenCV emitter, the configuration the paper numbers were measured with):

```
python -m praesens.session --condition bonafide --lanes optical+typing+pressure
python -m praesens.session --condition bonafide --lanes optical+typing+acoustic+pressure
```

Standalone checks: `python -m praesens.pressure --seconds 12` (type anything), `python -m praesens.acoustic`.

Tests: `python -m pytest -q` (no camera or microphone needed; on Linux run under `xvfb-run` because pynput
needs a display).
