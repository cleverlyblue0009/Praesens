"""Camera-capture format fix (2026-09-01): forcing MJPG on this camera was
found to unlock much higher throughput than the driver's default
uncompressed format -- see scripts/probe.py's before/after and NOTES.md.
configure_capture_format() is the one place FOURCC/resolution/fps get
negotiated for every capture site in the codebase; these tests exercise
its property-setting SEQUENCE against a mocked cv2.VideoCapture (no real
camera needed/available in CI) rather than trusting the DSHOW-safe
ordering claim unverified, matching this repo's "verify empirically,
don't assume" discipline applied elsewhere (e.g. the LFSR tap table,
the SNR passband).
"""
from unittest.mock import MagicMock, call, patch

import cv2
import pytest

import praesens.capture as capture_mod
from praesens.capture import CaptureConfig, configure_capture_format, measure_steady_state_fps


def _make_mock_cap(fourcc_readback: str = "MJPG", width: int = 640, height: int = 480,
                    reported_fps: float = 30.0):
    """A MagicMock standing in for cv2.VideoCapture: .set() calls are
    recorded (that's what the ordering assertions check), .get() returns
    fixed values keyed by property id so the readback/logging code in
    configure_capture_format doesn't choke on a bare MagicMock, and
    .grab()/.retrieve() succeed every call so measure_capture_fps (called
    internally) returns a real number instead of NaN."""
    cap = MagicMock()
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc_readback)
    get_values = {
        cv2.CAP_PROP_FOURCC: fourcc_code,
        cv2.CAP_PROP_FRAME_WIDTH: width,
        cv2.CAP_PROP_FRAME_HEIGHT: height,
        cv2.CAP_PROP_FPS: reported_fps,
    }
    cap.get.side_effect = lambda prop_id: get_values.get(prop_id, 0)
    cap.grab.return_value = True
    cap.retrieve.return_value = (True, None)
    return cap


def test_fourcc_is_set_before_width_and_height():
    cap = _make_mock_cap()
    config = CaptureConfig(fourcc="MJPG", width=640, height=480, requested_fps=30.0)

    configure_capture_format(cap, config)

    set_calls = [c for c in cap.set.mock_calls]
    prop_order = [c.args[0] for c in set_calls]
    assert prop_order.index(cv2.CAP_PROP_FOURCC) < prop_order.index(cv2.CAP_PROP_FRAME_WIDTH)
    assert prop_order.index(cv2.CAP_PROP_FOURCC) < prop_order.index(cv2.CAP_PROP_FRAME_HEIGHT)
    # full DSHOW-safe order: fourcc -> width -> height -> fps
    assert prop_order == [
        cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS,
    ]


def test_fourcc_set_with_the_right_encoded_value():
    cap = _make_mock_cap()
    config = CaptureConfig(fourcc="MJPG", width=640, height=480, requested_fps=30.0)

    configure_capture_format(cap, config)

    expected_code = cv2.VideoWriter_fourcc(*"MJPG")
    assert call(cv2.CAP_PROP_FOURCC, expected_code) in cap.set.mock_calls


def test_fourcc_null_skips_the_fourcc_call_entirely():
    cap = _make_mock_cap()
    config = CaptureConfig(fourcc=None, width=640, height=480, requested_fps=30.0)

    configure_capture_format(cap, config)

    called_props = [c.args[0] for c in cap.set.mock_calls]
    assert cv2.CAP_PROP_FOURCC not in called_props
    # width/height/fps still get set -- only the fourcc call is skipped
    assert cv2.CAP_PROP_FRAME_WIDTH in called_props
    assert cv2.CAP_PROP_FRAME_HEIGHT in called_props
    assert cv2.CAP_PROP_FPS in called_props


def test_width_height_fps_zero_or_falsy_are_skipped():
    """A CaptureConfig with width/height/requested_fps left as 0 (falsy)
    must not issue a meaningless cap.set(..., 0) call -- mirrors the
    fourcc=None skip, generalised to every field in the sequence."""
    cap = _make_mock_cap()
    config = CaptureConfig(fourcc="MJPG", width=0, height=0, requested_fps=0)

    configure_capture_format(cap, config)

    called_props = [c.args[0] for c in cap.set.mock_calls]
    assert called_props == [cv2.CAP_PROP_FOURCC]


def test_returns_requested_and_actual_values():
    cap = _make_mock_cap(fourcc_readback="MJPG", width=640, height=480, reported_fps=30.0)
    config = CaptureConfig(fourcc="MJPG", width=640, height=480, requested_fps=30.0)

    result = configure_capture_format(cap, config)

    assert result["fourcc_requested"] == "MJPG"
    assert result["fourcc_actual"] == "MJPG"
    assert result["width_actual"] == 640
    assert result["height_actual"] == 480
    assert result["fps_reported"] == pytest.approx(30.0)
    assert result["fps_measured"] > 0  # measure_capture_fps ran against the mocked grab/retrieve


def test_warns_when_measured_fps_is_below_threshold(monkeypatch):
    """The driver-negotiated format can still under-deliver in practice --
    warn_below_fps exists to catch that from the ACTUAL measured rate,
    not the (often unreliable) cap.get(CAP_PROP_FPS) readback."""
    import praesens.capture as capture_mod

    monkeypatch.setattr(capture_mod, "measure_capture_fps", lambda cap, n_frames=30: 8.0)
    cap = _make_mock_cap()
    config = CaptureConfig(fourcc="MJPG", width=640, height=480, requested_fps=30.0, warn_below_fps=20.0)
    warn_list: list = []

    configure_capture_format(cap, config, warn_list)

    assert len(warn_list) == 1
    assert "8.0" in warn_list[0]
    assert "20.0" in warn_list[0]


def test_no_warning_when_measured_fps_clears_the_threshold(monkeypatch):
    import praesens.capture as capture_mod

    monkeypatch.setattr(capture_mod, "measure_capture_fps", lambda cap, n_frames=30: 28.0)
    cap = _make_mock_cap()
    config = CaptureConfig(fourcc="MJPG", requested_fps=30.0, warn_below_fps=20.0)
    warn_list: list = []

    configure_capture_format(cap, config, warn_list)

    assert warn_list == []


def test_no_warn_list_given_does_not_raise(monkeypatch):
    import praesens.capture as capture_mod

    monkeypatch.setattr(capture_mod, "measure_capture_fps", lambda cap, n_frames=30: 8.0)
    cap = _make_mock_cap()
    config = CaptureConfig(fourcc="MJPG", warn_below_fps=20.0)

    configure_capture_format(cap, config, warn_list=None)  # must not raise despite the low measured fps


def test_capture_config_from_dict_ignores_unknown_keys():
    config = CaptureConfig.from_dict({"fourcc": "MJPG", "width": 800, "unrelated_key": "ignored"})
    assert config.fourcc == "MJPG"
    assert config.width == 800
    assert config.height == 480  # default, untouched


# ---------------------------------------------------------------------------
# measure_steady_state_fps -- 2026-09-01 preflight-staleness fix. Bug: the
# auto_chip_rate preflight measurement in praesens/session.py used to run
# BEFORE exposure was ever locked, measuring whatever fps the driver
# defaulted to on open rather than the fps achievable at the CONFIGURED
# exposure_value. These tests verify the ACTUAL ordering (lock, then
# warm-up frames, then measurement) against a mocked capture -- not just
# that the right functions get called, but that the warm-up frames are
# genuinely consumed BEFORE the measurement starts.
# ---------------------------------------------------------------------------

def test_measure_steady_state_fps_locks_exposure_before_any_grab():
    """lock_camera must run before even the warm-up loop starts -- a lock
    that happens after frames have already been grabbed doesn't fix
    anything, since those frames were captured under the OLD exposure."""
    cap = _make_mock_cap()
    optical_config = MagicMock()
    grab_count_at_lock_call = []

    def fake_lock_camera(cap_arg, config_arg, warn_list_arg):
        grab_count_at_lock_call.append(cap_arg.grab.call_count)
        return True

    with patch.object(capture_mod, "lock_camera", side_effect=fake_lock_camera) as mock_lock, \
         patch.object(capture_mod, "measure_capture_fps", return_value=16.0):
        measure_steady_state_fps(cap, optical_config, warmup_frames=30)

    mock_lock.assert_called_once()
    assert grab_count_at_lock_call == [0]  # no frames grabbed yet when lock_camera ran


def test_measure_steady_state_fps_measures_only_after_warmup_frames_consumed():
    """The core acceptance test: measure_capture_fps must not be called
    until exactly `warmup_frames` grab/retrieve pairs have already
    happened."""
    cap = _make_mock_cap()
    optical_config = MagicMock()
    grab_count_at_measure_call = []

    def fake_measure_capture_fps(cap_arg, n_frames=10):
        grab_count_at_measure_call.append(cap_arg.grab.call_count)
        return 16.0

    with patch.object(capture_mod, "lock_camera", return_value=True), \
         patch.object(capture_mod, "measure_capture_fps", side_effect=fake_measure_capture_fps):
        result = measure_steady_state_fps(cap, optical_config, warmup_frames=30)

    assert grab_count_at_measure_call == [30]
    assert cap.retrieve.call_count == 30  # every warm-up grab is paired with a retrieve (DSHOW-safe)
    assert result == 16.0


def test_measure_steady_state_fps_respects_a_different_warmup_frame_count():
    cap = _make_mock_cap()
    optical_config = MagicMock()
    grab_count_at_measure_call = []

    def fake_measure_capture_fps(cap_arg, n_frames=10):
        grab_count_at_measure_call.append(cap_arg.grab.call_count)
        return 16.0

    with patch.object(capture_mod, "lock_camera", return_value=True), \
         patch.object(capture_mod, "measure_capture_fps", side_effect=fake_measure_capture_fps):
        measure_steady_state_fps(cap, optical_config, warmup_frames=5)

    assert grab_count_at_measure_call == [5]


def test_measure_steady_state_fps_zero_warmup_frames_measures_immediately():
    """warmup_frames=0 is a valid (if inadvisable) configuration -- must
    not crash, and measurement starts with zero frames consumed."""
    cap = _make_mock_cap()
    optical_config = MagicMock()
    grab_count_at_measure_call = []

    def fake_measure_capture_fps(cap_arg, n_frames=10):
        grab_count_at_measure_call.append(cap_arg.grab.call_count)
        return 16.0

    with patch.object(capture_mod, "lock_camera", return_value=True), \
         patch.object(capture_mod, "measure_capture_fps", side_effect=fake_measure_capture_fps):
        measure_steady_state_fps(cap, optical_config, warmup_frames=0)

    assert grab_count_at_measure_call == [0]


def test_measure_steady_state_fps_forwards_warn_list_to_lock_camera():
    cap = _make_mock_cap()
    optical_config = MagicMock()
    warn_list: list = []

    with patch.object(capture_mod, "lock_camera") as mock_lock, \
         patch.object(capture_mod, "measure_capture_fps", return_value=16.0):
        measure_steady_state_fps(cap, optical_config, warmup_frames=1, warn_list=warn_list)

    mock_lock.assert_called_once_with(cap, optical_config, warn_list)


def test_measure_steady_state_fps_no_warn_list_does_not_raise():
    cap = _make_mock_cap()
    optical_config = MagicMock()

    with patch.object(capture_mod, "lock_camera", return_value=True), \
         patch.object(capture_mod, "measure_capture_fps", return_value=16.0):
        result = measure_steady_state_fps(cap, optical_config, warmup_frames=1, warn_list=None)

    assert result == 16.0
