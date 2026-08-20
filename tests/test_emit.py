"""Milestone 14 regression test: Emitter.drive_and_log() must be exactly
equivalent to calling render_frame() then log_redraw() separately (the
pattern demo/live.py's run loop used before Milestone 14, and the one
praesens.emit.Emitter._run()'s own background thread still uses) -- it's a
pure refactor into a single call so demo/live.py's run loop can drive
either this or praesens.spatial.SpatialEmitter without a type check, and
must not change what gets rendered or logged."""
import numpy as np

from praesens.challenge import Challenge
from praesens.emit import Emitter, EmitterConfig


def test_drive_and_log_returns_same_frame_as_render_frame():
    challenge = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=3, loop=True)
    emitter = Emitter(challenge, EmitterConfig())
    emitter.begin_manual_drive(start_time=0.0)

    expected_frame, *_ = emitter.render_frame(1.5)
    got_frame = emitter.drive_and_log(1.5)

    # render_frame is pure (same elapsed_s -> same frame); drive_and_log's
    # frame must match it exactly.
    assert np.array_equal(expected_frame, got_frame)


def test_drive_and_log_appends_exactly_one_log_entry_with_correct_fields():
    challenge = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=3, loop=True)
    emitter = Emitter(challenge, EmitterConfig())
    emitter.begin_manual_drive(start_time=0.0)

    assert emitter.get_log() == []
    emitter.drive_and_log(1.5)
    log = emitter.get_log()
    assert len(log) == 1
    assert set(log[0].keys()) == {"t", "elapsed_s", "chip_value", "luminance", "enabled"}

    emitter.drive_and_log(1.6)
    assert len(emitter.get_log()) == 2  # one call, one log entry, each time
