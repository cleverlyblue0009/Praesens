"""Milestone 9 acceptance tests: the spatial-assignment score must actually
distinguish "genuine per-region reflection" from "one estimated global
waveform applied everywhere" (the attack this milestone exists to close),
and both signal-processing scores must collapse to near-zero for a video
with no relationship to the challenge at all. All three run on synthetic
signals -- deterministic, no camera or display needed.
"""
import numpy as np
import pytest

from praesens.challenge import Challenge, derive_zone_challenges
from praesens.emit import EmitterConfig
from praesens.spatial import (
    SpatialEmitter, compute_correlation_matrix, spatial_assignment_score, global_score_from_logs,
)

ZONE_NAMES = ("left", "right", "top")
ROI_TO_ZONE = {"forehead": "top", "left_cheek": "left", "right_cheek": "right"}
DETREND_WINDOW_S = 1.5
LAG_MAX_MS = 300
LAG_STEP_MS = 5


def _make_zone_log(challenge: Challenge, sample_rate_hz: float = 100.0) -> list:
    t = np.arange(0, challenge.duration_s, 1.0 / sample_rate_hz)
    vals = challenge.value_at(t)
    return [{"t": float(tt), "chip_value": int(v)} for tt, v in zip(t, vals)]


def _make_global_log(zone_logs: dict) -> list:
    """Mirrors SpatialEmitter.get_global_log(): mean chip_value across
    zones at each shared timestamp."""
    names = list(zone_logs.keys())
    n = len(zone_logs[names[0]])
    out = []
    for i in range(n):
        t = zone_logs[names[0]][i]["t"]
        mean_chip = float(np.mean([zone_logs[name][i]["chip_value"] for name in names]))
        out.append({"t": t, "chip_value": mean_chip})
    return out


@pytest.fixture
def zones_and_logs():
    master = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=17)
    zones = derive_zone_challenges(master, ZONE_NAMES)
    zone_logs = {z.zone_name: _make_zone_log(z.challenge) for z in zones}
    zone_by_name = {z.zone_name: z.challenge for z in zones}
    frame_times = np.arange(0.2, 19.5, 1 / 29.97)
    return zone_by_name, zone_logs, frame_times


def test_zone_challenges_have_low_mutual_cross_correlation():
    """Regression test for derive_zone_challenges' core claim: shifted
    copies of the same m-sequence are near-orthogonal, by the same
    autocorrelation property Milestone 1 already verified."""
    master = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=17)
    zones = derive_zone_challenges(master, ZONE_NAMES)
    chips = np.array([z.challenge.chips for z in zones], dtype=np.float64)
    n = chips.shape[1]
    for i in range(3):
        for j in range(3):
            c = float(np.dot(chips[i], chips[j])) / n
            if i == j:
                assert c > 0.99, f"zone {i} should perfectly self-correlate, got {c}"
            else:
                assert abs(c) < 0.1, f"zones {i},{j} should be near-orthogonal, got {c}"


def test_genuine_per_zone_reflectance_gives_diagonal_dominant_matrix(zones_and_logs):
    """(a) Each ROI's measured luminance tracks ONLY its own geometrically
    correct zone -- the diagonal should clearly dominate every row."""
    zone_by_name, zone_logs, frame_times = zones_and_logs
    rng = np.random.default_rng(0)
    TRUE_LAG_S = 0.04

    per_roi_traces = {}
    for roi, zone_name in ROI_TO_ZONE.items():
        true_vals = zone_by_name[zone_name].value_at(frame_times - TRUE_LAG_S)
        measured = 120.0 + 8.0 * true_vals + rng.normal(0, 0.5, len(frame_times))
        per_roi_traces[roi] = (frame_times, measured)

    R, lag_matrix, roi_names, zone_names = compute_correlation_matrix(
        per_roi_traces, zone_logs, DETREND_WINDOW_S, LAG_MAX_MS, LAG_STEP_MS
    )
    spatial_score = spatial_assignment_score(R, roi_names, zone_names, ROI_TO_ZONE)

    for i, roi in enumerate(roi_names):
        j_expected = zone_names.index(ROI_TO_ZONE[roi])
        diag = R[i, j_expected]
        offdiag_max = max(R[i, j] for j in range(len(zone_names)) if j != j_expected)
        assert diag > 0.6, f"{roi}: diagonal score too low: {diag:.3f}"
        assert diag > offdiag_max + 0.3, (
            f"{roi}: diagonal ({diag:.3f}) not clearly dominant over best "
            f"off-diagonal ({offdiag_max:.3f})"
        )

    assert spatial_score > 0.4, f"spatial_assignment_score too low: {spatial_score:.3f}"


def test_global_brightness_multiply_collapses_spatial_but_not_global_score(zones_and_logs):
    """(b) An attacker who can only estimate ONE global waveform (the best
    a single-brightness-observer adversary can do) and applies it
    UNIFORMLY to every region: the old global-only score stays high (this
    attack used to pass), but the spatial assignment score -- which needs
    each ROI to prefer ITS OWN zone specifically -- collapses toward zero,
    because a uniform matrix has no diagonal dominance to find."""
    zone_by_name, zone_logs, frame_times = zones_and_logs
    rng = np.random.default_rng(1)

    global_vals = np.mean([zone_by_name[n].value_at(frame_times) for n in zone_by_name], axis=0)
    per_roi_traces = {}
    for roi in ROI_TO_ZONE:
        measured = 120.0 + 8.0 * global_vals + rng.normal(0, 0.5, len(frame_times))
        per_roi_traces[roi] = (frame_times, measured)

    R, lag_matrix, roi_names, zone_names = compute_correlation_matrix(
        per_roi_traces, zone_logs, DETREND_WINDOW_S, LAG_MAX_MS, LAG_STEP_MS
    )
    spatial_score = spatial_assignment_score(R, roi_names, zone_names, ROI_TO_ZONE)
    global_log = _make_global_log(zone_logs)
    global_score = global_score_from_logs(per_roi_traces, global_log, DETREND_WINDOW_S, LAG_MAX_MS, LAG_STEP_MS)

    assert global_score > 0.5, f"expected the OLD global score to stay high for this attack: {global_score:.3f}"
    assert abs(spatial_score) < 0.25, f"expected the NEW spatial score to collapse: {spatial_score:.3f}"


def test_unrelated_video_gives_near_zero_global_and_spatial_scores(zones_and_logs):
    """(c) A video with no relationship to the challenge at all (the
    baseline injection-attack case, no adaptation attempted) should
    collapse both scores, not just one."""
    zone_by_name, zone_logs, frame_times = zones_and_logs
    rng = np.random.default_rng(2)

    per_roi_traces = {}
    for roi in ROI_TO_ZONE:
        unrelated = 5.0 * np.sin(2 * np.pi * 0.3 * frame_times)
        measured = 120.0 + unrelated + rng.normal(0, 1.0, len(frame_times))
        per_roi_traces[roi] = (frame_times, measured)

    R, lag_matrix, roi_names, zone_names = compute_correlation_matrix(
        per_roi_traces, zone_logs, DETREND_WINDOW_S, LAG_MAX_MS, LAG_STEP_MS
    )
    spatial_score = spatial_assignment_score(R, roi_names, zone_names, ROI_TO_ZONE)
    global_log = _make_global_log(zone_logs)
    global_score = global_score_from_logs(per_roi_traces, global_log, DETREND_WINDOW_S, LAG_MAX_MS, LAG_STEP_MS)

    assert abs(global_score) < 0.3, f"expected near-zero global score for unrelated video: {global_score:.3f}"
    assert abs(spatial_score) < 0.3, f"expected near-zero spatial score for unrelated video: {spatial_score:.3f}"


def test_spatial_emitter_drive_and_log_matches_render_frame_and_logs_every_zone():
    """Milestone 14 regression test: SpatialEmitter.drive_and_log() must be
    exactly equivalent to render_frame()+log_redraw() (what
    praesens.spatial's own __main__ session loop still calls separately),
    and demo/panel.py's live-scoring loop depends on EVERY zone getting a
    log entry from a single drive_and_log() call, not just one merged
    entry -- that per-zone detail is what compute_correlation_matrix needs."""
    master = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=7, loop=True)
    zones = derive_zone_challenges(master, ZONE_NAMES)
    emitter = SpatialEmitter(zones, EmitterConfig())
    emitter.begin_manual_drive(start_time=0.0)

    expected_frame, *_ = emitter.render_frame(1.5)
    got_frame = emitter.drive_and_log(1.5)
    assert np.array_equal(expected_frame, got_frame)

    for name in ZONE_NAMES:
        log = emitter.get_zone_log(name)
        assert len(log) == 1
        assert set(log[0].keys()) == {"t", "chip_value", "enabled"}

    assert len(emitter.get_global_log()) == 1
