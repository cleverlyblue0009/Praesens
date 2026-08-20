"""Milestone 12 acceptance test: the adaptive injector's GLOBAL mode
attack must actually raise the (old, Milestone-3-style) global optical
score compared to an unrelated/no-attack video -- the literal acceptance
criterion -- while Milestone 9's spatial-assignment score must still
collapse, proving the spatial defence holds against a REAL implementation
of this attack (the actual luminance_multiply() image code path, not just
the simplified scalar-signal case tests/test_spatial.py already covers).
"""
import numpy as np

from praesens.challenge import Challenge, derive_zone_challenges
from praesens.optical import resample_emitted
from praesens.spatial import compute_correlation_matrix, spatial_assignment_score, global_score_from_logs
from attacks.adaptive_injector import luminance_multiply

ROI_TO_ZONE = {"forehead": "top", "left_cheek": "left", "right_cheek": "right"}


def _make_zone_log(challenge: Challenge, sample_rate_hz: float = 100.0) -> list:
    t = np.arange(0, challenge.duration_s, 1.0 / sample_rate_hz)
    vals = challenge.value_at(t)
    return [{"t": float(tt), "chip_value": int(v)} for tt, v in zip(t, vals)]


def _make_global_log(zone_logs: dict) -> list:
    """Chip-value (+/-1-ish) domain -- what praesens.optical's own
    cross_correlate_lag_search/resample_emitted expect, and what the
    VERIFIER'S scoring functions operate on."""
    names = list(zone_logs.keys())
    n = len(zone_logs[names[0]])
    out = []
    for i in range(n):
        t = zone_logs[names[0]][i]["t"]
        mean_chip = float(np.mean([zone_logs[name][i]["chip_value"] for name in names]))
        out.append({"t": t, "chip_value": mean_chip})
    return out


def _make_attacker_luminance_log(global_chip_log: list, base_luminance: float = 127.0,
                                  depth: float = 60.0) -> list:
    """LUMINANCE-value domain (0-255-ish) -- what the real
    AdaptiveInjector.record_sample() actually stores, since it comes from
    region_luminance() screen-capturing the emitter's REAL rendered
    border, not the abstract +/-1 chip signal. Feeding raw chip values
    into luminance_multiply()'s estimate parameter (which compares
    against a ~127 baseline) was the bug caught here: estimate-baseline
    was then always ~-127 regardless of the true chip value's sign,
    collapsing the multiplier to a near-constant instead of tracking the
    pattern -- this reconstructs what the attacker's tool would actually
    have measured from the screen."""
    return [{"t": e["t"], "chip_value": base_luminance + depth * e["chip_value"]} for e in global_chip_log]


def _synthetic_frame(mean_luminance: float, size=(16, 16)) -> np.ndarray:
    """A flat-colour synthetic 'face crop' at a given mean luminance --
    drives the REAL luminance_multiply() image code path with tiny arrays
    instead of a real camera frame, so this test exercises actual pixel
    processing, not a hand-derived formula standing in for it."""
    return np.full((size[0], size[1], 3), np.clip(mean_luminance, 0, 255), dtype=np.uint8)


def test_adaptive_injector_global_mode_raises_global_score_but_spatial_stays_dead():
    rng = np.random.default_rng(3)

    master = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=17)
    zones = derive_zone_challenges(master, ("left", "right", "top"))
    zone_logs = {z.zone_name: _make_zone_log(z.challenge) for z in zones}
    global_log = _make_global_log(zone_logs)

    frame_times = np.arange(0.2, 19.5, 1 / 29.97)
    baseline_source_luminance = 120.0

    # --- baseline: unrelated video, no attack at all ---
    per_roi_traces_baseline = {}
    for roi in ROI_TO_ZONE:
        unrelated = 5.0 * np.sin(2 * np.pi * 0.3 * frame_times)
        per_roi_traces_baseline[roi] = (frame_times, baseline_source_luminance + unrelated
                                         + rng.normal(0, 1.0, len(frame_times)))
    baseline_global = global_score_from_logs(per_roi_traces_baseline, global_log, 1.5, 300, 5)

    # --- attacked: the SAME unrelated source video, but run through the
    # real luminance_multiply() image code path using a PERFECT screen-
    # capture estimate (the attacker's strongest case -- no capture noise
    # on their side) of the true global waveform ---
    attacker_luminance_log = _make_attacker_luminance_log(global_log)
    per_roi_traces_attacked = {}
    for roi in ROI_TO_ZONE:
        unrelated = 5.0 * np.sin(2 * np.pi * 0.3 * frame_times) + rng.normal(0, 1.0, len(frame_times))
        attacked_lum = np.empty(len(frame_times))
        for i, t in enumerate(frame_times):
            estimate = float(resample_emitted(attacker_luminance_log, np.array([t]), lag_s=0.05)[0])
            src_frame = _synthetic_frame(baseline_source_luminance + unrelated[i])
            out_frame = luminance_multiply(src_frame, estimate, depth=1.0, baseline=127.0)
            attacked_lum[i] = float(out_frame.mean())
        per_roi_traces_attacked[roi] = (frame_times, attacked_lum)

    attacked_global = global_score_from_logs(per_roi_traces_attacked, global_log, 1.5, 300, 5)

    assert attacked_global > baseline_global + 0.3, (
        f"expected the adaptive injector to clearly raise the global score: "
        f"baseline={baseline_global:.3f} attacked={attacked_global:.3f}"
    )
    assert attacked_global > 0.5, f"expected a strongly raised global score, got {attacked_global:.3f}"

    # --- but the SPATIAL score must still collapse: uniform (global-mode)
    # modulation is applied identically to every ROI, so there's no
    # genuine per-region correspondence for the spatial score to find ---
    R, lag_matrix, roi_names, zone_names = compute_correlation_matrix(
        per_roi_traces_attacked, zone_logs, 1.5, 300, 5
    )
    spatial_score = spatial_assignment_score(R, roi_names, zone_names, ROI_TO_ZONE)
    assert abs(spatial_score) < 0.3, (
        f"expected the spatial defence to still catch this real attack implementation, "
        f"got spatial_score={spatial_score:.3f}"
    )


def test_luminance_multiply_preserves_colour_ratio():
    """The attack scales brightness, not colour -- a real attacker would
    do exactly this, and a naive all-channel scale would be a weaker,
    easier-to-catch-by-other-means attack than the one this tool models."""
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    frame[:, :, 2] = 200  # a reddish frame (BGR: R channel high)
    frame[:, :, 0] = 50
    out = luminance_multiply(frame, estimate=200.0, depth=1.0, baseline=127.0)
    # still reddish (R channel still clearly greater than B) after the attack
    assert out[:, :, 2].mean() > out[:, :, 0].mean()
