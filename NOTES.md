# PRAESENS — milestone notes

Records what was actually measured at each milestone, not what was intended.
Chronological, newest at the bottom.

---

## Milestone 9 — spatially-differential optical challenge

**What was built:** `praesens/challenge.py` gained `derive_zone_challenges()`,
deriving N zone sequences as cyclic shifts of the same session's full-period
m-sequence (not independently seeded -- for a fixed LFSR order, different
seeds are just different phases of the same cycle anyway). `praesens/optical.py`
gained `per_roi_luminance()` for forehead/left_cheek/right_cheek luminance
computed separately instead of merged into one mask. New `praesens/spatial.py`:
`SpatialEmitter` (mirrors `Emitter`'s API, border split into left/right/top
zones, each showing its own zone challenge), `compute_correlation_matrix()`
(ROI x zone peak correlation), `spatial_assignment_score()` (mean diagonal
minus mean best-off-diagonal), `global_score_from_logs()` (the old
Milestone-3-style whole-border score, kept as a separate sub-score, not
replaced). The existing single-zone `Emitter`/session.py/demo/eval pipeline
is completely untouched -- this is additive, not a modification of the
measured-identically pipeline those depend on.

**What was measured:**

- Zone cross-correlation (`tests/test_spatial.py::test_zone_challenges_have_low_mutual_cross_correlation`):
  diagonal >0.99, all off-diagonal <0.1 for 3 zones derived from one seed.
  Matches the same near-ideal autocorrelation property Milestone 1 verified,
  applied to shifted copies rather than the base sequence -- not a new
  unverified claim.
- Synthetic acceptance tests (`tests/test_spatial.py`, 4/4 pass):
  - genuine per-zone reflectance: diagonal >0.6 and >0.3 clear of the best
    off-diagonal on every ROI row; spatial_assignment_score=0.4+.
  - global-brightness-multiply attack (one estimated waveform applied
    uniformly to every ROI -- the attack this milestone exists to close):
    global_score stays >0.5 (this attack used to pass the old check),
    spatial_assignment_score collapses to <0.25 in magnitude.
  - unrelated video: both scores <0.3 in magnitude.
- Real session (`python -m praesens.spatial --seconds 20`, run on real
  hardware, empty room): auto_chip_rate correctly measured 7.9fps and
  picked chip_rate_hz=1.32, duration_s=45.4 (consistent with the exposure=-3
  constraint from earlier real-corpus work). Matrix computed and printed
  without error. global_score=0.005, spatial_score=0.004 -- both near zero,
  which is the CORRECT result for an empty room (matches the "unrelated
  video" synthetic case), but does NOT demonstrate a genuine diagonal-
  dominant result with a real face present. That demonstration still needs
  a real subject in frame; the synthetic tests are the rigorous evidence
  for the milestone's actual claim (the acceptance criterion's unit tests),
  this real run is evidence the plumbing doesn't crash on real hardware.

**Not yet done:** wiring SpatialEmitter into session.py/demo/panel (this
milestone is a standalone, independently-testable capability per the
brief's own scope; integration into the live demo happens later per the
milestone plan, not here).

**Config added:** `spatial:` section in `config.yaml` (zone_names,
min_shift_chips, detrend_window_s, lag_search_max_ms, lag_step_ms,
expected_assignment).
