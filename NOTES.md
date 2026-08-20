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

---

## Milestone 10 — typing lane (ACTIVE + PASSIVE)

**What was built:** `praesens/challenge.py` gained `generate_challenge_phrase()`
(offline word-list recombination from a hand-authored ~150-word list, seeded
deterministically from the session seed; `generative=True` raises
NotImplementedError on purpose so a stray config flag can't silently start
depending on an LLM/network call). New `praesens/typing.py`:
`KeystrokeCapture` (pynput-backed, ACTIVE mode compares each keypress
against the next expected character and stores ONLY the boolean result;
PASSIVE mode has no expected sequence at all), `HandActivityTracker`
(MediaPipe HandLandmarker, fingertip index derived from the library's own
connection sets the same way Milestone 9 derives ROI indices, not a
memorised landmark number), `keystroke_hand_coherence()` (Gaussian-
smoothed keystroke density cross-correlated against hand-activity over a
lag window bounded on BOTH sides of zero, since finger motion and key-down
are both downstream of one physical press rather than a one-directional
stimulus-response pair like light reflection), `compute_typing_status()`
(isolates the no_evidence decision from camera/keyboard I/O so it's
testable without hardware). New `praesens/inertial.py`: honest
`no_evidence` stub, documented interface, no faked signal.

**Privacy contract (rule 3), verified structurally, not just asserted:**
`tests/test_typing.py::test_keystroke_capture_never_stores_a_character`
presses a literal phrase ("secret") and asserts every stored event dict
has exactly the three keys {t_down, t_up, matched_expected} and no string
value anywhere in any event -- not a docstring claim, a check that fails
loudly if a future edit ever leaks a character into storage.

**What was measured:**

- `tests/test_typing.py` (16/16 pass, ~3.8s real compute time -- one CI run
  reported 659s total for the combined suite, traced to background-task
  scheduling overhead in the dev sandbox, not test logic: rerun with
  `--durations` showed the slowest individual test at 0.09s):
  - privacy: no character ever stored, PASSIVE mode never sets
    matched_expected, mismatches are flagged without recording what was
    actually typed.
  - coherence: genuine synchronised typing+hand-motion recovers a
    score >0.7 and the correct lag (within one grid step of a -50ms
    injected true lag); unrelated hand motion collapses the score to
    <0.4; insufficient hand data returns (nan, nan), which callers map to
    no_evidence rather than a zero score (rule 7).
  - status logic: typing that stops >=10s before session end reports
    no_evidence by the end even though earlier typing occurred in the
    same window (not "ok" just because something happened at some point);
    zero keystrokes in ACTIVE mode and insufficient hand data both
    correctly force no_evidence too.
- Real hardware (`python -m praesens.typing --mode passive --seconds 15`):
  camera + HandLandmarker pipeline ran clean, no crash. Did NOT use
  `pynput.Controller` to simulate real OS-level keystrokes for this run --
  synthetic keyboard injection would be delivered to whatever window
  actually has OS focus on the machine running it, which is a real,
  un-consented side-effect risk, not just a test-hygiene nicety. Correctly
  reported status=no_evidence, n_keystrokes=0,
  seconds_since_last_keystroke=inf, since nobody actually typed anything.
  The JSON log's `events` array is empty and the schema has no character
  field anywhere, consistent with the privacy contract. A real end-to-end
  run with an actual human typing the displayed phrase still needs a
  supervised session with someone at the keyboard -- the synthetic tests
  are the rigorous evidence for the coherence-detection claim itself.

**Not yet done:** wiring the typing lane into the live dashboard/panel
mode (Milestone 14's job) and into fusion (Milestone 11, next).

**Models added:** `models/hand_landmarker.task`, fetched via
`scripts/fetch_model.py --hand` (extended to support both models; official
Google-hosted asset, same source pattern as the face model, URL verified
by an actual successful download+load before building on it).

**Config added:** `typing:` section in `config.yaml` (n_words, generative
[hard-pinned false], hand_model_path, coherence window/lag/sigma,
min_valid_hand_samples, passive_decay_half_life_s,
passive_no_evidence_after_s, camera_index).
