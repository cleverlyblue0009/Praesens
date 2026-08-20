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

---

## Milestone 11 — fusion and adjudication

**What was built:** New `praesens/fusion.py`: `LaneResult` (uniform shape
every lane emits: lane_name, subscore, status, lag_ms, confidence,
diagnostics), `acoustic_lane_stub()` (registers the acoustic lane's
interface, permanently no_evidence, no audio I/O anywhere in this repo),
`compute_joint_score()` + `lane_contributes()` (confidence-weighted mean
of ONLY the lanes that are status=='ok' AND whose measured lag falls
within their own physically plausible window -- optical/spatial
[0,300]ms one-directional, typing [-300,300]ms bounded both sides since
finger motion and key-down are both downstream of the same press, not a
stimulus-then-response pair), `Adjudicator` (three-way ACCEPT/RE-CHALLENGE/
REJECT verdict with hysteresis so the demo doesn't flicker at a threshold
boundary, plus a structured `reason_text`/`per_lane_reasons` explanation
on every verdict). New `eval/ablate.py`: recomputes the joint verdict with
each lane held out.

**What was measured:**

- `tests/test_fusion.py` (11/11 pass, 0.22s): a lane with a high subscore
  but an implausible lag contributes NOTHING (joint score is undefined,
  not zero, with only that lane present) -- the actual claim this
  milestone makes, verified directly rather than just asserted. Joint
  score is confirmed to be exactly the confidence-weighted mean of
  contributing lanes (checked against a hand-computed expected value).
  Ablating the only contributing lane leaves RE-CHALLENGE, never a
  manufactured ACCEPT/REJECT. Hysteresis verified to actually hold a
  verdict through a value that would fail a FRESH threshold check, then
  correctly release it on a further drop. min_contributing_lanes=2
  correctly forces RE-CHALLENGE even with one lane scoring 0.95.
- Bug caught while building this: `_summarize()`'s reason-priority logic
  initially always named the acoustic stub as "the reason" for any
  non-ACCEPT verdict, since it's permanently no_evidence and no_evidence
  lanes were checked first -- correct per the code, useless in practice,
  since it would drown out whatever a REAL lane (optical/typing) actually
  did that window. Fixed by deprioritising any lane in a
  `_PERMANENTLY_STUBBED_LANES` set unless it's the only lane present;
  regression test `test_adjudicate_names_the_implausible_lag_lane_not_the_acoustic_stub`
  added specifically for this.
- `python -m eval.ablate` run against the real 14-session bonafide optical
  corpus: no real fused (optical+typing) session data exists yet (the
  corpus is optical-only; Milestone 10's typing lane has not been run
  against a live human typing, deliberately -- simulating real OS
  keystrokes is a genuine un-consented side-effect risk, not a test-
  hygiene nicety, see Milestone 10's notes). Each real optical session was
  paired with a FIXED, clearly-labelled SYNTHETIC typing LaneResult
  (subscore=0.75, values drawn from Milestone 10's own "genuine timing
  correlation" test, not invented fresh) and the real acoustic stub, under
  min_contributing_lanes=2 -- a configuration CHOICE ("this deployment
  needs 2 independent lanes to agree"), not a data manipulation. Result:
  for every session that reached ACCEPT with all lanes present (9 of 14),
  removing EITHER real lane (optical or typing) alone dropped it to
  RE-CHALLENGE with zero exceptions; removing the acoustic stub changed
  nothing (expected, since it never contributed). Full per-session table
  in `eval/ablation.json`.

**Not yet done:** a real fused session runner that captures optical AND
typing concurrently off one shared camera stream and saves a combined log
-- this milestone's acceptance criterion is the ablation harness itself,
which is real and tested; a genuine multi-lane corpus (replacing the
synthetic typing placeholder above with real data) is follow-up work, not
gating this milestone.

**Config added:** `fusion:` section in `config.yaml` (accept_threshold,
reject_threshold, hysteresis_margin, min_contributing_lanes).

---

## Milestone 12 — attack tooling

**What was built:** `scripts/record_source.py` records N seconds of clean
face video to `data/` (gitignored, hard rule 2) with a metadata sidecar,
for later use as attack source material. `scripts/run_corpus.py` gained
`--plan PLAN_YAML` (`load_plan_from_yaml()`), letting a corpus plan live in
a versionable YAML file instead of only the module's built-in constant;
`eval/corpus_plan.yaml` is a new 6-block/19-session plan covering bonafide,
inject_static, inject_swap, inject_reenact, inject_adaptive and
emitter_off, with printed manual-setup instructions per attack block
(nothing here can auto-drive OBS or a face-swap tool). `attacks/
adaptive_injector.py`: `AdaptiveInjector` screen-captures the emitter's own
border region(s) via `mss`, estimates the currently-emitted waveform from
that pixel brightness alone (no privileged access to the session seed or
true chip values -- exactly what a second camera pointed at the monitor
would see), and multiplies a supplied source video's luminance (Y-channel
only, colour preserved) by that estimate before sending it out through
`pyvirtualcam`. `--mode global` applies one estimate uniformly (the attack
Milestone 9 exists to catch); `--mode per-zone` estimates left/right/top
separately and modulates matching thirds of the frame -- the strongest
plausible version of this attack without also running live face detection
inside the attacker tool itself. `demo/panel.py` gained collapse-latency
instrumentation: `enter_attack()` arms a timestamp, `_check_collapse_latency()`
(called every frame) records the delta the first time the score is
observed to genuinely cross below threshold WITH a face still visible
(so a dropped/no-face frame can't masquerade as a fast collapse), and
`PanelDashboard.run()` now saves every recorded delta plus its summary
stats to `logs/collapse_latency_<timestamp>.json` on exit -- including an
honest `n=0` record if a panel session never triggered a genuine collapse.

**What was measured:**

- `tests/test_adaptive_injector.py` (2/2 pass): the REAL image-processing
  code path (not the simplified scalar-signal version `tests/test_spatial.py`
  already covers) -- an unrelated source video pushed through
  `luminance_multiply()` driven by a perfect screen-capture estimate --
  raises the global optical score from 0.053 (unrelated baseline) to 0.951,
  while Milestone 9's spatial_assignment_score stays at -0.020 (dead),
  confirming the spatial defence holds against a genuine implementation of
  its target attack, not just a hand-derived formula standing in for one.
  `luminance_multiply()` verified to preserve colour ratio (scales
  luminance only).
- Bug caught by the first test run: the test's own attacker-side log was
  built in the abstract +/-1 chip-value domain instead of the 0-255
  luminance domain `region_luminance()` actually produces, collapsing the
  attack multiplier to a near-constant regardless of the true signal
  (attacked_global=0.025, LOWER than baseline). Root-caused to the test's
  setup (not `AdaptiveInjector`/`region_luminance()`, which were already
  correct) and fixed with a `_make_attacker_luminance_log()` helper that
  converts chip values to a realistic luminance-domain log first.
- `tests/test_panel.py` (9/9 pass, new this milestone): `_check_collapse_latency`
  records a delta only once a switch is pending, a face is visible, AND the
  score has genuinely crossed the threshold -- verified separately that a
  no-face gap does NOT get recorded as a fast collapse, that a still-high
  score does not get recorded, and that a second low-score frame after
  recording doesn't add a duplicate entry. `collapse_latency_stats()`
  verified against a hand-computed mean/median/min/max. `_save_collapse_latencies`
  verified to write valid JSON with the full latency list and summary
  stats, including the empty (`n=0`) case.
- `scripts/run_corpus.py --plan eval/corpus_plan.yaml` verified to load 6
  blocks/19 sessions with the correct condition distribution; omitting
  `--plan` still gives the old built-in 15-session plan unchanged
  (backward compatible).
- `AdaptiveInjector` construction verified against the real screen
  resolution (`regions={'global': (0, 0, 1920, 300)}` on this machine) and
  its `run()` verified to fail cleanly with a clear `RuntimeError` (not a
  hang or silent no-op) when the supplied `--source` file doesn't exist.

**Not yet done:** the real end-to-end hardware path -- `AdaptiveInjector.run()`
actually driving a virtual camera, and `demo/panel.py` recording a real
20-switch collapse-latency distribution against it -- has NOT been run.
This needs (a) at least one recorded source video via
`scripts/record_source.py` (`data/` is currently empty) and (b) a working
OBS Virtual Camera driver installed and selectable by `pyvirtualcam`,
neither of which is confirmed present on this machine right now. `mss` and
`pyvirtualcam` import cleanly and the pure logic (region geometry,
luminance-multiply, estimate lookup, error handling) is exercised above;
the live camera-switching/collapse-latency distribution claim
("report it as a distribution over >=20 switches, not one anecdote")
remains a mechanism that is built and unit-tested, not yet backed by a
real 20-switch session. Similarly, `eval/corpus_plan.yaml`'s 19-session
attack corpus (inject_static/swap/reenact/adaptive) has not actually been
collected -- it is a starting plan, as its own header comment says, not a
claim that data exists.

**Config added:** `attacks: adaptive_injector:` section in `config.yaml`
(mode, lag_ms, depth, baseline_luminance, fps, border_fraction,
zone_names, history_seconds). `praesens/session.py`'s `VALID_CONDITIONS`
extended with `inject_static`, `inject_swap`, `inject_reenact`,
`inject_adaptive` (old `replay`/`swap` labels kept for the existing
corpus).
