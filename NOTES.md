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

---

## Milestone 13 — evaluation extensions

**What was built:** `eval/analyse.py` gained five additions, all pure
functions over already-loaded data (unit-testable without a live corpus):
`per_attack_type_metrics()` (APCER/BPCER/ACER per attack condition
individually, at the SAME threshold the pooled ROC/EER picked, so one easy
or hard attack type can't hide behind a pooled average); `robustness_table()`
(bona fide score/snr_db grouped by the FULL lighting x distance x makeup x
glasses x skin_tone combination actually present in the corpus, not one
field at a time -- the existing `breakdown_by()` single-field tables are
kept alongside it, not replaced); `collapse_latency_summary()` +
`load_collapse_latencies()` + `plot_collapse_latency_cdf()` (pools every
`logs/collapse_latency_*.json` file Milestone 12's `demo/panel.py` writes
and plots a CDF); `ablation_summary_table()` (reads Milestone 11's
`eval/ablation.json` back in and reports, among sessions that ACCEPTed with
every lane present, what fraction stop doing so when each lane is held
out); `load_cross_session_proxy()` (surfaces `eval/cross_session.py`'s
output as its own clearly-labelled PROXY section in the printed summary --
"(b'')" -- never merged into the real-attack ROC/EER, since it's
constructed from bona fide recordings scored against a mismatched
challenge, not a recording of a real attack). `config.yaml`'s
`eval.attack_conditions` was extended from `[replay, swap]` to include the
new `inject_*` labels from Milestone 12, since the pooled and per-type
metrics are only as complete as the conditions they're told to look for.

**What was measured:** `python -m eval.analyse` run against the real
19-session corpus (14 bonafide + 5 emitter_off, no real attack sessions
collected yet):

- Per-attack-type table (b'): every configured attack condition
  (`inject_static`/`inject_swap`/`inject_reenact`/`inject_adaptive`/
  `replay`/`swap`) correctly reports `n=0` honestly rather than a
  fabricated or zero-defaulted APCER -- confirmed by
  `tests/test_analyse.py::test_per_attack_type_metrics_reports_honest_zero_n_for_uncollected_type`.
  Pooled ROC/EER is also correctly skipped (no attack scores at all yet)
  rather than silently running on zero attack data.
- Cross-session proxy (b''): AUC=0.9046, EER=0.1413, APCER=0.1398,
  BPCER=0.1429, ACER=0.1413 -- real numbers from the existing
  `eval/cross_session.json`, printed in its own clearly-separated section
  so it is never mistaken for a real-attack ACER.
- Robustness table (d'): 6 distinct metadata combinations present across
  14 bona fide sessions (n=1 to n=6 each) -- e.g. `glasses=with` (n=2,
  score=0.376+/-0.376) is visibly both rarer and noisier than the
  `normal/60cm/none/without` baseline (n=6, score=0.727+/-0.042), an
  interaction the single-field `glasses` breakdown in (d) also shows but
  this table additionally pins to the exact lighting/distance/makeup it
  co-occurred with.
- Collapse latency (f): correctly reports "no data recorded yet" -- no
  `demo/panel.py` session with a real source switch has been run (same gap
  Milestone 12 already documented).
- Ablation table (g): `n_sessions=14, full_accept_count=10`; optical and
  typing both show `fraction_degraded=1.00` (10/10), acoustic
  `fraction_degraded=0.00` (0/10, expected -- it's a permanent stub) --
  matches Milestone 11's already-reported ablation finding, now surfaced
  through the general analysis report instead of only `eval/ablate.py`'s
  own output.
- `tests/test_analyse.py` (11/11 pass): per-attack-type metrics verified
  to actually differentiate an easy attack type (APCER=0.0) from a hard
  one (APCER=1.0) that a pooled computation over the same 10 trials would
  have reported as a single misleading 0.5 for both; BPCER confirmed
  identical across per-type rows (it doesn't depend on attack type);
  insufficient_signal correctly excluded from per-type APCER;
  robustness_table confirmed to group by the FULL combination (two rows
  sharing every field but `lighting` are NOT merged); ablation table's
  fraction-degraded arithmetic checked against a hand-built 3-session
  fixture with a known answer.
- Full regression: 49/49 tests pass across the whole suite (38 from
  Milestones 0-12 + 11 new).

**Not yet done:** the per-attack-type table and pooled ROC/EER both need
real `inject_*` session data (Milestone 12's corpus plan,
`eval/corpus_plan.yaml`, has not been run) before they report anything but
honest zeros -- this milestone built and tested the reporting machinery,
not the missing corpus. Same for the collapse-latency CDF: the mechanism
and its figure are real and tested, the >=20-switch distribution itself is
not yet collected (needs a real `demo/panel.py` session with OBS Virtual
Camera or the adaptive injector actually running, per Milestone 12's own
notes).

**Config added:** `eval.attack_conditions` extended to include the
Milestone 12 `inject_*` labels; `eval.ablation_json` (path to Milestone
11's ablation output); `eval.collapse_latency_glob` (glob pattern for
Milestone 12's collapse-latency logs under `logs_dir`).

---

## Milestone 14 — panel demo mode

**What was built:** Two small, behaviour-preserving additions first, to
make the rest possible: `praesens/emit.py`'s `Emitter` and `praesens/
spatial.py`'s `SpatialEmitter` both gained a `drive_and_log(elapsed_s)`
method (render_frame+log_redraw in one call, returning just the frame),
and `demo/live.py`'s run loop now calls `self.emitter.drive_and_log(...)`
instead of unpacking `render_frame`'s return value itself -- their
render_frame/log_redraw shapes genuinely differ (one scalar chip value vs.
a per-zone dict), but both now expose the same one-frame interface, so
`self.emitter` can be reassigned to either type without the run loop
knowing which one is active. Nothing about what gets rendered or logged
changed (verified in `tests/test_emit.py` and the new `SpatialEmitter`
test in `tests/test_spatial.py`).

On top of that, `demo/panel.py` (Milestone 8) gained three things:

1. It now drives Milestone 9's `SpatialEmitter` instead of the
   single-zone `Emitter`, and `_process_frame` was rewritten to run its
   own per-ROI face-landmark cadence (reusing `self.landmarker` directly,
   since the inherited `CadenceDetector` merges ROIs into one mask and
   can't produce per-ROI luminance) and continuously compute BOTH the
   global score and Milestone 9's spatial assignment score from the same
   rolling window (throttled to `spatial_update_interval_s`, a real
   cross-correlation lag search over 3 ROIs x N zones isn't free every
   frame). The ACCEPT/REJECT banner's verdict is still driven by the
   global score exactly as in Milestone 8 (`current_score` is kept as an
   alias for `current_global_score`) -- the spatial score is a visible
   SECOND opinion, not a second gate -- and the status row flags
   `GLOBAL PASSES BUT SPATIAL SCORE IS DEAD` when the two disagree past
   `spatial_score_threshold`, live.
2. A `[5]` typing-lane screen (`PanelState.TYPING`) runs Milestone 10's
   PASSIVE mode against the shared camera capture: a dedicated
   `HandLandmarker` instance (never sharing timestamps with the face
   landmarker -- MediaPipe's VIDEO mode requires strictly increasing
   timestamps PER INSTANCE, so two independently-paced callers on the
   SAME instance would violate that) plus `praesens.typing.KeystrokeCapture`
   started lazily. Both are wrapped individually so a missing hand model
   or a keystroke listener that fails to start (permission, no display,
   etc.) degrades that specific piece to a visible `no_evidence` message
   instead of crashing the panel -- never a fabricated score.
3. A `--rehearse` flag drives the whole sequence (`demo.rehearse_script`
   in `config.yaml`: live -> attack -> live -> replay -> typing, each with
   its own duration) on a timer instead of waiting for keypresses,
   re-entering the SAME `enter_live`/`enter_attack`/`enter_replay`/
   `enter_typing` methods a human pressing keys would use -- so rehearsing
   exercises the real transition code, not a separate mock path. Each
   scripted step is wrapped so a failing transition (e.g. no alternate
   camera found) logs a warning and the sequence continues rather than
   crashing; `_handle_key` returns `False` once the script completes,
   reusing the existing `[Q]` quit path.

**What was measured:**

- `tests/test_emit.py` (2/2) and the new `SpatialEmitter` test in
  `tests/test_spatial.py` (now 5/5, 1 new): `drive_and_log()` produces the
  IDENTICAL frame `render_frame()` would have and logs exactly one entry
  per call, with every zone represented for `SpatialEmitter` -- confirms
  the refactor changed nothing about what's rendered or logged.
- `tests/test_panel.py` (23/23, 14 new this milestone):
  `_update_spatial_scores` reproduces Milestone 9's own core claim live --
  genuine per-zone reflectance gives NO mismatch message; a synthetic
  global-brightness-multiply attack (the same construction as
  `tests/test_spatial.py`'s own test) raises the global score above
  threshold while the spatial score stays below `spatial_score_threshold`
  AND sets the mismatch status message; a later genuine update correctly
  CLEARS a stale mismatch message rather than leaving it stuck. Two
  degradation paths verified: too few valid samples reports global=0.0 but
  spatial=NaN (no_evidence, not a fabricated zero, rule 7), and a
  partially-empty per-ROI buffer (an ROI not yet detected this window)
  returns early without crashing -- this second case caught a bug in the
  TEST'S OWN fixture (a genuinely-missing dict key raised `KeyError`,
  since the real `PanelDashboard` always initialises all three ROI keys in
  `__init__` and this was never reachable in production; the fixture was
  fixed to match that real invariant, not the production code).
  `_update_typing_status` verified to report `no_evidence` with the actual
  unavailability reason when either the keystroke capture or hand
  landmarker never started, and to recover a real coherence score (>0.7)
  from the same synthetic genuine-timing construction
  `tests/test_typing.py` already uses. `_advance_rehearse`/`_check_rehearse`
  verified to step through a script in order, survive a step whose
  transition raises (status message set, sequence continues -- this test
  also caught that `_check_rehearse` calling `self._advance_rehearse()`
  needs that method actually bound on the test double, a test-harness
  detail, not a production bug), and only advance once the scheduled time
  has actually passed. `_handle_key` verified to exit cleanly once the
  rehearsal is done and to leave normal (non-rehearse) key handling
  unaffected. Full suite: 66/66 pass.
- Real hardware: `python -m demo.panel --max-seconds 8` ran end-to-end at
  the camera's ~8fps with the spatial-scoring rewrite active throughout,
  saved a screenshot and an honest `n=0` collapse-latency log (no
  RuntimeWarning after the fix below). `python -m demo.panel --rehearse
  --max-seconds 90` ran the full 5-step scripted sequence
  (live/attack/live/replay/typing, ~29s of scripted duration plus model-
  load/camera-probe overhead, ~35s wall clock total) to completion with
  ZERO crashes: the `attack` step correctly degraded to "no injected feed
  source found" (only camera index 0 exists on this machine, no OBS
  Virtual Camera running) and the `typing` step successfully loaded the
  hand landmarker and started the keystroke listener. This is the actual
  acceptance evidence for the hard rule "never crashes on a missing
  device" -- a real run hit a genuinely missing device (the alternate
  camera) and degraded exactly as designed. A `RuntimeWarning: Mean of
  empty slice` surfaced on the first such run (a live rolling window can
  contain a timestamp where every ROI is momentarily NaN between two good
  frames -- `tests/test_spatial.py`'s fixed 20s fixtures never hit this
  because they're fully populated) -- cosmetic only (NaN-in, NaN-out,
  handled correctly downstream), suppressed locally in
  `_update_spatial_scores` rather than touched in `praesens/spatial.py`
  itself, which needed no change.

**Not yet done:** the rehearsal ran with only the primary camera present
(no OBS Virtual Camera, no `attacks/adaptive_injector.py` actually running
as a second source on this machine) -- so while the ATTACK step's
graceful-degradation path is now real, verified evidence, the actual
"global score recovers, spatial score stays dead" visual has only been
demonstrated in `tests/test_adaptive_injector.py` (Milestone 12) and the
new synthetic `test_update_spatial_scores_flags_mismatch_under_global_multiply_attack`
here, not in a live panel session with a genuine competing camera source.
Same for the typing lane: the hand landmarker loaded and the keystroke
listener started successfully, but no keys were actually typed during the
smoke test (an automated script typing real OS-level keystrokes would be
the same un-consented side-effect risk Milestone 10's notes already
flagged), so the live coherence readout itself has not been exercised with
a real human typing during a `--rehearse` run -- only with the synthetic
fixture in `tests/test_panel.py`. A fully-populated ACTIVE-mode typing
screen (with a phrase to type) was deliberately left out of scope in
favour of PASSIVE mode, which needs no expected-sequence bookkeeping in a
panel context.

**Config added:** `demo.spatial_score_threshold`, `demo.spatial_update_interval_s`,
`demo.rehearse_script` (the 5-step scripted sequence, each entry a
`{state, duration_s, description}` block).

---

## Camera throughput investigation (2026-09-01) — MJPG did NOT fix it; the ceiling is exposure time, not pixel format

**What was built:** New `praesens/capture.py`: `CaptureConfig` dataclass
+ `configure_capture_format(cap, config, warn_list=None)`, the ONE shared
function that now sets FOURCC -> width -> height -> requested_fps (DSHOW-
safe order, empirically matching the ordering discipline this project
already applies to exposure/WB) on every real capture site --
`demo/live.py`, `demo/attack.py`'s `probe_camera`, `praesens/session.py`,
`praesens/spatial.py`'s `__main__`, `praesens/typing.py`'s `__main__`,
`scripts/diagnose.py`, `scripts/verify_timing.py`, `scripts/
record_source.py`. It reads back what the driver actually accepted
(requested and actual routinely differ), runs a real 30-frame
grab+retrieve measurement (reusing `praesens.optical.measure_capture_fps`,
not reimplemented), and appends a warning to a caller-owned `warn_list`
(same convention as `praesens.optical.lock_camera`) if the measured rate
falls under `capture.warn_below_fps`. It never touches exposure/WB itself
-- `lock_camera` is still called separately, at its existing call sites,
completely unchanged. Two deliberate exclusions: `scripts/probe.py`
(Milestone 0's pre-flight check, which must run before `praesens/` is
trusted, so it applies the same `capture:` block itself, independently,
with no import) and `attacks/adaptive_injector.py` (opens a video FILE or
a secondary camera as attack SOURCE material, not the optical lane's own
capture -- forcing live-camera properties on a file capture would be
wrong).

`scripts/probe.py` also gained a second, more honest FPS measurement:
`measure_fps_at_production_exposure()`, which locks exposure to
`optical.exposure_value` (the value the real pipeline actually runs
under) the same way `lock_camera()` does, then re-measures. This exists
because `probe_resolution_and_fps()`'s existing measurement runs BEFORE
any exposure lock, under the driver's fast auto-exposure default --
which is not a condition the real pipeline ever runs in, and was silently
overstating achievable throughput.

**What was measured -- the honest result, not the hypothesized one:**

The working hypothesis (documented in this milestone's own commit
message before testing) was that the camera was delivering an
uncompressed YUY2 stream that saturates the USB link at ~8fps, and that
forcing MJPG (on-camera JPEG compression, far less raw bandwidth) would
unlock a much higher rate. That hypothesis is **wrong for this camera**,
confirmed by direct measurement, not assumed:

- `python scripts/probe.py --skip-display` (`logs/probe_1788232170.json`):
  `fourcc_requested='MJPG' -> fourcc_actual='YUY2'` (the driver silently
  declined the FOURCC request and stayed YUY2) in every run.
  `probe_resolution_and_fps`'s measurement (taken before any exposure
  lock) varied run-to-run with the camera's residual driver state --
  29.97fps on one run, 8.0fps on another, since it depends on whatever
  auto-exposure state the driver happened to carry in from the previous
  process -- which is itself part of why a measurement taken before
  locking exposure isn't representative and needed replacing.
- The new production-exposure re-measurement (exposure locked to
  `optical.exposure_value=-3`, matching every real session) is the
  number that matters, and is NOT run-to-run noisy: **7.99fps** in the
  logged run above, **8.00fps** and **8.15fps** in two earlier runs of
  the same measurement -- all within measurement noise of each other and
  of the previously-documented ~8fps baseline (`config.yaml`'s own
  `-3 ~8fps WITH a face` note from the 2026-08-06 exposure sweep).
- A direct isolation sweep (`cv2.VideoCapture` driven by hand, exposure
  locked the same way as `lock_camera`, MJPG + explicit
  `CAP_PROP_FPS=30` requested either way): `exposure_value=-3 ->
  8.15fps`, `-4 -> 16.56fps`, `-5 -> 31.16fps`, `-6 -> 30.33fps`, **with
  or without MJPG requested -- the fourcc readback stayed YUY2 in every
  case, and fps tracked exposure_value only.** This exactly matches
  `config.yaml`'s already-documented `fps ~= 1/2^exposure_value`
  relationship: doubling the exposure time each step (`-3` -> `-4` ->
  `-5`) roughly doubles the achievable frame period, independent of
  pixel format.

**Conclusion, stated plainly per the brief's own instruction not to dress
up a null result: MJPG does not raise fps above ~8 once exposure is
locked to the value this room's lighting requires (`-3`). The ceiling is
the sensor's manual exposure TIME (a real physical constraint --
longer exposure per frame directly caps frames per second, independent
of how those frames are encoded/transferred), not the pixel format.**
This driver on this camera never actually honoured the MJPG FOURCC
request at all (readback stayed `YUY2` in every test run, at every
exposure value) -- whether that's a driver limitation specific to this
UVC device, or MJPG genuinely wasn't the bottleneck so the driver had no
reason to switch, wasn't distinguishable from the outside, but either way
the practical answer is the same: forcing MJPG bought nothing here.

The `-1.0` `Reported FPS` / 137ms max-gap symptom that originally
motivated the MJPG hypothesis is real and still present (still `-1.0`,
still large frame gaps under auto-exposure -- see the raw probe JSON),
but it turns out to be a red herring for the actual 8fps ceiling: it's a
driver readback quirk under YUY2, correlated with but not causally
responsible for the low production fps, which is fully explained by
exposure time alone.

**What this means for the real fix:** the only two levers that actually
move this camera's achievable fps at face-detectable exposure are (a)
more physical light near the subject, allowing a less-negative
`exposure_value` (each notch roughly doubles fps, per the sweep above),
or (b) continuing to rely on `auto_chip_rate` (already in place since
2026-08-06) to pick a chip rate the ACTUAL achievable fps can sample
correctly, rather than chasing a higher fps that this hardware/lighting
combination cannot deliver. `praesens/capture.py` is kept, not reverted --
it is a real, tested, useful refactor (one shared FOURCC/resolution/fps
setup point, honest warn_below_fps diagnostics) regardless of this
specific negative result, and would matter on a camera/driver
combination where MJPG negotiation actually succeeds.

**Tests:** `tests/test_capture.py` (9/9 pass) -- verifies the DSHOW-safe
property-setting order (FOURCC before width/height/fps) against a mocked
capture, that `fourcc: null` skips the FOURCC call entirely (and that
width/height/fps=0/falsy are each individually skipped the same way),
the requested-vs-actual readback contract, and the `warn_below_fps`
warning firing/not-firing correctly off a monkeypatched
`measure_capture_fps`. Full suite: 75/75 pass.

**Config added:** top-level `capture:` block (`fourcc`, `width`, `height`,
`requested_fps`, `warn_below_fps`).

---

## Preflight FPS staleness fix (2026-09-01) — auto_chip_rate was picking a chip rate for a camera state that no longer existed by the time the session ran

**What was found:** with `optical.exposure_value` moved to `-4` (more
light became available; `-3`'s ~8fps ceiling from the capture-throughput
investigation above no longer applied), real sessions were running at a
genuinely-achieved `measured_fps=16.0` with 100% face detection, but
`auto_chip_rate`'s preflight measurement still reported `8.0` and picked
`chip_rate_hz=0.80` -- SNR stayed stuck. Root cause confirmed by reading
the actual call order, not assumed: in `praesens/session.py`,
`preflight_fps = measure_capture_fps(cap)` (feeding `pick_auto_chip_rate`,
which fixes `chip_rate_hz` for the whole session) ran BEFORE exposure was
ever locked -- `lock_camera` only happened later, inside
`praesens.optical.run_session()`, called AFTER the chip-rate decision was
already baked into `challenge_cfg`. The preflight was measuring whatever
fps the driver defaulted to on open, not the fps achievable at the
CONFIGURED `exposure_value`. `praesens/spatial.py`'s equivalent block did
NOT have this specific bug (it already locked exposure before measuring)
but had no explicit warm-up burst either.

**What was built:** `praesens/capture.py` gained
`measure_steady_state_fps(cap, optical_config, warmup_frames, warn_list=None)`
-- locks exposure (`praesens.optical.lock_camera`, completely unchanged:
same candidate sweep, same `exposure_value`, same `disable_auto_wb`
logic), discards `warmup_frames` frames (grab+retrieve pairs, matching
`measure_capture_fps`'s own DSHOW-safe pattern), THEN measures. New config
key `optical.camera_warmup_frames` (default 30). `praesens/session.py`'s
auto_chip_rate block now calls this instead of a bare
`measure_capture_fps(cap)`; `lock_camera` still runs again later inside
`run_session()` exactly as before (idempotent, harmless -- deliberately
NOT removed, to keep this fix to ONLY the ordering of the preflight
measurement, not a restructuring of `run_session()` itself).
`praesens/spatial.py`'s `__main__` gained the same warm-up burst inline
(exposure there was already locked in the right place, just needed the
settle time).

**What was measured:** `python -m praesens.session --condition bonafide --camera-index 0`
(`logs/20260901T090233_241b5f1d.json`):

- `auto_chip_rate: measured preflight FPS=16.0 -> chip_rate_hz=1.60,
  duration_s=37.4` -- exactly `16.0 / auto_chip_rate_divisor(10.0) = 1.60`,
  matching the real achieved rate, not the stale 8.0 reading.
- `measured_fps=16.0` for the actual scored session -- IDENTICAL to the
  preflight prediction, confirming the mismatch is gone.
- `snr_db=7.72` (floor is `3.0`) and `insufficient_signal=False`.
- `score=0.786`, `n_frames=556`, `n_face_detected=556` (100%).

**Tests:** `tests/test_capture.py` gained 6 tests for
`measure_steady_state_fps` (21 total in that file): `lock_camera` runs
before any frame is grabbed; the measurement call happens only after
EXACTLY `warmup_frames` grab/retrieve pairs (verified by recording
`cap.grab.call_count` at the moment the (mocked) `measure_capture_fps` is
invoked -- the actual acceptance criterion, not just "the functions got
called in some order"); a different `warmup_frames` value is respected;
`warmup_frames=0` doesn't crash; `warn_list` is forwarded to
`lock_camera` unchanged; a missing `warn_list` doesn't raise. Full suite:
81/81 pass.

**Config added:** `optical.camera_warmup_frames` (default 30).

---

## Concurrent optical+typing session with one clean verdict (2026-09-02)

**What was built:** `python -m praesens.session` now runs optical and
typing CONCURRENTLY off ONE shared capture loop by default
(`--lanes optical+typing`, `--lanes optical` for the old single-lane path).

- **The decomposition that made this possible:** `praesens/optical.py`'s
  `run_session()` used to OWN its grab/retrieve loop; it's now a thin
  wrapper around a new `OpticalFrameProcessor` (per-frame ROI sampling +
  adaptive-boost tracking, extracted verbatim -- no scoring math changed)
  with a `.finalize()` that does the exact same post-loop detrend/cross-
  correlate/SNR computation as before. `praesens/typing.py`'s
  `run_typing_session()` got the identical treatment: a new
  `TypingFrameProcessor` for per-frame hand-tracking, `finalize_typing_result()`
  for the post-loop computation. Both single-lane wrappers are unchanged
  in behaviour for their existing callers (diagnose.py, each module's own
  smoke test) -- this was a pure extraction, verified by re-running real
  bonafide sessions before/after and confirming consistent scores.
- **`praesens/session.py`'s new `_run_concurrent_lanes()`** is the actual
  shared loop: locks exposure once, builds one `OpticalFrameProcessor` +
  one `TypingFrameProcessor` (its own hand landmarker -- a face landmarker
  and a hand landmarker are different objects with independent timestamp
  sequences, no MediaPipe VIDEO-mode collision, same discipline already
  established in Milestone 14's panel.py), starts the keystroke listener,
  generates the challenge phrase up front (`generate_challenge_phrase`,
  `generative:false` path, unchanged), then ONE loop: grab, retrieve,
  feed the SAME frame object to `optical_processor.process_frame()` then
  `typing_processor.process_frame()`, overlay the phrase (on a COPY, so
  the frame optical already scored is untouched) into the emitter
  preview. Mirrors the existing macOS/Windows platform branch for who
  owns the main thread.
- **Fusion wiring:** `optical_lane_result()` wraps `OpticalResult` into a
  `LaneResult` using the SAME confidence formula `eval/ablate.py`'s
  `optical_lane_result_from_log()` already established (SNR margin,
  clipped [0.05, 1.0]) -- reused, not re-derived. `typing_lane_result()`
  reuses `praesens.typing.passive_confidence()`'s own recency-decay
  formula. Both feed `Adjudicator.adjudicate()` unchanged (plus the
  acoustic stub) for one real `JointResult`, even in `--lanes optical`
  mode (optical+acoustic alone, so a single-lane session still gets a
  genuine three-way verdict, not a special-cased rule).
- **Terminal output:** `print_clean_block()` prints the short block from
  the brief; PASS/FAIL/NO EVIDENCE come from the lane's real `status` +
  whether it contributed to the joint score (fusion.py's own
  `compute_joint_score`) + whether its subscore itself clears
  `reject_threshold` (see the bug below) -- no new pass/fail threshold
  invented, all three checks are existing fusion.py concepts. REJECT/
  RE-CHALLENGE print the adjudicator's OWN `reason_text` verbatim; ACCEPT
  gets one fixed, honest phrase ("both lanes agree..." / "N lanes
  agree..." / "{lane} lane agrees...") sized to how many lanes actually
  contributed. Console noise (MediaPipe/TensorFlow C++ logging) is
  suppressed by default via `GLOG_minloglevel`/`TF_CPP_MIN_LOG_LEVEL` env
  vars PLUS an OS-file-descriptor-level redirect (`_quiet_console`) --
  `contextlib.redirect_stdout` alone does NOT catch it, since MediaPipe's
  C++ backend writes directly to the underlying fd, bypassing anything
  that only reassigns `sys.stdout`/`sys.stderr` at the Python level.
  `--verbose` skips the redirect; nothing is ever lost either way, it
  either shows live or goes to `logs/console_<ts>.log`.

**STOP-and-flag finding, resolved before building the above (per explicit
instruction to verify with the real Adjudicator and stop if it
reproduced):** confirmed a real bug in `Adjudicator.adjudicate()` -- a
confidence-weighted mean let a strong, confident typing lane numerically
outvote an optical lane that, on its own terms, already read as an
attack (subscore=0.05, confidence=0.20 -- the lowest confidence reachable
while status is still "ok" -- plus typing subscore=0.95 confidence=1.0
-> joint=0.80, ACCEPT). Fixed in `praesens/fusion.py`: after the normal
threshold branches compute a verdict, if it's ACCEPT and any
CONTRIBUTING lane's own subscore is below `reject_threshold`, downgrade
to REJECT -- scoped narrowly, only pulls back an about-to-be-lenient
ACCEPT. Also fixed a related `_summarize()` bug found while writing the
regression test: a RE-CHALLENGE where two real lanes both contributed
but the mean fell in the gap between thresholds used to name the
permanently-stubbed acoustic lane instead of the real joint score
(dead-code fallback, unreachable whenever acoustic was present).
15 fusion tests pass (11 original + 4 new), `eval/ablate.py` re-run
against the full real corpus with the same CONFIRMED result.

**A second real bug caught by hardware verification, not by review:** the
first version of `print_clean_block()` labelled a lane PASS whenever it
"contributed" (plausible lag), without checking whether its subscore
itself cleared `reject_threshold` -- a real session showed
"Optical lane : PASS (score 0.17)" with `reject_threshold=0.3`, an
outright misleading label for a session that correctly REJECTed. Fixed
by adding `reject_threshold` to the record's `fusion` dict (add-only) and
requiring `contributes AND subscore >= reject_threshold` for PASS.
Regression test added; re-verified on real hardware (now shows
"Optical lane : FAIL (subscore -0.03, below reject_threshold)" for a
session that scored badly).

**What was measured (real hardware, `logs/2026090[19]*`):**

- `--lanes optical+typing`, genuine face, no typing: optical PASS
  (score 0.81) + typing NO EVIDENCE (n_keystrokes=0) -> **ACCEPT**
  (joint 0.81), reason "optical lane agrees with this session's
  challenge" (the single-real-contributing-lane phrasing).
- `--lanes optical+typing`, genuine face, insufficient signal this
  particular take: optical NO EVIDENCE (insufficient_signal,
  snr_db=-2.17) + typing NO EVIDENCE -> **RE-CHALLENGE**, reason names
  the real cause.
- `--lanes optical+typing --camera-index 1` (OBS Virtual Camera,
  `inject_static`): optical NO EVIDENCE (insufficient_signal) across
  three separate runs, once optical FAIL (subscore -0.03, a genuine
  bonafide take that happened to score badly) -> REJECT/RE-CHALLENGE
  every time, never a false ACCEPT.
- `--lanes optical` (backward-compat path): identical clean-block
  presentation, single-lane verdict, "Starting..." line correctly omits
  the typing prompt.
- Exactly ONE `open_camera()` call confirmed both structurally (grep:
  one call site in the whole module) and dynamically
  (`tests/test_session.py`'s mocked-hardware test asserts
  `mock_open_camera.assert_called_once()` and that both lanes' per-frame
  processors receive the literal SAME frame object from a single
  grab()/retrieve() pair per loop iteration).
- Console suppression verified: default run shows only the clean block
  (11-line `logs/console_*.log` captures everything else, zero
  MediaPipe/TensorFlow spam on the terminal); `--verbose` shows
  everything live, confirmed.

**Not yet done:** the literal "I type the phrase while it plays, both
lanes PASS, VERDICT ACCEPT" case needs a human actually typing --
deliberately not simulated here (`pynput.keyboard.Controller()` would
deliver real OS-level keystrokes to whatever window has focus, the same
un-consented side-effect risk Milestone 10's notes already flagged for
this exact reason). Every OTHER path (no typing, insufficient signal,
optical fail, backward-compat single-lane, the shared-loop mechanics
themselves) is verified on real hardware above; the genuine dual-PASS
case is mechanically identical to what's already proven working (typing
already reached `status=ok` with real coherence numbers in Milestone
10's own hardware test, and optical PASS is demonstrated above) but
hasn't been observed end-to-end together with a live human typing during
a `--lanes optical+typing` run.

Also observed, not a code issue: real keystroke events (27, then 10) were
captured during two separate hardware runs where no one was
intentionally typing -- correctly resulted in NO EVIDENCE both times
(hand-tracking couldn't corroborate them, exactly the honest degradation
this lane is designed for) and, per the privacy contract, only
timing/counts were ever recorded, never characters. Source unidentified
(pynput's OS-level keyboard hook can pick up any process's keystrokes,
not just ones intended for this session) -- worth the operator's
awareness if running near other automation, though it does not affect
this milestone's correctness.

**Tests:** `tests/test_session.py` (new, 12 tests): `optical_lane_result`/
`typing_lane_result`'s confidence formulas; `print_clean_block`'s PASS/
FAIL/NO EVIDENCE logic including the reject_threshold regression case;
the shared-loop/single-open-camera mocked-hardware test; `--lanes optical`
never touches typing. Plus 1 new test in `tests/test_typing.py`
(`chars_matched` is a count, never a character). Full suite: 98/98 pass.

**Config added:** none new for this milestone (`typing:`/`fusion:` config
keys already existed from Milestones 10/11).
