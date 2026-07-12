# Confirmatory holdout v1 — immutable experiment specification

Spec version: `1.0.0`  
Experiment id: `confirmatory_holdout_v1`  
Prospective draft written before confirmatory outcomes were inspected:
2026-07-12 (Asia/Taipei). It becomes frozen only when `freeze.py` creates the
immutable freeze manifest; that manifest, not this heading, is the evidence of
the freeze time and exact content hashes.

This file and `configs/contract.json` are the prospective analysis contract.
After the freeze manifest is created, they must not be edited. Any semantic
change to the panel, probes, metrics, controls, thresholds, or decision rule
requires a new experiment id. Corrections that do not change meaning must be
recorded as a new append-only revision note and may not rewrite an existing
run.

## Research question and claim

Do the predeclared ChartGenEval measurements respond in the expected direction
when held-out human charts receive targeted, increasing damage, after accounting
for repeated courses from the same song, corruption randomness, and invariant
controls?

The experiment can support only probe–measurement sensitivity and
complementarity. It cannot establish player preference, global chart quality,
or that any single measurement is sufficient. C5 is especially constrained:
the primary evidence is repetition and surface-variety fit. Pattern-IC fit is an
ambiguity analysis because a common-pattern rewrite may legitimately move an
outlying chart toward the calibrated human band.

## Development/confirmatory boundary

The dataset is
`JacobLinCool/taiko-1000-parsed-clean` at revision
`b72da4616d643018e81f372cea06ce51349285e0`. The test split contains 140 raw
rows representing 120 song groups. Rows are streamed in the pinned HF order,
joined by the exact `(group_id, canonical, audio_sha256)` multiset to
`artifacts/splits/clean_split_manifest.json`, then filtered to
`canonical=True`. The resulting canonical-song order, not raw-row position,
defines the panel.

- Canonical-song indices `[0, 40)` are development-only. They may be used for
  smoke tests, debugging, metric design, probe selection, and threshold design.
- Canonical-song indices `[40, 120)` are the untouched confirmatory panel:
  exactly 80 unique `group_id` and `audio_sha256` identities, including every
  non-empty course chart in each group. The full run scans all 140 raw rows,
  records 20 filtered aliases, and writes context/metric records only for the
  120 canonical rows.
- No confirmatory chart, metric value, aggregate, or intermediate result may be
  inspected before the spec, contract, code bundle, calibration, and dependency
  inputs are frozen by `freeze.py`.
- A smoke configuration may use only synthetic data or canonical indices
  `[0, 40)`.
- Once a full run is attempted, its raw directory is immutable and append-only.
  The confirmatory anchor binds that first run id and prohibits a second
  confirmatory collection under this experiment id, preventing repeated looks.
  Before the anchor is written, the runner verifies that the requested run
  directory does not exist; after the anchor is written, even the same run id
  cannot be launched again.

The C5 language model is also frozen: all 924 source rows of the full `train`
split, including 111 aliases across 813 groups and 3,880 charts, in stored
order, trigram order 3, add-alpha 0.05. The run manifest
records both an ordered training-input fingerprint and a stable fingerprint of
the fitted per-course counts, contexts, and vocabulary.
The bundled calibration must independently declare that same dataset id,
revision, training split, 3,880-chart count, trigram order, and alpha; the
runner checks every field before scoring and the freeze hashes the artifact.

The experimental unit for corruption is one `(group_id, course, probe, dose,
replicate)` record. Statistical units are first collapsed within chart and then
clustered by immutable `group_id`; corruption replicates and courses are never
treated as independent songs. `sid` is derived mechanically as
`group_<group_id>` and is not based on split position.

## Frozen probes, repetitions, and primary measurements

Probe definitions and the exact three dose values are frozen in
`configs/contract.json`. Every non-zero corruption cell is run with the five
fixed base seeds `1103, 2207, 3301, 4409, 5519`. A record-specific RNG seed is
derived from the base seed and `(group_<group_id>, course, condition, dose)` by SHA-256.
Deterministic operators such as C5 can therefore yield identical replicates;
they are still collapsed before inference and never inflate sample size.

The ten primary probe–metric pairs are:

| Probe | Primary measurement | Expected raw response |
| --- | --- | --- |
| C1 timing jitter | `timing.clean_rate` | decrease |
| C1s sparse timing errors | `timing.absolute_error_p99_ms` | increase |
| C2 global anchor shift | `timing.grid_phase_offset_abs_ms` | increase |
| C3 note-type shuffle | `transition_validity_score` | decrease |
| C4 loop collapse | `repetition_adequacy_score` | decrease |
| C5 common-pattern rewrite | `repetition_adequacy_score` | decrease |
| C5 common-pattern rewrite | `surface_variety_adequacy_score` | decrease |
| C6 note-rate scale | `density_adequacy_score` | decrease |
| C7 burst insert | `density_spike_score` | decrease |
| C8 bar-order shuffle | `repetition_adequacy_score` | decrease |

`pattern_ic_adequacy_score` under C5 is a predeclared secondary ambiguity
measure with no directional pass criterion. Raw `pattern_nll` is also retained
as a diagnostic. No total quality scalar is analyzed.

## Timing reference

Timing measurements require an external chart-timing reference. The primary
reference is the authored per-course segment map in the dataset row: measure
timestamps, meter, and tempo are used to construct a lattice; generated note
placements are not used to estimate its phase. If this reference is missing,
timing values are explicitly missing and the affected primary pair is excluded
with a reason. A BPM-only, first-note-anchored fallback is forbidden.

## Invariant and negative controls

Controls test measurement implementation, not human judgments of chart quality.
They have explicit scope:

1. `CTRL_identity` re-evaluates the identical event stream and grid. It is an
   exact invariant for every retained measurement.
2. `CTRL_joint_time_origin_shift` adds one second to both note times and the
   authored grid, and extends the evaluation duration. It changes only the time
   coordinate origin and is an invariant for timing, density, grammar, and
   surface/form measurements in this experiment.
3. `CTRL_color_bijection` swaps Don and Ka while preserving size and all note
   times. It is an invariant only for timing and density/strain measurements.
   It is **not** quality-neutral for grammar, musical semantics, or form and is
   never used as their negative control.
4. C2 global translation is simultaneously a positive control for externally
   anchored timing and a scoped negative control for chart-only density,
   grammar, and surface/form measurements, which depend on intervals and note
   classes rather than absolute time.

The exact metric scopes and equivalence tolerances are frozen in the contract.
A scoped control failure prevents a directional verdict for the corresponding
family (`INCONCLUSIVE`); it is not averaged away. Provenance/hash failure is the
separate `INVALID` case.

Chart-only interval arithmetic uses a declared integer-microsecond metric
clock. Absolute input timestamps are rounded to that clock before subtracting
the first-note tick; values within one picosecond of a half-microsecond boundary
use a fixed half-down tie rule. This resolution is 1,000 times finer than the
smallest reported timing tolerance and prevents binary-float origin shifts from
changing half-open density-window membership or interval tokens.

## Canonical transformation and row grain

`run.py` writes one append-only JSONL raw record per attempted unit,
`source_scan.jsonl` with every scanned raw row (including filtered aliases),
`sampling_context.jsonl` with canonical rows only, and a sealed run manifest.
`transform.py` independently verifies that projection and deterministically
creates:

- `observations.jsonl`: one row per successful or failed attempted unit, with
  explicit status, missingness, seed, and retained measurements;
- `chart_dose_means.jsonl`: one row per
  `(group_id, course, condition, dose)`, after averaging successful corruption
  replicates; the official baseline remains one row;
- `exclusions.jsonl`: one row per task-level exclusion reason, or per
  task-by-metric missingness reason, keyed by `(task_id, reason, metric)`;
- `source_scan.jsonl`: a canonical copy of the raw stored-order scan, preserving
  the alias-filter evidence and canonical index assigned after filtering;
- `panels.json` and `MANIFEST.json`: fixed panel definitions, commands, row
  counts, and file hashes.

Raw evidence is never rewritten. Canonical and report directories are derived
and may be deleted and regenerated, but scripts refuse to overwrite an
existing output directory so accidental mixtures fail closed.

## Planned statistical analysis

For each primary pair, raw measurements are oriented so higher means better.
Within each chart, the five corruption replicates are averaged at each dose.
The predeclared dose statistic is the Spearman correlation between the frozen
severity values (clean plus three doses) and the oriented measurement. C6 uses
two-sided distance from the target density, with frozen severities
`[0, 0.5, 0.5, 1.0]`; tied severities receive average ranks. A second
predeclared statistic is the oriented maximum-dose target minus the pair's
scoped invariant control. The sham/control comparator is frozen per pair in
`configs/contract.json`: color bijection for C1, C1s, C6, and C7; joint
time-origin shift for C2; and global chart translation (C2) for C3, C4, C5,
and C8. A control value is averaged across its five replicates before the
contrast. These comparators are measurement-invariance controls in their stated
scope; they are not asserted to be universally quality-neutral charts.

If all four oriented dose values are constant, the within-chart Spearman
statistic is defined prospectively as zero sensitivity (rather than selecting a
different statistic after observing a ceiling). A course is available for a
pair only when its official baseline, all three target doses and all five
replicates, the fixed scoped-sham dose and all five sham replicates, and every
required scoped control are complete. A song enters when it has at least one
available course. Both the dose statistic and target-minus-sham contrast use
the identical available-course set, averaged within song before inference.

Each chart statistic is first averaged across courses within its `group_id`.
The point estimate is the mean across groups. Percentile confidence intervals are
obtained by resampling `group_id` clusters, with replacement, using 10,000 bootstrap
draws and seed `20260712`. Nominal 95% intervals are reported for estimation.
The confirmatory gate uses two-sided 99.5% intervals, a Bonferroni family-wise
adjustment for the ten frozen primary pairs. Because support requires both
statistics to be negative, this intersection rule does not add another family
of claims.

Provenance, panel, snapshot, or hash failure has priority and yields `INVALID`
(normally by failing closed before an analysis artifact is written). Control
completeness and the 72-song minimum are prerequisites for either directional
verdict; if either fails, the pair is `INCONCLUSIVE` regardless of its CI.

A pair `SURVIVES` only if:

- at least 72 of the 80 canonical groups have complete evaluable data;
- every required scoped invariant control passes its frozen equivalence bound;
- the adjusted upper confidence bound for mean within-chart dose correlation
  is below zero; and
- the adjusted upper confidence bound for the mean target-minus-control
  maximum-dose contrast is
  below zero.

A pair is `REFUTED` only after the same provenance, panel, minimum-sample, and
control prerequisites pass, and an adjusted lower confidence bound for either
statistic is at or above zero. All other cases are `INCONCLUSIVE`. These outcomes may
narrow or kill a paper claim; they may not trigger metric switching within this
experiment id.

## Failure, retry, and exclusion rules

- Dataset revision, score version, spec/config/code/calibration hashes, and the
  frozen source bundle must match before a confirmatory run can start.
- Missing authored grids exclude only timing pairs and remain visible.
- A probe no-op is recorded and excluded for that pair; it is never silently
  replaced with an identity result.
- Exceptions produce error records with type and message. The runner continues
  when the failure is local to a unit and seals the run as `completed_with_errors`.
- No retry occurs inside a run. There is exactly one confirmatory attempt. A
  fatal attempt is `INVALID`, remains preserved, and requires both a new
  experiment id and a genuinely untouched panel for any new prospective gate.
  Reusing canonical indices `[40, 120)` could only be labeled
  recovery/replication and cannot
  enter the prospective gate. Local unit
  failures remain explicit within the sole run and are handled by the frozen
  completeness rules.
- Duplicate raw keys, incomplete replicate sets, unregistered measurements,
  or hash mismatches fail the transformation or analysis.

## Execution commands and evidence paths

After implementation and calibration are final, freeze exactly once:

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/freeze.py \
  --config experiments/confirmatory_holdout_v1/configs/confirmatory.json \
  --out experiments/confirmatory_holdout_v1/frozen/confirmatory_freeze.json
```

Smoke tests (synthetic only) may run before or after freezing:

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/run.py \
  --config experiments/confirmatory_holdout_v1/configs/smoke_synthetic.json
```

Before freezing, the same full chain must also pass against the first two
`canonical=True` rows (canonical indices `[0, 2)`) using
`configs/smoke_development.json`. It stops immediately after the second
canonical row and therefore cannot scan into the confirmatory panel. This is the required
real-dataset schema, authored-grid, full-train-LM, and calibration wiring check;
it remains outside `[40, 120)`.

The full command is deliberately explicit and must not be run until authorized:

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/run.py \
  --config experiments/confirmatory_holdout_v1/configs/confirmatory.json \
  --freeze experiments/confirmatory_holdout_v1/frozen/confirmatory_freeze.json \
  --confirm-untouched-panel
```

Each run writes beneath `experiments/confirmatory_holdout_v1/runs/raw/<run_id>/`.
Transformation and analysis commands are printed into their manifests and are
documented in `RUNS.md`.

The confirmatory config locks one freeze path. At the first confirmatory
attempt, the runner exclusively creates `CONFIRMATORY_ANCHOR.json`, binding the
`primary_run_id`, freeze, spec, config, contract, code-bundle, dataset, and runtime
dependency versions. Because the bound run directory is exclusively created,
no second confirmatory attempt is possible. Failure or drift requires a new
experiment id plus a genuinely untouched panel for a new prospective test;
reuse of this panel is recovery/replication only. The freeze and run
must use the repository `.venv` Python (project range `>=3.10,<3.13`) and pin
the exact Python major/minor plus NumPy, SciPy, and datasets versions.

## Planned outputs and claim-evidence map

`analyze.py` produces machine-readable primary estimates, control diagnostics,
a verdict table, an analysis manifest, and `claim_evidence.md`. C5 is one
co-primary claim: it survives only if both the frozen repetition and
surface-variety pairs survive; one surviving pair cannot rescue the other.
Each claim entry
links the pair to this spec, the contract and freeze hashes, the selected raw
run, canonical table hashes, transformation/analysis commands, exclusions,
controls, confidence intervals, and limitations. Secondary C5 pattern-IC output
is labeled ambiguity/secondary and never upgraded to a primary claim.

## Known limitations

- Controlled corruptions establish sensitivity to specified synthetic damage,
  not criterion validity with players or expert mappers.
- Courses within a song share audio and authorship context; song clustering
  addresses independence for uncertainty, but does not model every author-level
  dependency.
- Authored timing metadata is an external timing map, not a direct audio-onset
  annotation.
- Five seeds characterize the frozen operators only. Deterministic operators
  do not gain new variation from repeated seeds.
- A two-sided calibration score can improve when corruption moves an unusual
  chart toward its course band. C5 pattern-IC therefore remains ambiguous even
  if repetition/variety evidence survives.

## Predeclared sampling-context comparison

The deterministic transformation also produces a descriptive, canonical-group
comparability table for canonical development indices `[0, 40)` and
confirmatory indices `[40, 120)`. For every group it preserves raw
`source_row_index`, generated `stored_split_row_id`, the true
`manifest_source_row_id` for canonical rows, `group_id`, `canonical`, and
`audio_sha256`, and records chart/course count plus the across-chart
mean BPM, active-note rate, and authored level; the analysis reports group
counts plus mean, median, p10, and p90. This table describes sampling context
only. It cannot be used to change probes, metrics, thresholds, exclusions,
hypotheses, or the primary panel after outcomes are available.

## Spec revision log

- 2026-07-12, v1.0.0 draft: awaiting the immutable freeze manifest. No
  confirmatory results inspected.
- 2026-07-12, pre-freeze correction: panel indexing was clarified as
  `canonical=True` filtering over all 140 raw stored-order test rows; canonical
  indices `[0,40)`/`[40,120)` replace any raw-row interpretation. The immutable
  cluster identity is `group_id`, and the split manifest is an order-independent
  identity contract. No canonical rows 40–119 or confirmatory outcomes were
  read while making this correction.
- 2026-07-12, pre-freeze numerical correction: chart-only time uses an explicit
  integer-microsecond clock, and the runner now verifies calibration corpus and
  LM hyperparameter provenance against the frozen contract. Both changes were
  made before the confirmatory freeze and without reading canonical rows
  40--119.
