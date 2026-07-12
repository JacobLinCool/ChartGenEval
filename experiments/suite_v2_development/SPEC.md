# `suite_v2_development` retrospective evidence contract

## Status and scope

- **Experiment id:** `suite_v2_development`
- **Spec version:** `1.0`
- **Phase:** `development_descriptive`
- **Status:** retrospective freeze of the released development analysis

This specification was written **after the development results had been
inspected and after the five measurements had been selected or revised**. It
is not a preregistration, a prospective analysis plan, or a confirmatory
protocol. It records the behavior of the current code and released artifacts
so that the paper's development-only statements can be audited without
upgrading their evidence tier.

The question is narrow: on the development panel, how do five proposed
chart measurements respond to nine deterministic controlled corruptions? The
experiment does not test player judgments, perceived quality, or superiority
over another evaluation suite.

## Units, conditions, and panel

The experimental unit is one human chart under one condition. Its stable key
is `(sid, course, probe, dose_index)`.

- Dataset: `JacobLinCool/taiko-1000-parsed-clean`
- Revision: `b72da4616d643018e81f372cea06ce51349285e0`
- Development panel: canonical test indices `[0, 40)` after alias filtering
- Panel size: 40 songs and 170 non-empty course charts
- Intact condition: `probe=official`, `dose_index=0`
- Corrupted conditions: nine probes at dose indices 1, 2, and 3
- Rows per source file: `170 * (1 + 9 * 3) = 4,760`
- C5 language model: all 3,880 training charts, order 3, additive smoothing
  `alpha=0.05`, conditioned on course
- Human-reference gap: separate training references by course, with 923 Easy,
  924 Normal, 924 Hard, 924 Oni, and 185 Ura charts

The structure and coupling/gap runners must produce the same set of 170
`(sid, course)` chart keys. There are no random replicates. Probe randomness is
deterministic from SHA-256 of `(sid, course, probe, dose_index)`.

### Conditions

| Probe | Doses in current code | Controlled edit |
|---|---|---|
| `C1_timing_jitter` | 10, 20, 30 ms | Independent uniform timing jitter per hit. |
| `C1s_sparse_jitter` | 0.5%, 1%, 2% | Move `max(1, round(n*p))` hits by exactly 60 ms with random sign. |
| `C2_anchor_shift` | +15, +30, +60 ms | Translate every hit by one constant offset. |
| `C3_type_shuffle` | 0.2, 0.4, 0.8 | Flip Don/Ka color with the stated probability. |
| `C4_loop_collapse` | 0.3, 0.6, 1.0 | Replace the leading fraction with the chart's most common four-hit cycle. |
| `C5_blandification` | 0.3, 0.6, 1.0 | Rewrite selected note types to the course LM's most probable continuation. |
| `C6_density_scale` | 0.5, 1.5, 2.0 | Delete or insert hits to scale density. |
| `C7_burst_insert` | 2, 4, 8 bursts | Insert 8--16-hit bursts sampled at 12--20 notes/s. |
| `C8_bar_shuffle` | 0.3, 0.6, 1.0 | Select and permute bars using the chart BPM. |

The trial budget is exhaustive rather than adaptively stopped: both runners
must finish all 4,760 rows. They have no timeout, retry, or resume policy and
buffer rows until completion, so an execution error must fail the run instead
of publishing a partial output. Missing authored bars, a missing or mismatched
audio identity, a missing or hash-mismatched mel file, an invalid mel shape or
dtype, a wrong-course gap reference, or a row-count mismatch is fatal. A
metric-level invalid input is represented as non-finite
and handled only by the complete-row rule below. The released runs contain no
fatal failures, incomplete outputs, no-op variants, or metric-level missingness.

## Measurements

All summary values are oriented so that larger means more of the declared
desirable property. This orientation is only a mathematical convention; it is
not a quality judgment.

| Summary name / raw key | Raw direction | Required inputs | Current operational definition and development reading |
|---|---:|---|---|
| `reciprocity` / `reciprocity` | Higher | Hits and authored metrical grid | Divide the chart into two-bar phrases on sixteenth-note slots. Let `S(i,j)` be one-slot-tolerant, chance-corrected rhythmic Dice multiplied by chance-corrected color agreement. Return the mean of `S(i,i+2) * (1-S(i,i+1))`. The development reading is lower under C4 loop collapse and C8 bar shuffle, with small C1/C2 response. |
| `stagnation_alienation` / `boredom_v2_raw` | Lower | Hits and authored metrical grid, or valid BPM | Mark active bars as stagnant when they are the fourth-or-later member of a consecutive similarity-at-least-0.8 run or when every eligible 16-hit color window is periodic with period at most four. Mark a bar alienated when no at-least-0.75-similar bar occurs within 32 bars in either direction. Return `abs(stagnant union alienated) / n_active_bars`. The code key is archival: this is a structural proxy, not validated player boredom. The development reading is higher raw value under C4 and small C1/C2 response. |
| `density_energy_response` / `density_energy_spearman` | Higher | Hits, authored downbeats, cached log-mel array | Compute hits/s and mean linear mel power (`exp(mel)`) in each bar, then Spearman-correlate the two series. At least eight windows and a 90th--10th percentile energy range of at least 5% of the median are required. The development reading is invariance to uniform C6 scaling and decrease under C8. The weak C7 response does not support a general burst-sensitive audio-fit reading and is a disclosed negative result. |
| `run_head_onset_support` / `energy_support_rate_raw` | Higher | Raw hit times and cached log-mel array | Standardize positive spectral flux. A run head is the first in-frame hit or a hit whose preceding in-frame inter-onset interval is at least 0.25 s. A head is supported when the maximum flux within three frames (about 35 ms) reaches the song-specific 60th percentile. Return the supported-head fraction. The development reading is decrease under C2 translation. |
| `human_chart_gap` / `manifold_gap_raw` | Lower | Hits, BPM, course, and a same-course training reference | Map a chart to the current 32-dimensional interval, type-bigram, per-beat-density, distinctness, big-note, entropy, and color-switch representation. Standardize by course and return mean Euclidean distance to the 20 nearest same-course training charts. The development reading is larger raw gap under several C3--C8 edits and near-invariance to C2; C5 is dose- and course-dependent. |

`boredom_v2_raw` and `manifold_gap_raw` are multiplied by `-1` before response
statistics are computed. The other three raw keys use multiplier `+1`.

## Record and summary contract

The two JSONL files are released per-variant numerical records. The runners
open outputs in exclusive-create mode, so reruns must use new paths; released
records must not be overwritten. Each source contains the intact chart and all
27 corrupted variants for every chart. `corruption_noop` is preserved.

Both released record files embed the experiment id, phase, dataset revision,
panel id, spec hash, LM contract, and stable chart/condition keys. The
coupling/gap rows additionally embed group, audio, and feature-manifest
identities. Companion run manifests record execution commands, current runtime
versions, source hashes, counts, and output hashes. The historical candidate
runs used during selection remain outside this release; the present records
are a clean rerun of the selected implementations under this retrospective
spec.

For each measurement--probe pair, a chart is complete only when the intact
value and all three dose values are present and finite. Handling is exact:

- incomplete charts are excluded for that measurement--probe row only;
- values are never imputed and missing values are never changed to zero;
- no-op variants are retained and analyzed as observed, with
  `corruption_noop=true`; they are not silently excluded;
- a within-chart four-dose sequence that is constant is assigned Spearman
  `0.0` by the current summarizer;
- pooled Spearman is computed directly by SciPy on all complete oriented
  chart-dose values; the current release is finite, but the code has no
  special replacement for a globally constant pooled series;
- ties in maximum-dose change contribute neither improvement nor worsening.

The current release has no missing measurement cells, no no-op cells, and
`n_complete_charts=170` in all 45 summary rows.

For raw values `x[c,d]`, orientation multiplier `s`, and
`y[c,d] = s*x[c,d]`, the deterministic summary contains five measurements by
nine probes and exactly these columns:

| Column | Definition |
|---|---|
| `metric`, `metric_key`, `probe` | Stable measurement and condition identifiers. |
| `orientation` | Always `higher_is_better` after applying `s`. |
| `n_complete_charts` | Number of charts satisfying the four-value completeness rule. |
| `pooled_spearman_dose` | Spearman correlation of repeated doses `[0,1,2,3]` with all corresponding `y[c,d]`. |
| `mean_within_chart_spearman` | Mean chart-level Spearman across the four doses; all-constant chart sequences equal 0. |
| `mean_maxdose_oriented_delta` | Mean of `y[c,3]-y[c,0]`. |
| `median_maxdose_raw_delta` | Median of `x[c,3]-x[c,0]`, before orientation. |
| `improve_minus_worsen` | `mean(y[c,3]>y[c,0]) - mean(y[c,3]<y[c,0])`. |
| `phase` | Fixed string `development_descriptive`. |
| `spec_sha256` | SHA-256 of this retrospective evidence contract. |
| `source_sha256` | SHA-256 of the source JSONL used for that measurement. |

There are no confidence intervals, significance tests, multiplicity claims,
or held-out verdicts in this summary.

## Execution and released artifacts

Run from the repository root. Because outputs are exclusive-create, the
commands below use fresh temporary paths.

```bash
.venv/bin/python experiments/build_coupling_gap_feature_manifest.py \
  --dataset JacobLinCool/taiko-1000-parsed-clean \
  --revision b72da4616d643018e81f372cea06ce51349285e0 \
  --features ../SoftChart/eval/features_clean \
  --audio-manifest ../SoftChart/eval/ext_provenance_clean/audio_manifest_clean.json \
  --extractor-script ../SoftChart/eval/ext_provenance_clean/features_clean.py \
  --preprocess-source ../SoftChart/src/softchart/preprocess.py \
  --vocab-source ../SoftChart/src/softchart/vocab.py \
  --spec experiments/suite_v2_development/SPEC.md \
  --out /tmp/coupling_gap_feature_manifest.json

.venv/bin/python experiments/certify_structure_v2.py \
  --dataset JacobLinCool/taiko-1000-parsed-clean \
  --revision b72da4616d643018e81f372cea06ce51349285e0 \
  --limit-songs 40 \
  --spec experiments/suite_v2_development/SPEC.md \
  --out /tmp/structure_v2_development.jsonl \
  --run-manifest /tmp/structure_run_manifest.json

.venv/bin/python experiments/certify_coupling_gap.py \
  --dataset JacobLinCool/taiko-1000-parsed-clean \
  --revision b72da4616d643018e81f372cea06ce51349285e0 \
  --limit-songs 40 \
  --features ../SoftChart/eval/features_clean \
  --audio-manifest ../SoftChart/eval/ext_provenance_clean/audio_manifest_clean.json \
  --feature-manifest /tmp/coupling_gap_feature_manifest.json \
  --spec experiments/suite_v2_development/SPEC.md \
  --out /tmp/corruption_coupling_gap_clean.jsonl \
  --run-manifest /tmp/coupling_gap_run_manifest.json

.venv/bin/python experiments/summarize_suite_v2_development.py \
  --structure /tmp/structure_v2_development.jsonl \
  --coupling-gap /tmp/corruption_coupling_gap_clean.jsonl \
  --spec experiments/suite_v2_development/SPEC.md \
  --out /tmp/suite_v2_development_results.csv \
  --manifest /tmp/summary_manifest.json

cmp /tmp/structure_v2_development.jsonl \
  artifacts/records/structure_v2_development.jsonl
cmp /tmp/corruption_coupling_gap_clean.jsonl \
  artifacts/records/corruption_coupling_gap_clean.jsonl
cmp /tmp/suite_v2_development_results.csv \
  artifacts/tables/suite_v2_development_results.csv
```

| Layer | Current path |
|---|---|
| Structure execution code | `experiments/certify_structure_v2.py` |
| Coupling/gap execution code | `experiments/certify_coupling_gap.py` |
| Cached-feature manifest builder | `experiments/build_coupling_gap_feature_manifest.py` |
| Cached-feature manifest | `artifacts/suite_v2_development/coupling_gap_feature_manifest.json` |
| Structure run manifest | `artifacts/suite_v2_development/structure_run_manifest.json` |
| Coupling/gap run manifest | `artifacts/suite_v2_development/coupling_gap_run_manifest.json` |
| Per-variant structure records | `artifacts/records/structure_v2_development.jsonl` |
| Per-variant coupling/gap records | `artifacts/records/corruption_coupling_gap_clean.jsonl` |
| Deterministic summarizer | `experiments/summarize_suite_v2_development.py` |
| Canonical 45-row summary | `artifacts/tables/suite_v2_development_results.csv` |
| Summary manifest | `artifacts/suite_v2_development/summary_manifest.json` |
| Claim map | `artifacts/suite_v2_development/claim_evidence.md` |
| Release verifier | `experiments/verify_release_artifacts.py` |

The companion run and summary manifests, followed by the release
`artifacts/MANIFEST.json`, are the authoritative locations for generated-file
hashes. They are deliberately not copied into this spec, because the spec hash
is itself embedded in those generated files.

## Feature provenance boundary

The feature-manifest builder checks a 40-row audio manifest against every
pinned dataset row's `group_id` and `audio_sha256`, then records each cached
mel file's SHA-256, byte count, shape, dtype, frame count, and finiteness. It
also records the feature index, extraction report, extraction contract, and
three extraction-source hashes. The coupling/gap runner reloads and verifies
every mel file against that manifest before scoring any chart.

This binds the cached feature bytes used now, but the historical extractor's
complete dependency snapshot and extractor run manifest were not preserved in this
repository. The audio and mel arrays are not redistributed. Consequently,
this release makes no end-to-end raw-audio-to-feature byte-identity claim and
does not claim that a fresh extraction will reproduce the mel cache bit for
bit.

## Selection and revision history visible in current code

All five measurements were selected or revised using this development panel.
Only the current implementations and released artifacts are evidence here:

- `reciprocity` is the current continuous v2 primary output;
  `reciprocity_hash` is retained only as a diagnostic.
- `boredom_v2_raw` is explicitly a stagnation--alienation structural proxy;
  the archival key is not a criterion-validity claim.
- `energy_support_rate_raw` uses run heads; the per-note
  `energy_support_rate_all` is diagnostic.
- `manifold_gap_raw` uses the current 32-dimensional `phi` v2, including the
  three variety dimensions in the current source.
- `density_energy_spearman` is the current primary bar-scale coupling leaf.

Current docstrings mention earlier candidate runs and revisions in another
research repository. Those older development records are not released here,
so their historical numbers are not incorporated into this contract and must
not be described as preregistered or independently verified evidence. The
released 40-song results were observed during metric selection; there is no
selection/evaluation split within this experiment.

The current released JSONL rows are regenerated after this retrospective
contract is frozen and contain its hash. That rerun improves provenance but
does not undo the fact that metric selection and revision preceded the spec. A
future prospective run must record its own spec hash and must not present these
development rows as confirmation.

## Reporting and kill boundaries

Allowed claims are limited to the exact descriptive responses on this panel,
the operational definitions above, and reproducibility of the released files
from the current code plus stated external cache. The following claims are out
of scope: preregistration, confirmation, unbiased metric selection, player
criterion validity, a universal chart-quality score, and generalization to
other games, corpora, feature extractors, or corruptions.

The development numerical claims must be withdrawn or narrowed if any of the
following occurs:

1. the panel, row counts, source hashes, metric keys, or 45-row summary no
   longer match this contract;
2. a clean rerun under the documented current inputs does not reproduce the
   released records and summary;
3. a paper number cannot be derived from the named summary row or explicitly
   traced to the per-variant source records;
4. a future untouched panel fails the claimed corruption direction, in which
   case the claim cannot be generalized beyond this development panel; or
5. a player study contradicts a perceptual interpretation, in which case only
   the structural/audio/statistical operationalization may remain.

Known negative and mixed results are already binding: density--energy response
must not be reported as a general audio-fit measure because C7 is weak, and the
human-chart gap must not be reported as uniformly monotone under C5 because
its low-dose and course-specific directions differ.

## Revision log

- **v1.0 — 2026-07-12, retrospective freeze.** Created after all released
  development results had been inspected. It documents the current panel,
  measurements, transformations, hashes, provenance gap, reporting limits, and
  claim-kill boundaries. It does not retroactively preregister or confirm the
  experiment. Any semantic metric or panel change requires a new metric/version
  name and a new experiment id; old and new results must not be silently mixed.
