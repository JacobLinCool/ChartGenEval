# ChartGenEval release artifacts

This directory contains the numerical evidence released with the paper. It
contains no audio, note-event arrays, generated charts, model checkpoints,
third-party source code, or input assets. `MANIFEST.json` records the byte size,
row count (where applicable), and SHA-256 digest of every released file. Verify
the bundle with:

```bash
python experiments/verify_release_artifacts.py
```

## Calibration and split

- `../src/chartgeneval/data/taiko_1000_parsed_clean_calibration_v2.json`
  contains per-course quantiles fitted to all 3,880 training charts at dataset
  revision `b72da4616d643018e81f372cea06ce51349285e0`. Its score version is
  `calibrated_diagnostics_v1_symmetric_halfwidth`.
- `splits/clean_split_manifest.json` records all 1,155 source rows in 1,013
  audio/title groups. The canonical train/validation/test panels contain
  813/80/120 groups, with no cross-split audio-SHA or normalized-title overlap.

## Sealed confirmatory evidence

`confirmatory_holdout_v1/` is a compact copy of the sealed confirmation on the
80 untouched canonical test songs at indices `[40, 120)`. The release includes
only the evidence needed to audit the result:

| File | Role |
|---|---|
| `freeze.json` | Frozen specification, runtime, dataset revision, calibration, and code-bundle hash. |
| `anchor.json` | Unique-run anchor binding the freeze to `confirmatory-primary-20260712`. |
| `run_manifest.json` | Completed-run status, panel fingerprints, counts, and hashes of the withheld raw records. |
| `table_manifest.json` | Deterministic transformation contract and hashes of the withheld canonical tables. |
| `analysis_manifest.json` | Cluster-bootstrap settings, provenance checks, and hashes of every released report. |
| `primary_results.json` | Ten preregistered primary probe--metric results. |
| `co_primary_results.json` | The preregistered formal C5 all-pairs verdict; its two rows were later shown to be algebraically equivalent. |
| `secondary_results.json` | Direction-free C5 pattern-IC and pattern-NLL diagnostics. |
| `control_diagnostics.json` | All 32 scoped-control checks and tolerances. |
| `pair_exclusions.json` | Four course-level exclusions produced by the frozen completeness rule. |
| `sampling_context_comparability.csv` | Descriptive development/confirmation context comparison. |
| `claim_evidence.md` | Human-readable claim-to-evidence and provenance map. |

The 50,283 per-attempt raw records and the 10,323 chart-dose rows are not
redistributed. Their SHA-256 digests and row counts remain sealed in the run
and table manifests. The manifest hash chain binds freeze -> anchor -> run ->
table -> analysis -> reports, so the compact bundle can be checked against the
full local evidence without publishing restricted chart content.

`tables/confirmatory_primary_results.csv` is a deterministic 10-row projection
of `primary_results.json`. `dose_mean` is the oriented within-chart dose
statistic. `target_minus_sham_mean` is the oriented target-minus-scoped-sham
contrast. Both intervals are the preregistered family-wise 99.5% cluster
bootstrap intervals. All ten frozen rows and all 32 required controls pass.
The C5 repetition and surface-variety rows are the same effective test: their
raw quantities sum to one, and mirrored symmetric calibration makes their
chart-level scores identical. The release therefore contains nine
nonredundant probe--measurement tests, not ten independent tests. The sealed
`co_primary_results.json` and `claim_evidence.md` preserve the preregistered
contract and reducer output; they are not evidence of metric independence.
The post-freeze proof, numerical checks, and interpretation boundary are in
[`C5_DEPENDENCY_AUDIT.md`](C5_DEPENDENCY_AUDIT.md).

## Development probe records

The five proposed development-only measurements are governed by the
retrospective contract in
`../experiments/suite_v2_development/SPEC.md`. The spec explicitly records
that metric selection and revision preceded this freeze; these records are not
confirmatory evidence.

| File | Rows | Role |
|---|---:|---|
| `records/corruption_probes_development.jsonl` | 4,760 | Current 40-song development probe panel: official + C1/C1s/C2--C8 at three doses, using the symmetric-half-width score version. |
| `records/corruption_borrowed_metrics_clean.jsonl` | 4,250 | Borrowed baselines and the chart-only C2-insensitivity archive. |
| `records/corruption_set_metrics_clean.jsonl` | 125 | Set-level Self-BLEU and type-token-ratio records. |
| `records/corruption_coupling_gap_clean.jsonl` | 4,760 | Audio-coupling and 32-dimensional, course-conditioned human-reference-gap records, bound to the dataset, spec, audio identities, and cached-feature manifest. The reference counts are Easy 923, Normal 924, Hard 924, Oni 924, and Ura 185. |
| `records/structure_v2_development.jsonl` | 4,760 | ABAB-reciprocity and stagnation--alienation structure records on the same development variants, bound to the dataset, spec, and run manifest. |
| `records/timing_corruptions_clean.jsonl` | 3,400 | Authored/estimated-grid timing records for C1, C1s, and C2. |
| `records/cross_lm_c5_clean.jsonl` | 171 | Cross-language-model C5 diagnostic archive. |

`tables/c2_shift_recovery_clean.csv` is the alias-aware aggregate of the C2
phase witness. The per-chart witness values remain in
`timing_corruptions_clean.jsonl`.

`tables/suite_v2_development_results.csv` is a deterministic 45-row summary of
the five proposed development-only measurements. It reports pooled
dose Spearman response, the mean within-chart Spearman response, oriented
maximum-dose change, and the fraction improving minus the fraction worsening.
These are descriptive development statistics: the table carries no held-out
verdict, confidence interval, or player-validity claim. Each row records the
SHA-256 digests of the governing spec and source JSONL file. The release
verifier reconstructs the full 170-chart by 28-condition Cartesian panel and
byte-compares a fresh in-memory summary, so duplicate, missing, or hand-edited
cells fail closed.

`suite_v2_development/` closes the input-to-claim provenance chain:

| File | Role |
|---|---|
| `coupling_gap_feature_manifest.json` | Forty dataset/audio identities plus per-mel SHA-256, byte count, shape, dtype, frame count, finiteness, extraction contract, and extraction-source hashes. |
| `structure_run_manifest.json` | Structure execution command, spec and source hashes, runtime versions, counts, and output hash. |
| `coupling_gap_run_manifest.json` | Coupling/gap execution command, feature/spec/source hashes, LM and reference contracts, runtime versions, counts, and output hash. |
| `summary_manifest.json` | Deterministic transformation command and hashes for both record inputs, the analysis script, and the 45-row output. |
| `claim_evidence.md` | Paper development claims mapped to exact rows, scripts, records, and interpretation limits. |

The cached mel arrays and source audio are not redistributed. Their historical
extraction dependency snapshot was not preserved; the feature manifest binds
the exact cached tensors used by the released records but does not establish
byte-identical end-to-end re-extraction from raw audio.

## Evaluated-system records

| File | Rows | Role |
|---|---:|---|
| `records/system_timing_clean.jsonl` | 1,320 | Authored/estimated timing profiles for the official row and five evaluated systems. |
| `records/system_coupling_clean.jsonl` | 660 | Game-agnostic audio-coupling profiles. |
| `records/system_baselines_clean.jsonl` | 380 | Official, Mapperatorinator, and TaikoNation borrowed-metric profiles. |
| `records/system_set_metrics_clean.jsonl` | 10 | System-level Self-BLEU and set statistics. |
| `records/system_profile_mapperatorinator.jsonl` | 170 | Current sanitized Mapperatorinator chart-side profile using the pinned calibration and symmetric-half-width scores. |
| `records/system_profile_taikonation.jsonl` | 40 | Current sanitized TaikoNation chart-side profile using the pinned calibration and symmetric-half-width scores. |

`tables/mapperatorinator_difficulty_summary.csv` is the deterministic
course-by-course summary used by the exploratory difficulty figure. Easy
through Oni contain 40 paired charts; Ura contains 10.

`systems/conditions.json` records source revisions, observed license status,
checkpoints, and explicit native-setting-to-reference-course mappings. Records
that use those mappings carry
`course_mapping_version=system_reference_course_v1`; undeclared mappings fail
instead of selecting a course implicitly. TaikoNation and AutoOsu declare no
software license in their public repositories, so this release contains only
their provenance and derived metric values.

## Rebuilding the paper figures

The paper figures read the released numerical records:

```bash
python experiments/make_paper_figures.py
```

Licensing and dataset provenance are defined in `../ARTIFACT_LICENSE.md`.
