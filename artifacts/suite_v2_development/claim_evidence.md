# Suite-v2 development claim--evidence map

## Evidence status

This map covers the five development-only measurements discussed in
`paper/paper.tex` under `sec:newmetricresults`. The measurements and the
reported responses were selected or revised after this 40-song development
panel had been inspected. The evidence is therefore descriptive and
retrospective: it is neither preregistered nor confirmatory.

The governing contract is
`experiments/suite_v2_development/SPEC.md` version 1.0. Its experiment id is
`suite_v2_development` and its phase is `development_descriptive`.

## Shared traceability chain

| Layer | Evidence |
|---|---|
| Dataset and panel | `JacobLinCool/taiko-1000-parsed-clean` at `b72da4616d643018e81f372cea06ce51349285e0`; canonical test indices `[0,40)`, 40 songs, 170 charts. |
| Conditions | Intact chart plus C1, C1s, and C2--C8 at three doses: 4,760 rows in each source. |
| Structure execution | `experiments/certify_structure_v2.py` |
| Coupling/gap execution | `experiments/certify_coupling_gap.py` |
| Governing spec | `experiments/suite_v2_development/SPEC.md`, SHA-256 `78bdc08cd414195af5e372c0b5c302f56040259d0689789ff135db1e7e1e948f` |
| Cached-feature binding | `artifacts/suite_v2_development/coupling_gap_feature_manifest.json`, SHA-256 `086b3cf9826b0a50eede3857b6d19fa11d0a2d4c184392be6dc7cfd23c14fbf9` |
| Structure records | `artifacts/records/structure_v2_development.jsonl`, SHA-256 `0babae4e8e0b673ed487a20ce5d90786297ab5ed0632180fbe842c9e927442c7` |
| Coupling/gap records | `artifacts/records/corruption_coupling_gap_clean.jsonl`, SHA-256 `46d2f6acea1184f6d1c995a0ef4738e5c9b239c61d3609e764e8cfff1c564298` |
| Deterministic analysis | `experiments/summarize_suite_v2_development.py` |
| Canonical summary | `artifacts/tables/suite_v2_development_results.csv`, 45 rows, SHA-256 `3deb8c56924a715a2600efa8e09cd304f6a35976e31204725c6efcf41951d698` |
| Run/analysis manifests | `artifacts/suite_v2_development/{structure_run_manifest,coupling_gap_run_manifest,summary_manifest}.json` |
| Semantic release checks | `_verify_suite_v2_development_evidence` in `experiments/verify_release_artifacts.py` |
| Metric implementations | `src/chartgeneval/metrics/structure.py`, `coupling.py`, and `gap.py` |
| Paper output | `paper/paper.tex`, subsection `sec:newmetricresults` |

The summary command is:

```bash
.venv/bin/python experiments/summarize_suite_v2_development.py \
  --structure artifacts/records/structure_v2_development.jsonl \
  --coupling-gap artifacts/records/corruption_coupling_gap_clean.jsonl \
  --spec experiments/suite_v2_development/SPEC.md \
  --out /tmp/suite_v2_development_results.csv \
  --manifest /tmp/suite_v2_development_summary_manifest.json
cmp /tmp/suite_v2_development_results.csv \
  artifacts/tables/suite_v2_development_results.csv
```

The summary includes only charts with a finite intact value and all three
finite dose values for that measurement--probe pair. It performs no
imputation. No-op rows are retained; all-constant within-chart dose sequences
receive Spearman 0. The current 45 rows each contain all 170 charts. See the
spec for the exact formulas and failure behavior.

## Claim 1: ABAB-style reciprocity responds to form edits

**Paper claim.** Reciprocity decreases under loop collapse and bar shuffling
while changing little under timing jitter and global translation.

**Released numbers.** These are `pooled_spearman_dose` values from rows with
`metric=reciprocity`:

| Probe | Pooled Spearman |
|---|---:|
| `C4_loop_collapse` | -0.6121628015 |
| `C8_bar_shuffle` | -0.5055849414 |
| `C1_timing_jitter` | -0.0862393215 |
| `C2_anchor_shift` | -0.0034174955 |

**Evidence path.** The raw key `reciprocity` is in every structure record. It
is computed by `call_response_reciprocity` in
`src/chartgeneval/metrics/structure.py`, emitted by
`experiments/certify_structure_v2.py`, and summarized by the canonical
45-row script and table named above. The definition and intended direction are
frozen retrospectively in the spec's Measurements section.

**Limit.** This supports controlled-edit sensitivity on the development panel.
It does not establish perceived call-and-response, musical quality, or
held-out robustness. `reciprocity_hash` is a diagnostic and is not evidence
for this claim.

## Claim 2: the stagnation--alienation proxy responds to loop collapse

**Paper claim.** The structural proxy changes strongly under loop collapse and
little under timing jitter or global translation.

**Released numbers.** These are oriented `pooled_spearman_dose` values from
rows with `metric=stagnation_alienation`:

| Probe | Pooled Spearman |
|---|---:|
| `C4_loop_collapse` | -0.9235501807 |
| `C1_timing_jitter` | -0.0074545991 |
| `C2_anchor_shift` | +0.0100611964 |

The raw key is `boredom_v2_raw`, for which lower is better; the summarizer
multiplies it by -1 before computing response statistics.

**Evidence path.** `boredom_v2` in
`src/chartgeneval/metrics/structure.py` computes the union of stagnant and
non-returning active bars. The structure runner, structure JSONL, summarizer,
summary CSV, and spec provide the complete chain.

**Limit.** The archival key does not validate player boredom. The supported
claim is about the current bar-fingerprint operationalization under these
edits. Selection and evaluation used the same development charts.

## Claim 3: density--energy response is bar-scale covariation

**Paper claim.** The measurement is stable under uniform density scaling,
decreases under bar shuffling, and has only a weak response to burst insertion;
the latter failed the development gate.

**Released numbers.** These are `pooled_spearman_dose` values from rows with
`metric=density_energy_response`:

| Probe | Pooled Spearman |
|---|---:|
| `C6_density_scale` | +0.0046234626 |
| `C8_bar_shuffle` | -0.3862333468 |
| `C7_burst_insert` | -0.0922950351 |

**Evidence path.** The primary raw key `density_energy_spearman` is computed by
`density_energy_response` in `src/chartgeneval/metrics/coupling.py` from
per-bar hit density and linear mel energy. The coupling/gap runner and JSONL
provide per-variant values; the summarizer and summary CSV provide the three
reported correlations. The spec binds the authored-grid, minimum-window, and
energy-dynamic-range requirements.

**Limit.** C7 is a released negative result. This evidence supports a
bar-scale covariance reading, not a general chart--audio fit or player-quality
score. It also depends on an external cached mel representation whose complete
historical extractor environment was not preserved.

## Claim 4: run-head onset support detects constant chart--audio shift

**Paper claim.** Run-head onset support decreases as all chart events are
translated away from the audio.

**Released number.** The row with
`metric=run_head_onset_support`, `metric_key=energy_support_rate_raw`, and
`probe=C2_anchor_shift` reports
`pooled_spearman_dose=-0.5592346818`.

**Evidence path.** `energy_peak_support_rate` in
`src/chartgeneval/metrics/coupling.py` defines run heads, spectral-flux support,
the three-frame window, and the song-level threshold. The coupling/gap runner,
coupling/gap JSONL, summarizer, summary row, and spec provide the complete
released chain.

**Limit.** The result establishes sensitivity to the synthetic +15/+30/+60 ms
translations on this development panel. It does not establish that every
musically appropriate hit should coincide with a detected onset, nor does it
validate the 0.25 s grouping threshold with players.

## Claim 5: the course-specific human-chart gap is descriptive and mixed

**Paper claim.** At maximum dose, the human-chart gap worsens for C3--C8 with
net directions ranging from approximately -0.32 to -0.88. It is invariant to
C2 in median, while C5 improves at the first two doses, worsens at the maximum
dose, and differs by course.

**Released maximum-dose numbers.** `human_chart_gap` orients
`manifold_gap_raw` by -1. The summary's `improve_minus_worsen` values are:

| Probe | Net direction |
|---|---:|
| `C3_type_shuffle` | -0.8352941176 |
| `C4_loop_collapse` | -0.8117647059 |
| `C5_blandification` | -0.3176470588 |
| `C6_density_scale` | -0.8588235294 |
| `C7_burst_insert` | -0.8823529412 |
| `C8_bar_shuffle` | -0.6941176471 |

The `C2_anchor_shift` summary row reports
`median_maxdose_raw_delta=0.0` and
`improve_minus_worsen=-0.0176470588`, yielding the paper's rounded `-.02`.

**C5 dose and course audit.** The bounded `manifold_score` is monotone
decreasing in raw gap. Direct pairing in the coupling/gap JSONL gives median
score changes of +0.01735, +0.02333, and -0.04987 at doses 1, 2, and 3, with
net directions +0.49412, +0.36471, and -0.31765. At dose 3, course-level net
directions are Easy -0.50, Normal -0.80, Hard -0.25, Oni +0.15, and Ura +0.20.
This is the basis for the paper's explicit mixed-response qualification.

**Evidence path.** `phi`, `fit_manifold_ref`, and `compute` in
`src/chartgeneval/metrics/gap.py` define the current 32-dimensional
course-specific reference-cloud distance. The coupling/gap runner fits the
training references and emits `manifold_gap_raw`, `manifold_score`, and
`manifold_n_ref`; the per-course counts are checked by the release verifier.
The 45-row summary supports the maximum-dose range and C2 statement.

**Traceability qualification.** The current summarizer does not emit the
per-dose or course-specific C5 breakdown. Those values are reproducibly
pairable from the released per-variant JSONL but are not materialized as a
separate committed analysis table. They must not be promoted to a stronger
claim without adding a deterministic derived artifact.

**Limit.** Distance from the same-course training cloud is atypicality, not a
verdict that an unusual chart is poor or creative. The C5 response is not
uniformly monotone, and the metric was revised on the same development data.

## Provenance and reporting boundaries

The feature manifest binds all 40 cached mel tensors to the pinned dataset
group id, audio SHA-256, per-file mel SHA-256, byte count, shape, dtype, frame
count, and finiteness. Its `feature_set_sha256` is
`0657c1d35cc25e836e03b4dafce4370bcd6afcce960919461ab22e2cf5f4df9c`.
The audio manifest SHA-256 is
`44f6a891ffd09812e39544f3b07fd0f3652ee8a2d227a62ed398b0868ba6fa45`.
The feature extractor's complete historical dependency snapshot is missing,
and neither audio nor mel arrays are released. No end-to-end
raw-audio-to-feature byte-identity claim is made.

Historical candidate results mentioned in current metric docstrings reside in
another research repository and are not part of this release. This map relies
only on current code, the two released 4,760-row record files, and the released
45-row summary. Those historical results must not be described as
preregistered, confirmatory, or independently reproduced evidence.

Any mismatch in panel identity, hashes, row counts, summary regeneration, or
paper rounding kills the affected numerical claim until resolved. Failure on a
future untouched panel prevents generalization beyond development; failure in
a player study prevents the corresponding perceptual interpretation. The full
allowed-claim and kill boundary is normative in the spec.
