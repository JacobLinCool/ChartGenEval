# Post hoc held-out baselines

This folder holds a **post hoc secondary analysis** that was not prespecified.
It applies the nine ChartGenEval tests (C1, C1s, C2-C8 with their frozen
doses, seeds, matched controls, and decision rule) to five common statistics
on the same 80 held-out Taiko songs used by the confirmatory run
`confirmatory-primary-20260712`. The held-out results of that run were already
known when this analysis was written, so it does not share the confirmatory
status of the nine primary tests. It is the source of Table 2 in the ICASSP 2027
paper (`paper/icassp2027/`).

The five statistics and the direction read as "worse":

| Statistic | Definition | Worse |
|---|---|---|
| LM perplexity | `exp(pattern_nll)` under the frozen trigram LM | increase |
| Self-similarity | long-band structureness indicator (lags 12-32, 4-token windows) | decrease (ambiguous; see `SPEC_posthoc_baselines.md`) |
| IOI-histogram JS | Jensen-Shannon divergence (bits) from the unedited chart's IOI histogram | increase |
| Onset F1 @ 50 ms | maximum-matching onset F1 against the unedited chart (`mir_eval` 0.8.2) | decrease |
| Onset F1 @ 20 ms | same, 20 ms window | decrease |

Seven further statistics (distinct-2/4, type-token ratio, IOI entropy and CV,
empty-beat rate, groove consistency) were computed as well; they are reported
descriptively in `dropped_baselines_check.json` and `distinct_ci.json`.

## Files

| File | Contents |
|---|---|
| `SPEC_posthoc_baselines.md` | Analysis specification: statistics, tests, controls, labels, validation gate. |
| `verdict_matrix.json` | Per statistic x test label (`DETECTS`, `BLIND`, `REWARDS`, `N/A-by-design`) with means, 99.5% intervals, and controls. Byte-identical to `paper/icassp2027/tables/baselines_verdict_matrix.json`, from which Table 2 is built. |
| `baseline_final_results.json` | Full per-cell results, label rules, input hashes, and the check that all ten frozen primary rows are rebuilt exactly. |
| `distinct_ci.json` | Distinct-4 responses with song-bootstrap 99.5% intervals. |
| `crosslm_c5_heldout.json` | Common-pattern rewrite (C5) scored by the frozen LM and three separately fitted LMs. |
| `dropped_baselines_check.json` | Descriptive responses of the seven further statistics. |
| `*.py` | The scripts that produced the JSON files above (copies of the versions in the archive). |
| `MANIFEST_final.json` | SHA-256 of the final analysis files as run. |
| `cge_baselines_records.tar.gz` | Working tree of the analysis: scripts, per-task metric records, logs, manifests (layout below). |

The archive `cge_baselines_records.tar.gz` unpacks to `cge_baselines/`:

- `baseline_holdout.py`, `full/shard_*.jsonl`: variant regeneration and the
  per-task metric records (50,283 tasks in four shards); `MANIFEST.json`
  lists their hashes.
- `final/`: onset-F1 maximum-matching shards (`final/shards/f1mm_*.jsonl`),
  `analyze_final.py`, `make_tex.py`, their logs and outputs, and
  `MANIFEST_final.json`.
- `verify/`: independent checks (record regeneration, statistics, split and
  dose checks, cross-LM C5, distinct-4 intervals) with `VERIFY_MANIFEST.json`.
- `analyze_baselines.py`, `check_c2_ulp.py`: the superseded greedy-matching
  analysis and the floating-point check described below.

Records contain task identifiers, song indices, SHA-256 hashes of each edited
event stream, and metric values. They contain no note events, chart text,
or audio. Two manuscript-drafting files listed in `final/MANIFEST_final.json`
(`PROVENANCE_new_sentences.md`, `results_text_draft.tex`) and a LaTeX layout
check directory are not included; they are neither inputs nor outputs of the
analysis.

## Whole-chart shift under chart-only statistics

In `verdict_matrix.json` and `baseline_final_results.json`, the cells
`lm_perplexity|C2_anchor_shift` and `self_similarity|C2_anchor_shift` report
`dose_statistic_mean` values of -0.0165 and -0.0077. Both statistics are
**exactly unchanged** by the whole-chart shift: every replicate at every dose
equals the unedited chart's value bit for bit
(`fraction_replicates_equal_unedited_by_dose` = 1.0; `check_c2_ulp.out` in the
archive reports a maximum replicate-level difference of 0.0), and the contrast
with the matched control is exactly 0. The nonzero dose correlations come from
floating-point summation noise in the five-seed mean (relative differences
below 1.4e-16), not from any response. The cells are labeled `BLIND` with
`report_as: "exactly unchanged at every seed and strength"`, and Table 2 reports
them that way.

## Reproducing

The analysis reads the frozen confirmatory harness in
`experiments/confirmatory_holdout_v1/` (contract, probes, seeds, LM, and
`analyze.py`, imported read-only), the sealed raw records of
`confirmatory-primary-20260712` (`runs/raw/confirmatory-primary-20260712/`),
and the `JacobLinCool/taiko-1000-parsed-clean` dataset from the Hugging Face
cache. The sealed raw records are not part of this release, so the release
supports checking the summary files; end-to-end regeneration additionally
needs those records. Environment used: Python 3.11 with numpy 2.2.6,
scipy 1.17.1, datasets 5.0.0, and mir_eval 0.8.2.

1. Unpack the archive and work inside `cge_baselines/`.
2. The scripts inside the archive are kept byte-identical to the versions
   that were run, so the archived manifests' hashes still match; they contain
   absolute paths from the original machine (the repository checkout, a
   sibling source checkout used only to record hashes of the copied baseline
   functions, and a temporary unpack directory). Point these constants at
   your checkout before rerunning. The top-level copies of
   `analyze_final.py`, `crosslm_c5_heldout.py`, and `distinct_ci.py` in this
   folder resolve paths relative to the repository instead; their hashes in
   `MANIFEST_final.json` are for these edited copies.
3. Run, with `PYTHONDONTWRITEBYTECODE=1`:
   - `python baseline_holdout.py --start A --stop B --out full/shard_A_B.jsonl`
     for `A` in 40, 60, 80, 100 (`B = A + 20`); aborts if any regenerated
     variant differs from the sealed records.
   - `python final/f1_maxmatch.py --start A --stop B --out final/shards/f1mm_A_B.jsonl`
     for the same ranges.
   - `python final/analyze_final.py`, then `python3 final/make_tex.py`.
   - Optional checks: the scripts in `verify/`.
4. Compare outputs against `final/MANIFEST_final.json` and
   `verify/VERIFY_MANIFEST.json`.
