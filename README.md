# chartgeneval

**Multi-dimensional rhythm-game chart metrics, tested with controlled
damage.**

`chartgeneval` is a self-contained evaluation toolkit for automatically
generated Taiko no Tatsujin-style charts. It provides the timing,
audio-response, structure, and human-gap measurements; difficulty-conditioned
adaptations of established chart statistics; and nine corruption probes (C1,
C1s, and C2-C8). The sealed confirmation covers seven outputs across nine
nonredundant probe--measurement tests. Five suite-v2 components have descriptive
development evidence and are labeled accordingly in the paper and artifacts.
The package does **not** ship or depend on any chart-generation model:
its corruption-sensitivity evidence is obtained by degrading human charts and
checking each declared measurement response.

- Core install is **numpy-only**. Audio grids, figures and dataset reproduction
  live in optional extras.
- The bundled calibration is the versioned 3,880-chart training artifact used
  by the paper's clean-split experiments.
- Every corruption probe is seed-deterministic: two runs give bit-identical
  output.

## Install

```bash
pip install -e .              # core (numpy only)
pip install -e ".[scipy]"     # + optimal-assignment timing matcher, Spearman p-values
pip install -e ".[audio]"     # + librosa grid + mel loading
pip install -e ".[plots]"     # + matplotlib for make_figures.py
pip install -e ".[data]"      # + datasets for the reproduction scripts
pip install -e ".[dev]"       # everything (incl. pytest)
```

Python 3.10-3.14. `madmom` is an *optional* downbeat-grid backend installed
separately (`pip install madmom`); it has no wheels for all Python versions, so
it is not in an extra. When absent, `MadmomGrid` raises a clear `ImportError`.

## 30-second raw-feature quickstart (no dataset needed)

```python
from chartgeneval.metrics.common import compute_chart_features
from chartgeneval import probes

# a tiny synthetic oni chart: alternating don/ka on every eighth note @160 BPM
bpm = 160.0
step = 60.0 / bpm / 2
events = [(i * step, "don" if i % 2 == 0 else "ka") for i in range(200)]

features = compute_chart_features(events, bpm)
for key in ("density_nps", "local_nps_p95", "ioi_entropy_bits",
            "repeat_4gram_rate", "unique_4gram_rate"):
    print(key, features[key])

# apply a corruption probe deterministically and re-score
jittered, _ = probes.corrupt(events, "C1_timing_jitter", dose_index=3,
                             sid="demo", course="oni", bpm=bpm)
print("after C1 dose-3:", len(jittered), "events")
```

Calibrated grammar diagnostics require an LM fitted on the exact corpus,
revision, order, and smoothing constant declared by the calibration artifact.
An empty or toy LM is not a valid fallback. Use the pinned reproduction scripts
below for the paper contract.

## Metric families

The suite is organised into five families with a uniform
`compute(events, ctx) -> dict` interface. Five proposed *suite-v2*
components are folded into their natural family; their empirical evidence is
development-only.

| Family | Module | What it scores | Key outputs | v2 candidate folded in |
|---|---|---|---|---|
| timing | `metrics.timing` | alignment to an authored or external musical-time grid | `grid_phase_offset_abs_ms` (C2 witness), `absolute_error_p99_ms`, `clean_rate`, `unsupported_rate` | — |
| coupling | `metrics.coupling` | density/strain constraints + audio coupling | `density_adequacy_score`, `strain_adequacy_score`, `density_energy_spearman`, `energy_support_rate_raw` (run-head normalized) | `density_energy_response`, `energy_peak_support_rate` |
| grammar | `metrics.grammar` | n-gram pattern grammar (course-conditioned LM) | `pattern_ic_adequacy_score`, `transition_validity_score` (interpreted as transition familiarity), `rare_ngram_score` | — |
| structure | `metrics.structure` | 4-gram repetition/uniqueness, stagnation--alienation, colour, call-response | `repetition_adequacy_score` (the algebraically equivalent `surface_variety_adequacy_score` is retained as a frozen field), `reciprocity` (v2 primary), `boredom_v2_raw` | `call_response_reciprocity`, `boredom_v2` |
| gap | `metrics.gap` | distance to a course-specific human reference cloud (phi v2, 32 dims) | `manifold_gap_raw`, `manifold_score` | `official_manifold_gap` |

`metrics.constraints` re-exposes the one-sided overload, spike, and chaos
checks plus the archival single-sided boredom heuristic. The proposed
`boredom_v2_raw` is reported more narrowly as a stagnation--alienation
structural proxy. `metrics.evaluate_profile` returns these calibrated leaves and
all namespaced family diagnostics. It intentionally does not average the axes
into a universal chart-quality total; ranking requires a declared task-specific
decision rule.

Timing assignment and timing error are distinct: candidate matching gates are
10–50 ms, whereas the absolute-error reporting band is 6 ms. The archival
`absolute_error_p99_ms` field reports the within-chart p99 of
`max(0, abs(offset) - 6 ms)`, called **excess-error p99** in the paper.
`absolute_offset_abs_p99_ms` retains the raw absolute-offset p99.
Relative excess error subtracts `max(6 ms, 0.025 * anchor_interval)` from
the absolute change between neighboring signed offsets. These are
psychophysics-motivated operating thresholds, not universal gameplay
inaudibility guarantees or event-matching tolerances.

### Pluggable grid sources

The timing and audio-coupled families need a beat grid, decoupled from any
generation model (`chartgeneval.grid`). Use the song's authored per-bar timing
metadata when it is available; use an audio-estimated grid for a new song.
Never evaluate a generator against a grid inferred from its own output.

- `MetadataGrid` -- a synthetic or explicitly phase-zero constant-tempo map.
  It is useful for tests or audio known to begin on a downbeat; it is not the
  paper's BPM-only fallback, because BPM by itself does not identify phase.
- `LibrosaGrid` -- `librosa.beat.beat_track` (`[audio]` extra).
- `MadmomGrid` -- madmom DBN downbeat tracker (optional; graceful `ImportError`).
- `ExternalJSONGrid` -- downbeats/bar from any external tracker's JSON.

## Corruption probes (C1, C1s, C2-C8)

`chartgeneval.probes` implements nine targeted edits of a
chart's event stream at three doses each, seeded deterministically from
`(sid, course, probe, dose_index)`.

| Probe | Operator | Target dimension |
|---|---|---|
| C1 | timing jitter (+-10/20/30 ms) | notes within 6 ms of the authored grid |
| C1s | sparse outliers (nominal 0.5/1/2%, actual count `max(1, round(n*p))`, +-60 ms) | timing-tail / grammar complementarity |
| C2 | global timing shift (+15/30/60 ms) | chart--audio alignment |
| C3 | colour shuffle (don<->ka, p up) | transition familiarity |
| C4 | loop collapse (most-common 4-cycle) | one 4-gram repetition/uniqueness axis + stagnation--alienation |
| C5 | blandification (LM argmax resample) | one effective 4-gram repetition/uniqueness test; the two frozen co-primary names are algebraically equivalent |
| C6 | density scale (x0.5/1.5/2) | density / strain |
| C7 | burst insert (dense clusters) | overload / spike / chaos |
| C8 | fixed-window shuffle | local repetition / window-boundary transitions |

C8 permutes selected occupied windows of duration `4 * 60 / BPM` seconds,
starting at the first note. Fixed windows deliberately require no authored
bar boundaries; the four-beat nominal scale does not imply alignment to
musical bars or phrases. The frozen identifier `C8_bar_shuffle` remains the
experiment/seed key; the operator and archived numerical results are unchanged.

For every chart, `repeat_4gram_rate = 1 - unique_4gram_rate`. Their mirrored
course bands and symmetric scoring therefore make `repetition_adequacy_score`
and `surface_variety_adequacy_score` identical up to floating-point roundoff.
They must not be counted as independent evidence. The frozen
`surface_structure_proxy_score` also includes this same axis twice and is kept
only as an archival field, not an inferential summary. See
[`artifacts/C5_DEPENDENCY_AUDIT.md`](artifacts/C5_DEPENDENCY_AUDIT.md).

## Audit gauntlet

`chartgeneval.audit` computes the paper's diagnostic tables from a scored probe
table: `dose_response`, `coupling_matrix` (the honest "specificity is coupling"
matrix), `length_orthogonality`, `baseline_reversal`, and `separation_auc`.

## Released artifact

The `artifacts/` directory contains the final clean-split calibration
provenance, grouped song split, dose-level and per-chart records, external-system
conditions, the sealed confirmatory summaries, and a checksum manifest. The records contain metric values and
provenance only: no audio, note-event arrays, third-party code, checkpoints, or
generated chart data. See `artifacts/README.md` for schemas and joins, and run:

```bash
python experiments/verify_release_artifacts.py
```

The paper source and compiled manuscript are under `paper/`.
The September 2026 terminology and disclosure revision applies to the English
manuscript; the Traditional Chinese companion has not yet been synchronized.
The confirmatory release supports summary-level auditability, but the exact
dirty source snapshot at freeze time and all raw confirmatory records are
not packaged. Current code and hash manifests alone do not provide independent
end-to-end reproduction of that frozen run.

## Reproducing the paper

> **Corpus not distributed.** The charts and audio are potentially
> copyrighted community and commercial content, so the corpus is not released
> with this project. The bundled calibration and all released paper records
> were computed from one frozen corpus snapshot; its identity and hash are
> pinned in `artifacts/confirmatory_holdout_v1/freeze.json`. Rebuilding the
> full training LM and calibration streams all 3,880 training charts and takes
> about nine minutes. The test suite requires the `[dev]` extra
> (`pip install -e ".[dev]"` before `pytest`).

The released records, calibration values, and figures can be rebuilt without
the corpus (step d below). The corpus-dependent steps take a dataset revision;
holders of an equivalent corpus substitute their own snapshot for
`<REVISION>`.

```bash
# (a) recompute the per-course calibration bands from the training split
python experiments/reproduce_calibration.py \
  --revision <REVISION> \
  --split train --out calibration.json

# (b) run all nine probes on the 170 official test charts
python experiments/reproduce_probes.py \
  --revision <REVISION> \
  --calibration calibration.json --limit-songs 40 --out probes.jsonl

# (c) score an arbitrary directory of JSON charts with the full profile
python experiments/evaluate_charts.py \
  --charts my_charts/ --system my_system --calibration calibration.json \
  --out charts_scored.jsonl

# (d) rebuild all paper figures from the released records
python experiments/make_paper_figures.py

# (e) bind the exact cached mel tensors to dataset/audio identities
python experiments/build_coupling_gap_feature_manifest.py \
  --revision <REVISION> \
  --features ../SoftChart/eval/features_clean \
  --audio-manifest ../SoftChart/eval/ext_provenance_clean/audio_manifest_clean.json \
  --extractor-script ../SoftChart/eval/ext_provenance_clean/features_clean.py \
  --preprocess-source ../SoftChart/src/softchart/preprocess.py \
  --vocab-source ../SoftChart/src/softchart/vocab.py \
  --out coupling_gap_feature_manifest.json

# (f) rebuild the development-only reciprocity and structure records
python experiments/certify_structure_v2.py \
  --revision <REVISION> \
  --out structure_v2_development.jsonl \
  --run-manifest structure_run_manifest.json

# (g) rebuild audio-response and per-course human-gap records
python experiments/certify_coupling_gap.py \
  --revision <REVISION> \
  --features ../SoftChart/eval/features_clean \
  --audio-manifest ../SoftChart/eval/ext_provenance_clean/audio_manifest_clean.json \
  --feature-manifest coupling_gap_feature_manifest.json \
  --out coupling_gap_development.jsonl \
  --run-manifest coupling_gap_run_manifest.json

# (h) summarize the five development-only proposed measurements
python experiments/summarize_suite_v2_development.py \
  --structure structure_v2_development.jsonl \
  --coupling-gap coupling_gap_development.jsonl \
  --out suite_v2_development_results.csv \
  --manifest suite_v2_summary_manifest.json
```

The full retrospective development contract, including evidence limits and
exact commands, is in
[`experiments/suite_v2_development/SPEC.md`](experiments/suite_v2_development/SPEC.md).

`reproduce_probes.py` filters aliases before assigning canonical indices and
hard-stops within development indices 0--39; the untouched 40--119 panel is
handled only by the frozen confirmatory workflow. Release chart scoring requires
a finite positive BPM and omits local source paths by default.

### Fidelity to the reference

With the bundled calibration artifact and an LM rebuilt from the same training
split, the unprefixed calibrated diagnostics use the same canonical symmetric
half-width contract as the reproduction scripts. Runs are deterministic when
the dataset revision, calibration artifact, LM fit, and source revision are
held fixed.

## Data requirements (honest notes)

- **Not ported:** TJA / osu chart parsing (`load_events_tja` / `load_events_osu`
  are documented stubs). Convert charts to the native JSON event format first.
- **Audio-coupled metrics** (`density_energy_response`, `energy_peak_support_rate`)
  require a mel spectrogram in `ctx["mel"]` (128 x T log-power @ 86.13 fps). The
  paper experiment uses the exact cached tensors bound by the released feature
  manifest. The extraction contract is recorded, but the historical dependency
  snapshot is unavailable; the release therefore does not claim byte-identical
  re-extraction from raw audio.
- **Distance from human charts** needs a fitted per-course reference
  (`gap.fit_manifold_ref(vectors, course=...)`) built separately from official
  charts for each course. References are course-tagged and rejected on mismatch;
  without one, only the `phi_*` vector is returned.

## License

The original ChartGenEval source code is licensed under the MIT License; see
`LICENSE`. The bundled calibration and released numerical records are derived
from the research/non-commercial dataset and are outside the MIT grant; see
`ARTIFACT_LICENSE.md`. Third-party LaTeX files retain their own terms as listed
in `THIRD_PARTY_NOTICES.md`.
