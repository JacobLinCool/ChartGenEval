# chartgeneval

**Calibrate before you rank: corruption-validated quality metrics for
rhythm-game chart generation.**

`chartgeneval` is a self-contained evaluation toolkit for automatically
generated Taiko no Tatsujin-style charts. It provides a calibrated, five-family
quality metric suite, the corruption probes (C1-C8) that *validate* those
metrics, and the audit gauntlet that turns the validation into tables and
figures. The package does **not** ship or depend on any chart-generation model:
its main evidence (degrading official human charts and watching the right score
fall) needs no model at all.

- Core install is **numpy-only**. Audio grids, figures and dataset reproduction
  live in optional extras.
- The bundled calibration artifact reproduces the reference per-chart scores to
  `< 1e-9`.
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

Python 3.10-3.12. `madmom` is an *optional* downbeat-grid backend installed
separately (`pip install madmom`); it has no wheels for all Python versions, so
it is not in an extra. When absent, `MadmomGrid` raises a clear `ImportError`.

## 30-second quickstart (no dataset needed)

```python
import numpy as np
from chartgeneval.calibration import load_bundled_calibration
from chartgeneval.metrics.common import NGramModel
from chartgeneval.metrics import evaluate_profile, compute_all_families
from chartgeneval.metrics.base import make_ctx
from chartgeneval import probes

# a tiny synthetic oni chart: alternating don/ka on every eighth note @160 BPM
bpm = 160.0
step = 60.0 / bpm / 2
events = [(i * step, "don" if i % 2 == 0 else "ka") for i in range(200)]

calibration = load_bundled_calibration()
lm = NGramModel(order=3, alpha=0.05)          # fit on your corpus; empty LM here
lm.add("oni", ["b08:DS", "b08:KS"] * 8)       # toy fit so grammar scores exist

ctx = make_ctx(course="oni", bpm=bpm, lm=lm, calibration=calibration)
scores = evaluate_profile(events, ctx)
print("chart_quality_proxy_score:", round(scores["chart_quality_proxy_score"], 4))

# apply a corruption probe deterministically and re-score
jittered, _ = probes.corrupt(events, "C1_timing_jitter", dose_index=3,
                             sid="demo", course="oni", bpm=bpm, lm=lm)
print("after C1 dose-3:", len(jittered), "events")
```

The scripts under `experiments/` all accept `--limit` for a fast smoke run.

## Metric families

The suite is organised into five families with a uniform
`compute(events, ctx) -> dict` interface. The five adopted *suite-v2* candidate
metrics (post gauntlet fix pass, 2026-07-11) are folded into their natural
family.

| Family | Module | What it scores | Key outputs | v2 candidate folded in |
|---|---|---|---|---|
| timing | `metrics.timing` | audio-anchored alignment vs an estimated beat grid | `grid_phase_offset_ms` (C2 witness), `relative_error_mean_ms`, `unsupported_rate` | — |
| coupling | `metrics.coupling` | density/strain constraints + audio coupling | `density_adequacy_score`, `strain_adequacy_score`, `density_energy_spearman`, `energy_support_rate_raw` (run-head normalized) | `density_energy_response`, `energy_peak_support_rate` |
| grammar | `metrics.grammar` | n-gram pattern grammar (course-conditioned LM) | `pattern_ic_adequacy_score`, `transition_validity_score`, `rare_ngram_score` | — |
| structure | `metrics.structure` | surface variety, repetition, boredom, colour, call-response | `surface_variety_adequacy_score`, `repetition_adequacy_score`, `reciprocity` (v2 primary), `boredom_v2_raw` | `call_response_reciprocity`, `boredom_v2` |
| gap | `metrics.gap` | distance to the official chart manifold (phi v2, 32 dims) | `manifold_gap_raw`, `manifold_score` | `official_manifold_gap` |

`metrics.constraints` re-exposes the hard playability gates (overload / spike /
chaos / boredom). `metrics.evaluate_profile` runs the full calibrated pipeline
(the golden path).

### Pluggable grid sources

The timing and audio-coupled families need a beat grid, decoupled from any
generation model (`chartgeneval.grid`):

- `MetadataGrid` -- constant-BPM 4/4 from metadata BPM + duration (default,
  zero deps).
- `LibrosaGrid` -- `librosa.beat.beat_track` (`[audio]` extra).
- `MadmomGrid` -- madmom DBN downbeat tracker (optional; graceful `ImportError`).
- `ExternalJSONGrid` -- downbeats/bar from any external tracker's JSON.

## Corruption probes (C1-C8)

`chartgeneval.probes` implements eight construction-guaranteed degradations of a
chart's event stream at three doses each, seeded deterministically from
`(sid, course, probe, dose_index)`.

| Probe | Operator | Target dimension |
|---|---|---|
| C1 | timing jitter (+-10/20/30 ms) | timing / rhythm complexity |
| C2 | anchor shift (+15/30/60 ms) | anchoring (chart-only blind control) |
| C3 | colour shuffle (don<->ka, p up) | transition validity / pattern IC |
| C4 | loop collapse (most-common 4-cycle) | repetition / variety / boredom |
| C5 | blandification (LM argmax resample) | pattern IC |
| C6 | density scale (x0.5/1.5/2) | density / strain |
| C7 | burst insert (dense clusters) | overload / spike / chaos |
| C8 | bar shuffle (destroy global form) | structure / repetition |

## Audit gauntlet

`chartgeneval.audit` computes the paper's diagnostic tables from a scored probe
table: `dose_response`, `coupling_matrix` (the honest "specificity is coupling"
matrix), `length_orthogonality`, `baseline_reversal`, and `separation_auc`.

## Reproducing the paper

> **Note on datasets and frozen numbers.** The bundled calibration artifact and
> all reference (golden) numbers were frozen on `JacobLinCool/taiko-1000-parsed`
> (the original corpus); to reproduce them byte-exactly pass
> `--dataset JacobLinCool/taiko-1000-parsed`. The default dataset is the
> go-forward clean split (`taiko-1000-parsed-clean`), which yields a different,
> self-consistent set of numbers. Also note the LM/calibration build streams the
> full train split (~9 min) regardless of `--limit-songs`, and running the test
> suite requires the `[dev]` extra (`pip install -e ".[dev]"` before `pytest`).


The dataset is gated on the Hugging Face Hub. The paper uses the cleaned split
`JacobLinCool/taiko-1000-parsed-clean`; a local HF token is required. Set
`--dataset` to switch corpora.

```bash
# (a) recompute the per-course calibration bands from the training split
python experiments/reproduce_calibration.py --split train --out calibration.json

# (b) run C1-C8 on the 167 official test charts, score with both metric families
python experiments/reproduce_probes.py --limit-songs 40 --out probes.jsonl

# (c) score an arbitrary directory of JSON charts with the full profile
python experiments/evaluate_charts.py --charts my_charts/ --out charts_scored.jsonl

# (d) render the three paper figures (dose-response, coupling matrix, reversal)
python experiments/make_figures.py --probes probes.jsonl --out figures/
```

Each script accepts `--limit` / `--limit-songs` for a fast smoke run that needs
only a handful of dataset rows.

### Fidelity to the reference

With the bundled calibration artifact and an LM rebuilt from the same training
split, `evaluate_profile` reproduces the reference official per-chart scores to
`< 1e-9`. `experiments/reproduce_probes.py`'s dose-0 rows are exactly the
reference official rows, which is the fidelity anchor the tests check.

## Data requirements (honest notes)

- **Not ported:** TJA / osu chart parsing (`load_events_tja` / `load_events_osu`
  are documented stubs). Convert charts to the native JSON event format first.
- **Audio-coupled metrics** (`density_energy_response`, `energy_peak_support_rate`)
  require a mel spectrogram in `ctx["mel"]` (128 x T log-power @ 86.13 fps). The
  reference pipeline read this from an internal cache; here you must supply it,
  e.g. via `librosa` from the audio in the dataset.
- **The manifold gap** needs a fitted per-course reference
  (`gap.fit_manifold_ref`) built from a corpus of official charts; without it,
  only the `phi_*` vector is returned.

## License

MIT (placeholder). **Pending author confirmation** of the final license and of
the redistribution terms for the bundled calibration artifact (which contains
aggregate quantile statistics derived from the gated dataset, not raw charts).
See `LICENSE`.
