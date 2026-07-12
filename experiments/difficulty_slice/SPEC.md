# Difficulty-slice analysis spec

## Research question / claim

How does a generator's fit to human chart ranges change with the requested
difficulty? The reportable claim is deliberately narrow: for
Mapperatorinator on the held-out songs, departures in note rate, short-interval
load, and local note patterns are concentrated in Easy through Hard, while Oni
and Ura are closer to their course-specific human ranges. Timing placement is
reported separately and is not expected to improve monotonically with course.

This is an exploratory, post-hoc slice requested after the primary evaluation.
It does not support a general claim that generators perform better at high
difficulty, nor a ranking across generators.

## Experimental unit

One generated chart paired with the human reference chart for the same song and
requested course.

## Systems / conditions compared

- Mapperatorinator v32, because it produced a chart for each available Taiko
  course and can be paired exactly with the corresponding reference chart.
- Human reference charts provide note-rate ratios and course-specific
  calibration ranges; they are not a competing system.

Other evaluated systems are excluded from the difficulty panel: TaikoNation,
DDC onset, and AutoOsu expose one setting. Gen\'eLive exposes five settings; its
timing and music-response records have been rerun with the explicit crosswalk
in `artifacts/systems/conditions.json`. We still exclude it here because that
cross-game crosswalk is an evaluation convention, whereas Mapperatorinator's
native settings are the same five Taiko courses as the human references.

## Corpus / panel

- Held-out song panel used by the paper.
- Easy, Normal, Hard, and Oni: 40 paired charts per course.
- Ura: 10 paired charts; figures and captions must show this smaller sample.
- Authored-grid timing rows only (`anchor_source=metadata`).

## Primary measurements

- Generated-to-reference note-rate ratio.
- Course-calibrated note-rate fit (`density_adequacy_score`).
- Course-calibrated short-interval-load fit (`strain_adequacy_score`).
- Familiar-transition fit (`transition_validity_score`).
- Local-pattern fit (`pattern_ic_adequacy_score`).
- Repetition and variety fits.
- Fraction of notes within 6 ms of the authored timing grid (`clean_rate`).

The note-rate ratio and timing panels report medians and p10--p90 intervals.
The six course-range fit columns report medians.

## Support and failure rules

The narrow mismatch claim is supported when Easy--Hard show lower note-rate,
short-interval, and local-pattern fit, together with larger note-rate ratios,
than Oni on the paired panel. The claim is weakened if the deterministic
analysis does not reproduce this ordering or if the paired song/course joins
are incomplete. Timing is interpreted on its own; no monotone timing trend is
required or claimed.

## Inputs and row grain

- `artifacts/records/system_profile_mapperatorinator.jsonl`: one row
  per generated song/course chart.
- `artifacts/records/corruption_probes_development.jsonl`: human-reference rows,
  one per song/course chart when `probe=official`.
- `artifacts/records/system_timing_clean.jsonl`: one row per
  system/song/course/reference source.

Stable join key: `(sid, course)`. No generated chart, source audio, or note
array is read by the analysis.

## Analysis and outputs

Script and command:

```bash
python experiments/make_paper_figures.py --only figE
```

Planned outputs:

- `paper/figs/figE_difficulty.pdf`
- `artifacts/tables/mapperatorinator_difficulty_summary.csv`

The figure script must regenerate both outputs from the released numerical
records. The summary table is the canonical source for prose values.

## Known limitations

- This slice covers one difficulty-conditioned Taiko generator.
- Ura has 10 charts rather than 40.
- The analysis was specified after inspecting the primary records, so it is
  labeled exploratory.
- Course-fit scores measure distance from human ranges, not player preference
  or overall chart quality.

## Revision log

- 2026-07-12, v3: switched the chart-side inputs to the pinned-revision,
  symmetric-half-width development and system-profile artifacts. The archived
  `*_clean_v2.jsonl` paths are no longer analysis inputs.
- 2026-07-11, v2: updated after the Gen\'eLive timing and music-response records
  were rerun with the explicit course crosswalk. The analysis remains limited
  to Mapperatorinator for the comparability reason stated above.
- 2026-07-11, v1: created after exploratory inspection of the released records
  in response to an author request for difficulty-stratified analysis.
