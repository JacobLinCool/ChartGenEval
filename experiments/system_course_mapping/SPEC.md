# External-system reference-course mapping correction

## Research question / claim

Do the reported timing and music-response measurements use the declared
reference course for every system setting? This correction replaces an
implicit ``same name, otherwise densest course'' fallback with an explicit,
versioned crosswalk.

## Experimental units

- Timing: one system output for one song and native difficulty setting, with one
  row against the declared metadata grid and one against the audio-estimated
  grid.
- Music response: the same output with one row using the declared course's bar
  windows and the song's audio features.

## Systems and crosswalk

The canonical crosswalk is `artifacts/systems/conditions.json` under
`configuration.reference_course_map`. Ordered candidate lists are intentional:
Gen\'eLive Challenge uses Ura when the song provides it and Oni otherwise.
The file also declares `reference_course_mapping_version`; the runners reject a
version mismatch. Undeclared systems, settings, or unavailable candidates are
errors.

## Inputs

- The first 40 songs in the frozen test-split order used by the paper.
- Frozen external-system sample directories and audio-feature cache from the
  original evaluation.
- `experiments/evaluate_external_timing.py`.
- `experiments/evaluate_external_coupling.py`.

The generated charts and audio features are execution inputs; the released
outputs are derived numerical records.

## Execution commands

```bash
python experiments/evaluate_external_timing.py \
  --features <research-repo>/eval/features_clean \
  --systems \
    taikonation=<research-repo>/eval/ext_taikonation_clean/samples \
    mapperatorinator=<research-repo>/eval/ext_mapperatorinator_clean/samples \
    ddconset=<research-repo>/eval/ext_ddconset_clean/samples \
    genelive=<research-repo>/eval/ext_genelive_clean/samples \
    autoosu=<research-repo>/eval/ext_autoosu_clean/samples \
  --out artifacts/records/system_timing_clean.jsonl

python experiments/evaluate_external_coupling.py \
  --features <research-repo>/eval/features_clean \
  --systems <same name=directory pairs> \
  --out artifacts/records/system_coupling_clean.jsonl
```

## Expected outputs and checks

- Timing: 1,320 rows; music response: 660 rows.
- Every row records `course_mapping_version=system_reference_course_v1`.
- Gen\'eLive maps Beginner/Easy/Medium/Hard to Easy/Normal/Hard/Oni.
- Gen\'eLive Challenge maps to Ura for 10 songs and Oni for 30 songs.
- The artifact manifest fixes row counts and SHA-256 hashes.

## Known limitations

The crosswalk is an evaluation convention, not an assertion that difficulty
scales are perceptually identical across games. Course-stratified claims are
therefore limited to Mapperatorinator, whose native settings are the five Taiko
courses.

The release manifest freezes the derived records, but the research execution
inputs do not yet have a published aggregate hash manifest. Any future record
replacement must first pin the dataset revision and the sample and feature
hashes; the current records remain the canonical output until then.

## Revision log

- 2026-07-11, v1: created after detecting that the earlier run silently used
  the densest available course for settings without an exact name match. The
  corrected records replace that canonical analysis output; the frozen system
  samples remain unchanged.
