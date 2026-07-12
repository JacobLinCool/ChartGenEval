# Confirmatory holdout v1 run book

The full panel has not been run when this workflow is introduced. Do not place
hand-computed values in this file. Every command below writes a manifest with
the exact invocation and input hashes.

## Freeze after implementation is final

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/freeze.py \
  --config experiments/confirmatory_holdout_v1/configs/confirmatory.json \
  --out experiments/confirmatory_holdout_v1/frozen/confirmatory_freeze.json
```

`freeze.py` uses exclusive creation and cannot update an existing snapshot. If
the snapshot becomes stale before any confirmatory run, create a new experiment
id; do not overwrite it.

## Synthetic smoke

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/run.py \
  --config experiments/confirmatory_holdout_v1/configs/smoke_synthetic.json \
  --run-id smoke-YYYYMMDDTHHMMSSZ

.venv/bin/python experiments/confirmatory_holdout_v1/transform.py \
  --run-dir experiments/confirmatory_holdout_v1/runs/raw/smoke-YYYYMMDDTHHMMSSZ

.venv/bin/python experiments/confirmatory_holdout_v1/analyze.py \
  --table-dir experiments/confirmatory_holdout_v1/runs/raw/smoke-YYYYMMDDTHHMMSSZ/tables
```

Synthetic verdicts are implementation checks only and cannot support paper
claims. The analysis labels every non-confirmatory verdict `SMOKE_ONLY`.

Before creating the freeze, run the same end-to-end chain on the first two
stored-order rows remaining after `canonical=True` filtering. The scanner stops
immediately after canonical index 1 and never reaches the held-out canonical
index 40. This validates the real dataset schema, split-manifest identity, authored
segment wiring, calibration, and full-train LM path without touching the
confirmatory panel:

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/run.py \
  --config experiments/confirmatory_holdout_v1/configs/smoke_development.json \
  --run-id smoke-development-YYYYMMDDTHHMMSSZ

.venv/bin/python experiments/confirmatory_holdout_v1/transform.py \
  --run-dir experiments/confirmatory_holdout_v1/runs/raw/smoke-development-YYYYMMDDTHHMMSSZ

.venv/bin/python experiments/confirmatory_holdout_v1/analyze.py \
  --table-dir experiments/confirmatory_holdout_v1/runs/raw/smoke-development-YYYYMMDDTHHMMSSZ/tables
```

## Untouched confirmatory panel

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/run.py \
  --config experiments/confirmatory_holdout_v1/configs/confirmatory.json \
  --freeze experiments/confirmatory_holdout_v1/frozen/confirmatory_freeze.json \
  --confirm-untouched-panel \
  --run-id confirmatory-YYYYMMDDTHHMMSSZ
```

The runner refuses a confirmatory config without both the freeze snapshot and
the explicit panel acknowledgement. It scans all 140 raw test rows, validates
their identity multiset against the frozen split manifest, filters 20 aliases,
and accepts only canonical indices `[40, 120)` with 80 unique group/audio IDs.
`CONFIRMATORY_ANCHOR.json` binds this command's run id as `primary_run_id`
before any held-out row is read. No second confirmatory collection—including a
repeat of the same run id—is allowed under this experiment id. A fatal first
attempt therefore requires a new experiment id **and a genuinely untouched
panel** for another prospective gate; reusing canonical `[40, 120)` would be recovery or
replication only. There is no predeclared resume path.

Then transform and analyze the selected run:

```bash
.venv/bin/python experiments/confirmatory_holdout_v1/transform.py \
  --run-dir experiments/confirmatory_holdout_v1/runs/raw/confirmatory-YYYYMMDDTHHMMSSZ

.venv/bin/python experiments/confirmatory_holdout_v1/analyze.py \
  --table-dir experiments/confirmatory_holdout_v1/runs/raw/confirmatory-YYYYMMDDTHHMMSSZ/tables
```

Generated evidence remains under the selected run directory:

```text
records.jsonl                  append-only raw attempts
source_scan.jsonl              append-only raw stored-order rows, including aliases
sampling_context.jsonl         append-only canonical-only context projection
manifest_events.jsonl          append-only run-start/run-end events
MANIFEST.json                  sealed run manifest
tables/                        deterministic canonical layer
tables/reports/                regenerable analysis and claim-evidence outputs
```
