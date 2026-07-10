#!/usr/bin/env python3
"""Run C1-C8 on the official test charts and score every variant.

For each official (human) test chart, applies the eight probes at three doses
(plus dose-0 = untouched) and scores every resulting chart with the calibrated
profile. Dose-0 rows are exactly the reference official rows (fidelity anchor).

Uses the bundled calibration artifact and an LM rebuilt from the training split
so scores reproduce the reference to <1e-9.

    python experiments/reproduce_probes.py --limit-songs 40 --out probes.jsonl
    python experiments/reproduce_probes.py --limit-songs 2   # fast smoke run

Requires the ``[data]`` extra and a local HF token for the gated dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import DEFAULT_DATASET, build_train_lm, iter_rows, official_charts  # noqa: E402

from chartgeneval.calibration import (  # noqa: E402
    evaluate_chart_quality,
    load_bundled_calibration,
    load_calibration,
)
from chartgeneval.probes import PROBES, corrupt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="test")
    ap.add_argument("--calibration-split", default="train")
    ap.add_argument("--calibration", default=None, help="calibration JSON (default: bundled)")
    ap.add_argument("--limit-songs", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None, help="alias for --limit-songs")
    ap.add_argument(
        "--probes",
        default=None,
        help="comma-separated probe ids to run (default: all registered probes)",
    )
    ap.add_argument("--out", default="probes.jsonl")
    args = ap.parse_args()

    limit_songs = args.limit if args.limit is not None else args.limit_songs
    probe_ids = list(PROBES)
    if args.probes:
        probe_ids = [p.strip() for p in args.probes.split(",") if p.strip()]
        unknown = [p for p in probe_ids if p not in PROBES]
        if unknown:
            raise SystemExit(f"unknown probes: {unknown}; registered: {list(PROBES)}")
    calibration = load_calibration(args.calibration) if args.calibration else load_bundled_calibration()

    lm, n_calib = build_train_lm(args.dataset, args.calibration_split)
    print(f"[lm] fitted on {n_calib} training charts", flush=True)

    rows = []
    n_songs = 0
    for sid, row in iter_rows(args.dataset, args.split, limit_songs):
        n_songs += 1
        title = (row.get("metadata") or {}).get("TITLE") or ""
        for course, chart in official_charts(row):
            bpm = chart.bpm
            base = {"sid": sid, "title": title, "course": course, "level": chart.level, "bpm": bpm}

            def emit(probe, dose_index, events, noop):
                rec = dict(base)
                rec.update({"probe": probe, "dose_index": dose_index, "corruption_noop": bool(noop)})
                rec.update(evaluate_chart_quality(events, bpm, course, lm, calibration))
                rows.append(rec)

            emit("official", 0, chart.events, False)
            for probe in probe_ids:
                for di in (1, 2, 3):
                    corrupted, noop = corrupt(
                        chart.events, probe, di, sid=sid, course=course, bpm=bpm, lm=lm
                    )
                    emit(probe, di, corrupted, noop)
        print(f"[{sid}] rows={len(rows)}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "n_songs": n_songs, "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
