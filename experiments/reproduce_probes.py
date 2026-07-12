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
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    build_train_lm,
    iter_canonical_rows,
    official_charts,
)

from chartgeneval.calibration import (  # noqa: E402
    calibration_content_sha256,
    evaluate_chart_quality,
    load_bundled_calibration,
    load_calibration,
    require_calibration_provenance,
)
from chartgeneval.probes import PROBES, corrupt  # noqa: E402


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    ap.add_argument("--split", choices=("test",), default="test")
    ap.add_argument("--calibration-split", choices=("train",), default="train")
    ap.add_argument("--calibration", default=None, help="calibration JSON (default: bundled)")
    ap.add_argument("--limit-songs", type=int, default=40)
    ap.add_argument(
        "--probes",
        default=None,
        help="comma-separated probe ids to run (default: all registered probes)",
    )
    ap.add_argument("--out", default="probes.jsonl")
    args = ap.parse_args()

    limit_songs = args.limit_songs
    if not 1 <= limit_songs <= 40:
        raise SystemExit("--limit-songs must stay within the development panel [1, 40]")
    probe_ids = list(PROBES)
    if args.probes:
        probe_ids = [p.strip() for p in args.probes.split(",") if p.strip()]
        unknown = [p for p in probe_ids if p not in PROBES]
        if unknown:
            raise SystemExit(f"unknown probes: {unknown}; registered: {list(PROBES)}")
    calibration = load_calibration(args.calibration) if args.calibration else load_bundled_calibration()
    require_calibration_provenance(
        calibration,
        dataset_id=args.dataset,
        dataset_revision=args.revision,
        split=args.calibration_split,
    )

    lm, n_calib = build_train_lm(
        args.dataset,
        args.calibration_split,
        order=calibration["lm_order"],
        alpha=calibration["lm_alpha"],
        revision=args.revision,
    )
    print(f"[lm] fitted on {n_calib} training charts", flush=True)
    provenance = {
        "lm_dataset_id": args.dataset,
        "lm_dataset_revision": args.revision,
        "lm_split": args.calibration_split,
        "lm_order": calibration["lm_order"],
        "lm_alpha": calibration["lm_alpha"],
        "calibration_content_sha256": calibration_content_sha256(calibration),
    }

    rows = []
    n_songs = 0
    for sid, row in iter_canonical_rows(
        args.dataset, args.split, limit_songs, revision=args.revision
    ):
        n_songs += 1
        title = (row.get("metadata") or {}).get("TITLE") or ""
        for course, chart in official_charts(row):
            bpm = chart.bpm
            if not math.isfinite(bpm) or bpm <= 0:
                raise ValueError(f"{sid}/{course}: a finite positive BPM is required")
            base = {
                "sid": sid,
                "title": title,
                "course": course,
                "level": chart.level,
                "bpm": bpm,
                **provenance,
            }

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

    with open(args.out, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "n_songs": n_songs, "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
