#!/usr/bin/env python3
"""Score an arbitrary directory of JSON charts with the full profile.

Reads native JSON event files (see :func:`chartgeneval.events.load_events_json`)
and emits one JSONL row of raw features + calibrated scores per chart. Course and
BPM come from the file if present, else from ``--course`` / ``--bpm`` defaults.

The n-gram model needed for grammar scores is rebuilt from the dataset training
split unless ``--no-lm`` is passed (in which case grammar scores are absent).

    python experiments/evaluate_charts.py --charts my_charts/ --out scored.jsonl
    python experiments/evaluate_charts.py --charts my_charts/ --no-lm   # no dataset

Filenames may encode ``{sid}_{course}.json`` to auto-fill the course.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chartgeneval.calibration import (  # noqa: E402
    evaluate_chart_quality,
    load_bundled_calibration,
    load_calibration,
)
from chartgeneval.events import COURSES, load_events_json  # noqa: E402
from chartgeneval.metrics.common import NGramModel  # noqa: E402


def infer_course(path, fallback):
    stem = Path(path).stem
    if "_" in stem:
        cand = stem.rsplit("_", 1)[-1]
        if cand in COURSES:
            return cand
    return fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", required=True, help="directory of *.json charts")
    ap.add_argument("--limit", type=int, default=0, help="max charts (0 = all)")
    ap.add_argument("--course", default="oni", help="fallback course")
    ap.add_argument("--bpm", type=float, default=0.0, help="fallback BPM")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--dataset", default=None, help="dataset id for LM (default clean split)")
    ap.add_argument("--no-lm", action="store_true", help="skip LM (no grammar scores)")
    ap.add_argument("--out", default="charts_scored.jsonl")
    args = ap.parse_args()

    calibration = load_calibration(args.calibration) if args.calibration else load_bundled_calibration()

    if args.no_lm:
        lm = NGramModel(order=3, alpha=0.05)  # empty LM -> grammar scores None
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _dataset import DEFAULT_DATASET, build_train_lm

        lm, n = build_train_lm(args.dataset or DEFAULT_DATASET, "train")
        print(f"[lm] fitted on {n} training charts", flush=True)

    paths = sorted(Path(args.charts).glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no *.json charts in {args.charts}")

    rows = []
    for p in paths:
        chart = load_events_json(p)
        course = chart.course or infer_course(p, args.course)
        bpm = chart.bpm or args.bpm
        rec = {"path": str(p), "course": course, "bpm": bpm, "n_notes": len(chart.events)}
        rec.update(evaluate_chart_quality(chart.events, bpm, course, lm, calibration))
        rows.append(rec)
        print(f"[{p.name}] quality={rec.get('chart_quality_proxy_score')}", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "n_charts": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
