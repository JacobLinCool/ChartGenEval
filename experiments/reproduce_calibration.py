#!/usr/bin/env python3
"""Recompute the per-course calibration bands from a taiko dataset split.

Streams the training split, fits the n-gram LM, and builds the p10-p90 official
bands (course-conditioned) written by :func:`chartgeneval.calibration.build_calibration`.
The output is byte-comparable to the bundled artifact when run on the same
corpus.

Requires the ``[data]`` extra and a local HF token for the gated dataset.

    python experiments/reproduce_calibration.py --split train --out calibration.json
    python experiments/reproduce_calibration.py --limit 50   # fast smoke run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import DEFAULT_DATASET, build_calibration_rows  # noqa: E402

from chartgeneval.calibration import build_calibration  # noqa: E402
from chartgeneval.metrics.common import NGramModel, event_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0, help="max songs (0 = all)")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()

    rows = build_calibration_rows(args.dataset, args.split, args.limit)
    if not rows:
        raise SystemExit("no calibration rows collected")

    lm = NGramModel(order=args.order, alpha=args.alpha)
    for r in rows:
        lm.add(r["course"], event_tokens(r["events"], r["bpm"]))

    calibration = build_calibration(rows, lm)
    Path(args.out).write_text(json.dumps(calibration, indent=2, sort_keys=True))
    n_by_course = {c: v.get("n") for c, v in calibration["courses"].items()}
    print(json.dumps({"out": args.out, "n_charts": len(rows), "n_by_course": n_by_course}, indent=2))


if __name__ == "__main__":
    main()
