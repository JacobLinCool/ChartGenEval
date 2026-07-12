#!/usr/bin/env python3
"""Recompute the per-course calibration bands from a taiko dataset split.

Streams the training split, fits the n-gram LM, and builds the p10-p90 official
bands (course-conditioned) written by :func:`chartgeneval.calibration.build_calibration`.
The output uses the same versioned schema as the bundled artifact when run on
the same corpus.

Requires the ``[data]`` extra and a local HF token for the gated dataset.

    python experiments/reproduce_calibration.py --split train --out calibration.json
    python experiments/reproduce_calibration.py --limit 50   # fast smoke run
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
    build_calibration_rows,
)

from chartgeneval.calibration import build_calibration  # noqa: E402
from chartgeneval.metrics.common import NGramModel, event_tokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    ap.add_argument("--split", choices=("train",), default="train")
    ap.add_argument("--limit", type=int, default=0, help="max songs (0 = all)")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", default="calibration.json")
    args = ap.parse_args()
    if args.order < 2:
        raise SystemExit("--order must be at least 2")
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        raise SystemExit("--alpha must be finite and positive")

    rows = build_calibration_rows(
        args.dataset, args.split, args.limit, revision=args.revision
    )
    if not rows:
        raise SystemExit("no calibration rows collected")

    lm = NGramModel(order=args.order, alpha=args.alpha)
    for r in rows:
        lm.add(r["course"], event_tokens(r["events"], r["bpm"]))

    calibration = {
        "artifact_version": 1,
        "dataset_id": args.dataset,
        "dataset_revision": args.revision,
        "split": args.split,
        "n_charts": len(rows),
        "lm_order": args.order,
        "lm_alpha": args.alpha,
        **build_calibration(rows, lm),
    }
    with Path(args.out).open("x", encoding="utf-8") as stream:
        json.dump(calibration, stream, indent=2, sort_keys=True)
        stream.write("\n")
    n_by_course = {c: v.get("n") for c, v in calibration["courses"].items()}
    print(json.dumps({"out": args.out, "n_charts": len(rows), "n_by_course": n_by_course}, indent=2))


if __name__ == "__main__":
    main()
