#!/usr/bin/env python3
"""Aggregate corpus-coverage statistics from the training split.

The human calibration bands are fit on the training-split difficulty-course
charts (see ``reproduce_calibration.py``). This script summarizes that same set
along two axes the source metadata does carry -- star level per course and tempo
-- and writes a derived, non-identifying aggregate record. The appendix coverage
figure then regenerates from this record without redistributing the corpus.

Note on what the source metadata does *not* carry: the parsed dataset exposes no
charter/author field and no game-version field, and its GENRE field is empty for
almost every song. Author-, version-, and genre-level concentration therefore
cannot be reported; only level and tempo coverage are recoverable.

    python experiments/build_corpus_coverage.py

Requires the ``[data]`` extra (``datasets``) and a local HF token for the
gated dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from _dataset import DEFAULT_DATASET, DEFAULT_DATASET_REVISION, build_calibration_rows
from chartgeneval.events import COURSES

# Fixed tempo bins (40-400 BPM, 20 BPM wide) safely bracket the observed range
# so the histogram is comparable across any rebuild of the record.
BPM_BIN_EDGES = list(range(40, 401, 20))


def _hist(values):
    if not values:
        return [0] * (len(BPM_BIN_EDGES) - 1)
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=BPM_BIN_EDGES)
    return [int(c) for c in counts]


def _summary(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    p10, p50, p90 = (float(x) for x in np.percentile(a, [10, 50, 90]))
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "p10": p10,
        "p50": p50,
        "p90": p90,
        "max": float(a.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1]
            / "artifacts"
            / "records"
            / "corpus_coverage.json"
        ),
    )
    args = ap.parse_args()

    rows = build_calibration_rows(
        args.dataset, args.split, args.limit, revision=args.revision
    )

    per_course_levels = defaultdict(Counter)
    per_course_bpm = defaultdict(list)
    songs = set()
    n_charts = 0
    for r in rows:
        songs.add(r["sid"])
        n_charts += 1
        level = r.get("level")
        if level is not None:
            per_course_levels[r["course"]][int(level)] += 1
        bpm = r.get("bpm")
        if bpm is not None and float(bpm) > 0:
            per_course_bpm[r["course"]].append(float(bpm))

    all_bpm = [v for vals in per_course_bpm.values() for v in vals]
    courses_out = {}
    for course in COURSES:
        level_counts = per_course_levels.get(course, Counter())
        bpm_vals = per_course_bpm.get(course, [])
        courses_out[course] = {
            "n_charts": int(sum(level_counts.values())),
            "level_counts": {str(k): int(v) for k, v in sorted(level_counts.items())},
            "bpm_hist": _hist(bpm_vals),
            "bpm_summary": _summary(bpm_vals),
        }

    payload = {
        "dataset_id": args.dataset,
        "dataset_revision": args.revision,
        "split": args.split,
        "n_songs": len(songs),
        "n_charts": n_charts,
        "bpm_bin_edges": BPM_BIN_EDGES,
        "bpm_hist_all": _hist(all_bpm),
        "bpm_summary_all": _summary(all_bpm),
        "courses": courses_out,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print("wrote", out, "| songs", len(songs), "charts", n_charts)


if __name__ == "__main__":
    main()
