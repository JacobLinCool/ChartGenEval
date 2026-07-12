#!/usr/bin/env python3
"""Score an arbitrary directory of JSON charts with the full profile.

Reads native JSON event files (see :func:`chartgeneval.events.load_events_json`)
and emits one JSONL row of raw features + calibrated scores per chart. Course and
BPM come from the file if present, else from ``--course`` / ``--bpm`` defaults.

The n-gram model needed for grammar scores is rebuilt from one immutable
dataset revision.  The calibration must declare the same dataset, revision,
split, n-gram order, and smoothing constant or the run aborts.

    python experiments/evaluate_charts.py \
      --charts my_charts/ --system my_system --out scored.jsonl

Filenames may encode ``{sid}_{course}.json`` to auto-fill the course.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    build_train_lm,
)

from chartgeneval.calibration import (  # noqa: E402
    calibration_content_sha256,
    evaluate_chart_quality,
    load_bundled_calibration,
    load_calibration,
    require_calibration_provenance,
)
from chartgeneval.events import COURSES, load_events_json  # noqa: E402


def chart_identity(path, fallback):
    """Parse a collision-free chart id and optional course suffix."""
    stem = Path(path).stem
    suffix = stem.rsplit("_", 1)[-1] if "_" in stem else None
    course = suffix if suffix in COURSES else fallback
    sid = stem[: -(len(suffix) + 1)] if suffix in COURSES else stem
    if not sid or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", sid):
        raise ValueError(f"unsupported chart filename identity: {Path(path).name!r}")
    return sid, course


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--charts", required=True, help="directory of *.json charts")
    ap.add_argument("--limit", type=int, default=0, help="max charts (0 = all)")
    ap.add_argument("--course", choices=COURSES, default="oni", help="fallback course")
    ap.add_argument("--bpm", type=float, default=0.0, help="fallback BPM")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset id for LM")
    ap.add_argument(
        "--revision",
        default=DEFAULT_DATASET_REVISION,
        help="immutable 40-character dataset commit used to fit the LM",
    )
    ap.add_argument(
        "--system",
        required=True,
        help="system identifier recorded on every output row",
    )
    ap.add_argument(
        "--include-source-paths",
        action="store_true",
        help="include local chart paths for private debugging (release default omits them)",
    )
    ap.add_argument(
        "--bpm-map",
        default=None,
        help="JSON file {sid: bpm} used when a chart file carries no BPM "
        "(e.g. the song's authored BPM). IOI tokenization needs a real BPM; "
        "a zero fallback silently poisons every grammar score.",
    )
    ap.add_argument("--out", default="charts_scored.jsonl")
    args = ap.parse_args()
    if args.limit < 0:
        raise SystemExit("--limit must be non-negative")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.system):
        raise SystemExit("--system must match [a-z0-9][a-z0-9_-]*")

    bpm_map = {}
    if args.bpm_map:
        with open(args.bpm_map, encoding="utf-8") as stream:
            bpm_map = json.load(stream)

    calibration = load_calibration(args.calibration) if args.calibration else load_bundled_calibration()
    require_calibration_provenance(
        calibration,
        dataset_id=args.dataset,
        dataset_revision=args.revision,
        split="train",
    )
    lm, n = build_train_lm(
        args.dataset,
        "train",
        order=calibration["lm_order"],
        alpha=calibration["lm_alpha"],
        revision=args.revision,
    )
    print(f"[lm] fitted on {n} training charts", flush=True)
    provenance = {
        "lm_dataset_id": args.dataset,
        "lm_dataset_revision": args.revision,
        "lm_split": "train",
        "lm_order": calibration["lm_order"],
        "lm_alpha": calibration["lm_alpha"],
        "calibration_content_sha256": calibration_content_sha256(calibration),
    }

    paths = sorted(Path(args.charts).glob("*.json"))
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        raise SystemExit(f"no *.json charts in {args.charts}")

    rows = []
    seen_identities = set()
    for p in paths:
        chart = load_events_json(p)
        parsed_sid, parsed_course = chart_identity(p, args.course)
        sid = parsed_sid
        course = chart.course or parsed_course
        identity = (sid, course)
        if identity in seen_identities:
            raise RuntimeError(f"duplicate output chart identity: {identity}")
        seen_identities.add(identity)
        bpm = chart.bpm or bpm_map.get(sid) or args.bpm
        if not isinstance(bpm, (int, float)) or not math.isfinite(bpm) or bpm <= 0:
            raise ValueError(f"{p.name}: a finite positive BPM is required")
        rec = {
            "sid": sid,
            "system": args.system,
            "course": course,
            "bpm": bpm,
            "n_notes": len(chart.events),
            **provenance,
        }
        if args.include_source_paths:
            rec["path"] = str(p)
        rec.update(evaluate_chart_quality(chart.events, bpm, course, lm, calibration))
        rows.append(rec)
        available = sum(
            value is not None
            for key, value in rec.items()
            if key.endswith("_score")
        )
        print(f"[{p.name}] diagnostic_scores={available}", flush=True)

    with open(args.out, "x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "n_charts": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
