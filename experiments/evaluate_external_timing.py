#!/usr/bin/env python3
"""Timing-family evaluation of external-system sample dirs over the grid hierarchy.

For every generated chart, computes the full timing family against BOTH grid
reference tiers:

  * authored  -- per-course TJA segments from the dataset row (per-bar meter),
                 selected by the system's explicit reference-course map
  * estimated -- beat-tracker downbeats from the features dir (``<sid>.beats.npz``)

Official charts are evaluated as the ``official`` reference system. Sample
dirs follow the SoftChart sample format (``{gen: {hits: [{t, type}]}}``,
filenames ``<sid>_<course>.json`` or ``<sid>.json``). Chart vocabularies other
than taiko are fine here: the timing family only reads note times (this is the
game-agnostic subset that powers the applicability matrix).

    python experiments/evaluate_external_timing.py \
        --features ../SoftChart/eval/features_clean \
        --systems taikonation=../SoftChart/eval/ext_taikonation_clean/samples \
        --out ext_timing.jsonl

Requires the ``[data]`` extra (dataset streaming), scipy, and numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import iter_rows  # noqa: E402
from _system_conditions import (  # noqa: E402
    COURSE_MAPPING_VERSION,
    DEFAULT_CONDITIONS,
    load_reference_course_maps,
    pick_reference_course,
)

from chartgeneval.events import COURSES, load_taiko_parsed_course, sorted_hits  # noqa: E402
from chartgeneval.metrics import timing  # noqa: E402

def load_grid_context(row):
    """Per-course authored segments + official bpm for one dataset row."""
    out = {}
    for course in COURSES:
        struct = row.get(course)
        if not struct:
            continue
        segments = struct.get("segments")
        chart = load_taiko_parsed_course(struct)
        if segments and chart is not None:
            out[course] = {"segments": segments, "bpm": chart.bpm, "chart": chart}
    return out


def est_grid(features_dir, sid):
    path = Path(features_dir) / f"{sid}.beats.npz"
    if not path.exists():
        return None
    z = np.load(path)
    db = np.asarray(z.get("downbeats"), dtype=float)
    beats = np.asarray(z.get("beats"), dtype=float)
    if db is None or len(db) < 2:
        return None
    bpm = 60.0 / float(np.median(np.diff(beats))) if len(beats) >= 2 else None
    return {"downbeats": [float(t) for t in db], "bar": None, "bpm": bpm}


def sample_files(samples_dir):
    for p in sorted(Path(samples_dir).glob("*.json")):
        if p.name in ("run_report.json", "index.json"):
            continue
        yield p


def parse_sid_course(stem):
    parts = stem.split("_")
    # sid = test_00000 (two tokens); optional trailing course token(s)
    if len(parts) >= 3:
        return "_".join(parts[:2]), "_".join(parts[2:])
    return stem, None


def eval_one(note_times, grid, bpm, duration):
    ctx = {"grid": grid, "bpm": bpm, "duration": duration}
    return timing.compute([(t, "don") for t in note_times], ctx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="JacobLinCool/taiko-1000-parsed-clean")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--features", required=True)
    ap.add_argument("--conditions", default=str(DEFAULT_CONDITIONS))
    ap.add_argument("--systems", nargs="*", default=[], help="name=samples_dir pairs")
    ap.add_argument("--skip-official", action="store_true")
    ap.add_argument("--out", default="ext_timing.jsonl")
    args = ap.parse_args()
    course_maps = load_reference_course_maps(args.conditions)

    systems = []
    for spec in args.systems:
        name, _, d = spec.partition("=")
        systems.append((name, Path(d)))

    print(f"[rows] streaming {args.dataset} {args.split} limit={args.limit}", flush=True)
    grid_by_sid = {}
    for sid, row in iter_rows(args.dataset, args.split, args.limit):
        grid_by_sid[sid] = load_grid_context(row)
    if args.limit and len(grid_by_sid) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} dataset rows, received {len(grid_by_sid)}"
        )

    records = []

    def emit(system, sid, course, note_times, grid_ctx):
        est = est_grid(args.features, sid)
        if not note_times:
            raise ValueError(f"empty generated chart for {system}/{sid}/{course}")
        if est is None:
            raise FileNotFoundError(f"missing or invalid estimated grid for {sid}")
        gcourse = pick_reference_course(system, course, grid_ctx, course_maps)
        g = grid_ctx[gcourse]
        duration = max(note_times) + 1.0
        for src, grid, bpm in (
            ("metadata", {"segments": g["segments"]}, g["bpm"]),
            ("estimated", est, (est or {}).get("bpm")),
        ):
            m = eval_one(note_times, grid, bpm, duration)
            m.update({
                "system": system, "sid": sid, "course": course,
                "grid_course": gcourse, "anchor_source": src,
                "course_mapping_version": COURSE_MAPPING_VERSION,
            })
            records.append(m)

    if not args.skip_official:
        for sid, gctx in grid_by_sid.items():
            for course, g in gctx.items():
                times = sorted(t for t, _ in sorted_hits(g["chart"].events))
                emit("official", sid, course, times, gctx)
        print(f"[official] records={len(records)}", flush=True)

    for name, d in systems:
        n0 = len(records)
        for p in sample_files(d):
            sid, course = parse_sid_course(p.stem)
            gctx = grid_by_sid.get(sid)
            if not gctx:
                raise ValueError(f"sample {p} refers to unknown song {sid!r}")
            with p.open() as stream:
                data = json.load(stream)
            hits = (data.get("gen") or {}).get("hits") or []
            times = sorted(float(h["t"]) for h in hits)
            emit(name, sid, course, times, gctx)
        print(f"[{name}] +{len(records) - n0} records", flush=True)

    keys = [
        (r["system"], r["sid"], r["course"], r["anchor_source"])
        for r in records
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate system/song/course/reference rows in timing output")

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
