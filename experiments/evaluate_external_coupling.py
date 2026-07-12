#!/usr/bin/env python3
"""Coupling-family evaluation of external-system sample dirs (audio-coupled).

For every generated chart: ``density_energy_response`` (bar-windowed rank
correlation of density vs mel energy) and ``energy_peak_support_rate``
(run-head onset-envelope support). Both read note times only, so any chart
vocabulary qualifies -- this is the second game-agnostic family of the
applicability matrix. Bar windows come from the authored segments (grid
hierarchy tier 1); mel from the features dir (same STFT/mel path as the
calibration corpus, FPS matches ``chartgeneval.metrics.coupling.FPS``).

    python experiments/evaluate_external_coupling.py \
        --features ../SoftChart/eval/features_clean \
        --systems taikonation=../SoftChart/eval/ext_taikonation_clean/samples

Requires the ``[data]`` extra and numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import iter_rows, official_charts  # noqa: E402
from _system_conditions import (  # noqa: E402
    COURSE_MAPPING_VERSION,
    DEFAULT_CONDITIONS,
    load_reference_course_maps,
    pick_reference_course,
)

from chartgeneval.events import COURSES, load_taiko_parsed_course, sorted_hits  # noqa: E402
from chartgeneval.metrics.coupling import density_energy_response, energy_peak_support_rate  # noqa: E402
from chartgeneval.metrics.timing import _segment_bars  # noqa: E402

def parse_sid_course(stem):
    parts = stem.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:2]), "_".join(parts[2:])
    return stem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="JacobLinCool/taiko-1000-parsed-clean")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--features", required=True)
    ap.add_argument("--conditions", default=str(DEFAULT_CONDITIONS))
    ap.add_argument("--systems", nargs="*", default=[], help="name=samples_dir pairs")
    ap.add_argument("--out", default="ext_coupling.jsonl")
    args = ap.parse_args()
    course_maps = load_reference_course_maps(args.conditions)

    systems = [(n, Path(d)) for n, _, d in (s.partition("=") for s in args.systems)]

    ctx_by_sid = {}
    charts_by_sid = {}
    for sid, row in iter_rows(args.dataset, args.split, args.limit):
        bars_by_course = {}
        charts = {}
        for course in COURSES:
            struct = row.get(course)
            if not struct:
                continue
            bars, _beats = _segment_bars(struct.get("segments"))
            chart = load_taiko_parsed_course(struct)
            if bars and chart is not None:
                bars_by_course[course] = (bars, chart.bpm)
                charts[course] = chart
        mel_path = Path(args.features) / f"{sid}.mel.npy"
        mel = np.load(mel_path) if mel_path.exists() else None
        ctx_by_sid[sid] = (bars_by_course, mel)
        charts_by_sid[sid] = charts
        print(f"[{sid}] courses={list(bars_by_course)}", flush=True)
    if args.limit and len(ctx_by_sid) != args.limit:
        raise RuntimeError(
            f"requested {args.limit} dataset rows, received {len(ctx_by_sid)}"
        )

    records = []

    def emit(system, sid, course, note_times):
        if sid not in ctx_by_sid:
            raise ValueError(f"sample refers to unknown song {sid!r}")
        bars_by_course, mel = ctx_by_sid[sid]
        if mel is None:
            raise FileNotFoundError(f"missing mel features for {sid}")
        if not note_times:
            raise ValueError(f"empty generated chart for {system}/{sid}/{course}")
        gcourse = pick_reference_course(system, course, bars_by_course, course_maps)
        bars, bpm = bars_by_course[gcourse]
        ctx = {"mel": mel, "grid": {"downbeats": bars}, "bpm": bpm}
        events = [(t, "don") for t in note_times]
        m = {}
        m.update(density_energy_response(events, ctx))
        m.update(energy_peak_support_rate(events, ctx))
        m = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in m.items()}
        m.update({
            "system": system,
            "sid": sid,
            "course": course,
            "grid_course": gcourse,
            "course_mapping_version": COURSE_MAPPING_VERSION,
            "n_notes": len(note_times),
        })
        records.append(m)

    for sid, charts in charts_by_sid.items():
        for course, chart in charts.items():
            emit("official", sid, course, sorted(t for t, _ in sorted_hits(chart.events)))
    print(f"[official] records={len(records)}", flush=True)

    for name, d in systems:
        n0 = len(records)
        for p in sorted(d.glob("*.json")):
            if p.name in ("run_report.json", "index.json"):
                continue
            sid, course = parse_sid_course(p.stem)
            data = json.load(open(p))
            hits = (data.get("gen") or {}).get("hits") or []
            emit(name, sid, course, sorted(float(h["t"]) for h in hits))
        print(f"[{name}] +{len(records) - n0}", flush=True)

    keys = [(r["system"], r["sid"], r["course"]) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate system/song/course rows in music-response output")

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
