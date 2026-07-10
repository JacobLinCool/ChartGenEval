"""Certify tail-sensitive timing fields against C1 (dense jitter) and C1s
(sparse severe outliers) on official charts, over authored + estimated grids.

For each official test chart, applies C1 and C1s at three doses and computes
the full timing family on both grid sources. Dose-0 rows are the untouched
officials. Prints a dose-response table (cross-chart median per field) plus
paired per-chart direction consistency, the acceptance evidence for the
tail statistics (violation rates, P99/max, salience) and the expected
near-blindness of mean-based summaries under C1s.

Grid input: the grids JSON artifact with per-sid ``est_downbeats`` /
``meta_downbeats`` (authored timing metadata, not note positions).

Requires the ``[data]`` extra and a local HF token for the gated dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _dataset import iter_rows, official_charts  # noqa: E402

from chartgeneval.metrics import timing  # noqa: E402
from chartgeneval.probes import corrupt  # noqa: E402

PROBES = ["C1_timing_jitter", "C1s_sparse_jitter"]

TABLE_FIELDS = [
    "clean_rate",
    "absolute_violation_rate",
    "absolute_violation_rate_2x",
    "absolute_violation_rate_3x",
    "n_absolute_violation",
    "absolute_error_mean_ms",
    "absolute_error_p99_ms",
    "relative_violation_rate",
    "relative_violation_rate_2x",
    "relative_violation_rate_3x",
    "unsupported_rate",
    "anchor_salience_mean",
    "signed_offset_median_ms",
]


def build_sources(g):
    sources = {}
    meta = g.get("meta_downbeats")
    if meta and len(meta) >= 2:
        sources["metadata"] = {"downbeats": meta, "bar": None, "bpm": None}
    if g.get("est_ok") and g.get("est_downbeats") and len(g["est_downbeats"]) >= 2:
        sources["estimated"] = {
            "downbeats": g["est_downbeats"],
            "bar": None,
            "bpm": g.get("est_bpm"),
        }
    return sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="JacobLinCool/taiko-1000-parsed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--grids", required=True, help="grids JSON artifact path")
    ap.add_argument("--out", default="certify_timing_tail_records.jsonl")
    args = ap.parse_args()

    payload = json.load(open(args.grids))
    grids = payload.get("grids", payload)

    records = []
    n_charts = 0
    for sid, row in iter_rows(args.dataset, args.split, args.limit):
        g = grids.get(sid)
        if not g:
            continue
        duration = g.get("audio_duration_s")
        sources = build_sources(g)
        if not sources:
            continue
        for course, chart in official_charts(row):
            variants = [("official", 0, chart.events, False)]
            for probe in PROBES:
                for dose in (1, 2, 3):
                    ev, noop = corrupt(
                        chart.events, probe, dose, sid=sid, course=course, bpm=chart.bpm
                    )
                    variants.append((probe, dose, ev, noop))
            for src, grid in sources.items():
                ctx = {"grid": grid, "bpm": chart.bpm, "duration": duration}
                for tid, dose, ev, noop in variants:
                    m = timing.compute(ev, ctx)
                    m.update(
                        {
                            "sid": sid,
                            "course": course,
                            "target_id": tid,
                            "dose_index": dose,
                            "anchor_source": src,
                            "noop": bool(noop),
                        }
                    )
                    records.append(m)
            n_charts += 1
        print(f"[{sid}] cumulative charts={n_charts}", flush=True)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} records -> {args.out}", flush=True)

    ok = [r for r in records if not r["noop"]]
    by = defaultdict(list)
    for r in ok:
        by[(r["anchor_source"], r["target_id"], r["dose_index"], r["sid"], r["course"])].append(r)

    def chart_vals(src, tid, dose, field):
        out = {}
        for r in ok:
            if r["anchor_source"] == src and r["target_id"] == tid and r["dose_index"] == dose:
                v = r.get(field)
                if v is not None:
                    out[(r["sid"], r["course"])] = v
        return out

    for src in ["metadata", "estimated"]:
        for probe in PROBES:
            print(f"\n===== {probe} @ {src} (cross-chart median; paired frac(dose>dose0)) =====")
            base = {f: chart_vals(src, "official", 0, f) for f in TABLE_FIELDS}
            header = f"{'field':<28}" + "".join(f"{'d'+str(d):>11}" for d in range(4))
            header += "   " + "".join(f"{'dir'+str(d):>7}" for d in (1, 2, 3))
            print(header)
            for f in TABLE_FIELDS:
                meds, dirs = [], []
                for d in range(4):
                    tid = "official" if d == 0 else probe
                    cv = chart_vals(src, tid, d, f)
                    meds.append(np.median(list(cv.values())) if cv else float("nan"))
                    if d > 0:
                        common = set(cv) & set(base[f])
                        dirs.append(
                            np.mean([cv[k] > base[f][k] for k in common]) if common else float("nan")
                        )
                line = f"{f:<28}" + "".join(f"{m:>11.4f}" for m in meds)
                line += "   " + "".join(f"{x:>7.2f}" for x in dirs)
                print(line)


if __name__ == "__main__":
    main()
