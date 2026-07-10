#!/usr/bin/env python3
"""Corruption certification for the coupling and gap families (clean test set).

Applies all registered corruption operators at three doses to the official
test charts and computes (a) the coupling family against the song's mel
spectrogram and authored bar lines, and (b) the manifold-gap score against a
reference fitted on the training corpus. Closes the audit gap for the two
families whose inputs (audio; feature manifold) the chart-only probe run
cannot cover.

    python experiments/certify_coupling_gap.py \
        --features ../SoftChart/eval/features_clean --out coupling_gap.jsonl

Requires the ``[data]`` extra, numpy, and the features dir (mel npy files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import iter_rows, official_charts  # noqa: E402

from chartgeneval.metrics import coupling, gap  # noqa: E402
from chartgeneval.metrics.timing import _segment_bars  # noqa: E402
from chartgeneval.probes import PROBES, corrupt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="JacobLinCool/taiko-1000-parsed-clean")
    ap.add_argument("--split", default="test")
    ap.add_argument("--calibration-split", default="train")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", default="coupling_gap_probes.jsonl")
    args = ap.parse_args()

    # Manifold reference + LM (for C5) from the training corpus.
    from _dataset import build_train_lm
    print("[ref] fitting manifold reference + LM on train", flush=True)
    lm, _ = build_train_lm(args.dataset, args.calibration_split)
    ref_rows = []
    for sid, row in iter_rows(args.dataset, args.calibration_split, 0):
        for course, chart in official_charts(row):
            ref_rows.append({"course": course, "events": chart.events, "bpm": chart.bpm})
    phis = [v for v in (gap.phi(r["events"], r["bpm"]) for r in ref_rows) if v is not None]
    manifold_ref = gap.fit_manifold_ref(phis)
    print(f"[ref] fitted on {len(ref_rows)} charts", flush=True)

    records = []
    for sid, row in iter_rows(args.dataset, args.split, args.limit):
        mel_path = Path(args.features) / f"{sid}.mel.npy"
        mel = np.load(mel_path) if mel_path.exists() else None
        for course, chart in official_charts(row):
            bars, _beats = _segment_bars(row[course].get("segments"))
            ctx = {
                "mel": mel,
                "grid": {"downbeats": bars} if bars else None,
                "bpm": chart.bpm,
                "course": course,
                "official_manifold_ref": manifold_ref,
            }

            def emit(probe, dose, events, noop):
                m = {}
                m.update(coupling.density_energy_response(events, ctx))
                m.update(coupling.energy_peak_support_rate(events, ctx))
                try:
                    m.update(gap.compute(events, ctx))
                except Exception:
                    pass
                m = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                     for k, v in m.items()}
                m.update({"sid": sid, "course": course, "probe": probe,
                          "dose_index": dose, "corruption_noop": bool(noop)})
                records.append(m)

            emit("official", 0, chart.events, False)
            for probe in PROBES:
                for di in (1, 2, 3):
                    ev, noop = corrupt(chart.events, probe, di, sid=sid,
                                       course=course, bpm=chart.bpm, lm=lm)
                    emit(probe, di, ev, noop)
        print(f"[{sid}] records={len(records)}", flush=True)

    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
