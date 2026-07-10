"""Standalone fidelity check (NOT a pytest test; needs the gated dataset).

Rebuilds the full-train LM from taiko-1000-parsed, scores the first N official
test charts with chartgeneval.evaluate_chart_quality using the bundled
calibration, and compares every shared numeric field against the golden
chart_quality_matrix official rows. Passes iff max abs diff < 1e-9.

Run:
    .venv/bin/python tests/_fidelity_check.py --golden <path> --n 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from _dataset import build_train_lm, iter_rows, official_charts  # noqa: E402

from chartgeneval.calibration import evaluate_chart_quality, load_bundled_calibration  # noqa: E402


def fnum(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="JacobLinCool/taiko-1000-parsed")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--n", type=int, default=10, help="number of test songs")
    args = ap.parse_args()

    golden = {}
    for line in open(args.golden):
        r = json.loads(line)
        if r.get("system_id") == "official" and r.get("status") == "ok":
            golden[(r["sid"], r["course"])] = r
    print(f"golden official rows: {len(golden)}")

    calibration = load_bundled_calibration()
    lm, n_calib = build_train_lm(args.dataset, "train")
    print(f"LM fitted on {n_calib} training charts")

    max_diff = 0.0
    n_values = 0
    n_charts = 0
    worst = None
    for sid, row in iter_rows(args.dataset, "test", args.n):
        for course, chart in official_charts(row):
            key = (sid, course)
            g = golden.get(key)
            if g is None:
                continue
            n_charts += 1
            scored = evaluate_chart_quality(chart.events, chart.bpm, course, lm, calibration)
            for k, v in scored.items():
                a, b = fnum(v), fnum(g.get(k))
                if a is None or b is None:
                    continue
                d = abs(a - b)
                n_values += 1
                if d > max_diff:
                    max_diff = d
                    worst = (key, k, a, b)
    print(json.dumps({
        "n_charts": n_charts,
        "n_values": n_values,
        "max_abs_diff": max_diff,
        "worst": worst,
        "PASS": max_diff < 1e-9,
    }, indent=2))
    sys.exit(0 if max_diff < 1e-9 else 1)


if __name__ == "__main__":
    main()
