#!/usr/bin/env python3
"""Render the three paper figures from a scored probe table.

Reads the JSONL produced by ``reproduce_probes.py`` and draws:

  1. dose_response.png  -- target-score dose curves for each probe.
  2. coupling_matrix.png -- probe x metric standardized-delta heatmap (the honest
                            "specificity is coupling" figure).
  3. baseline_reversal.png -- baseline foils improving under corruption (C4
                            structureness/self-BLEU proxies, C5 perplexity).

Requires the ``[plots]`` extra (matplotlib).

    python experiments/make_figures.py --probes probes.jsonl --out figures/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from chartgeneval.audit import coupling_matrix, dose_response
from chartgeneval.probes import PROBE_ORDER

# Target calibrated score(s) each probe attacks (paper Table).
PROBE_TARGETS = {
    "C1_timing_jitter": "rhythm_complexity_adequacy_score",
    "C2_anchor_shift": "grid_phase_offset_abs_ms",
    "C3_type_shuffle": "transition_validity_score",
    "C4_loop_collapse": "repetition_adequacy_score",
    "C5_blandification": "repetition_adequacy_score",
    "C6_density_scale": "density_adequacy_score",
    "C7_burst_insert": "overload_score",
    "C8_bar_shuffle": "surface_structure_proxy_score",
}

MATRIX_METRICS = [
    "density_adequacy_score",
    "strain_adequacy_score",
    "rhythm_complexity_adequacy_score",
    "color_switch_adequacy_score",
    "surface_variety_adequacy_score",
    "repetition_adequacy_score",
    "pattern_ic_adequacy_score",
    "transition_validity_score",
    "overload_score",
    "density_spike_score",
    "pattern_chaos_score",
    "boredom_score",
]

REVERSAL_METRICS = ["pattern_ngram_perplexity", "unique_4gram_rate", "repeat_4gram_rate"]


def load_rows(path):
    return [json.loads(l) for l in open(path)]


def fig_dose_response(rows, out):
    import matplotlib.pyplot as plt

    dr = {(d["probe"], d["metric"]): d for d in dose_response(rows, list(PROBE_TARGETS.values()))}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for probe in PROBE_ORDER:
        metric = PROBE_TARGETS[probe]
        d = dr.get((probe, metric))
        if not d:
            continue
        ys = [d.get(f"mean_dose{i}") for i in range(4)]
        if any(y is None for y in ys):
            continue
        ys = np.asarray(ys, dtype=float)
        y0 = ys[0] if abs(ys[0]) > 1e-9 else 1.0
        ax.plot(range(4), ys / y0, marker="o", label=f"{probe}: {metric}")
    ax.set_xlabel("dose index")
    ax.set_ylabel("target metric (normalized to dose 0)")
    ax.set_title("Dose-response: target metric falls as corruption strengthens")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_coupling_matrix(rows, out):
    import matplotlib.pyplot as plt

    cm = {(c["probe"], c["metric"]): c.get("std_delta_maxdose") for c in coupling_matrix(rows, MATRIX_METRICS)}
    M = np.full((len(PROBE_ORDER), len(MATRIX_METRICS)), np.nan)
    for i, probe in enumerate(PROBE_ORDER):
        for j, metric in enumerate(MATRIX_METRICS):
            v = cm.get((probe, metric))
            if v is not None:
                M[i, j] = v
    fig, ax = plt.subplots(figsize=(9, 4.5))
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    im = ax.imshow(M, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(MATRIX_METRICS)))
    ax.set_xticklabels(MATRIX_METRICS, rotation=90, fontsize=6)
    ax.set_yticks(range(len(PROBE_ORDER)))
    ax.set_yticklabels(PROBE_ORDER, fontsize=7)
    ax.set_title("Coupling matrix: standardized score change at max dose")
    fig.colorbar(im, ax=ax, shrink=0.8, label="std delta (official SD)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_baseline_reversal(rows, out):
    import matplotlib.pyplot as plt

    # C4 -> repetition/variety baselines; C5 -> perplexity.
    dr = {(d["probe"], d["metric"]): d for d in dose_response(rows, REVERSAL_METRICS)}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    panels = [
        ("C4_loop_collapse", "repeat_4gram_rate"),
        ("C4_loop_collapse", "unique_4gram_rate"),
        ("C5_blandification", "pattern_ngram_perplexity"),
    ]
    for probe, metric in panels:
        d = dr.get((probe, metric))
        if not d:
            continue
        ys = [d.get(f"mean_dose{i}") for i in range(4)]
        if any(y is None for y in ys):
            continue
        ys = np.asarray(ys, dtype=float)
        y0 = ys[0] if abs(ys[0]) > 1e-9 else 1.0
        ax.plot(range(4), ys / y0, marker="s", label=f"{probe}: {metric}")
    ax.set_xlabel("dose index")
    ax.set_ylabel("baseline metric (normalized to dose 0)")
    ax.set_title("Baseline reversal: naive metrics move the 'wrong' way under corruption")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True, help="probes.jsonl from reproduce_probes.py")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()

    rows = load_rows(args.probes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dose_response(rows, out_dir / "dose_response.png")
    fig_coupling_matrix(rows, out_dir / "coupling_matrix.png")
    fig_baseline_reversal(rows, out_dir / "baseline_reversal.png")
    print(json.dumps({"out_dir": str(out_dir), "figures": 3, "n_rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
