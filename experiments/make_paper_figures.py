#!/usr/bin/env python3
"""Render the evaluation-paper figures into ``paper/figs/``.

Figure set (visual protocol -- see ``chartgeneval.plots``):

  fig1_reductio.pdf        C4/C5: baselines improve while calibrated scores hold.
  fig2_anchor_flip.pdf     C2: chart-only + matching offset blind; fixed-grid
                           phase offset recovers the injected shift.
  figA_complementarity.pdf The bidirectional blindness exhibit: C5 fools the
                           grammar baseline (band holds); C1s is absorbed by
                           the timing lattice (grammar catches). Paired
                           per-chart net direction on both panels.
  figB_timing_tiers.pdf    Perceptual-bucket stacked bars (clean / 1-2x / 2-3x
                           / >3x deadzone / unsupported) + per-chart clean-rate
                           strip. DRAFT: uses official + corrupted variants as
                           known-damage rulers until the system re-evaluation
                           on the clean split lands.
  figC_coupling.pdf        Probe x calibrated-score coupling heatmap with the
                           C1s row merged in (its grammar response is
                           discovered coupling -- no designated target box).

Inputs are the SoftChart experiment records/reports (paths via
``--softchart-root``); they will ship as the paper's records artifact.

    python experiments/make_paper_figures.py --softchart-root ../SoftChart

Requires the ``[plots]`` extra (matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chartgeneval.audit import coupling_matrix
from chartgeneval.plots import BASE_C, OURS_C, net_direction, plot_dose_response, plot_timing_tiers, timing_buckets

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "lines.linewidth": 1.4,
                     "lines.markersize": 4})


def rows(path):
    return list(csv.DictReader(open(path)))


def jrows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


# ---------------- fig1: reductio (ported unchanged) ----------------


def fig1(cp, out):
    rev = rows(cp / "baseline_reversal.csv")
    dr = rows(cp / "dose_response.csv")

    def dose_means(probe, metric, table):
        for r in table:
            if r["probe"] == probe and r["metric"] == metric:
                return [float(r[f"dose{i}"]) if f"dose{i}" in r else float(r[f"mean_dose{i}"])
                        for i in range(4)]
        raise KeyError((probe, metric))

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3))
    doses = [0, 1, 2, 3]

    ax = axes[0]
    for metric, label, color, ls in [
        ("structureness_indicator_long", "structureness (baseline)", BASE_C, "-"),
        ("self_bleu", "Self-BLEU (baseline)", BASE_C, "--"),
    ]:
        v = dose_means("C4_loop_collapse", metric, rev)
        ax.plot(doses, [x / v[0] for x in v], ls, color=color, marker="o", label=label)
    for metric, label, ls in [
        ("repetition_adequacy_score", "repetition adequacy (ours)", "-"),
        ("surface_variety_adequacy_score", "surface variety (ours)", "--"),
    ]:
        v = dose_means("C4_loop_collapse", metric, dr)
        ax.plot(doses, [x / v[0] for x in v], ls, color=OURS_C, marker="s", label=label)
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_title("C4 loop-collapse (charts made repetitive)")
    ax.set_xlabel("corruption dose")
    ax.set_ylabel("value relative to intact chart")
    ax.set_xticks(doses)
    ax.annotate("baselines say\n“better”", xy=(3, 1.4), ha="center", color=BASE_C, fontsize=7)
    ax.legend(loc="center left", frameon=False)

    ax = axes[1]
    v = dose_means("C5_blandification", "pattern_ngram_perplexity", rev)
    ax.plot(doses, [x / v[0] for x in v], "-", color=BASE_C, marker="o",
            label="perplexity (baseline, “lower=better”)")
    v = dose_means("C5_blandification", "pattern_ic_adequacy_score", rev)
    ax.plot(doses, [x / v[0] for x in v], "-", color=OURS_C, marker="s",
            label="pattern-IC band score (ours)")
    v = dose_means("C5_blandification", "repetition_adequacy_score", dr)
    ax.plot(doses, [x / v[0] for x in v], "--", color=OURS_C, marker="s",
            label="repetition adequacy (ours)")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax.set_title("C5 blandification (LM-argmax rewrite)")
    ax.set_xlabel("corruption dose")
    ax.set_xticks(doses)
    ax.annotate("perplexity says\n“better”", xy=(3, 0.76), ha="center", color=BASE_C, fontsize=7)
    ax.legend(loc="lower left", frameon=False)
    save(fig, out, "fig1_reductio")


# ---------------- fig2: C2 flip (ported unchanged) ----------------


def fig2(ti, out):
    c2 = [r for r in rows(ti / "c2_flip.csv") if r["anchor_source"] == "estimated_est_ok"]
    shift = [float(r["shift_ms"]) for r in c2]
    grid_delta = [float(r["grid_phase_offset_delta_vs_dose0_ms"]) for r in c2]
    match = [float(r["signed_offset_match_mean_ms"]) for r in c2]
    score = [float(r["score_grid_offset_mean"]) for r in c2]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(6.9, 2.2))
    ax.plot([0, 60], [0, 60], ":", color="gray", lw=0.8, label="ideal (y = x)")
    ax.plot(shift, grid_delta, "-", color=OURS_C, marker="s",
            label="fixed-grid phase offset (ours, audio-anchored)")
    ax.plot(shift, [m - match[0] for m in match], "-", color=BASE_C, marker="o",
            label="matching-based signed offset (blind)")
    ax.annotate("all 35 chart-only metrics:\nmax |Δ| = 0.0004 SD (blind)",
                xy=(38, 8), fontsize=7, color="dimgray")
    ax.set_xlabel("injected global shift (ms)")
    ax.set_ylabel("measured offset change (ms)")
    ax.set_title("C2 anchor shift: who can see it?")
    ax.legend(loc="upper left", frameon=False)

    ax2.plot(shift, score, "-", color=OURS_C, marker="s")
    ax2.set_xlabel("injected global shift (ms)")
    ax2.set_ylabel("calibrated anchor score")
    ax2.set_title("calibrated score response")
    ax2.set_ylim(0, 0.6)
    save(fig, out, "fig2_anchor_flip")


# ---------------- figA: bidirectional blindness ----------------


def figA(probe_records, cert_records, c1s_records, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.4), sharey=True)

    plot_dose_response(
        probe_records,
        ["pattern_ngram_perplexity", "pattern_ic_adequacy_score"],
        probe="C5_blandification",
        styles={
            "pattern_ngram_perplexity": {
                "label": "n-gram perplexity (baseline) — falls: “better”",
                "color": BASE_C, "marker": "o"},
            "pattern_ic_adequacy_score": {
                "label": "pattern-IC band score (ours) — holds",
                "color": OURS_C},
        },
        ax=axes[0],
        title="C5 blandification: grammar baseline fooled",
    )
    axes[0].annotate("fooled", xy=(2.5, -0.85), color=BASE_C, fontsize=7, ha="center")

    cert_meta = [r for r in cert_records if r.get("anchor_source") == "metadata"]
    plot_dose_response(
        cert_meta,
        ["absolute_error_mean_ms", "clean_rate"],
        probe="C1s_sparse_jitter",
        styles={
            "absolute_error_mean_ms": {
                "label": "mean timing summary — blind",
                "color": BASE_C, "ls": "--", "marker": "o"},
            "clean_rate": {
                "label": "clean rate (timing) — partial: lattice absorbs 72%",
                "color": "#7fb3d5"},
        },
        ax=axes[1],
        title="C1s sparse outliers: timing lattice absorbs",
    )
    nd = net_direction(c1s_records, "pattern_nll", "C1s_sparse_jitter")
    ds = sorted(nd)
    axes[1].plot([0] + ds, [0.0] + [nd[d] for d in ds], "-", color=OURS_C, marker="s",
                 label="pattern NLL (grammar) — catches")
    axes[1].legend(frameon=False, fontsize=6.4)
    axes[0].set_ylabel("per-chart net direction vs intact")
    axes[1].set_ylabel("")
    save(fig, out, "figA_complementarity")


# ---------------- figB: perceptual tier bars (draft) ----------------


def figB(ext_records, out):
    """Real-system perceptual tier bars from the clean-40 external timing run."""
    meta = [r for r in ext_records if r.get("anchor_source") == "metadata"]
    by_system = defaultdict(list)
    for r in meta:
        by_system[r["system"]].append(r)
    # official first, then descending median clean rate.
    order = sorted(
        (s for s in by_system if s != "official"),
        key=lambda s: -np.median([r["clean_rate"] for r in by_system[s]]),
    )
    systems = ["official"] + order
    tier_rows, labels, strips = [], [], {}
    for s in systems:
        rs = by_system[s]
        B = np.array([timing_buckets(r) for r in rs])
        label = f"{s} (n={len(rs)})"
        tier_rows.append({"buckets": B.mean(axis=0)})
        labels.append(label)
        strips[label] = np.array([r["clean_rate"] for r in rs if r.get("clean_rate") is not None])
    fig, ax = plot_timing_tiers(tier_rows, labels=labels, chart_clean_rates=strips)
    ax.set_title("timing profile of published systems (clean test, authored grid)", fontsize=8)
    save(fig, out, "figB_timing_tiers")


# ---------------- figC: coupling heatmap + C1s row ----------------


def figC(cp, c1s_records, out):
    spec = rows(cp / "specificity_matrix.csv")
    AGG = {"chart_quality_proxy_score", "local_pattern_score",
           "surface_structure_proxy_score", "playability_proxy_score"}
    cells = defaultdict(dict)
    targets = set()
    for r in spec:
        if r["metric_kind"] != "incumbent" or r["metric"] in AGG:
            continue
        cells[r["metric"]][r["probe"]] = float(r["std_delta_maxdose"])
        if r["is_target"] == "True":
            targets.add((r["metric"], r["probe"]))

    # C1s column from the raw C1s profile run, same standardized-delta
    # semantics (audit.coupling_matrix). Its grammar response is discovered
    # coupling: C1s has no designated chart-only target, so no box.
    for cell in coupling_matrix(c1s_records, sorted(cells)):
        if cell["probe"] == "C1s_sparse_jitter" and cell["std_delta_maxdose"] is not None:
            cells[cell["metric"]]["C1s_sparse_jitter"] = cell["std_delta_maxdose"]

    probes = sorted({p for m in cells.values() for p in m})
    metrics = sorted(cells, key=lambda m: -max(abs(v) for v in cells[m].values()))
    M = np.array([[cells[m].get(p, np.nan) for p in probes] for m in metrics])

    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    vmax = 4.0
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([p.split("_", 1)[0] for p in probes])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace("_score", "").replace("_adequacy", "") for m in metrics],
                       fontsize=6.5)
    for i, m in enumerate(metrics):
        for j, p in enumerate(probes):
            if (m, p) in targets:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.4))
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Δ score at max dose (official-SD units, clipped ±4)")
    ax.set_title("Metric coupling under targeted corruption "
                 "(boxes = designated targets; C1s: discovered coupling)")
    save(fig, out, "figC_coupling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softchart-root", default=str(Path(__file__).resolve().parents[2] / "SoftChart"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "figs"))
    ap.add_argument("--only", default=None, help="comma list: fig1,fig2,figA,figB,figC")
    args = ap.parse_args()

    root = Path(args.softchart_root)
    cp = root / "experiments/corruption_probes_v1/runs/reports/corruption_probes_20260710"
    ti = root / "experiments/timing_integration_v1/runs/reports/timing_integration_20260710"
    probe_rec = root / "experiments/corruption_probes_v1/runs/raw/records/corruption_probes_20260710.jsonl"
    cert_rec = root / "experiments/timing_integration_v1/runs/raw/records/certify_timing_tail_6ms.jsonl"
    c1s_rec = root / "experiments/corruption_probes_v1/runs/raw/records/c1s_full_profile.jsonl"
    ext_rec = root / "experiments/timing_integration_v1/runs/raw/records/ext_clean_timing_final_20260711.jsonl"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else None

    def want(name):
        return only is None or name in only

    if want("fig1"):
        fig1(cp, out)
    if want("fig2"):
        fig2(ti, out)
    if want("figA"):
        figA(jrows(probe_rec), jrows(cert_rec), jrows(c1s_rec), out)
    if want("figB"):
        figB(jrows(ext_rec), out)
    if want("figC"):
        figC(cp, jrows(c1s_rec), out)


if __name__ == "__main__":
    main()
