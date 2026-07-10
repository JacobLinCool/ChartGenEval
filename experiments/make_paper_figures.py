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
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
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


def _net_series(records, metric, probe):
    from chartgeneval.plots import net_direction
    nd = net_direction(records, metric, probe)
    return [nd.get(d, float("nan")) for d in (1, 2, 3)]


def figA(probe_records, cert_records, out):
    """Grouped horizontal bars of per-chart net direction (doses 1-3)."""
    cert_meta = [r for r in cert_records if r.get("anchor_source") == "metadata"]
    panels = [
        ("C5 flattening (LM-argmax rewrite)", [
            ("n-gram NLL (baseline)", BASE_C, _net_series(probe_records, "pattern_nll", "C5_blandification")),
            ("pattern-IC band score", OURS_C, _net_series(probe_records, "pattern_ic_adequacy_score", "C5_blandification")),
            ("repetition adequacy", "#7fb3d5", _net_series(probe_records, "repetition_adequacy_score", "C5_blandification")),
        ]),
        ("C1s sparse outliers", [
            ("mean timing summary", BASE_C, _net_series(cert_meta, "absolute_error_mean_ms", "C1s_sparse_jitter")),
            ("timing clean rate", "#7fb3d5", _net_series(cert_meta, "clean_rate", "C1s_sparse_jitter")),
            ("grammar NLL", OURS_C, _net_series(probe_records, "pattern_nll", "C1s_sparse_jitter")),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), sharex=True)
    plt.rcParams.update({})
    for ax, (title, series) in zip(axes, panels):
        ax.axvspan(-0.15, 0.15, color="#d5d8dc", alpha=0.5, lw=0, zorder=0)
        ax.axvline(0, color="gray", lw=0.6)
        yticks, ylabels = [], []
        for i, (label, color, vals) in enumerate(series):
            base_y = -i * 1.0
            for d, v in enumerate(vals):
                ax.barh(base_y + (1 - d) * 0.26, v, height=0.24,
                        color=color, alpha=0.45 + 0.275 * d, zorder=2)
            yticks.append(base_y)
            ylabels.append(label)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=9.5)
        ax.set_xlim(-1.05, 1.05)
        ax.tick_params(axis="x", labelsize=9)
        ax.set_title(title, fontsize=10.5)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(left=False)
    fig.supxlabel("per-chart net direction vs. intact (doses 1→3, light→dark)",
                  fontsize=9.5, y=0.01)
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
        key=lambda s: -np.mean([r["clean_rate"] for r in by_system[s]]),
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
    ax.set_title("timing profiles of published systems (held-out test songs, authored grid)", fontsize=8)
    save(fig, out, "figB_timing_tiers")


# ---------------- figC: coupling heatmap + C1s row ----------------


PROBE_TARGET_SCORES = {
    "C1_timing_jitter": "rhythm_complexity_adequacy_score",
    "C3_type_shuffle": "transition_validity_score",
    "C4_loop_collapse": "repetition_adequacy_score",
    "C5_blandification": "pattern_ic_adequacy_score",
    "C6_density_scale": "density_adequacy_score",
    "C7_burst_insert": "density_spike_score",
    "C8_bar_shuffle": "repetition_adequacy_score",
    # C2's designated target is the audio-anchored witness (not chart-only);
    # C1s has no designated chart-only target (discovered coupling).
}


def figC(probe_records, out):
    """Probe x calibrated-score coupling heatmap, built entirely from the
    clean-split probe run (audit.coupling_matrix standardized deltas)."""
    AGG = {"chart_quality_proxy_score", "local_pattern_score",
           "playability_proxy_score"}
    r0 = next(r for r in probe_records if r["probe"] == "official")
    metrics_all = sorted(
        k for k in r0
        if k.endswith("_score") and k not in AGG and isinstance(r0[k], (int, float))
    )
    cells = defaultdict(dict)
    for cell in coupling_matrix(probe_records, metrics_all):
        if cell["std_delta_maxdose"] is not None:
            cells[cell["metric"]][cell["probe"]] = cell["std_delta_maxdose"]
    targets = {(m, p) for p, m in PROBE_TARGET_SCORES.items()}

    probes = sorted({p for m in cells.values() for p in m})
    metrics = sorted(cells, key=lambda m: -max(abs(v) for v in cells[m].values()))
    M = np.array([[cells[m].get(p, np.nan) for p in probes] for m in metrics])

    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    vmax = 4.0
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([p.split("_", 1)[0] for p in probes])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m.replace("_score", "").replace("_adequacy", "").replace("_", " ") for m in metrics],
                       fontsize=6.5)
    for i, m in enumerate(metrics):
        for j, p in enumerate(probes):
            if (m, p) in targets:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.4))
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Δ score at max dose (official-SD units, clipped ±4)")
    ax.set_title("Band-score responses under targeted corruption "
                 "(boxes = designated targets; C1s/C2: no chart-only target)")
    save(fig, out, "figC_coupling")


# ---------------- figD: radar profile + constraint gates ----------------


def figD(probe_records, ext_records, profiles, out):
    """Radar over the ranker axes for the full-profile systems (taiko
    vocabulary only; cross-vocabulary systems are covered by the timing
    figure), plus a separate constraint-gate panel: gates veto, they are
    not axes."""
    axes_cols = [
        ("timing clean", None),
        ("density", "density_adequacy_score"),
        ("strain", "strain_adequacy_score"),
        ("trans.\nvalidity", "transition_validity_score"),
        ("pattern IC", "pattern_ic_adequacy_score"),
        ("repetition", "repetition_adequacy_score"),
        ("variety", "surface_variety_adequacy_score"),
    ]
    gate_keys = [("overload", "overload_score"), ("spike", "density_spike_score"),
                 ("playable", "playability_proxy_score")]

    timing = defaultdict(list)
    for r in ext_records:
        if r.get("anchor_source") == "metadata" and r.get("clean_rate") is not None:
            timing[r["system"]].append(r["clean_rate"])
    chartside = {"official": [r for r in probe_records if r["probe"] == "official"]}
    chartside.update(profiles)

    def med(rows_, key):
        v = [r[key] for r in rows_ if r.get(key) is not None]
        return float(np.median(v)) if v else None

    order = ["official", "mapperatorinator", "taikonation"]
    colors = {"official": "#2c3e50", "mapperatorinator": "#2471a3", "taikonation": "#c0392b"}
    fig = plt.figure(figsize=(6.9, 3.1))
    axr = fig.add_axes([0.06, 0.08, 0.5, 0.84], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(axes_cols), endpoint=False)
    for sys_ in order:
        vals = [float(np.median(timing[sys_]))]
        vals += [med(chartside[sys_], key) for _, key in axes_cols[1:]]
        closed = np.concatenate([angles, angles[:1]])
        axr.plot(closed, vals + vals[:1], marker="o", ms=3, lw=1.6,
                 color=colors[sys_], label=sys_)
    axr.set_xticks(angles)
    axr.set_xticklabels([c for c, _ in axes_cols], fontsize=7.5)
    axr.set_ylim(0, 1.0)
    axr.set_yticks([0.25, 0.5, 0.75, 1.0])
    axr.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=6)
    axr.legend(loc="upper left", bbox_to_anchor=(1.02, 1.08), frameon=False, fontsize=8)

    axg = fig.add_axes([0.66, 0.10, 0.3, 0.55])
    axg.set_title("constraint gates (veto, not axes)", fontsize=8)
    for j, (gname, _) in enumerate(gate_keys):
        axg.text(j, len(order) - 0.4, gname, ha="center", fontsize=7.5)
    for i, sys_ in enumerate(order):
        y = len(order) - 1.4 - i
        axg.text(-0.8, y, sys_, ha="right", va="center", fontsize=7.5)
        for j, (_, key) in enumerate(gate_keys):
            v = med(chartside[sys_], key)
            ok = v is not None and v >= 0.5
            axg.text(j, y, "\u2713" if ok else "\u2717", ha="center", va="center",
                     fontsize=11, color="#1e8449" if ok else "#c0392b")
    axg.set_xlim(-2.6, len(gate_keys) - 0.5)
    axg.set_ylim(-0.8, len(order))
    axg.axis("off")
    save(fig, out, "figD_profile_radar")


def figD_heat(probe_records, ext_records, profiles, out):
    """Variant: 3-system annotated heatmap + gate glyphs (zeros legible)."""
    from chartgeneval.plots import plot_profile
    cols = [("timing clean", None), ("density", "density_adequacy_score"),
            ("strain", "strain_adequacy_score"), ("trans. validity", "transition_validity_score"),
            ("pattern IC", "pattern_ic_adequacy_score"), ("repetition", "repetition_adequacy_score"),
            ("variety", "surface_variety_adequacy_score")]
    con_keys = {"overload": "overload_score", "spike": "density_spike_score",
                "playable": "playability_proxy_score"}
    timing = defaultdict(list)
    for r in ext_records:
        if r.get("anchor_source") == "metadata" and r.get("clean_rate") is not None:
            timing[r["system"]].append(r["clean_rate"])
    chartside = {"official": [r for r in probe_records if r["probe"] == "official"]}
    chartside.update(profiles)
    def med(rows_, key):
        v = [r[key] for r in rows_ if r.get(key) is not None]
        return float(np.median(v)) if v else None
    order = ["official", "mapperatorinator", "taikonation"]
    scores, constraints = {}, {}
    for sys_ in order:
        row = {"timing clean": float(np.median(timing[sys_]))}
        for label, key in cols[1:]:
            row[label] = med(chartside[sys_], key)
        scores[sys_] = row
        constraints[sys_] = {n: (med(chartside[sys_], k) or 0) >= 0.5 for n, k in con_keys.items()}
    fig, ax = plot_profile(scores, columns=[c for c, _ in cols], constraints=constraints)
    save(fig, out, "figD_profile_matrix")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--softchart-root", default=str(Path(__file__).resolve().parents[2] / "SoftChart"))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "figs"))
    ap.add_argument("--only", default=None, help="comma list: fig1,fig2,figA,figB,figC")
    args = ap.parse_args()

    root = Path(args.softchart_root)
    cp = root / "experiments/corruption_probes_v1/runs/reports/corruption_probes_20260710"
    ti = root / "experiments/timing_integration_v1/runs/reports/timing_integration_20260710"
    probe_rec = root / "experiments/corruption_probes_v1/runs/raw/records/clean_probes_v2rational_20260711.jsonl"
    cert_rec = root / "experiments/timing_integration_v1/runs/raw/records/certify_timing_tail_6ms_clean.jsonl"
    ext_rec = root / "experiments/timing_integration_v1/runs/raw/records/ext_clean_timing_final_20260711.jsonl"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # fig1/fig2 read old-split report CSVs; superseded by figA and the C2 dual
    # table -- regenerate only on explicit request.
    only = set(args.only.split(",")) if args.only else {"figA", "figB", "figC", "figD"}

    def want(name):
        return name in only

    if want("fig1"):
        fig1(cp, out)
    if want("fig2"):
        fig2(ti, out)
    if want("figA"):
        figA(jrows(probe_rec), jrows(cert_rec), out)
    if want("figB"):
        figB(jrows(ext_rec), out)
    if want("figC"):
        figC(jrows(probe_rec), out)
    if want("figD"):
        profiles = {
            "mapperatorinator": jrows(root / "experiments/chart_quality_metrics_v1/runs/ext_mapperatorinator_clean_profile_v2.jsonl"),
            "taikonation": jrows(root / "experiments/chart_quality_metrics_v1/runs/ext_taikonation_clean_profile_v2.jsonl"),
        }
        figD(jrows(probe_rec), jrows(ext_rec), profiles, out)
        figD_heat(jrows(probe_rec), jrows(ext_rec), profiles, out)


if __name__ == "__main__":
    main()
