#!/usr/bin/env python3
"""Render the evaluation-paper figures into ``paper/figs/``.

Figure set (visual protocol -- see ``chartgeneval.plots``):

  fig0_overview.pdf          Plain-language overview of controlled-damage
                             validation and the six questions in the profile.
  figT_timing_alignment.pdf  Why local nearest-grid matching can miss a global
                             shift and how a whole-chart offset search finds it.

  figA_complementarity.pdf The bidirectional blindness exhibit: C5 improves
                           both LM loss and its pattern-IC band score while one
                           effective 4-gram repetition/uniqueness axis catches
                           the collapse; C1s is
                           absorbed by the timing lattice while grammar reacts.
  figB_timing_tiers.pdf    Timing-error-range stacked bars (within 6 ms / 6-12 /
                           12-18 / >18 / unmatched) + a per-chart strip for the
                           evaluated systems.
  figC_coupling.pdf        Probe x calibrated-score coupling heatmap with the
                           C1s row merged in (its grammar response is
                           discovered coupling -- no designated target box).
  figD_profile_matrix.pdf  Three-system calibrated profile + separate
                           constraint gates.
  figE_difficulty.pdf      Exploratory course-by-course Mapperatorinator slice.

Inputs default to the versioned records under ``artifacts/records``.

    python experiments/make_paper_figures.py

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
from matplotlib.patches import FancyBboxPatch

from _system_conditions import COURSE_MAPPING_VERSION
from chartgeneval.audit import coupling_matrix
from chartgeneval.plots import BASE_C, OURS_C, net_direction, plot_dose_response, plot_timing_tiers, timing_buckets

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 6.8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "lines.linewidth": 1.4,
                     "lines.markersize": 4})


def jrows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


# ---------------- fig0: plain-language paper overview ----------------


def fig0(out):
    """Explain the validation idea before introducing metric terminology."""
    fig = plt.figure(figsize=(6.9, 2.45), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.15, 1.0, 2.0), wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("1. Add one known flaw", fontweight="bold")
    for x in np.arange(0, 4.01, 0.5):
        ax.axvline(x, color="#d9d9d9", lw=0.6, zorder=0)
    original = np.array([0.5, 1.0, 1.75, 2.5, 3.0, 3.5])
    damaged = original + 0.18
    ax.scatter(original, np.full_like(original, 0.72), s=26, color=OURS_C, zorder=3)
    ax.scatter(damaged, np.full_like(damaged, 0.28), s=26, color=BASE_C, zorder=3)
    for x0, x1 in zip(original, damaged):
        ax.annotate("", xy=(x1, 0.35), xytext=(x0, 0.65),
                    arrowprops={"arrowstyle": "-", "color": "#8c8c8c", "lw": 0.7})
    ax.text(-0.05, 0.72, "human", ha="right", va="center", fontsize=7.5)
    ax.text(-0.05, 0.28, "edited", ha="right", va="center", fontsize=7.5)
    ax.text(2.0, 0.05, "same chart, controlled change", ha="center", fontsize=7)
    ax.set(xlim=(-0.25, 4.05), ylim=(-0.02, 1.02), xticks=[], yticks=[])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("2. Ask what the measure does", fontweight="bold")
    strength = np.arange(4)
    ax.plot(strength, [1.0, 0.78, 0.52, 0.25], "o-", color=OURS_C, label="detects flaw")
    ax.plot(strength, [0.76, 0.75, 0.76, 0.74], "s--", color="#777777", label="misses flaw")
    ax.plot(strength, [0.55, 0.62, 0.73, 0.88], "^--", color=BASE_C, label="rewards flaw")
    ax.set(xticks=strength, xticklabels=["none", "low", "mid", "high"],
           xlabel="strength of added flaw", ylabel="reported quality")
    ax.set_ylim(0.1, 1.05)
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, fontsize=6.8, loc="lower left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("3. Keep the answers separate", fontweight="bold")
    ax.set_axis_off()
    questions = [
        ("Timing", "on the song\ngrid?"),
        ("Note sequence", "local transitions\nfamiliar?"),
        ("Form", "repeats without\ncollapse?"),
        ("Music response", "tracks accents\nand energy?"),
        ("Human charts", "overall profile\nsimilar?"),
        ("Playability", "feasible density\nand strain?"),
    ]
    for i, (name, question) in enumerate(questions):
        col, row = i % 2, i // 2
        x, y = 0.01 + col * 0.5, 0.68 - row * 0.31
        box = FancyBboxPatch(
            (x, y), 0.47, 0.23,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            transform=ax.transAxes, facecolor="#f5f8fb", edgecolor="#8aa8bf", lw=0.8,
        )
        ax.add_patch(box)
        ax.text(x + 0.025, y + 0.16, name, transform=ax.transAxes,
                fontsize=6.8, fontweight="bold", va="center")
        ax.text(x + 0.025, y + 0.065, question, transform=ax.transAxes,
                fontsize=5.9, va="center", linespacing=0.95)
    save(fig, out, "fig0_overview")


# ---------------- figT: timing-reference schematic ----------------


def figT(out):
    """Show local re-pairing versus one chart-wide timing shift."""
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": (1.2, 1)})

    ax = axes[0]
    ax.set_title("Nearest-grid matching can re-pair notes", fontweight="bold")
    fine = np.arange(0, 481, 30)
    major = np.arange(0, 481, 120)
    for i, x in enumerate(fine):
        ax.axvline(
            x,
            color="#e6e6e6",
            lw=0.55,
            zorder=0,
            label="grid subdivisions" if i == 0 else None,
        )
    for x in major:
        ax.axvline(x, color="#8aa8bf", lw=1.1, zorder=0)
    original = np.array([0, 120, 240, 360, 480], dtype=float)
    shifted = original[:-1] + 60
    ax.scatter(original, np.full_like(original, 0.72), s=23, color=OURS_C, label="fixed music grid")
    ax.scatter(shifted, np.full_like(shifted, 0.32), s=23, color=BASE_C, label="chart notes: +60 ms")
    for x in shifted:
        ax.annotate("", xy=(x, 0.66), xytext=(x, 0.38),
                    arrowprops={"arrowstyle": "->", "color": "#777777", "lw": 0.7})
    ax.text(240, 0.08, "dense subdivisions make local errors look small",
            ha="center", fontsize=7)
    ax.set(xlim=(-20, 500), ylim=(0, 1), xlabel="time (ms)", yticks=[])
    ax.legend(frameon=False, fontsize=5.8, loc="upper center", ncol=3)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.set_title("Estimate one offset for the whole chart", fontweight="bold")
    shifts = np.linspace(-120, 120, 241)
    agreement = np.exp(-0.5 * ((shifts - 60) / 18) ** 2)
    agreement += 0.18 * np.exp(-0.5 * ((shifts + 60) / 25) ** 2)
    agreement /= agreement.max()
    ax.plot(shifts, agreement, color=OURS_C, lw=1.8)
    ax.axvline(60, color=BASE_C, lw=1, ls="--")
    ax.scatter([60], [1], color=BASE_C, s=26, zorder=3)
    ax.annotate("estimated: +60 ms", xy=(60, 1), xytext=(-5, 0.78),
                arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.7},
                fontsize=7.2)
    ax.set(xlabel="candidate chart offset (ms)", ylabel="grid agreement",
           xlim=(-120, 120), ylim=(0, 1.08), yticks=[])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save(fig, out, "figT_timing_alignment")


# ---------------- figA: bidirectional blindness ----------------


def _net_series(records, metric, probe):
    from chartgeneval.plots import net_direction
    nd = net_direction(records, metric, probe)
    return [nd.get(d, float("nan")) for d in (1, 2, 3)]


def figA(probe_records, cert_records, out):
    """Grouped horizontal bars of per-chart net direction (doses 1-3)."""
    cert_meta = [r for r in cert_records if r.get("anchor_source") == "metadata"]
    panels = [
        ("C5 common-pattern rewrite", [
            ("LM loss (lower looks better)", BASE_C, _net_series(probe_records, "pattern_nll", "C5_blandification")),
            ("pattern-IC band score", OURS_C, _net_series(probe_records, "pattern_ic_adequacy_score", "C5_blandification")),
            ("4-gram repetition/uniqueness score", "#58a67e", _net_series(probe_records, "repetition_adequacy_score", "C5_blandification")),
        ]),
        ("C1s a few large timing errors", [
            ("mean timing error", BASE_C, _net_series(cert_meta, "absolute_error_mean_ms", "C1s_sparse_jitter")),
            ("notes within 6 ms", "#7fb3d5", _net_series(cert_meta, "clean_rate", "C1s_sparse_jitter")),
            ("note-sequence LM loss", OURS_C, _net_series(probe_records, "pattern_nll", "C1s_sparse_jitter")),
        ]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.65), sharex=True)
    plt.rcParams.update({})
    for ax, (title, series) in zip(axes, panels):
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
    fig.supxlabel("fraction moving up minus fraction moving down vs. intact (low→high, light→dark)",
                  fontsize=9.5, y=0.01)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "figA_complementarity")


# ---------------- figB: perceptual tier bars (draft) ----------------


def figB(ext_records, out):
    """Real-system perceptual tier bars from the 40-song development run."""
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
        display = "human reference" if s == "official" else s
        label = f"{display} (n={len(rs)})"
        tier_rows.append({"buckets": B.mean(axis=0)})
        labels.append(label)
        strips[label] = np.array([r["clean_rate"] for r in rs if r.get("clean_rate") is not None])
    fig, ax = plot_timing_tiers(tier_rows, labels=labels, chart_clean_rates=strips)
    ax.set_title("timing of evaluated systems (development songs, metadata-based grid)", fontsize=8)
    save(fig, out, "figB_timing_tiers")


# ---------------- figC: coupling heatmap + C1s row ----------------


PROBE_TARGET_SCORES = {
    "C3_type_shuffle": ("transition_validity_score",),
    "C4_loop_collapse": ("repetition_adequacy_score",),
    # The surface-variety score is algebraically identical to repetition fit:
    # unique_4gram_rate + repeat_4gram_rate == 1, and the calibrated bands are
    # mirrored. Plot and box the effective axis once.
    "C5_blandification": ("repetition_adequacy_score",),
    "C6_density_scale": ("density_adequacy_score",),
    "C7_burst_insert": ("density_spike_score",),
    "C8_bar_shuffle": ("repetition_adequacy_score",),
    # C1/C1s/C2 targets require an authored timing reference and therefore do
    # not appear in this chart-only matrix.
}


def figC(probe_records, out):
    """Probe x calibrated-score coupling heatmap, built entirely from the
    clean-split probe run (audit.coupling_matrix standardized deltas)."""
    # surface_structure_proxy_score double-weights the algebraically identical
    # repetition/variety axis, so it is an archival aggregate rather than a
    # nonredundant plotted score.
    AGG = {
        "local_pattern_score",
        "playability_proxy_score",
        "surface_structure_proxy_score",
    }
    r0 = next(r for r in probe_records if r["probe"] == "official")
    metrics_all = sorted(
        k for k in r0
        if k.endswith("_score")
        and k not in AGG | {"surface_variety_adequacy_score"}
        and isinstance(r0[k], (int, float))
    )
    cells = defaultdict(dict)
    for cell in coupling_matrix(probe_records, metrics_all):
        if cell["std_delta_maxdose"] is not None:
            cells[cell["metric"]][cell["probe"]] = cell["std_delta_maxdose"]
    targets = {
        (metric, probe)
        for probe, metrics in PROBE_TARGET_SCORES.items()
        for metric in metrics
    }

    probes = sorted({p for m in cells.values() for p in m})
    metrics = sorted(cells, key=lambda m: -max(abs(v) for v in cells[m].values()))
    M = np.array([[cells[m].get(p, np.nan) for p in probes] for m in metrics])

    fig, ax = plt.subplots(figsize=(6.9, 3.4))
    vmax = 4.0
    im = ax.imshow(np.clip(M, -vmax, vmax), cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    probe_labels = {
        "C1_timing_jitter": "C1\njitter",
        "C1s_sparse_jitter": "C1s\nfew errors",
        "C2_anchor_shift": "C2\nwhole shift",
        "C3_type_shuffle": "C3\ntypes",
        "C4_loop_collapse": "C4\nloop",
        "C5_blandification": "C5\ncommon",
        "C6_density_scale": "C6\nrate",
        "C7_burst_insert": "C7\nbursts",
        "C8_bar_shuffle": "C8\nbars",
    }
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([probe_labels.get(p, p) for p in probes], fontsize=5.8)
    ax.set_yticks(range(len(metrics)))
    plain_labels = {
        "density_spike_score": "local density spike",
        "overload_score": "overload",
        "repetition_adequacy_score": "4-gram repetition/uniqueness score",
        "rare_ngram_score": "rare patterns",
        "density_adequacy_score": "note-rate score",
        "strain_adequacy_score": "short-interval score",
        "surface_structure_proxy_score": "form summary",
        "color_switch_adequacy_score": "note-type switching",
        "rhythm_complexity_adequacy_score": "rhythm variety",
        "density_variation_adequacy_score": "density variation",
        "pattern_ic_adequacy_score": "local-pattern score",
        "transition_validity_score": "familiar transitions",
        "pattern_chaos_score": "pattern irregularity",
        "boredom_score": "local repetition",
        "big_note_adequacy_score": "large-note rate",
    }
    ax.set_yticklabels([plain_labels.get(m, m) for m in metrics], fontsize=6.5)
    for i, m in enumerate(metrics):
        for j, p in enumerate(probes):
            if (m, p) in targets:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.4))
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("change at highest strength (human SD, clipped ±4)")
    ax.set_title("Calibrated chart-only scores under controlled damage")
    fig.tight_layout()
    save(fig, out, "figC_coupling")


# ---------------- figD: system profile matrix + constraint gates ------------


def figD(probe_records, ext_records, profiles, out):
    """Three-system annotated heatmap + constraint-gate glyphs."""
    from chartgeneval.plots import plot_profile
    cols = [("timing within 6 ms", None), ("note rate", "density_adequacy_score"),
            ("short intervals", "strain_adequacy_score"), ("familiar trans.", "transition_validity_score"),
            ("local patterns", "pattern_ic_adequacy_score"),
            ("4-gram repetition/uniqueness", "repetition_adequacy_score")]
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
        row = {cols[0][0]: float(np.median(timing[sys_]))}
        for label, key in cols[1:]:
            row[label] = med(chartside[sys_], key)
        display = "human reference" if sys_ == "official" else sys_
        scores[display] = row
        constraints[display] = {n: (med(chartside[sys_], k) or 0) >= 0.5 for n, k in con_keys.items()}
    fig, ax = plot_profile(
        scores,
        columns=[c for c, _ in cols],
        constraints=constraints,
        outline_failed_rows=False,
    )
    fig.tight_layout()
    save(fig, out, "figD_profile_matrix")


# ---------------- figE: exploratory difficulty slice -------------------------


DIFFICULTY_PLOT_METRICS = [
    ("note-rate score", "density_adequacy_score"),
    ("short-interval score", "strain_adequacy_score"),
    ("familiar transitions", "transition_validity_score"),
    ("local patterns", "pattern_ic_adequacy_score"),
    ("4-gram repetition/uniqueness", "repetition_adequacy_score"),
]

# Keep both frozen names in the released audit table while plotting their
# algebraically equivalent score only once.
DIFFICULTY_TABLE_METRICS = DIFFICULTY_PLOT_METRICS + [
    ("surface variety (redundant)", "surface_variety_adequacy_score"),
]


def figE(probe_records, mapper_records, timing_records, out, tables_out):
    """Difficulty-conditioned Mapperatorinator profile from paired records."""
    courses = ["easy", "normal", "hard", "oni", "ura"]
    labels = ["Easy", "Normal", "Hard", "Oni", "Ura"]
    short_labels = ["E", "N", "H", "O", "U"]

    def unique_by_song_course(rows, name):
        index = {}
        for row in rows:
            key = (row.get("sid"), row.get("course"))
            if None in key:
                raise ValueError(f"{name} row lacks sid/course: {row!r}")
            if key in index:
                raise ValueError(f"duplicate {name} row for {key!r}")
            index[key] = row
        return index

    official = unique_by_song_course(
        [r for r in probe_records if r.get("probe") == "official"],
        "human reference",
    )
    mapper = unique_by_song_course(mapper_records, "Mapperatorinator profile")
    mapper_timing_rows = [
        r for r in timing_records
        if r.get("system") == "mapperatorinator"
        and r.get("anchor_source") == "metadata"
    ]
    for row in mapper_timing_rows:
        if row.get("course_mapping_version") != COURSE_MAPPING_VERSION:
            raise ValueError("Mapperatorinator timing row has a stale course mapping")
        if row.get("grid_course") != row.get("course"):
            raise ValueError(
                f"Mapperatorinator timing course mismatch: "
                f"{row.get('course')!r} -> {row.get('grid_course')!r}"
            )
    mapper_timing = unique_by_song_course(
        mapper_timing_rows,
        "Mapperatorinator timing",
    )
    if set(mapper) != set(official) or set(mapper) != set(mapper_timing):
        raise ValueError(
            "difficulty inputs must contain the same (sid, course) pairs: "
            f"profile={len(mapper)}, human={len(official)}, timing={len(mapper_timing)}"
        )

    mapper_by_course = {
        course: [row for (_, c), row in sorted(mapper.items()) if c == course]
        for course in courses
    }
    timing_by_course = {
        course: [row for (_, c), row in sorted(mapper_timing.items()) if c == course]
        for course in courses
    }

    summary = []
    for course in courses:
        rows = mapper_by_course[course]
        ratios = []
        for row in rows:
            reference = official.get((row["sid"], course))
            if reference is None or not reference.get("density_nps"):
                raise ValueError(f"missing paired reference for {row['sid']}/{course}")
            ratios.append(float(row["density_nps"]) / float(reference["density_nps"]))
        clean = [float(r["clean_rate"]) for r in timing_by_course[course]]
        if len(clean) != len(rows):
            raise ValueError(
                f"timing/profile row mismatch for {course}: {len(clean)} != {len(rows)}"
            )
        entry = {"course": course, "n": len(rows)}
        for prefix, values in (("note_rate_ratio", ratios), ("timing_clean", clean)):
            p10, median, p90 = np.percentile(values, [10, 50, 90])
            entry[f"{prefix}_p10"] = float(p10)
            entry[f"{prefix}_median"] = float(median)
            entry[f"{prefix}_p90"] = float(p90)
        for label, metric in DIFFICULTY_TABLE_METRICS:
            entry[metric] = float(np.median([r[metric] for r in rows]))
        summary.append(entry)

    tables_out.mkdir(parents=True, exist_ok=True)
    table_path = tables_out / "mapperatorinator_difficulty_summary.csv"
    fields = list(summary[0])
    with table_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in summary:
            writer.writerow({
                key: (f"{value:.6f}" if isinstance(value, float) else value)
                for key, value in row.items()
            })
    print("wrote", table_path)

    fig = plt.figure(figsize=(6.9, 3.0), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=(1.05, 2.35, 1.05), wspace=0.38)
    x = np.arange(len(courses))

    ax = fig.add_subplot(gs[0, 0])
    ratio_q = np.array([
        [r["note_rate_ratio_p10"], r["note_rate_ratio_median"], r["note_rate_ratio_p90"]]
        for r in summary
    ])
    ax.axhline(1, color="#777777", lw=0.8, ls="--")
    ax.vlines(x, ratio_q[:, 0], ratio_q[:, 2], color=OURS_C, lw=2)
    ax.scatter(x, ratio_q[:, 1], color=OURS_C, s=26, zorder=3)
    ax.set_title("(a) Generated / human\nnote rate", fontweight="bold")
    ax.set(xticks=x, xticklabels=short_labels, xlabel="course", ylabel="ratio",
           ylim=(0.6, max(3.2, ratio_q[:, 2].max() + 0.1)))
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    matrix = np.array([[r[metric] for _, metric in DIFFICULTY_PLOT_METRICS] for r in summary])
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_title("(b) Calibrated scores by declared role", fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels([f"{label}  n={row['n']}" for label, row in zip(labels, summary)])
    ax.set_xticks(np.arange(len(DIFFICULTY_PLOT_METRICS)))
    ax.set_xticklabels([label for label, _ in DIFFICULTY_PLOT_METRICS], rotation=35, ha="right")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if matrix[i, j] > 0.55 else "#1a1a1a")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cb.set_ticks([0, 0.5, 1])
    cb.set_label("score (higher = better by declared role)")

    ax = fig.add_subplot(gs[0, 2])
    timing_q = np.array([
        [r["timing_clean_p10"], r["timing_clean_median"], r["timing_clean_p90"]]
        for r in summary
    ])
    ax.vlines(x, timing_q[:, 0], timing_q[:, 2], color=BASE_C, lw=2)
    ax.scatter(x, timing_q[:, 1], color=BASE_C, s=26, zorder=3)
    ax.set_title("(c) Notes within\n6 ms of the grid", fontweight="bold")
    ax.set(xticks=x, xticklabels=short_labels, xlabel="course",
           ylabel="fraction of notes", ylim=(0, 1.02))
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    save(fig, out, "figE_difficulty")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts-root",
        default=str(Path(__file__).resolve().parents[1] / "artifacts"),
    )
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "figs"))
    ap.add_argument("--only", default=None, help="comma list: figA,figB,figC,figD")
    args = ap.parse_args()

    root = Path(args.artifacts_root)
    records = root / "records"
    probe_rec = records / "corruption_probes_development.jsonl"
    cert_rec = records / "timing_corruptions_clean.jsonl"
    ext_rec = records / "system_timing_clean.jsonl"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    only = set(args.only.split(",")) if args.only else {
        "fig0", "figT", "figA", "figB", "figC", "figD", "figE"
    }

    def want(name):
        return name in only

    if want("fig0"):
        fig0(out)
    if want("figT"):
        figT(out)
    if want("figA"):
        figA(jrows(probe_rec), jrows(cert_rec), out)
    if want("figB"):
        figB(jrows(ext_rec), out)
    if want("figC"):
        figC(jrows(probe_rec), out)
    if want("figD"):
        profiles = {
            "mapperatorinator": jrows(records / "system_profile_mapperatorinator.jsonl"),
            "taikonation": jrows(records / "system_profile_taikonation.jsonl"),
        }
        figD(jrows(probe_rec), jrows(ext_rec), profiles, out)
    if want("figE"):
        figE(
            jrows(probe_rec),
            jrows(records / "system_profile_mapperatorinator.jsonl"),
            jrows(ext_rec),
            out,
            root / "tables",
        )


if __name__ == "__main__":
    main()
