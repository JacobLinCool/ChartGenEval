"""Reusable visualization library for chart-quality profiles (``[plots]`` extra).

One idiom per role tier, so every figure in the protocol speaks the same
visual language:

    L0 constraints    -> pass/fail glyph column on the profile heatmap
    L1 family scores  -> :func:`plot_profile` (heatmap default; radar for demos)
    L2 family drill   -> :func:`plot_timing_tiers` (timing's idiom),
                         :func:`plot_dose_response` (probe certification),
                         :func:`plot_manifold` (official-envelope scatter)
    L3 per-chart tail -> quantile strips on the tier bars

Design rules encoded here (paper section "presentation protocol"):

* perceptual buckets (deadzone multiples), never raw-ms distributions
* per-chart aggregation first; pooled-note statistics are not drawn
* n/a is data: rendered as an explicit grey dash, never imputed
* radar: fixed axis order, outline only, never filled -- area is not a score
* two-hue scheme: blue = calibrated/ours, red = baseline/blind metric

All functions lazily import matplotlib and return ``(fig, ax)``. numpy is the
only hard dependency; :func:`plot_chart_diagnostics` additionally needs scipy
(the timing family's matcher).
"""

from __future__ import annotations

import numpy as np

OURS_C = "#2471a3"  # calibrated / catching metric (blue)
BASE_C = "#c0392b"  # baseline / blind metric (red)

# Perceptual severity buckets for the timing family (luminance-ordered so the
# stacked bar survives grayscale printing).
TIER_BUCKETS = [
    ("within 6 ms", "#d6eaf8"),
    ("6–12 ms", "#f5b041"),
    ("12–18 ms", "#dc7633"),
    (">18 ms", "#922b21"),
    ("unmatched", "#5d6d7e"),
]


def _mpl():
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        return plt
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for chartgeneval.plots; install the [plots] extra"
        ) from e


# ---------------------------------------------------------------- L1: profile


def plot_profile(
    scores,
    *,
    columns=None,
    constraints=None,
    kind="heatmap",
    vmin=0.0,
    vmax=1.0,
    fmt="{:.2f}",
    outline_failed_rows=True,
    ax=None,
):
    """System x family profile. ``scores``: ``{system: {column: value|None}}``.

    ``kind="heatmap"`` (default, evidence-grade): value-annotated matrix;
    ``None``/missing cells render as a grey dash (the applicability matrix is
    part of the data). ``constraints``: ``{system: {name: bool}}`` adds a
    pass/fail glyph block right of the heatmap. Set
    ``outline_failed_rows=False`` when the glyphs alone are clearer.

    ``kind="radar"`` (demo-grade): fixed axis order = ``columns``, outline
    only, no fill. Systems with any missing column are skipped with a warning
    text -- a broken polygon would silently lie.
    """
    plt = _mpl()
    systems = list(scores)
    if columns is None:
        columns = sorted({c for row in scores.values() for c in row})

    if kind == "radar":
        angles = np.linspace(0, 2 * np.pi, len(columns), endpoint=False)
        if ax is None:
            fig, ax = plt.subplots(figsize=(3.4, 3.4), subplot_kw={"polar": True})
        else:
            fig = ax.figure
        skipped = []
        for i, s in enumerate(systems):
            vals = [scores[s].get(c) for c in columns]
            if any(v is None for v in vals):
                skipped.append(s)
                continue
            closed = np.concatenate([angles, angles[:1]])
            ax.plot(closed, vals + vals[:1], marker="o", label=s, lw=1.4)
        ax.set_xticks(angles)
        ax.set_xticklabels(columns)
        ax.set_ylim(vmin, vmax)
        ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.1), frameon=False, fontsize=7)
        if skipped:
            ax.set_title(f"skipped (incomplete axes): {', '.join(skipped)}", fontsize=7)
        return fig, ax

    if kind != "heatmap":
        raise ValueError(f"unknown kind: {kind!r}")

    M = np.full((len(systems), len(columns)), np.nan)
    for i, s in enumerate(systems):
        for j, c in enumerate(columns):
            v = scores[s].get(c)
            if v is not None:
                M[i, j] = float(v)

    con_names = []
    if constraints:
        con_names = sorted({k for row in constraints.values() for k in row})

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(1.1 * len(columns) + 0.9 * len(con_names) + 2.2, 0.42 * len(systems) + 1.2)
        )
    else:
        fig = ax.figure

    masked = np.ma.masked_invalid(M)
    cmap = plt.get_cmap("Blues").with_extremes(bad="#eeeeee")
    ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(len(systems)):
        for j in range(len(columns)):
            if np.isnan(M[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#888888")
            else:
                dark = (M[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.55
                ax.text(
                    j, i, fmt.format(M[i, j]),
                    ha="center", va="center", fontsize=8.5,
                    color="white" if dark else "#1a1a1a",
                )
    if con_names:
        # Separate the pass/fail checks from the score heatmap with a light
        # divider so the glyphs read as their own labelled columns.
        ax.axvline(len(columns) - 0.5 + 0.28, color="#bbbbbb", lw=0.8)
    for k, name in enumerate(con_names):
        x = len(columns) - 0.5 + 0.62 + k * 0.62
        ax.text(x, -0.72, name, ha="center", va="bottom", fontsize=7.5, rotation=30)
        for i, s in enumerate(systems):
            v = (constraints.get(s) or {}).get(name)
            glyph, color = ("✓", "#1e8449") if v else ("✗", BASE_C)
            if v is None:
                glyph, color = "—", "#888888"
            ax.text(x, i, glyph, ha="center", va="center", color=color, fontsize=11)
        if constraints and any((constraints.get(s) or {}).get(name) is False for s in systems):
            pass
    for i, s in enumerate(systems):
        if (
            outline_failed_rows
            and constraints
            and any(v is False for v in (constraints.get(s) or {}).values())
        ):
            ax.add_patch(
                plt.Rectangle(
                    (-0.5, i - 0.5), len(columns), 1,
                    fill=False, edgecolor=BASE_C, lw=1.2,
                )
            )
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=30, ha="right")
    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems)
    ax.set_xlim(-0.5, len(columns) - 0.5 + (0.7 + 0.62 * len(con_names) if con_names else 0))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig, ax


# ------------------------------------------------- L2: timing tier drill-down


def timing_buckets(row):
    """Partition all notes of one chart into the five perceptual buckets.

    ``row`` is a timing-family result (or per-chart aggregate of them) with
    ``clean_rate`` / ``matched_rate`` / ``absolute_violation_rate{,_2x,_3x}``
    / ``unsupported_rate``. Returns bucket fractions summing to 1. If the row
    carries a precomputed ``buckets`` vector (e.g. the mean of per-chart
    bucket vectors -- the exact aggregation), it is passed through.
    """
    if row.get("buckets") is not None:
        parts = np.clip(np.asarray(row["buckets"], dtype=float), 0.0, None)
        total = parts.sum()
        return parts / total if total > 0 else parts
    matched = row.get("matched_rate")
    unsup = row.get("unsupported_rate") or 0.0
    if matched is None:
        matched = 1.0 - unsup
    v1 = row.get("absolute_violation_rate") or 0.0
    v2 = row.get("absolute_violation_rate_2x") or 0.0
    v3 = row.get("absolute_violation_rate_3x") or 0.0
    clean = row.get("clean_rate")
    if clean is None:
        clean = matched * (1.0 - v1)
    parts = np.array(
        [clean, matched * (v1 - v2), matched * (v2 - v3), matched * v3, unsup],
        dtype=float,
    )
    parts = np.clip(parts, 0.0, None)
    total = parts.sum()
    return parts / total if total > 0 else parts


def plot_timing_tiers(rows, *, labels=None, chart_clean_rates=None, ax=None):
    """Stacked perceptual-bucket bars, one per system.

    ``rows``: list of timing aggregates (means of per-chart values -- aggregate
    per chart first). ``chart_clean_rates``: optional ``{label: array}`` of
    per-chart clean rates; draws a p10--median--p90 strip with the worst chart
    marked ×, making tail charts visible next to the corpus bar.
    """
    plt = _mpl()
    n = len(rows)
    if labels is None:
        labels = [r.get("label", f"system {i}") for i, r in enumerate(rows)]
    strip = chart_clean_rates is not None
    if ax is not None:
        fig = ax.figure
        ax_strip = None
        strip = False
    else:
        if strip:
            fig, (ax, ax_strip) = plt.subplots(
                1, 2, figsize=(6.9, 0.44 * n + 1.5), sharey=True,
                gridspec_kw={"width_ratios": [4, 1], "wspace": 0.04},
            )
        else:
            fig, ax = plt.subplots(figsize=(6.9, 0.44 * n + 1.5))
            ax_strip = None

    y = np.arange(n)[::-1]
    left = np.zeros(n)
    B = np.array([timing_buckets(r) for r in rows])
    for b, (name, color) in enumerate(TIER_BUCKETS):
        ax.barh(y, B[:, b], left=left, color=color, edgecolor="white", lw=0.4, label=name)
        left += B[:, b]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("fraction of notes by timing-error range")
    ax.legend(loc="upper center", ncols=len(TIER_BUCKETS), frameon=False,
              fontsize=6.4, bbox_to_anchor=(0.62, -0.30))

    if strip and ax_strip is not None:
        for yi, lab in zip(y, labels):
            vals = np.asarray(chart_clean_rates.get(lab, []), dtype=float)
            if len(vals) == 0:
                continue
            p10, med, p90 = np.percentile(vals, [10, 50, 90])
            ax_strip.plot([p10, p90], [yi, yi], color="#5d6d7e", lw=1.4)
            ax_strip.plot([med], [yi], "o", color=OURS_C, ms=3.5)
            ax_strip.plot([vals.min()], [yi], "x", color=BASE_C, ms=4)
        ax_strip.set_xlim(-0.05, 1.05)
        ax_strip.set_xticks([0, 0.5, 1])
        ax_strip.set_xlabel("within-6-ms fraction", fontsize=7)
        ax_strip.set_title("p10–med–p90, × = worst", fontsize=6.2, color="dimgray")
        for spine in ("top", "right", "left"):
            ax_strip.spines[spine].set_visible(False)
        ax_strip.tick_params(left=False)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return fig, ax


# ------------------------------------------------ L2: probe dose certification


def net_direction(records, metric, probe, *, dose0_probe="official", key=("sid", "course")):
    """Per-dose paired net direction in ``[-1, 1]``.

    ``frac(charts where value rose vs its own dose-0) - frac(fell)``. This is
    the certification statistic: magnitude medians saturate under re-matching
    absorption, paired direction does not.
    """
    def kf(r):
        return tuple(r.get(k) for k in key)

    def noop(r):
        return bool(r.get("corruption_noop") or r.get("noop"))

    probe_field = "probe" if any("probe" in r for r in records[:1]) else "target_id"
    base = {kf(r): r for r in records if r.get(probe_field) == dose0_probe and not noop(r)}
    out = {}
    doses = sorted({r["dose_index"] for r in records if r.get(probe_field) == probe})
    for d in doses:
        ups = downs = n = 0
        for r in records:
            if r.get(probe_field) != probe or r["dose_index"] != d or noop(r):
                continue
            b = base.get(kf(r))
            if b is None or r.get(metric) is None or b.get(metric) is None:
                continue
            n += 1
            ups += r[metric] > b[metric]
            downs += r[metric] < b[metric]
        out[d] = (ups - downs) / n if n else np.nan
    return out


def plot_dose_response(records, metrics, *, probe, styles=None, ax=None, title=None):
    """Paired net-direction dose curves for one probe.

    ``metrics``: list of metric field names; ``styles``: optional
    ``{metric: {"label", "color", "ls"}}``. Blind metrics hug 0; catching
    metrics run to ±1.
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
    else:
        fig = ax.figure
    styles = styles or {}
    for m in metrics:
        nd = net_direction(records, m, probe)
        ds = sorted(nd)
        st = styles.get(m, {})
        ax.plot(
            [0] + ds, [0.0] + [nd[d] for d in ds],
            st.get("ls", "-"), color=st.get("color", OURS_C),
            marker=st.get("marker", "s"), label=st.get("label", m),
        )
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("corruption dose")
    ax.set_ylabel("per-chart net direction")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=6.4)
    return fig, ax


# -------------------------------------------------- L2: official-envelope map


def plot_manifold(official_xy, systems, *, xlabel, ylabel, ax=None):
    """Official charts as a grey cloud; systems as labelled markers.

    Gaming one axis (e.g. perplexity driven below the human range) shows up as
    leaving the cloud -- the visual form of ``official_manifold_gap``.
    """
    plt = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.8))
    else:
        fig = ax.figure
    xy = np.asarray(official_xy, dtype=float)
    ax.scatter(xy[:, 0], xy[:, 1], s=8, color="#b0b8bf", alpha=0.55, lw=0,
               label="official charts")
    markers = ["o", "s", "D", "^", "v", "P", "X"]
    for i, (label, pt) in enumerate(systems.items()):
        ax.scatter([pt[0]], [pt[1]], s=46, marker=markers[i % len(markers)],
                   edgecolor="black", lw=0.7, zorder=3, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=6.4)
    return fig, ax


# ---------------------------------------------- case study: single-chart strip


def plot_chart_diagnostics(events, ctx, *, window=None, ax=None):
    """Timeline strip of one chart: notes coloured by perceptual bucket.

    Downbeats from ``ctx['grid']`` as vertical lines; each note tick coloured
    clean / 1–2× / 2–3× / >3× / unsupported. Requires
    scipy (timing matcher). ``window=(t0, t1)`` crops the strip.
    """
    plt = _mpl()
    from .events import sorted_hits
    from .metrics import timing as T

    hits = sorted_hits(events)
    times = np.array([t for t, _ in hits])
    grid = ctx.get("grid") or {}
    downbeats = sorted(float(x) for x in grid.get("downbeats") or [])
    if len(downbeats) < 2:
        raise ValueError("ctx['grid']['downbeats'] with >=2 entries required")
    anchors = T.finalize_anchors(
        T.merge_anchors(T.estimated_grid_anchors(downbeats, audio_duration=ctx.get("duration")))
    )
    matches, unsupported = T.match_notes_to_anchors(times, anchors)
    bucket = np.full(len(times), 4, dtype=int)  # default: unsupported
    for m in matches:
        e = abs(m["error"])
        if e <= T.TAU_ABS_S:
            bucket[m["note_index"]] = 0
        elif e <= 2 * T.TAU_ABS_S:
            bucket[m["note_index"]] = 1
        elif e <= 3 * T.TAU_ABS_S:
            bucket[m["note_index"]] = 2
        else:
            bucket[m["note_index"]] = 3

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.9, 1.4))
    else:
        fig = ax.figure
    for db in downbeats:
        ax.axvline(db, color="#d5d8dc", lw=0.6, zorder=1)
    for b, (name, color) in enumerate(TIER_BUCKETS):
        sel = bucket == b
        if not sel.any():
            continue
        ax.vlines(times[sel], 0.15, 0.85, color=color,
                  lw=1.6 if b else 1.0, zorder=2 + b, label=name)
    if window:
        ax.set_xlim(*window)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="upper right", ncols=5, frameon=False, fontsize=5.8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    return fig, ax
