#!/usr/bin/env python3
"""Render the evaluation-paper figures into ``paper/figs/``.

Figure set (visual protocol -- see ``chartgeneval.plots``):

  fig0_thesis.pdf            First-page map of the six evaluation dimensions.
  figT_timing_alignment.pdf  Why local nearest-grid matching can miss a global
                             shift and how a whole-chart offset search finds it.

  figA_complementarity.pdf Two wrong-way incentives: C5 improves both LM loss
                           and its pattern-IC band score, while C4 loop collapse
                           increases self-similarity. The 4-gram typicality
                           axis moves against both failures.
  figB_timing_tiers.pdf    Timing-error-range stacked bars (within 6 ms / 6-12 /
                           12-18 / >18 / unmatched) + a per-chart strip for the
                           evaluated systems.
  figC_coupling.pdf        Probe x calibrated-score coupling heatmap with the
                           C1s row merged in (its grammar response is
                           discovered coupling -- no designated target box).
  figD_profile_matrix.pdf  Three-system calibrated profile + separate
                           feasibility checks.
  figE_difficulty.pdf      Exploratory course-by-course Mapperatorinator slice.
  figM_metric_constructs.pdf
                           Visual guide to the six measurement roles.
  figP_corruption_atlas.pdf
                           Visual atlas of the nine controlled corruptions and
                           their prespecified target readings.
  figCC_corpus_coverage.pdf  Human-reference corpus coverage: star level per
                           course and the tempo histogram.

Inputs default to the versioned records under ``artifacts/records``.

    python experiments/make_paper_figures.py

Requires the ``[plots]`` extra. Traditional Chinese schematics require a
variable Noto Sans TC font discoverable by Matplotlib or supplied explicitly
with ``--cjk-font``; generation fails if the font or its weight axis is absent.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch

from _system_conditions import COURSE_MAPPING_VERSION
from chartgeneval.audit import coupling_matrix
from chartgeneval.plots import BASE_C, OURS_C, net_direction, plot_dose_response, plot_timing_tiers, timing_buckets
from confirmatory_holdout_v1.contract import load_contract

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.color": "#263238",
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8,
    "axes.labelcolor": "#263238",
    "axes.edgecolor": "#b7c6cf",
    "axes.linewidth": 0.7,
    "legend.fontsize": 6.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.color": "#6b7780",
    "ytick.color": "#6b7780",
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#2471a3"
SKY = "#56b4e9"
TEAL = "#009e73"
GOLD = "#e69f00"
PURPLE = "#8e6c8a"
VERMILLION = "#d55e00"
INK = "#263238"
MID = "#6b7780"
LIGHT = "#f4f7f9"

DIMENSIONS = [
    ("timing", BLUE, "Timing", "時值對齊"),
    ("music", TEAL, "Music response", "音樂呼應"),
    ("sequence", PURPLE, "Note sequence", "音符序列"),
    ("form", GOLD, "Repetition & form", "重複與曲式"),
    ("human", VERMILLION, "Human-chart distance", "人類譜面距離"),
    ("difficulty", "#607d8b", "Difficulty & limits", "難度與限制"),
]
DIMENSION_COLORS = {key: color for key, color, _, _ in DIMENSIONS}
DIMENSION_LABELS = {
    "en": {key: en for key, _, en, _ in DIMENSIONS},
    "zh-TW": {key: zh for key, _, _, zh in DIMENSIONS},
}

_CJK_FONT_SOURCE = None
_CJK_FONT_TEMP = None


def configure_cjk_font(path=None):
    """Select and validate the variable CJK font used by localized figures."""
    global _CJK_FONT_SOURCE, _CJK_FONT_TEMP
    if path is None:
        try:
            resolved = Path(font_manager.findfont(
                FontProperties(family="Noto Sans TC", weight=100),
                fallback_to_default=False,
            ))
        except ValueError as exc:
            raise FileNotFoundError(
                "Noto Sans TC is required for Traditional Chinese figures; "
                "install it or pass --cjk-font PATH"
            ) from exc
    else:
        resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Traditional Chinese font not found: {resolved}")

    with TTFont(resolved) as font:
        axes = {axis.axisTag for axis in font["fvar"].axes} if "fvar" in font else set()
    if "wght" not in axes:
        raise ValueError(
            f"Traditional Chinese font must expose a variable wght axis: {resolved}"
        )

    if _CJK_FONT_TEMP is not None:
        _CJK_FONT_TEMP.cleanup()
    _CJK_FONT_SOURCE = resolved
    _CJK_FONT_TEMP = None
    _cjk_font_instances.cache_clear()


@lru_cache(maxsize=1)
def _cjk_font_instances():
    """Instantiate regular and bold static fonts from one validated source."""
    global _CJK_FONT_TEMP
    if _CJK_FONT_SOURCE is None:
        configure_cjk_font()

    _CJK_FONT_TEMP = tempfile.TemporaryDirectory(prefix="chartgeneval-cjk-")
    font_dir = Path(_CJK_FONT_TEMP.name)
    paths = []
    for label, weight in (("regular", 400), ("bold", 700)):
        with TTFont(_CJK_FONT_SOURCE) as variable_font:
            instance = instantiateVariableFont(
                variable_font,
                {"wght": weight},
                inplace=False,
                updateFontNames=True,
            )
        instance_path = font_dir / f"NotoSansTC-{label}.ttf"
        instance.save(instance_path)
        instance.close()
        paths.append(instance_path)
    return tuple(FontProperties(fname=str(path)) for path in paths)


def locale_fonts(locale):
    """Return regular/bold font properties for schematic text."""
    if locale == "zh-TW":
        return _cjk_font_instances()
    return None, None


def localized_name(base, locale):
    return base if locale == "en" else f"{base}.zh-TW"


def tx(ax, x, y, text, *, locale="en", bold=False, **kwargs):
    regular, strong = locale_fonts(locale)
    if regular is not None:
        kwargs["fontproperties"] = strong if bold else regular
    elif bold:
        kwargs["fontweight"] = "bold"
    return ax.text(x, y, text, **kwargs)

# Display names for the evaluated systems, matching the formal names used in
# the paper body rather than the lowercase record keys.
DISPLAY_NAMES = {
    "official": "human reference",
    "mapperatorinator": "Mapperatorinator",
    "taikonation": "TaikoNation",
    "genelive": "GenéLive!",
    "ddconset": "DDC onset",
    "autoosu": "AutoOsu",
}


def display_name(system):
    return DISPLAY_NAMES.get(system, system)


def jrows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(fig, out, name):
    """Write a fixed-size vector PDF after checking the physical safe area."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    width, height = fig.get_size_inches()
    margin = 2.5 / 72.0
    if (
        tight.x0 < margin
        or tight.y0 < margin
        or tight.x1 > width - margin
        or tight.y1 > height - margin
    ):
        raise ValueError(
            f"{name} escapes the fixed canvas: "
            f"tight=({tight.x0:.3f}, {tight.y0:.3f}, {tight.x1:.3f}, {tight.y1:.3f}), "
            f"canvas=({width:.3f}, {height:.3f})"
        )
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print("wrote", out / f"{name}.pdf")


# ---------------- schematic helpers -----------------------------------------


def rounded_box(ax, xy, width, height, *, facecolor=LIGHT, edgecolor="#b7c6cf",
                linewidth=0.9, radius=0.025, zorder=1):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        clip_on=False,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MID, style="->", linewidth=1.1, linestyle="-"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "lw": linewidth,
            "linestyle": linestyle,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )


def assert_text_within(ax, artists, bounds, context, inset=0.0):
    """Fail figure generation when rendered text escapes its visual box."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    x, y, width, height = bounds
    allowed = (
        x + inset,
        y + inset,
        x + width - inset,
        y + height - inset,
    )
    for artist in artists:
        if not artist.get_text():
            continue
        actual = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        if (
            actual.x0 < allowed[0]
            or actual.y0 < allowed[1]
            or actual.x1 > allowed[2]
            or actual.y1 > allowed[3]
        ):
            raise ValueError(
                f"text escapes {context}: {artist.get_text()!r}; "
                f"actual=({actual.x0:.3f}, {actual.y0:.3f}, "
                f"{actual.x1:.3f}, {actual.y1:.3f}), "
                f"allowed={allowed}"
            )


def assert_text_avoids_bounds(ax, artists, bounds, context, pad=0.0):
    """Fail when prose labels intersect the visual encoding region."""
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    x, y, width, height = bounds
    forbidden = (
        x - pad,
        y - pad,
        x + width + pad,
        y + height + pad,
    )
    for artist in artists:
        if not artist.get_text():
            continue
        actual = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        intersects = not (
            actual.x1 <= forbidden[0]
            or actual.x0 >= forbidden[2]
            or actual.y1 <= forbidden[1]
            or actual.y0 >= forbidden[3]
        )
        if intersects:
            raise ValueError(
                f"text intersects {context}: {artist.get_text()!r}; "
                f"actual=({actual.x0:.3f}, {actual.y0:.3f}, "
                f"{actual.x1:.3f}, {actual.y1:.3f}), "
                f"forbidden={forbidden}"
            )


def draw_grid_notes(ax, x0, x1, y, *, note_x=None, note_color=BLUE,
                    grid_color="#cbd6dc", note_y=None, size=20):
    for x in np.linspace(x0, x1, 9):
        ax.plot([x, x], [y - 0.075, y + 0.075], color=grid_color, lw=0.6, zorder=1)
    if note_x is None:
        note_x = np.linspace(x0 + 0.02, x1 - 0.02, 6)
    note_x = np.asarray(note_x)
    if note_y is None:
        note_y = np.full_like(note_x, y, dtype=float)
    ax.scatter(note_x, note_y, s=size, color=note_color, edgecolor="white", lw=0.35, zorder=3)


# ---------------- fig0: first-page thesis graphic ---------------------------


def fig0_thesis(out, locale="en"):
    """Show the six evaluation questions without implying a shared input path."""
    labels = {
        "en": {
            "groups": ["MUSICAL FIT", "PLAY-RELEVANT CHARACTERISTICS"],
            "cards": [
                ("TIMING", "On the musical\ngrid?"),
                ("MUSIC RESPONSE", "Follows musical\nactivity?"),
                ("NOTE SEQUENCE", "Local note\npatterns?"),
                ("REPETITION &\nFORM", "Structure across\nsections?"),
                ("HUMAN-CHART\nDISTANCE", "Near same-level\ncharts?"),
                ("DIFFICULTY &\nLIMITS", "Typical and\nfeasible demands?"),
            ],
        },
        "zh-TW": {
            "groups": ["音樂契合度", "遊玩相關特性"],
            "cards": [
                ("時值對齊", "對齊音樂格線？"),
                ("音樂呼應", "呼應音樂變化？"),
                ("音符序列", "局部音符模式？"),
                ("重複與曲式", "跨段落的結構？"),
                ("人類譜面距離", "接近同難度人類譜？"),
                ("難度與限制", "操作需求典型且可行？"),
            ],
        },
    }[locale]

    fig, ax = plt.subplots(figsize=(6.9, 1.35))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()

    left = 0.01
    right = 0.99
    gap = 0.008
    card_width = (right - left - 5 * gap) / 6
    card_x = [left + i * (card_width + gap) for i in range(6)]
    group_bounds = [
        (left, 0.79, 2 * card_width + gap, 0.17),
        (card_x[2], 0.79, 4 * card_width + 3 * gap, 0.17),
    ]
    group_styles = [
        ("#f2f8fc", "#8bb4ca", BLUE),
        ("#faf7fc", "#b49bb8", PURPLE),
    ]
    group_artists = []
    for bounds, label, (face, edge, color) in zip(
        group_bounds, labels["groups"], group_styles
    ):
        rounded_box(ax, bounds[:2], bounds[2], bounds[3], facecolor=face,
                    edgecolor=edge, linewidth=0.9, radius=0.014)
        group_artists.append(tx(
            ax, bounds[0] + bounds[2] / 2, bounds[1] + bounds[3] / 2,
            label, locale=locale,
            bold=True, ha="center", va="center", fontsize=7.0, color=color,
        ))

    colors = [BLUE, TEAL, PURPLE, GOLD, VERMILLION, "#607d8b"]
    card_bounds = []
    card_texts = []
    icon_bounds = []
    for i, ((card_title, question), color) in enumerate(zip(labels["cards"], colors)):
        x = card_x[i]
        bounds = (x, 0.04, card_width, 0.70)
        card_bounds.append(bounds)
        icon_bounds.append((x + 0.012, 0.455, card_width - 0.024, 0.145))
        rounded_box(ax, bounds[:2], bounds[2], bounds[3], facecolor="white",
                    edgecolor=color, linewidth=1.0, radius=0.014)

        icon_x = x + card_width / 2
        if i == 0:  # notes relative to a metrical grid
            for gx in np.linspace(x + 0.025, x + card_width - 0.025, 5):
                ax.plot([gx, gx], [0.465, 0.585], color="#d5dfe4", lw=0.65)
            ax.scatter([x + 0.039, x + 0.078, x + 0.117], [0.525] * 3,
                       s=20, color=color, edgecolor="white", lw=0.3, zorder=3)
        elif i == 1:  # chart activity following musical activity
            xs = np.linspace(x + 0.026, x + card_width - 0.026, 5)
            heights = np.array([0.035, 0.075, 0.045, 0.095, 0.060])
            ax.bar(xs, heights, width=0.012, bottom=0.475, color="#bfe3d8",
                   edgecolor="none", zorder=2)
            ax.plot(xs, 0.49 + heights, "o-", color=color, ms=2.2, lw=0.9,
                    zorder=3)
        elif i == 2:  # local note transitions
            xs = np.linspace(x + 0.032, x + card_width - 0.032, 4)
            ys = [0.515, 0.555, 0.505, 0.55]
            ax.plot(xs, ys, color="#aa91ae", lw=0.9, zorder=2)
            ax.scatter(xs, ys, s=21, color=[PURPLE, BLUE, PURPLE, GOLD],
                       edgecolor="white", lw=0.3, zorder=3)
        elif i == 3:  # phrase-level return with variation
            phrase_colors = [GOLD, "#f3d69a", GOLD, "#f0c27a"]
            bx = x + 0.019
            total_width = card_width - 0.038
            segment_width = total_width / 4
            for j, (letter, fill) in enumerate(zip("ABAC", phrase_colors)):
                sx = bx + j * segment_width
                ax.add_patch(plt.Rectangle(
                    (sx + 0.002, 0.493), segment_width - 0.004, 0.064,
                    facecolor=fill, edgecolor="none", zorder=2.2,
                ))
                if j:
                    ax.plot([sx, sx], [0.493, 0.557], color=color, lw=0.45,
                            zorder=3)
                tx(ax, sx + segment_width / 2, 0.525, letter, bold=True,
                   ha="center", va="center", fontsize=5.1, color=INK,
                   zorder=4)
            ax.add_patch(FancyBboxPatch(
                (bx, 0.49), total_width, 0.07,
                boxstyle="round,pad=0,rounding_size=0.006",
                facecolor="none", edgecolor=color, linewidth=0.7, zorder=3,
            ))
        elif i == 4:  # generated point relative to human-chart cluster
            cluster_x = np.array([x + 0.038, x + 0.055, x + 0.064, x + 0.073,
                                  x + 0.084, x + 0.093])
            cluster_y = np.array([0.515, 0.55, 0.498, 0.535, 0.565, 0.515])
            ax.scatter(cluster_x, cluster_y, s=13, color="#d6a28b",
                       edgecolor="white", lw=0.25, zorder=3)
            generated = (x + 0.123, 0.56)
            nearest = np.argmin(
                (cluster_x - generated[0]) ** 2 + (cluster_y - generated[1]) ** 2
            )
            ax.plot(
                [cluster_x[nearest], generated[0]],
                [cluster_y[nearest], generated[1]],
                color=color, lw=0.8, ls="--",
            )
            ax.scatter(*generated, s=28, color=color, marker="D",
                       edgecolor="white", lw=0.35, zorder=3)
        else:  # demand profile relative to a typical band and hard limit
            ax.fill_between([x + 0.024, x + card_width - 0.024], 0.505, 0.555,
                            color="#dfe8ec", zorder=1)
            ax.plot([x + 0.024, x + card_width - 0.024], [0.57, 0.57],
                    color=VERMILLION, lw=0.8, ls="--")
            demand_x = np.linspace(x + 0.032, x + card_width - 0.032, 5)
            ax.plot(demand_x, [0.515, 0.53, 0.542, 0.522, 0.548], "o-",
                    color=color, ms=2.2, lw=0.9, zorder=3)

        title_artist_card = tx(
            ax, icon_x, 0.36, card_title, locale=locale, bold=True,
            ha="center", va="center", fontsize=6.35, color=INK,
        )
        question_artist = tx(
            ax, icon_x, 0.17, question, locale=locale,
            ha="center", va="center", fontsize=5.9, color=MID,
            linespacing=1.2,
        )
        card_texts.append((title_artist_card, question_artist))

    fig.tight_layout(pad=0.40)
    for i, (artist, bounds) in enumerate(zip(group_artists, group_bounds)):
        assert_text_within(ax, [artist], bounds, f"{locale} group {i + 1}", inset=0.004)
    for i, (artists, bounds) in enumerate(zip(card_texts, card_bounds)):
        assert_text_within(ax, list(artists), bounds,
                           f"{locale} dimension card {i + 1}", inset=0.004)
        assert_text_avoids_bounds(
            ax, list(artists), icon_bounds[i],
            f"{locale} dimension icon {i + 1}", pad=0.004,
        )
    save(fig, out, localized_name("fig0_thesis", locale))


# ---------------- figT: timing-reference schematic ----------------


def figT(out, locale="en"):
    """Show local re-pairing versus one chart-wide timing shift."""
    labels = {
        "en": {
            "local": "LOCAL MATCHING",
            "grid": "fixed grid",
            "notes": "notes +60 ms",
            "time": "time (ms)",
            "whole": "WHOLE-CHART OFFSET",
            "estimate": "estimate: +60 ms",
            "candidate": "candidate chart offset (ms)",
            "agreement": "grid agreement",
        },
        "zh-TW": {
            "local": "局部配對",
            "grid": "固定格線",
            "notes": "音符 +60 ms",
            "time": "時間（ms）",
            "whole": "整譜偏移",
            "estimate": "估計：+60 ms",
            "candidate": "候選整譜偏移（ms）",
            "agreement": "格線一致度",
        },
    }[locale]
    regular, strong = locale_fonts(locale)
    regular_kw = {"fontproperties": regular} if regular is not None else {}
    strong_kw = {"fontproperties": strong} if strong is not None else {}
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), gridspec_kw={"width_ratios": (1.2, 1)})

    ax = axes[0]
    ax.set_title(labels["local"], loc="left", color=BLUE, **strong_kw)
    fine = np.arange(0, 481, 30)
    major = np.arange(0, 481, 120)
    for x in fine:
        ax.plot([x, x], [0.62, 0.82], color="#d9e1e5", lw=0.55, zorder=0)
    for x in major:
        ax.plot([x, x], [0.58, 0.86], color="#8aa8bf", lw=1.1, zorder=0)
    original = np.array([0, 120, 240, 360, 480], dtype=float)
    shifted = original[:-1] + 60
    ax.scatter(original, np.full_like(original, 0.72), s=23, color=BLUE)
    ax.scatter(shifted, np.full_like(shifted, 0.32), s=27, color=VERMILLION,
               marker="D")
    for x in shifted:
        # Pairing is an undirected nearest-grid relation. A short connector
        # communicates the match without an unnecessary arrowhead.
        ax.plot([x, x], [0.38, 0.60], color="#777777", lw=0.7, zorder=1)
    ax.text(-82, 0.72, labels["grid"], ha="left", va="center",
            fontsize=6.8, color=BLUE, **regular_kw)
    ax.text(-82, 0.32, labels["notes"], ha="left", va="center",
            fontsize=6.8, color=VERMILLION, **regular_kw)
    ax.set(xlim=(-90, 500), ylim=(0, 1), yticks=[])
    ax.set_xlabel(labels["time"], **regular_kw)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    ax.set_title(labels["whole"], loc="left", color=TEAL, **strong_kw)
    shifts = np.linspace(-120, 120, 241)
    agreement = np.exp(-0.5 * ((shifts - 60) / 18) ** 2)
    agreement += 0.18 * np.exp(-0.5 * ((shifts + 60) / 25) ** 2)
    agreement /= agreement.max()
    ax.plot(shifts, agreement, color=BLUE, lw=1.8)
    ax.axvline(60, color=VERMILLION, lw=1, ls="--")
    ax.scatter([60], [1], color=VERMILLION, s=28, marker="D", zorder=3)
    ax.text(60, 1.02, labels["estimate"], ha="center", va="bottom",
            fontsize=7.2, bbox={"facecolor": "white", "edgecolor": "none",
                               "pad": 1.0, "alpha": 1.0}, **regular_kw)
    ax.set(xlim=(-120, 120), ylim=(0, 1.08), yticks=[])
    ax.set_xlabel(labels["candidate"], **regular_kw)
    ax.set_ylabel(labels["agreement"], **regular_kw)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.8, w_pad=1.3)
    save(fig, out, localized_name("figT_timing_alignment", locale))


# ---------------- figA: bidirectional blindness ----------------


def _net_series(records, metric, probe):
    from chartgeneval.plots import net_direction
    nd = net_direction(records, metric, probe)
    return [nd.get(d, float("nan")) for d in (1, 2, 3)]


def figA(probe_records, borrowed_records, out, locale="en"):
    """Grouped horizontal bars of per-chart net direction (doses 1-3)."""
    labels = {
        "en": {
            "c5": "C5 common-pattern rewrite",
            "lm": "LM loss (lower looks better)",
            "band": "pattern-IC band score",
            "fourgram": "4-gram repetition / uniqueness",
            "c4": "C4 loop collapse",
            "selfsim": "self-similarity (higher looks better)",
            "net": "Net direction vs. intact",
        },
        "zh-TW": {
            "c5": "C5 常見模式改寫",
            "lm": "LM loss（越低看似越好）",
            "band": "pattern-IC 範圍分數",
            "fourgram": "4-gram 重複／唯一性",
            "c4": "C4 循環崩解",
            "selfsim": "自相似度（越高看似越好）",
            "net": "相對原譜的淨方向",
        },
    }[locale]
    panels = [
        (labels["c5"], [
            (labels["lm"], BASE_C, _net_series(probe_records, "pattern_nll", "C5_blandification")),
            (labels["band"], OURS_C, _net_series(probe_records, "pattern_ic_adequacy_score", "C5_blandification")),
            (labels["fourgram"], TEAL,
             _net_series(probe_records, "repetition_adequacy_score", "C5_blandification")),
        ]),
        (labels["c4"], [
            (labels["selfsim"], BASE_C,
             _net_series(borrowed_records, "structureness_indicator_long", "C4_loop_collapse")),
            (labels["fourgram"], TEAL,
             _net_series(probe_records, "repetition_adequacy_score", "C4_loop_collapse")),
        ]),
    ]
    regular, strong = locale_fonts(locale)
    fig, axes = plt.subplots(2, 1, figsize=(3.9, 4.1), sharex=True)
    plt.rcParams.update({})
    for ax, (title, series) in zip(axes, panels):
        ax.axvline(0, color="#a9b4ba", lw=0.7)
        yticks, ylabels = [], []
        for i, (label, color, vals) in enumerate(series):
            base_y = -i * 1.0
            for d, v in enumerate(vals):
                ax.barh(base_y + (1 - d) * 0.26, v, height=0.24,
                        color=color, alpha=0.45 + 0.275 * d, zorder=2)
            yticks.append(base_y)
            ylabels.append(label)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7.2, fontproperties=regular)
        ax.set_xlim(-1.05, 1.05)
        ax.set_xticks([-1, 0, 1])
        ax.tick_params(axis="x", labelsize=7)
        ax.set_title(title, fontsize=8.2, loc="left", color=INK,
                     fontproperties=strong)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(left=False)
    fig.supxlabel(labels["net"], fontsize=8, y=0.025, color=INK,
                  fontproperties=regular)
    fig.tight_layout(rect=(0, 0.06, 1, 1), pad=0.8, h_pad=1.1)
    save(fig, out, localized_name("figA_complementarity", locale))


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
        display = "human reference" if s == "official" else display_name(s)
        label = f"{display} (n={len(rs)})"
        tier_rows.append({"buckets": B.mean(axis=0)})
        labels.append(label)
        strips[label] = np.array([r["clean_rate"] for r in rs if r.get("clean_rate") is not None])
    fig, ax = plot_timing_tiers(tier_rows, labels=labels, chart_clean_rates=strips)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.96, bottom=0.24, wspace=0.12)
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

    fig, ax = plt.subplots(figsize=(6.9, 3.25))
    vmax = 4.0
    clipped = np.ma.masked_invalid(np.clip(M, -vmax, vmax))
    x_edges = np.arange(len(probes) + 1) - 0.5
    y_edges = np.arange(len(metrics) + 1) - 0.5
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        clipped,
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "chartgeneval_diverging", [VERMILLION, "#fffdf9", BLUE]
        ),
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
        rasterized=False,
    )
    ax.set_ylim(len(metrics) - 0.5, -0.5)
    probe_labels = {
        "C1_timing_jitter": "C1\njitter",
        "C1s_sparse_jitter": "C1s\nsparse",
        "C2_anchor_shift": "C2\nshift",
        "C3_type_shuffle": "C3\ntypes",
        "C4_loop_collapse": "C4\nloop",
        "C5_blandification": "C5\ncommon",
        "C6_density_scale": "C6\nrate",
        "C7_burst_insert": "C7\nburst",
        "C8_bar_shuffle": "C8\nshuffle",
    }
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([probe_labels.get(p, p) for p in probes], fontsize=6.2)
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
    ax.set_yticklabels([plain_labels.get(m, m) for m in metrics], fontsize=6.2)
    for i, m in enumerate(metrics):
        for j, p in enumerate(probes):
            if (m, p) in targets:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="black", lw=1.6))
                # A signed value inside each selected-target cell provides a
                # second cue that remains legible in grayscale and across
                # color-vision differences.
                val = cells[m].get(p)
                if val is not None:
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                            fontsize=5.6, fontweight="bold",
                            color="white" if abs(val) > 1.6 else "#1a1a1a")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor("face")
    cb.set_label("Change at strongest corruption (human SD)")
    fig.subplots_adjust(left=0.27, right=0.90, bottom=0.16, top=0.97)
    save(fig, out, "figC_coupling")


# ---------------- figD: system profile matrix + feasibility checks ----------


def figD(probe_records, ext_records, profiles, out):
    """Three-system annotated heatmap with feasibility-check glyphs."""
    from chartgeneval.plots import plot_profile
    cols = [
        ("Timing\nwithin 6 ms", None),
        ("Note\nrate", "density_adequacy_score"),
        ("Short\nintervals", "strain_adequacy_score"),
        ("Familiar\ntransitions", "transition_validity_score"),
        ("Local\npatterns", "pattern_ic_adequacy_score"),
        ("4-gram\nrepetition /\nuniqueness", "repetition_adequacy_score"),
    ]
    con_keys = [
        ("overload", "overload_score"),
        ("density\nspike", "density_spike_score"),
        ("playability", "playability_proxy_score"),
    ]
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
        display = "human reference" if sys_ == "official" else display_name(sys_)
        scores[display] = row
        constraints[display] = {
            name: (med(chartside[sys_], key) or 0) >= 0.5
            for name, key in con_keys
        }
    # Draw into a fixed ~text-width canvas so the rendered figure is close to
    # 1:1 in the paper and its cell text stays legible (roughly 8 pt) instead
    # of being downscaled from an oversized auto-sized figure.
    fig, ax = plt.subplots(figsize=(6.9, 2.25))
    plot_profile(
        scores,
        columns=[c for c, _ in cols],
        constraints=constraints,
        constraint_order=[name for name, _ in con_keys],
        outline_failed_rows=False,
        ax=ax,
    )
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.20, top=0.94)
    save(fig, out, "figD_profile_matrix")


# ---------------- figE: exploratory difficulty slice -------------------------


DIFFICULTY_PLOT_METRICS = [
    ("note-rate score", "density_adequacy_score"),
    ("short-interval score", "strain_adequacy_score"),
    ("familiar transitions", "transition_validity_score"),
    ("local patterns", "pattern_ic_adequacy_score"),
    ("4-gram repetition/uniqueness", "repetition_adequacy_score"),
]

# Preserve both prespecified output names in the released audit table while
# plotting their algebraically equivalent score only once.
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

    fig = plt.figure(figsize=(6.9, 3.15), layout="constrained")
    fig.get_layout_engine().set(w_pad=5 / 72, h_pad=5 / 72)
    gs = fig.add_gridspec(1, 3, width_ratios=(1.1, 2.45, 1.1), wspace=0.20)
    x = np.arange(len(courses))

    ax = fig.add_subplot(gs[0, 0])
    ratio_q = np.array([
        [r["note_rate_ratio_p10"], r["note_rate_ratio_median"], r["note_rate_ratio_p90"]]
        for r in summary
    ])
    ax.axhline(1, color="#777777", lw=0.8, ls="--")
    ax.vlines(x, ratio_q[:, 0], ratio_q[:, 2], color=OURS_C, lw=2)
    ax.scatter(x, ratio_q[:, 1], color=OURS_C, s=26, zorder=3)
    ax.set_title("(a) NOTE-RATE RATIO", loc="left", color=BLUE)
    ax.set(xticks=x, xticklabels=short_labels, xlabel="course", ylabel="ratio",
           ylim=(0.6, max(3.2, ratio_q[:, 2].max() + 0.1)))
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = fig.add_subplot(gs[0, 1])
    matrix = np.array([[r[metric] for _, metric in DIFFICULTY_PLOT_METRICS] for r in summary])
    x_edges = np.arange(matrix.shape[1] + 1) - 0.5
    y_edges = np.arange(matrix.shape[0] + 1) - 0.5
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=1,
        shading="flat",
        rasterized=False,
    )
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)
    ax.set_title("(b) CALIBRATED SCORES", loc="left", color=PURPLE)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{label}  n={row['n']}" for label, row in zip(labels, summary)])
    ax.set_xticks(np.arange(len(DIFFICULTY_PLOT_METRICS)))
    score_labels = [
        "note\nrate",
        "short\nIOI",
        "transition\nfamiliarity",
        "pattern\nfit",
        "4-gram\nfit",
    ]
    ax.set_xticklabels(score_labels, rotation=0, ha="center", fontsize=6.1,
                       linespacing=1.2)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="white" if matrix[i, j] > 0.55 else "#1a1a1a")
    ax = fig.add_subplot(gs[0, 2])
    timing_q = np.array([
        [r["timing_clean_p10"], r["timing_clean_median"], r["timing_clean_p90"]]
        for r in summary
    ])
    ax.vlines(x, timing_q[:, 0], timing_q[:, 2], color=BASE_C, lw=2)
    ax.scatter(x, timing_q[:, 1], color=BASE_C, s=26, zorder=3)
    ax.set_title("(c) TIMING", loc="left", color=VERMILLION,
                 fontsize=7.8)
    ax.set(xticks=x, xticklabels=short_labels, xlabel="course",
           ylabel="fraction of notes", ylim=(0, 1.02))
    ax.grid(axis="y", color="#e6e6e6", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.align_ylabels()

    save(fig, out, "figE_difficulty")


# ---------------- figM: metric-to-construct visual guide --------------------


def _metric_panel_visual(ax, index):
    if index == 0:  # timing
        for x in np.linspace(0.12, 0.88, 9):
            ax.plot([x, x], [0.34, 0.64], color="#cad5db", lw=0.7)
        xs = np.array([0.16, 0.34, 0.50, 0.67, 0.84])
        ax.scatter(xs, np.full_like(xs, 0.49), s=24, color=BLUE, zorder=3)
        ax.scatter(xs + 0.045, np.full_like(xs, 0.39), s=18, color=VERMILLION,
                   marker="x", zorder=3)
    elif index == 1:  # sequence
        xs = np.linspace(0.16, 0.84, 5)
        colors = [BLUE, GOLD, BLUE, TEAL, GOLD]
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(ax, (a + 0.03, 0.49), (b - 0.03, 0.49), color="#9ca8af", linewidth=0.8)
        ax.scatter(xs, np.full_like(xs, 0.49), s=55, color=colors,
                   edgecolor="white", lw=0.6, zorder=3)
    elif index == 2:  # form
        labels = ["A", "B", "A", "B"]
        for i, label in enumerate(labels):
            x = 0.12 + i * 0.20
            rounded_box(ax, (x, 0.43), 0.15, 0.16, facecolor="white",
                        edgecolor=[BLUE, GOLD, BLUE, GOLD][i], radius=0.015)
            ax.text(x + 0.075, 0.51, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", color=INK)
    elif index == 3:  # music response
        xs = np.linspace(0.10, 0.90, 100)
        # Two aligned tracks avoid implying that one signal occludes the other.
        energy = 0.56 + 0.045 * np.sin(xs * 9) + 0.020 * np.sin(xs * 23)
        ax.plot(xs, energy, color=TEAL, lw=1.4)
        bars_x = np.linspace(0.14, 0.86, 9)
        heights = 0.045 + 0.075 * (np.sin(bars_x * 9) + 1) / 2
        ax.bar(bars_x, heights, width=0.04, bottom=0.32, color=SKY, alpha=0.85)
    elif index == 4:  # human distance
        cloud = np.array([
            [0.25, 0.48], [0.32, 0.56], [0.38, 0.43], [0.45, 0.54],
            [0.50, 0.46], [0.56, 0.57], [0.61, 0.42], [0.67, 0.50],
        ])
        ax.scatter(cloud[:, 0], cloud[:, 1], s=24, color=SKY, alpha=0.85)
        generated = np.array([0.84, 0.60])
        ax.scatter([generated[0]], [generated[1]], s=46, color=VERMILLION,
                   marker="D", zorder=3)
        nearest = np.argmin(np.sum((cloud - generated) ** 2, axis=1))
        ax.plot(
            [cloud[nearest, 0], generated[0]],
            [cloud[nearest, 1], generated[1]],
            color=VERMILLION, lw=0.9, ls="--",
        )
    else:  # difficulty and limits
        ax.fill_between([0.12, 0.88], [0.42, 0.42], [0.58, 0.58], color="#dcefe9")
        xs = np.linspace(0.16, 0.84, 9)
        vals = np.array([0.48, 0.52, 0.47, 0.55, 0.50, 0.54, 0.49, 0.64, 0.51])
        ordinary = np.arange(len(xs)) != len(xs) - 2
        ax.scatter(xs[ordinary], vals[ordinary], color=BLUE, s=18, zorder=3)
        ax.scatter([xs[-2]], [vals[-2]], color=VERMILLION, s=28, zorder=3)


def figM(out, locale="en"):
    """Map each construct to its question, representative output, and evidence."""
    text_linespacing = 1.4 if locale == "zh-TW" else 1.2
    panels = {
        "en": [
            ("Timing", "Are notes aligned\nto musical time?",
             "clean fraction | residual tail\nchart-wide offset", "held-out subset"),
            ("Note sequence", "Are local transitions\nfamiliar?",
             "LM fit | transition familiarity\npattern typicality", "held-out subset"),
            ("Repetition & form", "Does material return\nwithout stagnating?",
             "4-gram typicality | reciprocity\nstagnation-alienation", "held-out subset"),
            ("Music response", "Does chart activity\nfollow the song?",
             "density-energy response\nonset support", "development"),
            ("Human-chart distance", "How far is the generated chart\nfrom same-course\nhuman charts?",
             "32-D nearest-neighbor gap", "development"),
            ("Difficulty & limits", "Are chart demands typical\nfor the requested course and\nwithin declared limits?",
             "rate/load typicality\noverload and spike checks", "held-out subset"),
        ],
        "zh-TW": [
            ("時值對齊", "音符是否對齊\n音樂時間？",
             "乾淨比例 | 殘差尾端\n整譜位移", "留出子集"),
            ("音符序列", "局部轉移是否\n符合熟悉模式？",
             "LM 配適 | 轉移熟悉度\n模式典型度", "留出子集"),
            ("重複與曲式", "素材是否回返\n而不陷入停滯？",
             "4-gram 典型度 | 交互回應\n停滯-疏離", "留出子集"),
            ("音樂呼應", "譜面活動是否\n跟隨歌曲？",
             "密度-能量回應\n起音支持", "發展組"),
            ("人類譜面距離", "生成譜與同難度人類譜\n相距多遠？",
             "32 維最近鄰距離", "發展組"),
            ("難度與限制", "譜面負荷是否符合指定難度\n並落在宣告限制內？",
             "速率/負荷典型度\n過載與尖峰檢查", "留出子集"),
        ],
    }[locale]

    fig, axes = plt.subplots(2, 3, figsize=(6.9, 4.45))
    colors = [
        DIMENSION_COLORS["timing"],
        DIMENSION_COLORS["sequence"],
        DIMENSION_COLORS["form"],
        DIMENSION_COLORS["music"],
        DIMENSION_COLORS["human"],
        DIMENSION_COLORS["difficulty"],
    ]
    evidence_artists = []
    copy_artists = []
    for i, (ax, (title, question, outputs, evidence)) in enumerate(zip(axes.flat, panels)):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        rounded_box(ax, (0.015, 0.04), 0.97, 0.92, facecolor="#fbfcfd",
                    edgecolor=colors[i], linewidth=1.0, radius=0.025)
        title_artist = tx(
            ax, 0.06, 0.87, title, locale=locale, bold=True,
            ha="left", va="center", fontsize=8.5, color=colors[i],
        )
        question_artist = tx(
            ax, 0.06, 0.75, question, locale=locale,
            ha="left", va="center", fontsize=6.2, color=INK,
            linespacing=text_linespacing,
        )
        _metric_panel_visual(ax, i)
        output_artist = tx(
            ax, 0.06, 0.18, outputs, locale=locale,
            ha="left", va="center", fontsize=5.7, color=INK,
            linespacing=text_linespacing,
        )
        held_out = "held-out" in evidence or "留出" in evidence
        rounded_box(ax, (0.60, 0.075), 0.36, 0.10,
                    facecolor="#e8f4f1" if held_out else "white",
                    edgecolor=TEAL if held_out else MID,
                    linewidth=0.9, radius=0.012)
        evidence_artists.append(tx(
            ax, 0.78, 0.125, evidence, locale=locale, bold=True,
            ha="center", va="center", fontsize=5.7,
            color=TEAL if held_out else MID,
        ))
        copy_artists.append((title_artist, question_artist, output_artist))
    fig.tight_layout(pad=0.35, h_pad=0.55, w_pad=0.55)
    for i, (ax, evidence_artist, panel_copy) in enumerate(
        zip(axes.flat, evidence_artists, copy_artists)
    ):
        assert_text_within(
            ax,
            [artist for artist in ax.texts if artist.get_text()],
            (0.015, 0.04, 0.97, 0.92),
            f"{locale} metric panel {i + 1}",
            inset=0.008,
        )
        assert_text_within(
            ax,
            [evidence_artist],
            (0.60, 0.075, 0.36, 0.10),
            f"{locale} metric evidence badge {i + 1}",
            inset=0.004,
        )
        assert_text_avoids_bounds(
            ax,
            [*panel_copy, evidence_artist],
            (0.08, 0.28, 0.84, 0.39),
            f"{locale} metric visual {i + 1}",
            pad=0.008,
        )
    save(fig, out, localized_name("figM_metric_constructs", locale))


# ---------------- figP: controlled-corruption atlas -------------------------


def _corruption_visual(ax, probe):
    if probe in {"C1_timing_jitter", "C1s_sparse_jitter", "C2_anchor_shift"}:
        xs = np.linspace(0.23, 0.87, 6)
        for x in np.linspace(0.19, 0.91, 13):
            ax.plot([x, x], [0.31, 0.68], color="#d7dfe4", lw=0.55)
        ax.scatter(xs, np.full_like(xs, 0.60), s=20, color=BLUE, zorder=3)
        if probe == "C1_timing_jitter":
            edited = xs + np.array([-0.025, 0.020, -0.018, 0.030, -0.024, 0.015])
        elif probe == "C1s_sparse_jitter":
            edited = xs.copy()
            edited[4] += 0.085
        else:
            edited = xs + 0.055
        ax.scatter(edited, np.full_like(edited, 0.39), s=22, color=VERMILLION,
                   marker="x", linewidth=1.2, zorder=3)
    elif probe == "C3_type_shuffle":
        xs = np.linspace(0.23, 0.87, 6)
        before = [BLUE, GOLD, BLUE, GOLD, GOLD, BLUE]
        after = [GOLD, BLUE, GOLD, BLUE, GOLD, GOLD]
        sizes = [24, 24, 42, 24, 42, 42]
        ax.scatter(xs, np.full_like(xs, 0.60), s=sizes, color=before,
                   edgecolor="white", lw=0.5)
        ax.scatter(xs, np.full_like(xs, 0.39), s=sizes, color=after,
                   edgecolor="white", lw=0.5)
    elif probe in {"C4_loop_collapse", "C5_blandification", "C8_bar_shuffle"}:
        if probe == "C4_loop_collapse":
            before, after = ["ABCD", "EFGH"], ["ABCD", "ABCD"]
        elif probe == "C5_blandification":
            before, after = ["A", "C", "D", "B"], ["A", "B", "D", "C"]
        else:
            before, after = ["1", "2", "3", "4"], ["3", "1", "4", "2"]
        for row_y, labels, color in ((0.54, before, BLUE), (0.33, after, VERMILLION)):
            width = 0.29 if len(labels) == 2 else 0.14
            x0 = 0.23 if len(labels) == 2 else 0.21
            step = 0.35 if len(labels) == 2 else 0.18
            for i, label in enumerate(labels):
                x = x0 + i * step
                rounded_box(ax, (x, row_y), width, 0.12, facecolor="white",
                            edgecolor=color, radius=0.012)
                ax.text(x + width / 2, row_y + 0.06, label, ha="center", va="center",
                        fontsize=7, fontweight="bold", color=color)
    elif probe == "C6_density_scale":
        before = np.linspace(0.23, 0.87, 5)
        after = np.linspace(0.21, 0.89, 10)
        ax.scatter(before, np.full_like(before, 0.60), s=22, color=BLUE)
        ax.scatter(after, np.full_like(after, 0.39), s=18, color=VERMILLION, marker="x")
    elif probe == "C7_burst_insert":
        before = np.linspace(0.23, 0.87, 6)
        burst = np.concatenate([before, np.linspace(0.51, 0.66, 7)])
        ax.scatter(before, np.full_like(before, 0.60), s=20, color=BLUE)
        ax.scatter(burst, np.full_like(burst, 0.39), s=10, color=VERMILLION,
                   marker="x", linewidth=0.75)


def figP(contract, out, locale="en"):
    """Render the nine contract-defined corruptions as before/after panels."""
    text_linespacing = 1.4 if locale == "zh-TW" else 1.2
    order = [
        "C1_timing_jitter", "C1s_sparse_jitter", "C2_anchor_shift",
        "C3_type_shuffle", "C4_loop_collapse", "C5_blandification",
        "C6_density_scale", "C7_burst_insert", "C8_bar_shuffle",
    ]
    if set(contract["probes"]) != set(order):
        raise ValueError("corruption atlas and confirmatory contract disagree on probe IDs")
    by_probe = defaultdict(list)
    for pair in contract["primary_pairs"]:
        by_probe[pair["probe"]].append(pair["metric"])
    if set(by_probe) != set(order):
        raise ValueError("every corruption must have at least one primary pair")

    metric_names = {
        "en": {
            "timing.clean_rate": "timing clean fraction",
            "timing.absolute_error_p99_ms": "timing-error p99",
            "timing.grid_phase_offset_abs_ms": "chart-wide phase offset",
            "transition_validity_score": "transition familiarity",
            "repetition_adequacy_score": "4-gram typicality",
            "surface_variety_adequacy_score": "4-gram typicality",
            "density_adequacy_score": "note-rate typicality",
            "density_spike_score": "density-spike limit",
        },
        "zh-TW": {
            "timing.clean_rate": "時值乾淨比例",
            "timing.absolute_error_p99_ms": "時值誤差 p99",
            "timing.grid_phase_offset_abs_ms": "整譜相位位移",
            "transition_validity_score": "轉移熟悉度",
            "repetition_adequacy_score": "4-gram 典型度",
            "surface_variety_adequacy_score": "4-gram 典型度",
            "density_adequacy_score": "音符速率典型度",
            "density_spike_score": "密度尖峰限制",
        },
    }[locale]
    titles = {
        "en": {
            "C1_timing_jitter": "C1  dense timing jitter",
            "C1s_sparse_jitter": "C1s  sparse 60 ms errors",
            "C2_anchor_shift": "C2  whole-chart shift",
            "C3_type_shuffle": "C3  note-type shuffle",
            "C4_loop_collapse": "C4  loop collapse",
            "C5_blandification": "C5  common-pattern rewrite",
            "C6_density_scale": "C6  note-rate scaling",
            "C7_burst_insert": "C7  local burst insertion",
            "C8_bar_shuffle": "C8  bar-order shuffle",
        },
        "zh-TW": {
            "C1_timing_jitter": "C1  密集時值抖動",
            "C1s_sparse_jitter": "C1s  稀疏 60 ms 誤差",
            "C2_anchor_shift": "C2  整譜平移",
            "C3_type_shuffle": "C3  音符類型重排",
            "C4_loop_collapse": "C4  循環崩塌",
            "C5_blandification": "C5  常見模式改寫",
            "C6_density_scale": "C6  音符速率縮放",
            "C7_burst_insert": "C7  局部密集插入",
            "C8_bar_shuffle": "C8  小節順序重排",
        },
    }[locale]
    before_after = ("intact", "edited") if locale == "en" else ("原始", "改動後")
    target_prefix = "target" if locale == "en" else "目標"

    border_colors = {
        "C1_timing_jitter": DIMENSION_COLORS["timing"],
        "C1s_sparse_jitter": DIMENSION_COLORS["timing"],
        "C2_anchor_shift": DIMENSION_COLORS["timing"],
        "C3_type_shuffle": DIMENSION_COLORS["sequence"],
        "C4_loop_collapse": DIMENSION_COLORS["form"],
        "C5_blandification": DIMENSION_COLORS["form"],
        "C6_density_scale": DIMENSION_COLORS["difficulty"],
        "C7_burst_insert": DIMENSION_COLORS["difficulty"],
        "C8_bar_shuffle": DIMENSION_COLORS["form"],
    }
    fig, axes = plt.subplots(3, 3, figsize=(6.9, 5.45))
    copy_artists = []
    for ax, probe in zip(axes.flat, order):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_axis_off()
        rounded_box(ax, (0.015, 0.035), 0.97, 0.93, facecolor="#fbfcfd",
                    edgecolor=border_colors[probe], linewidth=1.0, radius=0.025)
        title_artist = tx(
            ax, 0.06, 0.87, titles[probe], locale=locale, bold=True,
            ha="left", va="center", fontsize=7.4, color=INK,
        )
        intact_artist = tx(
            ax, 0.06, 0.60, before_after[0], locale=locale,
            ha="left", va="center", fontsize=5.5, color=BLUE,
        )
        edited_artist = tx(
            ax, 0.06, 0.39, before_after[1], locale=locale,
            ha="left", va="center", fontsize=5.5, color=VERMILLION,
        )
        _corruption_visual(ax, probe)
        targets = []
        for metric in by_probe[probe]:
            name = metric_names[metric]
            if name not in targets:
                targets.append(name)
        target = " + ".join(targets)
        if probe == "C5_blandification":
            target += "\n(one effective axis)" if locale == "en" else "\n（單一有效軸）"
        target_artist = tx(
            ax, 0.50, 0.13, f"{target_prefix}: {target}",
            locale=locale, bold=True, ha="center", va="center",
            fontsize=5.8, color=TEAL, linespacing=text_linespacing,
        )
        copy_artists.append(
            (title_artist, intact_artist, edited_artist, target_artist)
        )
    fig.tight_layout(pad=0.35, h_pad=0.55, w_pad=0.55)
    for i, (ax, panel_copy) in enumerate(zip(axes.flat, copy_artists)):
        assert_text_within(
            ax,
            [artist for artist in ax.texts if artist.get_text()],
            (0.015, 0.035, 0.97, 0.93),
            f"{locale} corruption panel {i + 1}",
            inset=0.008,
        )
        assert_text_avoids_bounds(
            ax,
            list(panel_copy),
            (0.18, 0.28, 0.76, 0.43),
            f"{locale} corruption visual {i + 1}",
            pad=0.006,
        )
    save(fig, out, localized_name("figP_corruption_atlas", locale))


# ---------------- figCC: human-reference corpus coverage --------------------


def figCC(coverage, out, locale="en"):
    """Coverage of the human reference bands along the two axes the source
    metadata carries: star level per course and tempo. Rendered from derived
    aggregate counts, so it needs no chart content."""
    regular, strong = locale_fonts(locale)

    def _title(ax, text, color):
        kw = {"loc": "left", "color": color}
        if strong is not None:
            kw["fontproperties"] = strong
        else:
            kw["fontweight"] = "bold"
        ax.set_title(text, **kw)

    def _axis_label(ax, xlabel=None, ylabel=None):
        kw = {"fontproperties": regular} if regular is not None else {}
        if xlabel is not None:
            ax.set_xlabel(xlabel, **kw)
        if ylabel is not None:
            ax.set_ylabel(ylabel, **kw)

    text = {
        "en": {
            "t_a": "(a) star level per course",
            "t_b": "(b) tempo distribution",
            "x_a": "star level",
            "x_b": "beats per minute",
            "y_b": "charts",
            "note": "median {med:.0f} BPM\nrange {lo:.0f}–{hi:.0f}",
        },
        "zh-TW": {
            "t_a": "(a) 各 course 的星級分布",
            "t_b": "(b) 速度（BPM）分布",
            "x_a": "星級",
            "x_b": "每分鐘拍數",
            "y_b": "譜面數",
            "note": "中位 {med:.0f} BPM\n範圍 {lo:.0f}–{hi:.0f}",
        },
    }[locale]

    courses = ["easy", "normal", "hard", "oni", "ura"]
    course_labels = ["Easy", "Normal", "Hard", "Oni", "Ura"]
    course_data = coverage["courses"]
    max_level = max(int(k) for c in courses for k in course_data[c]["level_counts"])
    levels = list(range(1, max_level + 1))
    matrix = np.array(
        [[course_data[c]["level_counts"].get(str(lv), 0) for lv in levels] for c in courses],
        dtype=float,
    )

    fig = plt.figure(figsize=(6.9, 2.5))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.6, 1.0))

    ax = fig.add_subplot(gs[0, 0])
    vmax = matrix.max()
    x_edges = np.arange(len(levels) + 1) - 0.5
    y_edges = np.arange(len(courses) + 1) - 0.5
    ax.pcolormesh(x_edges, y_edges, matrix, cmap="Blues", vmin=0, vmax=vmax,
                  shading="flat", rasterized=False)
    ax.set_ylim(len(courses) - 0.5, -0.5)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([str(lv) for lv in levels], fontsize=7.5)
    ax.set_yticks(range(len(courses)))
    ax.set_yticklabels(
        [f"{lab}  n={int(course_data[c]['n_charts'])}"
         for lab, c in zip(course_labels, courses)]
    )
    ax.tick_params(length=0)
    for i in range(len(courses)):
        for j in range(len(levels)):
            v = int(matrix[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5,
                        color="white" if matrix[i, j] > vmax * 0.55 else "#1a1a1a")
    _title(ax, text["t_a"], BLUE)
    _axis_label(ax, xlabel=text["x_a"])

    ax2 = fig.add_subplot(gs[0, 1])
    edges = coverage["bpm_bin_edges"]
    counts = coverage["bpm_hist_all"]
    centers = [(edges[k] + edges[k + 1]) / 2 for k in range(len(counts))]
    ax2.bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color=SKY,
            edgecolor="white", lw=0.4)
    summary = coverage["bpm_summary_all"]
    ax2.axvline(summary["p50"], color=VERMILLION, lw=1.0, ls="--")
    note = ax2.text(
        0.96, 0.95,
        text["note"].format(med=summary["p50"], lo=summary["min"], hi=summary["max"]),
        transform=ax2.transAxes, ha="right", va="top", fontsize=6.8,
        color=INK, linespacing=1.3,
    )
    if regular is not None:
        note.set_fontproperties(regular)
    _title(ax2, text["t_b"], TEAL)
    _axis_label(ax2, xlabel=text["x_b"], ylabel=text["y_b"])
    ax2.set_xlim(edges[0], edges[-1])
    ax2.grid(axis="y", color="#e6e6e6", lw=0.6)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    fig.subplots_adjust(left=0.16, right=0.955, bottom=0.19, top=0.88, wspace=0.32)
    save(fig, out, localized_name("figCC_corpus_coverage", locale))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifacts-root",
        default=str(Path(__file__).resolve().parents[1] / "artifacts"),
    )
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "paper" / "figs"))
    ap.add_argument(
        "--run-config",
        default=str(
            Path(__file__).resolve().parent
            / "confirmatory_holdout_v1"
            / "configs"
            / "confirmatory.json"
        ),
        help="validated confirmatory run config used to locate the atlas contract",
    )
    ap.add_argument(
        "--cjk-font",
        default=None,
        help="path to a Traditional Chinese variable font with a wght axis",
    )
    ap.add_argument(
        "--only",
        default=None,
        help="comma list: fig0,figT,figA,figB,figC,figD,figE,figM,figP,figCC",
    )
    args = ap.parse_args()

    figure_names = {
        "fig0", "figT", "figA", "figB", "figC", "figD", "figE",
        "figM", "figP", "figCC",
    }
    if args.only is None:
        only = figure_names
    else:
        if not args.only.strip():
            ap.error("--only requires at least one figure name")
        requested = [name.strip() for name in args.only.split(",")]
        if any(not name for name in requested):
            ap.error("--only contains an empty figure name")
        unknown = sorted(set(requested) - figure_names)
        if unknown:
            ap.error(f"unknown figure name(s) for --only: {', '.join(unknown)}")
        duplicates = sorted({name for name in requested if requested.count(name) > 1})
        if duplicates:
            ap.error(f"duplicate figure name(s) for --only: {', '.join(duplicates)}")
        only = set(requested)

    if args.cjk_font is not None:
        configure_cjk_font(args.cjk_font)

    root = Path(args.artifacts_root)
    records = root / "records"
    probe_rec = records / "corruption_probes_development.jsonl"
    ext_rec = records / "system_timing_clean.jsonl"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def want(name):
        return name in only

    if want("fig0"):
        fig0_thesis(out, "en")
        fig0_thesis(out, "zh-TW")
    if want("figT"):
        figT(out, "en")
        figT(out, "zh-TW")
    if want("figA"):
        probe_rows = jrows(probe_rec)
        borrowed_rows = jrows(records / "corruption_borrowed_metrics_clean.jsonl")
        figA(probe_rows, borrowed_rows, out, "en")
        figA(probe_rows, borrowed_rows, out, "zh-TW")
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
    if want("figM"):
        figM(out, "en")
        figM(out, "zh-TW")
    if want("figP"):
        _config, contract, _config_path, _contract_path = load_contract(args.run_config)
        figP(contract, out, "en")
        figP(contract, out, "zh-TW")
    if want("figCC"):
        coverage = json.loads(
            (records / "corpus_coverage.json").read_text(encoding="utf-8")
        )
        figCC(coverage, out, "en")
        figCC(coverage, out, "zh-TW")


if __name__ == "__main__":
    main()
