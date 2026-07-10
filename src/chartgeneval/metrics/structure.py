"""Structure family: surface variety, repetition, boredom, colour, reciprocity.

Covers the long-range organisation of a chart: 4-gram surface variety,
repetition rate, windowed boredom, colour-switch adequacy, plus the adopted
v2 candidate ``call_response_reciprocity`` (call-and-response phrase structure).
Band scores need ``ctx['calibration']``; reciprocity needs ``ctx['grid']``.
"""

from __future__ import annotations

import numpy as np

from ..calibration import score_chart_features
from ..events import sorted_hits
from .common import compute_chart_features, fold_hit
from .grid_utils import jaccard, resolve_grid

METRIC_NAME = "structure"

# --- adopted v2 candidate: call_response_reciprocity ---
TAU_R = 0.6
COLOR_LO, COLOR_HI = 0.3, 0.9
PHRASE_BARS = 2
SUBDIV = 4
MIN_PHRASES = 4


def _phrase_slots(hits, p_start, p_end, subdiv, meter):
    phrase_len = p_end - p_start
    if phrase_len <= 0:
        return set(), {}
    n_slots = int(round(subdiv * meter * PHRASE_BARS))
    if n_slots <= 0:
        return set(), {}
    slot_colors = {}
    slot_set = set()
    for t, cls in hits:
        if t < p_start or t >= p_end:
            continue
        frac = (t - p_start) / phrase_len
        slot = int(round(frac * n_slots))
        if slot < 0 or slot > n_slots:
            continue
        slot_set.add(slot)
        slot_colors.setdefault(slot, []).append(fold_hit(cls))
    main = {}
    for slot, cols in slot_colors.items():
        d = cols.count("D")
        k = cols.count("K")
        main[slot] = "D" if d >= k else "K"
    return slot_set, main


def call_response_reciprocity(events, ctx):
    """Raw call-and-response and reciprocity values (adopted v2 candidate)."""
    out = {
        "call_response_rate": float("nan"),
        "reciprocity": float("nan"),
        "n_phrases": float("nan"),
        "mean_rhythm_similarity": float("nan"),
        "mean_color_contrast": float("nan"),
    }
    hits = sorted_hits(events)
    if len(hits) < 4:
        return out
    g = resolve_grid(ctx)
    if g is None:
        return out
    downbeats, beat_period, meter = g
    dbs = list(downbeats)
    bounds = dbs[::PHRASE_BARS]
    if len(bounds) < 2:
        return out
    phrases = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    if len(phrases) < MIN_PHRASES:
        out["n_phrases"] = float(len(phrases))
        return out
    out["n_phrases"] = float(len(phrases))

    reps = [_phrase_slots(hits, s0, s1, SUBDIV, meter) for (s0, s1) in phrases]

    cr_events = 0
    n_pairs = 0
    rsims = []
    ccontrasts = []
    for i in range(len(reps) - 1):
        s_a, c_a = reps[i]
        s_b, c_b = reps[i + 1]
        if not s_a and not s_b:
            continue
        n_pairs += 1
        rsim = jaccard(s_a, s_b)
        rsims.append(rsim)
        aligned = s_a & s_b
        if aligned:
            flips = sum(1 for slot in aligned if c_a.get(slot) != c_b.get(slot))
            ccon = flips / len(aligned)
            ccontrasts.append(ccon)
        else:
            ccon = None
        if rsim > TAU_R and ccon is not None and COLOR_LO <= ccon <= COLOR_HI:
            cr_events += 1

    if n_pairs > 0:
        out["call_response_rate"] = cr_events / n_pairs
    if rsims:
        out["mean_rhythm_similarity"] = float(np.mean(rsims))
    if ccontrasts:
        out["mean_color_contrast"] = float(np.mean(ccontrasts))

    labels = []
    for (ss, mc) in reps:
        rhash = hash(frozenset(ss))
        d = sum(1 for v in mc.values() if v == "D")
        k = sum(1 for v in mc.values() if v == "K")
        maincol = "D" if d >= k else "K"
        labels.append((rhash, maincol))
    if len(labels) >= 3:
        alt = 0
        denom = 0
        for i in range(len(labels) - 2):
            denom += 1
            if labels[i] == labels[i + 2] and labels[i] != labels[i + 1]:
                alt += 1
        out["reciprocity"] = alt / denom if denom else float("nan")
    return out


RAW_KEYS = (
    "unique_4gram_rate",
    "repeat_4gram_rate",
    "color_switch_rate",
    "boredom_rate",
    "local_nps_cv",
)
SCORE_KEYS = (
    "surface_variety_adequacy_score",
    "repetition_adequacy_score",
    "color_switch_adequacy_score",
    "density_variation_adequacy_score",
    "boredom_score",
    "surface_structure_proxy_score",
)


def compute(events, ctx):
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    course = ctx.get("course") if isinstance(ctx, dict) else None
    calibration = ctx.get("calibration") if isinstance(ctx, dict) else None
    lm = ctx.get("lm") if isinstance(ctx, dict) else None

    features = compute_chart_features(events, bpm)
    out = {k: features.get(k) for k in RAW_KEYS}

    if calibration is not None:
        if lm is not None:
            from .common import event_tokens

            stats = lm.sequence_stats(course, event_tokens(events, bpm))
            features["invalid_transition_rate"] = stats.get("invalid_transition_rate")
            features["rare_ngram_rate"] = stats.get("rare_ngram_rate")
            features["pattern_nll"] = stats.get("pattern_nll")
            from .common import quantile

            features["pattern_window_nll_p95"] = quantile(stats.get("window_nlls", []), 95)
        scores = score_chart_features(features, calibration, course)
        for k in SCORE_KEYS:
            out[k] = scores.get(k)
    else:
        for k in SCORE_KEYS:
            out[k] = None

    # adopted v2 candidate (raw; band-scored downstream if desired)
    out.update(call_response_reciprocity(events, ctx))
    return out
