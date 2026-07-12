"""Constraint layer: hard playability bounds (overload / spike / chaos).

These are the lower-is-better excess quantities that gate a chart independently
of the two-sided adequacy bands: local-density overload, density-spike, and
windowed pattern chaos. They are computed inside :mod:`coupling` and
:mod:`grammar` (they share the same feature bundle and thresholds); this module
re-exposes them as a standalone constraint report for callers that want just the
safety gates.
"""

from __future__ import annotations

from ..calibration import score_chart_features
from .common import compute_chart_features, event_tokens, quantile

METRIC_NAME = "constraints"

SCORE_KEYS = ("overload_score", "density_spike_score", "pattern_chaos_score", "boredom_score")
RAW_KEYS = ("overload_excess_nps", "density_spike_excess_nps", "pattern_chaos_excess_nll", "boredom_rate")


def compute(events, ctx):
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    course = ctx.get("course") if isinstance(ctx, dict) else None
    calibration = ctx.get("calibration") if isinstance(ctx, dict) else None
    lm = ctx.get("lm") if isinstance(ctx, dict) else None

    features = compute_chart_features(events, bpm)
    if lm is not None:
        stats = lm.sequence_stats(course, event_tokens(events, bpm))
        features["invalid_transition_rate"] = stats.get("invalid_transition_rate")
        features["rare_ngram_rate"] = stats.get("rare_ngram_rate")
        features["pattern_nll"] = stats.get("pattern_nll")
        features["pattern_window_nll_p95"] = quantile(stats.get("window_nlls", []), 95)

    out = {}
    if calibration is not None:
        scores = score_chart_features(features, calibration, course)
        for k in SCORE_KEYS:
            out[k] = scores.get(k)
        for k in RAW_KEYS:
            out[k] = features.get(k)
    else:
        for k in (*SCORE_KEYS, *RAW_KEYS):
            out[k] = features.get(k)
    return out
