"""Grammar family: n-gram pattern grammar of the hit-token stream.

Scores how "chartable" the local pattern grammar is under a course-conditioned
n-gram model fitted on official charts: information content (band-scored, so
neither chaos nor blandness is rewarded), invalid-transition rate, rare-n-gram
rate and windowed pattern chaos. Requires ``ctx['lm']`` (a fitted
:class:`~chartgeneval.metrics.common.NGramModel`); band scoring additionally
needs ``ctx['calibration']``.
"""

from __future__ import annotations

import math

from ..calibration import score_chart_features
from .common import compute_chart_features, event_tokens, quantile

METRIC_NAME = "grammar"

RAW_KEYS = (
    "pattern_nll",
    "invalid_transition_rate",
    "rare_ngram_rate",
    "pattern_window_nll_p95",
    "pattern_ngram_perplexity",
)
SCORE_KEYS = (
    "pattern_ic_adequacy_score",
    "pattern_chaos_score",
    "transition_validity_score",
    "rare_ngram_score",
    "local_pattern_score",
)


def _nan_result():
    return {k: None for k in (*RAW_KEYS, *SCORE_KEYS)}


def compute(events, ctx):
    lm = ctx.get("lm") if isinstance(ctx, dict) else None
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    course = ctx.get("course") if isinstance(ctx, dict) else None
    calibration = ctx.get("calibration") if isinstance(ctx, dict) else None
    if lm is None:
        return _nan_result()

    tokens = event_tokens(events, bpm)
    stats = lm.sequence_stats(course, tokens)
    window_nlls = stats.pop("window_nlls", [])
    nll = stats.get("pattern_nll")
    out = {
        "pattern_nll": nll,
        "invalid_transition_rate": stats.get("invalid_transition_rate"),
        "rare_ngram_rate": stats.get("rare_ngram_rate"),
        "pattern_window_nll_p95": quantile(window_nlls, 95),
        "pattern_ngram_perplexity": float(math.exp(nll)) if nll is not None else None,
    }
    if calibration is None:
        for k in SCORE_KEYS:
            out[k] = None
        return out

    features = compute_chart_features(events, bpm)
    features.update({k: out[k] for k in ("pattern_nll", "invalid_transition_rate", "rare_ngram_rate")})
    features["pattern_window_nll_p95"] = out["pattern_window_nll_p95"]
    features["boredom_rate"] = features.get("boredom_rate")
    scores = score_chart_features(features, calibration, course)
    for k in SCORE_KEYS:
        out[k] = scores.get(k)
    return out
