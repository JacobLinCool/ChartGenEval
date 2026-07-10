"""Corpus calibration: per-course official bands and band -> score conversion.

Two layers:

* *raw* chart features (diagnostic, unbounded) from
  :func:`chartgeneval.metrics.common.compute_chart_features` + the n-gram model;
* *calibrated* scores in ``[0, 1]`` produced by comparing each feature against
  the official corpus band (p10-p90, course-conditioned) and mapping excess
  overloads through a lower-is-better decay.

The package ships a bundled calibration artifact (aggregate statistics computed
from the training split of ``taiko-1000-parsed``; these are summary quantiles,
not raw charts). :func:`load_bundled_calibration` returns it. Users can rebuild
bands from their own corpus with :func:`build_calibration`.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from .metrics.common import (
    COURSES,
    SCORE_VERSION,
    band_score,
    band_sigma,
    compute_chart_features,
    event_tokens,
    geometric_mean,
    lower_better_score,
    quantile,
    quantile_summary,
)

# Metrics for which a two-sided p10-p90 official band is computed.
BAND_METRICS = (
    "density_nps",
    "local_nps_p95",
    "local_nps_cv",
    "ioi_entropy_bits",
    "color_switch_rate",
    "big_note_rate",
    "unique_4gram_rate",
    "repeat_4gram_rate",
    "pattern_nll",
)

BUNDLED_CALIBRATION = "chart_quality_full_20260708_calibration.json"


def load_bundled_calibration() -> dict:
    """Load the calibration artifact shipped inside the package."""
    with resources.files("chartgeneval.data").joinpath(BUNDLED_CALIBRATION).open() as f:
        return json.load(f)


def load_calibration(path: str | Path) -> dict:
    """Load a calibration artifact from an arbitrary path."""
    with open(path) as f:
        return json.load(f)


def build_calibration(calib_rows, lm) -> dict:
    """Build per-course official bands from calibration rows.

    ``calib_rows`` is an iterable of dicts with keys ``events``, ``bpm``,
    ``course``. ``lm`` is a fitted :class:`NGramModel`.
    """
    rows = []
    for row in calib_rows:
        features = compute_chart_features(row["events"], row["bpm"])
        tokens = event_tokens(row["events"], row["bpm"])
        features.update(
            {k: v for k, v in lm.sequence_stats(row["course"], tokens).items() if k != "window_nlls"}
        )
        stats = lm.sequence_stats(row["course"], tokens)
        features["pattern_window_nll_p95"] = quantile(stats.get("window_nlls", []), 95)
        rows.append({**row, **features})

    by_course = {}
    for course in COURSES:
        cr = [r for r in rows if r["course"] == course]
        if not cr:
            continue
        entry = {"n": len(cr), "bands": {}, "thresholds": {}}
        for metric in BAND_METRICS:
            entry["bands"][metric] = quantile_summary([r.get(metric) for r in cr])
        entry["thresholds"]["local_nps_p95_official_p95"] = quantile(
            [r.get("local_nps_p95") for r in cr], 95, default=8.0
        )
        entry["thresholds"]["local_nps_delta_p95_official_p95"] = quantile(
            [r.get("local_nps_delta_p95") for r in cr], 95, default=4.0
        )
        entry["thresholds"]["pattern_window_nll_p95"] = quantile(
            [r.get("pattern_window_nll_p95") for r in cr], 95, default=8.0
        )
        by_course[course] = entry
    return {"score_version": SCORE_VERSION, "courses": by_course}


def _course_calibration(calibration, course):
    courses = calibration.get("courses") or {}
    return courses.get(course) or next(iter(courses.values()))


def score_chart_features(features, calibration, course) -> dict:
    """Convert raw features to calibrated [0, 1] scores against the bands."""
    c = _course_calibration(calibration, course)
    bands = c.get("bands") or {}
    thresholds = c.get("thresholds") or {}
    scores = {}

    def band(metric, score_name):
        q = bands.get(metric) or {}
        lower = q.get("p10")
        upper = q.get("p90")
        if lower is None or upper is None:
            scores[score_name] = None
            return
        width = max(float(upper) - float(lower), 1e-6)
        scores[score_name] = band_score(
            features.get(metric),
            lower,
            upper,
            sigma_low=max(width * 0.75, 1e-6),
            sigma_high=max(width * 0.50, 1e-6),
        )
        # saturation-free companion: signed band position in half-widths
        scores[score_name.replace("_score", "_band_sigma")] = band_sigma(
            features.get(metric), lower, upper
        )

    band("density_nps", "density_adequacy_score")
    band("local_nps_p95", "strain_adequacy_score")
    band("local_nps_cv", "density_variation_adequacy_score")
    band("ioi_entropy_bits", "rhythm_complexity_adequacy_score")
    band("color_switch_rate", "color_switch_adequacy_score")
    band("big_note_rate", "big_note_adequacy_score")
    band("unique_4gram_rate", "surface_variety_adequacy_score")
    band("repeat_4gram_rate", "repetition_adequacy_score")
    band("pattern_nll", "pattern_ic_adequacy_score")

    overload_thr = thresholds.get("local_nps_p95_official_p95") or 8.0
    spike_thr = thresholds.get("local_nps_delta_p95_official_p95") or 4.0
    chaos_thr = thresholds.get("pattern_window_nll_p95") or 8.0

    local_p95 = features.get("local_nps_p95")
    local_delta = features.get("local_nps_delta_p95")
    pattern_window_nll_p95 = features.get("pattern_window_nll_p95")

    features["overload_excess_nps"] = (
        max(0.0, float(local_p95) - float(overload_thr)) if local_p95 is not None else None
    )
    features["density_spike_excess_nps"] = (
        max(0.0, float(local_delta) - float(spike_thr)) if local_delta is not None else None
    )
    features["pattern_chaos_excess_nll"] = (
        max(0.0, float(pattern_window_nll_p95) - float(chaos_thr))
        if pattern_window_nll_p95 is not None
        else None
    )

    scores["overload_score"] = lower_better_score(features.get("overload_excess_nps"), 2.0)
    scores["density_spike_score"] = lower_better_score(features.get("density_spike_excess_nps"), 1.5)
    scores["pattern_chaos_score"] = lower_better_score(features.get("pattern_chaos_excess_nll"), 1.5)
    scores["boredom_score"] = lower_better_score(features.get("boredom_rate"), 0.10)
    scores["transition_validity_score"] = lower_better_score(
        features.get("invalid_transition_rate"), 0.08
    )
    scores["rare_ngram_score"] = lower_better_score(features.get("rare_ngram_rate"), 0.08)

    scores["playability_proxy_score"] = geometric_mean(
        [
            scores.get("density_adequacy_score"),
            scores.get("strain_adequacy_score"),
            scores.get("overload_score"),
            scores.get("density_spike_score"),
        ]
    )
    scores["local_pattern_score"] = geometric_mean(
        [
            scores.get("pattern_ic_adequacy_score"),
            scores.get("pattern_chaos_score"),
            scores.get("boredom_score"),
            scores.get("transition_validity_score"),
            scores.get("rare_ngram_score"),
        ]
    )
    scores["surface_structure_proxy_score"] = geometric_mean(
        [
            scores.get("surface_variety_adequacy_score"),
            scores.get("repetition_adequacy_score"),
            scores.get("density_variation_adequacy_score"),
        ]
    )
    scores["chart_quality_proxy_score"] = geometric_mean(
        [
            scores.get("playability_proxy_score"),
            scores.get("local_pattern_score"),
            scores.get("surface_structure_proxy_score"),
            scores.get("rhythm_complexity_adequacy_score"),
            scores.get("color_switch_adequacy_score"),
        ]
    )
    return scores


def evaluate_chart_quality(events, bpm, course, lm, calibration) -> dict:
    """Full raw features + calibrated scores for one chart (the golden path)."""
    features = compute_chart_features(events, bpm)
    tokens = event_tokens(events, bpm)
    lm_stats = lm.sequence_stats(course, tokens)
    window_nlls = lm_stats.pop("window_nlls", [])
    features.update(lm_stats)
    features["pattern_window_nll_p95"] = quantile(window_nlls, 95)
    scores = score_chart_features(features, calibration, course)
    return {**features, **scores}
