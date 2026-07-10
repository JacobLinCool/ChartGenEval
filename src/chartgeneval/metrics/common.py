"""Shared primitives for the metric families.

This is the fidelity-critical core: tokenization, IOI beat-binning, the
course-conditioned n-gram model, band / lower-better scoring, and the chart
feature extractor. The numerics here must reproduce the reference pipeline
bit-for-bit, so the logic is a faithful, dependency-stripped port -- no
behavioural changes.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from ..events import HIT_CLASSES, sorted_hits

COURSES = ("easy", "normal", "hard", "oni", "ura")

# Beat-relative inter-onset-interval bins used by the tokenizer and features.
IOI_BEAT_BINS = (
    1 / 12,
    1 / 8,
    1 / 6,
    1 / 4,
    1 / 3,
    1 / 2,
    2 / 3,
    3 / 4,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
)

SCORE_VERSION = "chart_quality_proxy_v1"


def fold_hit(cls: str) -> str:
    """Colour fold: don -> D, ka -> K, else X."""
    if "don" in cls:
        return "D"
    if "ka" in cls:
        return "K"
    return "X"


def size_hit(cls: str) -> str:
    """Size fold: big -> B, else S."""
    return "B" if "big" in cls else "S"


def entropy_from_counts(counts) -> float | None:
    total = float(sum(counts))
    if total <= 0:
        return None
    probs = [c / total for c in counts if c > 0]
    return float(-sum(p * math.log2(p) for p in probs))


def geometric_mean(values) -> float | None:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not xs:
        return None
    if any(v <= 0 for v in xs):
        return 0.0
    return float(math.exp(sum(math.log(v) for v in xs) / len(xs)))


def lower_better_score(x, scale, floor=1e-9) -> float | None:
    """Map a non-negative excess (lower is better) to (0, 1]."""
    if x is None or not np.isfinite(float(x)):
        return None
    scale = max(float(scale or 0.0), floor)
    return float(math.exp(-max(0.0, float(x)) / scale))


def band_score(x, lower, upper, sigma_low=None, sigma_high=None) -> float | None:
    """1.0 inside [lower, upper], Gaussian falloff outside."""
    if x is None or not np.isfinite(float(x)):
        return None
    x = float(x)
    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        lower, upper = upper, lower
    width = max(upper - lower, 1e-9)
    sigma_low = max(float(sigma_low) if sigma_low is not None else width * 0.5, 1e-9)
    sigma_high = max(float(sigma_high) if sigma_high is not None else width * 0.5, 1e-9)
    if lower <= x <= upper:
        return 1.0
    if x < lower:
        return float(math.exp(-((lower - x) / sigma_low) ** 2))
    return float(math.exp(-((x - upper) / sigma_high) ** 2))


def quantile(values, q, default=None):
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not xs:
        return default
    return float(np.percentile(xs, q))


def quantile_summary(values) -> dict:
    xs = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not xs:
        return {}
    return {
        "n": len(xs),
        "p05": float(np.percentile(xs, 5)),
        "p10": float(np.percentile(xs, 10)),
        "p25": float(np.percentile(xs, 25)),
        "p50": float(np.percentile(xs, 50)),
        "p75": float(np.percentile(xs, 75)),
        "p90": float(np.percentile(xs, 90)),
        "p95": float(np.percentile(xs, 95)),
    }


def snap_ioi_bin(ioi_beats) -> str:
    """Snap a beat-relative IOI to the nearest metrical bin label."""
    if ioi_beats is None or not np.isfinite(float(ioi_beats)) or ioi_beats <= 0:
        return "bad"
    x = float(ioi_beats)
    rel = [abs(x - b) / b for b in IOI_BEAT_BINS]
    j = int(np.argmin(rel))
    if rel[j] <= 0.18:
        return f"b{j:02d}"
    if x < IOI_BEAT_BINS[0]:
        return "short"
    if x > IOI_BEAT_BINS[-1]:
        return "long"
    return "off"


def event_tokens(events, bpm) -> list[str]:
    """Tokenize a hit stream into ``{ioi_bin}:{color}{size}`` tokens."""
    hits = sorted_hits(events)
    if not hits:
        return []
    tokens = []
    prev_t = None
    for t, cls in hits:
        note = fold_hit(cls) + size_hit(cls)
        if prev_t is None or not bpm or bpm <= 0:
            ioi = "start"
        else:
            ioi = snap_ioi_bin((t - prev_t) * bpm / 60.0)
        tokens.append(f"{ioi}:{note}")
        prev_t = t
    return tokens


def _local_density_series(times, window_s=2.0, step_s=0.5):
    if len(times) < 2:
        return np.zeros(0, dtype=np.float64)
    start = float(times[0])
    end = float(times[-1])
    if end <= start:
        return np.zeros(0, dtype=np.float64)
    starts = np.arange(start, max(start, end - window_s) + step_s, step_s)
    vals = []
    times = np.asarray(times, dtype=np.float64)
    for s in starts:
        vals.append(float(np.sum((times >= s) & (times < s + window_s))) / window_s)
    return np.asarray(vals, dtype=np.float64)


def _rolling_windows(seq, size):
    if len(seq) < size:
        return []
    return [seq[i : i + size] for i in range(len(seq) - size + 1)]


def _window_boredom(tokens, colors, ioi_bins, size=8):
    if len(tokens) < size:
        return 0.0
    bad = 0
    total = 0
    for i in range(len(tokens) - size + 1):
        total += 1
        t = tokens[i : i + size]
        c = colors[i : i + size]
        b = ioi_bins[i : i + size]
        unique_ratio = len(set(t)) / size
        color_ent = entropy_from_counts(Counter(c).values()) or 0.0
        ioi_ent = entropy_from_counts(Counter(b).values()) or 0.0
        if unique_ratio <= 0.30 or (color_ent <= 0.35 and ioi_ent <= 0.75):
            bad += 1
    return float(bad / total) if total else 0.0


def compute_chart_features(events, bpm) -> dict:
    """Chart-only feature bundle: density, IOI, colour, variety, boredom."""
    hits = sorted_hits(events)
    times = np.asarray([t for t, _ in hits], dtype=np.float64)
    classes = [c for _, c in hits]
    n = len(hits)
    out = {
        "n_notes": n,
        "duration_active_s": None,
        "density_nps": None,
        "local_nps_mean": None,
        "local_nps_p95": None,
        "local_nps_max": None,
        "local_nps_cv": None,
        "local_nps_delta_p95": None,
        "ioi_entropy_bits": None,
        "ioi_cv": None,
        "short_ioi_rate": None,
        "color_switch_rate": None,
        "big_note_rate": None,
        "unique_4gram_rate": None,
        "repeat_4gram_rate": None,
        "boredom_rate": None,
    }
    if n == 0:
        return out

    if n >= 2:
        duration = max(float(times[-1] - times[0]), 1e-9)
        out["duration_active_s"] = duration
        out["density_nps"] = float(n / duration) if duration >= 1.0 else None
        local = _local_density_series(times)
        if len(local):
            out["local_nps_mean"] = float(np.mean(local))
            out["local_nps_p95"] = float(np.percentile(local, 95))
            out["local_nps_max"] = float(np.max(local))
            mean = max(float(np.mean(local)), 1e-9)
            out["local_nps_cv"] = float(np.std(local) / mean)
            deltas = np.abs(np.diff(local))
            out["local_nps_delta_p95"] = (
                float(np.percentile(deltas, 95)) if len(deltas) else 0.0
            )

        iois = np.diff(times)
        if len(iois):
            out["ioi_cv"] = float(np.std(iois) / max(np.mean(iois), 1e-9))
            if bpm and bpm > 0:
                beat_iois = iois * bpm / 60.0
                bins = [snap_ioi_bin(x) for x in beat_iois]
                out["ioi_entropy_bits"] = entropy_from_counts(Counter(bins).values())
                out["short_ioi_rate"] = float(np.mean(beat_iois < 0.125))

    colors = [fold_hit(c) for c in classes]
    if n >= 2:
        switches = [colors[i] != colors[i - 1] for i in range(1, n)]
        out["color_switch_rate"] = float(np.mean(switches))
    out["big_note_rate"] = float(np.mean(["big" in c for c in classes]))

    tokens = event_tokens(hits, bpm)
    if len(tokens) >= 4:
        grams = [tuple(tokens[i : i + 4]) for i in range(len(tokens) - 3)]
        out["unique_4gram_rate"] = float(len(set(grams)) / len(grams))
        gram_counts = Counter(grams)
        repeated = sum(c - 1 for c in gram_counts.values() if c > 1)
        out["repeat_4gram_rate"] = float(repeated / len(grams))
    if len(tokens) >= 8:
        ioi_bins = [tok.split(":", 1)[0] for tok in tokens]
        out["boredom_rate"] = _window_boredom(tokens, colors, ioi_bins)
    else:
        out["boredom_rate"] = 0.0
    return out


@dataclass
class NGramModel:
    """Course-conditioned add-alpha n-gram model over chart tokens."""

    order: int = 3
    alpha: float = 0.05

    def __post_init__(self):
        self.course_counts = defaultdict(Counter)
        self.course_context = defaultdict(Counter)
        self.all_counts = Counter()
        self.all_context = Counter()
        self.vocab = set()

    def add(self, course, tokens):
        if len(tokens) < self.order:
            return
        self.vocab.update(tokens)
        for i in range(len(tokens) - self.order + 1):
            ctx = tuple(tokens[i : i + self.order - 1])
            ng = tuple(tokens[i : i + self.order])
            self.course_counts[course][ng] += 1
            self.course_context[course][ctx] += 1
            self.all_counts[ng] += 1
            self.all_context[ctx] += 1

    def _prob(self, course, ng):
        ctx = tuple(ng[:-1])
        v = max(len(self.vocab), 1)
        counts = self.course_counts.get(course) or self.all_counts
        contexts = self.course_context.get(course) or self.all_context
        numerator = counts.get(tuple(ng), 0.0) + self.alpha
        denominator = contexts.get(ctx, 0.0) + self.alpha * v
        if denominator <= 0:
            numerator = self.all_counts.get(tuple(ng), 0.0) + self.alpha
            denominator = self.all_context.get(ctx, 0.0) + self.alpha * v
        return float(numerator / max(denominator, 1e-12))

    def sequence_stats(self, course, tokens, rare_threshold=1e-4):
        if len(tokens) < self.order:
            return {
                "pattern_nll": None,
                "invalid_transition_rate": None,
                "rare_ngram_rate": None,
                "window_nlls": [],
            }
        nlls = []
        invalid = []
        rare = []
        counts = self.course_counts.get(course) or Counter()
        for i in range(len(tokens) - self.order + 1):
            ng = tuple(tokens[i : i + self.order])
            p = self._prob(course, ng)
            nlls.append(-math.log(max(p, 1e-12)))
            invalid.append(1.0 if counts.get(ng, 0) == 0 else 0.0)
            rare.append(1.0 if p < rare_threshold else 0.0)
        window_nlls = []
        for w in _rolling_windows(tokens, 12):
            ws = []
            for i in range(len(w) - self.order + 1):
                ws.append(-math.log(max(self._prob(course, tuple(w[i : i + self.order])), 1e-12)))
            if ws:
                window_nlls.append(float(np.mean(ws)))
        return {
            "pattern_nll": float(np.mean(nlls)) if nlls else None,
            "invalid_transition_rate": float(np.mean(invalid)) if invalid else None,
            "rare_ngram_rate": float(np.mean(rare)) if rare else None,
            "window_nlls": window_nlls,
        }
