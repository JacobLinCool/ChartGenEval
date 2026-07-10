"""Grid-resolution helpers shared by audio/structure metric families.

Ported from the metric-candidate shared helpers. These resolve the ``ctx["grid"]``
dict into concrete downbeat arrays, beat periods and metrical phases, with the
same graceful-degradation contract (return None rather than raise).
"""

from __future__ import annotations

import bisect
import math

import numpy as np


def resolve_grid(ctx):
    """Return ``(downbeats, beat_period, meter)`` or ``None``.

    ``beat_period = bar / meter``; ``meter`` recovered as ``round(bar*bpm/60)``,
    clamped to ``[2, 12]`` (falls back to 4).
    """
    grid = ctx.get("grid") if isinstance(ctx, dict) else None
    if not grid:
        return None
    dbs = grid.get("downbeats")
    if dbs is None or len(dbs) < 2:
        return None
    dbs = np.asarray([float(x) for x in dbs], dtype=float)
    dbs.sort()
    bar = grid.get("bar")
    if bar is None or not np.isfinite(float(bar)) or float(bar) <= 0:
        diffs = np.diff(dbs)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return None
        bar = float(np.median(diffs))
    bar = float(bar)
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    if (not bpm) and isinstance(grid, dict):
        bpm = grid.get("bpm")
    meter = 4
    if bpm is not None and np.isfinite(float(bpm)) and float(bpm) > 0:
        beat_period_guess = 60.0 / float(bpm)
        if beat_period_guess > 0:
            m = int(round(bar / beat_period_guess))
            if 2 <= m <= 12:
                meter = m
    beat_period = bar / meter
    if not np.isfinite(beat_period) or beat_period <= 0:
        return None
    return dbs, beat_period, meter


def phase_of(t, downbeats, beat_period, meter):
    """Metrical phase (beats within a bar) of ``t`` in ``[0, meter)`` or None."""
    idx = bisect.bisect_right(downbeats, t) - 1
    if idx < 0:
        return None
    db = float(downbeats[idx])
    return float(((t - db) / beat_period) % meter)


def jsd(p, q, base=2.0, eps=1e-9):
    """Jensen-Shannon divergence between two discrete distributions."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    logb = math.log(base)

    def _kl(a, b):
        return float(np.sum(a * (np.log(a / b) / logb)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = len(a | b)
    if u == 0:
        return 1.0
    return len(a & b) / u
