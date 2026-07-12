"""Unified metric-family interface.

Every metric family exposes ``compute(events, ctx) -> dict[str, float]``.

``events`` : list of ``(t_seconds, class)`` hit tuples.
``ctx``    : evaluation context dict. Recognised keys:

    course   : str | None   -- difficulty course (band conditioning).
    bpm      : float | None -- tempo (tokenizer + audio-coupled features).
    duration : float | None -- audio duration in seconds.
    grid     : dict | None  -- beat grid from a :mod:`chartgeneval.grid` adapter
                               ({"downbeats", "bar", "bpm"}).
    mel      : np.ndarray | None -- (128, T) log-power mel spectrogram for
                               audio-coupled metrics (energy_peak, density_energy).
    lm       : NGramModel | None -- fitted n-gram model (grammar family).
    calibration : dict | None -- calibration artifact (band scoring).
    official_manifold_ref : dict | None -- fitted manifold reference (gap family).

Families degrade gracefully: a missing dependency in ``ctx`` yields NaN outputs
rather than an exception. The five families are ``timing``, ``coupling``,
``grammar``, ``structure`` and ``gap``.
"""

from __future__ import annotations

from typing import Protocol


class MetricFamily(Protocol):
    name: str

    def compute(self, events, ctx) -> dict:
        ...


def make_ctx(
    *,
    course=None,
    bpm=None,
    duration=None,
    grid=None,
    mel=None,
    lm=None,
    calibration=None,
    official_manifold_ref=None,
    **extra,
) -> dict:
    """Assemble a metric context dict with the recognised keys."""
    ctx = {
        "course": course,
        "bpm": bpm,
        "duration": duration,
        "grid": grid,
        "mel": mel,
        "lm": lm,
        "calibration": calibration,
        "official_manifold_ref": official_manifold_ref,
    }
    ctx.update(extra)
    return ctx
