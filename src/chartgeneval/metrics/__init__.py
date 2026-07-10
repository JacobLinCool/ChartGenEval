"""Metric families and the full calibrated profile.

Five families, each with a uniform ``compute(events, ctx) -> dict`` entry point:

    timing      -- audio-anchored alignment vs an estimated beat grid.
    coupling    -- density/strain constraints + audio coupling (density_energy,
                   energy_peak).
    grammar     -- n-gram pattern grammar (pattern IC, transitions, rare n-grams).
    structure   -- surface variety / repetition / boredom / colour + reciprocity.
    gap         -- distance to the official chart manifold (manifold_gap).

``constraints`` re-exposes the hard playability gates.

:func:`evaluate_profile` runs the full calibrated chart-quality pipeline (the
golden path) and returns every raw feature and calibrated score in one dict.

Family submodules are imported lazily (via ``__getattr__``) so that
``chartgeneval.metrics.common`` can be imported by ``chartgeneval.calibration``
without triggering the family imports that depend on calibration.
"""

from __future__ import annotations

import importlib

_FAMILY_NAMES = ("timing", "coupling", "grammar", "structure", "gap", "constraints")

__all__ = [
    "FAMILIES",
    *_FAMILY_NAMES,
    "evaluate_profile",
    "compute_all_families",
]


def __getattr__(name):
    if name in _FAMILY_NAMES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name == "FAMILIES":
        fams = {n: importlib.import_module(f".{n}", __name__) for n in _FAMILY_NAMES}
        globals()["FAMILIES"] = fams
        return fams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def evaluate_profile(events, ctx) -> dict:
    """Full calibrated chart-quality profile (raw features + all scores).

    Requires ``ctx['lm']`` and ``ctx['calibration']``; ``bpm`` and ``course``
    condition the tokenizer and bands. This is the exact golden-path computation
    whose outputs match the reference to <1e-9.
    """
    from ..calibration import evaluate_chart_quality

    lm = ctx.get("lm")
    calibration = ctx.get("calibration")
    if lm is None or calibration is None:
        raise ValueError("evaluate_profile requires ctx['lm'] and ctx['calibration']")
    return evaluate_chart_quality(events, ctx.get("bpm"), ctx.get("course"), lm, calibration)


def compute_all_families(events, ctx) -> dict:
    """Run every family's ``compute`` and merge their outputs (family-prefixed).

    Each family output is namespaced by its family name to avoid key collisions.
    """
    out = {}
    for name in _FAMILY_NAMES:
        mod = __getattr__(name)
        try:
            res = mod.compute(events, ctx)
        except Exception as e:  # graceful: a family failure must not sink the run
            res = {"_error": f"{type(e).__name__}: {e}"}
        for k, v in res.items():
            out[f"{name}.{k}"] = v
    return out
