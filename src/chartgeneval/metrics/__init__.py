"""Metric families and the full calibrated profile.

Five families, each with a uniform ``compute(events, ctx) -> dict`` entry point:

    timing      -- alignment against an authored or audio-estimated grid.
    coupling    -- density/strain constraints + audio coupling (density_energy,
                   energy_peak).
    grammar     -- n-gram pattern grammar (pattern IC, transitions, rare n-grams).
    structure   -- surface variety / repetition / boredom / colour + reciprocity.
    gap         -- distance to a course-specific human reference cloud.

``constraints`` re-exposes the hard playability gates.

:func:`evaluate_profile` returns the calibrated chart diagnostics together with
all family outputs. It deliberately exposes no cross-family total score.

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
    """Full diagnostic profile: calibrated leaves plus every metric family.

    Requires ``ctx['lm']`` and ``ctx['calibration']``; ``bpm`` and ``course``
    condition the tokenizer and bands. Calibrated chart diagnostics remain
    unprefixed; family-interface results are namespaced as ``family.metric``.
    No cross-family total is computed.
    """
    from ..calibration import evaluate_chart_quality

    lm = ctx.get("lm")
    calibration = ctx.get("calibration")
    if lm is None or calibration is None:
        raise ValueError("evaluate_profile requires ctx['lm'] and ctx['calibration']")
    calibrated = evaluate_chart_quality(
        events, ctx.get("bpm"), ctx.get("course"), lm, calibration
    )
    return {**calibrated, **compute_all_families(events, ctx)}


def compute_all_families(events, ctx) -> dict:
    """Run every family's ``compute`` and merge their outputs (family-prefixed).

    Each family output is namespaced by its family name to avoid key collisions.
    """
    out = {}
    for name in _FAMILY_NAMES:
        mod = __getattr__(name)
        res = mod.compute(events, ctx)
        for k, v in res.items():
            out[f"{name}.{k}"] = v
    return out
