"""chartgeneval -- corruption-validated quality metrics for rhythm-game charts.

A self-contained evaluation toolkit: a calibrated, corruption-validated quality
metric suite for automatically generated Taiko-style charts, plus the corruption
probes and audit gauntlet used to validate it. The package carries no dependency
on any chart-generation model.

Quickstart::

    from chartgeneval.calibration import load_bundled_calibration
    from chartgeneval.metrics.common import NGramModel
    from chartgeneval.metrics import evaluate_profile
    from chartgeneval.metrics.base import make_ctx

    calibration = load_bundled_calibration()
    lm = NGramModel(order=3, alpha=0.05)   # fit on your official corpus, or reuse
    ctx = make_ctx(course="oni", bpm=160.0, lm=lm, calibration=calibration)
    scores = evaluate_profile(events, ctx)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .events import Chart, load_events_json, load_taiko_parsed_course, sorted_hits

__all__ = [
    "__version__",
    "Chart",
    "load_events_json",
    "load_taiko_parsed_course",
    "sorted_hits",
]
