"""chartgeneval -- a multi-dimensional metric suite for rhythm-game charts.

A self-contained evaluation toolkit with ChartGenEval chart-to-music and
structure measurements, difficulty-conditioned chart statistics, and the
controlled edits used to test declared corruption responses. The package
carries no dependency on any chart-generation model.

Quickstart::

    from chartgeneval.calibration import load_bundled_calibration
    from chartgeneval.metrics.common import NGramModel
    from chartgeneval.metrics import evaluate_profile
    from chartgeneval.metrics.base import make_ctx

    calibration = load_bundled_calibration()
    # Fit this LM on the exact dataset revision/order/alpha recorded by the
    # calibration artifact; experiments/reproduce_calibration.py is canonical.
    lm = NGramModel(order=calibration["lm_order"], alpha=calibration["lm_alpha"])
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
