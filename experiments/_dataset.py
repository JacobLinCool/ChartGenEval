"""Shared dataset helpers for the reproduction scripts.

Loads charts from a taiko-1000-parsed(-clean) Hugging Face dataset row and builds
the training-split n-gram model + calibration rows. Requires the ``[data]``
extra (``datasets``) and a local HF token for the gated dataset.
"""

from __future__ import annotations

from chartgeneval.events import COURSES, load_taiko_parsed_course
from chartgeneval.metrics.common import NGramModel, event_tokens

DEFAULT_DATASET = "JacobLinCool/taiko-1000-parsed-clean"


def iter_rows(dataset_id, split, limit=0):
    """Stream dataset rows, yielding ``(sid, row)``. Audio is not decoded."""
    from datasets import Audio, load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        yield f"{split}_{i:05d}", row


def official_charts(row):
    """Yield ``(course, Chart)`` for every non-empty course in a row."""
    for course in COURSES:
        chart = load_taiko_parsed_course(row.get(course))
        if chart is not None and chart.events:
            chart.course = course
            yield course, chart


def build_train_lm(dataset_id, split="train", limit=0, order=3, alpha=0.05, verbose=True):
    """Fit the course-conditioned n-gram model on the training split."""
    lm = NGramModel(order=order, alpha=alpha)
    n_charts = 0
    for i, (sid, row) in enumerate(iter_rows(dataset_id, split, limit)):
        for course, chart in official_charts(row):
            lm.add(course, event_tokens(chart.events, chart.bpm))
            n_charts += 1
        if verbose and i % 100 == 0:
            print(f"[lm {split}] songs={i + 1} charts={n_charts}", flush=True)
    return lm, n_charts


def build_calibration_rows(dataset_id, split="train", limit=0, verbose=True):
    """Collect ``{sid, course, events, bpm, level}`` rows for band building."""
    rows = []
    for i, (sid, row) in enumerate(iter_rows(dataset_id, split, limit)):
        for course, chart in official_charts(row):
            rows.append(
                {
                    "sid": sid,
                    "course": course,
                    "events": chart.events,
                    "bpm": chart.bpm,
                    "level": chart.level,
                }
            )
        if verbose and i % 100 == 0:
            print(f"[calib {split}] songs={i + 1} charts={len(rows)}", flush=True)
    return rows
