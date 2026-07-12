"""Shared fixtures: synthetic charts, a toy LM, a toy calibration, a grid."""

from __future__ import annotations

import pytest

from chartgeneval.calibration import build_calibration
from chartgeneval.grid import MetadataGrid
from chartgeneval.metrics.common import NGramModel, event_tokens

BPM = 160.0
BEAT = 60.0 / BPM
EIGHTH = BEAT / 2.0


@pytest.fixture
def alt_eighths():
    """Alternating don/ka on every eighth note for 200 hits @160 BPM."""
    return [(i * EIGHTH, "don" if i % 2 == 0 else "ka") for i in range(200)]


@pytest.fixture
def constant_dons():
    """All-don on every quarter note for 128 hits (maximally repetitive)."""
    return [(i * BEAT, "don") for i in range(128)]


@pytest.fixture
def bpm():
    return BPM


@pytest.fixture
def grid():
    """Metadata 4/4 grid covering ~60 s @160 BPM."""
    return MetadataGrid().grid_for(bpm=BPM, duration=64.0)


@pytest.fixture
def toy_lm(alt_eighths, constant_dons):
    """Small course-conditioned LM fitted on the two synthetic charts."""
    lm = NGramModel(order=3, alpha=0.05)
    lm.add("oni", event_tokens(alt_eighths, BPM))
    lm.add("oni", event_tokens(constant_dons, BPM))
    return lm


@pytest.fixture
def toy_calibration(alt_eighths, constant_dons, toy_lm):
    """Calibration bands from the synthetic charts (enough for band scoring)."""
    rows = [
        {"course": "oni", "events": alt_eighths, "bpm": BPM, "level": 9},
        {"course": "oni", "events": constant_dons, "bpm": BPM, "level": 9},
    ]
    return build_calibration(rows, toy_lm)
