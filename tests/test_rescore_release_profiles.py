"""Tests for the deterministic release-profile scoring transform."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from chartgeneval.metrics.common import SCORE_VERSION


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "experiments" / "rescore_release_profiles.py"
SPEC = importlib.util.spec_from_file_location("rescore_release_profiles", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_rescore_record_removes_retired_total_and_uses_symmetric_band():
    calibration = {
        "courses": {
            "oni": {
                "bands": {
                    metric: {"p10": 4.0, "p90": 6.0}
                    for metric in (
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
                },
                "thresholds": {},
            }
        }
    }
    record = {
        "sid": "s0",
        "course": "oni",
        "chart_quality_proxy_score": 0.9,
        "density_nps": 3.0,
        "local_nps_p95": 7.0,
        "local_nps_cv": 5.0,
        "ioi_entropy_bits": 5.0,
        "color_switch_rate": 5.0,
        "big_note_rate": 5.0,
        "unique_4gram_rate": 5.0,
        "repeat_4gram_rate": 5.0,
        "pattern_nll": 5.0,
    }
    result = MODULE.rescore_record(record, calibration)
    assert "chart_quality_proxy_score" not in result
    assert result["density_adequacy_score"] == result["strain_adequacy_score"] == 0.5
    assert result["score_version"] == SCORE_VERSION
