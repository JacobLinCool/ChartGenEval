"""Event loader, calibration, and audit tests."""

from __future__ import annotations

import json

import pytest

from chartgeneval.audit import dose_response, separation_auc, spearman
from chartgeneval.calibration import build_calibration, load_bundled_calibration
from chartgeneval.events import (
    Chart,
    load_events_json,
    load_events_osu,
    load_events_tja,
    load_taiko_parsed_course,
    sorted_hits,
)
from chartgeneval.metrics.common import NGramModel, event_tokens


# ------------------------------------------------------------------ events ----
def test_load_events_json_flat_list(tmp_path):
    p = tmp_path / "chart.json"
    p.write_text(json.dumps([{"t": 0.5, "type": "don"}, {"t": 0.0, "type": "ka"}]))
    chart = load_events_json(p)
    assert isinstance(chart, Chart)
    # sorted by time; roll/other classes dropped
    assert chart.events == [(0.0, "ka"), (0.5, "don")]


def test_load_events_json_object_with_hits(tmp_path):
    p = tmp_path / "chart.json"
    p.write_text(json.dumps({"bpm": 160.0, "course": "oni", "hits": [{"t": 1.0, "type": "don_big"}]}))
    chart = load_events_json(p)
    assert chart.bpm == 160.0
    assert chart.course == "oni"
    assert chart.events == [(1.0, "don_big")]


def test_load_events_json_gen_wrapper(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text(json.dumps({"gen": {"hits": [{"t": 0.0, "type": "ka"}]}, "course": "hard"}))
    chart = load_events_json(p)
    assert chart.events == [(0.0, "ka")]
    assert chart.course == "hard"


def test_load_taiko_parsed_course():
    struct = {
        "level": 9,
        "segments": [
            {"notes": [{"timestamp": 0.0, "note_type": "Don", "bpm": 160.0},
                       {"timestamp": 0.5, "note_type": "Ka", "bpm": 160.0},
                       {"timestamp": 1.0, "note_type": "Roll", "bpm": 160.0}]},  # roll dropped
        ],
    }
    chart = load_taiko_parsed_course(struct)
    assert chart.level == 9
    assert chart.bpm == 160.0
    assert chart.events == [(0.0, "don"), (0.5, "ka")]


def test_load_taiko_parsed_empty():
    assert load_taiko_parsed_course(None) is None
    assert load_taiko_parsed_course({"segments": []}) is None


def test_tja_osu_stubs_raise(tmp_path):
    with pytest.raises(NotImplementedError):
        load_events_tja(tmp_path / "x.tja")
    with pytest.raises(NotImplementedError):
        load_events_osu(tmp_path / "x.osu")


def test_sorted_hits_filters_and_sorts():
    ev = [(1.0, "don"), (0.0, "roll"), (0.5, "ka")]
    assert sorted_hits(ev) == [(0.5, "ka"), (1.0, "don")]


# ------------------------------------------------------------- calibration ----
def test_bundled_calibration_shape():
    cal = load_bundled_calibration()
    assert cal["score_version"] == "chart_quality_proxy_v1"
    assert set(cal["courses"].keys()) == {"easy", "normal", "hard", "oni", "ura"}
    oni = cal["courses"]["oni"]
    assert "density_nps" in oni["bands"]
    assert {"p10", "p90"} <= set(oni["bands"]["density_nps"].keys())


def test_build_calibration_roundtrip():
    bpm = 160.0
    beat = 60.0 / bpm
    ev_a = [(i * beat, "don" if i % 2 == 0 else "ka") for i in range(100)]
    ev_b = [(i * beat, "don") for i in range(100)]
    lm = NGramModel(order=3, alpha=0.05)
    for ev in (ev_a, ev_b):
        lm.add("oni", event_tokens(ev, bpm))
    cal = build_calibration(
        [{"course": "oni", "events": ev_a, "bpm": bpm}, {"course": "oni", "events": ev_b, "bpm": bpm}], lm
    )
    assert cal["courses"]["oni"]["n"] == 2
    assert cal["courses"]["oni"]["bands"]["density_nps"]["p10"] is not None


# ------------------------------------------------------------------- audit ----
def test_spearman_monotone():
    assert spearman([0, 1, 2, 3], [10, 8, 6, 4]) == pytest.approx(-1.0)
    assert spearman([0, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(1.0)


def test_separation_auc():
    assert separation_auc([3, 4, 5], [0, 1, 2]) == 1.0
    assert separation_auc([0, 1, 2], [3, 4, 5]) == 0.0
    assert separation_auc([1, 2, 3], [1, 2, 3]) == pytest.approx(0.5)


def test_dose_response_detects_drop():
    # synthetic probe table: score falls with dose for one metric
    rows = []
    for sid in range(20):
        rows.append({"sid": f"s{sid}", "course": "oni", "probe": "official", "dose_index": 0, "m": 1.0})
        for di, val in [(1, 0.8), (2, 0.6), (3, 0.4)]:
            rows.append({"sid": f"s{sid}", "course": "oni", "probe": "C1", "dose_index": di, "m": val})
    dr = {(d["probe"], d["metric"]): d for d in dose_response(rows, ["m"])}
    d = dr[("C1", "m")]
    assert d["spearman_dose_score"] < 0
    assert d["monotone_nonincreasing"] is True
    assert d["mean_dose0"] > d["mean_dose3"]
