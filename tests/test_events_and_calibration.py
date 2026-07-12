"""Event loader, calibration, and audit tests."""

from __future__ import annotations

import json

import pytest

from chartgeneval.audit import dose_response, separation_auc, spearman
from chartgeneval.calibration import (
    build_calibration,
    calibration_content_sha256,
    load_bundled_calibration,
    load_calibration,
    require_calibration_provenance,
    score_chart_features,
)
from chartgeneval.events import (
    Chart,
    load_events_json,
    load_events_osu,
    load_events_tja,
    load_taiko_parsed_course,
    sorted_hits,
)
from chartgeneval.metrics.common import (
    NGramModel,
    SCORE_VERSION,
    compute_chart_features,
    event_tokens,
)


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
    assert cal["artifact_version"] == 1
    assert cal["dataset_id"] == "JacobLinCool/taiko-1000-parsed-clean"
    assert cal["dataset_revision"] == "b72da4616d643018e81f372cea06ce51349285e0"
    assert cal["split"] == "train"
    assert cal["n_charts"] == 3880
    assert cal["lm_order"] == 3
    assert cal["lm_alpha"] == 0.05
    assert cal["score_version"] == "calibrated_diagnostics_v1_symmetric_halfwidth"
    assert set(cal["courses"].keys()) == {"easy", "normal", "hard", "oni", "ura"}
    oni = cal["courses"]["oni"]
    assert "density_nps" in oni["bands"]
    assert {"p10", "p90"} <= set(oni["bands"]["density_nps"].keys())
    assert calibration_content_sha256(cal) == calibration_content_sha256(dict(cal))


def test_calibration_provenance_mismatch_is_rejected():
    calibration = load_bundled_calibration()
    with pytest.raises(ValueError, match="dataset_revision"):
        require_calibration_provenance(
            calibration,
            dataset_id=calibration["dataset_id"],
            dataset_revision="0" * 40,
            split="train",
        )


def test_golden_chart_record_declares_scoring_contract(toy_lm, toy_calibration, bpm):
    from chartgeneval.calibration import evaluate_chart_quality

    events = [(i * 0.25, "don" if i % 2 else "ka") for i in range(16)]
    record = evaluate_chart_quality(events, bpm, "oni", toy_lm, toy_calibration)
    assert record["score_version"] == SCORE_VERSION


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


def test_load_calibration_rejects_stale_score_version(tmp_path):
    path = tmp_path / "stale.json"
    path.write_text(json.dumps({"score_version": "chart_quality_proxy_v1", "courses": {"oni": {}}}))
    with pytest.raises(ValueError, match="incompatible calibration"):
        load_calibration(path)


@pytest.mark.parametrize("offset_s", [0.060, 1.0])
def test_chart_only_features_are_exactly_time_origin_invariant(
    offset_s, bpm, toy_calibration
):
    """Decimal shifts cannot move boundary notes across density windows."""
    # Every half-second note lies on a local-window boundary; the repeated
    # timestamp at 4 s also creates a non-trivial calibrated spike penalty.
    times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    times += [4.0] * 10
    times += [4.5, 5.0, 5.5, 6.0]
    events = [(t, "don" if i % 2 == 0 else "ka") for i, t in enumerate(times)]
    shifted = [(t + offset_s, cls) for t, cls in events]

    baseline_features = compute_chart_features(events, bpm)
    shifted_features = compute_chart_features(shifted, bpm)

    # This intentionally checks the complete chart-only bundle with exact
    # equality, not a tolerance that could hide a boundary-membership change.
    assert shifted_features == baseline_features
    assert event_tokens(shifted, bpm) == event_tokens(events, bpm)

    baseline_scoring_features = baseline_features.copy()
    shifted_scoring_features = shifted_features.copy()
    baseline_scores = score_chart_features(
        baseline_scoring_features, toy_calibration, "oni"
    )
    shifted_scores = score_chart_features(
        shifted_scoring_features, toy_calibration, "oni"
    )

    assert baseline_scores["density_spike_score"] < 1.0
    for key in (
        "density_nps",
        "local_nps_mean",
        "local_nps_p95",
        "local_nps_max",
        "local_nps_cv",
        "local_nps_delta_p95",
        "overload_excess_nps",
        "density_spike_excess_nps",
    ):
        assert shifted_scoring_features[key] == baseline_scoring_features[key]
    for key in (
        "density_adequacy_score",
        "strain_adequacy_score",
        "density_variation_adequacy_score",
        "overload_score",
        "density_spike_score",
        "playability_proxy_score",
    ):
        assert shifted_scores[key] == baseline_scores[key]


@pytest.mark.parametrize("offset_s", [0.060, 1.0])
def test_time_origin_invariance_survives_near_window_boundary_float(offset_s, bpm):
    """A near-boundary binary float cannot change half-open window membership."""
    origin = 31.33766764983761
    relative = [0.0, 0.5, 1.0, 1.5, 2.0 - 5.03e-13, 2.5, 3.0, 3.5, 4.0]
    events = [(origin + t, "don" if i % 2 else "ka") for i, t in enumerate(relative)]
    shifted = [(t + offset_s, cls) for t, cls in events]
    assert compute_chart_features(shifted, bpm) == compute_chart_features(events, bpm)
    assert event_tokens(shifted, bpm) == event_tokens(events, bpm)


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
