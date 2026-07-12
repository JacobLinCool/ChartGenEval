"""Smoke tests for chartgeneval.plots (skipped without the [plots] extra)."""

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")

from chartgeneval import plots


def _records():
    rows = []
    for i in range(6):
        key = {"sid": f"s{i:02d}", "course": "oni"}
        rows.append({"probe": "official", "dose_index": 0, "m": 1.0, **key})
        for d in (1, 2, 3):
            rows.append({"probe": "CX", "dose_index": d, "m": 1.0 + 0.1 * d, **key})
    return rows


def test_net_direction_all_up():
    nd = plots.net_direction(_records(), "m", "CX")
    assert nd == {1: 1.0, 2: 1.0, 3: 1.0}


def test_net_direction_target_id_and_noop_keys():
    rows = _records()
    for r in rows:
        r["target_id"] = r.pop("probe")
    rows.append({"target_id": "CX", "dose_index": 3, "m": 99.0, "noop": True,
                 "sid": "s00", "course": "oni"})
    nd = plots.net_direction(rows, "m", "CX")
    assert nd[3] == 1.0  # noop row excluded


def test_timing_buckets_partition():
    row = {"clean_rate": 0.9, "matched_rate": 0.95, "unsupported_rate": 0.05,
           "absolute_violation_rate": 0.05, "absolute_violation_rate_2x": 0.02,
           "absolute_violation_rate_3x": 0.01}
    b = plots.timing_buckets(row)
    assert b.shape == (5,)
    assert np.isclose(b.sum(), 1.0)
    assert b[0] == pytest.approx(0.9, abs=0.02)


def test_timing_buckets_passthrough():
    b = plots.timing_buckets({"buckets": [2.0, 1.0, 1.0, 0.0, 0.0]})
    assert np.isclose(b.sum(), 1.0) and b[0] == 0.5


def test_plot_profile_heatmap_with_na_and_constraints(tmp_path):
    scores = {
        "official": {"timing": 0.97, "grammar": 0.96},
        "genelive": {"timing": 0.79, "grammar": None},
    }
    constraints = {"official": {"gate": True}, "genelive": {"gate": False}}
    fig, ax = plots.plot_profile(scores, columns=["timing", "grammar"],
                                 constraints=constraints)
    fig.savefig(tmp_path / "profile.png")


def test_plot_profile_radar_skips_incomplete(tmp_path):
    scores = {
        "a": {"x": 0.5, "y": 0.6, "z": 0.7},
        "b": {"x": 0.5, "y": None, "z": 0.7},
    }
    fig, ax = plots.plot_profile(scores, columns=["x", "y", "z"], kind="radar")
    assert "b" in ax.get_title()
    fig.savefig(tmp_path / "radar.png")


def test_plot_timing_tiers(tmp_path):
    rows = [
        {"clean_rate": 0.98, "matched_rate": 1.0, "unsupported_rate": 0.0,
         "absolute_violation_rate": 0.02, "absolute_violation_rate_2x": 0.0,
         "absolute_violation_rate_3x": 0.0},
        {"buckets": [0.6, 0.2, 0.1, 0.05, 0.05]},
    ]
    strips = {"A": np.array([0.9, 0.95, 1.0]), "B": np.array([0.5, 0.6, 0.7])}
    fig, ax = plots.plot_timing_tiers(rows, labels=["A", "B"], chart_clean_rates=strips)
    fig.savefig(tmp_path / "tiers.png")


def test_plot_dose_response(tmp_path):
    fig, ax = plots.plot_dose_response(_records(), ["m"], probe="CX")
    fig.savefig(tmp_path / "dose.png")


def test_plot_manifold(tmp_path):
    rng = np.random.default_rng(0)
    fig, ax = plots.plot_manifold(rng.normal(size=(50, 2)),
                                  {"sys": (0.5, 0.5)}, xlabel="x", ylabel="y")
    fig.savefig(tmp_path / "manifold.png")


def test_plot_chart_diagnostics(tmp_path):
    pytest.importorskip("scipy")
    events = [{"t": 0.5 * i, "type": "don"} for i in range(16)]
    events[7]["t"] += 0.05  # one 50 ms outlier
    ctx = {"grid": {"downbeats": [0.0, 2.0, 4.0, 6.0, 8.0], "bar": 2.0, "bpm": 120.0},
           "duration": 8.0}
    fig, ax = plots.plot_chart_diagnostics(events, ctx)
    fig.savefig(tmp_path / "diag.png")
