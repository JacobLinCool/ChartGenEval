"""Probe determinism + behaviour tests."""

from __future__ import annotations

import numpy as np

from chartgeneval.probes import PROBES, corrupt, seed_for


def _chart(bpm=160.0, n=256):
    beat = 60.0 / bpm
    eighth = beat / 2
    return [(i * eighth, "don" if i % 2 == 0 else "ka") for i in range(n)]


def test_seed_determinism():
    a = seed_for("test_00000", "oni", "C1_timing_jitter", 2)
    b = seed_for("test_00000", "oni", "C1_timing_jitter", 2)
    assert a == b
    # different cell -> (almost surely) different seed
    assert a != seed_for("test_00000", "oni", "C1_timing_jitter", 3)


def test_all_probes_bit_identical_two_runs():
    """Same (sid, course, probe, dose) -> bit-identical output across two runs."""
    ev = _chart()
    for probe in PROBES:
        needs_lm = PROBES[probe][2]
        lm = None
        if needs_lm:
            from chartgeneval.metrics.common import NGramModel, event_tokens

            lm = NGramModel(order=3, alpha=0.05)
            lm.add("oni", event_tokens(ev, 160.0))
        for di in (1, 2, 3):
            out1, noop1 = corrupt(ev, probe, di, sid="s0", course="oni", bpm=160.0, lm=lm)
            out2, noop2 = corrupt(ev, probe, di, sid="s0", course="oni", bpm=160.0, lm=lm)
            assert noop1 == noop2
            assert out1 == out2, f"{probe} dose {di} not deterministic"


def test_c2_anchor_shift_exact():
    ev = _chart()
    out, noop = corrupt(ev, "C2_anchor_shift", 3, sid="s0", course="oni", bpm=160.0)
    assert not noop
    # dose 3 = +60 ms constant shift
    shifts = [round(o[0] - e[0], 9) for e, o in zip(ev, out)]
    assert all(abs(s - 0.060) < 1e-9 for s in shifts)


def test_dose0_is_untouched():
    ev = _chart()
    out, noop = corrupt(ev, "C1_timing_jitter", 0, sid="s0", course="oni", bpm=160.0)
    assert out == sorted(ev)


def test_c6_density_scale_changes_count():
    ev = _chart(n=256)
    fewer, _ = corrupt(ev, "C6_density_scale", 1, sid="s0", course="oni", bpm=160.0)  # x0.5
    more, _ = corrupt(ev, "C6_density_scale", 3, sid="s0", course="oni", bpm=160.0)   # x2
    assert len(fewer) < len(ev) < len(more)


def test_c4_loop_collapse_reduces_variety():
    from chartgeneval.metrics.common import compute_chart_features

    ev = _chart()
    collapsed, _ = corrupt(ev, "C4_loop_collapse", 3, sid="s0", course="oni", bpm=160.0)
    v0 = compute_chart_features(ev, 160.0)["unique_4gram_rate"]
    v1 = compute_chart_features(collapsed, 160.0)["unique_4gram_rate"]
    assert v1 <= v0
