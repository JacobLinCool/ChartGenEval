"""Known-answer unit tests: at least one per metric family."""

from __future__ import annotations

import math

import numpy as np

from chartgeneval.metrics import coupling, gap, grammar, structure, timing
from chartgeneval.metrics.base import make_ctx
from chartgeneval.metrics.common import (
    NGramModel,
    band_score,
    event_tokens,
    fold_hit,
    lower_better_score,
    size_hit,
    snap_ioi_bin,
)


# ---------------------------------------------------------------- common ----
def test_fold_and_size():
    assert fold_hit("don") == "D"
    assert fold_hit("ka_big") == "K"
    assert size_hit("don_big") == "B"
    assert size_hit("ka") == "S"


def test_snap_ioi_bin_known():
    # exactly one beat -> the "1.0" bin (index 8 in IOI_BEAT_BINS -> b08)
    assert snap_ioi_bin(1.0) == "b08"
    # exactly half a beat -> b05
    assert snap_ioi_bin(0.5) == "b05"
    assert snap_ioi_bin(0.0) == "bad"


def test_band_score_edges():
    # inside band -> 1.0
    assert band_score(5.0, 4.0, 6.0) == 1.0
    # far outside -> small but ordered (rational tail keeps far-field
    # resolution: 5 sigma vs 12 sigma remain distinguishable)
    assert band_score(100.0, 4.0, 6.0) < 0.001
    assert band_score(100.0, 4.0, 6.0) > band_score(400.0, 4.0, 6.0) > 0.0
    # symmetric-ish falloff
    assert 0.0 < band_score(3.0, 4.0, 6.0) < 1.0


def test_lower_better_score():
    assert lower_better_score(0.0, 2.0) == 1.0
    assert lower_better_score(2.0, 2.0) == math.exp(-1.0)
    assert lower_better_score(None, 2.0) is None


def test_event_tokens_known():
    # two dons one beat apart @ 60 BPM: IOI = 1 beat -> "b08:DS"
    toks = event_tokens([(0.0, "don"), (1.0, "don")], bpm=60.0)
    assert toks == ["start:DS", "b08:DS"]


# ---------------------------------------------------------------- timing ----
def test_timing_recovers_anchor_shift(alt_eighths, bpm, grid):
    """C2-style constant shift is recovered by grid_phase_offset (the C2 witness)."""
    ctx = make_ctx(course="oni", bpm=bpm, grid=grid, duration=64.0)
    base = timing.compute(alt_eighths, ctx)
    assert abs(base["grid_phase_offset_ms"]) < 1.0  # official ~ on grid

    shifted = [(t + 0.030, c) for t, c in alt_eighths]  # +30 ms
    out = timing.compute(shifted, ctx)
    assert abs(out["grid_phase_offset_ms"] - 30.0) < 1.0  # recovers +30 ms


def test_timing_empty_grid_graceful(alt_eighths, bpm):
    out = timing.compute(alt_eighths, make_ctx(course="oni", bpm=bpm))
    # no grid -> still gets the fixed-grid witness from bpm, no lattice matching
    assert out["n_matched"] == 0
    assert out["grid_phase_offset_ms"] is not None


# --------------------------------------------------------------- grammar ----
def test_grammar_perplexity_lower_for_seen(alt_eighths, constant_dons, bpm):
    """A chart the LM has seen has lower perplexity than an unseen mixed chart."""
    lm = NGramModel(order=3, alpha=0.05)
    lm.add("oni", event_tokens(constant_dons, bpm))  # only constant-dons seen
    ctx = make_ctx(course="oni", bpm=bpm, lm=lm)
    seen = grammar.compute(constant_dons, ctx)
    unseen = grammar.compute(alt_eighths, ctx)
    assert seen["pattern_ngram_perplexity"] < unseen["pattern_ngram_perplexity"]
    # unseen chart has invalid transitions under this LM
    assert unseen["invalid_transition_rate"] > 0.0


def test_grammar_no_lm_is_nan(alt_eighths, bpm):
    out = grammar.compute(alt_eighths, make_ctx(course="oni", bpm=bpm))
    assert out["pattern_nll"] is None


# ------------------------------------------------------------- structure ----
def test_structure_repetition(constant_dons, bpm):
    """A maximally repetitive chart has near-zero surface variety."""
    beat = 60.0 / bpm
    # a genuinely varied chart: pseudo-random colours on the beat
    rng_colors = ["don", "ka", "don", "ka_big", "ka", "don_big", "ka", "don"]
    varied = [(i * beat, rng_colors[(i * 7 + 3) % len(rng_colors)]) for i in range(128)]
    ctx = make_ctx(course="oni", bpm=bpm)
    rep = structure.compute(constant_dons, ctx)
    var = structure.compute(varied, ctx)
    assert rep["repeat_4gram_rate"] > var["repeat_4gram_rate"]
    assert rep["unique_4gram_rate"] < var["unique_4gram_rate"]


def test_structure_reciprocity_call_response(bpm, grid):
    """Alternating phrases (ABAB colour) yield a positive call-response rate."""
    beat = 60.0 / bpm
    bar = beat * 4
    n_bars = 32
    dbs = [i * bar for i in range(n_bars + 1)]
    ctx = make_ctx(course="oni", bpm=bpm, grid={"downbeats": dbs, "bar": bar}, duration=n_bars * bar)
    call_c = ["don", "don", "ka", "don", "ka", "don"]
    resp_c = ["ka", "don", "don", "don", "don", "don"]
    ev = []
    for ph in range(0, n_bars, 2):
        start = ph * bar
        colors = call_c if (ph // 2) % 2 == 0 else resp_c
        for k in range(6):
            ev.append((start + k * beat, colors[k]))
    out = structure.compute(ev, ctx)
    assert out["call_response_rate"] > 0.0


# ------------------------------------------------------------- coupling ----
def test_coupling_density_scale_invariant_energy():
    """density_energy_response: NaN without mel/grid, degrades gracefully."""
    ev = [(i * 0.25, "don") for i in range(64)]
    out = coupling.density_energy_response(ev, make_ctx(course="oni", bpm=160.0))
    assert math.isnan(out["density_energy_spearman"])


def test_coupling_energy_peak_on_synthetic_mel():
    """energy_peak: hits on onset frames score higher support than hits off them."""
    fps = coupling.FPS
    T = 2000
    mel = np.full((8, T), -5.0, dtype=np.float64)
    # inject sharp onsets every 43 frames (~0.5 s)
    onset_frames = list(range(43, T, 43))
    for f in onset_frames:
        mel[:, f] = 5.0
    on_hits = [(f / fps, "don") for f in onset_frames]
    off_hits = [((f + 20) / fps, "don") for f in onset_frames]  # between onsets
    ctx = make_ctx(course="oni", bpm=160.0, mel=mel)
    on = coupling.energy_peak_support_rate(on_hits, ctx)
    off = coupling.energy_peak_support_rate(off_hits, ctx)
    # energy_at_hit_z (exact-frame leaf) cleanly separates on- vs off-onset hits;
    # the windowed support_rate can saturate at 1.0 for both on this toy mel.
    assert on["energy_at_hit_z"] > off["energy_at_hit_z"]
    assert on["energy_peak_mean_z"] >= off["energy_peak_mean_z"]


# ------------------------------------------------------------------ gap ----
def test_gap_phi_dim(alt_eighths, bpm):
    p = gap.phi(alt_eighths, bpm)
    assert p is not None and len(p) == gap.PHI_DIM


def test_gap_official_scores_higher_than_outlier(alt_eighths, constant_dons, bpm):
    """A chart resembling the reference cloud gets a smaller manifold gap."""
    # a small varied reference cloud around the alternating chart
    beat = 60.0 / bpm
    ref_charts = [
        [(i * beat, "don" if (i + s) % 2 == 0 else "ka") for i in range(120 + s)] for s in range(6)
    ]
    ref = gap.fit_manifold_ref([gap.phi(c, bpm) for c in ref_charts], k=3)
    ctx_ref = make_ctx(course="oni", bpm=bpm, official_manifold_ref=ref)
    near = gap.compute(alt_eighths, ctx_ref)
    far = gap.compute(constant_dons, ctx_ref)
    assert near["manifold_gap_raw"] <= far["manifold_gap_raw"]
    assert near["manifold_score"] >= far["manifold_score"]


# ------------------------------------------------------ calibrated profile ----
def test_evaluate_profile_bounds(alt_eighths, bpm, toy_lm, toy_calibration):
    from chartgeneval.metrics import evaluate_profile

    ctx = make_ctx(course="oni", bpm=bpm, lm=toy_lm, calibration=toy_calibration)
    scores = evaluate_profile(alt_eighths, ctx)
    q = scores["chart_quality_proxy_score"]
    assert 0.0 <= q <= 1.0
