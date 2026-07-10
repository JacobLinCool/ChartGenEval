"""Tests for the five adopted suite-v2 candidate metrics.

Two layers per metric:

* behaviour tests on synthetic charts (the constructs do what they claim);
* pinned fixed-value equivalence tests -- the expected constants were computed
  by running the *source* research implementations (SoftChart
  ``experiments/metric_candidates_v1/candidates/``, post-fix, 2026-07-11) on
  the exact same deterministic inputs, and the port must reproduce them
  bitwise (plain ``==``). The full-scale bitwise equivalence run (synthetic
  battery + real official charts + real-audio mel) lives in
  ``experiments/equivalence_v2_candidates.py``.
"""

from __future__ import annotations

import math
import random

import numpy as np

from chartgeneval.metrics import coupling, gap, structure
from chartgeneval.metrics.base import make_ctx

BPM = 160.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
SR, HOP, N_MELS = 22050, 256, 128
FPS = SR / HOP  # 86.1328125


def _ctx(n_bars, mel=None):
    dbs = [i * BAR for i in range(n_bars + 1)]
    return make_ctx(
        course="oni",
        bpm=BPM,
        grid={"downbeats": dbs, "bar": BAR},
        duration=n_bars * BAR,
        mel=mel,
    )


def _abab_chart(n_bars=32):
    """Same rhythm every 2-bar phrase; response phrases flip 3/6 colours."""
    call_c = ["don", "don", "ka", "don", "ka", "don"]
    resp_c = ["ka", "don", "don", "don", "don", "don"]
    ev = []
    for ph in range(0, n_bars, 2):
        colors = call_c if (ph // 2) % 2 == 0 else resp_c
        ev.extend((ph * BAR + k * BEAT, colors[k]) for k in range(6))
    return ev


def _loop_chart(n_bars=40):
    """One bar figure repeated verbatim (copy-paste collapse)."""
    return [
        (b * BAR + k * BEAT, c)
        for b in range(n_bars)
        for k, c in enumerate(["don", "don", "ka", "don"])
    ]


def _developed_chart(n_bars=40):
    """2-bar figures, restated with variation, sections return."""
    figs = [
        [(0.0, "don"), (1.0, "don"), (2.0, "ka"), (3.0, "don")],
        [(0.0, "don"), (0.5, "don"), (1.0, "ka"), (2.0, "ka"), (3.0, "don")],
        [(0.0, "ka"), (1.0, "don"), (1.5, "don"), (2.0, "ka"), (3.0, "don")],
    ]
    return [
        (b * BAR + off * BEAT, c) for b in range(n_bars) for off, c in figs[(b // 2) % 3]
    ]


# ------------------------------------------------- reciprocity v2 (structure) --
def test_reciprocity_v2_abab_full_credit():
    """Textbook ABAB earns full reciprocity; the old hash leaf scored it 0.0."""
    out = structure.call_response_reciprocity(_abab_chart(), _ctx(32))
    # pinned source values
    assert out["reciprocity"] == 1.0
    assert out["call_response_rate"] == 1.0
    assert out["mean_rhythm_similarity"] == 1.0
    assert out["mean_color_contrast"] == 0.5
    assert out["n_phrases"] == 16.0
    # the floor-bound legacy leaf (kept as diagnostic) measures 0.0 here --
    # the very measurement failure that forced the v2 redesign
    assert out["reciprocity_hash"] == 0.0


def test_reciprocity_v2_copy_paste_scores_zero():
    """Copy-paste collapse (same rhythm AND colour every phrase) earns nothing."""
    ev = [(ph * BAR + k * BEAT, "don") for ph in range(0, 32, 2) for k in range(6)]
    out = structure.call_response_reciprocity(ev, _ctx(32))
    assert out["reciprocity"] < 1e-9


def test_reciprocity_v2_developed_pinned():
    """Fixed-value equivalence pin from the source implementation."""
    out = structure.call_response_reciprocity(_developed_chart(), _ctx(40))
    assert out["reciprocity"] == 0.19148671789949429


def test_reciprocity_v2_needs_grid():
    out = structure.call_response_reciprocity(_abab_chart(), make_ctx(course="oni", bpm=BPM))
    assert math.isnan(out["reciprocity"])


# ------------------------------------------------------ boredom_v2 (structure) --
def test_boredom_v2_loop_stagnation_pinned():
    """Verbatim looping reads as stagnation (pinned source values)."""
    out = structure.boredom_v2(_loop_chart(), _ctx(40))
    assert out["boredom_v2_raw"] == 0.925
    assert out["stagnation_frac"] == 0.925
    assert out["alienation_frac"] == 0.0
    assert out["n_bars_active"] == 40.0


def test_boredom_v2_developed_beats_loop_and_chaos():
    """Repetition-with-variation is the engaged optimum (Berlyne inverted-U)."""
    ctx = _ctx(40)
    dev = structure.boredom_v2(_developed_chart(), ctx)
    loop = structure.boredom_v2(_loop_chart(), ctx)
    rng = random.Random(0)
    scatter, t = [], 0.0
    while t < 40 * BAR:
        scatter.append((t, rng.choice(["don", "ka"])))
        t += rng.choice([BEAT / 4, BEAT / 3, BEAT / 2, BEAT * 0.75, BEAT])
    rand = structure.boredom_v2(scatter, ctx)
    # pinned source value for the developed chart
    assert dev["boredom_v2_raw"] == 0.0
    # both failure extremes score worse than the developed chart
    assert loop["boredom_v2_raw"] > dev["boredom_v2_raw"]
    assert rand["boredom_v2_raw"] > dev["boredom_v2_raw"]


def test_boredom_v2_invalid_inputs_nan():
    """No grid and no valid bpm (bpm<=0) -> NaN, never a fabricated score."""
    ev = _loop_chart()
    out = structure.boredom_v2(ev, make_ctx(course="oni", bpm=0.0))
    assert math.isnan(out["boredom_v2_raw"])
    out2 = structure.boredom_v2(ev[:4], _ctx(40))  # too few hits
    assert math.isnan(out2["boredom_v2_raw"])


def test_structure_compute_exports_v2_keys():
    out = structure.compute(_developed_chart(), _ctx(40))
    for key in ("reciprocity", "reciprocity_hash", "boredom_v2_raw", "stagnation_frac"):
        assert key in out


# ------------------------------------------------------------ phi v2 (gap) ----
def _mixed_chart():
    pat = ["don", "ka", "don", "don_big", "ka", "ka_big", "don", "ka"]
    return [(i * BEAT / 2, pat[i % 8]) for i in range(200)]


def test_gap_phi_v2_pinned_vector():
    """Fixed-value equivalence pin: the full 32-dim phi from the source module."""
    expected = [
        0.0, 0.0, 0.0, 0.0, 5.333333333333333, 0.0, 0.0, 0.0,
        0.0, 0.25125628140703515, 0.12562814070351758, 0.0,
        0.24623115577889448, 0.0, 0.0, 0.12562814070351758,
        0.0, 0.12562814070351758, 0.0, 0.0,
        0.12562814070351758, 0.0, 0.0, 0.0,
        2.0, 2.0, 2.0,
        0.04568527918781726, 0.25,
        1.811278124459133, 2.5024577319614685, 0.7487437185929648,
    ]
    p = gap.phi(_mixed_chart(), BPM)
    assert p is not None and len(p) == gap.PHI_DIM == 32
    assert p.tolist() == expected


def test_gap_phi_v2_variety_dims_fall_under_blandification():
    """The three appended variety dims are monotone anti-blandness axes."""
    varied = gap.phi(_mixed_chart(), BPM)
    bland = gap.phi([(i * BEAT / 2, "don") for i in range(200)], BPM)
    # dims: -3 type unigram entropy, -2 bigram entropy, -1 colour switch rate
    assert bland[-3] < varied[-3]
    assert bland[-2] < varied[-2]
    assert bland[-1] == 0.0 < varied[-1]


# ------------------------------------------------- coupling: audio candidates --
def _ramp_mel(n_bars=32):
    """Deterministic mel: beat-locked onsets whose height ramps with bar index."""
    T = int(round(n_bars * BAR * SR / HOP))
    mel = np.full((N_MELS, T), -6.0)
    for b in range(n_bars):
        h = 1.0 + 3.0 * (b / n_bars)
        for q in range(4):
            f = int(round((b * BAR + q * BEAT) * FPS))
            if 0 <= f < T:
                mel[:, f] += h
    return mel


def _tracking_chart(n_bars=32, invert=False):
    """Quarter notes in one half, eighths in the other (density ramp)."""
    ev = []
    for b in range(n_bars):
        dense = (b >= n_bars // 2) ^ invert
        step = BEAT / 2 if dense else BEAT
        t = b * BAR
        while t < (b + 1) * BAR - 1e-9:
            ev.append((t, "don" if int(round(t / step)) % 2 == 0 else "ka"))
            t += step
    return ev


def test_density_energy_tracking_pinned():
    """Density that follows the energy ramp couples positively (pinned values)."""
    out = coupling.density_energy_response(_tracking_chart(), _ctx(32, mel=_ramp_mel()))
    assert out["density_energy_spearman"] == 0.8664485777182117
    assert out["density_energy_partial"] == -0.25733137829912023
    assert out["density_energy_lag_robust"] == 0.8664485777182117
    assert out["density_energy_n_windows"] == 32.0


def test_density_energy_inverted_is_negative():
    """Cramming the quiet half and emptying the loud half flips the sign."""
    out = coupling.density_energy_response(
        _tracking_chart(invert=True), _ctx(32, mel=_ramp_mel())
    )
    assert out["density_energy_spearman"] < -0.5


def test_density_energy_missing_mel_nan():
    out = coupling.density_energy_response(_tracking_chart(), _ctx(32))
    assert math.isnan(out["density_energy_spearman"])


def _run_chart(n_bars=32):
    """Big anchor on each downbeat + an 8-note 16th run from beat 2."""
    ev = []
    for b in range(n_bars):
        ev.append((b * BAR, "don_big"))
        ev.extend(
            (b * BAR + 2 * BEAT + j * BEAT / 4, "don" if j % 2 else "ka") for j in range(8)
        )
    return ev


def test_energy_peak_run_head_pinned():
    """Run-head normalization: one probe per perceptual group (pinned values).

    288 hits collapse to 33 run-heads (the 16th runs bridge into the next
    downbeat, so each bar contributes exactly one >=0.25 s group boundary).
    """
    out = coupling.energy_peak_support_rate(_run_chart(), _ctx(32, mel=_ramp_mel()))
    assert out["energy_n_hits"] == 288.0
    assert out["energy_n_heads"] == 33.0
    assert out["energy_support_rate_raw"] == 1.0
    assert out["energy_support_rate_all"] == 1.0
    assert out["energy_support_rate_global"] == 0.9696969696969697
    assert out["energy_peak_mean_z"] == 1.636909684097685
    assert out["energy_at_hit_z"] == 1.636909684097685
    assert out["energy_theta_song"] == -0.16781657080778872


def test_energy_peak_head_rate_decouples_from_run_interiors():
    """Off-onset run interiors do not dilute the head rate; off-onset heads do.

    Chart A anchors a dense 16th run on every spiked downbeat: its 256 interior
    notes are off-onset but sub-group (IOI < 0.25 s), so the head-based support
    stays 1.0. Chart B plays sparse off-onset hits (every hit its own head) and
    scores 0.0 on the identical mel/threshold.
    """
    n_bars = 33  # bar 0 stays silent: flux needs a rise, undefined at frame 0
    T = int(round(n_bars * BAR * SR / HOP))
    mel = np.full((N_MELS, T), -6.0)
    for b in range(1, n_bars):
        mel[:, int(round(b * BAR * FPS))] += 8.0  # onset spike on each downbeat
    ev_runs, ev_sparse = [], []
    for b in range(1, n_bars):
        ev_runs.append((b * BAR, "don_big"))  # on the spike; the group head
        ev_runs.extend(
            (b * BAR + j * BEAT / 4, "don" if j % 2 else "ka") for j in range(1, 9)
        )
        ev_sparse.extend(
            (b * BAR + q * BEAT + BEAT / 2, "don") for q in range(4)  # off-spike
        )
    ctx = _ctx(n_bars, mel=mel)
    ctx["energy_theta_global"] = 1.0  # above the quiet floor, below the spikes
    runs = coupling.energy_peak_support_rate(ev_runs, ctx)
    sparse = coupling.energy_peak_support_rate(ev_sparse, ctx)
    assert runs["energy_n_hits"] == 288.0 and runs["energy_n_heads"] == 32.0
    assert runs["energy_support_rate_global"] == 1.0
    assert sparse["energy_n_heads"] == sparse["energy_n_hits"]  # no grouping
    assert sparse["energy_support_rate_global"] == 0.0


def test_energy_peak_missing_mel_nan():
    out = coupling.energy_peak_support_rate(_run_chart(), _ctx(32))
    for k in ("energy_support_rate_raw", "energy_support_rate_all", "energy_n_heads"):
        assert math.isnan(out[k])
