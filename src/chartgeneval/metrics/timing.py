"""Timing family: audio-anchored alignment against an estimated beat grid.

Note times are matched to a hierarchical meter lattice laid inside each grid
span (from :mod:`chartgeneval.grid`), separating unsupported notes, absolute
offset, relative-interval distortion, and signed bias. It also computes the
fixed-grid ``grid_phase_offset`` that survives re-matching and is the sole
witness of a global anchor shift (probe C2).

Ported from the reference perceptual-timing module with the internal audio
onset path removed (the metric here is anchored on the meter lattice; audio
onset anchors were an ablation and required internal mel caches).

Requires an externally supplied authored or audio-derived grid. BPM alone does
not identify phase and is therefore never used to construct a chart-derived
reference. The optimal-assignment matcher requires scipy; if scipy is absent,
only the scipy-free ``grid_phase_offset`` witness is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ..events import sorted_hits

METRIC_NAME = "timing"

# Perception-grounded thresholds (timing just-noticeable-difference literature).
# These are perceptual constants, distinct from the matcher engineering
# parameters (anchor gates 10-50 ms, merge window 4 ms) defined below.
# The absolute deadzone is aligned with the relative JND floor: a single
# 6 ms perceptual constant. Deviations below it are imperceptible; violation
# tiers are expressed as multiples of it (1x detectable, 2x/3x severity).
TAU_ABS_S = 0.006  # absolute deviation deadzone: deviations below are imperceptible
REL_FLOOR_S = 0.006  # relative interval-distortion detection floor
REL_FRAC = 0.025  # Weber fraction of the local inter-anchor interval
VIOLATION_TIERS = (1, 2, 3)  # exceedance tiers as multiples of the deadzone


@dataclass(frozen=True)
class Anchor:
    time: float
    salience: float
    kind: str


def _put_frac(out, q, salience, kind):
    q = Fraction(q).limit_denominator(256)
    if q < 0 or q > 1:
        return
    prev = out.get(q)
    if prev is None or salience > prev[0]:
        out[q] = (float(salience), kind)


def _meter_group_boundaries(num, den):
    if den == 8 and num in (6, 9, 12):
        return [Fraction(i, num // 3) for i in range(num // 3 + 1)]
    if den == 8 and num == 5:
        acc = 0
        out = [Fraction(0, 1)]
        for g in (3, 2):
            acc += g
            out.append(Fraction(acc, num))
        return out
    if den == 8 and num == 7:
        acc = 0
        out = [Fraction(0, 1)]
        for g in (2, 2, 3):
            acc += g
            out.append(Fraction(acc, num))
        return out
    return []


def _coerce_meter(num=4, den=4):
    try:
        num = int(num or 4)
        den = int(den or 4)
    except Exception:
        num, den = 4, 4
    if num <= 0 or den <= 0:
        num, den = 4, 4
    return num, den


def _is_regular_meter(num, den):
    if den not in (1, 2, 4, 8, 16):
        return False
    if num <= 0 or num > 16:
        return False
    quarter_units = 4.0 * num / den
    return abs(quarter_units - round(quarter_units)) < 1e-6 or den in (8, 16)


def _put_span_subdivisions(out, start, stop, divs, salience, kind):
    if stop <= start:
        return
    span = stop - start
    for d in divs:
        for i in range(1, d):
            _put_frac(out, start + span * Fraction(i, d), salience, kind)


def meter_fractions(num=4, den=4, include_fine=False):
    """Hierarchical meter lattice fractions in one span with salience labels."""
    num, den = _coerce_meter(num, den)
    out = {}
    _put_frac(out, Fraction(0, 1), 1.0, "bar")
    _put_frac(out, Fraction(1, 1), 1.0, "bar")

    for q in _meter_group_boundaries(num, den):
        _put_frac(out, q, 0.88, "pulse")

    regular_meter = _is_regular_meter(num, den)
    units = max(1, num) if regular_meter else 0
    if units:
        for i in range(units + 1):
            _put_frac(out, Fraction(i, units), 0.78, "beat")

    span_divs = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32)
    for d in span_divs:
        for i in range(1, d):
            if d <= 4:
                sal = 0.50
            elif d <= 8:
                sal = 0.38
            else:
                sal = 0.28
            _put_frac(out, Fraction(i, d), sal, "span_subdivision")

    if regular_meter:
        for i in range(units):
            a = Fraction(i, units)
            b = Fraction(i + 1, units)
            _put_span_subdivisions(out, a, b, (2, 3, 4), 0.58, "subdivision")
            _put_span_subdivisions(out, a, b, (6, 8, 12), 0.44, "dense_subdivision")
            _put_span_subdivisions(out, a, b, (5,), 0.32, "tuplet")
            if include_fine:
                _put_span_subdivisions(out, a, b, (16,), 0.24, "fine_grid")

    pulses = sorted(set(_meter_group_boundaries(num, den)))
    for a, b in zip(pulses, pulses[1:]):
        _put_span_subdivisions(out, a, b, (2, 3, 4), 0.52, "pulse_subdivision")

    return out


def estimated_grid_anchors(downbeats, audio_duration=None, meter_num=4, meter_den=4, include_fine=False):
    """Build meter-aware anchors from an estimated downbeat grid."""
    db = np.asarray(sorted(float(x) for x in downbeats), dtype=np.float64)
    if len(db) < 2:
        return []
    fracs = meter_fractions(meter_num, meter_den, include_fine=include_fine)
    anchors = []
    for a, b in zip(db[:-1], db[1:]):
        span = b - a
        if span <= 0:
            continue
        for q, (sal, kind) in fracs.items():
            t = a + float(q) * span
            if audio_duration is not None and not (-0.05 <= t <= audio_duration + 0.05):
                continue
            anchors.append(Anchor(float(t), sal, f"meter:{kind}"))
    return anchors


def segment_grid_anchors(segments, audio_duration=None, include_fine=False):
    """Build meter-aware anchors from parsed TJA segment timing metadata.

    One segment per authored measure: ``timestamp`` + ``measure_num/den`` +
    per-note ``bpm``. Each bar gets its own meter lattice, so variable-tempo
    and variable-meter charts (3/4<->4/4, 5/16 bars, mid-song BPM ramps) are
    anchored exactly. This is the authored-grid tier of the grid reference
    hierarchy; flattened downbeat lists lose the per-bar meter and misanchor
    such charts.
    """
    if not segments:
        return []
    segs = [s for s in segments if s.get("timestamp") is not None]
    segs.sort(key=lambda s: float(s["timestamp"]))
    if not segs:
        return []

    anchors = []
    for i, seg in enumerate(segs):
        start = float(seg["timestamp"])
        if i + 1 < len(segs):
            end = float(segs[i + 1]["timestamp"])
        else:
            end = None
            bpm = None
            notes = seg.get("notes") or []
            if notes and notes[0].get("bpm"):
                bpm = float(notes[0]["bpm"])
            if bpm is None and i > 0:
                notes = segs[i - 1].get("notes") or []
                if notes and notes[0].get("bpm"):
                    bpm = float(notes[0]["bpm"])
            if bpm and bpm > 0:
                num = float(seg.get("measure_num") or 4)
                den = float(seg.get("measure_den") or 4)
                end = start + (240.0 / bpm) * (num / den)
            elif audio_duration is not None:
                end = float(audio_duration)
        if end is None:
            continue
        if audio_duration is not None and start > audio_duration + 0.1:
            continue
        end = min(end, float(audio_duration)) if audio_duration is not None else end
        if end <= start:
            anchors.append(Anchor(float(start), 1.0, "meter:bar"))
            continue
        span = end - start
        for q, (sal, kind) in meter_fractions(
            seg.get("measure_num") or 4,
            seg.get("measure_den") or 4,
            include_fine=include_fine,
        ).items():
            t = start + float(q) * span
            if audio_duration is not None and not (-0.05 <= t <= audio_duration + 0.05):
                continue
            anchors.append(Anchor(float(t), sal, f"meter:{kind}"))
    return anchors


def merge_anchors(anchors, merge_ms=4.0):
    if not anchors:
        return []
    merge_s = merge_ms / 1000.0
    xs = sorted(anchors, key=lambda a: a.time)
    groups = []
    cur = [xs[0]]
    for a in xs[1:]:
        if a.time - cur[-1].time <= merge_s:
            cur.append(a)
        else:
            groups.append(cur)
            cur = [a]
    groups.append(cur)
    merged = []
    for g in groups:
        best = max(g, key=lambda a: (a.salience, -abs(a.time)))
        sal = min(1.0, max(a.salience for a in g) + 0.03 * (len(g) - 1))
        kinds = "+".join(sorted(set(a.kind for a in g)))
        merged.append(Anchor(float(best.time), float(sal), kinds))
    return merged


def finalize_anchors(anchors, max_gate_s=0.050, min_gate_s=0.010):
    anchors = sorted(anchors, key=lambda a: a.time)
    if not anchors:
        return {"time": np.zeros(0), "salience": np.zeros(0), "gate": np.zeros(0), "kind": []}
    t = np.asarray([a.time for a in anchors], dtype=np.float64)
    sal = np.asarray([a.salience for a in anchors], dtype=np.float64)
    prev_d = np.concatenate([[np.inf], np.diff(t)])
    next_d = np.concatenate([np.diff(t), [np.inf]])
    nearest = np.minimum(prev_d, next_d)
    gate = np.minimum(float(max_gate_s), 0.5 * nearest)
    gate[~np.isfinite(gate)] = float(max_gate_s)
    gate = np.maximum(gate, float(min_gate_s))
    return {"time": t, "salience": sal, "gate": gate, "kind": [a.kind for a in anchors]}


def match_notes_to_anchors(note_times, anchors, lambda_salience=0.08, dummy_cost=1.25):
    """One-to-one min-cost matching of notes to anchors within gates."""
    from scipy.optimize import linear_sum_assignment

    times = np.asarray(note_times, dtype=np.float64)
    g = np.asarray(anchors["time"], dtype=np.float64)
    sal = np.asarray(anchors["salience"], dtype=np.float64)
    gate = np.asarray(anchors["gate"], dtype=np.float64)
    kinds = anchors.get("kind", [])
    n, m = len(times), len(g)
    if n == 0:
        return [], []
    if m == 0:
        return [], list(range(n))

    big = np.float32(1e6)
    cost = np.full((n, m + n), big, dtype=np.float32)
    for i, t in enumerate(times):
        lo = np.searchsorted(g, t - 0.050, side="left")
        hi = np.searchsorted(g, t + 0.050, side="right")
        if hi > lo:
            d = np.abs(g[lo:hi] - t)
            ok = d <= gate[lo:hi]
            if np.any(ok):
                js = np.nonzero(ok)[0] + lo
                cost[i, js] = (d[ok] / gate[js] - lambda_salience * sal[js]).astype(np.float32)
        cost[i, m + i] = np.float32(dummy_cost)

    rows, cols = linear_sum_assignment(cost)
    matches = []
    unsupported = []
    for r, c in zip(rows, cols):
        if c < m and cost[r, c] < dummy_cost:
            matches.append(
                {
                    "note_index": int(r),
                    "anchor_index": int(c),
                    "t": float(times[r]),
                    "g": float(g[c]),
                    "error": float(times[r] - g[c]),
                    "gate": float(gate[c]),
                    "salience": float(sal[c]),
                    "kind": kinds[c] if c < len(kinds) else "",
                }
            )
        else:
            unsupported.append(int(r))
    matches.sort(key=lambda x: (x["g"], x["t"]))
    unsupported.sort()
    return matches, unsupported


def _summary_ms(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return None, None
    return float(np.mean(x) * 1000.0), float(np.percentile(x, 95) * 1000.0)


def _percentile_ms(x, q):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return None
    return float(np.percentile(x, q) * 1000.0)


def grid_phase_offset(note_times, beat_period, phase, subdiv=2, tol_ms=12.0, n_steps=241):
    """Global signed anchor offset against fixed eighth + beat lattices.

    This is the C2-sensitive detector that the re-matching offset is blind to.
    A constant anchor shift moves the whole stream off the grid; the support
    objective averages eighth-lattice and beat-lattice agreement, so its
    period is one full beat and the argmax is unique over +-beat/2. (An
    eighth-only objective wraps at +-eighth/2 and misreads >=60 ms shifts on
    high-BPM charts.) Returns ``(offset_ms, support_frac, step_ms)`` with
    ``step_ms`` the fine (eighth) lattice step.
    """
    t = np.asarray(note_times, dtype=np.float64)
    if len(t) < 4 or not beat_period or beat_period <= 0:
        return None, None, None
    beat = float(beat_period)
    step = beat / float(subdiv)
    step_ms = step * 1000.0
    tol_f = min(tol_ms / 1000.0, 0.45 * step)
    tol_b = min(tol_ms / 1000.0, 0.45 * beat)
    half_f = step / 2.0
    half_b = beat / 2.0
    rel = t - float(phase)
    # Odd candidate count so d = 0 sits exactly on the search grid; otherwise
    # an on-grid chart loses a sliver of support to the endpoint aliases.
    n_c = int(round(n_steps * beat / step)) | 1
    cands = np.linspace(-half_b, half_b, n_c)
    sup = np.empty(len(cands))
    for i, d in enumerate(cands):
        r_f = np.abs(((rel - d + half_f) % step) - half_f)
        r_b = np.abs(((rel - d + half_b) % beat) - half_b)
        sup[i] = 0.5 * np.mean(np.maximum(0.0, 1.0 - r_f / tol_f)) + 0.5 * np.mean(
            np.maximum(0.0, 1.0 - r_b / tol_b)
        )
    best_s = float(sup.max())
    # Near-ties (e.g. beat-symmetric charts, where d and d±beat/2 are truly
    # indistinguishable) resolve to the minimum-norm offset.
    near = sup >= best_s - 1e-9
    best_d = float(cands[near][np.argmin(np.abs(cands[near]))])
    return best_d * 1000.0, best_s, step_ms


def lattice_phase_offset(
    note_times,
    bar_times,
    beats_per_bar=None,
    tol_ms=12.0,
    search_ms=250.0,
    step_search_ms=0.5,
):
    """Piecewise fixed-lattice anchor-shift witness (tempo-change robust).

    The lattice is laid inside each authored bar (eighth + beat + bar-line
    scales, weights 0.5/0.3/0.2), so it is aperiodic under tempo changes: a
    constant-period witness accumulates phase drift on BPM ramps and picks a
    far alias, this one does not. Bar-scale support breaks full-beat aliases;
    near-ties resolve to the minimum-norm offset. Returns
    ``(offset_ms, support_frac, fine_step_ms)``.
    """
    t = np.asarray(note_times, dtype=np.float64)
    bars = np.asarray(sorted(float(x) for x in bar_times), dtype=np.float64)
    if len(t) < 4 or len(bars) < 2:
        return None, None, None

    fine, beat = [], []
    for i in range(len(bars) - 1):
        a, b = bars[i], bars[i + 1]
        if b <= a:
            continue
        m = 4.0
        if beats_per_bar is not None and i < len(beats_per_bar) and beats_per_bar[i]:
            m = float(beats_per_bar[i])
        nb = max(1, int(round(m)))
        for j in range(nb):
            beat.append(a + (b - a) * j / nb)
        for j in range(2 * nb):
            fine.append(a + (b - a) * j / (2 * nb))
    if not fine:
        return None, None, None
    lattices = (
        (np.asarray(sorted(set(fine))), 0.5),
        (np.asarray(sorted(set(beat))), 0.3),
        (bars, 0.2),
    )
    fine_arr = lattices[0][0]
    fine_step_ms = float(np.median(np.diff(fine_arr)) * 1000.0) if len(fine_arr) > 1 else None
    tol = tol_ms / 1000.0

    def dist(x, L):
        idx = np.clip(np.searchsorted(L, x), 1, len(L) - 1)
        return np.minimum(np.abs(x - L[idx - 1]), np.abs(L[idx] - x))

    search = search_ms / 1000.0
    step = step_search_ms / 1000.0
    n_half = int(round(search / step))
    cands = np.arange(-n_half, n_half + 1) * step  # symmetric, includes 0
    sup = np.zeros(len(cands))
    for i, d in enumerate(cands):
        x = t - d
        s = 0.0
        for L, w in lattices:
            if len(L) < 2:
                continue
            s += w * float(np.mean(np.maximum(0.0, 1.0 - dist(x, L) / tol)))
        sup[i] = s
    best_s = float(sup.max())
    # Equivalent aliases (uniform-stream charts, +-one beat) differ by O(1e-4)
    # support; genuinely distinct peaks differ by >=0.05. Minimum-norm among
    # near-ties picks the physical offset, not the alias.
    near = sup >= best_s - 2e-3
    best_d = float(cands[near][np.argmin(np.abs(cands[near]))])
    return best_d * 1000.0, best_s, fine_step_ms


def _segment_bars(segments):
    """(bar_times, beats_per_bar) from parsed TJA segments, incl. final end."""
    segs = [s for s in segments or [] if s.get("timestamp") is not None]
    segs.sort(key=lambda s: float(s["timestamp"]))
    if len(segs) < 2:
        return None, None
    bars = [float(s["timestamp"]) for s in segs]
    beats = []
    for s in segs:
        num = float(s.get("measure_num") or 4)
        den = float(s.get("measure_den") or 4)
        beats.append(num * 4.0 / den)
    # extend one bar past the last line so trailing notes have a span
    bars.append(bars[-1] + (bars[-1] - bars[-2] if bars[-1] > bars[-2] else 1.0))
    return bars, beats


def _empty_result():
    return {
        "n_notes": 0,
        "n_matched": 0,
        "n_unsupported": 0,
        "n_anchors": 0,
        "unsupported_rate": None,
        "matched_rate": None,
        "absolute_tolerance_ms": TAU_ABS_S * 1000.0,
        "absolute_error_mean_ms": None,
        "absolute_error_p90_ms": None,
        "absolute_error_p95_ms": None,
        "absolute_error_p99_ms": None,
        "absolute_offset_abs_mean_ms": None,
        "absolute_offset_abs_p90_ms": None,
        "absolute_offset_abs_p95_ms": None,
        "absolute_offset_abs_p99_ms": None,
        "absolute_offset_abs_max_ms": None,
        "absolute_violation_rate": None,
        "absolute_violation_rate_2x": None,
        "absolute_violation_rate_3x": None,
        "n_absolute_violation": None,
        "n_absolute_violation_2x": None,
        "n_absolute_violation_3x": None,
        "clean_rate": None,
        "relative_interval_abs_mean_ms": None,
        "relative_interval_abs_p90_ms": None,
        "relative_interval_abs_p95_ms": None,
        "relative_interval_abs_p99_ms": None,
        "relative_error_mean_ms": None,
        "relative_error_p90_ms": None,
        "relative_error_p95_ms": None,
        "relative_error_p99_ms": None,
        "relative_violation_rate": None,
        "relative_violation_rate_2x": None,
        "relative_violation_rate_3x": None,
        "n_relative_violation": None,
        "signed_offset_mean_ms": None,
        "signed_offset_median_ms": None,
        "early_rate": None,
        "late_rate": None,
        "anchor_salience_mean": None,
        "grid_phase_offset_ms": None,
        "grid_phase_offset_abs_ms": None,
        "grid_phase_support_frac": None,
        "grid_phase_step_ms": None,
    }


def compute(events, ctx):
    """Timing metrics for a hit stream against the ctx grid.

    Uses ``ctx['grid']`` downbeats or authored segments to build the anchor
    lattice and fixed-grid offset witness. Without that external reference,
    timing values are unavailable even if a BPM is present, because BPM alone
    does not define phase.
    """
    hits = sorted_hits(events)
    note_times = [t for t, _ in hits]
    out = _empty_result()
    out["n_notes"] = len(note_times)
    if len(note_times) < 2:
        return out

    grid = ctx.get("grid") if isinstance(ctx, dict) else None
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    duration = ctx.get("duration") if isinstance(ctx, dict) else None
    if grid and (not bpm):
        bpm = grid.get("bpm")

    segments = grid.get("segments") if grid else None
    downbeats = grid.get("downbeats") if grid else None
    meter = 4
    beat_period = None
    phase = None
    if segments:
        # Authored grid: per-bar meter lattice; fixed-grid witness anchored at
        # the first authored bar line with the course BPM (as the reference
        # runner wires the metadata control).
        ts = sorted(float(s["timestamp"]) for s in segments if s.get("timestamp") is not None)
        if ts and bpm and bpm > 0:
            beat_period = 60.0 / float(bpm)
            phase = ts[0]
    elif downbeats and len(downbeats) >= 2:
        db = sorted(float(x) for x in downbeats)
        bar = grid.get("bar") or float(np.median(np.diff(db)))
        if bpm and bpm > 0:
            m = int(round(bar / (60.0 / bpm)))
            if 2 <= m <= 12:
                meter = m
        beat_period = bar / meter
        phase = db[0]
    # Fixed-lattice global offset (C2 witness), scipy-free. Any bar-line list
    # (authored segments or estimated downbeats) gets the piecewise lattice
    # witness (tempo-change robust, +-250 ms unique window). A constant-grid
    # witness is used only when its phase came from that external grid.
    go = gsupport = gstep = None
    if segments:
        bars, beats = _segment_bars(segments)
        if bars:
            go, gsupport, gstep = lattice_phase_offset(note_times, bars, beats)
    elif downbeats and len(downbeats) >= 2:
        db = sorted(float(x) for x in downbeats)
        go, gsupport, gstep = lattice_phase_offset(note_times, db, [meter] * (len(db) - 1))
    if go is None and beat_period is not None and phase is not None:
        go, gsupport, gstep = grid_phase_offset(note_times, beat_period, phase)
    out["grid_phase_offset_ms"] = go
    out["grid_phase_offset_abs_ms"] = abs(go) if go is not None else None
    out["grid_phase_support_frac"] = gsupport
    out["grid_phase_step_ms"] = gstep

    # Lattice matching (needs scipy + segments or a downbeat grid).
    anchors = None
    if segments:
        try:
            anchors = finalize_anchors(
                merge_anchors(segment_grid_anchors(segments, audio_duration=duration))
            )
        except Exception:
            anchors = None
    if anchors is None or not len(anchors.get("time", [])):
        if not downbeats or len(downbeats) < 2:
            return out
        try:
            anchors = finalize_anchors(
                merge_anchors(estimated_grid_anchors(downbeats, audio_duration=duration, meter_num=meter))
            )
        except Exception:
            return out
    out["n_anchors"] = int(len(anchors.get("time", [])))
    try:
        matches, unsupported = match_notes_to_anchors(note_times, anchors)
    except ImportError:
        return out
    n_notes = len(note_times)
    n_matched = len(matches)
    out["n_matched"] = n_matched
    out["n_unsupported"] = len(unsupported)
    out["unsupported_rate"] = float(len(unsupported) / n_notes) if n_notes else 0.0
    out["matched_rate"] = float(n_matched / n_notes) if n_notes else 1.0
    if n_matched == 0:
        return out

    e = np.asarray([m["error"] for m in matches], dtype=np.float64)
    abs_e = np.abs(e)
    abs_resid = np.maximum(0.0, abs_e - TAU_ABS_S)
    out["absolute_error_mean_ms"], out["absolute_error_p95_ms"] = _summary_ms(abs_resid)
    out["absolute_error_p90_ms"] = _percentile_ms(abs_resid, 90)
    out["absolute_error_p99_ms"] = _percentile_ms(abs_resid, 99)
    out["absolute_offset_abs_mean_ms"], out["absolute_offset_abs_p95_ms"] = _summary_ms(abs_e)
    out["absolute_offset_abs_p90_ms"] = _percentile_ms(abs_e, 90)
    out["absolute_offset_abs_p99_ms"] = _percentile_ms(abs_e, 99)
    out["absolute_offset_abs_max_ms"] = float(np.max(abs_e) * 1000.0)
    for tier in VIOLATION_TIERS:
        suffix = "" if tier == 1 else f"_{tier}x"
        mask = abs_e > tier * TAU_ABS_S
        out[f"absolute_violation_rate{suffix}"] = float(np.mean(mask))
        out[f"n_absolute_violation{suffix}"] = int(np.sum(mask))
    # Clean rate: fraction of ALL notes (unsupported included in the
    # denominator) that are matched to an anchor within the deadzone.
    n_clean = n_matched - out["n_absolute_violation"]
    out["clean_rate"] = float(n_clean / n_notes) if n_notes else None
    out["signed_offset_mean_ms"] = float(np.mean(e) * 1000.0)
    out["signed_offset_median_ms"] = float(np.median(e) * 1000.0)
    out["early_rate"] = float(np.mean(e < -TAU_ABS_S))
    out["late_rate"] = float(np.mean(e > TAU_ABS_S))
    out["anchor_salience_mean"] = float(np.mean([m["salience"] for m in matches]))

    if n_matched >= 2:
        g = np.asarray([m["g"] for m in matches], dtype=np.float64)
        delta = np.diff(g)
        de = np.diff(e)
        ok = delta > 1e-6
        if np.any(ok):
            r = np.abs(de[ok])
            theta = np.maximum(REL_FLOOR_S, REL_FRAC * delta[ok])
            rel_resid = np.maximum(0.0, r - theta)
            out["relative_interval_abs_mean_ms"], out["relative_interval_abs_p95_ms"] = _summary_ms(r)
            out["relative_interval_abs_p90_ms"] = _percentile_ms(r, 90)
            out["relative_interval_abs_p99_ms"] = _percentile_ms(r, 99)
            out["relative_error_mean_ms"], out["relative_error_p95_ms"] = _summary_ms(rel_resid)
            out["relative_error_p90_ms"] = _percentile_ms(rel_resid, 90)
            out["relative_error_p99_ms"] = _percentile_ms(rel_resid, 99)
            # r is the raw (pre-deadzone) interval distortion; the deadzone
            # envelope theta is applied only here, after the relative
            # deviation is computed.
            for tier in VIOLATION_TIERS:
                suffix = "" if tier == 1 else f"_{tier}x"
                mask = r > tier * theta
                out[f"relative_violation_rate{suffix}"] = float(np.mean(mask))
                if tier == 1:
                    out["n_relative_violation"] = int(np.sum(mask))
    return out
