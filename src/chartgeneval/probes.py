"""Corruption probes C1-C8: targeted, seed-deterministic degradations.

Each probe is a construction-guaranteed degradation of a chart's event stream,
applied at three increasing doses. They operate purely on ``(t, class)`` events;
audio is untouched. Every probe is seeded deterministically from
``(sid, course, probe, dose_index)`` via SHA-256, so two runs with the same key
produce bit-identical output.

Probe -> target dimension (paper Table):

    C1 timing_jitter    -> timing / rhythm complexity
    C1s sparse_jitter   -> timing tail (sparse severe outliers; mean-blind)
    C2 anchor_shift     -> anchoring (designed chart-only blind control)
    C3 type_shuffle     -> transition validity / pattern IC
    C4 loop_collapse    -> repetition / variety / boredom
    C5 blandification   -> pattern IC (LM argmax resample; needs ctx['lm'])
    C6 density_scale    -> density / strain constraints
    C7 burst_insert     -> overload / spike / chaos
    C8 bar_shuffle      -> global structure / repetition

Each operator returns ``(new_events, noop_flag)``; ``noop_flag`` is True when the
chart is too short / degenerate for the probe to apply.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

from .events import HIT_CLASSES, sorted_hits
from .metrics.common import event_tokens, fold_hit, size_hit

LM_ORDER = 3

NOTE_BY_COLOR_SIZE = {
    ("D", "S"): "don",
    ("K", "S"): "ka",
    ("D", "B"): "don_big",
    ("K", "B"): "ka_big",
}
NOTE_TOKENS = ["DS", "KS", "DB", "KB"]
NOTE_TOKEN_TO_CLASS = {"DS": "don", "KS": "ka", "DB": "don_big", "KB": "ka_big"}


def seed_for(sid, course, probe, dose_index) -> int:
    """Deterministic 32-bit seed for a (sid, course, probe, dose) cell."""
    key = f"{sid}|{course}|{probe}|{dose_index}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def _flip_color_class(cls):
    color = "K" if fold_hit(cls) == "D" else "D"
    return NOTE_BY_COLOR_SIZE[(color, size_hit(cls))]


def c1_timing_jitter(hits, delta_ms, rng, bpm):
    if len(hits) < 2:
        return hits, True
    out = []
    for t, c in hits:
        jit = float(rng.uniform(-delta_ms, delta_ms)) / 1000.0
        out.append((max(0.0, t + jit), c))
    return sorted(out), False


def c1s_sparse_jitter(hits, p, rng, bpm, mag_ms=60.0):
    """Sparse severe outliers: displace a fraction ``p`` of notes by a large,
    clearly perceptible offset (10x the 6 ms relative-JND floor), random sign.

    The affected count is ``max(1, round(n * p))`` so the probe always applies.
    Mean-based timing summaries barely move under this probe; tail statistics
    (P99, violation rates) must respond.
    """
    n = len(hits)
    if n < 4:
        return hits, True
    k = max(1, int(round(n * p)))
    idx = set(int(i) for i in rng.choice(n, size=k, replace=False))
    off = mag_ms / 1000.0
    out = []
    for i, (t, c) in enumerate(hits):
        if i in idx:
            sign = 1.0 if rng.random() < 0.5 else -1.0
            t = max(0.0, t + sign * off)
        out.append((t, c))
    return sorted(out), False


def c2_anchor_shift(hits, shift_ms, rng, bpm):
    if not hits:
        return hits, True
    off = shift_ms / 1000.0
    return [(max(0.0, t + off), c) for t, c in hits], False


def c3_type_shuffle(hits, p, rng, bpm):
    if len(hits) < 4:
        return hits, True
    out = []
    for t, c in hits:
        if rng.random() < p:
            c = _flip_color_class(c)
        out.append((t, c))
    return out, False


def _most_common_4cycle(hits):
    notes = [fold_hit(c) + size_hit(c) for _, c in hits]
    if len(notes) < 4:
        return None
    grams = Counter(tuple(notes[i : i + 4]) for i in range(len(notes) - 3))
    if not grams:
        return None
    best = max(grams.items(), key=lambda kv: (kv[1], tuple(-ord(ch) for ch in "".join(kv[0]))))
    return best[0]


def c4_loop_collapse(hits, p, rng, bpm):
    if len(hits) < 4:
        return hits, True
    cycle = _most_common_4cycle(hits)
    if cycle is None:
        return hits, True
    times = [t for t, _ in hits]
    notes = [fold_hit(c) + size_hit(c) for _, c in hits]
    k = int(round(p * len(hits)))
    for i in range(k):
        notes[i] = cycle[i % 4]
    out = [(times[i], NOTE_TOKEN_TO_CLASS[notes[i]]) for i in range(len(hits))]
    return sorted(out), False


def c5_blandification(hits, p, rng, bpm, lm, course):
    if len(hits) < LM_ORDER + 1 or lm is None:
        return hits, True
    tokens = event_tokens(hits, bpm)
    times = [t for t, _ in hits]
    n = len(tokens)
    k = int(round(p * n))
    if k <= 0:
        return hits, True
    if k >= n:
        targets = set(range(n))
    else:
        step = n / k
        targets = {int(np.floor(j * step)) for j in range(k)}
    for i in range(n):
        if i < LM_ORDER - 1 or i not in targets:
            continue
        ioi_i = tokens[i].split(":", 1)[0]
        ctx = tuple(tokens[i - (LM_ORDER - 1) : i])
        best_note, best_p = None, -1.0
        for nt in NOTE_TOKENS:
            cand = f"{ioi_i}:{nt}"
            pr = lm._prob(course, ctx + (cand,))
            if pr > best_p:
                best_p, best_note = pr, nt
        tokens[i] = f"{ioi_i}:{best_note}"
    out = []
    for i in range(n):
        note = tokens[i].split(":", 1)[1]
        out.append((times[i], NOTE_TOKEN_TO_CLASS[note]))
    return sorted(out), False


def c6_density_scale(hits, f, rng, bpm):
    n = len(hits)
    if n < 4:
        return hits, True
    if f < 1.0:
        target = max(2, int(round(f * n)))
        if target >= n:
            return hits, True
        interior = list(range(1, n - 1))
        n_keep_interior = target - 2
        if n_keep_interior <= 0:
            keep = [0, n - 1]
        else:
            chosen = rng.choice(len(interior), size=n_keep_interior, replace=False)
            keep = sorted([0, n - 1] + [interior[j] for j in chosen])
        return [hits[i] for i in keep], False
    n_add = int(round((f - 1.0) * n))
    if n_add <= 0:
        return hits, True
    out = list(hits)
    for _ in range(n_add):
        i = int(rng.integers(0, n - 1))
        t_new = (hits[i][0] + hits[i + 1][0]) / 2.0
        out.append((t_new, hits[i][1]))
    return sorted(out), False


def c7_burst_insert(hits, k, rng, bpm):
    if len(hits) < 2:
        return hits, True
    t_lo = hits[0][0]
    t_hi = hits[-1][0]
    if t_hi <= t_lo:
        return hits, True
    small_colors = ["don", "ka"]
    out = list(hits)
    for _ in range(int(k)):
        anchor = float(rng.uniform(t_lo, t_hi))
        m = int(rng.integers(8, 17))
        nps = float(rng.uniform(12.0, 20.0))
        gap = 1.0 / nps
        for j in range(m):
            t = anchor + j * gap
            c = small_colors[int(rng.integers(0, 2))]
            out.append((t, c))
    return sorted(out), False


def c8_bar_shuffle(hits, p, rng, bpm):
    if len(hits) < 4 or not bpm or bpm <= 0:
        return hits, True
    bar_s = 4.0 * 60.0 / bpm
    t0 = hits[0][0]
    bars = defaultdict(list)
    for t, c in hits:
        b = int(np.floor((t - t0) / bar_s))
        bars[b].append((t, c))
    keys = sorted(bars)
    if len(keys) < 2:
        return hits, True
    selected = [b for b in keys if rng.random() < p]
    if len(selected) < 2:
        return hits, True
    perm = list(selected)
    rng.shuffle(perm)
    mapping = dict(zip(selected, perm))
    out = []
    for b in keys:
        src = mapping.get(b, b)
        for t, c in bars[src]:
            out.append((t - src * bar_s + b * bar_s, c))
    return sorted(out), False


# probe id -> (operator, dose tuple, needs_lm)
PROBES = {
    "C1_timing_jitter": (c1_timing_jitter, (10, 20, 30), False),
    "C1s_sparse_jitter": (c1s_sparse_jitter, (0.005, 0.01, 0.02), False),
    "C2_anchor_shift": (c2_anchor_shift, (15, 30, 60), False),
    "C3_type_shuffle": (c3_type_shuffle, (0.2, 0.4, 0.8), False),
    "C4_loop_collapse": (c4_loop_collapse, (0.3, 0.6, 1.0), False),
    "C5_blandification": (c5_blandification, (0.3, 0.6, 1.0), True),
    "C6_density_scale": (c6_density_scale, (0.5, 1.5, 2.0), False),
    "C7_burst_insert": (c7_burst_insert, (2, 4, 8), False),
    "C8_bar_shuffle": (c8_bar_shuffle, (0.3, 0.6, 1.0), False),
}

PROBE_ORDER = list(PROBES.keys())


def apply_probe(probe, events, dose_value, rng, bpm, lm=None, course=None):
    """Apply one probe operator at a dose. Returns ``(new_events, noop_flag)``."""
    hits = sorted_hits(events)
    op, _doses, needs_lm = PROBES[probe]
    if needs_lm:
        return op(hits, dose_value, rng, bpm, lm, course)
    return op(hits, dose_value, rng, bpm)


def corrupt(events, probe, dose_index, *, sid, course, bpm, lm=None):
    """Deterministically apply ``probe`` at ``dose_index`` (1..3).

    ``dose_index`` 0 returns the untouched chart. The RNG is seeded from
    ``(sid, course, probe, dose_index)``, giving reproducible output.
    """
    if dose_index == 0:
        return sorted_hits(events), False
    _op, doses, _needs = PROBES[probe]
    dose_value = doses[dose_index - 1]
    rng = np.random.default_rng(seed_for(sid, course, probe, dose_index))
    return apply_probe(probe, events, dose_value, rng, bpm, lm=lm, course=course)
