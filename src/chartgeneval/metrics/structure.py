"""Structure family: surface variety, repetition, boredom, colour, reciprocity.

Covers the long-range organisation of a chart: 4-gram surface variety,
repetition rate, windowed boredom, colour-switch adequacy, plus two adopted
suite-v2 candidates ported bitwise-identically from the fixed SoftChart
research modules (``experiments/metric_candidates_v1/candidates/``):

* ``call_response_reciprocity`` -- call-and-response phrase structure, with the
  redesigned continuous ``reciprocity`` v2 as the primary leaf (the old
  exact-hash value survives as the ``reciprocity_hash`` diagnostic).
* ``boredom_v2`` -- two-sided engagement failure (stagnation + alienation),
  replacing the single-sided incumbent ``boredom_rate`` heuristic as the
  quality read (the incumbent stays as a raw diagnostic).

Band scores need ``ctx['calibration']``; reciprocity and boredom_v2 need
``ctx['grid']`` (boredom_v2 alternatively accepts a valid ``ctx['bpm']``).
Missing/invalid inputs (no grid, bpm <= 0, too few hits/bars/phrases) yield
NaN outputs, never silent garbage.
"""

from __future__ import annotations

import numpy as np

from ..calibration import score_chart_features
from ..events import sorted_hits
from .common import compute_chart_features, fold_hit
from .grid_utils import jaccard, resolve_grid

METRIC_NAME = "structure"

# --- adopted v2 candidate: call_response_reciprocity ---
TAU_R = 0.6
COLOR_LO, COLOR_HI = 0.3, 0.9
PHRASE_BARS = 2
SUBDIV = 4
MIN_PHRASES = 4


def _phrase_slots(hits, p_start, p_end, subdiv, meter):
    phrase_len = p_end - p_start
    if phrase_len <= 0:
        return set(), {}
    n_slots = int(round(subdiv * meter * PHRASE_BARS))
    if n_slots <= 0:
        return set(), {}
    slot_colors = {}
    slot_set = set()
    for t, cls in hits:
        if t < p_start or t >= p_end:
            continue
        frac = (t - p_start) / phrase_len
        slot = int(round(frac * n_slots))
        if slot < 0 or slot > n_slots:
            continue
        slot_set.add(slot)
        slot_colors.setdefault(slot, []).append(fold_hit(cls))
    main = {}
    for slot, cols in slot_colors.items():
        d = cols.count("D")
        k = cols.count("K")
        main[slot] = "D" if d >= k else "K"
    return slot_set, main


def _match_slots(sa, sb, tol=1):
    """Nearest-slot matching within +-tol. Returns (matched_a_count,
    matched_b_count, matched_pairs [(slot_a, slot_b)] using nearest-b for
    each a-slot)."""
    if not sa or not sb:
        return 0, 0, []
    a = sorted(sa)
    b = sorted(sb)
    b_arr = np.asarray(b, dtype=float)
    pairs = []
    ma = 0
    for s in a:
        j = int(np.searchsorted(b_arr, s))
        best = None
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(b):
                d = abs(b[jj] - s)
                if d <= tol and (best is None or d < abs(b[best] - s)):
                    best = jj
        if best is not None:
            ma += 1
            pairs.append((s, b[best]))
    a_arr = np.asarray(a, dtype=float)
    mb = 0
    for s in b:
        j = int(np.searchsorted(a_arr, s))
        for jj in (j - 1, j, j + 1):
            if 0 <= jj < len(a) and abs(a[jj] - s) <= tol:
                mb += 1
                break
    return ma, mb, pairs


def _expected_match_prob(n_other, n_slots, tol=1):
    """P(a random slot has >= 1 of n_other uniformly-placed slots within
    +-tol), i.e. within a window of (2*tol+1) positions."""
    if n_slots <= 0:
        return 1.0
    w = min(2 * tol + 1, n_slots)
    p_miss = 1.0
    # without-replacement product: C(n_slots - w, n_other) / C(n_slots, n_other)
    for k in range(int(n_other)):
        num = n_slots - w - k
        den = n_slots - k
        if num <= 0 or den <= 0:
            return 1.0
        p_miss *= num / den
    return 1.0 - p_miss


def _pairsim(rep_a, rep_b, n_slots, tol=1):
    """Chance-corrected tolerant Dice x colour agreement for two phrases.
    rep = (slot_set, {slot: main_color}). Returns None if either empty."""
    sa, ca = rep_a
    sb, cb = rep_b
    if not sa or not sb:
        return None
    ma, mb, pairs = _match_slots(sa, sb, tol=tol)
    sim = (ma + mb) / (len(sa) + len(sb))
    pa = _expected_match_prob(len(sb), n_slots, tol)
    pb = _expected_match_prob(len(sa), n_slots, tol)
    e_sim = (len(sa) * pa + len(sb) * pb) / (len(sa) + len(sb))
    if e_sim >= 1.0 - 1e-9:
        adjsim = 0.0
    else:
        adjsim = max(0.0, (sim - e_sim) / (1.0 - e_sim))
    if pairs:
        agree = sum(1 for s_a, s_b in pairs if ca.get(s_a) == cb.get(s_b))
        colagree = agree / len(pairs)
        # chance-corrected colour agreement (kappa) from matched-pair marginals
        pa_d = sum(1 for s_a, _ in pairs if ca.get(s_a) == "D") / len(pairs)
        pb_d = sum(1 for _, s_b in pairs if cb.get(s_b) == "D") / len(pairs)
        e_agree = pa_d * pb_d + (1.0 - pa_d) * (1.0 - pb_d)
        if e_agree >= 1.0 - 1e-9:
            kappa = 0.0  # no colour information (monochrome pair)
        else:
            kappa = max(0.0, (colagree - e_agree) / (1.0 - e_agree))
    else:
        kappa = 0.0
    return adjsim * kappa


def call_response_reciprocity(events, ctx):
    """Call-and-response phrase structure; primary leaf = ``reciprocity`` v2.

    Construct: charters love call-and-response -- a phrase thrown out, then a
    reply with a similar rhythm but a contrasting colour. Phrases are 2-bar
    windows quantised to 16th slots.

    ``reciprocity`` v2 (fix 2026-07-11, promoted primary): mean over phrase
    triples (i, i+1, i+2) of ``pairsim(i, i+2) * (1 - pairsim(i, i+1))`` --
    A's material returns at lag 2 while the interleaved B differs. ``pairsim =
    adjsim * kappa_color`` with +-1-slot tolerant matching (one 16th; wider
    than the 30 ms max C1 jitter at any charted tempo), chance-corrected Dice
    for rhythm (density normalisation: dense phrases are not trivially
    similar) and Cohen-kappa-style chance-corrected colour agreement (random
    recolouring, mode-cycle collapse and full blandification all drive kappa
    to 0, so none of them earns conversation credit). The old exact-hash ABAB
    leaf was floor-bound (official median 0.067, 16% exact zeros; a textbook
    ABAB construction scored 0.0) and is kept as ``reciprocity_hash``
    (diagnostic only).

    Gauntlet record (fixes_20260710.md ticket 2; records ``experiments/
    metric_candidates_v1/runs/raw/records/assess_20260710_crfix_{probes,
    systems}.jsonl`` in the SoftChart research repo): 8/8 probes pass -- C3
    type-shuffle leak closed (+0.3728 -> rho -0.1890, drop +0.0340), C4
    loop-collapse rho -0.5438, C8 bar-shuffle rho -0.4360, C1/C2 still, C6
    density direction corrected (drop -0.012 -> +0.0418); official mean 0.0991
    (no floor); length coupling +0.3234 (< 0.5, disclosed); AUC
    official-vs-generated 0.6927 -> 0.7107; ladder rho +0.7381.
    """
    out = {
        "call_response_rate": float("nan"),
        "reciprocity": float("nan"),
        "reciprocity_hash": float("nan"),
        "n_phrases": float("nan"),
        "mean_rhythm_similarity": float("nan"),
        "mean_color_contrast": float("nan"),
    }
    hits = sorted_hits(events)
    if len(hits) < 4:
        return out
    g = resolve_grid(ctx)
    if g is None:
        return out
    downbeats, beat_period, meter = g
    dbs = list(downbeats)
    bounds = dbs[::PHRASE_BARS]
    if len(bounds) < 2:
        return out
    phrases = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    if len(phrases) < MIN_PHRASES:
        out["n_phrases"] = float(len(phrases))
        return out
    out["n_phrases"] = float(len(phrases))

    reps = [_phrase_slots(hits, s0, s1, SUBDIV, meter) for (s0, s1) in phrases]

    cr_events = 0
    n_pairs = 0
    rsims = []
    ccontrasts = []
    for i in range(len(reps) - 1):
        s_a, c_a = reps[i]
        s_b, c_b = reps[i + 1]
        if not s_a and not s_b:
            continue
        n_pairs += 1
        rsim = jaccard(s_a, s_b)
        rsims.append(rsim)
        aligned = s_a & s_b
        if aligned:
            flips = sum(1 for slot in aligned if c_a.get(slot) != c_b.get(slot))
            ccon = flips / len(aligned)
            ccontrasts.append(ccon)
        else:
            ccon = None
        if rsim > TAU_R and ccon is not None and COLOR_LO <= ccon <= COLOR_HI:
            cr_events += 1

    if n_pairs > 0:
        out["call_response_rate"] = cr_events / n_pairs
    if rsims:
        out["mean_rhythm_similarity"] = float(np.mean(rsims))
    if ccontrasts:
        out["mean_color_contrast"] = float(np.mean(ccontrasts))

    # reciprocity v2 (primary): continuous ABAB alternation. A phrase's
    # material returns two phrases later (tolerant, chance-corrected, colour-
    # aware similarity) while the interleaved phrase differs.
    n_slots = int(round(SUBDIV * meter * PHRASE_BARS))
    recips = []
    for i in range(len(reps) - 2):
        s2 = _pairsim(reps[i], reps[i + 2], n_slots)
        s1 = _pairsim(reps[i], reps[i + 1], n_slots)
        if s2 is None or s1 is None:
            continue
        recips.append(s2 * (1.0 - s1))
    if recips:
        out["reciprocity"] = float(np.mean(recips))

    # legacy exact-hash alternation (diagnostic only)
    labels = []
    for (ss, mc) in reps:
        rhash = hash(frozenset(ss))
        d = sum(1 for v in mc.values() if v == "D")
        k = sum(1 for v in mc.values() if v == "K")
        maincol = "D" if d >= k else "K"
        labels.append((rhash, maincol))
    if len(labels) >= 3:
        alt = 0
        denom = 0
        for i in range(len(labels) - 2):
            denom += 1
            if labels[i] == labels[i + 2] and labels[i] != labels[i + 1]:
                alt += 1
        out["reciprocity_hash"] = alt / denom if denom else float("nan")
    return out


# --- adopted v2 candidate: boredom_v2 ---
BOREDOM_SUBDIV = 4  # 16ths per beat
STAG_SIM = 0.8  # near-identical bar chain threshold
STAG_RUN = 4  # 4th+ consecutive statement = habituated
MECH_WIN = 16  # colour-stream window (notes): 4 statements of p<=4
MECH_MAX_P = 4  # short-cycle bound (C4 produces a 4-cycle; p=1 = spam)
REC_SIM = 0.75  # recognizable return keeps >= 3/4 of the bar
REC_WINDOW = 32  # bars; ~45-60 s section-recall span
MIN_BARS = 8


def _bar_reps(hits, downbeats, bar_len, meter):
    """Bar index -> (slot_set, {slot: main_color}). Slots at 16th resolution."""
    n_slots = int(round(BOREDOM_SUBDIV * meter))
    dbs = list(downbeats)
    if not dbs or n_slots <= 0:
        return {}
    t0 = dbs[0]
    reps = {}
    for t, cls in hits:
        b = int(np.floor((t - t0) / bar_len))
        if b < 0:
            continue
        frac = (t - (t0 + b * bar_len)) / bar_len
        slot = int(round(frac * n_slots))
        if slot >= n_slots:  # rolls into the next bar's downbeat
            b += 1
            slot = 0
        ss, cols = reps.setdefault(b, (set(), {}))
        ss.add(slot)
        cols.setdefault(slot, []).append(fold_hit(cls))
    out = {}
    for b, (ss, cols) in reps.items():
        main = {}
        for slot, cs in cols.items():
            d = cs.count("D")
            k = cs.count("K")
            main[slot] = "D" if d >= k else "K"
        out[b] = (frozenset(ss), main)
    return out


def _adjsim(sa, sb, n_slots, tol=1):
    """Chance-corrected tolerant Dice between two slot sets."""
    if not sa or not sb:
        return 0.0
    sb_exp = set()
    for s in sb:
        sb_exp.update((s - 1, s, s + 1))
    sa_exp = set()
    for s in sa:
        sa_exp.update((s - 1, s, s + 1))
    ma = len(sa & sb_exp)
    mb = len(sb & sa_exp)
    sim = (ma + mb) / (len(sa) + len(sb))
    pa = _expected_match_prob(len(sb), n_slots, tol)
    pb = _expected_match_prob(len(sa), n_slots, tol)
    e_sim = (len(sa) * pa + len(sb) * pb) / (len(sa) + len(sb))
    if e_sim >= 1.0 - 1e-9:
        return 0.0
    return max(0.0, (sim - e_sim) / (1.0 - e_sim))


def _colagree(rep_a, rep_b):
    """Raw colour agreement over exact-common slots (1.0 if none common)."""
    sa, ca = rep_a
    sb, cb = rep_b
    common = sa & sb
    if not common:
        return 1.0
    agree = sum(1 for s in common if ca.get(s) == cb.get(s))
    return agree / len(common)


def boredom_v2(events, ctx):
    """Two-sided boredom: stagnation (habituation) + alienation (unlearnability).

    Construct (Berlyne's inverted-U: hedonic value peaks at intermediate
    novelty; both extremes fail). Bar-level rhythm/colour fingerprints from the
    metrical grid -- deliberately decoupled from the LM event-token stream the
    pattern family shares. A bar is boring when it is either:

    * STAGNANT -- (a) the 4th-or-later consecutive near-identical statement
      (full similarity >= 0.8 chained; habituation sets in within a few
      repetitions, Rankin et al. 2009), or (b) colour-stream mechanical: every
      hit of the bar closes a 16-note colour window that is exactly periodic
      with period <= 4 (what C4 loop-collapse actually produces);
    * ALIEN -- its material (rhythm AND colours) never returns with full
      similarity >= 0.75 within +-32 bars (~45-60 s, listeners' recognition
      span for verbatim section returns, Margulis 2014).

    ``boredom_v2_raw = |stagnant U alien| / n_active_bars`` (higher = worse).
    Similarity machinery matches reciprocity v2 (+-1-slot tolerance, density
    chance correction); colour agreement is deliberately NOT kappa-corrected --
    identical monochrome bars are the canonical boring case and must count as
    identical.

    Why the incumbent needed replacing: ``boredom_rate`` fires only on token
    repetition, so zero repetition = zero boredom -- TaikoNation's
    structureless charts scored boredom_score = 1.000 ("not boring") with
    repeat_4gram 0.004 vs the official 0.45-0.72 band.

    Gauntlet record (fixes_20260710.md ticket 4; records ``experiments/
    metric_candidates_v1/runs/raw/records/assess_20260710_bv2fix_{probes,
    systems}.jsonl`` in the SoftChart research repo): 8/8 probes pass -- C4
    loop-collapse (target) rho -0.9324, drop +0.6751, z -5.93, 100% per-chart;
    C5 flatten rho -0.9314 (spam reads as boring); C1 jitter rho -0.0004,
    C2 shift rho +0.0153 (still). TaikoNation counterexample: mean 0.458 vs
    official-oni 0.260, AUC(official vs TN) = 0.871, all alienation-driven;
    length orthogonality 0.0448; AUC official-vs-generated 0.8597 (strongest
    separation of the adopted candidates).
    """
    out = {
        "boredom_v2_raw": float("nan"),
        "stagnation_frac": float("nan"),
        "alienation_frac": float("nan"),
        "n_bars_active": float("nan"),
    }
    hits = sorted_hits(events)
    if len(hits) < 8:
        return out

    g = resolve_grid(ctx)
    if g is not None:
        downbeats, beat_period, meter = g
        bar_len = beat_period * meter
        dbs = list(downbeats)
    else:
        bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
        if not bpm or not np.isfinite(float(bpm)) or float(bpm) <= 0:
            return out
        meter = 4
        bar_len = 4 * 60.0 / float(bpm)
        dbs = [hits[0][0]]
    n_slots = int(round(BOREDOM_SUBDIV * meter))

    reps = _bar_reps(hits, dbs, bar_len, meter)
    if len(reps) < MIN_BARS:
        return out
    bars = sorted(reps)

    # stagnation route (a): chains of consecutive near-identical bars
    stagnant = set()
    run = 1
    for i in range(1, len(bars)):
        b_prev, b_cur = bars[i - 1], bars[i]
        if b_cur == b_prev + 1:
            s_full = _adjsim(reps[b_prev][0], reps[b_cur][0], n_slots) * _colagree(
                reps[b_prev], reps[b_cur]
            )
            run = run + 1 if s_full >= STAG_SIM else 1
        else:
            run = 1
        if run >= STAG_RUN:
            stagnant.add(b_cur)

    # stagnation route (b): colour-stream mechanicalness. A hit is mechanical
    # if the MECH_WIN-note colour window it closes is exactly periodic with
    # some period <= MECH_MAX_P; a bar is stagnant when all of its
    # window-eligible hits are mechanical.
    folds = [fold_hit(c) for _, c in hits]
    mech = [False] * len(folds)
    for i in range(MECH_WIN - 1, len(folds)):
        w = folds[i - MECH_WIN + 1 : i + 1]
        for p in range(1, MECH_MAX_P + 1):
            if all(w[j] == w[j - p] for j in range(p, MECH_WIN)):
                mech[i] = True
                break
    t0 = dbs[0]
    hits_by_bar = {}
    for k, (t, _c) in enumerate(hits):
        b = int(np.floor((t - t0) / bar_len))
        if b < 0:
            continue
        hits_by_bar.setdefault(b, []).append(k)
    for b in bars:
        idx = [k for k in hits_by_bar.get(b, []) if k >= MECH_WIN - 1]
        if idx and all(mech[k] for k in idx):
            stagnant.add(b)

    # alienation: material never recurs (full sim >= REC_SIM) within +-REC_WINDOW
    alien = set()
    for i, b in enumerate(bars):
        found = False
        # scan outward, nearest first (early exit; neighbors usually match)
        for off in range(1, len(bars)):
            done_lo = i - off < 0
            done_hi = i + off >= len(bars)
            if done_lo and done_hi:
                break
            for j in (i - off, i + off):
                if j < 0 or j >= len(bars):
                    continue
                if abs(bars[j] - b) > REC_WINDOW:
                    continue
                s_full = _adjsim(reps[b][0], reps[bars[j]][0], n_slots) * _colagree(
                    reps[b], reps[bars[j]]
                )
                if s_full >= REC_SIM:
                    found = True
                    break
            if found:
                break
            if (done_lo or abs(bars[i - off] - b) > REC_WINDOW) and (
                done_hi or abs(bars[i + off] - b) > REC_WINDOW
            ):
                break
        if not found:
            alien.add(b)

    n_active = len(bars)
    out["stagnation_frac"] = len(stagnant) / n_active
    out["alienation_frac"] = len(alien) / n_active
    out["boredom_v2_raw"] = len(stagnant | alien) / n_active
    out["n_bars_active"] = float(n_active)
    return out


RAW_KEYS = (
    "unique_4gram_rate",
    "repeat_4gram_rate",
    "color_switch_rate",
    "boredom_rate",
    "local_nps_cv",
)
SCORE_KEYS = (
    "surface_variety_adequacy_score",
    "repetition_adequacy_score",
    "color_switch_adequacy_score",
    "density_variation_adequacy_score",
    "boredom_score",
    "surface_structure_proxy_score",
)


def compute(events, ctx):
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    course = ctx.get("course") if isinstance(ctx, dict) else None
    calibration = ctx.get("calibration") if isinstance(ctx, dict) else None
    lm = ctx.get("lm") if isinstance(ctx, dict) else None

    features = compute_chart_features(events, bpm)
    out = {k: features.get(k) for k in RAW_KEYS}

    if calibration is not None:
        if lm is not None:
            from .common import event_tokens

            stats = lm.sequence_stats(course, event_tokens(events, bpm))
            features["invalid_transition_rate"] = stats.get("invalid_transition_rate")
            features["rare_ngram_rate"] = stats.get("rare_ngram_rate")
            features["pattern_nll"] = stats.get("pattern_nll")
            from .common import quantile

            features["pattern_window_nll_p95"] = quantile(stats.get("window_nlls", []), 95)
        scores = score_chart_features(features, calibration, course)
        for k in SCORE_KEYS:
            out[k] = scores.get(k)
    else:
        for k in SCORE_KEYS:
            out[k] = None

    # adopted v2 candidates (raw; band-scored downstream if desired)
    out.update(call_response_reciprocity(events, ctx))
    out.update(boredom_v2(events, ctx))
    return out
