"""Gap family: distance to the official chart manifold.

The adopted v2 candidate ``official_manifold_gap``: good charts look like
official charts, bad ones are far from the official "shape". A joint feature
vector phi(chart) (IOI histogram + type-bigram + per-beat density quantiles +
distinct ratio + big-note ratio) is compared against a reference cloud of
official charts. This catches the "each dimension is band-normal but the
combination is not official" collapse that per-dimension bands cannot see.

The module always returns the phi vector (``phi_*``). When the runner injects a
fitted reference via ``ctx['official_manifold_ref'] = {mean, std, feats, scale,
k}`` for the matching course, it also returns ``manifold_gap_raw`` /
``manifold_score``. Fitting a reference from a corpus is a two-pass operation;
:func:`fit_manifold_ref` builds one from phi vectors.

Ported from the candidate module.
"""

from __future__ import annotations

import numpy as np

from ..events import HIT_CLASSES

METRIC_NAME = "gap"
_TYPE_INDEX = {"DS": 0, "KS": 1, "DB": 2, "KB": 3}
_IOI_LOG2_EDGES = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.float64)
PHI_DIM = 8 + 16 + 3 + 1 + 1  # 29


def _fold_size(cls: str) -> str:
    color = "D" if "don" in cls else ("K" if "ka" in cls else "X")
    size = "B" if "big" in cls else "S"
    return color + size


def phi(events, bpm):
    """Density/length-invariant feature vector (no absolute timing offset)."""
    hits = sorted((float(t), str(c)) for t, c in events if c in HIT_CLASSES)
    if len(hits) < 4:
        return None
    times = np.array([t for t, _ in hits], dtype=np.float64)
    classes = [_fold_size(c) for _, c in hits]
    duration = float(times[-1] - times[0]) if times[-1] > times[0] else 1.0

    ioi_hist = np.zeros(8, dtype=np.float64)
    if bpm and bpm > 0:
        beat = 60.0 / float(bpm)
        iois_beats = np.diff(times) / beat
        iois_beats = iois_beats[iois_beats > 1e-6]
        if iois_beats.size:
            logs = np.log2(iois_beats)
            idx = np.clip(np.digitize(logs, _IOI_LOG2_EDGES), 0, 7)
            for k in idx:
                ioi_hist[k] += 1.0
    ioi_hist = ioi_hist / max(duration, 1e-6)

    bigram = np.zeros((4, 4), dtype=np.float64)
    for a, b in zip(classes[:-1], classes[1:]):
        ia, ib = _TYPE_INDEX.get(a), _TYPE_INDEX.get(b)
        if ia is not None and ib is not None:
            bigram[ia, ib] += 1.0
    tot = bigram.sum()
    if tot > 0:
        bigram = bigram / tot
    bigram = bigram.reshape(-1)

    dens_q = np.zeros(3, dtype=np.float64)
    if bpm and bpm > 0 and times.size >= 4:
        beat = 60.0 / float(bpm)
        edges = np.arange(times[0], times[-1] + beat, beat)
        if edges.size >= 3:
            counts, _ = np.histogram(times, bins=edges)
            if counts.size:
                dens_q = np.percentile(counts.astype(np.float64), [10, 50, 90])

    if bpm and bpm > 0:
        beat = 60.0 / float(bpm)
        toks = []
        prev = None
        for i, (t, _) in enumerate(hits):
            if prev is None:
                io = "s"
            else:
                r = (t - prev) / beat
                io = "x" if r < 0.4 else "e" if r < 0.9 else "q" if r < 1.9 else "h"
            toks.append(io + classes[i])
            prev = t
    else:
        toks = classes
    grams = [tuple(toks[i : i + 4]) for i in range(len(toks) - 3)]
    distinct = (len(set(grams)) / len(grams)) if grams else 0.0

    big = sum(1 for c in classes if c.endswith("B"))
    big_ratio = big / len(classes) if classes else 0.0

    return np.concatenate([ioi_hist, bigram, dens_q, [distinct], [big_ratio]]).astype(np.float64)


def fit_manifold_ref(phi_vectors, k=20):
    """Fit a per-course manifold reference from a list of phi vectors.

    Returns ``{mean, std, feats (z-scored), scale, k}``. ``scale`` is the p90 of
    leave-one-out kNN gaps over the reference set (a robust normaliser).
    """
    P = np.asarray([v for v in phi_vectors if v is not None], dtype=np.float64)
    if P.shape[0] < 2:
        return None
    mean = P.mean(axis=0)
    std = P.std(axis=0)
    std_safe = np.where(std > 1e-9, std, 1.0)
    Z = (P - mean) / std_safe
    # LOO kNN gaps for scale
    kk = min(k, Z.shape[0] - 1)
    gaps = []
    for i in range(Z.shape[0]):
        d = np.linalg.norm(Z - Z[i][None, :], axis=1)
        d = np.sort(d)[1 : kk + 1]  # skip self
        if d.size:
            gaps.append(float(np.mean(d)))
    scale = float(np.percentile(gaps, 90)) if gaps else 1.0
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "feats": Z.tolist(),
        "scale": scale if scale > 0 else 1.0,
        "k": int(k),
    }


def compute(events, ctx):
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    grid = ctx.get("grid") if isinstance(ctx, dict) else None
    if (not bpm or bpm <= 0) and grid:
        bpm = grid.get("bpm")

    p = phi(events, bpm)
    if p is None:
        return {
            "manifold_gap_raw": float("nan"),
            "manifold_gap_mahal": float("nan"),
            "manifold_score": float("nan"),
            "manifold_n_ref": float("nan"),
        }

    out = {f"phi_{k:02d}": float(p[k]) for k in range(len(p))}
    ref = ctx.get("official_manifold_ref") if isinstance(ctx, dict) else None
    if not ref:
        out.update(
            {
                "manifold_gap_raw": float("nan"),
                "manifold_gap_mahal": float("nan"),
                "manifold_score": float("nan"),
                "manifold_n_ref": float("nan"),
            }
        )
        return out

    mean = np.asarray(ref["mean"], dtype=np.float64)
    std = np.asarray(ref["std"], dtype=np.float64)
    feats = np.asarray(ref["feats"], dtype=np.float64)
    scale = float(ref.get("scale", 1.0)) or 1.0
    std_safe = np.where(std > 1e-9, std, 1.0)
    z = np.clip((p - mean) / std_safe, -50.0, 50.0)

    k = int(ref.get("k", 20))
    d = np.linalg.norm(feats - z[None, :], axis=1)
    d.sort()
    kk = min(k, d.size)
    gap_knn = float(np.mean(d[:kk])) if kk > 0 else float("nan")

    gap_mahal = float("nan")
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            cov = np.cov(feats, rowvar=False)
            if cov.ndim == 0:
                cov = np.array([[float(cov)]])
            lam = 0.1
            cov_s = (1 - lam) * cov + lam * np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
            inv = np.linalg.pinv(cov_s)
            diff = z - feats.mean(axis=0)
            q = float(diff @ inv @ diff)
        if np.isfinite(q) and q >= 0.0:
            gap_mahal = float(np.sqrt(q))
    except Exception:
        gap_mahal = float("nan")

    out.update(
        {
            "manifold_gap_raw": gap_knn,
            "manifold_gap_mahal": gap_mahal,
            "manifold_score": float(np.exp(-gap_knn / scale)) if np.isfinite(gap_knn) else float("nan"),
            "manifold_n_ref": float(feats.shape[0]),
        }
    )
    return out
