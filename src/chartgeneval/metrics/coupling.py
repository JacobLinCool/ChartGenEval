"""Coupling family: density / strain constraints and audio coupling.

Two sub-groups:

* *density constraints* -- density adequacy, strain (local p95), overload and
  density-spike lower-better scores (band + threshold; needs ``ctx['calibration']``).
* *audio coupling* -- the adopted v2 candidates ``density_energy_response`` (note
  density tracks musical energy) and ``energy_peak_support_rate`` (notes land on
  onset-envelope peaks). Both need ``ctx['mel']``; density_energy also needs a
  grid.

Ported from the reference density-score pipeline and the two candidate modules.
"""

from __future__ import annotations

import numpy as np

from ..calibration import score_chart_features
from ..events import HIT_CLASSES
from .common import compute_chart_features

METRIC_NAME = "coupling"

FPS = 86.1328125
MIN_WINDOWS = 8
MIN_ENERGY_DYNRANGE = 0.05
HALF_WIN = 3  # +-3 frames ~= +-35 ms


# --------------------------------------------------------------------------
# density_energy_response
# --------------------------------------------------------------------------
def _rankdata(a):
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sa = a[order]
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    rx = _rankdata(x) - _rankdata(x).mean()
    ry = _rankdata(y) - _rankdata(y).mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom <= 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _detrend(y):
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 3:
        return y
    x = np.arange(n, dtype=np.float64)
    x = x - x.mean()
    yy = y - y.mean()
    sxx = (x * x).sum()
    if sxx <= 1e-12:
        return yy
    slope = (x * yy).sum() / sxx
    return yy - slope * x


def density_energy_response(events, ctx):
    """Spearman coupling of per-bar note density and audio energy (candidate)."""
    nan = {
        "density_energy_spearman": float("nan"),
        "density_energy_partial": float("nan"),
        "density_energy_lag_robust": float("nan"),
        "density_energy_n_windows": float("nan"),
    }
    mel = ctx.get("mel") if isinstance(ctx, dict) else None
    grid = ctx.get("grid") if isinstance(ctx, dict) else None
    if mel is None or grid is None:
        return nan
    mel = np.asarray(mel, dtype=np.float64)
    if mel.ndim != 2 or mel.shape[1] < 3:
        return nan
    downbeats = grid.get("downbeats")
    if downbeats is None or len(downbeats) < 2:
        return nan
    db = np.unique(np.asarray([float(x) for x in downbeats], dtype=np.float64))
    db.sort()
    if db.size < MIN_WINDOWS + 1:
        return nan

    hits = sorted((float(t), str(c)) for t, c in events if c in HIT_CLASSES)
    times = np.array([t for t, _ in hits], dtype=np.float64)
    T = mel.shape[1]
    lin = np.exp(mel)

    note_density, audio_energy = [], []
    for b in range(db.size - 1):
        a, e = db[b], db[b + 1]
        if e <= a:
            continue
        dur = e - a
        n_hits = float(np.sum((times >= a) & (times < e))) if times.size else 0.0
        note_density.append(n_hits / dur)
        f0 = max(0, min(T, int(round(a * FPS))))
        f1 = max(0, min(T, int(round(e * FPS))))
        audio_energy.append(0.0 if f1 <= f0 else float(lin[:, f0:f1].mean()))

    note_density = np.asarray(note_density, dtype=np.float64)
    audio_energy = np.asarray(audio_energy, dtype=np.float64)
    n = len(note_density)
    if n < MIN_WINDOWS:
        return nan
    e_med = float(np.median(audio_energy)) if np.median(audio_energy) > 0 else 1e-12
    dyn = (np.percentile(audio_energy, 90) - np.percentile(audio_energy, 10)) / e_med
    if not np.isfinite(dyn) or dyn < MIN_ENERGY_DYNRANGE:
        return nan

    def _z(v):
        s = v.std()
        return (v - v.mean()) / s if s > 1e-12 else v - v.mean()

    d_z, e_z = _z(note_density), _z(audio_energy)
    primary = _spearman(d_z, e_z)
    partial = _spearman(_detrend(note_density), _detrend(audio_energy))
    lag_vals = [primary]
    if n >= MIN_WINDOWS + 1:
        lag_vals.append(_spearman(d_z[1:], e_z[:-1]))
        lag_vals.append(_spearman(d_z[:-1], e_z[1:]))
    lag_vals = [v for v in lag_vals if np.isfinite(v)]
    lag_robust = float(max(lag_vals, key=abs)) if lag_vals else float("nan")
    return {
        "density_energy_spearman": float(primary),
        "density_energy_partial": float(partial) if np.isfinite(partial) else float("nan"),
        "density_energy_lag_robust": lag_robust,
        "density_energy_n_windows": float(n),
    }


# --------------------------------------------------------------------------
# energy_peak_support_rate
# --------------------------------------------------------------------------
def _onset_envelope(mel):
    mel = np.asarray(mel, dtype=np.float64)
    if mel.ndim != 2 or mel.shape[1] < 3:
        return None
    diff = mel[:, 1:] - mel[:, :-1]
    flux = np.maximum(0.0, diff).sum(axis=0)
    o = np.zeros(mel.shape[1], dtype=np.float64)
    o[1:] = flux
    std = o.std()
    if std <= 1e-9:
        return None
    return (o - o.mean()) / std


def energy_peak_support_rate(events, ctx):
    """Rate of hits landing on onset-envelope peaks (candidate; C2 witness)."""
    nan = {
        "energy_support_rate_raw": float("nan"),
        "energy_support_rate_global": float("nan"),
        "energy_peak_mean_z": float("nan"),
        "energy_at_hit_z": float("nan"),
        "energy_theta_song": float("nan"),
        "energy_n_hits": float("nan"),
    }
    mel = ctx.get("mel") if isinstance(ctx, dict) else None
    if mel is None:
        return nan
    o = _onset_envelope(mel)
    if o is None:
        return nan
    T = o.shape[0]
    hits = sorted((float(t), str(c)) for t, c in events if c in HIT_CLASSES)
    if not hits:
        return nan
    peaks, at_hit = [], []
    for t, _ in hits:
        i = int(round(t * FPS))
        if i < 0 or i >= T:
            continue
        lo = max(0, i - HALF_WIN)
        hi = min(T, i + HALF_WIN + 1)
        peaks.append(float(np.max(o[lo:hi])))
        at_hit.append(float(o[i]))
    if not peaks:
        return nan
    peaks = np.asarray(peaks, dtype=np.float64)
    at_hit = np.asarray(at_hit, dtype=np.float64)
    theta_song = float(np.percentile(o, 60))
    theta_global = 0.0
    if isinstance(ctx, dict) and ctx.get("energy_theta_global") is not None:
        try:
            theta_global = float(ctx["energy_theta_global"])
        except (TypeError, ValueError):
            theta_global = 0.0
    return {
        "energy_support_rate_raw": float(np.mean(peaks >= theta_song)),
        "energy_support_rate_global": float(np.mean(peaks >= theta_global)),
        "energy_peak_mean_z": float(np.mean(peaks)),
        "energy_at_hit_z": float(np.mean(at_hit)),
        "energy_theta_song": theta_song,
        "energy_n_hits": float(len(peaks)),
    }


RAW_KEYS = (
    "density_nps",
    "local_nps_p95",
    "local_nps_delta_p95",
    "overload_excess_nps",
    "density_spike_excess_nps",
)
SCORE_KEYS = (
    "density_adequacy_score",
    "strain_adequacy_score",
    "overload_score",
    "density_spike_score",
    "playability_proxy_score",
)


def compute(events, ctx):
    bpm = ctx.get("bpm") if isinstance(ctx, dict) else None
    course = ctx.get("course") if isinstance(ctx, dict) else None
    calibration = ctx.get("calibration") if isinstance(ctx, dict) else None

    features = compute_chart_features(events, bpm)
    out = {k: features.get(k) for k in ("density_nps", "local_nps_p95", "local_nps_delta_p95")}
    if calibration is not None:
        scores = score_chart_features(features, calibration, course)
        out["overload_excess_nps"] = features.get("overload_excess_nps")
        out["density_spike_excess_nps"] = features.get("density_spike_excess_nps")
        for k in SCORE_KEYS:
            out[k] = scores.get(k)
    else:
        for k in ("overload_excess_nps", "density_spike_excess_nps", *SCORE_KEYS):
            out[k] = None

    out.update(density_energy_response(events, ctx))
    out.update(energy_peak_support_rate(events, ctx))
    return out
