"""Audit gauntlet: dose-response, two-sided AUC, length orthogonality, coupling.

Given a table of scored charts under the probe ladder (one row per
``(sid, course, probe, dose_index, <metric columns>)``), this computes the
paper's diagnostic tables:

* :func:`dose_response` -- Spearman(dose, score) + monotonicity + per-dose means.
* :func:`coupling_matrix` -- probe x metric standardized-delta matrix at max dose
  (the "specificity is really coupling" honesty figure).
* :func:`length_orthogonality` -- Spearman(n_notes, metric) over official charts,
  flagging metrics a generator could game with length alone.
* :func:`separation_auc` -- rank-AUC of a metric separating one row group from
  another (e.g. official vs a generated system).

Only numpy is required; if scipy is present, Spearman p-values are reported.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np


def _fnum(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


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


def spearman(x, y):
    """Spearman rho (numpy-only). Returns None if degenerate."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(y, dtype=np.float64)
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    rx = _rankdata(xs) - _rankdata(xs).mean()
    ry = _rankdata(ys) - _rankdata(ys).mean()
    denom = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    if denom <= 1e-12:
        return None
    return float((rx * ry).sum() / denom)


def _index_by_probe(rows, dose0_probe="official"):
    """probe -> dose_index -> [rows]; dose-0 pool shared across probes."""
    by_probe = defaultdict(lambda: defaultdict(list))
    official = [r for r in rows if r.get("probe") == dose0_probe]
    probes = sorted({r.get("probe") for r in rows if r.get("probe") != dose0_probe})
    for r in official:
        for probe in probes:
            by_probe[probe][0].append(r)
    for r in rows:
        if r.get("probe") == dose0_probe:
            continue
        by_probe[r["probe"]][int(r["dose_index"])].append(r)
    return by_probe, probes, official


def _per_dose_means(rows_by_dose, metric, max_dose=3):
    means = {}
    for di in range(max_dose + 1):
        vals = [_fnum(r.get(metric)) for r in rows_by_dose.get(di, [])]
        vals = [v for v in vals if v is not None]
        means[di] = float(np.mean(vals)) if vals else None
    return means


def _monotone_nonincreasing(means, max_dose=3):
    seq = [means[di] for di in range(max_dose + 1) if means.get(di) is not None]
    if len(seq) < 2:
        return None
    return all(seq[i] >= seq[i + 1] - 1e-9 for i in range(len(seq) - 1))


def dose_response(rows, metrics, max_dose=3, dose0_probe="official"):
    """Per (probe, metric) dose-response: Spearman, monotonicity, per-dose means.

    ``rows`` : list of dicts with ``probe``, ``dose_index`` and metric columns.
    Returns a list of result dicts (one per probe x metric).
    """
    by_probe, probes, _official = _index_by_probe(rows, dose0_probe)
    out = []
    for probe in probes:
        rbd = by_probe[probe]
        for metric in metrics:
            means = _per_dose_means(rbd, metric, max_dose)
            xs, ys = [], []
            for di in range(max_dose + 1):
                for r in rbd.get(di, []):
                    v = _fnum(r.get(metric))
                    if v is not None:
                        xs.append(di)
                        ys.append(v)
            rho = spearman(xs, ys) if len(set(xs)) >= 2 else None
            out.append(
                {
                    "probe": probe,
                    "metric": metric,
                    "spearman_dose_score": rho,
                    "monotone_nonincreasing": _monotone_nonincreasing(means, max_dose),
                    **{f"mean_dose{di}": means[di] for di in range(max_dose + 1)},
                }
            )
    return out


def _official_sd(official_rows, metric):
    vals = [_fnum(r.get(metric)) for r in official_rows]
    vals = [v for v in vals if v is not None]
    if len(vals) < 2:
        return None
    sd = float(np.std(vals, ddof=1))
    return sd if sd > 1e-9 else None


def _standardized_delta(rows_by_dose, official_rows, metric, max_dose=3):
    sd = _official_sd(official_rows, metric)
    d0 = {(r.get("sid"), r.get("course")): _fnum(r.get(metric)) for r in rows_by_dose.get(0, [])}
    diffs = []
    for r in rows_by_dose.get(max_dose, []):
        key = (r.get("sid"), r.get("course"))
        a = _fnum(r.get(metric))
        b = d0.get(key)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return None, None
    raw = float(np.mean(diffs))
    return raw, (None if sd is None else float(raw / sd))


def coupling_matrix(rows, metrics, max_dose=3, dose0_probe="official"):
    """Full probe x metric standardized-delta matrix at max dose (every cell)."""
    by_probe, probes, official = _index_by_probe(rows, dose0_probe)
    out = []
    for probe in probes:
        rbd = by_probe[probe]
        for metric in metrics:
            raw, sdn = _standardized_delta(rbd, official, metric, max_dose)
            out.append(
                {
                    "probe": probe,
                    "metric": metric,
                    "raw_delta_maxdose": raw,
                    "std_delta_maxdose": sdn,
                    "abs_std_delta": None if sdn is None else abs(sdn),
                }
            )
    return out


def length_orthogonality(rows, metrics, length_key="n_notes", dose0_probe="official"):
    """Spearman(length, metric) over dose-0 official charts per metric.

    A |rho| near 1 flags a metric a generator could move by changing chart length
    alone -- a length confound the paper reports for e.g. distinct-n.
    """
    official = [r for r in rows if r.get("probe", dose0_probe) == dose0_probe]
    out = []
    for metric in metrics:
        xs, ys = [], []
        for r in official:
            L = _fnum(r.get(length_key))
            v = _fnum(r.get(metric))
            if L is not None and v is not None:
                xs.append(L)
                ys.append(v)
        out.append({"metric": metric, "spearman_len_metric": spearman(xs, ys), "n": len(xs)})
    return out


def separation_auc(pos_values, neg_values):
    """Rank-AUC that ``pos`` > ``neg`` (Mann-Whitney U / (n_pos*n_neg)).

    Returns None if either group is empty. AUC 0.5 = no separation, 1.0 = perfect.
    """
    pos = [v for v in (_fnum(x) for x in pos_values) if v is not None]
    neg = [v for v in (_fnum(x) for x in neg_values) if v is not None]
    if not pos or not neg:
        return None
    allv = pos + neg
    ranks = _rankdata(allv)
    r_pos = float(np.sum(ranks[: len(pos)]))
    n1, n2 = len(pos), len(neg)
    u = r_pos - n1 * (n1 + 1) / 2.0
    return float(u / (n1 * n2))


def baseline_reversal(rows, foils, max_dose=3, dose0_probe="official"):
    """Per-dose means for named baseline foils under their probes.

    ``foils`` : list of ``(probe, metric)`` pairs. Returns per-foil dose curves;
    the paper's Figure 1 reads these to show baselines improving under corruption.
    """
    by_probe, _probes, _official = _index_by_probe(rows, dose0_probe)
    out = []
    for probe, metric in foils:
        means = _per_dose_means(by_probe[probe], metric, max_dose)
        out.append({"probe": probe, "metric": metric, **{f"dose{di}": means[di] for di in range(max_dose + 1)}})
    return out
