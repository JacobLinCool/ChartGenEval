"""Descriptive check of baselines computed by the scout but dropped from the spec (distinct-n, TTR,
IOI entropy/CV, empty-beat rate, groove): song-mean strongest-dose change vs official and the sign
of the per-chart Spearman (oriented so that 'higher distinct/TTR = better'). Descriptive only."""
import json, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
base = Path(__file__).resolve().parent.parent
recs = [json.loads(l) for p in sorted(glob.glob(str(base / "full/shard_*.jsonl"))) for l in open(p)]
M = ["distinct_2", "distinct_4", "type_token_ratio", "ioi_entropy_bits", "ioi_cv", "empty_beat_rate", "groove_consistency"]
by = defaultdict(lambda: defaultdict(list))
for r in recs:
    by[(int(r["sid"].split("_")[1]), r["course"])][(r["condition"], r["dose_index"])].append(r["metrics"])
out = {}
for probe in ("C2_anchor_shift", "C4_loop_collapse", "C5_blandification", "C3_type_shuffle"):
    sev = [0, .25, .5, 1] if probe in ("C2_anchor_shift", "C3_type_shuffle") else [0, .3, .6, 1]
    for m in M:
        song_d, song_rho = defaultdict(list), defaultdict(list)
        for (g, c), d in by.items():
            vals = [d[("official", 0)][0][m]] + [np.mean([x[m] for x in d[(probe, k)]]) if all(x[m] is not None for x in d[(probe, k)]) else None for k in (1, 2, 3)]
            if any(v is None for v in vals):
                continue
            song_d[g].append(vals[3] - vals[0])
            song_rho[g].append(0.0 if len(set(vals)) == 1 else float(spearmanr(sev, vals).statistic))
        dm = np.array([np.mean(v) for v in song_d.values()]); rm = np.array([np.mean(v) for v in song_rho.values()])
        out[f"{m}|{probe}"] = {"songs": len(dm), "raw_delta_song_mean": round(float(dm.mean()), 5),
                               "raw_rho_song_mean": round(float(rm.mean()), 3), "exactly_zero": bool(np.all(dm == 0) and np.all(rm == 0))}
for k, v in out.items():
    print(f"{k:40s} {v}")
(Path(__file__).parent / "dropped_baselines_check.json").write_text(json.dumps(out, indent=1))
