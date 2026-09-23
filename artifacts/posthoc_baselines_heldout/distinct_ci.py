"""Held-out frozen-gate-style stats (own implementation) for distinct-4 and TTR, read as 'higher = more diverse = better'.
Values from the regenerated baseline shards (hash-verified against sealed records); status/noop from sealed records."""
import json, glob, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
base = Path(__file__).resolve().parent.parent
RAW = Path(__file__).resolve().parents[2] / "experiments/confirmatory_holdout_v1/runs/raw/confirmatory-primary-20260712/records.jsonl"
st = {}
for l in open(RAW):
    r = json.loads(l); st[r["task_id"]] = (r["status"] == "success" and not r["corruption_noop"], r["variant_events_sha256"])
vals = defaultdict(list)
for p in sorted(glob.glob(str(base / "full/shard_*.jsonl"))):
    for l in open(p):
        r = json.loads(l)
        ok, h = st[r["task_id"]]
        assert h == r["variant_events_sha256"]
        vals[(int(r["sid"].split("_")[1]), r["course"], r["condition"], r["dose_index"])].append((ok, r["metrics"]))
SEV = {"C3_type_shuffle": [0, .25, .5, 1], "C4_loop_collapse": [0, .3, .6, 1], "C5_blandification": [0, .3, .6, 1]}
def mean(k, m):
    v = vals.get(k); need = 1 if k[2] == "official" else 5
    if not v or len(v) != need or not all(ok and x[m] is not None for ok, x in v): return None
    xs = [x[m] for _, x in v]
    return xs[0] if len(set(xs)) == 1 else float(np.mean(xs))
out = {}
rng = np.random.default_rng(4242)
for m in ("distinct_4", "type_token_ratio"):
    for probe, sev in SEV.items():
        song = defaultdict(list)
        for (g, c, cond, d) in list(vals):
            if cond != "official": continue
            xs = [mean((g, c, "official", 0), m)] + [mean((g, c, probe, k), m) for k in (1, 2, 3)] + [mean((g, c, "C2_anchor_shift", 3), m)]
            if any(x is None for x in xs): continue
            o = xs  # worse = decrease -> oriented = value
            rho = 0.0 if len(set(o[:4])) == 1 else float(spearmanr(sev, o[:4]).statistic)
            song[g].append((rho, o[3] - o[4]))
        X = np.array([[np.mean([a for a, _ in v]), np.mean([b for _, b in v])] for v in song.values()])
        B = X[rng.integers(0, len(X), size=(10000, len(X)))].mean(axis=1)
        q = lambda c: [round(float(np.quantile(B[:, c], .0025)), 4), round(float(np.quantile(B[:, c], .9975)), 4)]
        out[f"{m}|{probe}"] = {"n_songs": len(X), "rho": round(float(X[:, 0].mean()), 4), "rho_ci995": q(0),
                               "delta": round(float(X[:, 1].mean()), 4), "delta_ci995": q(1)}
for k, v in out.items(): print(k, v)
(Path(__file__).parent / "distinct_ci.json").write_text(json.dumps(out, indent=1))
