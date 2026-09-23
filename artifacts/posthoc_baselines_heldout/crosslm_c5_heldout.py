"""Robustness check: held-out C5 (common-pattern rewrite) perplexity with SEPARATE scoring LMs.

Rewriter = frozen LM (full train, order 3, alpha 0.05; fingerprint checked against the run manifest),
so the C5 variants are the sealed ones (hash-checked). Scorers (post hoc, defined here):
  A: trigram, alpha 0.05, fit only on training songs with EVEN group_id (half of train)
  B: trigram, alpha 0.05, fit only on training songs with ODD group_id (the other half)
  C: 4-gram, alpha 0.05, full train
C5 is deterministic (argmax), so its 5 replicates are identical; sham = C2 strongest = official tokens.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/confirmatory_holdout_v1"
sys.path.insert(0, str(EXP)); sys.path.insert(0, str(REPO / "src"))
import run as FR  # noqa: E402
from chartgeneval.events import sorted_hits  # noqa: E402
from chartgeneval.metrics.common import NGramModel, event_tokens  # noqa: E402
from chartgeneval.probes import apply_probe  # noqa: E402

D = Path.home() / ".cache/huggingface/hub/datasets--JacobLinCool--taiko-1000-parsed-clean/snapshots/b72da4616d643018e81f372cea06ce51349285e0/data"
RAW = EXP / "runs/raw/confirmatory-primary-20260712/records.jsonl"
HERE = Path(__file__).resolve().parent
COLS = ["group_id", "canonical", "easy", "normal", "hard", "oni", "ura"]

lms = {"frozen": NGramModel(order=3, alpha=0.05), "A_even_tri": NGramModel(order=3, alpha=0.05),
       "B_odd_tri": NGramModel(order=3, alpha=0.05), "C_full_4gram": NGramModel(order=4, alpha=0.05)}
n_rows = 0
for f in sorted(D.glob("train-*.parquet")):
    for row in pq.read_table(f, columns=COLS).to_pylist():
        n_rows += 1
        for course, chart, _ in FR._official_charts(row):
            tok = event_tokens(chart.events, chart.bpm)
            lms["frozen"].add(course, tok); lms["C_full_4gram"].add(course, tok)
            lms["A_even_tri" if int(row["group_id"]) % 2 == 0 else "B_odd_tri"].add(course, tok)
fp = FR._lm_state_fingerprint(lms["frozen"])
manifest_ok = fp in (EXP / "runs/raw/confirmatory-primary-20260712/MANIFEST.json").read_text()

sealed = {}
for line in open(RAW):
    r = json.loads(line)
    if r["condition"] in ("C5_blandification", "official"):
        sealed[r["task_id"]] = (r["variant_events_sha256"], r["status"], r["corruption_noop"], r["metrics"].get("pattern_nll"))

test = []
for f in sorted(D.glob("test-*.parquet")):
    test += pq.read_table(f, columns=COLS).to_pylist()
canon = [r for r in test if r["canonical"]][40:120]

per = defaultdict(lambda: defaultdict(list))  # scorer -> gid -> [(rho, delta_raw, base)]
hash_fail = nll_fail = excluded = 0
for row in canon:
    sid = f"group_{int(row['group_id'])}"
    for course, chart, _ in FR._official_charts(row):
        ev = sorted_hits(chart.events)
        variants = [ev]
        ok = True
        for di, dv in enumerate((0.3, 0.6, 1.0), start=1):
            reps = []
            for ri in range(1, 6):
                s = sealed[f"{sid}|{course}|C5_blandification|d{di}|r{ri}"]
                v, noop = apply_probe("C5_blandification", ev, dv, np.random.default_rng(0), chart.bpm, lm=lms["frozen"], course=course)
                hash_fail += FR._events_hash(v) != s[0]
                if s[1] != "success" or s[2]:
                    ok = False
                reps.append(v)
            variants.append(reps[0])
        if not ok:
            excluded += 1
            continue
        for name, lm in lms.items():
            nll = [lm.sequence_stats(course, event_tokens(v, chart.bpm))["pattern_nll"] for v in variants]
            if name == "frozen":
                nll_fail += abs(nll[3] - sealed[f"{sid}|{course}|C5_blandification|d3|r1"][3]) > 1e-12
            ppl = [math.exp(x) for x in nll]
            o = [-p for p in ppl]  # worse = increase -> oriented = -value
            rho = 0.0 if len(set(o)) == 1 else float(spearmanr([0, .3, .6, 1], o).statistic)
            per[name][int(row["group_id"])].append((rho, ppl[3] - ppl[0], ppl[0], ppl[3]))

out = {"train_rows": n_rows, "frozen_lm_fp": fp, "frozen_fp_in_manifest": manifest_ok,
       "c5_hash_fail": hash_fail, "frozen_nll_fail": nll_fail, "excluded_charts": excluded}
rng = np.random.default_rng(777)
for name, g in per.items():
    gids = sorted(g)
    X = np.array([[np.mean([a[0] for a in g[k]]), np.mean([a[1] for a in g[k]]),
                   np.mean([a[2] for a in g[k]]), np.mean([a[3] for a in g[k]])] for k in gids])
    idx = rng.integers(0, len(X), size=(10000, len(X)))
    boot = X[idx].mean(axis=1)
    ci = lambda c: [round(float(np.quantile(boot[:, c], .0025)), 4), round(float(np.quantile(boot[:, c], .9975)), 4)]
    chart_rows = [a for k in gids for a in g[k]]
    out[name] = {"n_songs": len(gids), "n_charts": len(chart_rows),
                 "rho_oriented_mean": round(float(X[:, 0].mean()), 4), "rho_ci995": ci(0),
                 "delta_ppl_song_mean": round(float(X[:, 1].mean()), 4), "delta_ci995": ci(1),
                 "official_ppl_song_mean": round(float(X[:, 2].mean()), 4), "c5d3_ppl_song_mean": round(float(X[:, 3].mean()), 4),
                 "chart_mean_official_ppl": round(float(np.mean([a[2] for a in chart_rows])), 4),
                 "chart_mean_c5d3_ppl": round(float(np.mean([a[3] for a in chart_rows])), 4)}
print(json.dumps(out, indent=1))
(HERE / "crosslm_c5_heldout.json").write_text(json.dumps(out, indent=1))
