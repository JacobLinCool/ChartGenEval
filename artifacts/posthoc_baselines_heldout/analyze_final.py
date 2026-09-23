"""POST-HOC SECONDARY held-out baseline analysis (not preregistered) -- FINAL.

Same harness as ../analyze_baselines.py (frozen analyze.py functions imported
read-only, frozen replicate reducer, frozen bootstrap seed rule, 99.5% song
bootstrap, >=72/80 songs), with these conventions:
  * onset F1 uses MAXIMUM matching (mir_eval.onset.f_measure; shards/f1mm_*.jsonl);
    the greedy values are kept only as a superseded cross-check;
  * cells where every seed at every strength equals the unedited value are
    reported as exactly unchanged (the 5-seed float mean can differ by 1 ulp);
  * time-only metrics (IOI-JS, onset F1) under type-only edits (C3, C4, C5) are
    N/A by design, not empirical failures.
Writes only into this directory. Caller sets PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
EXP = REPO / "experiments/confirmatory_holdout_v1"
RUN = EXP / "runs/raw/confirmatory-primary-20260712"
TABLES = RUN / "tables"
PRIMARY = REPO / "artifacts/confirmatory_holdout_v1/primary_results.json"
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(REPO / "src"))
import analyze as FA  # noqa: E402  frozen analysis, read-only


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_jsonl(p: Path) -> list[dict]:
    with open(p) as fh:
        return [json.loads(line) for line in fh if line.strip()]


contract = json.load(open(RUN / "spec_snapshots/contract.json"))
acfg = contract["analysis"]
DRAWS = int(acfg["bootstrap_draws"]); NOM = float(acfg["nominal_confidence_level"])
ADJ = float(acfg["familywise_confidence_level"]); MIN_G = int(acfg["minimum_confirmatory_group_clusters"])
MASTER = int(acfg["bootstrap_seed"]); ADJ_KEY = f"{ADJ:.6f}"


def verdict_of(summary, controls_pass):
    if not controls_pass or summary["n_group_clusters"] < MIN_G:
        return "INCONCLUSIVE"
    d_lo, d_hi = summary["confidence_intervals"][ADJ_KEY]["dose_statistic"]
    c_lo, c_hi = summary["confidence_intervals"][ADJ_KEY]["target_minus_sham"]
    if d_hi < 0.0 and c_hi < 0.0:
        return "SURVIVES"
    if d_lo >= 0.0 or c_lo >= 0.0:
        return "REFUTED"
    return "INCONCLUSIVE"


# ---- step 1: harness validation (frozen primary rows reproduced exactly)
observations = read_jsonl(TABLES / "observations.jsonl")
frozen_means = read_jsonl(TABLES / "chart_dose_means.jsonl")
index = FA._index_means(frozen_means)
_, ctrl_pass, ctrl_course = FA._control_diagnostics(observations, contract)
sealed_primary = {r["pair_id"]: r for r in json.load(open(PRIMARY))}
validation = []
for pair in contract["primary_pairs"]:
    gv, _ex, cc = FA._pair_group_values(index, pair, contract, ctrl_course)
    s = FA._cluster_summary(gv, draws=DRAWS, seed=FA._pair_seed(MASTER, pair["pair_id"]), levels=(NOM, ADJ))
    req = [(c, pair["metric"]) for c, ctl in contract["controls"].items()
           if pair["family"] in ctl["scoped_families"] and pair["metric"] in ctl["metrics"]]
    v = verdict_of(s, all(ctrl_pass.get(k, False) for k in req))
    ref = sealed_primary[pair["pair_id"]]
    ok = (s["dose_statistic_mean"] == ref["dose_statistic_mean"] and s["target_minus_sham_mean"] == ref["target_minus_sham_mean"]
          and s["confidence_intervals"] == ref["confidence_intervals"] and s["n_group_clusters"] == ref["n_group_clusters"]
          and len(cc) == ref["n_complete_courses"] and v == ref["verdict"])
    validation.append({"pair_id": pair["pair_id"], "probe": pair["probe"], "metric": pair["metric"], "exact_match": ok,
                       "verdict": v, "dose_statistic_mean": s["dose_statistic_mean"],
                       "target_minus_sham_mean": s["target_minus_sham_mean"],
                       "ci995": s["confidence_intervals"][ADJ_KEY], "n_songs": s["n_group_clusters"]})
if not all(r["exact_match"] for r in validation):
    raise SystemExit("VALIDATION FAILED: frozen primary rows not reproduced")
print("validation: all 10 frozen primary rows reproduced exactly")

# ---- step 2: baseline records (old verified shards) + max-matching onset F1 (new shards)
old_shards = sorted((PARENT / "full").glob("shard_*.jsonl"))
new_shards = sorted((HERE / "shards").glob("f1mm_*.jsonl"))
brecs = []
for p in old_shards:
    brecs.extend(read_jsonl(p))
f1recs = {}
for p in new_shards:
    for r in read_jsonl(p):
        if r["task_id"] in f1recs:
            raise SystemExit(f"duplicate f1 task {r['task_id']}")
        f1recs[r["task_id"]] = r
if len(brecs) != 50283 or len(f1recs) != 50283:
    raise SystemExit(f"record count {len(brecs)} / {len(f1recs)} != 50283")
if not all(r["sealed_hash_match"] and r["sealed_pattern_nll_match"] for r in brecs):
    raise SystemExit("sealed hash / pattern_nll mismatch in old shards")
if not all(r["sealed_hash_match"] and r["greedy_equals_previous_shard"] for r in f1recs.values()):
    raise SystemExit("sealed hash mismatch or greedy != previous shard in new F1 shards")
bidx = {}
for r in brecs:
    f = f1recs[r["task_id"]]
    if f["variant_events_sha256"] != r["variant_events_sha256"]:
        raise SystemExit(f"hash differs between old and new shard: {r['task_id']}")
    r["metrics"].update({k: v for k, v in f["metrics"].items()})
    gid = int(r["sid"].split("_")[1])
    k = (gid, r["course"], r["condition"], int(r["dose_index"]), int(r["replicate_index"]))
    if k in bidx:
        raise SystemExit(f"duplicate key {k}")
    bidx[k] = r

BASELINES = {
    # key: (field, worse-direction, chart_only, time_only, reported)
    "lm_perplexity": ("lm_perplexity", "increase", True, False, True),
    "self_similarity": ("self_similarity_si_long", "decrease", True, False, True),
    "ioi_hist_js": ("ioi_hist_js_vs_ref_bits", "increase", True, True, True),
    "onset_f1_50ms": ("onset_f1_maxmatch_50ms", "decrease", False, True, True),
    "onset_f1_20ms": ("onset_f1_maxmatch_20ms", "decrease", False, True, True),
    "onset_f1_50ms_greedy_superseded": ("onset_f1_greedy_50ms", "decrease", False, True, False),
    "onset_f1_20ms_greedy_superseded": ("onset_f1_greedy_20ms", "decrease", False, True, False),
}
TYPE_ONLY_EDITS = {"C3_type_shuffle", "C4_loop_collapse", "C5_blandification"}
TOL = 1e-9

bobs = []
for o in observations:
    k = (int(o["group_id"]), o["course"], o["condition"], int(o["dose_index"]), int(o["replicate_index"]))
    r = bidx.get(k)
    if r is None or r["variant_events_sha256"] != o["variant_events_sha256"]:
        raise SystemExit(f"observation without matching baseline record: {k}")
    bobs.append({**o, "metrics": {b: r["metrics"].get(spec[0]) for b, spec in BASELINES.items()}})
grouped = defaultdict(list)
for o in bobs:
    grouped[(int(o["group_id"]), o["course"], o["condition"], int(o["dose_index"]))].append(o)
bmeans = []
for k, rows in sorted(grouped.items()):
    expected = 1 if rows[0]["condition_kind"] == "official" else 5
    assert len(rows) == expected, k
    usable = [r for r in rows if r["status"] == "success" and not r["corruption_noop"]]
    mm = {}
    for b in BASELINES:
        vals = [float(r["metrics"][b]) for r in usable if r["metrics"].get(b) is not None]
        mm[b] = float(np.mean(vals)) if len(vals) == expected else None  # frozen reducer (unchanged)
    bmeans.append({"group_id": k[0], "course": k[1], "condition": k[2], "dose_index": k[3],
                   "status": "valid" if len(usable) == expected else "invalid", "metrics": mm})
fstat = {(int(r["group_id"]), r["course"], r["condition"], int(r["dose_index"])): r["status"] for r in frozen_means}
bstat = {(r["group_id"], r["course"], r["condition"], r["dose_index"]): r["status"] for r in bmeans}
if fstat != bstat:
    raise SystemExit("chart-dose status differs from frozen table")
bindex = FA._index_means(bmeans)

bcontract = {"probes": contract["probes"], "controls": {}, "control_absolute_tolerances": {b: TOL for b in BASELINES}}
for cid in ("CTRL_identity", "CTRL_joint_time_origin_shift", "CTRL_color_bijection", "C2_anchor_shift"):
    fams, mets = [], []
    for b, (_f, _d, chart_only, time_only, _rep) in BASELINES.items():
        if (cid in ("CTRL_identity", "CTRL_joint_time_origin_shift") or (cid == "C2_anchor_shift" and chart_only)
                or (cid == "CTRL_color_bijection" and time_only)):
            fams.append(f"bl_{b}"); mets.append(b)
    bcontract["controls"][cid] = {"scoped_families": fams, "metrics": mets}
bdiag, bpass, bcourse = FA._control_diagnostics(bobs, bcontract)

FROZEN_SHAM = {p["probe"]: (p["scoped_sham"], int(p["scoped_sham_dose_index"])) for p in contract["primary_pairs"]}
EDITS = list(contract["probes"])


def song_mean(cc, cond, dose, b):
    per = defaultdict(list)
    for gid, course in cc:
        per[gid].append(bindex[(gid, course, cond, dose)]["metrics"][b])
    return float(np.mean([np.mean(v) for v in per.values()])) if per else None


results = []
for b, (field, worse, chart_only, time_only, reported) in BASELINES.items():
    fam = f"bl_{b}"
    required = [(c, b) for c, ctl in bcontract["controls"].items() if fam in ctl["scoped_families"]]
    controls_pass = all(bpass.get(k, False) for k in required)
    for probe in EDITS:
        sham, sham_dose = FROZEN_SHAM[probe]
        sham_rule = "frozen"
        if not any(c == sham for c, _ in required):
            sham, sham_dose, sham_rule = "CTRL_identity", 0, "identity_fallback"
        pair = {"pair_id": f"POSTHOC_BASELINE__{probe}__{b}", "probe": probe, "metric": b, "family": fam,
                "expected_raw_direction": worse, "scoped_sham": sham, "scoped_sham_dose_index": sham_dose}
        gv, ex, cc = FA._pair_group_values(bindex, pair, bcontract, bcourse)
        s = FA._cluster_summary(gv, draws=DRAWS, seed=FA._pair_seed(MASTER, pair["pair_id"]), levels=(NOM, ADJ))
        v = verdict_of(s, controls_pass)
        ci = s["confidence_intervals"].get(ADJ_KEY, {})
        vals = np.asarray(list(gv.values()), dtype=float) if gv else np.zeros((0, 2))
        all_zero = bool(len(vals)) and bool(np.all(vals == 0.0))
        per_dose_invariant = {}
        for di in (1, 2, 3):
            eq = [bidx[(gid, course, probe, di, ri)]["metrics"][field] == bidx[(gid, course, "official", 0, 0)]["metrics"][field]
                  for gid, course in cc for ri in range(1, 6)]
            per_dose_invariant[di] = float(np.mean(eq)) if eq else None
        rep_invariant = bool(cc) and all(x == 1.0 for x in per_dose_invariant.values())
        if not controls_pass:
            label = "CONTROL-FAIL"
        elif time_only and probe in TYPE_ONLY_EDITS:
            label = "N/A-by-design" if rep_invariant else "N/A-by-design?CHANGED"
        elif v == "SURVIVES":
            label = "DETECTS"
        elif all_zero or rep_invariant:
            label = "BLIND"
        elif ci and (ci["dose_statistic"][0] > 0 or ci["target_minus_sham"][0] > 0):
            label = "REWARDS"
        else:
            label = "MISSES"
        dose_song_means = {"official": song_mean(cc, "official", 0, b),
                           **{f"d{di}": song_mean(cc, probe, di, b) for di in (1, 2, 3)}}
        base = dose_song_means["official"]; strong = dose_song_means["d3"]
        results.append({
            "analysis": "POST_HOC_SECONDARY_heldout_baselines", "baseline": b, "field": field, "reported": reported,
            "worse_direction": worse, "probe": probe, "dose_values": contract["probes"][probe]["dose_values"],
            "sham": f"{sham}:d{sham_dose}", "sham_rule": sham_rule,
            "required_controls": [c for c, _ in required], "controls_pass": controls_pass,
            "n_songs": s["n_group_clusters"], "n_courses": len(cc), "n_excluded_courses": len(ex),
            "dose_statistic_mean": s["dose_statistic_mean"], "dose_ci995": ci.get("dose_statistic"),
            "contrast_mean": s["target_minus_sham_mean"], "contrast_ci995": ci.get("target_minus_sham"),
            "all_song_values_exactly_zero": all_zero,
            "fraction_replicates_equal_unedited_by_dose": per_dose_invariant,
            "replicate_level_exactly_invariant": rep_invariant,
            "report_as": ("exactly unchanged at every seed and strength" if rep_invariant else "statistics"),
            "frozen_rule_verdict": v, "label": label,
            "song_mean_by_dose": dose_song_means,
            "strongest_minus_unedited_song_mean": (strong - base) if (strong is not None and base is not None) else None,
            "strongest_minus_unedited_pct": (100.0 * (strong - base) / base) if (base not in (None, 0.0) and strong is not None) else None,
        })

# ChartGenEval matched-measurement row (sealed, prespecified; reproduced exactly above)
cge_row = {}
for probe in EDITS:
    rows = [r for r in validation if r["probe"] == probe]
    cge_row[probe] = {"pairs": [r["pair_id"] for r in rows], "verdicts": [r["verdict"] for r in rows],
                      "label": "DETECTS" if rows and all(r["verdict"] == "SURVIVES" for r in rows) else "NOT-ALL-SURVIVE"}

out = {
    "analysis": "POST_HOC_SECONDARY_heldout_baselines_FINAL (not preregistered; held-out labels already opened by confirmatory-primary-20260712; the paper's 'fixed before inspecting held-out results' applies only to the nine primary tests)",
    "onset_f1_definition": "mir_eval 0.8.2 onset.f_measure: maximum bipartite matching (Hopcroft-Karp), inclusive window |ref-est|<=tol, tol in {0.05, 0.02} s, reference = unedited official chart (shifted with the notes under the joint time-origin control); cross-checked against scipy maximum_bipartite_matching on every record",
    "label_rules": {
        "DETECTS": "frozen rule SURVIVES (both 99.5% upper bounds < 0 in the worse direction, controls pass, >=72 songs)",
        "BLIND": "every seed at every strength equals the unedited chart's value exactly",
        "REWARDS": "a 99.5% lower bound > 0, i.e. moves toward the conventionally better value (lower perplexity; higher self-similarity -- only if higher is read as better, which is ambiguous)",
        "N/A-by-design": "time-only metric (IOI-JS, onset F1) under a type-only edit (C3, C4, C5); exactly unchanged by construction, not an empirical failure",
    },
    "inputs": {"observations.jsonl": sha(TABLES / "observations.jsonl"), "chart_dose_means.jsonl": sha(TABLES / "chart_dose_means.jsonl"),
               "contract.json": sha(RUN / "spec_snapshots/contract.json"), "primary_results.json": sha(PRIMARY),
               "analyze.py": sha(EXP / "analyze.py"), "SPEC_posthoc_baselines.md": sha(PARENT / "SPEC_posthoc_baselines.md"),
               **{f"full/{p.name}": sha(p) for p in old_shards}, **{f"final/shards/{p.name}": sha(p) for p in new_shards},
               "final/f1_maxmatch.py": sha(HERE / "f1_maxmatch.py"), "baseline_holdout.py": sha(PARENT / "baseline_holdout.py")},
    "validation_frozen_rows": validation,
    "chartgeneval_measurement_row": cge_row,
    "control_diagnostics_required": bdiag,
    "results": results,
}
(HERE / "baseline_final_results.json").write_text(json.dumps(out, indent=1))


def f(x, n=3):
    return "NA" if x is None else f"{x:+.{n}f}"


for r in results:
    d = r["dose_ci995"] or [None, None]; c = r["contrast_ci995"] or [None, None]
    sm = r["song_mean_by_dose"]
    print(f"{r['baseline'][:22]:22s} {r['probe'][:17]:17s} rho {f(r['dose_statistic_mean'])} [{f(d[0])},{f(d[1])}] "
          f"dlt {f(r['contrast_mean'],4)} [{f(c[0],4)},{f(c[1],4)}] n={r['n_songs']} {r['frozen_rule_verdict'][:5]} {r['label']:14s} "
          f"means {f(sm['official'],3)}|{f(sm['d1'],3)}|{f(sm['d2'],3)}|{f(sm['d3'],3)} inv {r['fraction_replicates_equal_unedited_by_dose']} sham={r['sham']}")
print("controls:", all(d["passes"] for d in bdiag), max(d["max_absolute_delta"] or 0 for d in bdiag))
print("CGE row:", {k: v["label"] for k, v in cge_row.items()})
