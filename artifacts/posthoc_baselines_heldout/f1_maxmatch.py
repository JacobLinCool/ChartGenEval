"""POST-HOC SECONDARY held-out baseline analysis (not preregistered): onset F1
recomputed with MAXIMUM bipartite matching (mir_eval.onset.f_measure).

Regenerates the frozen confirmatory_holdout_v1 variants for canonical songs in
[start, stop) with the verified pipeline in ../baseline_holdout.py (frozen
run.py / contract / probes / seeds / LM, imported read-only), re-checks every
variant hash, input hash and pattern_nll against the sealed raw records, and
computes onset F1 against the unedited chart at 50 ms and 20 ms with:
  * mir_eval 0.8.2 onset.f_measure (Hopcroft-Karp maximum matching)  -> reported
  * scipy.sparse.csgraph.maximum_bipartite_matching (independent check)
  * the old greedy matcher (must equal the values in ../full/shard_*.jsonl)
Writes only to this scratch directory. Caller sets PYTHONDONTWRITEBYTECODE=1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mir_eval
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import baseline_holdout as BH  # noqa: E402  verified regeneration pipeline (read-only)

from contract import derive_rng_seed, load_contract, load_json  # noqa: E402
from chartgeneval.events import sorted_hits  # noqa: E402
from chartgeneval.probes import apply_probe  # noqa: E402

frozen_run = BH.frozen_run


def f1_mireval(ref, est, tol):
    f, p, r = mir_eval.onset.f_measure(np.asarray(ref, float), np.asarray(est, float), window=tol)
    n = len(mir_eval.util.match_events(np.asarray(ref, float), np.asarray(est, float), tol)) if len(ref) and len(est) else 0
    return float(f), n


def n_match_scipy(ref, est, tol):
    ref = np.asarray(ref, float); est = np.asarray(est, float)
    if len(ref) == 0 or len(est) == 0:
        return 0
    hr, he = BH_hits(ref, est, tol)
    if not hr:
        return 0
    g = csr_matrix((np.ones(len(hr), dtype=np.int8), (np.asarray(he), np.asarray(hr))), shape=(len(est), len(ref)))
    m = maximum_bipartite_matching(g, perm_type="column")
    return int(np.sum(m >= 0))


def BH_hits(ref, est, tol):
    # same inclusive window as mir_eval._fast_hit_windows, written independently
    order = np.argsort(ref, kind="stable")
    rs = ref[order]
    hr, he = [], []
    lo = np.searchsorted(rs, est - tol, side="left")
    hi = np.searchsorted(rs, est + tol, side="right")
    for j in range(len(est)):
        for k in range(lo[j], hi[j]):
            hr.append(int(order[k])); he.append(j)
    return hr, he


def f1_from(n, n_ref, n_est):
    if n_ref == 0 or n_est == 0:
        return 0.0
    p = n / n_est; r = n / n_ref
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--stop", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    config, contract, _, _ = load_contract(BH.EXP / "configs/confirmatory.json")
    sealed = {}
    with open(BH.RAW) as fh:
        for line in fh:
            r = json.loads(line)
            if args.start <= r["canonical_song_index"] < args.stop:
                sealed[(r["sid"], r["task_id"])] = (
                    r["variant_events_sha256"], r["input_chart_sha256"], r["metrics"].get("pattern_nll"))
    # old greedy values from the verified shards (for an equality check)
    old = {}
    for p in sorted((HERE.parent / "full").glob("shard_*.jsonl")):
        with open(p) as fh:
            for line in fh:
                r = json.loads(line)
                if args.start <= r["canonical_song_index"] < args.stop:
                    old[r["task_id"]] = (r["metrics"]["onset_f1_vs_ref_50ms"], r["metrics"]["onset_f1_vs_ref_20ms"], r["variant_events_sha256"])
    lm, n_rows, n_charts = BH.train_lm(contract)
    lm_fp = frozen_run._lm_state_fingerprint(lm)
    manifest = load_json(BH.RUN_MANIFEST)
    seeds = contract["replicate_base_seeds"]
    n = mism = nll_mism = old_mism = scipy_mism = 0
    with open(args.out, "w") as out:
        for raw_index, cidx, row in BH.iter_test_rows(args.start, args.stop):
            sid = f"group_{int(row['group_id'])}"
            for course, chart, course_struct in frozen_run._official_charts(row):
                events = sorted_hits(chart.events)
                segments = (course_struct or {}).get("segments") or []
                grid = {"segments": segments} if segments else None
                duration = (events[-1][0] + 1.0) if events else 0.0
                input_hash = frozen_run._chart_input_hash(course, chart, course_struct)
                cells = [("official", 0, 0, 0, events)]
                for ctrl in ("CTRL_identity", "CTRL_joint_time_origin_shift", "CTRL_color_bijection"):
                    for ri, _bs in enumerate(seeds, start=1):
                        ev, _g, _d = frozen_run._control_variant(ctrl, events, grid, duration)
                        cells.append((ctrl, 0, 0, ri, ev))
                for probe_id, pc in contract["probes"].items():
                    for di, dv in enumerate(pc["dose_values"], start=1):
                        for ri, bs in enumerate(seeds, start=1):
                            rng = np.random.default_rng(derive_rng_seed(bs, sid, course, probe_id, di))
                            ev, _noop = apply_probe(probe_id, events, dv, rng, chart.bpm, lm=lm, course=course)
                            cells.append((probe_id, di, dv, ri, ev))
                for cond, di, dv, ri, ev in cells:
                    task_id = f"{sid}|{course}|{cond}|d{di}|r{ri}"
                    vh = frozen_run._events_hash(ev)
                    s = sealed.get((sid, task_id))
                    ok = s is not None and s[0] == vh and s[1] == input_hash
                    ref_ev = [(t + 1.0, c) for t, c in events] if cond == "CTRL_joint_time_origin_shift" else events
                    ref_t = [t for t, _ in sorted_hits(ref_ev)]
                    est_t = [t for t, _ in sorted_hits(ev)]
                    m = {}
                    for tol, tag in ((0.05, "50ms"), (0.02, "20ms")):
                        f_me, n_me = f1_mireval(ref_t, est_t, tol)
                        n_sp = n_match_scipy(ref_t, est_t, tol)
                        scipy_ok = (n_sp == n_me) and abs(f1_from(n_sp, len(ref_t), len(est_t)) - f_me) <= 1e-15
                        scipy_mism += (not scipy_ok)
                        m[f"onset_f1_maxmatch_{tag}"] = f_me
                        m[f"onset_matches_maxmatch_{tag}"] = n_me
                        m[f"onset_f1_greedy_{tag}"] = BH.onset_f1(ref_t, est_t, tol)
                    m["n_ref"] = len(ref_t); m["n_est"] = len(est_t)
                    o = old.get(task_id)
                    old_ok = o is not None and o[2] == vh and o[0] == m["onset_f1_greedy_50ms"] and o[1] == m["onset_f1_greedy_20ms"]
                    old_mism += (not old_ok)
                    mism += (not ok)
                    n += 1
                    out.write(json.dumps({
                        "analysis": "POST_HOC_SECONDARY_heldout_baselines_onsetF1_maxmatch",
                        "sid": sid, "canonical_song_index": cidx, "course": course,
                        "condition": cond, "dose_index": di, "dose_value": dv, "replicate_index": ri,
                        "task_id": task_id, "variant_events_sha256": vh,
                        "sealed_hash_match": ok, "greedy_equals_previous_shard": old_ok,
                        "metrics": m}) + "\n")
    print(json.dumps({
        "records": n, "sealed_records_in_range": len(sealed), "sealed_hash_mismatches": mism,
        "greedy_vs_previous_shard_mismatches": old_mism, "mireval_vs_scipy_mismatches": scipy_mism,
        "lm_state_sha256": lm_fp, "run_manifest_contains_lm_fp": lm_fp in json.dumps(manifest),
        "mir_eval_version": mir_eval.__version__, "sec": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
