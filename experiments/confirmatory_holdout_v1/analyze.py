#!/usr/bin/env python3
"""Run the predeclared song-cluster confirmatory analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from contract import (
    REPO_ROOT,
    file_sha256,
    invocation,
    load_json,
    resolve_repo_path,
    runtime_dependency_contract,
    stable_json,
    utc_now,
    verify_source_snapshot_files,
    write_json_exclusive,
)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _validate_analysis_source_scan(
    scan_rows: list[dict],
    context_rows: list[dict],
    run_manifest: dict,
    run_config: dict,
) -> None:
    """Recheck the raw-row-to-canonical-context projection at analysis time."""
    dataset = run_config["dataset"]
    if not scan_rows:
        raise ValueError("analysis source scan is empty")
    split = str(dataset["split"])
    canonical_rows = []
    aliases = []
    for expected_index, row in enumerate(scan_rows):
        if row.get("schema_version") != "chartgeneval.source_scan_raw.v1":
            raise ValueError("analysis source-scan schema drift")
        if int(row.get("source_row_index")) != expected_index:
            raise ValueError("analysis source scan is not in raw stored-row order")
        if dataset["source"] == "huggingface" and row.get(
            "stored_split_row_id"
        ) != f"{split}_row_{expected_index:05d}":
            raise ValueError("analysis stored split row ID/index mismatch")
        if row.get("canonical") is True:
            if row.get("action") != "canonical_context":
                raise ValueError("analysis canonical source row has wrong action")
            canonical_rows.append(row)
        elif row.get("canonical") is False:
            if (
                row.get("action") != "filtered_alias"
                or row.get("canonical_song_index") is not None
                or row.get("manifest_source_row_id") is not None
            ):
                raise ValueError("analysis alias source row leaked into canonical metadata")
            aliases.append(row)
        else:
            raise ValueError("analysis source-scan canonical flag is not boolean")
    if [int(row["canonical_song_index"]) for row in canonical_rows] != list(
        range(len(canonical_rows))
    ):
        raise ValueError("analysis canonical indices do not follow filtered row order")
    if len({int(row["group_id"]) for row in canonical_rows}) != len(canonical_rows):
        raise ValueError("analysis source scan has duplicate canonical group_id values")
    if len({str(row["audio_sha256"]) for row in canonical_rows}) != len(
        canonical_rows
    ):
        raise ValueError("analysis source scan has duplicate canonical audio hashes")

    context_start = int(dataset["sampling_context_canonical_start_inclusive"])
    context_stop = int(dataset["sampling_context_canonical_stop_exclusive"])
    scanned_context = [
        row
        for row in canonical_rows
        if context_start <= int(row["canonical_song_index"]) < context_stop
    ]

    def identity(row: dict) -> tuple:
        return (
            int(row["source_row_index"]),
            row["stored_split_row_id"],
            row.get("manifest_source_row_id"),
            int(row["canonical_song_index"]),
            int(row["group_id"]),
            str(row["audio_sha256"]),
        )

    if [identity(row) for row in scanned_context] != [
        identity(row) for row in context_rows
    ]:
        raise ValueError("analysis context includes aliases or differs from source scan")
    observed_counts = (
        len(scan_rows),
        len(canonical_rows),
        len(aliases),
    )
    manifest_counts = (
        int(run_manifest["counts"]["scanned_source_rows"]),
        int(run_manifest["counts"]["canonical_context_rows"]),
        int(run_manifest["counts"]["filtered_alias_rows"]),
    )
    if observed_counts != manifest_counts:
        raise ValueError("analysis source-scan counts differ from run manifest")
    if dataset["scan_full_split"]:
        expected_counts = (
            int(dataset["expected_source_rows"]),
            int(dataset["expected_canonical_songs"]),
            int(dataset["expected_filtered_alias_rows"]),
        )
        if observed_counts != expected_counts:
            raise ValueError("analysis raw/canonical/alias contract drift")
    elif len(canonical_rows) != context_stop or scan_rows[-1]["canonical"] is not True:
        raise ValueError("analysis development scan extends past canonical prefix")

    if dataset["source"] != "huggingface":
        return
    split_manifest = load_json(
        resolve_repo_path(run_config["clean_split_manifest_path"])
    )["manifest"]
    entries = [
        (manifest_row_id, entry)
        for manifest_row_id, entry in split_manifest.items()
        if entry.get("split") == split
    ]
    remaining = Counter(
        (
            int(entry["group_id"]),
            bool(entry["canonical"]),
            str(entry["audio_sha256"]),
        )
        for _manifest_row_id, entry in entries
    )
    canonical_manifest_ids = {
        (int(entry["group_id"]), str(entry["audio_sha256"])): manifest_row_id
        for manifest_row_id, entry in entries
        if bool(entry["canonical"])
    }
    if len(canonical_manifest_ids) != sum(
        1 for _manifest_row_id, entry in entries if bool(entry["canonical"])
    ):
        raise ValueError("analysis split manifest has duplicate canonical identities")
    for row in scan_rows:
        key = (
            int(row["group_id"]),
            bool(row["canonical"]),
            str(row["audio_sha256"]),
        )
        if remaining[key] <= 0:
            raise ValueError("analysis source row is absent/exhausted in split manifest")
        remaining[key] -= 1
        if row["canonical"] is True and row.get(
            "manifest_source_row_id"
        ) != canonical_manifest_ids.get((key[0], key[2])):
            raise ValueError("analysis canonical row has wrong manifest source ID")
    if dataset["scan_full_split"] and any(remaining.values()):
        raise ValueError("analysis full source scan does not exhaust split manifest")


def _rank_average(values: list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _spearman_or_zero(x: list[float], y: list[float]) -> float:
    """Spearman rho; a constant response is frozen as zero sensitivity."""
    rx = _rank_average(x)
    ry = _rank_average(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _oriented(value: float, direction: str) -> float:
    return float(value) if direction == "decrease" else -float(value)


def _pair_seed(master: int, pair_id: str) -> int:
    digest = hashlib.sha256(f"{master}|{pair_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _cluster_summary(group_values: dict[int, tuple[float, float]], *, draws: int, seed: int, levels: tuple[float, ...]) -> dict:
    group_ids = sorted(group_values)
    values = np.asarray([group_values[group_id] for group_id in group_ids], dtype=float)
    if len(values) == 0:
        return {
            "n_group_clusters": 0,
            "group_ids": [],
            "dose_statistic_mean": None,
            "target_minus_sham_mean": None,
            "confidence_intervals": {},
        }
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, len(values), size=(draws, len(values)))
    boot = values[sample_index].mean(axis=1)
    intervals = {}
    for level in levels:
        tail = (1.0 - level) / 2.0
        intervals[f"{level:.6f}"] = {
            "dose_statistic": [
                float(np.quantile(boot[:, 0], tail)),
                float(np.quantile(boot[:, 0], 1.0 - tail)),
            ],
            "target_minus_sham": [
                float(np.quantile(boot[:, 1], tail)),
                float(np.quantile(boot[:, 1], 1.0 - tail)),
            ],
        }
    return {
        "n_group_clusters": len(group_ids),
        "group_ids": group_ids,
        "dose_statistic_mean": float(np.mean(values[:, 0])),
        "target_minus_sham_mean": float(np.mean(values[:, 1])),
        "confidence_intervals": intervals,
    }


def _index_means(rows: list[dict]) -> dict[tuple[int, str, str, int], dict]:
    out = {}
    for row in rows:
        key = (
            int(row["group_id"]),
            row["course"],
            row["condition"],
            int(row["dose_index"]),
        )
        if key in out:
            raise ValueError(f"duplicate canonical chart-dose key: {key}")
        out[key] = row
    return out


def _control_diagnostics(observations: list[dict], contract: dict) -> tuple[list[dict], dict[tuple[str, str], bool], dict[tuple[str, str, str, str], bool]]:
    observation_index = {}
    for row in observations:
        key = (
            int(row["group_id"]),
            row["course"],
            row["condition"],
            int(row["dose_index"]),
            int(row["replicate_index"]),
        )
        if key in observation_index:
            raise ValueError(f"duplicate canonical observation key: {key}")
        observation_index[key] = row
    official_keys = sorted(
        (int(row["group_id"]), row["course"])
        for row in observations
        if row["condition"] == "official" and int(row["dose_index"]) == 0
    )
    diagnostics = []
    passes = {}
    course_passes = {}
    for control_id, control in contract["controls"].items():
        doses = (1, 2, 3) if control_id == "C2_anchor_shift" else (0,)
        for metric in control["metrics"]:
            deltas = []
            missing = 0
            n_baseline_evaluable = 0
            for group_id, course in official_keys:
                base = observation_index[(group_id, course, "official", 0, 0)]
                base_value = (
                    base["metrics"].get(metric)
                    if base["status"] == "success" and not base["corruption_noop"]
                    else None
                )
                if base_value is None:
                    continue
                n_baseline_evaluable += 1
                chart_complete = True
                for dose in doses:
                    for replicate in range(1, 6):
                        control_row = observation_index.get(
                            (group_id, course, control_id, dose, replicate)
                        )
                        value = None
                        if (
                            control_row is not None
                            and control_row["status"] == "success"
                            and not control_row["corruption_noop"]
                        ):
                            value = control_row["metrics"].get(metric)
                        if value is None:
                            missing += 1
                            chart_complete = False
                        else:
                            delta = abs(float(value) - float(base_value))
                            deltas.append(delta)
                            if delta > float(contract["control_absolute_tolerances"][metric]):
                                chart_complete = False
                course_passes[(group_id, course, control_id, metric)] = chart_complete
            tolerance = float(contract["control_absolute_tolerances"][metric])
            maximum = max(deltas) if deltas else None
            expected_cells = n_baseline_evaluable * len(doses) * 5
            passed = (
                n_baseline_evaluable > 0
                and len(deltas) == expected_cells
                and missing == 0
                and maximum is not None
                and maximum <= tolerance
            )
            diagnostics.append(
                {
                    "control": control_id,
                    "metric": metric,
                    "scoped_families": control["scoped_families"],
                    "n_baseline_evaluable_charts": n_baseline_evaluable,
                    "n_expected_control_replicates": expected_cells,
                    "n_evaluable_control_replicates": len(deltas),
                    "n_missing_control_replicates": missing,
                    "max_absolute_delta": maximum,
                    "absolute_tolerance": tolerance,
                    "passes": passed,
                }
            )
            passes[(control_id, metric)] = passed
    return diagnostics, passes, course_passes


def _pair_group_values(index: dict, pair: dict, contract: dict, control_course_passes: dict) -> tuple[dict, list[dict], list[tuple[int, str]]]:
    probe = pair["probe"]
    metric = pair["metric"]
    severity = [float(x) for x in contract["probes"][probe]["severity"]]
    direction = pair["expected_raw_direction"]
    sham = pair["scoped_sham"]
    sham_dose = int(pair["scoped_sham_dose_index"])
    official_courses = sorted(
        (group_id, course)
        for group_id, course, condition, dose in index
        if condition == "official" and dose == 0
    )
    chart_values = defaultdict(list)
    excluded = []
    complete_courses = []
    required_controls = [
        control_id
        for control_id, control in contract["controls"].items()
        if pair.get("family") in control["scoped_families"] and metric in control["metrics"]
    ]
    for group_id, course in official_courses:
        rows = [index.get((group_id, course, "official", 0))]
        rows.extend(index.get((group_id, course, probe, dose)) for dose in (1, 2, 3))
        sham_row = index.get((group_id, course, sham, sham_dose))
        rows.append(sham_row)
        labels = ["official", f"{probe}:1", f"{probe}:2", f"{probe}:3", f"{sham}:{sham_dose}"]
        reason = None
        if any(row is None for row in rows):
            reason = "missing_chart_condition_dose"
        elif any(row["status"] != "valid" for row in rows):
            reason = "invalid_replicate_group"
        else:
            values = [row["metrics"].get(metric) for row in rows]
            if any(value is None for value in values):
                reason = "missing_registered_metric"
        if reason:
            excluded.append(
                {
                    "pair_id": pair["pair_id"],
                    "group_id": group_id,
                    "course": course,
                    "reason": reason,
                    "required_cells": labels,
                }
            )
            continue
        failed_controls = [
            control_id
            for control_id in required_controls
            if not control_course_passes.get(
                (group_id, course, control_id, metric), False
            )
        ]
        if failed_controls:
            excluded.append(
                {
                    "pair_id": pair["pair_id"],
                    "group_id": group_id,
                    "course": course,
                    "reason": "required_control_incomplete_or_outside_tolerance",
                    "required_cells": failed_controls,
                }
            )
            continue
        target_raw = [float(rows[i]["metrics"][metric]) for i in range(4)]
        sham_raw = float(sham_row["metrics"][metric])
        oriented = [_oriented(value, direction) for value in target_raw]
        chart_values[group_id].append(
            (
                _spearman_or_zero(severity, oriented),
                oriented[3] - _oriented(sham_raw, direction),
            )
        )
        complete_courses.append((group_id, course))
    # Frozen available-complete-course rule: one or more complete courses makes
    # a group available; both statistics use the same complete courses.
    group_values = {
        group_id: (
            float(np.mean([value[0] for value in values])),
            float(np.mean([value[1] for value in values])),
        )
        for group_id, values in sorted(chart_values.items())
        if values
    }
    return group_values, excluded, complete_courses


def _summarize_context(context_rows: list[dict]) -> list[dict]:
    fields = (
        "chart_count",
        "course_count",
        "song_mean_bpm",
        "song_mean_note_rate_nps",
        "song_mean_level",
    )
    output = []
    groups = sorted({row["panel_group"] for row in context_rows})
    for group in groups:
        rows = [row for row in context_rows if row["panel_group"] == group]
        for field in fields:
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            output.append(
                {
                    "panel_group": group,
                    "metric": field,
                    "n_songs": len(rows),
                    "n_nonmissing": len(values),
                    "mean": float(np.mean(values)) if values else None,
                    "median": float(np.median(values)) if values else None,
                    "p10": float(np.quantile(values, 0.10)) if values else None,
                    "p90": float(np.quantile(values, 0.90)) if values else None,
                }
            )
    return output


def _dose_descriptives(
    index: dict,
    complete_courses: list[tuple[int, str]],
    *,
    probe: str,
    metric: str,
    direction: str | None,
    draws: int,
    seed: int,
    confidence_level: float,
) -> list[dict]:
    output = []
    for dose in (0, 1, 2, 3):
        condition = "official" if dose == 0 else probe
        by_group = defaultdict(list)
        for group_id, course in complete_courses:
            row = index[(group_id, course, condition, dose)]
            value = float(row["metrics"][metric])
            by_group[group_id].append(value)
        song_raw = np.asarray(
            [float(np.mean(values)) for _group_id, values in sorted(by_group.items())], dtype=float
        )
        tail = (1.0 - confidence_level) / 2.0
        if len(song_raw):
            rng = np.random.default_rng(seed + dose)
            sample_idx = rng.integers(0, len(song_raw), size=(draws, len(song_raw)))
            boot = song_raw[sample_idx].mean(axis=1)
            ci = [float(np.quantile(boot, tail)), float(np.quantile(boot, 1.0 - tail))]
            raw_mean = float(np.mean(song_raw))
        else:
            ci, raw_mean = [None, None], None
        output.append(
            {
                "dose_index": dose,
                "n_group_clusters": len(song_raw),
                "raw_song_mean": raw_mean,
                "raw_song_mean_ci": ci,
                "oriented_song_mean": (
                    float(np.mean([_oriented(value, direction) for value in song_raw]))
                    if direction is not None and len(song_raw)
                    else None
                ),
            }
        )
    return output


def _seed_robustness(
    observations: list[dict],
    complete_courses: list[tuple[int, str]],
    pair: dict,
) -> list[dict]:
    index = {
        (
            int(row["group_id"]),
            row["course"],
            row["condition"],
            int(row["dose_index"]),
            int(row["replicate_index"]),
        ): row
        for row in observations
    }
    output = []
    for replicate in range(1, 6):
        by_group = defaultdict(list)
        for group_id, course in complete_courses:
            target = index[(group_id, course, pair["probe"], 3, replicate)]
            sham = index[
                (
                    group_id,
                    course,
                    pair["scoped_sham"],
                    int(pair["scoped_sham_dose_index"]),
                    replicate,
                )
            ]
            target_value = target["metrics"].get(pair["metric"])
            sham_value = sham["metrics"].get(pair["metric"])
            if target_value is None or sham_value is None:
                raise ValueError("complete-course seed robustness encountered missing metric")
            by_group[group_id].append(
                _oriented(float(target_value), pair["expected_raw_direction"])
                - _oriented(float(sham_value), pair["expected_raw_direction"])
            )
        group_values = [float(np.mean(v)) for _group_id, v in sorted(by_group.items())]
        output.append(
            {
                "replicate_index": replicate,
                "replicate_base_seed": target["replicate_base_seed"] if complete_courses else None,
                "n_group_clusters": len(group_values),
                "target_minus_sham_song_mean": (
                    float(np.mean(group_values)) if group_values else None
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0]) if rows else []
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(table_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    current_runtime = runtime_dependency_contract()
    table_dir = resolve_repo_path(table_dir)
    table_manifest_path = table_dir / "MANIFEST.json"
    table_manifest = load_json(table_manifest_path)
    if table_manifest.get("runtime_dependency_contract") != current_runtime:
        raise ValueError("analysis runtime differs from canonical transformation runtime")
    for name, expected in table_manifest["table_hashes"].items():
        if file_sha256(table_dir / name) != expected:
            raise ValueError(f"canonical table hash mismatch: {name}")
    run_dir = table_dir.parent
    run_manifest_path = run_dir / "MANIFEST.json"
    run_manifest = load_json(run_manifest_path)
    verify_source_snapshot_files(run_manifest["code_version"])
    if file_sha256(run_manifest_path) != table_manifest["source_run_manifest_sha256"]:
        raise ValueError("run manifest changed after transformation")
    if table_manifest.get("source_raw_hashes") != run_manifest.get("raw_hashes"):
        raise ValueError("canonical manifest raw hashes differ from run manifest")
    for name, expected in run_manifest["raw_hashes"].items():
        if file_sha256(run_dir / name) != expected:
            raise ValueError(f"raw evidence changed before analysis: {name}")
    if table_manifest.get("cartesian_validation", {}).get("validated") is not True:
        raise ValueError("canonical manifest lacks successful Cartesian validation")
    if table_manifest.get("source_spec_snapshot_hashes") != run_manifest.get(
        "spec_snapshot_hashes"
    ):
        raise ValueError("canonical source snapshot hashes differ from run manifest")
    snapshots = run_dir / "spec_snapshots"
    for name, expected in run_manifest["spec_snapshot_hashes"].items():
        if file_sha256(snapshots / name) != expected:
            raise ValueError(f"analysis snapshot hash mismatch: {name}")
    panels = load_json(table_dir / "panels.json")
    if panels["primary_panel"]["selected_group_ids_ordered"] != run_manifest[
        "selected_group_ids_ordered"
    ]:
        raise ValueError("analysis panel differs from run manifest")
    if panels["primary_panel"]["selected_group_ids_ordered_sha256"] != run_manifest[
        "selected_group_ids_ordered_sha256"
    ]:
        raise ValueError("analysis panel group_id hash differs from run manifest")
    if run_manifest["phase"] == "confirmatory":
        anchor_path = resolve_repo_path(run_manifest["confirmatory_anchor_path"])
        if file_sha256(anchor_path) != run_manifest["confirmatory_anchor_sha256"]:
            raise ValueError("live confirmatory anchor changed before analysis")
        if file_sha256(snapshots / "freeze.json") != run_manifest["freeze_sha256"]:
            raise ValueError("freeze snapshot mismatch before analysis")
        if load_json(snapshots / "freeze.json")["runtime_dependency_contract"] != current_runtime:
            raise ValueError("analysis runtime differs from confirmatory freeze")
    contract = load_json(run_dir / "spec_snapshots" / "contract.json")
    run_config = load_json(run_dir / "spec_snapshots" / "config.json")
    if run_manifest.get("language_model_contract") != contract.get("language_model"):
        raise ValueError("run language-model contract differs from frozen contract")
    lm_fingerprints = run_manifest.get("language_model_fingerprints") or {}
    if (
        int(lm_fingerprints.get("n_train_rows") or 0)
        != int(run_manifest["counts"]["n_train_rows"])
        or int(lm_fingerprints.get("n_train_charts") or 0)
        != int(run_manifest["counts"]["n_train_charts"])
        or len(str(lm_fingerprints.get("ordered_training_input_sha256") or "")) != 64
        or len(str(lm_fingerprints.get("model_state_sha256") or "")) != 64
    ):
        raise ValueError("language-model fingerprints are missing or inconsistent")
    if run_config["dataset"]["source"] == "huggingface" and int(
        lm_fingerprints.get("n_train_unique_group_ids") or 0
    ) != int(contract["language_model"]["expected_canonical_groups"]):
        raise ValueError("full-train unique group count drift")
    dataset_fingerprints = run_manifest.get("datasets_fingerprints") or {}
    required_dataset_fingerprints = (
        {"train", "test"}
        if run_config["dataset"]["source"] == "huggingface"
        else {"synthetic_reference", "synthetic"}
    )
    if not required_dataset_fingerprints.issubset(dataset_fingerprints):
        raise ValueError("pinned datasets fingerprints are incomplete")
    if run_config["dataset"]["source"] == "huggingface":
        split_manifest_path = resolve_repo_path(run_config["clean_split_manifest_path"])
        if (
            run_manifest.get("clean_split_manifest_path")
            != run_config["clean_split_manifest_path"]
            or run_manifest.get("clean_split_manifest_sha256")
            != file_sha256(split_manifest_path)
        ):
            raise ValueError("analysis clean split manifest path/hash mismatch")
    context_rows = _read_jsonl(table_dir / "sampling_context_songs.jsonl")
    source_scan_rows = _read_jsonl(table_dir / "source_scan.jsonl")
    if int(table_manifest["row_counts"].get("source_scan.jsonl", -1)) != len(
        source_scan_rows
    ):
        raise ValueError("analysis source-scan row count differs from table manifest")
    _validate_analysis_source_scan(
        source_scan_rows, context_rows, run_manifest, run_config
    )
    selected_context = [
        row
        for row in context_rows
        if run_config["dataset"]["canonical_song_index_start_inclusive"]
        <= int(row["canonical_song_index"])
        < run_config["dataset"]["canonical_song_index_stop_exclusive"]
    ]
    if [int(row["group_id"]) for row in selected_context] != run_manifest[
        "selected_group_ids_ordered"
    ]:
        raise ValueError("analysis context does not reproduce selected group_id order")
    if any(
        row.get("canonical") is not True
        or row.get("sid") != f"group_{int(row['group_id'])}"
        for row in context_rows
    ):
        raise ValueError("analysis context contains noncanonical/non-group SID rows")
    if (
        len({int(row["group_id"]) for row in context_rows}) != len(context_rows)
        or len({str(row["audio_sha256"]) for row in context_rows}) != len(context_rows)
    ):
        raise ValueError("analysis context group_id/audio_sha256 values are not unique")
    provenance_checks = {
        "table_hashes": True,
        "run_manifest_hash": True,
        "raw_evidence_hashes": True,
        "cartesian_151_per_chart": True,
        "spec_snapshots": True,
        "runtime_contract": True,
        "source_snapshot": True,
        "language_model_contract_and_fingerprints": True,
        "dataset_fingerprints": True,
        "selected_panel": True,
        "source_scan_and_alias_filter": True,
    }
    if run_manifest["phase"] == "confirmatory":
        if [int(row["canonical_song_index"]) for row in context_rows] != list(range(120)):
            raise ValueError("analysis confirmatory canonical context is not indices 0..119")
        if [int(row["canonical_song_index"]) for row in selected_context] != list(
            range(40, 120)
        ):
            raise ValueError("analysis confirmatory canonical panel is not indices 40..119")
        if any(row["panel_group"] != "confirmatory" for row in selected_context):
            raise ValueError("analysis confirmatory panel group label drift")
        if (
            run_manifest["counts"]["scanned_source_rows"] != 140
            or run_manifest["counts"]["filtered_alias_rows"] != 20
            or run_manifest["counts"]["canonical_context_rows"] != 120
        ):
            raise ValueError("analysis confirmatory raw/canonical/alias counts drift")
        split_manifest = load_json(
            resolve_repo_path(run_config["clean_split_manifest_path"])
        )["manifest"]
        expected_test_set = {
            (int(entry["group_id"]), str(entry["audio_sha256"]))
            for entry in split_manifest.values()
            if entry.get("split") == "test" and bool(entry["canonical"])
        }
        observed_test_set = {
            (int(row["group_id"]), str(row["audio_sha256"])) for row in context_rows
        }
        if observed_test_set != expected_test_set:
            raise ValueError("analysis context differs from clean split canonical test set")
        frozen = load_json(snapshots / "freeze.json")
        if frozen["source_snapshot"] != run_manifest["code_version"]:
            raise ValueError("analysis run source snapshot differs from freeze")
        anchor = load_json(snapshots / "confirmatory_anchor.json")
        if (
            anchor.get("primary_run_id") != run_manifest["run_id"]
            or anchor.get("freeze_sha256") != run_manifest["freeze_sha256"]
            or anchor.get("code_bundle_sha256")
            != run_manifest["code_version"]["code_bundle_sha256"]
        ):
            raise ValueError("analysis anchor binding differs from run/freeze/code")
        provenance_checks.update(
            {
                "confirmatory_canonical_indices_40_120": True,
                "clean_split_test_set_equality": True,
                "freeze_snapshot": True,
                "unique_run_anchor": True,
            }
        )
    elif run_manifest["phase"] == "smoke_development":
        if [int(row["canonical_song_index"]) for row in context_rows] != [0, 1]:
            raise ValueError("analysis development smoke must be canonical indices 0 and 1")
        split_manifest = load_json(
            resolve_repo_path(run_config["clean_split_manifest_path"])
        )["manifest"]
        expected_test_set = {
            (int(entry["group_id"]), str(entry["audio_sha256"]))
            for entry in split_manifest.values()
            if entry.get("split") == "test" and bool(entry["canonical"])
        }
        observed = {
            (int(row["group_id"]), str(row["audio_sha256"])) for row in context_rows
        }
        if not observed.issubset(expected_test_set):
            raise ValueError("development smoke is outside clean split canonical test set")
        provenance_checks["development_canonical_prefix_0_2"] = True
    rows = _read_jsonl(table_dir / "chart_dose_means.jsonl")
    observations = _read_jsonl(table_dir / "observations.jsonl")
    index = _index_means(rows)
    control_rows, control_passes, control_course_passes = _control_diagnostics(
        observations, contract
    )
    analysis_cfg = contract["analysis"]
    phase = run_manifest["phase"]
    nominal = float(analysis_cfg["nominal_confidence_level"])
    adjusted = float(analysis_cfg["familywise_confidence_level"])
    draws = int(analysis_cfg["bootstrap_draws"])
    minimum = int(analysis_cfg["minimum_confirmatory_group_clusters"])

    primary = []
    pair_exclusions = []
    provenance_valid = all(provenance_checks.values()) and run_manifest["status"] in {
        "completed",
        "completed_with_errors",
    }
    if phase == "confirmatory":
        provenance_valid = (
            provenance_valid
            and len(run_manifest["selected_group_ids_ordered"]) == 80
            and len(set(run_manifest["selected_group_ids_ordered"])) == 80
            and len(set(run_manifest["selected_audio_sha256_ordered"])) == 80
            and panels["primary_panel"]["id"]
            == "confirmatory_canonical_40_120_v1"
        )
    for pair in contract["primary_pairs"]:
        group_values, exclusions, complete_courses = _pair_group_values(
            index, pair, contract, control_course_passes
        )
        pair_exclusions.extend(exclusions)
        summary = _cluster_summary(
            group_values,
            draws=draws,
            seed=_pair_seed(int(analysis_cfg["bootstrap_seed"]), pair["pair_id"]),
            levels=(nominal, adjusted),
        )
        required_controls = [
            (control_id, pair["metric"])
            for control_id, control in contract["controls"].items()
            if pair["family"] in control["scoped_families"]
            and pair["metric"] in control["metrics"]
        ]
        controls_pass = all(control_passes.get(key, False) for key in required_controls)
        if not provenance_valid:
            verdict = "INVALID"
        elif phase != "confirmatory":
            verdict = "SMOKE_ONLY"
        elif not controls_pass or summary["n_group_clusters"] < minimum:
            verdict = "INCONCLUSIVE"
        else:
            interval = summary["confidence_intervals"][f"{adjusted:.6f}"]
            dose_lo, dose_hi = interval["dose_statistic"]
            contrast_lo, contrast_hi = interval["target_minus_sham"]
            if dose_hi < 0.0 and contrast_hi < 0.0:
                verdict = "SURVIVES"
            elif dose_lo >= 0.0 or contrast_lo >= 0.0:
                verdict = "REFUTED"
            else:
                verdict = "INCONCLUSIVE"
        primary.append(
            {
                **pair,
                **summary,
                "descriptive_dose_response": _dose_descriptives(
                    index,
                    complete_courses,
                    probe=pair["probe"],
                    metric=pair["metric"],
                    direction=pair["expected_raw_direction"],
                    draws=draws,
                    seed=_pair_seed(
                        int(analysis_cfg["bootstrap_seed"]), pair["pair_id"] + "|doses"
                    ),
                    confidence_level=nominal,
                ),
                "descriptive_seed_robustness": _seed_robustness(
                    observations, complete_courses, pair
                ),
                "required_controls": [f"{a}|{b}" for a, b in required_controls],
                "controls_pass": controls_pass,
                "minimum_group_clusters": minimum if phase == "confirmatory" else None,
                "n_complete_courses": len(complete_courses),
                "verdict": verdict,
            }
        )

    by_pair = {row["pair_id"]: row for row in primary}
    co_primary = []
    for claim in contract["co_primary_claims"]:
        verdicts = [by_pair[pair_id]["verdict"] for pair_id in claim["pair_ids"]]
        if any(v == "INVALID" for v in verdicts):
            verdict = "INVALID"
        elif phase != "confirmatory":
            verdict = "SMOKE_ONLY"
        elif all(v == "SURVIVES" for v in verdicts):
            verdict = "SURVIVES"
        elif any(v == "REFUTED" for v in verdicts):
            verdict = "REFUTED"
        else:
            verdict = "INCONCLUSIVE"
        co_primary.append({**claim, "pair_verdicts": verdicts, "verdict": verdict})

    secondary = []
    for item in contract["secondary_metrics"]:
        pair = {
            "pair_id": item["analysis_id"],
            "probe": item["probe"],
            "metric": item["metric"],
            "expected_raw_direction": "decrease",
            "scoped_sham": "official",
            "scoped_sham_dose_index": 0,
        }
        values, exclusions, secondary_complete_courses = _pair_group_values(
            index, pair, contract, control_course_passes
        )
        pair_exclusions.extend(exclusions)
        summary = _cluster_summary(
            values,
            draws=draws,
            seed=_pair_seed(int(analysis_cfg["bootstrap_seed"]), item["analysis_id"]),
            levels=(nominal,),
        )
        raw_interpretation = (
            "two-sided calibrated adequacy; movement in either raw direction is ambiguous"
            if item["metric"] == "pattern_ic_adequacy_score"
            else "raw NLL: lower means greater LM likelihood, not higher chart quality"
        )
        secondary.append(
            {
                **item,
                "raw_dose_statistic_mean": summary["dose_statistic_mean"],
                "raw_max_minus_clean_mean": summary["target_minus_sham_mean"],
                "confidence_intervals": summary["confidence_intervals"],
                "n_group_clusters": summary["n_group_clusters"],
                "group_ids": summary["group_ids"],
                "raw_dose_response": _dose_descriptives(
                    index,
                    secondary_complete_courses,
                    probe=item["probe"],
                    metric=item["metric"],
                    direction=None,
                    draws=draws,
                    seed=_pair_seed(
                        int(analysis_cfg["bootstrap_seed"]), item["analysis_id"] + "|doses"
                    ),
                    confidence_level=nominal,
                ),
                "raw_interpretation": raw_interpretation,
                "orientation": "none",
                "verdict": "SECONDARY_NO_DIRECTIONAL_GATE",
            }
        )

    context_summary = _summarize_context(context_rows)
    output = resolve_repo_path(output_dir) if output_dir else table_dir / "reports"
    output.mkdir(parents=False, exist_ok=False)
    write_json_exclusive(output / "primary_results.json", primary)
    write_json_exclusive(output / "co_primary_results.json", co_primary)
    write_json_exclusive(output / "secondary_results.json", secondary)
    write_json_exclusive(output / "control_diagnostics.json", control_rows)
    write_json_exclusive(output / "pair_exclusions.json", pair_exclusions)
    _write_csv(output / "sampling_context_comparability.csv", context_summary)

    pair_exclusion_counts = defaultdict(int)
    for exclusion in pair_exclusions:
        pair_exclusion_counts[exclusion["pair_id"]] += 1
    control_by_key = {
        f"{row['control']}|{row['metric']}": row for row in control_rows
    }
    claim_lines = [
        "# Claim–evidence map",
        "",
        f"- Experiment: `experiments/confirmatory_holdout_v1/SPEC.md`",
        f"- Phase: `{phase}`",
        f"- Spec snapshot hash: `{run_manifest['spec_snapshot_hashes']['SPEC.md']}`",
        f"- Config snapshot hash: `{run_manifest['spec_snapshot_hashes']['config.json']}`",
        f"- Contract snapshot hash: `{run_manifest['spec_snapshot_hashes']['contract.json']}`",
        f"- Freeze hash: `{run_manifest.get('freeze_sha256')}`",
        f"- Confirmatory anchor hash: `{run_manifest.get('confirmatory_anchor_sha256')}`",
        f"- Code-bundle hash: `{run_manifest['code_version']['code_bundle_sha256']}`",
        f"- Runtime: `{stable_json(run_manifest['runtime_environment'])}`",
        f"- Language-model contract: `{stable_json(run_manifest['language_model_contract'])}`",
        f"- Language-model fingerprints: `{stable_json(run_manifest['language_model_fingerprints'])}`",
        f"- Raw run manifest: `{run_manifest_path.relative_to(REPO_ROOT)}`",
        f"- Raw execution command: `{stable_json(run_manifest['execution_command'])}`",
        f"- Raw evidence hashes: `{stable_json(run_manifest['raw_hashes'])}`",
        f"- Canonical manifest: `{table_manifest_path.relative_to(REPO_ROOT)}`",
        f"- Transformation command: `{stable_json(table_manifest['transformation_command'])}`",
        f"- Canonical table hashes: `{stable_json(table_manifest['table_hashes'])}`",
        f"- Analysis command: `{stable_json(invocation())}`",
        f"- Dataset revision: `{run_manifest['dataset_revision']}`",
        f"- Clean split manifest hash: `{run_manifest.get('clean_split_manifest_sha256')}`",
        f"- Datasets fingerprints: `{stable_json(run_manifest['datasets_fingerprints'])}`",
        f"- Raw/canonical/alias scan counts: `{run_manifest['counts']['scanned_source_rows']}` / "
        f"`{run_manifest['counts']['canonical_context_rows']}` / "
        f"`{run_manifest['counts']['filtered_alias_rows']}`",
        f"- Selected group_id hash: `{run_manifest['selected_group_ids_ordered_sha256']}`",
        f"- Selected audio_sha256 hash: `{run_manifest['selected_audio_sha256_ordered_sha256']}`",
        f"- Selected panel fingerprint: `{run_manifest['selected_panel_fingerprint_sha256']}`",
        "",
        "Sampling-context comparisons are descriptive only and cannot revise hypotheses.",
        "",
    ]
    for row in primary:
        nominal_ci = row["confidence_intervals"].get(f"{nominal:.6f}")
        adjusted_ci = row["confidence_intervals"].get(f"{adjusted:.6f}")
        required_control_details = [
            control_by_key.get(key) for key in row["required_controls"]
        ]
        claim_lines.extend(
            [
                f"## {row['pair_id']}",
                "",
                f"- Probe / metric: `{row['probe']}` / `{row['metric']}`",
                f"- Scoped sham: `{row['scoped_sham']}` dose `{row['scoped_sham_dose_index']}`",
                f"- group_id clusters: {row['n_group_clusters']}",
                f"- Complete courses: {row['n_complete_courses']}",
                f"- Mean within-chart dose statistic: `{row['dose_statistic_mean']}`",
                f"- Mean target-minus-scoped-sham contrast: `{row['target_minus_sham_mean']}`",
                f"- Nominal 95% intervals: `{stable_json(nominal_ci)}`",
                f"- Family-wise 99.5% intervals: `{stable_json(adjusted_ci)}`",
                f"- Required controls / pass: `{stable_json(required_control_details)}` / `{row['controls_pass']}`",
                f"- Pair exclusions: {pair_exclusion_counts[row['pair_id']]}; rule: "
                "official + all target doses/replicates + fixed sham + scoped controls must be complete",
                f"- Verdict: **{row['verdict']}**",
                "- Evidence: `primary_results.json`, `control_diagnostics.json`, "
                "`../chart_dose_means.jsonl`, `../source_scan.jsonl`, "
                "`../../records.jsonl`",
                "- Exclusions: `pair_exclusions.json`",
                "",
            ]
        )
    for claim in co_primary:
        claim_lines.extend(
            [
                f"## {claim['claim_id']}",
                "",
                f"- Combination rule: `{claim['combination_rule']}`",
                f"- Pair verdicts: `{stable_json(dict(zip(claim['pair_ids'], claim['pair_verdicts'])))}`",
                f"- Co-primary verdict: **{claim['verdict']}**",
                "",
            ]
        )
    claim_lines.extend(
        [
            "## C5 interpretation boundary",
            "",
            "C5 is co-primary across repetition and surface variety using an all-pairs-survive rule. "
            "Pattern-IC and raw pattern NLL are secondary ambiguity/diagnostic outputs with no directional gate.",
            "",
            "## Limitations",
            "",
            "- Controlled corruption establishes sensitivity to the frozen edits, not player or expert criterion validity.",
            "- Authored timing metadata is an external timing map, not direct audio-onset annotation.",
            "- Courses are collapsed within song; remaining author-level dependence is not modeled.",
            "- Deterministic probes can produce identical five-seed replicates and never gain artificial sample size.",
            "- Sampling-context comparisons are descriptive and cannot modify the frozen hypotheses or panel.",
            "",
        ]
    )
    claim_path = output / "claim_evidence.md"
    with claim_path.open("x", encoding="utf-8") as f:
        f.write("\n".join(claim_lines))

    output_names = [
        "primary_results.json",
        "co_primary_results.json",
        "secondary_results.json",
        "control_diagnostics.json",
        "pair_exclusions.json",
        "sampling_context_comparability.csv",
        "claim_evidence.md",
    ]
    analysis_manifest = {
        "schema_version": "chartgeneval.confirmatory_analysis_manifest.v1",
        "generated_at_utc": utc_now(),
        "analysis_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "analysis_command": invocation(),
        "phase": phase,
        "input_table_manifest": str(table_manifest_path.relative_to(REPO_ROOT)),
        "input_table_manifest_sha256": file_sha256(table_manifest_path),
        "input_table_hashes": table_manifest["table_hashes"],
        "source_spec_snapshot_hashes": run_manifest["spec_snapshot_hashes"],
        "source_freeze_sha256": run_manifest.get("freeze_sha256"),
        "source_confirmatory_anchor_sha256": run_manifest.get(
            "confirmatory_anchor_sha256"
        ),
        "source_code_bundle_sha256": run_manifest["code_version"][
            "code_bundle_sha256"
        ],
        "source_raw_hashes": run_manifest["raw_hashes"],
        "source_clean_split_manifest_sha256": run_manifest.get(
            "clean_split_manifest_sha256"
        ),
        "source_execution_command": run_manifest["execution_command"],
        "source_transformation_command": table_manifest["transformation_command"],
        "panel_id": (
            "confirmatory_canonical_40_120_v1"
            if phase == "confirmatory"
            else "smoke_only"
        ),
        "cluster_unit": "group_id",
        "replicate_reducer": analysis_cfg["replicate_reducer"],
        "course_reducer": analysis_cfg["course_reducer"],
        "bootstrap_draws": draws,
        "bootstrap_seed": analysis_cfg["bootstrap_seed"],
        "nominal_confidence_level": nominal,
        "familywise_confidence_level": adjusted,
        "provenance_valid": provenance_valid,
        "provenance_checks": provenance_checks,
        "runtime_dependency_contract": current_runtime,
        "output_hashes": {name: file_sha256(output / name) for name in output_names},
        "notes": "C5 pattern-IC is secondary; sampling-context table is descriptive only.",
    }
    write_json_exclusive(output / "MANIFEST.json", analysis_manifest)
    print(stable_json({"report_dir": str(output), "phase": phase, "n_pairs": len(primary)}))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--table-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    analyze(args.table_dir, args.output_dir)


if __name__ == "__main__":
    main()
