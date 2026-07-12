#!/usr/bin/env python3
"""Deterministically normalize append-only raw evidence into canonical tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from contract import (
    REPO_ROOT,
    derive_rng_seed,
    file_sha256,
    invocation,
    load_json,
    resolve_repo_path,
    runtime_dependency_contract,
    stable_json,
    sha256_text,
    utc_now,
    verify_source_snapshot_files,
    write_json_exclusive,
)


def _validate_cartesian(raw: list[dict], context_rows: list[dict], manifest: dict, contract: dict, retained: list[str]) -> None:
    selected = set(int(x) for x in manifest["selected_group_ids_ordered"])
    mother = {}
    for song in context_rows:
        group_id = int(song["group_id"])
        if group_id not in selected:
            continue
        for chart in song["charts"]:
            key = (group_id, chart["course"])
            if key in mother:
                raise ValueError(f"duplicate selected sampling-context chart: {key}")
            mother[key] = {
                "sid": f"group_{group_id}",
                "source_row_index": int(song["source_row_index"]),
                "stored_split_row_id": song["stored_split_row_id"],
                "manifest_source_row_id": song.get("manifest_source_row_id"),
                "canonical_song_index": int(song["canonical_song_index"]),
                "group_id": group_id,
                "audio_sha256": song["audio_sha256"],
                "input_chart_sha256": chart["chart_input_sha256"],
                "events_sha256": chart["events_sha256"],
                "bpm": chart["bpm"],
                "level": chart["level"],
            }
    official_charts = {
        (int(row.get("group_id")), row.get("course"))
        for row in raw
        if row.get("condition") == "official"
    }
    if official_charts != set(mother):
        raise ValueError("official chart rows do not equal the sampling-context mother set")

    expected_cells = set()
    expected_cells.add(("official", 0, 0))
    for control in ("CTRL_identity", "CTRL_joint_time_origin_shift", "CTRL_color_bijection"):
        expected_cells.update((control, 0, replicate) for replicate in range(1, 6))
    for probe in contract["probes"]:
        expected_cells.update(
            (probe, dose, replicate) for dose in (1, 2, 3) for replicate in range(1, 6)
        )
    if len(expected_cells) != 151:
        raise ValueError("internal Cartesian contract is not 151 cells")

    observed_by_chart = defaultdict(set)
    for row in raw:
        if row.get("experiment_id") != manifest["experiment_id"]:
            raise ValueError("raw experiment_id mismatch")
        if row.get("run_id") != manifest["run_id"] or row.get("phase") != manifest["phase"]:
            raise ValueError("raw run_id/phase mismatch")
        chart_key = (int(row.get("group_id")), row.get("course"))
        if chart_key not in mother:
            raise ValueError(f"raw row outside selected chart mother set: {chart_key}")
        chart = mother[chart_key]
        if (
            row.get("sid") != chart["sid"]
            or int(row.get("source_row_index")) != chart["source_row_index"]
            or row.get("stored_split_row_id") != chart["stored_split_row_id"]
            or row.get("manifest_source_row_id")
            != chart["manifest_source_row_id"]
            or int(row.get("canonical_song_index")) != chart["canonical_song_index"]
            or row.get("audio_sha256") != chart["audio_sha256"]
            or row.get("canonical") is not True
        ):
            raise ValueError("raw canonical/group/source identity mismatch")
        if row.get("input_chart_sha256") != chart["input_chart_sha256"]:
            raise ValueError("raw input chart hash mismatch")
        condition = row.get("condition")
        dose = int(row.get("dose_index"))
        replicate = int(row.get("replicate_index"))
        cell = (condition, dose, replicate)
        if cell not in expected_cells:
            raise ValueError(f"unexpected chart-condition-dose-replicate cell: {cell}")
        if cell in observed_by_chart[chart_key]:
            raise ValueError(f"duplicate Cartesian cell for {chart_key}: {cell}")
        observed_by_chart[chart_key].add(cell)
        expected_task = f"{row['sid']}|{row['course']}|{condition}|d{dose}|r{replicate}"
        if row.get("task_id") != expected_task:
            raise ValueError("raw task_id is inconsistent with row keys")

        if condition == "official":
            expected_kind, expected_value = "official", 0
            if row.get("replicate_base_seed") is not None or row.get("rng_seed") is not None:
                raise ValueError("official row unexpectedly has a seed")
            if row.get("variant_events_sha256") != chart["events_sha256"]:
                raise ValueError("official variant events differ from sampling-context input")
            if row.get("corruption_noop"):
                raise ValueError("official baseline cannot be marked corruption_noop")
        elif condition.startswith("CTRL_"):
            expected_kind, expected_value = "invariant_control", 0
        else:
            expected_kind = "corruption"
            expected_value = contract["probes"][condition]["dose_values"][dose - 1]
            if row.get("variant_events_sha256") == chart["events_sha256"] and not row.get(
                "corruption_noop"
            ):
                raise ValueError("semantic no-op corruption was not marked no-op")
        if row.get("condition_kind") != expected_kind or float(row.get("dose_value")) != float(
            expected_value
        ):
            raise ValueError("condition kind or dose value drift")
        if replicate:
            expected_base_seed = contract["replicate_base_seeds"][replicate - 1]
            expected_rng_seed = derive_rng_seed(
                expected_base_seed, row["sid"], row["course"], condition, dose
            )
            if row.get("replicate_base_seed") != expected_base_seed or row.get(
                "rng_seed"
            ) != expected_rng_seed:
                raise ValueError("replicate seed drift")

        status = row.get("status")
        metrics = row.get("metrics")
        if status == "success":
            if not isinstance(metrics, dict) or set(metrics) != set(retained):
                raise ValueError("successful raw record metric key set drift")
            if row.get("error") is not None:
                raise ValueError("successful raw record unexpectedly carries an error")
        elif status == "error":
            if metrics != {} or not isinstance(row.get("error"), dict):
                raise ValueError("error raw record must have empty metrics and structured error")
        else:
            raise ValueError(f"unknown raw status: {status!r}")
    for chart_key, observed in observed_by_chart.items():
        if observed != expected_cells:
            missing = sorted(expected_cells - observed)
            extra = sorted(observed - expected_cells)
            raise ValueError(f"incomplete Cartesian matrix for {chart_key}: missing={missing}, extra={extra}")
    if set(observed_by_chart) != set(mother):
        raise ValueError("Cartesian chart set differs from sampling-context mother set")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _write_jsonl_exclusive(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(stable_json(row))
            f.write("\n")


def _validate_source_scan(
    scan_rows: list[dict],
    context_rows: list[dict],
    manifest: dict,
    dataset: dict,
) -> list[dict]:
    """Validate alias filtering and the canonical index in raw stored order."""
    if not scan_rows:
        raise ValueError("raw source scan is empty")
    split = str(dataset["split"])
    canonical_rows = []
    alias_rows = []
    for expected_index, row in enumerate(scan_rows):
        if row.get("schema_version") != "chartgeneval.source_scan_raw.v1":
            raise ValueError("unsupported source-scan schema")
        if int(row.get("source_row_index")) != expected_index:
            raise ValueError("source scan is not a zero-based stored-order prefix")
        if dataset["source"] == "huggingface" and row.get(
            "stored_split_row_id"
        ) != f"{split}_row_{expected_index:05d}":
            raise ValueError("stored split row ID/index mismatch")
        if row.get("canonical") is True:
            if row.get("action") != "canonical_context":
                raise ValueError("canonical source row has non-context action")
            canonical_rows.append(row)
        elif row.get("canonical") is False:
            if (
                row.get("action") != "filtered_alias"
                or row.get("canonical_song_index") is not None
                or row.get("manifest_source_row_id") is not None
            ):
                raise ValueError("alias source row leaked into canonical metadata")
            alias_rows.append(row)
        else:
            raise ValueError("source scan canonical flag is not a JSON boolean")
    if [int(row["canonical_song_index"]) for row in canonical_rows] != list(
        range(len(canonical_rows))
    ):
        raise ValueError("canonical index is not based on filtered stored-row order")
    if len({int(row["group_id"]) for row in canonical_rows}) != len(canonical_rows):
        raise ValueError("source scan has duplicate canonical group_id values")
    if len({str(row["audio_sha256"]) for row in canonical_rows}) != len(
        canonical_rows
    ):
        raise ValueError("source scan has duplicate canonical audio_sha256 values")

    context_start = int(dataset["sampling_context_canonical_start_inclusive"])
    context_stop = int(dataset["sampling_context_canonical_stop_exclusive"])
    scanned_context = [
        row
        for row in canonical_rows
        if context_start <= int(row["canonical_song_index"]) < context_stop
    ]
    scan_identity = [
        (
            int(row["source_row_index"]),
            row["stored_split_row_id"],
            row.get("manifest_source_row_id"),
            int(row["canonical_song_index"]),
            int(row["group_id"]),
            str(row["audio_sha256"]),
        )
        for row in scanned_context
    ]
    context_identity = [
        (
            int(row["source_row_index"]),
            row["stored_split_row_id"],
            row.get("manifest_source_row_id"),
            int(row["canonical_song_index"]),
            int(row["group_id"]),
            str(row["audio_sha256"]),
        )
        for row in context_rows
    ]
    if context_identity != scan_identity:
        raise ValueError("sampling context is not the canonical-only source-scan projection")
    observed_counts = {
        "scanned_source_rows": len(scan_rows),
        "canonical_context_rows": len(canonical_rows),
        "filtered_alias_rows": len(alias_rows),
    }
    for key, value in observed_counts.items():
        if int(manifest["counts"].get(key, -1)) != value:
            raise ValueError(f"source scan count differs from run manifest: {key}")
    if dataset["scan_full_split"]:
        expected = (
            int(dataset["expected_source_rows"]),
            int(dataset["expected_canonical_songs"]),
            int(dataset["expected_filtered_alias_rows"]),
        )
        observed = (len(scan_rows), len(canonical_rows), len(alias_rows))
        if observed != expected:
            raise ValueError(
                f"raw/canonical/alias source counts drift: observed={observed}, expected={expected}"
            )
    elif (
        len(canonical_rows) != context_stop
        or scan_rows[-1].get("canonical") is not True
    ):
        raise ValueError("development scan did not stop immediately at canonical prefix")
    return canonical_rows


def _record_sort_key(row: dict) -> tuple:
    return (
        int(row["canonical_song_index"]),
        int(row["group_id"]),
        row["course"],
        row["condition"],
        int(row["dose_index"]),
        int(row["replicate_index"]),
    )


def _canonical_observation(row: dict, retained: list[str]) -> dict:
    metrics = row.get("metrics") or {}
    return {
        "schema_version": "chartgeneval.confirmatory_observation.v1",
        "experiment_id": row["experiment_id"],
        "phase": row["phase"],
        "run_id": row["run_id"],
        "attempt_id": row["attempt_id"],
        "task_id": row["task_id"],
        "sid": row["sid"],
        "source_row_index": row["source_row_index"],
        "stored_split_row_id": row["stored_split_row_id"],
        "manifest_source_row_id": row.get("manifest_source_row_id"),
        "canonical_song_index": row["canonical_song_index"],
        "group_id": row["group_id"],
        "canonical": row["canonical"],
        "audio_sha256": row["audio_sha256"],
        "course": row["course"],
        "level": row.get("level"),
        "bpm": row.get("bpm"),
        "input_chart_sha256": row["input_chart_sha256"],
        "variant_events_sha256": row.get("variant_events_sha256"),
        "variant_n_notes": row.get("variant_n_notes"),
        "condition": row["condition"],
        "condition_kind": row["condition_kind"],
        "dose_index": row["dose_index"],
        "dose_value": row["dose_value"],
        "replicate_index": row["replicate_index"],
        "replicate_base_seed": row.get("replicate_base_seed"),
        "rng_seed": row.get("rng_seed"),
        "status": row["status"],
        "corruption_noop": bool(row.get("corruption_noop")),
        "metrics": {metric: metrics.get(metric) for metric in retained},
        "error": row.get("error"),
    }


def _build_chart_dose_means(observations: list[dict], retained: list[str]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in observations:
        key = (
            row["group_id"],
            row["canonical_song_index"],
            row["course"],
            row["condition"],
            row["condition_kind"],
            row["dose_index"],
            stable_json(row["dose_value"]),
        )
        grouped[key].append(row)

    means = []
    exclusions = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda r: int(r["replicate_index"]))
        group_id, canonical_song_index, course, condition, kind, dose_index, _dose_json = key
        sid = f"group_{group_id}"
        expected = 1 if kind == "official" else 5
        if len(rows) != expected:
            raise ValueError(
                f"incomplete attempted replicate set for {sid}/{course}/{condition}/d{dose_index}: "
                f"observed={len(rows)}, expected={expected}"
            )
        expected_replicates = [0] if expected == 1 else [1, 2, 3, 4, 5]
        observed_replicates = [int(row["replicate_index"]) for row in rows]
        if observed_replicates != expected_replicates:
            raise ValueError(
                f"replicate index drift for {sid}/{course}/{condition}/d{dose_index}: "
                f"{observed_replicates}"
            )
        usable = [r for r in rows if r["status"] == "success" and not r["corruption_noop"]]
        metric_means = {}
        metric_valid_counts = {}
        for metric in retained:
            values = [r["metrics"].get(metric) for r in usable]
            values = [float(v) for v in values if v is not None]
            metric_valid_counts[metric] = len(values)
            metric_means[metric] = float(np.mean(values)) if len(values) == expected else None
        group_status = "valid" if len(usable) == expected else "invalid"
        means.append(
            {
                "schema_version": "chartgeneval.chart_dose_mean.v1",
                "sid": sid,
                "source_row_index": rows[0]["source_row_index"],
                "stored_split_row_id": rows[0]["stored_split_row_id"],
                "manifest_source_row_id": rows[0].get("manifest_source_row_id"),
                "canonical_song_index": canonical_song_index,
                "group_id": group_id,
                "canonical": True,
                "audio_sha256": rows[0]["audio_sha256"],
                "course": course,
                "input_chart_sha256": rows[0]["input_chart_sha256"],
                "condition": condition,
                "condition_kind": kind,
                "dose_index": dose_index,
                "dose_value": rows[0]["dose_value"],
                "attempted_replicates": len(rows),
                "successful_non_noop_replicates": len(usable),
                "status": group_status,
                "metrics": metric_means,
                "metric_valid_counts": metric_valid_counts,
            }
        )
        for row in rows:
            reason = None
            if row["status"] != "success":
                reason = f"raw_status_{row['status']}"
            elif row["corruption_noop"]:
                reason = "semantic_or_operator_noop"
            if reason:
                exclusions.append(
                    {
                        "schema_version": "chartgeneval.confirmatory_exclusion.v1",
                        "task_id": row["task_id"],
                        "sid": sid,
                        "group_id": group_id,
                        "course": course,
                        "condition": condition,
                        "dose_index": dose_index,
                        "replicate_index": row["replicate_index"],
                        "reason": reason,
                        "metric": None,
                        "error": row.get("error"),
                    }
                )
            if row["status"] == "success":
                for metric in retained:
                    if row["metrics"].get(metric) is None:
                        exclusions.append(
                            {
                                "schema_version": "chartgeneval.confirmatory_exclusion.v1",
                                "task_id": row["task_id"],
                                "sid": sid,
                                "group_id": group_id,
                                "course": course,
                                "condition": condition,
                                "dose_index": dose_index,
                                "replicate_index": row["replicate_index"],
                                "reason": "registered_metric_missing",
                                "metric": metric,
                                "error": None,
                            }
                        )
    exclusion_keys = [
        (row["task_id"], row["reason"], row.get("metric")) for row in exclusions
    ]
    if len(exclusion_keys) != len(set(exclusion_keys)):
        raise ValueError("duplicate exclusion stable key")
    return means, sorted(
        exclusions, key=lambda row: (row["task_id"], row["reason"], row.get("metric") or "")
    )


def transform(run_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    current_runtime = runtime_dependency_contract()
    run_dir = resolve_repo_path(run_dir)
    manifest_path = run_dir / "MANIFEST.json"
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "chartgeneval.confirmatory_run_manifest.v1":
        raise ValueError("unsupported raw run manifest")
    if manifest.get("status") == "failed":
        raise ValueError("cannot canonicalize a fatally failed run")
    verify_source_snapshot_files(manifest["code_version"])
    run_runtime = manifest.get("runtime_dependency_contract")
    if current_runtime != run_runtime:
        raise ValueError("transformation runtime differs from execution runtime contract")

    snapshots = run_dir / "spec_snapshots"
    for name, expected in manifest.get("spec_snapshot_hashes", {}).items():
        if file_sha256(snapshots / name) != expected:
            raise ValueError(f"spec snapshot hash mismatch: {name}")
    required_snapshots = {"SPEC.md", "config.json", "contract.json"}
    if manifest.get("phase") == "confirmatory":
        required_snapshots.update({"freeze.json", "confirmatory_anchor.json"})
        anchor_path = resolve_repo_path(manifest["confirmatory_anchor_path"])
        if file_sha256(anchor_path) != manifest.get("confirmatory_anchor_sha256"):
            raise ValueError("live confirmatory anchor hash mismatch")
        frozen = load_json(snapshots / "freeze.json")
        if frozen["source_snapshot"] != manifest["code_version"]:
            raise ValueError("run source snapshot differs from confirmatory freeze")
        if current_runtime != frozen["runtime_dependency_contract"]:
            raise ValueError("transformation runtime differs from confirmatory freeze")
        if file_sha256(snapshots / "SPEC.md") != frozen["spec_sha256"]:
            raise ValueError("SPEC snapshot does not match confirmatory freeze")
        if file_sha256(snapshots / "config.json") != frozen["config_sha256"]:
            raise ValueError("config snapshot does not match confirmatory freeze")
        if file_sha256(snapshots / "contract.json") != frozen["contract_sha256"]:
            raise ValueError("contract snapshot does not match confirmatory freeze")
        if file_sha256(snapshots / "freeze.json") != manifest.get("freeze_sha256"):
            raise ValueError("freeze snapshot does not match run manifest")
        if file_sha256(snapshots / "confirmatory_anchor.json") != manifest.get(
            "confirmatory_anchor_sha256"
        ):
            raise ValueError("anchor snapshot does not match run manifest")
        if load_json(snapshots / "confirmatory_anchor.json").get(
            "primary_run_id"
        ) != manifest.get("run_id"):
            raise ValueError("run_id differs from anchor primary_run_id")
    if not required_snapshots.issubset(manifest.get("spec_snapshot_hashes", {})):
        raise ValueError("required spec snapshots are absent from run manifest")

    records_path = run_dir / "records.jsonl"
    context_path = run_dir / "sampling_context.jsonl"
    source_scan_path = run_dir / "source_scan.jsonl"
    for path in (
        records_path,
        context_path,
        source_scan_path,
        run_dir / "manifest_events.jsonl",
    ):
        expected = manifest["raw_hashes"].get(path.name)
        if expected != file_sha256(path):
            raise ValueError(f"raw evidence hash mismatch: {path.name}")

    contract = load_json(run_dir / "spec_snapshots" / "contract.json")
    run_config = load_json(run_dir / "spec_snapshots" / "config.json")
    retained = sorted(
        {p["metric"] for p in contract["primary_pairs"]}
        | {m["metric"] for m in contract["secondary_metrics"]}
        | {
            metric
            for control in contract["controls"].values()
            for metric in control["metrics"]
        }
    )
    raw = _read_jsonl(records_path)
    context_rows = sorted(
        _read_jsonl(context_path), key=lambda row: int(row["canonical_song_index"])
    )
    source_scan_rows = _read_jsonl(source_scan_path)
    dataset = run_config["dataset"]
    _validate_source_scan(source_scan_rows, context_rows, manifest, dataset)
    selected_context = [
        row
        for row in context_rows
        if dataset["canonical_song_index_start_inclusive"]
        <= int(row["canonical_song_index"])
        < dataset["canonical_song_index_stop_exclusive"]
    ]
    if any(
        row.get("canonical") is not True
        or row.get("sid") != f"group_{int(row['group_id'])}"
        for row in context_rows
    ):
        raise ValueError("sampling context has noncanonical or non-group-derived SID rows")
    if len({int(row["group_id"]) for row in context_rows}) != len(context_rows):
        raise ValueError("sampling context group_id values are not unique")
    if len({str(row["audio_sha256"]) for row in context_rows}) != len(context_rows):
        raise ValueError("sampling context audio_sha256 values are not unique")
    selected_group_ids = [int(row["group_id"]) for row in selected_context]
    selected_audio = [str(row["audio_sha256"]) for row in selected_context]
    if selected_group_ids != manifest["selected_group_ids_ordered"]:
        raise ValueError("selected group_id order does not match canonical panel boundaries")
    if sha256_text(stable_json(selected_group_ids)) != manifest[
        "selected_group_ids_ordered_sha256"
    ]:
        raise ValueError("selected group_id ordered hash mismatch")
    if selected_audio != manifest["selected_audio_sha256_ordered"]:
        raise ValueError("selected audio_sha256 order mismatch")
    if sha256_text(stable_json(selected_audio)) != manifest[
        "selected_audio_sha256_ordered_sha256"
    ]:
        raise ValueError("selected audio_sha256 ordered hash mismatch")
    fingerprint_material = [
        {
            "source_row_index": int(row["source_row_index"]),
            "stored_split_row_id": row["stored_split_row_id"],
            "manifest_source_row_id": row.get("manifest_source_row_id"),
            "canonical_song_index": int(row["canonical_song_index"]),
            "group_id": int(row["group_id"]),
            "audio_sha256": row["audio_sha256"],
            "chart_input_sha256": [chart["chart_input_sha256"] for chart in row["charts"]],
        }
        for row in selected_context
    ]
    if sha256_text(stable_json(fingerprint_material)) != manifest[
        "selected_panel_fingerprint_sha256"
    ]:
        raise ValueError("selected panel fingerprint mismatch")
    if run_config["dataset"]["source"] == "huggingface":
        split_manifest_path = resolve_repo_path(run_config["clean_split_manifest_path"])
        if (
            manifest.get("clean_split_manifest_path")
            != run_config["clean_split_manifest_path"]
            or manifest.get("clean_split_manifest_sha256")
            != file_sha256(split_manifest_path)
        ):
            raise ValueError("clean split manifest path/hash mismatch")
        dataset_fingerprints = manifest.get("datasets_fingerprints") or {}
        train_fp = dataset_fingerprints.get("train") or {}
        test_fp = dataset_fingerprints.get(run_config["dataset"]["split"]) or {}
        if (
            train_fp.get("kind") != "ordered_parsed_chart_content_v1"
            or train_fp.get("dataset_revision") != run_config["dataset"]["revision"]
            or train_fp.get("range") != "full_split"
            or len(str(train_fp.get("sha256") or "")) != 64
            or int(train_fp.get("n_source_rows") or 0)
            != int(manifest["counts"]["n_train_rows"])
            or int(train_fp.get("n_charts") or 0)
            != int(manifest["counts"]["n_train_charts"])
        ):
            raise ValueError("full-train content fingerprint is missing or inconsistent")
        context_material = [
            {
                "source_row_index": int(row["source_row_index"]),
                "stored_split_row_id": row["stored_split_row_id"],
                "manifest_source_row_id": row.get("manifest_source_row_id"),
                "canonical_song_index": int(row["canonical_song_index"]),
                "group_id": int(row["group_id"]),
                "audio_sha256": row["audio_sha256"],
                "chart_input_sha256": [
                    chart["chart_input_sha256"] for chart in row["charts"]
                ],
            }
            for row in context_rows
        ]
        if (
            test_fp.get("kind") != "ordered_canonical_chart_content_v1"
            or test_fp.get("dataset_revision") != run_config["dataset"]["revision"]
            or test_fp.get("range")
            != [
                run_config["dataset"]["sampling_context_canonical_start_inclusive"],
                run_config["dataset"]["sampling_context_canonical_stop_exclusive"],
            ]
            or int(test_fp.get("scanned_source_rows") or 0)
            != int(manifest["counts"]["scanned_source_rows"])
            or int(test_fp.get("filtered_alias_rows") or 0)
            != int(manifest["counts"]["filtered_alias_rows"])
            or int(test_fp.get("n_canonical_songs") or 0) != len(context_rows)
            or test_fp.get("source_scan_kind")
            != "ordered_raw_source_identity_v1"
            or test_fp.get("source_scan_sha256")
            != sha256_text(stable_json(source_scan_rows))
            or test_fp.get("sha256") != sha256_text(stable_json(context_material))
        ):
            raise ValueError("test sampling-context content fingerprint mismatch")
        split_payload = load_json(split_manifest_path)
        split_entries = [
            (manifest_source_row_id, entry)
            for manifest_source_row_id, entry in split_payload["manifest"].items()
            if entry.get("split") == "test"
        ]
        expected_canonical_set = {
            (int(entry["group_id"]), str(entry["audio_sha256"]))
            for _manifest_source_row_id, entry in split_entries
            if bool(entry["canonical"])
        }
        expected_identity_counter = Counter(
            (
                int(entry["group_id"]),
                bool(entry["canonical"]),
                str(entry["audio_sha256"]),
            )
            for _manifest_row_id, entry in split_entries
        )
        canonical_manifest_ids = {
            (int(entry["group_id"]), str(entry["audio_sha256"])): manifest_row_id
            for manifest_row_id, entry in split_entries
            if bool(entry["canonical"])
        }
        if len(canonical_manifest_ids) != len(expected_canonical_set):
            raise ValueError("clean split manifest has duplicate canonical identities")
        remaining = expected_identity_counter.copy()
        for row in source_scan_rows:
            identity = (
                int(row["group_id"]),
                bool(row["canonical"]),
                str(row["audio_sha256"]),
            )
            if remaining[identity] <= 0:
                raise ValueError("source-scan identity is absent/exhausted in split manifest")
            remaining[identity] -= 1
            if row["canonical"] is True:
                canonical_identity = (identity[0], identity[2])
                if row.get("manifest_source_row_id") != canonical_manifest_ids.get(
                    canonical_identity
                ):
                    raise ValueError("canonical source row lacks its true manifest source ID")
        if manifest["phase"] == "confirmatory" and any(remaining.values()):
            raise ValueError("confirmatory source scan does not exhaust split manifest rows")
        context_identity_set = {
            (int(row["group_id"]), str(row["audio_sha256"])) for row in context_rows
        }
        if manifest["phase"] == "confirmatory":
            if context_identity_set != expected_canonical_set:
                raise ValueError("120-song context differs from clean split canonical test set")
        elif not context_identity_set.issubset(expected_canonical_set):
            raise ValueError("development smoke context is outside canonical test set")
    if manifest["phase"] == "confirmatory":
        if [int(row["canonical_song_index"]) for row in context_rows] != list(range(120)):
            raise ValueError("confirmatory canonical context must be exact indices 0..119")
        if [int(row["canonical_song_index"]) for row in selected_context] != list(
            range(40, 120)
        ):
            raise ValueError("confirmatory selected canonical panel must be indices 40..119")
        if (
            len(selected_group_ids) != 80
            or len(set(selected_group_ids)) != 80
            or len(set(selected_audio)) != 80
        ):
            raise ValueError("confirmatory panel must have 80 unique group/audio identities")
        if (
            manifest["counts"]["scanned_source_rows"] != 140
            or manifest["counts"]["filtered_alias_rows"] != 20
            or manifest["counts"]["canonical_context_rows"] != 120
        ):
            raise ValueError("confirmatory raw/canonical/alias counts are not 140/120/20")
        for row in context_rows:
            expected_group = (
                "development"
                if int(row["canonical_song_index"]) < 40
                else "confirmatory"
            )
            if row.get("panel_group") != expected_group:
                raise ValueError("confirmatory sampling-context group label drift")
    elif manifest["phase"] == "smoke_development":
        if [int(row["canonical_song_index"]) for row in context_rows] != [0, 1]:
            raise ValueError("development smoke must contain canonical indices 0 and 1 only")
        if any(row["canonical"] is not True for row in context_rows):
            raise ValueError("development smoke contains a noncanonical row")
    _validate_cartesian(raw, context_rows, manifest, contract, retained)
    task_ids = [row.get("task_id") for row in raw]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("duplicate raw task_id")
    if len(raw) != manifest["counts"]["records"]:
        raise ValueError("raw record count does not match run manifest")
    observations = sorted(
        [_canonical_observation(row, retained) for row in raw], key=_record_sort_key
    )
    chart_means, exclusions = _build_chart_dose_means(observations, retained)
    if len(context_rows) != manifest["counts"]["sampling_context_rows"]:
        raise ValueError("sampling-context count does not match run manifest")
    output = resolve_repo_path(output_dir) if output_dir else run_dir / "tables"
    output.mkdir(parents=False, exist_ok=False)
    _write_jsonl_exclusive(output / "observations.jsonl", observations)
    _write_jsonl_exclusive(output / "chart_dose_means.jsonl", chart_means)
    _write_jsonl_exclusive(output / "exclusions.jsonl", exclusions)
    _write_jsonl_exclusive(output / "sampling_context_songs.jsonl", context_rows)
    _write_jsonl_exclusive(output / "source_scan.jsonl", source_scan_rows)
    panels = {
        "schema_version": "chartgeneval.confirmatory_panels.v1",
        "phase": manifest["phase"],
        "primary_panel": {
            "id": (
                "confirmatory_canonical_40_120_v1"
                if manifest["phase"] == "confirmatory"
                else "smoke_only"
            ),
            "selected_group_ids_ordered": manifest["selected_group_ids_ordered"],
            "selected_group_ids_ordered_sha256": manifest[
                "selected_group_ids_ordered_sha256"
            ],
            "selected_audio_sha256_ordered_sha256": manifest[
                "selected_audio_sha256_ordered_sha256"
            ],
            "cluster_unit": "group_id",
        },
        "sampling_context_groups": {
            group: [
                int(row["group_id"])
                for row in context_rows
                if row["panel_group"] == group
            ]
            for group in sorted({row["panel_group"] for row in context_rows})
        },
    }
    write_json_exclusive(output / "panels.json", panels)
    table_names = [
        "observations.jsonl",
        "chart_dose_means.jsonl",
        "exclusions.jsonl",
        "sampling_context_songs.jsonl",
        "source_scan.jsonl",
        "panels.json",
    ]
    table_manifest = {
        "schema_version": "chartgeneval.confirmatory_tables_manifest.v1",
        "generated_at_utc": utc_now(),
        "transformation_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "transformation_command": invocation(),
        "source_run_manifest": str(manifest_path.relative_to(REPO_ROOT)),
        "source_run_manifest_sha256": file_sha256(manifest_path),
        "source_raw_hashes": manifest["raw_hashes"],
        "source_spec_snapshot_hashes": manifest["spec_snapshot_hashes"],
        "runtime_dependency_contract": current_runtime,
        "cartesian_validation": {
            "expected_attempts_per_chart": 151,
            "validated": True,
            "mother_set": "sampling_context selected group_id/course/chart_input_sha256",
        },
        "schema_grains": {
            "observations.jsonl": "one row per attempted chart-condition-dose-replicate",
            "chart_dose_means.jsonl": "one row per chart-condition-dose after replicate reduction",
            "exclusions.jsonl": "one row per task-level reason or task-by-metric missingness reason; stable key=(task_id, reason, metric)",
            "sampling_context_songs.jsonl": "one row per canonical song after filtering raw stored-order rows",
            "source_scan.jsonl": "one row per scanned raw stored-order row, including aliases filtered before canonical indexing",
        },
        "row_counts": {
            "observations.jsonl": len(observations),
            "chart_dose_means.jsonl": len(chart_means),
            "exclusions.jsonl": len(exclusions),
            "sampling_context_songs.jsonl": len(context_rows),
            "source_scan.jsonl": len(source_scan_rows),
        },
        "table_hashes": {name: file_sha256(output / name) for name in table_names},
        "retained_metrics": retained,
        "missingness_rule": "metric mean is null unless every planned replicate is successful, non-noop, and nonmissing",
    }
    write_json_exclusive(output / "MANIFEST.json", table_manifest)
    print(stable_json({"table_dir": str(output), "row_counts": table_manifest["row_counts"]}))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    transform(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
