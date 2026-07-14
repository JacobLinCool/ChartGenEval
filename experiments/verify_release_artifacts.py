#!/usr/bin/env python3
"""Verify the released artifact bundle against its checksum manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from pathlib import PurePosixPath

from _coupling_feature_contract import (
    EXPERIMENT_ID,
    PANEL_ID,
    RECORD_SCHEMA,
    RUN_SCHEMA,
    SUMMARY_SCHEMA,
    load_feature_manifest,
)
from _dataset import DEFAULT_DATASET, DEFAULT_DATASET_REVISION
from summarize_suite_v2_development import (
    FIELDNAMES as SUITE_V2_SUMMARY_FIELDS,
    _load as _load_suite_v2_records,
    _validate_panel as _validate_suite_v2_panel,
    build_summaries as _build_suite_v2_summaries,
    render_summary_csv as _render_suite_v2_summary_csv,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "MANIFEST.json"
PROHIBITED_RECORD_KEYS = {"audio", "events", "hits", "notes"}
PROHIBITED_CONFIRMATORY_ROW_KEYS = {
    "manifest_source_row_id",
    "source_row_id",
    "stored_split_row_id",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATASET_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
CONFIRMATORY_PREFIX = "artifacts/confirmatory_holdout_v1/"
CONFIRMATORY_FILES = {
    "analysis_manifest.json",
    "anchor.json",
    "claim_evidence.md",
    "co_primary_results.json",
    "control_diagnostics.json",
    "freeze.json",
    "pair_exclusions.json",
    "primary_results.json",
    "run_manifest.json",
    "sampling_context_comparability.csv",
    "secondary_results.json",
    "table_manifest.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect(path: Path) -> tuple[int | None, set[str]]:
    if path.suffix == ".jsonl":
        rows = 0
        keys: set[str] = set()
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: record is not an object")
                rows += 1
                keys.update(row)
        return rows, keys
    if path.suffix == ".json":
        with path.open() as stream:
            value = json.load(stream)
        return (len(value), set()) if isinstance(value, list) else (None, set())
    if path.suffix == ".csv":
        with path.open(newline="") as stream:
            rows = sum(1 for _ in csv.DictReader(stream))
        return rows, set()
    return None, set()


def _load_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def _require_equal(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: {actual!r}; expected {expected!r}")


def _audit_confirmatory_payload(value, *, location: str) -> None:
    """Reject raw chart payloads and row-level source identifiers recursively."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if key in PROHIBITED_CONFIRMATORY_ROW_KEYS:
                raise ValueError(f"{child_location}: restricted row identifier")
            if key in PROHIBITED_RECORD_KEYS and isinstance(child, (dict, list)):
                raise ValueError(f"{child_location}: prohibited raw payload")
            _audit_confirmatory_payload(child, location=child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _audit_confirmatory_payload(child, location=f"{location}[{index}]")


def _verify_confirmatory_bundle(listed_paths: set[str]) -> None:
    """Verify the compact evidence bundle and its sealed hash chain."""

    expected_paths = {CONFIRMATORY_PREFIX + name for name in CONFIRMATORY_FILES}
    actual_paths = {path for path in listed_paths if path.startswith(CONFIRMATORY_PREFIX)}
    _require_equal(actual_paths, expected_paths, "confirmatory evidence files")

    bundle = ROOT / CONFIRMATORY_PREFIX
    freeze_path = bundle / "freeze.json"
    anchor_path = bundle / "anchor.json"
    run_path = bundle / "run_manifest.json"
    table_path = bundle / "table_manifest.json"
    analysis_path = bundle / "analysis_manifest.json"

    freeze = _load_json(freeze_path)
    anchor = _load_json(anchor_path)
    run = _load_json(run_path)
    table = _load_json(table_path)
    analysis = _load_json(analysis_path)
    for name in CONFIRMATORY_FILES:
        path = bundle / name
        if path.suffix == ".json":
            _audit_confirmatory_payload(_load_json(path), location=name)

    freeze_sha256 = _sha256(freeze_path)
    anchor_sha256 = _sha256(anchor_path)
    run_sha256 = _sha256(run_path)
    table_sha256 = _sha256(table_path)
    _require_equal(anchor["freeze_sha256"], freeze_sha256, "anchor -> freeze")
    _require_equal(run["freeze_sha256"], freeze_sha256, "run -> freeze")
    _require_equal(
        run["confirmatory_anchor_sha256"], anchor_sha256, "run -> anchor"
    )
    _require_equal(
        table["source_run_manifest_sha256"], run_sha256, "table -> run"
    )
    _require_equal(
        analysis["input_table_manifest_sha256"], table_sha256, "analysis -> table"
    )
    _require_equal(
        analysis["source_freeze_sha256"], freeze_sha256, "analysis -> freeze"
    )
    _require_equal(
        analysis["source_confirmatory_anchor_sha256"],
        anchor_sha256,
        "analysis -> anchor",
    )
    _require_equal(run["status"], "completed", "confirmatory run status")
    _require_equal(run["fatal_error"], None, "confirmatory fatal_error")
    _require_equal(analysis["provenance_valid"], True, "analysis provenance")
    if not all(analysis["provenance_checks"].values()):
        raise ValueError("analysis provenance_checks contains a failure")

    for filename, expected_sha256 in analysis["output_hashes"].items():
        _require_equal(
            _sha256(bundle / filename),
            expected_sha256,
            f"analysis output {filename}",
        )

    primary = _load_json(bundle / "primary_results.json")
    controls = _load_json(bundle / "control_diagnostics.json")
    _require_equal(len(primary), 10, "primary result count")
    if any(row.get("verdict") != "SURVIVES" for row in primary):
        raise ValueError("primary_results contains a non-SURVIVES verdict")
    _require_equal(len(controls), 32, "control diagnostic count")
    if any(row.get("passes") is not True for row in controls):
        raise ValueError("control_diagnostics contains a failed control")


def _verify_primary_summary_csv() -> None:
    """Check that the compact CSV is an exact projection of sealed JSON."""

    primary = _load_json(
        ROOT / CONFIRMATORY_PREFIX / "primary_results.json"
    )
    summary_path = ROOT / "artifacts/tables/confirmatory_primary_results.csv"
    with summary_path.open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    _require_equal(len(summary), len(primary), "confirmatory CSV row count")
    for result, row in zip(primary, summary, strict=True):
        interval = result["confidence_intervals"]["0.995000"]
        expected = {
            "pair_id": result["pair_id"],
            "probe": result["probe"],
            "metric": result["metric"],
            "n_groups": str(result["n_group_clusters"]),
            "n_courses": str(result["n_complete_courses"]),
            "dose_mean": str(result["dose_statistic_mean"]),
            "dose_ci_99_5_lower": str(interval["dose_statistic"][0]),
            "dose_ci_99_5_upper": str(interval["dose_statistic"][1]),
            "target_minus_sham_mean": str(result["target_minus_sham_mean"]),
            "target_minus_sham_ci_99_5_lower": str(
                interval["target_minus_sham"][0]
            ),
            "target_minus_sham_ci_99_5_upper": str(
                interval["target_minus_sham"][1]
            ),
            "controls_pass": str(result["controls_pass"]).lower(),
            "verdict": result["verdict"],
        }
        _require_equal(row, expected, f"confirmatory CSV row {result['pair_id']}")


def _verify_c5_dependency_disclosure(listed_paths: set[str]) -> None:
    """Recompute the released evidence for the post-freeze C5 dependency."""

    audit_relative = "artifacts/C5_DEPENDENCY_AUDIT.md"
    if audit_relative not in listed_paths:
        raise ValueError("release manifest omits the C5 dependency audit")
    audit_text = (ROOT / audit_relative).read_text()
    for required in (
        "nine nonredundant probe--measurement tests",
        "does not support independent repetition-versus-variety corroboration",
        "surface_structure_proxy_score",
    ):
        if required not in audit_text:
            raise ValueError(f"C5 dependency audit omits: {required!r}")

    development_path = ROOT / "artifacts/records/corruption_probes_development.jsonl"
    n_scores = 0
    n_c5_scores = 0
    max_raw_residual = 0.0
    max_score_delta = 0.0
    max_c5_score_delta = 0.0
    n_structure_proxies = 0
    max_structure_proxy_delta = 0.0
    with development_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            repeated_raw = row.get("repeat_4gram_rate")
            unique_raw = row.get("unique_4gram_rate")
            if repeated_raw is not None and unique_raw is not None:
                max_raw_residual = max(
                    max_raw_residual,
                    abs(float(repeated_raw) + float(unique_raw) - 1.0),
                )
            repeated_score = row.get("repetition_adequacy_score")
            variety_score = row.get("surface_variety_adequacy_score")
            if repeated_score is None or variety_score is None:
                continue
            delta = abs(float(repeated_score) - float(variety_score))
            n_scores += 1
            max_score_delta = max(max_score_delta, delta)
            if row.get("probe") == "C5_blandification":
                n_c5_scores += 1
                max_c5_score_delta = max(max_c5_score_delta, delta)
            proxy_inputs = [
                row.get("surface_variety_adequacy_score"),
                row.get("repetition_adequacy_score"),
                row.get("density_variation_adequacy_score"),
            ]
            proxy = row.get("surface_structure_proxy_score")
            if proxy is not None and all(value is not None for value in proxy_inputs):
                expected_proxy = (
                    0.0
                    if any(float(value) <= 0.0 for value in proxy_inputs)
                    else math.exp(
                        sum(math.log(float(value)) for value in proxy_inputs) / 3.0
                    )
                )
                n_structure_proxies += 1
                max_structure_proxy_delta = max(
                    max_structure_proxy_delta,
                    abs(float(proxy) - expected_proxy),
                )

    _require_equal(n_scores, 4760, "development dependency score count")
    _require_equal(n_c5_scores, 510, "C5 development dependency score count")
    _require_equal(max_raw_residual, 0.0, "development raw complement residual")
    _require_equal(
        max_score_delta,
        8.881784197001252e-16,
        "development dependency maximum score delta",
    )
    _require_equal(
        max_c5_score_delta,
        6.661338147750939e-16,
        "C5 development dependency maximum score delta",
    )
    _require_equal(n_structure_proxies, 4760, "surface structure proxy count")
    if not math.isclose(
        max_structure_proxy_delta,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "surface structure proxy geometric-mean residual: "
            f"{max_structure_proxy_delta!r}; expected at most 1e-15"
        )

    calibration = _load_json(
        ROOT / "src/chartgeneval/data/taiko_1000_parsed_clean_calibration_v2.json"
    )
    for course, course_data in calibration["courses"].items():
        bands = course_data["bands"]
        repeated = bands["repeat_4gram_rate"]
        unique = bands["unique_4gram_rate"]
        if (
            abs(float(repeated["p10"]) - (1.0 - float(unique["p90"]))) > 1e-12
            or abs(float(repeated["p90"]) - (1.0 - float(unique["p10"]))) > 1e-12
        ):
            raise ValueError(f"{course}: 4-gram calibration bands are not mirrored")

    primary = {
        row["pair_id"]: row
        for row in _load_json(ROOT / CONFIRMATORY_PREFIX / "primary_results.json")
    }
    repetition = primary["C5__repetition"]
    variety = primary["C5__surface_variety"]
    for key in ("dose_statistic_mean", "target_minus_sham_mean"):
        if abs(float(repetition[key]) - float(variety[key])) > 1e-12:
            raise ValueError(f"C5 confirmatory point estimates differ for {key}")


def _verify_suite_v2_development_evidence(listed_paths: set[str]) -> None:
    """Verify the complete traceability chain for proposed development metrics."""

    relatives = {
        "spec": "experiments/suite_v2_development/SPEC.md",
        "feature": "artifacts/suite_v2_development/coupling_gap_feature_manifest.json",
        "structure_run": "artifacts/suite_v2_development/structure_run_manifest.json",
        "coupling_run": "artifacts/suite_v2_development/coupling_gap_run_manifest.json",
        "summary_manifest": "artifacts/suite_v2_development/summary_manifest.json",
        "claim_map": "artifacts/suite_v2_development/claim_evidence.md",
        "coupling": "artifacts/records/corruption_coupling_gap_clean.jsonl",
        "structure": "artifacts/records/structure_v2_development.jsonl",
        "summary": "artifacts/tables/suite_v2_development_results.csv",
    }
    missing = set(relatives.values()) - listed_paths
    if missing:
        raise ValueError(f"release omits suite-v2 evidence: {sorted(missing)}")
    paths = {name: ROOT / relative for name, relative in relatives.items()}

    spec_sha256 = _sha256(paths["spec"])
    feature_payload, feature_rows = load_feature_manifest(
        paths["feature"],
        expected_dataset=DEFAULT_DATASET,
        expected_revision=DEFAULT_DATASET_REVISION,
    )
    _require_equal(feature_payload["spec_sha256"], spec_sha256, "feature spec hash")
    _require_equal(
        sum(int(row["mel_bytes"]) for row in feature_rows.values()),
        113633024,
        "cached mel byte total",
    )
    feature_sha256 = _sha256(paths["feature"])

    records = {
        "structure": _load_suite_v2_records(paths["structure"]),
        "coupling_gap": _load_suite_v2_records(paths["coupling"]),
    }
    panels = {}
    record_spec_hashes = {}
    for label, rows in records.items():
        panels[label], record_spec_hashes[label] = _validate_suite_v2_panel(rows, label)
    _require_equal(panels["structure"], panels["coupling_gap"], "suite-v2 panel")
    _require_equal(set(record_spec_hashes.values()), {spec_sha256}, "record spec hashes")

    group_maps = {}
    for label, rows in records.items():
        observed = defaultdict(set)
        for row in rows:
            observed[str(row["sid"])].add(int(row["group_id"]))
        if any(len(values) != 1 for values in observed.values()):
            raise ValueError(f"{label}: group_id varies within a sid")
        group_maps[label] = {sid: next(iter(values)) for sid, values in observed.items()}
    feature_groups = {sid: int(row["group_id"]) for sid, row in feature_rows.items()}
    _require_equal(group_maps["structure"], feature_groups, "structure/feature groups")
    _require_equal(group_maps["coupling_gap"], feature_groups, "coupling/feature groups")

    for row in records["coupling_gap"]:
        sid = str(row["sid"])
        _require_equal(
            row["feature_manifest_sha256"], feature_sha256, f"{sid}: feature hash"
        )
        _require_equal(
            row["audio_sha256"], feature_rows[sid]["audio_sha256"], f"{sid}: audio hash"
        )

    expected_refs = {
        "easy": 923.0,
        "normal": 924.0,
        "hard": 924.0,
        "oni": 924.0,
        "ura": 185.0,
    }
    observed_refs = defaultdict(set)
    for row in records["coupling_gap"]:
        observed_refs[str(row["course"])].add(float(row["manifold_n_ref"]))
    if any(len(values) != 1 for values in observed_refs.values()):
        raise ValueError("manifold reference count varies within a course")
    _require_equal(
        {course: next(iter(values)) for course, values in observed_refs.items()},
        expected_refs,
        "course-conditioned reference counts",
    )

    output_hashes = {
        "structure": _sha256(paths["structure"]),
        "coupling_gap": _sha256(paths["coupling"]),
    }
    for runner, manifest_name in (
        ("structure", "structure_run"),
        ("coupling_gap", "coupling_run"),
    ):
        manifest = _load_json(paths[manifest_name])
        _require_equal(manifest.get("schema_version"), RUN_SCHEMA, f"{runner} run schema")
        _require_equal(manifest.get("experiment_id"), EXPERIMENT_ID, f"{runner} experiment")
        _require_equal(manifest.get("phase"), "development_descriptive", f"{runner} phase")
        _require_equal(manifest.get("runner"), runner, f"{runner} id")
        _require_equal(manifest["dataset"]["id"], DEFAULT_DATASET, f"{runner} dataset")
        _require_equal(
            manifest["dataset"]["revision"], DEFAULT_DATASET_REVISION, f"{runner} revision"
        )
        _require_equal(manifest["dataset"]["panel_id"], PANEL_ID, f"{runner} panel")
        _require_equal(manifest["spec"]["sha256"], spec_sha256, f"{runner} spec")
        _require_equal(manifest["counts"]["songs"], 40, f"{runner} songs")
        _require_equal(manifest["counts"]["charts"], 170, f"{runner} charts")
        _require_equal(manifest["counts"]["records"], 4760, f"{runner} records")
        _require_equal(manifest["counts"]["noops"], 0, f"{runner} noops")
        _require_equal(
            manifest["output"]["record_schema_version"], RECORD_SCHEMA, f"{runner} record schema"
        )
        _require_equal(
            manifest["output"]["sha256"], output_hashes[runner], f"{runner} output hash"
        )
        for relative, digest in manifest["source_hashes"].items():
            if relative.startswith("/") or ".." in PurePosixPath(relative).parts:
                raise ValueError(f"{runner}: unsafe source path {relative!r}")
            _require_equal(_sha256(ROOT / relative), digest, f"{runner} source {relative}")
        if runner == "coupling_gap":
            _require_equal(
                manifest["inputs"]["audio_manifest_sha256"],
                feature_payload["audio_manifest_sha256"],
                "coupling audio input",
            )
            _require_equal(
                manifest["inputs"]["feature_manifest_sha256"],
                feature_sha256,
                "coupling feature input",
            )

    expected_summaries, built_spec_sha256 = _build_suite_v2_summaries(
        paths["structure"], paths["coupling"]
    )
    _require_equal(built_spec_sha256, spec_sha256, "summary spec")
    expected_csv = _render_suite_v2_summary_csv(expected_summaries).encode("utf-8")
    if paths["summary"].read_bytes() != expected_csv:
        raise ValueError("suite-v2 summary is not the deterministic record transformation")

    with paths["summary"].open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    _require_equal(len(summary), 45, "suite-v2 summary row count")
    _require_equal(list(summary[0]), SUITE_V2_SUMMARY_FIELDS, "suite-v2 summary columns")
    if any(int(row["n_complete_charts"]) != 170 for row in summary):
        raise ValueError("suite-v2 summary must retain all 170 development charts")

    summary_manifest = _load_json(paths["summary_manifest"])
    _require_equal(summary_manifest.get("schema_version"), SUMMARY_SCHEMA, "summary schema")
    _require_equal(summary_manifest.get("experiment_id"), EXPERIMENT_ID, "summary experiment")
    _require_equal(summary_manifest["spec"]["sha256"], spec_sha256, "summary manifest spec")
    _require_equal(
        summary_manifest["inputs"]["structure"]["sha256"],
        output_hashes["structure"],
        "summary structure input hash",
    )
    _require_equal(
        summary_manifest["inputs"]["coupling_gap"]["sha256"],
        output_hashes["coupling_gap"],
        "summary coupling input hash",
    )
    _require_equal(summary_manifest["output"]["sha256"], _sha256(paths["summary"]), "summary hash")
    _require_equal(summary_manifest["output"]["rows"], 45, "summary manifest rows")
    _require_equal(
        summary_manifest["output"]["columns"], SUITE_V2_SUMMARY_FIELDS, "summary manifest columns"
    )
    analysis_script = summary_manifest["analysis_script"]
    _require_equal(
        _sha256(ROOT / analysis_script["path"]),
        analysis_script["sha256"],
        "summary analysis script hash",
    )


def verify_release_artifacts(manifest_path: Path = DEFAULT_MANIFEST) -> dict:
    with manifest_path.open() as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != "chartgeneval.release_manifest.v1":
        raise ValueError("unsupported release manifest schema")
    if not DATASET_REVISION_RE.fullmatch(str(manifest.get("dataset_revision", ""))):
        raise ValueError("dataset_revision must be a lowercase 40-character commit")

    checked = 0
    total_bytes = 0
    total_rows = 0
    listed_paths: set[str] = set()
    for entry in manifest.get("files", []):
        relative = entry["path"]
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.as_posix() != relative
        ):
            raise ValueError(f"unsafe manifest path: {relative!r}")
        if relative in listed_paths:
            raise ValueError(f"duplicate manifest path: {relative}")
        listed_paths.add(relative)
        if not isinstance(entry.get("bytes"), int) or entry["bytes"] < 0:
            raise ValueError(f"{relative}: bytes must be a non-negative integer")
        if not SHA256_RE.fullmatch(str(entry.get("sha256", ""))):
            raise ValueError(f"{relative}: invalid sha256")
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{relative}: bytes={size}; expected {entry['bytes']}")
        digest = _sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"{relative}: sha256={digest}; expected {entry['sha256']}")
        rows, keys = _inspect(path)
        expected_rows = entry.get("rows")
        if expected_rows is not None and rows != expected_rows:
            raise ValueError(f"{relative}: rows={rows}; expected {expected_rows}")
        if relative.startswith("artifacts/records/"):
            forbidden = sorted(PROHIBITED_RECORD_KEYS & keys)
            if forbidden:
                raise ValueError(f"{relative}: prohibited record keys {forbidden}")
        checked += 1
        total_bytes += size
        total_rows += rows or 0

    required_external = {
        "ARTIFACT_LICENSE.md",
        "src/chartgeneval/data/taiko_1000_parsed_clean_calibration_v2.json",
    }
    missing_external = required_external - listed_paths
    if missing_external:
        raise ValueError(f"manifest omits required files: {sorted(missing_external)}")
    actual_artifacts = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "artifacts").rglob("*")
        if path.is_file() and path != DEFAULT_MANIFEST
    }
    listed_artifacts = {
        path for path in listed_paths if path.startswith("artifacts/")
    }
    _require_equal(listed_artifacts, actual_artifacts, "artifact file inventory")
    _verify_confirmatory_bundle(listed_paths)
    _verify_primary_summary_csv()
    _verify_c5_dependency_disclosure(listed_paths)
    _verify_suite_v2_development_evidence(listed_paths)

    expected_total_bytes = manifest.get("total_bytes")
    if total_bytes != expected_total_bytes:
        raise ValueError(
            f"manifest total_bytes={expected_total_bytes}; calculated {total_bytes}"
        )

    result = {"files": checked, "bytes": total_bytes, "rows": total_rows, "status": "ok"}
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    verify_release_artifacts(args.manifest)
