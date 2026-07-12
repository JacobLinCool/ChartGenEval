"""Shared, fail-closed contract and hashing utilities.

This module contains no analysis choices: those live in the immutable JSON
contract.  It validates that the executable registry still matches that
contract and supplies stable serialization for all evidence layers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
EXPERIMENT_ID = "confirmatory_holdout_v1"
SPEC_PATH = HERE / "SPEC.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def runtime_dependency_contract() -> dict:
    import importlib.metadata
    import platform

    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"required frozen dependency is not installed: {name}") from exc

    python_parts = tuple(int(x) for x in platform.python_version().split(".")[:2])
    if not ((3, 10) <= python_parts < (3, 13)):
        raise RuntimeError(
            f"unsupported experiment Python {python_parts[0]}.{python_parts[1]}; "
            "use the repository .venv (requires >=3.10,<3.13)"
        )
    expected_venv = (REPO_ROOT / ".venv").resolve()
    if Path(sys.prefix).resolve() != expected_venv:
        raise RuntimeError(
            f"experiment must run inside repository .venv: expected {expected_venv}, "
            f"observed {Path(sys.prefix).resolve()}"
        )
    pyvenv_cfg = expected_venv / "pyvenv.cfg"
    if not pyvenv_cfg.is_file():
        raise RuntimeError("repository .venv has no pyvenv.cfg")
    return {
        "venv_path": ".venv",
        "pyvenv_cfg_sha256": file_sha256(pyvenv_cfg),
        "python_major_minor": f"{python_parts[0]}.{python_parts[1]}",
        "numpy": version("numpy"),
        "scipy": version("scipy"),
        "datasets": version("datasets"),
    }


def stable_json(value: Any) -> str:
    """Canonical JSON used for record and content hashes."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def clean_json(value: Any) -> Any:
    """Convert NumPy scalars/non-finite floats to strict JSON values."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if hasattr(value, "item"):
        try:
            return clean_json(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    p = p.resolve()
    try:
        p.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError(f"path escapes repository root: {path}") from exc
    return p


def write_json_exclusive(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(stable_json(clean_json(value)))
        f.write("\n")


def append_jsonl(handle, value: Any) -> None:
    handle.write(stable_json(clean_json(value)))
    handle.write("\n")
    handle.flush()


def _as_float_list(values: Iterable[Any]) -> list[float]:
    return [float(x) for x in values]


def load_contract(config_path: str | Path) -> tuple[dict, dict, Path, Path]:
    config_path = resolve_repo_path(config_path)
    config = load_json(config_path)
    if config.get("schema_version") != "chartgeneval.confirmatory_run_config.v1":
        raise ValueError("unsupported run config schema")
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("run config experiment_id mismatch")
    contract_path = resolve_repo_path(config.get("contract_path", ""))
    contract = load_json(contract_path)
    validate_contract(contract, config)
    return config, contract, config_path, contract_path


def validate_contract(contract: dict, config: dict) -> None:
    if contract.get("schema_version") != "chartgeneval.confirmatory_contract.v1":
        raise ValueError("unsupported confirmatory contract schema")
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("contract experiment_id mismatch")
    if contract.get("spec_version") != "1.0.0":
        raise ValueError("unexpected spec version")
    if contract.get("language_model") != {
        "dataset_id": "JacobLinCool/taiko-1000-parsed-clean",
        "dataset_revision": "b72da4616d643018e81f372cea06ce51349285e0",
        "split": "train",
        "range": "full_split",
        "row_policy": "all_source_rows_including_aliases",
        "expected_source_rows": 924,
        "expected_canonical_groups": 813,
        "expected_charts": 3880,
        "order": 3,
        "alpha": 0.05,
    }:
        raise ValueError("language-model contract drift")

    seeds = contract.get("replicate_base_seeds")
    if not isinstance(seeds, list) or len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("contract must define exactly five unique replicate seeds")
    if any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("replicate seeds must be integers")

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from chartgeneval.metrics.common import SCORE_VERSION
    from chartgeneval.probes import PROBES

    if SCORE_VERSION != contract.get("expected_score_version"):
        raise ValueError(
            f"score version drift: executable={SCORE_VERSION!r}, "
            f"contract={contract.get('expected_score_version')!r}"
        )
    declared = contract.get("probes") or {}
    if set(declared) != set(PROBES):
        raise ValueError(
            f"probe registry drift: executable={sorted(PROBES)}, contract={sorted(declared)}"
        )
    for probe_id, (_op, doses, _needs_lm) in PROBES.items():
        if _as_float_list(doses) != _as_float_list(declared[probe_id].get("dose_values", [])):
            raise ValueError(f"dose drift for {probe_id}")
        severity = declared[probe_id].get("severity")
        if not isinstance(severity, list) or len(severity) != 4 or float(severity[0]) != 0.0:
            raise ValueError(f"invalid four-point severity vector for {probe_id}")

    pairs = contract.get("primary_pairs") or []
    if len(pairs) != 10 or len({p.get("pair_id") for p in pairs}) != 10:
        raise ValueError("contract must define ten unique primary pairs")
    controls = contract.get("controls") or {}
    for pair in pairs:
        if pair.get("probe") not in declared:
            raise ValueError(f"unknown pair probe: {pair}")
        if pair.get("expected_raw_direction") not in {"increase", "decrease"}:
            raise ValueError(f"invalid direction: {pair}")
        sham = pair.get("scoped_sham")
        if sham not in controls:
            raise ValueError(f"unknown scoped sham {sham!r}")
        if pair.get("family") not in controls[sham].get("scoped_families", []):
            raise ValueError(f"sham {sham!r} is out of scope for pair {pair.get('pair_id')}")
        if pair.get("metric") not in controls[sham].get("metrics", []):
            raise ValueError(f"sham {sham!r} does not register metric {pair.get('metric')}")
        expected_sham_dose = 3 if sham == "C2_anchor_shift" else 0
        if pair.get("scoped_sham_dose_index") != expected_sham_dose:
            raise ValueError(f"scoped sham dose drift for pair {pair.get('pair_id')}")
    c5 = [p for p in pairs if p.get("probe") == "C5_blandification"]
    if {p.get("metric") for p in c5} != {
        "repetition_adequacy_score",
        "surface_variety_adequacy_score",
    }:
        raise ValueError("C5 co-primary pair drift")
    claims = contract.get("co_primary_claims") or []
    if claims != [
        {
            "claim_id": "C5__repetition_and_variety",
            "pair_ids": ["C5__repetition", "C5__surface_variety"],
            "combination_rule": "all_pairs_must_survive",
        }
    ]:
        raise ValueError("C5 co-primary combination rule drift")

    tolerances = contract.get("control_absolute_tolerances") or {}
    retained = retained_metric_names(contract)
    if not retained.issubset(tolerances):
        missing = sorted(retained - set(tolerances))
        raise ValueError(f"control tolerances missing metrics: {missing}")
    if any(float(tolerances[m]) < 0 for m in retained):
        raise ValueError("control tolerances must be non-negative")
    analysis = contract.get("analysis") or {}
    if analysis.get("constant_response_dose_statistic") != 0.0:
        raise ValueError("constant-response statistic drift")
    if analysis.get("cluster_unit") != "group_id":
        raise ValueError("cluster unit drift")
    if analysis.get("complete_case_rule") != (
        "course_requires_official_all_three_target_doses_and_fixed_scoped_sham; "
        "song_requires_at_least_one_complete_course; both_statistics_use_identical_courses"
    ):
        raise ValueError("complete-case rule drift")

    dataset = config.get("dataset") or {}
    start = dataset.get("canonical_song_index_start_inclusive")
    stop = dataset.get("canonical_song_index_stop_exclusive")
    context_start = dataset.get("sampling_context_canonical_start_inclusive")
    context_stop = dataset.get("sampling_context_canonical_stop_exclusive")
    if not all(isinstance(v, int) for v in (start, stop, context_start, context_stop)):
        raise ValueError("dataset panel indices must be integers")
    if not (0 <= context_start <= start < stop <= context_stop):
        raise ValueError("invalid scoring/context panel boundaries")
    phase = config.get("phase")
    if phase == "confirmatory":
        expected = {
            "source": "huggingface",
            "id": "JacobLinCool/taiko-1000-parsed-clean",
            "revision": "b72da4616d643018e81f372cea06ce51349285e0",
            "split": "test",
            "canonical_filter": True,
            "canonical_song_index_start_inclusive": 40,
            "canonical_song_index_stop_exclusive": 120,
            "sampling_context_canonical_start_inclusive": 0,
            "sampling_context_canonical_stop_exclusive": 120,
            "expected_selected_canonical_songs": 80,
            "expected_source_rows": 140,
            "expected_canonical_songs": 120,
            "expected_filtered_alias_rows": 20,
            "scan_full_split": True,
            "ordering": "stored_order_at_pinned_revision",
        }
        for key, value in expected.items():
            if dataset.get(key) != value:
                raise ValueError(f"confirmatory panel drift for dataset.{key}")
        if not config.get("require_freeze_manifest"):
            raise ValueError("confirmatory run must require a freeze manifest")
        if config.get("freeze_path") != (
            "experiments/confirmatory_holdout_v1/frozen/confirmatory_freeze.json"
        ):
            raise ValueError("confirmatory freeze path drift")
        if config.get("confirmatory_anchor_path") != (
            "experiments/confirmatory_holdout_v1/CONFIRMATORY_ANCHOR.json"
        ):
            raise ValueError("confirmatory anchor path drift")
        if config.get("required_pre_freeze_smoke_phase") != "smoke_development":
            raise ValueError("pre-freeze smoke requirement drift")
        if config.get("clean_split_manifest_path") != (
            "artifacts/splits/clean_split_manifest.json"
        ):
            raise ValueError("clean split manifest path drift")
    elif phase == "smoke_synthetic":
        expected = {
            "source": "synthetic",
            "canonical_filter": True,
            "canonical_song_index_start_inclusive": 0,
            "canonical_song_index_stop_exclusive": 4,
            "sampling_context_canonical_start_inclusive": 0,
            "sampling_context_canonical_stop_exclusive": 4,
            "expected_selected_canonical_songs": 4,
            "expected_source_rows": 4,
            "expected_canonical_songs": 4,
            "expected_filtered_alias_rows": 0,
            "scan_full_split": True,
        }
        for key, value in expected.items():
            if dataset.get(key) != value:
                raise ValueError(f"synthetic smoke contract drift for dataset.{key}")
        if config.get("require_freeze_manifest"):
            raise ValueError("synthetic smoke must not require confirmatory freeze")
    elif phase == "smoke_development":
        expected = {
            "source": "huggingface",
            "id": "JacobLinCool/taiko-1000-parsed-clean",
            "revision": "b72da4616d643018e81f372cea06ce51349285e0",
            "split": "test",
            "canonical_filter": True,
            "canonical_song_index_start_inclusive": 0,
            "canonical_song_index_stop_exclusive": 2,
            "sampling_context_canonical_start_inclusive": 0,
            "sampling_context_canonical_stop_exclusive": 2,
            "expected_selected_canonical_songs": 2,
            "stop_after_canonical_songs": 2,
            "scan_full_split": False,
            "ordering": "stored_order_at_pinned_revision",
        }
        for key, value in expected.items():
            if dataset.get(key) != value:
                raise ValueError(f"development smoke panel drift for dataset.{key}")
        if config.get("require_freeze_manifest"):
            raise ValueError("development smoke must not require confirmatory freeze")
        if config.get("clean_split_manifest_path") != (
            "artifacts/splits/clean_split_manifest.json"
        ):
            raise ValueError("clean split manifest path drift")
    else:
        raise ValueError(f"unregistered phase: {phase!r}")

    if contract.get("clean_split_contract") != {
        "path": "artifacts/splits/clean_split_manifest.json",
        "train_source_rows": 924,
        "train_canonical_groups": 813,
        "test_source_rows": 140,
        "test_canonical_groups": 120,
        "test_alias_rows": 20,
    }:
        raise ValueError("clean split contract drift")
    split_payload = load_json(resolve_repo_path(contract["clean_split_contract"]["path"]))
    split_manifest = split_payload.get("manifest")
    if not isinstance(split_manifest, dict):
        raise ValueError("clean split artifact schema drift")
    for split, expected_rows, expected_groups in (
        ("train", 924, 813),
        ("test", 140, 120),
    ):
        entries = [entry for entry in split_manifest.values() if entry.get("split") == split]
        canonical_entries = [entry for entry in entries if bool(entry.get("canonical"))]
        if (
            len(entries) != expected_rows
            or len(canonical_entries) != expected_groups
            or len({int(entry["group_id"]) for entry in canonical_entries})
            != expected_groups
            or len({str(entry["audio_sha256"]) for entry in canonical_entries})
            != expected_groups
        ):
            raise ValueError(f"clean split {split} artifact count/uniqueness drift")


def retained_metric_names(contract: dict) -> set[str]:
    names = {p["metric"] for p in contract.get("primary_pairs", [])}
    names.update(m["metric"] for m in contract.get("secondary_metrics", []))
    for control in (contract.get("controls") or {}).values():
        names.update(control.get("metrics", []))
    return names


def source_snapshot(contract: dict, config_path: str | Path) -> dict:
    paths = list(contract.get("frozen_source_paths") or [])
    paths.extend(
        [
            str(resolve_repo_path(config_path).relative_to(REPO_ROOT)),
            "experiments/confirmatory_holdout_v1/freeze.py",
        ]
    )
    for pattern in contract.get("frozen_source_globs") or []:
        matches = sorted(REPO_ROOT.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"frozen source glob has no matches: {pattern}")
        paths.extend(str(path.relative_to(REPO_ROOT)) for path in matches if path.is_file())
    paths = sorted(set(paths))
    files = []
    for relative in paths:
        path = resolve_repo_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"frozen source is missing: {relative}")
        files.append(
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    bundle_hash = sha256_text(stable_json(files))
    return {"files": files, "code_bundle_sha256": bundle_hash}


def verify_source_snapshot_files(snapshot: dict) -> None:
    files = snapshot.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("source snapshot has no frozen files")
    observed = []
    for entry in files:
        path = resolve_repo_path(entry["path"])
        if not path.is_file():
            raise FileNotFoundError(f"frozen source disappeared: {entry['path']}")
        current = {
            "path": entry["path"],
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        if current != entry:
            raise ValueError(f"live source differs from frozen snapshot: {entry['path']}")
        observed.append(current)
    if sha256_text(stable_json(observed)) != snapshot.get("code_bundle_sha256"):
        raise ValueError("frozen source bundle hash mismatch")


def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "head_sha": run("rev-parse", "HEAD"),
        "is_dirty": bool(status) if status is not None else None,
    }


def find_pre_freeze_smoke(config: dict, snapshot: dict) -> dict:
    """Locate one completed real-data [0,2) chain matching current code/runtime."""
    raw_root = resolve_repo_path(config["raw_root"])
    if not raw_root.exists():
        raise RuntimeError("required development smoke has not been run")
    for manifest_path in sorted(raw_root.glob("*/MANIFEST.json"), reverse=True):
        manifest = load_json(manifest_path)
        if manifest.get("phase") != config.get("required_pre_freeze_smoke_phase"):
            continue
        if manifest.get("status") != "completed":
            continue
        if manifest.get("code_version") != snapshot:
            continue
        if manifest.get("runtime_dependency_contract") != runtime_dependency_contract():
            continue
        if manifest.get("dataset_revision") != config["dataset"]["revision"]:
            continue
        if (
            len(manifest.get("selected_group_ids_ordered") or []) != 2
            or len(set(manifest["selected_group_ids_ordered"])) != 2
            or len(set(manifest.get("selected_audio_sha256_ordered") or [])) != 2
            or int(manifest.get("counts", {}).get("canonical_context_rows") or 0) != 2
        ):
            continue
        table_manifest_path = manifest_path.parent / "tables" / "MANIFEST.json"
        analysis_manifest_path = manifest_path.parent / "tables" / "reports" / "MANIFEST.json"
        if not table_manifest_path.is_file() or not analysis_manifest_path.is_file():
            continue
        table_manifest = load_json(table_manifest_path)
        analysis_manifest = load_json(analysis_manifest_path)
        if table_manifest.get("source_run_manifest_sha256") != file_sha256(manifest_path):
            continue
        if analysis_manifest.get("input_table_manifest_sha256") != file_sha256(
            table_manifest_path
        ):
            continue
        if analysis_manifest.get("provenance_valid") is not True:
            continue
        if analysis_manifest.get("runtime_dependency_contract") != runtime_dependency_contract():
            continue
        return {
            "run_id": manifest["run_id"],
            "run_manifest_path": str(manifest_path.relative_to(REPO_ROOT)),
            "run_manifest_sha256": file_sha256(manifest_path),
            "table_manifest_sha256": file_sha256(table_manifest_path),
            "analysis_manifest_sha256": file_sha256(analysis_manifest_path),
            "selected_group_ids_ordered_sha256": manifest[
                "selected_group_ids_ordered_sha256"
            ],
            "selected_audio_sha256_ordered_sha256": manifest[
                "selected_audio_sha256_ordered_sha256"
            ],
            "selected_panel_fingerprint_sha256": manifest[
                "selected_panel_fingerprint_sha256"
            ],
        }
    raise RuntimeError(
        "no completed development [0,2) run→transform→analyze chain matches current source/runtime"
    )


def build_freeze_payload(config_path: str | Path) -> dict:
    config, contract, config_path, contract_path = load_contract(config_path)
    dataset = config["dataset"]
    snapshot = source_snapshot(contract, config_path)
    payload = {
        "schema_version": "chartgeneval.confirmatory_freeze.v1",
        "experiment_id": EXPERIMENT_ID,
        "spec_version": contract["spec_version"],
        "created_at_utc": utc_now(),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": file_sha256(config_path),
        "contract_path": str(contract_path.relative_to(REPO_ROOT)),
        "contract_sha256": file_sha256(contract_path),
        "spec_path": str(SPEC_PATH.relative_to(REPO_ROOT)),
        "spec_sha256": file_sha256(SPEC_PATH),
        "expected_score_version": contract["expected_score_version"],
        "dataset": {
            "id": dataset["id"],
            "revision": dataset["revision"],
            "split": dataset["split"],
            "canonical_filter": dataset["canonical_filter"],
            "canonical_song_index_start_inclusive": dataset[
                "canonical_song_index_start_inclusive"
            ],
            "canonical_song_index_stop_exclusive": dataset[
                "canonical_song_index_stop_exclusive"
            ],
            "ordering": dataset["ordering"],
            "clean_split_manifest_sha256": file_sha256(
                resolve_repo_path(config["clean_split_manifest_path"])
            ),
        },
        "source_snapshot": snapshot,
        "runtime_dependency_contract": runtime_dependency_contract(),
        "git_state_at_freeze": git_state(),
    }
    if config["phase"] == "confirmatory":
        payload["pre_freeze_smoke_evidence"] = find_pre_freeze_smoke(config, snapshot)
    return payload


def verify_freeze(freeze_path: str | Path, config_path: str | Path) -> dict:
    freeze_path = resolve_repo_path(freeze_path)
    observed = load_json(freeze_path)
    expected = build_freeze_payload(config_path)
    if observed.get("schema_version") != "chartgeneval.confirmatory_freeze.v1":
        raise ValueError("unsupported freeze schema")
    # Time and git dirtiness are provenance, not equality inputs. Every
    # experiment-defining content hash and dataset field must match exactly.
    for key in (
        "experiment_id",
        "spec_version",
        "config_path",
        "config_sha256",
        "contract_path",
        "contract_sha256",
        "spec_path",
        "spec_sha256",
        "expected_score_version",
        "dataset",
        "source_snapshot",
        "runtime_dependency_contract",
        "pre_freeze_smoke_evidence",
    ):
        if observed.get(key) != expected.get(key):
            raise ValueError(f"stale or mismatched freeze field: {key}")
    verify_source_snapshot_files(observed["source_snapshot"])
    return observed


def derive_rng_seed(base_seed: int, sid: str, course: str, condition: str, dose: int) -> int:
    key = f"{base_seed}|{sid}|{course}|{condition}|{dose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def invocation() -> list[str]:
    return [os.path.realpath(sys.executable), *sys.argv]
