#!/usr/bin/env python3
"""Execute the frozen corruption panel and append raw evidence.

The confirmatory panel requires both a matching immutable freeze snapshot and
an explicit command-line acknowledgement. Synthetic smoke runs exercise the
same five-replicate path without touching the held-out song groups.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from contract import (
    EXPERIMENT_ID,
    REPO_ROOT,
    append_jsonl,
    clean_json,
    derive_rng_seed,
    file_sha256,
    git_state,
    invocation,
    load_contract,
    load_json,
    resolve_repo_path,
    runtime_dependency_contract,
    sha256_text,
    source_snapshot,
    stable_json,
    utc_now,
    verify_freeze,
    write_json_exclusive,
)

sys.path.insert(0, str(REPO_ROOT / "src"))

from chartgeneval.calibration import (  # noqa: E402
    build_calibration,
    evaluate_chart_quality,
    load_bundled_calibration,
    require_calibration_provenance,
)
from chartgeneval.events import COURSES, Chart, load_taiko_parsed_course, sorted_hits  # noqa: E402
from chartgeneval.metrics import timing  # noqa: E402
from chartgeneval.metrics.common import NGramModel, SCORE_VERSION, event_tokens  # noqa: E402
from chartgeneval.probes import PROBES, apply_probe  # noqa: E402


def _write_text_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(text)


def _ensure_confirmatory_anchor(
    config: dict,
    freeze: dict,
    freeze_path: Path,
    raw_root: Path,
    run_id: str,
) -> tuple[dict, Path]:
    anchor_path = resolve_repo_path(config["confirmatory_anchor_path"])
    existing_manifests = []
    if raw_root.exists():
        for manifest_path in sorted(raw_root.glob("*/MANIFEST.json")):
            manifest = load_json(manifest_path)
            if manifest.get("phase") == "confirmatory":
                existing_manifests.append((manifest_path, manifest))
    if anchor_path.exists() or existing_manifests:
        raise RuntimeError(
            "this experiment id already has a confirmatory anchor or manifest; "
            "a second collection is forbidden"
        )
    fixed = {
        "schema_version": "chartgeneval.confirmatory_anchor.v1",
        "experiment_id": EXPERIMENT_ID,
        "primary_run_id": run_id,
        "freeze_path": str(freeze_path.relative_to(REPO_ROOT)),
        "freeze_sha256": file_sha256(freeze_path),
        "spec_sha256": freeze["spec_sha256"],
        "contract_sha256": freeze["contract_sha256"],
        "config_sha256": freeze["config_sha256"],
        "code_bundle_sha256": freeze["source_snapshot"]["code_bundle_sha256"],
        "runtime_dependency_contract": freeze["runtime_dependency_contract"],
        "dataset": freeze["dataset"],
    }
    try:
        write_json_exclusive(anchor_path, {**fixed, "created_at_utc": utc_now()})
    except FileExistsError as exc:
        raise RuntimeError("concurrent or repeated confirmatory collection is forbidden") from exc
    anchor = load_json(anchor_path)
    for key, value in fixed.items():
        if anchor.get(key) != value:
            raise RuntimeError(f"confirmatory anchor drift: {key}")
    return anchor, anchor_path


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_environment() -> dict:
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "datasets": _package_version("datasets"),
            "chartgeneval": _package_version("chartgeneval"),
        },
        "score_version": SCORE_VERSION,
    }


def _official_charts(row: dict) -> Iterator[tuple[str, Chart, dict]]:
    for course in COURSES:
        course_struct = row.get(course)
        chart = load_taiko_parsed_course(course_struct)
        if chart is not None and chart.events:
            chart.course = course
            yield course, chart, course_struct


def _synthetic_course(bpm: float, level: int, bars: int, subdivisions: int, offset: int) -> dict:
    beat = 60.0 / bpm
    bar_s = 4.0 * beat
    note_names = ("Don", "Ka", "Don", "KaBig", "Ka", "DonBig", "Ka", "Don")
    segments = []
    for bar_i in range(bars):
        start = 0.5 + bar_i * bar_s
        notes = []
        for j in range(subdivisions):
            notes.append(
                {
                    "timestamp": start + j * bar_s / subdivisions,
                    "note_type": note_names[(j * 3 + bar_i + offset) % len(note_names)],
                    "bpm": bpm,
                }
            )
        segments.append(
            {
                "timestamp": start,
                "measure_num": 4,
                "measure_den": 4,
                "notes": notes,
            }
        )
    return {"level": level, "segments": segments}


def _synthetic_rows(n: int, prefix: str) -> Iterator[tuple[int, str, dict]]:
    for index in range(n):
        bpm = float(112 + 7 * index)
        row = {
            "metadata": {"TITLE": f"synthetic-{prefix}-{index}"},
            "group_id": index,
            "canonical": True,
            "audio_sha256": sha256_text(f"synthetic-audio|{prefix}|{index}"),
            "easy": _synthetic_course(bpm, 2 + index % 3, 8, 4, index),
            "oni": _synthetic_course(bpm, 7 + index % 4, 8, 8, index + 2),
        }
        yield index, f"{prefix}_{index:05d}", row


def _hf_rows(
    dataset: dict,
    split: str,
    expected_identity_counts: Counter,
    *,
    require_exhaustion: bool,
    stop: int | None = None,
) -> Iterator[tuple[int, str, dict]]:
    from datasets import Audio, load_dataset

    ds = load_dataset(
        dataset["id"],
        split=split,
        revision=dataset["revision"],
        streaming=True,
    )
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))
    remaining = expected_identity_counts.copy()
    for index, row in enumerate(ds):
        if stop is not None and index >= stop:
            break
        observed = (
            int(row.get("group_id")),
            bool(row.get("canonical")),
            str(row.get("audio_sha256")),
        )
        if remaining[observed] <= 0:
            raise RuntimeError(
                f"HF {split} raw row {index} identity is absent/exhausted in clean split manifest"
            )
        remaining[observed] -= 1
        stored_split_row_id = f"{split}_row_{index:05d}"
        yield index, stored_split_row_id, row
    if require_exhaustion and any(count != 0 for count in remaining.values()):
        raise RuntimeError(f"HF {split} rows do not exhaust the clean split manifest multiset")


def _clean_split_entries(config: dict, contract: dict, split: str) -> list[tuple[str, dict]]:
    path = resolve_repo_path(config["clean_split_manifest_path"])
    payload = load_json(path)
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("clean split artifact has no manifest mapping")
    entries = [
        (manifest_source_row_id, entry)
        for manifest_source_row_id, entry in manifest.items()
        if entry.get("split") == split
    ]
    split_contract = contract["clean_split_contract"]
    expected_rows = split_contract.get(f"{split}_source_rows")
    if expected_rows is not None and len(entries) != int(expected_rows):
        raise ValueError(f"clean split {split} row-count drift")
    return entries


def _identity_counter(entries: list[tuple[str, dict]]) -> Counter:
    return Counter(
        (
            int(entry["group_id"]),
            bool(entry["canonical"]),
            str(entry["audio_sha256"]),
        )
        for _manifest_row_id, entry in entries
    )


def _canonical_manifest_id_map(entries: list[tuple[str, dict]]) -> dict[tuple[int, str], str]:
    out = {}
    for manifest_source_row_id, entry in entries:
        if not bool(entry["canonical"]):
            continue
        key = (int(entry["group_id"]), str(entry["audio_sha256"]))
        if key in out:
            raise ValueError(f"duplicate canonical identity in split manifest: {key}")
        out[key] = manifest_source_row_id
    return out


def _iter_source_panel(
    rows: Iterable[tuple[int, str, dict]],
    *,
    context_stop: int,
    scan_full_split: bool,
    canonical_manifest_ids: dict[tuple[int, str], str] | None,
) -> Iterator[tuple[dict, dict]]:
    """Attach canonical indices while preserving the raw stored-row order.

    Alias rows are emitted as source-scan evidence but never enter the
    canonical context.  Development scans terminate immediately after the
    requested canonical prefix, so this iterator cannot consume a later
    holdout row by accident.
    """
    canonical_song_index = 0
    seen_group_ids: set[int] = set()
    seen_audio_sha256: set[str] = set()
    for source_row_index, stored_split_row_id, row in rows:
        if not isinstance(row.get("canonical"), bool):
            raise RuntimeError("source row canonical flag is not a JSON boolean")
        is_canonical = row["canonical"]
        group_id = int(row["group_id"])
        audio_sha256 = str(row["audio_sha256"])
        manifest_source_row_id = None
        if is_canonical:
            if canonical_song_index >= context_stop:
                raise RuntimeError(
                    "more canonical source songs than the registered context contract"
                )
            if group_id in seen_group_ids:
                raise RuntimeError(f"duplicate canonical group_id in source scan: {group_id}")
            if audio_sha256 in seen_audio_sha256:
                raise RuntimeError(
                    f"duplicate canonical audio_sha256 in source scan: {audio_sha256}"
                )
            seen_group_ids.add(group_id)
            seen_audio_sha256.add(audio_sha256)
            if canonical_manifest_ids is not None:
                manifest_source_row_id = canonical_manifest_ids.get(
                    (group_id, audio_sha256)
                )
                if manifest_source_row_id is None:
                    raise RuntimeError(
                        "canonical source row is absent from the clean split manifest"
                    )
        scan_row = {
            "schema_version": "chartgeneval.source_scan_raw.v1",
            "source_row_index": int(source_row_index),
            "stored_split_row_id": str(stored_split_row_id),
            "manifest_source_row_id": manifest_source_row_id,
            "group_id": group_id,
            "canonical": is_canonical,
            "audio_sha256": audio_sha256,
            "canonical_song_index": canonical_song_index if is_canonical else None,
            "action": "canonical_context" if is_canonical else "filtered_alias",
        }
        yield scan_row, row
        if is_canonical:
            canonical_song_index += 1
            if not scan_full_split and canonical_song_index >= context_stop:
                return


def _validate_source_scan_contract(
    scan_rows: list[dict],
    dataset: dict,
    *,
    expected_canonical_set: set[tuple[int, str]] | None,
) -> dict[str, int]:
    """Fail closed on raw/canonical/alias counts and canonical identities."""
    if not scan_rows:
        raise RuntimeError("source scan is empty")
    canonical_rows = [row for row in scan_rows if row["canonical"] is True]
    alias_rows = [row for row in scan_rows if row["canonical"] is False]
    split = str(dataset["split"])
    for expected_index, row in enumerate(scan_rows):
        if int(row["source_row_index"]) != expected_index:
            raise RuntimeError("source scan row indices are not a zero-based stored-order prefix")
        if dataset["source"] == "huggingface" and row["stored_split_row_id"] != (
            f"{split}_row_{expected_index:05d}"
        ):
            raise RuntimeError("stored split row ID does not match its raw row index")
        if row["canonical"] is True:
            if row["action"] != "canonical_context":
                raise RuntimeError("canonical source row has a non-context action")
            if dataset["source"] == "huggingface" and not row.get(
                "manifest_source_row_id"
            ):
                raise RuntimeError("canonical HF row lacks its true split-manifest source ID")
        elif (
            row["action"] != "filtered_alias"
            or row.get("canonical_song_index") is not None
            or row.get("manifest_source_row_id") is not None
        ):
            raise RuntimeError("alias source row leaked into canonical metadata")
    canonical_indices = [int(row["canonical_song_index"]) for row in canonical_rows]
    if canonical_indices != list(range(len(canonical_rows))):
        raise RuntimeError("canonical indices do not follow filtered stored-row order")
    canonical_groups = [int(row["group_id"]) for row in canonical_rows]
    canonical_audio = [str(row["audio_sha256"]) for row in canonical_rows]
    if len(canonical_groups) != len(set(canonical_groups)):
        raise RuntimeError("canonical source scan contains duplicate group_id values")
    if len(canonical_audio) != len(set(canonical_audio)):
        raise RuntimeError("canonical source scan contains duplicate audio_sha256 values")
    observed_set = set(zip(canonical_groups, canonical_audio, strict=True))
    if dataset["scan_full_split"]:
        expected_counts = (
            int(dataset["expected_source_rows"]),
            int(dataset["expected_canonical_songs"]),
            int(dataset["expected_filtered_alias_rows"]),
        )
        observed_counts = (len(scan_rows), len(canonical_rows), len(alias_rows))
        if observed_counts != expected_counts:
            raise RuntimeError(
                "raw/canonical/alias source counts drift: "
                f"observed={observed_counts}, expected={expected_counts}"
            )
        if expected_canonical_set is not None and observed_set != expected_canonical_set:
            raise RuntimeError("canonical source set differs from clean split test set")
    else:
        expected_prefix = int(dataset["sampling_context_canonical_stop_exclusive"])
        if len(canonical_rows) != expected_prefix or scan_rows[-1]["canonical"] is not True:
            raise RuntimeError("development source scan did not stop at its canonical prefix")
        if expected_canonical_set is not None and not observed_set.issubset(
            expected_canonical_set
        ):
            raise RuntimeError("development canonical prefix is outside the clean split test set")
    return {
        "scanned_source_rows": len(scan_rows),
        "canonical_context_rows": len(canonical_rows),
        "filtered_alias_rows": len(alias_rows),
    }


def _lm_state_fingerprint(lm: NGramModel) -> str:
    def counter_rows(counter) -> list:
        return [[list(key), int(value)] for key, value in sorted(counter.items())]

    payload = {
        "order": int(lm.order),
        "alpha": float(lm.alpha),
        "vocab": sorted(lm.vocab),
        "course_counts": {
            course: counter_rows(counter) for course, counter in sorted(lm.course_counts.items())
        },
        "course_context": {
            course: counter_rows(counter) for course, counter in sorted(lm.course_context.items())
        },
        "all_counts": counter_rows(lm.all_counts),
        "all_context": counter_rows(lm.all_context),
    }
    return sha256_text(stable_json(payload))


def _prepare_models(
    config: dict, contract: dict, dataset_fingerprints: dict
) -> tuple[NGramModel, dict, dict]:
    dataset = config["dataset"]
    lm_contract = contract["language_model"]
    lm = NGramModel(order=int(lm_contract["order"]), alpha=float(lm_contract["alpha"]))
    n_train_charts = 0
    input_hasher = hashlib.sha256()
    train_dataset_hasher = hashlib.sha256()
    n_train_rows = 0
    train_group_ids = set()

    def add_training_chart(sid: str, course: str, chart: Chart) -> None:
        nonlocal n_train_charts
        tokens = event_tokens(chart.events, chart.bpm)
        lm.add(course, tokens)
        material = {
            "sid": sid,
            "course": course,
            "bpm": float(chart.bpm),
            "events_sha256": _events_hash(chart.events),
            "tokens_sha256": sha256_text(stable_json(tokens)),
        }
        input_hasher.update((stable_json(material) + "\n").encode("utf-8"))
        n_train_charts += 1

    if dataset["source"] == "synthetic":
        dataset_fingerprints["synthetic_reference"] = "chartgeneval_confirmatory_synthetic_v1"
        calibration_rows = []
        for _index, sid, row in _synthetic_rows(16, "synthetic_train"):
            n_train_rows += 1
            train_group_ids.add(int(row["group_id"]))
            for course, chart, _course_struct in _official_charts(row):
                add_training_chart(sid, course, chart)
                calibration_rows.append(
                    {
                        "sid": sid,
                        "course": course,
                        "events": chart.events,
                        "bpm": chart.bpm,
                        "level": chart.level,
                    }
                )
        calibration = build_calibration(calibration_rows, lm)
        return lm, calibration, {
            "n_train_rows": n_train_rows,
            "n_train_charts": n_train_charts,
            "n_train_unique_group_ids": len(train_group_ids),
            "ordered_training_input_sha256": input_hasher.hexdigest(),
            "model_state_sha256": _lm_state_fingerprint(lm),
        }

    if (
        dataset["id"] != lm_contract["dataset_id"]
        or dataset["revision"] != lm_contract["dataset_revision"]
        or dataset["lm_split"] != lm_contract["split"]
        or lm_contract["range"] != "full_split"
    ):
        raise ValueError("language-model corpus contract drift")
    train_entries = _clean_split_entries(config, contract, lm_contract["split"])
    for _index, stored_split_row_id, row in _hf_rows(
        dataset,
        lm_contract["split"],
        _identity_counter(train_entries),
        require_exhaustion=True,
        stop=None,
    ):
        charts = list(_official_charts(row))
        train_dataset_hasher.update(
            (
                stable_json(
                    {
                        "source_row_index": _index,
                        "stored_split_row_id": stored_split_row_id,
                        "group_id": int(row["group_id"]),
                        "canonical": bool(row["canonical"]),
                        "audio_sha256": str(row["audio_sha256"]),
                        "charts": [
                            {
                                "course": course,
                                "chart_input_sha256": _chart_input_hash(
                                    course, chart, course_struct
                                ),
                            }
                            for course, chart, course_struct in charts
                        ],
                    }
                )
                + "\n"
            ).encode("utf-8")
        )
        n_train_rows += 1
        train_group_ids.add(int(row["group_id"]))
        for course, chart, _course_struct in charts:
            add_training_chart(stored_split_row_id, course, chart)
    if (
        n_train_rows != int(lm_contract["expected_source_rows"])
        or len(train_group_ids) != int(lm_contract["expected_canonical_groups"])
        or n_train_charts != int(lm_contract["expected_charts"])
    ):
        raise RuntimeError("full-train row/group/chart count drift")
    calibration = load_bundled_calibration()
    require_calibration_provenance(
        calibration,
        dataset_id=lm_contract["dataset_id"],
        dataset_revision=lm_contract["dataset_revision"],
        split=lm_contract["split"],
    )
    if (
        calibration["n_charts"] != int(lm_contract["expected_charts"])
        or calibration["lm_order"] != int(lm_contract["order"])
        or float(calibration["lm_alpha"]) != float(lm_contract["alpha"])
    ):
        raise ValueError("bundled calibration and language-model contract drift")
    dataset_fingerprints["train"] = {
        "kind": "ordered_parsed_chart_content_v1",
        "dataset_revision": dataset["revision"],
        "split": lm_contract["split"],
        "range": "full_split",
        "n_source_rows": n_train_rows,
        "n_unique_group_ids": len(train_group_ids),
        "n_charts": n_train_charts,
        "sha256": train_dataset_hasher.hexdigest(),
    }
    return lm, calibration, {
        "n_train_rows": n_train_rows,
        "n_train_charts": n_train_charts,
        "n_train_unique_group_ids": len(train_group_ids),
        "ordered_training_input_sha256": input_hasher.hexdigest(),
        "model_state_sha256": _lm_state_fingerprint(lm),
    }


def _strict_float_mean(values: list[float | None]) -> float | None:
    xs = [float(v) for v in values if v is not None]
    return float(np.mean(xs)) if xs else None


def _chart_summary(course: str, chart: Chart, course_struct: dict) -> dict:
    hits = sorted_hits(chart.events)
    active = hits[-1][0] - hits[0][0] if len(hits) >= 2 else 0.0
    return {
        "course": course,
        "n_notes": len(hits),
        "bpm": float(chart.bpm),
        "level": chart.level,
        "active_duration_s": float(active),
        "note_rate_nps": float(len(hits) / active) if active > 0 else None,
        "chart_input_sha256": _chart_input_hash(course, chart, course_struct),
        "events_sha256": _events_hash(chart.events),
    }


def _sampling_context_record(
    *,
    source_row_index: int,
    stored_split_row_id: str,
    manifest_source_row_id: str | None,
    canonical_song_index: int,
    row: dict,
    panel_group: str,
    charts: list[tuple],
) -> dict:
    if row.get("canonical") is not True:
        raise ValueError("sampling context accepts canonical rows only")
    summaries = [_chart_summary(course, chart, course_struct) for course, chart, course_struct in charts]
    group_id = int(row["group_id"])
    return {
        "schema_version": "chartgeneval.sampling_context_raw.v1",
        "experiment_id": EXPERIMENT_ID,
        "source_row_index": source_row_index,
        "stored_split_row_id": stored_split_row_id,
        "manifest_source_row_id": manifest_source_row_id,
        "canonical_song_index": canonical_song_index,
        "sid": f"group_{group_id}",
        "group_id": group_id,
        "canonical": bool(row["canonical"]),
        "audio_sha256": str(row["audio_sha256"]),
        "panel_group": panel_group,
        "chart_count": len(summaries),
        "course_count": len({c["course"] for c in summaries}),
        "courses": [c["course"] for c in summaries],
        "song_mean_bpm": _strict_float_mean([c["bpm"] for c in summaries]),
        "song_mean_note_rate_nps": _strict_float_mean([c["note_rate_nps"] for c in summaries]),
        "song_mean_level": _strict_float_mean([c["level"] for c in summaries]),
        "charts": summaries,
    }


def _chart_input_hash(course: str, chart: Chart, course_struct: dict) -> str:
    material = {
        "course": course,
        "bpm": float(chart.bpm),
        "level": chart.level,
        "events": [[float(t), c] for t, c in sorted_hits(chart.events)],
        "authored_segments": course_struct.get("segments") if course_struct else None,
    }
    return sha256_text(stable_json(clean_json(material)))


def _events_hash(events) -> str:
    return sha256_text(stable_json([[float(t), c] for t, c in sorted_hits(events)]))


def _shift_grid(grid: dict | None, offset_s: float) -> dict | None:
    if not grid:
        return grid
    shifted = copy.deepcopy(grid)
    for segment in shifted.get("segments") or []:
        if segment.get("timestamp") is not None:
            segment["timestamp"] = float(segment["timestamp"]) + offset_s
        for note in segment.get("notes") or []:
            if note.get("timestamp") is not None:
                note["timestamp"] = float(note["timestamp"]) + offset_s
    if shifted.get("downbeats") is not None:
        shifted["downbeats"] = [float(t) + offset_s for t in shifted["downbeats"]]
    return shifted


def _control_variant(condition: str, events, grid, duration: float) -> tuple[list, dict | None, float]:
    hits = sorted_hits(events)
    if condition == "CTRL_identity":
        return hits, copy.deepcopy(grid), duration
    if condition == "CTRL_joint_time_origin_shift":
        return [(t + 1.0, c) for t, c in hits], _shift_grid(grid, 1.0), duration + 1.0
    if condition == "CTRL_color_bijection":
        color_swap = {
            "don": "ka",
            "ka": "don",
            "don_big": "ka_big",
            "ka_big": "don_big",
        }
        return [(t, color_swap[c]) for t, c in hits], copy.deepcopy(grid), duration
    raise ValueError(f"unknown executable control: {condition}")


def _evaluate(events, *, bpm: float, course: str, grid: dict | None, duration: float, lm, calibration, retained: set[str]) -> dict:
    chart_metrics = evaluate_chart_quality(events, bpm, course, lm, calibration)
    timing_metrics = timing.compute(
        events,
        {"grid": grid, "bpm": bpm, "duration": duration},
    )
    merged = dict(chart_metrics)
    merged.update({f"timing.{key}": value for key, value in timing_metrics.items()})
    missing = sorted(retained - set(merged))
    if missing:
        raise RuntimeError(f"registered metrics missing from executable output: {missing}")
    return clean_json({name: merged[name] for name in sorted(retained)})


def _record_base(
    *,
    run_id: str,
    phase: str,
    sid: str,
    source_row_index: int,
    stored_split_row_id: str,
    manifest_source_row_id: str | None,
    canonical_song_index: int,
    group_id: int,
    audio_sha256: str,
    course: str,
    chart: Chart,
    input_hash: str,
    condition: str,
    condition_kind: str,
    dose_index: int,
    dose_value,
    replicate_index: int,
    base_seed: int | None,
    rng_seed: int | None,
) -> dict:
    return {
        "schema_version": "chartgeneval.confirmatory_raw.v1",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "run_id": run_id,
        "attempt_id": 1,
        "task_id": f"{sid}|{course}|{condition}|d{dose_index}|r{replicate_index}",
        "timestamp_utc": utc_now(),
        "sid": sid,
        "source_row_index": source_row_index,
        "stored_split_row_id": stored_split_row_id,
        "manifest_source_row_id": manifest_source_row_id,
        "canonical_song_index": canonical_song_index,
        "group_id": group_id,
        "canonical": True,
        "audio_sha256": audio_sha256,
        "course": course,
        "level": chart.level,
        "bpm": float(chart.bpm),
        "input_chart_sha256": input_hash,
        "condition": condition,
        "condition_kind": condition_kind,
        "dose_index": dose_index,
        "dose_value": dose_value,
        "replicate_index": replicate_index,
        "replicate_base_seed": base_seed,
        "rng_seed": rng_seed,
    }


def _emit_variant(handle, record: dict, *, events, grid, duration, lm, calibration, retained, noop: bool) -> str:
    try:
        metrics = _evaluate(
            events,
            bpm=record["bpm"],
            course=record["course"],
            grid=grid,
            duration=duration,
            lm=lm,
            calibration=calibration,
            retained=retained,
        )
        record.update(
            {
                "status": "success",
                "corruption_noop": bool(noop),
                "variant_events_sha256": _events_hash(events),
                "variant_n_notes": len(sorted_hits(events)),
                "metrics": metrics,
                "error": None,
            }
        )
        status = "noop" if noop else "success"
    except Exception as exc:  # preserve local failure and continue
        record.update(
            {
                "status": "error",
                "corruption_noop": bool(noop),
                "variant_events_sha256": _events_hash(events),
                "variant_n_notes": len(sorted_hits(events)),
                "metrics": {},
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        status = "error"
    append_jsonl(handle, record)
    return status


def _score_chart(
    handle,
    *,
    run_id: str,
    phase: str,
    source_row_index: int,
    stored_split_row_id: str,
    manifest_source_row_id: str | None,
    canonical_song_index: int,
    group_id: int,
    audio_sha256: str,
    sid: str,
    course: str,
    chart: Chart,
    course_struct: dict,
    lm,
    calibration,
    contract: dict,
) -> dict[str, int]:
    counts = {"records": 0, "success": 0, "noop": 0, "error": 0}
    events = sorted_hits(chart.events)
    input_hash = _chart_input_hash(course, chart, course_struct)
    segments = copy.deepcopy((course_struct or {}).get("segments") or [])
    grid = {"segments": segments} if segments else None
    duration = (events[-1][0] + 1.0) if events else 0.0
    retained = {
        p["metric"] for p in contract["primary_pairs"]
    } | {m["metric"] for m in contract["secondary_metrics"]}
    for control in contract["controls"].values():
        retained.update(control["metrics"])

    def emit(record, variant_events, variant_grid, variant_duration, noop=False):
        outcome = _emit_variant(
            handle,
            record,
            events=variant_events,
            grid=variant_grid,
            duration=variant_duration,
            lm=lm,
            calibration=calibration,
            retained=retained,
            noop=noop,
        )
        counts["records"] += 1
        counts[outcome] += 1

    official = _record_base(
        run_id=run_id,
        phase=phase,
        sid=sid,
        source_row_index=source_row_index,
        stored_split_row_id=stored_split_row_id,
        manifest_source_row_id=manifest_source_row_id,
        canonical_song_index=canonical_song_index,
        group_id=group_id,
        audio_sha256=audio_sha256,
        course=course,
        chart=chart,
        input_hash=input_hash,
        condition="official",
        condition_kind="official",
        dose_index=0,
        dose_value=0,
        replicate_index=0,
        base_seed=None,
        rng_seed=None,
    )
    emit(official, events, grid, duration)

    seeds = contract["replicate_base_seeds"]
    for control_id in ("CTRL_identity", "CTRL_joint_time_origin_shift", "CTRL_color_bijection"):
        for replicate_index, base_seed in enumerate(seeds, start=1):
            rng_seed = derive_rng_seed(base_seed, sid, course, control_id, 0)
            variant_events, variant_grid, variant_duration = _control_variant(
                control_id, events, grid, duration
            )
            record = _record_base(
                run_id=run_id,
                phase=phase,
                sid=sid,
                source_row_index=source_row_index,
                stored_split_row_id=stored_split_row_id,
                manifest_source_row_id=manifest_source_row_id,
                canonical_song_index=canonical_song_index,
                group_id=group_id,
                audio_sha256=audio_sha256,
                course=course,
                chart=chart,
                input_hash=input_hash,
                condition=control_id,
                condition_kind="invariant_control",
                dose_index=0,
                dose_value=0,
                replicate_index=replicate_index,
                base_seed=base_seed,
                rng_seed=rng_seed,
            )
            emit(record, variant_events, variant_grid, variant_duration)

    for probe_id, probe_contract in contract["probes"].items():
        for dose_index, dose_value in enumerate(probe_contract["dose_values"], start=1):
            for replicate_index, base_seed in enumerate(seeds, start=1):
                rng_seed = derive_rng_seed(base_seed, sid, course, probe_id, dose_index)
                rng = np.random.default_rng(rng_seed)
                variant_events, noop = apply_probe(
                    probe_id,
                    events,
                    dose_value,
                    rng,
                    chart.bpm,
                    lm=lm,
                    course=course,
                )
                noop = bool(noop) or _events_hash(variant_events) == _events_hash(events)
                record = _record_base(
                    run_id=run_id,
                    phase=phase,
                    sid=sid,
                    source_row_index=source_row_index,
                    stored_split_row_id=stored_split_row_id,
                    manifest_source_row_id=manifest_source_row_id,
                    canonical_song_index=canonical_song_index,
                    group_id=group_id,
                    audio_sha256=audio_sha256,
                    course=course,
                    chart=chart,
                    input_hash=input_hash,
                    condition=probe_id,
                    condition_kind="corruption",
                    dose_index=dose_index,
                    dose_value=dose_value,
                    replicate_index=replicate_index,
                    base_seed=base_seed,
                    rng_seed=rng_seed,
                )
                emit(record, variant_events, grid, duration, noop=noop)
    return counts


def _run(args) -> Path:
    runtime_dependency_contract()
    config, contract, config_path, contract_path = load_contract(args.config)
    phase = config["phase"]
    if phase == "confirmatory":
        if not args.confirm_untouched_panel:
            raise SystemExit("confirmatory run requires --confirm-untouched-panel")
        if not args.freeze:
            raise SystemExit("confirmatory run requires --freeze")
        configured_freeze = resolve_repo_path(config["freeze_path"])
        supplied_freeze = resolve_repo_path(args.freeze)
        if supplied_freeze != configured_freeze:
            raise SystemExit("--freeze must equal the unique path locked in confirmatory.json")
        freeze = verify_freeze(supplied_freeze, config_path)
    else:
        if args.confirm_untouched_panel:
            raise SystemExit("--confirm-untouched-panel is valid only for the frozen panel")
        freeze = None

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        f"{phase}-%Y%m%dT%H%M%SZ"
    )
    if run_id in {".", ".."} or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) is None:
        raise SystemExit("--run-id must be one safe path segment (letters, digits, '.', '_', '-')")
    raw_root = resolve_repo_path(config["raw_root"])
    run_dir = raw_root / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_dir}")
    if phase == "confirmatory":
        anchor, anchor_path = _ensure_confirmatory_anchor(
            config, freeze, supplied_freeze, raw_root, run_id
        )
    else:
        anchor, anchor_path = None, None
    run_dir.mkdir(parents=True, exist_ok=False)
    records_path = run_dir / "records.jsonl"
    context_path = run_dir / "sampling_context.jsonl"
    source_scan_path = run_dir / "source_scan.jsonl"
    events_path = run_dir / "manifest_events.jsonl"

    # Immutable snapshots remove reliance on later edits to the source configs.
    snapshots = run_dir / "spec_snapshots"
    _write_text_exclusive(snapshots / "config.json", config_path.read_text(encoding="utf-8"))
    _write_text_exclusive(snapshots / "contract.json", contract_path.read_text(encoding="utf-8"))
    _write_text_exclusive(
        snapshots / "SPEC.md",
        (resolve_repo_path("experiments/confirmatory_holdout_v1/SPEC.md")).read_text(
            encoding="utf-8"
        ),
    )
    if freeze is not None:
        _write_text_exclusive(
            snapshots / "freeze.json", supplied_freeze.read_text(encoding="utf-8")
        )
    if anchor is not None:
        _write_text_exclusive(
            snapshots / "confirmatory_anchor.json", anchor_path.read_text(encoding="utf-8")
        )

    snapshot_files = sorted(path for path in snapshots.iterdir() if path.is_file())
    snapshot_hashes = {path.name: file_sha256(path) for path in snapshot_files}

    snapshot = source_snapshot(contract, config_path)
    state = {
        "records": 0,
        "success": 0,
        "noop": 0,
        "error": 0,
        "n_scored_charts": 0,
        "n_train_rows": 0,
        "n_train_charts": 0,
        "scanned_source_rows": 0,
        "filtered_alias_rows": 0,
        "canonical_context_rows": 0,
        "lm_fingerprints": None,
        "selected_group_ids": [],
        "selected_audio_sha256": [],
        "selected_panel_fingerprint_material": [],
        "sampling_context_rows": 0,
        "datasets_fingerprints": {},
        "sampling_context_fingerprint_material": [],
    }
    fatal = None
    started = utc_now()
    with records_path.open("x", encoding="utf-8") as records_handle, context_path.open(
        "x", encoding="utf-8"
    ) as context_handle, source_scan_path.open(
        "x", encoding="utf-8"
    ) as source_scan_handle, events_path.open("x", encoding="utf-8") as events_handle:
        append_jsonl(
            events_handle,
            {
                "event": "run_started",
                "timestamp_utc": started,
                "experiment_id": EXPERIMENT_ID,
                "phase": phase,
                "run_id": run_id,
                "invocation": invocation(),
            },
        )
        try:
            lm, calibration, state["lm_fingerprints"] = _prepare_models(
                config, contract, state["datasets_fingerprints"]
            )
            state["n_train_rows"] = state["lm_fingerprints"]["n_train_rows"]
            state["n_train_charts"] = state["lm_fingerprints"]["n_train_charts"]
            dataset = config["dataset"]
            start = dataset["canonical_song_index_start_inclusive"]
            stop = dataset["canonical_song_index_stop_exclusive"]
            context_start = dataset["sampling_context_canonical_start_inclusive"]
            context_stop = dataset["sampling_context_canonical_stop_exclusive"]
            expected_canonical_set = None
            canonical_manifest_ids = None
            if dataset["source"] == "synthetic":
                rows = _synthetic_rows(dataset["expected_source_rows"], "synthetic_test")
            else:
                test_entries = _clean_split_entries(config, contract, dataset["split"])
                expected_canonical_set = {
                    (int(entry["group_id"]), str(entry["audio_sha256"]))
                    for _source_row_id, entry in test_entries
                    if bool(entry["canonical"])
                }
                canonical_manifest_ids = _canonical_manifest_id_map(test_entries)
                rows = _hf_rows(
                    dataset,
                    dataset["split"],
                    _identity_counter(test_entries),
                    require_exhaustion=bool(dataset["scan_full_split"]),
                    stop=None,
                )
            if dataset["source"] == "synthetic":
                state["datasets_fingerprints"]["synthetic"] = {
                    "kind": "ordered_canonical_chart_content_v1",
                    "dataset_revision": dataset["revision"],
                }
            observed_canonical_set = set()
            source_scan_rows = []
            panel_rows = _iter_source_panel(
                rows,
                context_stop=context_stop,
                scan_full_split=bool(dataset["scan_full_split"]),
                canonical_manifest_ids=canonical_manifest_ids,
            )
            for scan_row, row in panel_rows:
                source_scan_rows.append(scan_row)
                append_jsonl(source_scan_handle, scan_row)
                state["scanned_source_rows"] += 1
                is_canonical = bool(scan_row["canonical"])
                group_id = int(scan_row["group_id"])
                audio_sha256 = str(scan_row["audio_sha256"])
                identity = (group_id, audio_sha256)
                if not is_canonical:
                    state["filtered_alias_rows"] += 1
                    continue
                source_row_index = int(scan_row["source_row_index"])
                stored_split_row_id = str(scan_row["stored_split_row_id"])
                manifest_source_row_id = scan_row["manifest_source_row_id"]
                canonical_song_index = int(scan_row["canonical_song_index"])
                observed_canonical_set.add(identity)
                if canonical_song_index < context_start:
                    continue
                charts = list(_official_charts(row))
                if phase == "confirmatory":
                    panel_group = (
                        "development" if canonical_song_index < 40 else "confirmatory"
                    )
                elif phase == "smoke_development":
                    panel_group = "development_smoke"
                else:
                    panel_group = "smoke"
                context_record = _sampling_context_record(
                    source_row_index=source_row_index,
                    stored_split_row_id=stored_split_row_id,
                    manifest_source_row_id=manifest_source_row_id,
                    canonical_song_index=canonical_song_index,
                    row=row,
                    panel_group=panel_group,
                    charts=charts,
                )
                append_jsonl(context_handle, context_record)
                state["sampling_context_rows"] += 1
                state["canonical_context_rows"] += 1
                state["sampling_context_fingerprint_material"].append(
                    {
                        "source_row_index": source_row_index,
                        "stored_split_row_id": stored_split_row_id,
                        "manifest_source_row_id": manifest_source_row_id,
                        "canonical_song_index": canonical_song_index,
                        "group_id": group_id,
                        "audio_sha256": audio_sha256,
                        "chart_input_sha256": [
                            chart["chart_input_sha256"] for chart in context_record["charts"]
                        ],
                    }
                )
                if start <= canonical_song_index < stop:
                    sid = f"group_{group_id}"
                    state["selected_group_ids"].append(group_id)
                    state["selected_audio_sha256"].append(audio_sha256)
                    state["selected_panel_fingerprint_material"].append(
                        {
                            "source_row_index": source_row_index,
                            "stored_split_row_id": stored_split_row_id,
                            "manifest_source_row_id": manifest_source_row_id,
                            "canonical_song_index": canonical_song_index,
                            "group_id": group_id,
                            "audio_sha256": audio_sha256,
                            "chart_input_sha256": [
                                c["chart_input_sha256"] for c in context_record["charts"]
                            ],
                        }
                    )
                    for course, chart, course_struct in charts:
                        chart_counts = _score_chart(
                            records_handle,
                            run_id=run_id,
                            phase=phase,
                            source_row_index=source_row_index,
                            stored_split_row_id=stored_split_row_id,
                            manifest_source_row_id=manifest_source_row_id,
                            canonical_song_index=canonical_song_index,
                            group_id=group_id,
                            audio_sha256=audio_sha256,
                            sid=sid,
                            course=course,
                            chart=chart,
                            course_struct=course_struct,
                            lm=lm,
                            calibration=calibration,
                            contract=contract,
                        )
                        for key, value in chart_counts.items():
                            state[key] += value
                        state["n_scored_charts"] += 1
            scan_counts = _validate_source_scan_contract(
                source_scan_rows,
                dataset,
                expected_canonical_set=expected_canonical_set,
            )
            for key, observed in scan_counts.items():
                if int(state[key]) != int(observed):
                    raise RuntimeError(f"source scan state drift for {key}")
            if len(state["selected_group_ids"]) != dataset[
                "expected_selected_canonical_songs"
            ]:
                raise RuntimeError(
                    f"selected {len(state['selected_group_ids'])} canonical songs; "
                    f"expected {dataset['expected_selected_canonical_songs']}"
                )
            if state["sampling_context_rows"] != context_stop - context_start:
                raise RuntimeError("sampling-context panel is incomplete")
            if len(set(state["selected_group_ids"])) != len(state["selected_group_ids"]):
                raise RuntimeError("selected group_id values are not unique")
            if len(set(state["selected_audio_sha256"])) != len(
                state["selected_audio_sha256"]
            ):
                raise RuntimeError("selected audio_sha256 values are not unique")
            if dataset["scan_full_split"]:
                if expected_canonical_set is not None and observed_canonical_set != expected_canonical_set:
                    raise RuntimeError("HF canonical context differs from clean split test set")
            if dataset["source"] == "huggingface":
                state["datasets_fingerprints"][dataset["split"]] = {
                    "kind": "ordered_canonical_chart_content_v1",
                    "dataset_revision": dataset["revision"],
                    "split": dataset["split"],
                    "range": [context_start, context_stop],
                    "scanned_source_rows": state["scanned_source_rows"],
                    "filtered_alias_rows": state["filtered_alias_rows"],
                    "n_canonical_songs": state["sampling_context_rows"],
                    "source_scan_kind": "ordered_raw_source_identity_v1",
                    "source_scan_sha256": sha256_text(stable_json(source_scan_rows)),
                    "sha256": sha256_text(
                        stable_json(state["sampling_context_fingerprint_material"])
                    ),
                }
        except BaseException as exc:
            fatal = exc
            append_jsonl(
                events_handle,
                {
                    "event": "run_failed",
                    "timestamp_utc": utc_now(),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
        else:
            append_jsonl(
                events_handle,
                {
                    "event": "run_completed",
                    "timestamp_utc": utc_now(),
                    "local_error_records": state["error"],
                },
            )

    selected_group_ids = state.pop("selected_group_ids")
    selected_audio_sha256 = state.pop("selected_audio_sha256")
    fingerprint_material = state.pop("selected_panel_fingerprint_material")
    state.pop("sampling_context_fingerprint_material")
    lm_fingerprints = state.pop("lm_fingerprints")
    datasets_fingerprints = state.pop("datasets_fingerprints")
    status = (
        "failed"
        if fatal is not None
        else ("completed_with_errors" if state["error"] else "completed")
    )
    manifest = {
        "schema_version": "chartgeneval.confirmatory_run_manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "run_id": run_id,
        "attempt_id": 1,
        "status": status,
        "started_at_utc": started,
        "ended_at_utc": utc_now(),
        "execution_script": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "execution_command": invocation(),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": file_sha256(config_path),
        "contract_path": str(contract_path.relative_to(REPO_ROOT)),
        "contract_sha256": file_sha256(contract_path),
        "freeze_path": (
            str(resolve_repo_path(args.freeze).relative_to(REPO_ROOT)) if args.freeze else None
        ),
        "freeze_sha256": file_sha256(resolve_repo_path(args.freeze)) if args.freeze else None,
        "confirmatory_anchor_path": (
            str(anchor_path.relative_to(REPO_ROOT)) if anchor_path is not None else None
        ),
        "confirmatory_anchor_sha256": file_sha256(anchor_path) if anchor_path is not None else None,
        "spec_snapshot_hashes": snapshot_hashes,
        "code_version": snapshot,
        "git_state": git_state(),
        "runtime_environment": _runtime_environment(),
        "runtime_dependency_contract": runtime_dependency_contract(),
        "dataset": config["dataset"],
        "dataset_revision": config["dataset"]["revision"],
        "clean_split_manifest_path": config.get("clean_split_manifest_path"),
        "clean_split_manifest_sha256": (
            file_sha256(resolve_repo_path(config["clean_split_manifest_path"]))
            if config.get("clean_split_manifest_path")
            else None
        ),
        "datasets_fingerprints": datasets_fingerprints,
        "language_model_contract": contract["language_model"],
        "language_model_fingerprints": lm_fingerprints,
        "cluster_unit": "group_id",
        "selected_group_ids_ordered": selected_group_ids,
        "selected_group_ids_ordered_sha256": sha256_text(
            stable_json(selected_group_ids)
        ),
        "selected_audio_sha256_ordered": selected_audio_sha256,
        "selected_audio_sha256_ordered_sha256": sha256_text(
            stable_json(selected_audio_sha256)
        ),
        "selected_panel_fingerprint_sha256": sha256_text(
            stable_json(fingerprint_material)
        ),
        "raw_paths": [
            "records.jsonl",
            "sampling_context.jsonl",
            "source_scan.jsonl",
            "manifest_events.jsonl",
        ],
        "raw_hashes": {
            "records.jsonl": file_sha256(records_path),
            "sampling_context.jsonl": file_sha256(context_path),
            "source_scan.jsonl": file_sha256(source_scan_path),
            "manifest_events.jsonl": file_sha256(events_path),
        },
        "counts": state,
        "fatal_error": (
            {"type": type(fatal).__name__, "message": str(fatal)} if fatal is not None else None
        ),
    }
    write_json_exclusive(run_dir / "MANIFEST.json", manifest)
    print(stable_json({"run_dir": str(run_dir), "status": status, "counts": state}))
    if fatal is not None:
        raise fatal
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--freeze")
    parser.add_argument("--confirm-untouched-panel", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    _run(args)


if __name__ == "__main__":
    main()
