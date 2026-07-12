"""Fail-closed contract for cached mel features used by development evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np


EXPERIMENT_ID = "suite_v2_development"
PANEL_ID = "canonical_test_0_40_development"
FEATURE_SCHEMA = "chartgeneval.coupling_gap_feature_manifest.v1"
RECORD_SCHEMA = "chartgeneval.suite_v2_development_record.v1"
RUN_SCHEMA = "chartgeneval.suite_v2_development_run.v1"
SUMMARY_SCHEMA = "chartgeneval.suite_v2_development_summary.v1"

EXTRACTION_CONTRACT = {
    "audio_decode": "soundfile.read(dtype=float32,always_2d=True)",
    "channel_mix": "arithmetic_mean_over_channels",
    "target_sample_rate_hz": 22050,
    "resample_when_needed": "librosa.resample(res_type=soxr_hq)",
    "stft_backend": "torch.stft",
    "n_fft": 2048,
    "hop_length": 256,
    "window": "hann",
    "center": True,
    "power": 2,
    "mel_backend": "librosa.filters.mel",
    "n_mels": 128,
    "fmin_hz": 20.0,
    "fmax_hz": 11025.0,
    "log_transform": "natural_log(mel_power+1e-5)",
    "output_dtype": "float16",
    "output_shape": "[128,n_frames]",
    "frames_per_second": 86.1328125,
    "historical_runtime_dependency_snapshot_available": False,
}

ROW_FIELDS = {
    "sid",
    "group_id",
    "audio_sha256",
    "audio_bytes",
    "mel_file",
    "mel_sha256",
    "mel_bytes",
    "mel_shape",
    "mel_dtype",
    "n_frames",
    "all_finite",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value, label: str) -> str:
    value = str(value)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def load_feature_manifest(
    path: Path,
    *,
    expected_dataset: str,
    expected_revision: str,
    expected_count: int = 40,
) -> tuple[dict, dict[str, dict]]:
    """Load and validate the released feature manifest without reading mel files."""

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("schema_version") != FEATURE_SCHEMA:
        raise ValueError("unsupported coupling-gap feature manifest schema")
    if payload.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("feature manifest experiment_id mismatch")
    if payload.get("panel_id") != PANEL_ID:
        raise ValueError("feature manifest panel_id mismatch")
    if payload.get("dataset_id") != expected_dataset:
        raise ValueError("feature manifest dataset_id mismatch")
    if payload.get("dataset_revision") != expected_revision:
        raise ValueError("feature manifest dataset_revision mismatch")
    if payload.get("split") != "test":
        raise ValueError("feature manifest split must be test")
    if payload.get("extraction_contract") != EXTRACTION_CONTRACT:
        raise ValueError("feature manifest extraction contract mismatch")
    _require_sha256(payload.get("spec_sha256"), "spec_sha256")
    _require_sha256(payload.get("audio_manifest_sha256"), "audio_manifest_sha256")
    _require_sha256(payload.get("features_index_sha256"), "features_index_sha256")
    _require_sha256(payload.get("features_run_report_sha256"), "features_run_report_sha256")
    source_hashes = payload.get("extraction_source_hashes")
    if not isinstance(source_hashes, dict) or set(source_hashes) != {
        "feature_extractor",
        "softchart_preprocess",
        "softchart_vocab",
    }:
        raise ValueError("feature manifest extraction_source_hashes mismatch")
    for label, value in source_hashes.items():
        _require_sha256(value, f"extraction_source_hashes.{label}")

    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(
            f"feature manifest must contain {expected_count} rows; got "
            f"{len(rows) if isinstance(rows, list) else 'non-list'}"
        )
    if rows != sorted(rows, key=lambda row: str(row.get("sid", ""))):
        raise ValueError("feature manifest rows must be sorted by sid")
    if canonical_sha256(rows) != payload.get("feature_set_sha256"):
        raise ValueError("feature_set_sha256 does not match the manifest rows")

    by_sid: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise ValueError("feature manifest row schema mismatch")
        sid = str(row["sid"])
        if sid in by_sid:
            raise ValueError(f"duplicate feature-manifest sid: {sid}")
        if not re.fullmatch(r"test_[0-9]{5}", sid):
            raise ValueError(f"invalid feature sid: {sid!r}")
        if int(row["group_id"]) < 0:
            raise ValueError(f"{sid}: group_id must be nonnegative")
        _require_sha256(row["audio_sha256"], f"{sid}.audio_sha256")
        _require_sha256(row["mel_sha256"], f"{sid}.mel_sha256")
        if int(row["audio_bytes"]) <= 0 or int(row["mel_bytes"]) <= 0:
            raise ValueError(f"{sid}: byte counts must be positive")
        if row["mel_file"] != f"{sid}.mel.npy":
            raise ValueError(f"{sid}: unexpected mel_file")
        if row["mel_dtype"] != "float16":
            raise ValueError(f"{sid}: mel_dtype must be float16")
        shape = row["mel_shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or int(shape[0]) != 128
            or int(shape[1]) <= 0
        ):
            raise ValueError(f"{sid}: mel_shape must be [128, positive n_frames]")
        if int(row["n_frames"]) != int(shape[1]):
            raise ValueError(f"{sid}: n_frames does not match mel_shape")
        if row["all_finite"] is not True:
            raise ValueError(f"{sid}: feature manifest requires all-finite mel values")
        by_sid[sid] = row
    return payload, by_sid


def verify_mel_file(path: Path, row: dict) -> np.ndarray:
    """Verify one cached tensor against the manifest before returning it."""

    if not path.is_file():
        raise FileNotFoundError(f"missing mel feature: {path}")
    if path.stat().st_size != int(row["mel_bytes"]):
        raise ValueError(f"{row['sid']}: mel byte count mismatch")
    if sha256(path) != row["mel_sha256"]:
        raise ValueError(f"{row['sid']}: mel SHA-256 mismatch")
    mel = np.load(path, allow_pickle=False)
    if list(mel.shape) != [int(value) for value in row["mel_shape"]]:
        raise ValueError(f"{row['sid']}: mel shape mismatch")
    if str(mel.dtype) != row["mel_dtype"]:
        raise ValueError(f"{row['sid']}: mel dtype mismatch")
    if not bool(np.isfinite(mel).all()):
        raise ValueError(f"{row['sid']}: mel contains a non-finite value")
    fps = EXTRACTION_CONTRACT["frames_per_second"]
    derived_fps = (
        EXTRACTION_CONTRACT["target_sample_rate_hz"]
        / EXTRACTION_CONTRACT["hop_length"]
    )
    if not math.isclose(fps, derived_fps, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("feature extraction contract has inconsistent frame rate")
    return mel
