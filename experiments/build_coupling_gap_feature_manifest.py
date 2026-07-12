#!/usr/bin/env python3
"""Freeze the cached mel tensors used by suite-v2 development evidence.

The manifest binds each tensor to the pinned dataset song identity and audio
digest.  It records the extraction contract and source hashes, while explicitly
not claiming that the historical dependency environment can be reconstructed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _coupling_feature_contract import (  # noqa: E402
    EXPERIMENT_ID,
    EXTRACTION_CONTRACT,
    FEATURE_SCHEMA,
    PANEL_ID,
    canonical_sha256,
    sha256,
)
from _dataset import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    iter_canonical_rows,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments/suite_v2_development/SPEC.md"


def _load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _unique_rows(rows: list[dict], key: str, label: str) -> dict[str, dict]:
    output = {}
    for row in rows:
        value = str(row[key])
        if value in output:
            raise ValueError(f"duplicate {label} {key}: {value}")
        output[value] = row
    return output


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--limit-songs", type=int, default=40)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--audio-manifest", type=Path, required=True)
    parser.add_argument("--extractor-script", type=Path, required=True)
    parser.add_argument("--preprocess-source", type=Path, required=True)
    parser.add_argument("--vocab-source", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.limit_songs != 40:
        raise SystemExit("the release feature manifest requires exactly 40 songs")
    for path in (
        args.features,
        args.audio_manifest,
        args.extractor_script,
        args.preprocess_source,
        args.vocab_source,
        args.spec,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    audio_payload = _load_json(args.audio_manifest)
    if not isinstance(audio_payload, list):
        raise ValueError("audio manifest must be a JSON list")
    audio = _unique_rows(audio_payload, "sid", "audio-manifest")

    index_path = args.features / "index.json"
    report_path = args.features / "run_report.json"
    index_payload = _load_json(index_path)
    report_payload = _load_json(report_path)
    if not isinstance(index_payload, list):
        raise ValueError("feature index must be a JSON list")
    index = _unique_rows(index_payload, "id", "feature-index")

    dataset = {}
    for sid, row in iter_canonical_rows(
        args.dataset,
        args.split,
        args.limit_songs,
        revision=args.revision,
    ):
        dataset[sid] = {
            "group_id": int(row["group_id"]),
            "audio_sha256": str(row["audio_sha256"]),
        }
    expected_sids = set(dataset)
    if len(expected_sids) != 40:
        raise RuntimeError(f"expected 40 canonical songs, observed {len(expected_sids)}")
    if set(audio) != expected_sids:
        raise ValueError("audio manifest sid set does not match the development panel")
    if set(index) != expected_sids:
        raise ValueError("feature index sid set does not match the development panel")
    if report_payload != {"ok": sorted(expected_sids), "fail": []}:
        raise ValueError("feature run report is not an exact all-success panel report")

    rows = []
    for sid in sorted(expected_sids):
        audio_row = audio[sid]
        dataset_row = dataset[sid]
        observed_identity = {
            "group_id": int(audio_row["group_id"]),
            "audio_sha256": str(audio_row["audio_sha256"]),
        }
        if observed_identity != dataset_row:
            raise ValueError(
                f"{sid}: dataset/audio-manifest identity mismatch: "
                f"{dataset_row!r} != {observed_identity!r}"
            )
        mel_path = args.features / f"{sid}.mel.npy"
        if not mel_path.is_file():
            raise FileNotFoundError(mel_path)
        mel = np.load(mel_path, mmap_mode="r", allow_pickle=False)
        if mel.ndim != 2 or mel.shape[0] != 128 or mel.shape[1] <= 0:
            raise ValueError(f"{sid}: mel shape must be [128, positive n_frames]")
        if str(mel.dtype) != "float16":
            raise ValueError(f"{sid}: mel dtype must be float16")
        if not bool(np.isfinite(mel).all()):
            raise ValueError(f"{sid}: mel contains non-finite values")
        n_frames = int(mel.shape[1])
        if int(index[sid]["n_frames"]) != n_frames:
            raise ValueError(f"{sid}: index n_frames does not match the mel tensor")
        rows.append(
            {
                "sid": sid,
                "group_id": dataset_row["group_id"],
                "audio_sha256": dataset_row["audio_sha256"],
                "audio_bytes": int(audio_row["bytes"]),
                "mel_file": mel_path.name,
                "mel_sha256": sha256(mel_path),
                "mel_bytes": mel_path.stat().st_size,
                "mel_shape": [int(mel.shape[0]), n_frames],
                "mel_dtype": str(mel.dtype),
                "n_frames": n_frames,
                "all_finite": True,
            }
        )

    payload = {
        "schema_version": FEATURE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": "development_descriptive",
        "panel_id": PANEL_ID,
        "dataset_id": args.dataset,
        "dataset_revision": args.revision,
        "split": args.split,
        "spec_sha256": sha256(args.spec),
        "audio_manifest_file": args.audio_manifest.name,
        "audio_manifest_sha256": sha256(args.audio_manifest),
        "features_index_file": index_path.name,
        "features_index_sha256": sha256(index_path),
        "features_run_report_file": report_path.name,
        "features_run_report_sha256": sha256(report_path),
        "extraction_contract": EXTRACTION_CONTRACT,
        "extraction_source_hashes": {
            "feature_extractor": sha256(args.extractor_script),
            "softchart_preprocess": sha256(args.preprocess_source),
            "softchart_vocab": sha256(args.vocab_source),
        },
        "feature_set_sha256": canonical_sha256(rows),
        "limitations": [
            "This manifest binds the cached tensors used by the released derived records.",
            "The historical extraction dependency snapshot was not recorded, so the release does not claim byte-identical end-to-end re-extraction from raw audio.",
        ],
        "rows": rows,
    }
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "rows": len(rows),
                "mel_bytes": sum(row["mel_bytes"] for row in rows),
                "feature_set_sha256": payload["feature_set_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
