#!/usr/bin/env python3
"""Corruption certification for the coupling and gap families (clean test set).

Applies all registered corruption operators at three doses to the official
test charts and computes (a) the coupling family against the song's mel
spectrogram and authored bar lines, and (b) the manifold-gap score against a
reference fitted on the training corpus. Closes the audit gap for the two
families whose inputs (audio; feature manifold) the chart-only probe run
cannot cover.

    python experiments/certify_coupling_gap.py \
        --features ../SoftChart/eval/features_clean \
        --audio-manifest ../SoftChart/eval/ext_provenance_clean/audio_manifest_clean.json \
        --out coupling_gap.jsonl

Requires the ``[data]`` extra, numpy, and the features dir (mel npy files).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    iter_canonical_rows,
    iter_rows,
    official_charts,
)
from _coupling_feature_contract import (  # noqa: E402
    EXPERIMENT_ID,
    EXTRACTION_CONTRACT,
    PANEL_ID,
    RECORD_SCHEMA,
    RUN_SCHEMA,
    load_feature_manifest,
    sha256,
    verify_mel_file,
)

from chartgeneval.metrics import coupling, gap  # noqa: E402
from chartgeneval.metrics.timing import _segment_bars  # noqa: E402
from chartgeneval.probes import PROBES, corrupt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments/suite_v2_development/SPEC.md"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    ap.add_argument("--split", choices=("test",), default="test")
    ap.add_argument("--calibration-split", choices=("train",), default="train")
    ap.add_argument("--limit-songs", type=int, default=40)
    ap.add_argument("--features", required=True)
    ap.add_argument(
        "--audio-manifest",
        required=True,
        help="JSON list binding each feature sid to group_id and audio_sha256",
    )
    ap.add_argument(
        "--feature-manifest",
        required=True,
        help="Frozen per-song mel hashes and extraction contract",
    )
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    ap.add_argument("--out", default="coupling_gap_probes.jsonl")
    ap.add_argument("--run-manifest", required=True)
    args = ap.parse_args()

    if args.limit_songs != 40:
        raise SystemExit("the released development run requires exactly 40 songs")
    out_path = Path(args.out)
    run_manifest_path = Path(args.run_manifest)
    for path in (out_path, run_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    started_at = datetime.now(timezone.utc).isoformat()
    if not args.spec.is_file():
        raise FileNotFoundError(args.spec)
    spec_sha256 = sha256(args.spec)
    audio_manifest_path = Path(args.audio_manifest)
    with audio_manifest_path.open(encoding="utf-8") as stream:
        audio_manifest_rows = json.load(stream)
    if not isinstance(audio_manifest_rows, list):
        raise ValueError("audio manifest must be a JSON list")
    audio_manifest = {}
    for row in audio_manifest_rows:
        sid = str(row["sid"])
        if sid in audio_manifest:
            raise ValueError(f"duplicate audio-manifest sid: {sid}")
        audio_manifest[sid] = {
            "group_id": int(row["group_id"]),
            "audio_sha256": str(row["audio_sha256"]),
        }
    feature_manifest_path = Path(args.feature_manifest)
    feature_manifest, feature_rows = load_feature_manifest(
        feature_manifest_path,
        expected_dataset=args.dataset,
        expected_revision=args.revision,
        expected_count=args.limit_songs,
    )
    if feature_manifest["spec_sha256"] != spec_sha256:
        raise ValueError("feature manifest was built against a different experiment spec")
    if feature_manifest["audio_manifest_sha256"] != sha256(audio_manifest_path):
        raise ValueError("audio manifest SHA-256 does not match the feature manifest")
    if set(audio_manifest) != set(feature_rows):
        raise ValueError("audio and feature manifests have different sid sets")
    if abs(EXTRACTION_CONTRACT["frames_per_second"] - coupling.FPS) > 1e-12:
        raise ValueError("cached-feature FPS does not match the coupling metric")
    feature_manifest_sha256 = sha256(feature_manifest_path)

    # Manifold reference + LM (for C5) from the training corpus.
    from _dataset import build_train_lm
    print("[ref] fitting manifold reference + LM on train", flush=True)
    lm, n_train_charts = build_train_lm(
        args.dataset,
        args.calibration_split,
        revision=args.revision,
    )
    phis_by_course = defaultdict(list)
    for sid, row in iter_rows(
        args.dataset,
        args.calibration_split,
        0,
        revision=args.revision,
    ):
        for course, chart in official_charts(row):
            vector = gap.phi(chart.events, chart.bpm)
            if vector is not None:
                phis_by_course[course].append(vector)
    manifold_refs = {
        course: gap.fit_manifold_ref(vectors, course=course)
        for course, vectors in phis_by_course.items()
    }
    ref_counts = {course: ref["n_ref"] for course, ref in manifold_refs.items()}
    print(f"[ref] fitted per course: {ref_counts}", flush=True)

    records = []
    n_songs = 0
    n_charts = 0
    for sid, row in iter_canonical_rows(
        args.dataset,
        args.split,
        args.limit_songs,
        revision=args.revision,
    ):
        n_songs += 1
        expected_audio = audio_manifest.get(sid)
        observed_audio = {
            "group_id": int(row["group_id"]),
            "audio_sha256": str(row["audio_sha256"]),
        }
        if expected_audio != observed_audio:
            raise RuntimeError(
                f"{sid}: audio-feature identity mismatch: "
                f"manifest={expected_audio!r}, dataset={observed_audio!r}"
            )
        feature_row = feature_rows.get(sid)
        if feature_row is None:
            raise RuntimeError(f"{sid}: absent from the feature manifest")
        if {
            "group_id": int(feature_row["group_id"]),
            "audio_sha256": str(feature_row["audio_sha256"]),
        } != observed_audio:
            raise RuntimeError(f"{sid}: feature manifest identity mismatch")
        mel_path = Path(args.features) / feature_row["mel_file"]
        mel = verify_mel_file(mel_path, feature_row)
        for course, chart in official_charts(row):
            n_charts += 1
            if course not in manifold_refs:
                raise RuntimeError(f"no fitted manifold reference for course {course!r}")
            bars, _beats = _segment_bars(row[course].get("segments"))
            ctx = {
                "mel": mel,
                "grid": {"downbeats": bars} if bars else None,
                "bpm": chart.bpm,
                "course": course,
                "official_manifold_ref": manifold_refs[course],
            }

            def emit(probe, dose, events, noop):
                m = {}
                m.update(coupling.density_energy_response(events, ctx))
                m.update(coupling.energy_peak_support_rate(events, ctx))
                m.update(gap.compute(events, ctx))
                m = {k: (None if isinstance(v, float) and np.isnan(v) else v)
                     for k, v in m.items()}
                primary = (
                    "density_energy_spearman",
                    "energy_support_rate_raw",
                    "manifold_gap_raw",
                    "manifold_score",
                )
                if any(m.get(key) is None or not np.isfinite(float(m[key])) for key in primary):
                    raise RuntimeError(
                        f"{sid}/{course}/{probe}/{dose}: non-finite primary measurement"
                    )
                m.update(
                    {
                        "record_schema_version": RECORD_SCHEMA,
                        "experiment_id": EXPERIMENT_ID,
                        "phase": "development_descriptive",
                        "dataset_id": args.dataset,
                        "dataset_revision": args.revision,
                        "panel_id": PANEL_ID,
                        "spec_sha256": spec_sha256,
                        "feature_manifest_sha256": feature_manifest_sha256,
                        "sid": sid,
                        "group_id": int(row["group_id"]),
                        "audio_sha256": str(row["audio_sha256"]),
                        "course": course,
                        "probe": probe,
                        "dose_index": dose,
                        "corruption_noop": bool(noop),
                        "lm_split": args.calibration_split,
                        "lm_order": lm.order,
                        "lm_alpha": lm.alpha,
                        "lm_n_train_charts": n_train_charts,
                    }
                )
                records.append(m)

            emit("official", 0, chart.events, False)
            for probe in PROBES:
                for di in (1, 2, 3):
                    ev, noop = corrupt(chart.events, probe, di, sid=sid,
                                       course=course, bpm=chart.bpm, lm=lm)
                    emit(probe, di, ev, noop)
        print(f"[{sid}] records={len(records)}", flush=True)

    expected_records = n_charts * (1 + 3 * len(PROBES))
    if len(records) != expected_records:
        raise RuntimeError(
            f"record-count mismatch: expected {expected_records}, observed {len(records)}"
        )
    cells = {
        (record["sid"], record["course"], record["probe"], record["dose_index"])
        for record in records
    }
    if len(cells) != expected_records:
        raise RuntimeError("duplicate or missing chart/probe/dose cells")
    with out_path.open("x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n")

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "experiments/_coupling_feature_contract.py",
        ROOT / "experiments/_dataset.py",
        ROOT / "src/chartgeneval/metrics/coupling.py",
        ROOT / "src/chartgeneval/metrics/gap.py",
        ROOT / "src/chartgeneval/probes.py",
    ]
    run_manifest = {
        "schema_version": RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": "development_descriptive",
        "runner": "coupling_gap",
        "started_at_utc": started_at,
        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_command": sys.argv,
        "dataset": {
            "id": args.dataset,
            "revision": args.revision,
            "split": args.split,
            "panel_id": PANEL_ID,
        },
        "spec": {"path": str(args.spec), "sha256": spec_sha256},
        "inputs": {
            "audio_manifest_sha256": sha256(audio_manifest_path),
            "feature_manifest_sha256": feature_manifest_sha256,
        },
        "language_model": {
            "split": args.calibration_split,
            "order": lm.order,
            "alpha": lm.alpha,
            "n_train_charts": n_train_charts,
        },
        "reference_counts": ref_counts,
        "counts": {
            "songs": n_songs,
            "charts": n_charts,
            "records": len(records),
            "noops": sum(bool(record["corruption_noop"]) for record in records),
        },
        "output": {
            "path": str(out_path),
            "sha256": sha256(out_path),
            "record_schema_version": RECORD_SCHEMA,
        },
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in source_paths
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "datasets": version("datasets"),
            "chartgeneval": version("chartgeneval"),
        },
    }
    with run_manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(run_manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "out": str(out_path),
                "run_manifest": str(run_manifest_path),
                "n_songs": n_songs,
                "n_charts": n_charts,
                "n_records": len(records),
                "reference_counts": ref_counts,
                "audio_manifest_sha256": sha256(audio_manifest_path),
                "feature_manifest_sha256": feature_manifest_sha256,
                "spec_sha256": spec_sha256,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
