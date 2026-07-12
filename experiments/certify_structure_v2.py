#!/usr/bin/env python3
"""Development certification records for ChartGenEval's new structure metrics.

Runs the frozen C1/C1s/C2--C8 corruption operators on the first 40 canonical
test-song groups and records the project-defined reciprocity and
stagnation--alienation measurements.  The output is development evidence; it
is deliberately separate from the sealed confirmatory holdout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dataset import (  # noqa: E402
    DEFAULT_DATASET,
    DEFAULT_DATASET_REVISION,
    build_train_lm,
    iter_canonical_rows,
    official_charts,
)
from _coupling_feature_contract import (  # noqa: E402
    EXPERIMENT_ID,
    PANEL_ID,
    RECORD_SCHEMA,
    RUN_SCHEMA,
    sha256,
)
from chartgeneval.metrics import structure  # noqa: E402
from chartgeneval.metrics.timing import _segment_bars  # noqa: E402
from chartgeneval.probes import PROBES, corrupt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments/suite_v2_development/SPEC.md"


def _json_value(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--revision", default=DEFAULT_DATASET_REVISION)
    parser.add_argument("--split", choices=("test",), default="test")
    parser.add_argument("--calibration-split", choices=("train",), default="train")
    parser.add_argument("--limit-songs", type=int, default=40)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--out", default="structure_v2_development.jsonl")
    parser.add_argument("--run-manifest", required=True)
    args = parser.parse_args()

    if args.limit_songs != 40:
        raise SystemExit("the released development run requires exactly 40 songs")
    out_path = Path(args.out)
    run_manifest_path = Path(args.run_manifest)
    for path in (out_path, run_manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    if not args.spec.is_file():
        raise FileNotFoundError(args.spec)
    spec_sha256 = sha256(args.spec)
    started_at = datetime.now(timezone.utc).isoformat()

    lm, n_train_charts = build_train_lm(
        args.dataset,
        args.calibration_split,
        revision=args.revision,
    )
    provenance = {
        "record_schema_version": RECORD_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": "development_descriptive",
        "dataset_id": args.dataset,
        "dataset_revision": args.revision,
        "panel_id": PANEL_ID,
        "spec_sha256": spec_sha256,
        "lm_split": args.calibration_split,
        "lm_order": lm.order,
        "lm_alpha": lm.alpha,
        "lm_n_train_charts": n_train_charts,
    }

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
        group_id = int(row["group_id"])
        title = (row.get("metadata") or {}).get("TITLE") or ""
        for course, chart in official_charts(row):
            n_charts += 1
            bars, _beats = _segment_bars(row[course].get("segments"))
            if bars is None:
                raise RuntimeError(f"{sid}/{course}: authored bar grid is required")
            ctx = {
                "bpm": chart.bpm,
                "course": course,
                "grid": {"downbeats": bars},
            }

            def emit(probe: str, dose_index: int, events, noop: bool) -> None:
                values = {}
                values.update(structure.call_response_reciprocity(events, ctx))
                values.update(structure.boredom_v2(events, ctx))
                primary = ("reciprocity", "boredom_v2_raw")
                if any(
                    values.get(key) is None
                    or not math.isfinite(float(values[key]))
                    for key in primary
                ):
                    raise RuntimeError(
                        f"{sid}/{course}/{probe}/{dose_index}: "
                        "non-finite primary structure measurement"
                    )
                record = {
                    **provenance,
                    "sid": sid,
                    "group_id": group_id,
                    "title": title,
                    "course": course,
                    "probe": probe,
                    "dose_index": dose_index,
                    "corruption_noop": bool(noop),
                    **{key: _json_value(value) for key, value in values.items()},
                }
                records.append(record)

            emit("official", 0, chart.events, False)
            for probe in PROBES:
                for dose_index in (1, 2, 3):
                    events, noop = corrupt(
                        chart.events,
                        probe,
                        dose_index,
                        sid=sid,
                        course=course,
                        bpm=chart.bpm,
                        lm=lm,
                    )
                    emit(probe, dose_index, events, noop)
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
    with out_path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            )

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "experiments/_coupling_feature_contract.py",
        ROOT / "experiments/_dataset.py",
        ROOT / "src/chartgeneval/metrics/structure.py",
        ROOT / "src/chartgeneval/probes.py",
    ]
    run_manifest = {
        "schema_version": RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": "development_descriptive",
        "runner": "structure",
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
        "language_model": {
            "split": args.calibration_split,
            "order": lm.order,
            "alpha": lm.alpha,
            "n_train_charts": n_train_charts,
        },
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
                "spec_sha256": spec_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
