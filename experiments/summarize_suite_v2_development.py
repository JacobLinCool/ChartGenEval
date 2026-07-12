#!/usr/bin/env python3
"""Summarize development responses for ChartGenEval's project-defined metrics.

The summary is descriptive.  It computes one within-chart Spearman response
over intact plus three doses, then averages those chart-level values.  It also
reports the oriented maximum-dose change and the fraction of charts that improve
minus the fraction that worsen.  No confidence interval or held-out verdict is
attached to these development statistics.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _coupling_feature_contract import (  # noqa: E402
    EXPERIMENT_ID,
    PANEL_ID,
    RECORD_SCHEMA,
    SUMMARY_SCHEMA,
    sha256,
)
from _dataset import DEFAULT_DATASET, DEFAULT_DATASET_REVISION  # noqa: E402
from chartgeneval.probes import PROBES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = ROOT / "experiments/suite_v2_development/SPEC.md"

FIELDNAMES = [
    "metric",
    "metric_key",
    "probe",
    "orientation",
    "n_complete_charts",
    "pooled_spearman_dose",
    "mean_within_chart_spearman",
    "mean_maxdose_oriented_delta",
    "median_maxdose_raw_delta",
    "improve_minus_worsen",
    "phase",
    "spec_sha256",
    "source_sha256",
]

EXPECTED_CONDITIONS = {("official", 0)} | {
    (probe, dose) for probe in PROBES for dose in (1, 2, 3)
}

METRICS = {
    "reciprocity": {
        "source": "structure",
        "key": "reciprocity",
        "orientation": 1.0,
    },
    "stagnation_alienation": {
        "source": "structure",
        "key": "boredom_v2_raw",
        "orientation": -1.0,
    },
    "density_energy_response": {
        "source": "coupling_gap",
        "key": "density_energy_spearman",
        "orientation": 1.0,
    },
    "run_head_onset_support": {
        "source": "coupling_gap",
        "key": "energy_support_rate_raw",
        "orientation": 1.0,
    },
    "human_chart_gap": {
        "source": "coupling_gap",
        "key": "manifold_gap_raw",
        "orientation": -1.0,
    },
}

def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(value)
    return rows


def _validate_panel(rows: list[dict], label: str) -> tuple[set[tuple[str, str]], str]:
    if len(rows) != 4760:
        raise ValueError(f"{label}: expected 4,760 rows, observed {len(rows)}")
    songs = {str(row["sid"]) for row in rows}
    charts = {
        (str(row["sid"]), str(row["course"]))
        for row in rows
        if row.get("probe") == "official" and int(row.get("dose_index", -1)) == 0
    }
    if len(songs) != 40 or len(charts) != 170:
        raise ValueError(
            f"{label}: expected 40 songs/170 charts, observed {len(songs)}/{len(charts)}"
        )
    cells = set()
    spec_hashes = set()
    for row in rows:
        if row.get("record_schema_version") != RECORD_SCHEMA:
            raise ValueError(f"{label}: record schema mismatch")
        if row.get("experiment_id") != EXPERIMENT_ID:
            raise ValueError(f"{label}: experiment_id mismatch")
        if row.get("phase") != "development_descriptive":
            raise ValueError(f"{label}: phase mismatch")
        if row.get("dataset_id") != DEFAULT_DATASET:
            raise ValueError(f"{label}: dataset_id mismatch")
        if row.get("dataset_revision") != DEFAULT_DATASET_REVISION:
            raise ValueError(f"{label}: dataset_revision mismatch")
        if row.get("panel_id") != PANEL_ID:
            raise ValueError(f"{label}: panel_id mismatch")
        if not isinstance(row.get("corruption_noop"), bool):
            raise ValueError(f"{label}: corruption_noop must be boolean")
        probe = str(row["probe"])
        dose = int(row["dose_index"])
        if (probe, dose) not in EXPECTED_CONDITIONS:
            raise ValueError(f"{label}: unexpected condition {(probe, dose)!r}")
        cell = (str(row["sid"]), str(row["course"]), probe, dose)
        if cell in cells:
            raise ValueError(f"{label}: duplicate cell {cell!r}")
        cells.add(cell)
        spec_hashes.add(str(row.get("spec_sha256")))
    expected_cells = {
        (sid, course, probe, dose)
        for sid, course in charts
        for probe, dose in EXPECTED_CONDITIONS
    }
    if cells != expected_cells:
        missing = sorted(expected_cells - cells)[:3]
        extra = sorted(cells - expected_cells)[:3]
        raise ValueError(
            f"{label}: Cartesian panel mismatch; missing={missing}, extra={extra}"
        )
    if len(spec_hashes) != 1:
        raise ValueError(f"{label}: records do not share one spec_sha256")
    spec_sha256 = next(iter(spec_hashes))
    if len(spec_sha256) != 64:
        raise ValueError(f"{label}: invalid spec_sha256")
    return charts, spec_sha256


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _summaries(
    rows: list[dict],
    metric_name: str,
    metric: dict,
    source_sha256: str,
    spec_sha256: str,
) -> list[dict]:
    key = metric["key"]
    orientation = float(metric["orientation"])
    official = {}
    by_probe = defaultdict(dict)
    for row in rows:
        chart = (str(row["sid"]), str(row["course"]))
        probe = str(row["probe"])
        dose = int(row["dose_index"])
        if probe == "official" and dose == 0:
            official[chart] = row.get(key)
        elif dose in (1, 2, 3):
            by_probe[probe].setdefault(chart, {})[dose] = row.get(key)

    output = []
    for probe in sorted(by_probe):
        chart_rhos = []
        raw_deltas = []
        oriented_deltas = []
        pooled_doses = []
        pooled_values = []
        for chart, doses in by_probe[probe].items():
            values = [official.get(chart), doses.get(1), doses.get(2), doses.get(3)]
            if not all(_finite(value) for value in values):
                continue
            raw = np.asarray(values, dtype=float)
            oriented = raw * orientation
            pooled_doses.extend((0.0, 1.0, 2.0, 3.0))
            pooled_values.extend(oriented.tolist())
            if np.all(oriented == oriented[0]):
                rho = 0.0
            else:
                rho = float(spearmanr([0.0, 1.0, 2.0, 3.0], oriented).statistic)
            chart_rhos.append(rho)
            raw_deltas.append(float(raw[-1] - raw[0]))
            oriented_deltas.append(float(oriented[-1] - oriented[0]))

        oriented_array = np.asarray(oriented_deltas, dtype=float)
        improve_minus_worsen = float(
            np.mean(oriented_array > 0.0) - np.mean(oriented_array < 0.0)
        )
        output.append(
            {
                "metric": metric_name,
                "metric_key": key,
                "probe": probe,
                "orientation": "higher_is_better",
                "n_complete_charts": len(chart_rhos),
                "pooled_spearman_dose": float(
                    spearmanr(pooled_doses, pooled_values).statistic
                ),
                "mean_within_chart_spearman": float(np.mean(chart_rhos)),
                "mean_maxdose_oriented_delta": float(np.mean(oriented_array)),
                "median_maxdose_raw_delta": float(np.median(raw_deltas)),
                "improve_minus_worsen": improve_minus_worsen,
                "phase": "development_descriptive",
                "spec_sha256": spec_sha256,
                "source_sha256": source_sha256,
            }
        )
    return output


def build_summaries(
    structure_path: Path, coupling_gap_path: Path
) -> tuple[list[dict], str]:
    sources = {
        "structure": (structure_path, _load(structure_path)),
        "coupling_gap": (coupling_gap_path, _load(coupling_gap_path)),
    }
    panels = {}
    spec_hashes = {}
    for name, (_path, rows) in sources.items():
        panels[name], spec_hashes[name] = _validate_panel(rows, name)
    if panels["structure"] != panels["coupling_gap"]:
        raise ValueError("structure and coupling/gap records use different chart panels")
    if len(set(spec_hashes.values())) != 1:
        raise ValueError("structure and coupling/gap records use different specs")
    spec_sha256 = spec_hashes["structure"]

    required_primary = {
        "structure": ("reciprocity", "boredom_v2_raw"),
        "coupling_gap": (
            "density_energy_spearman",
            "energy_support_rate_raw",
            "manifold_gap_raw",
            "manifold_score",
        ),
    }
    for source, keys in required_primary.items():
        for row in sources[source][1]:
            if any(not _finite(row.get(key)) for key in keys):
                raise ValueError(
                    f"{source}: non-finite primary measurement in "
                    f"{row.get('sid')}/{row.get('course')}/{row.get('probe')}/"
                    f"{row.get('dose_index')}"
                )

    summaries = []
    for name, metric in METRICS.items():
        path, rows = sources[metric["source"]]
        summaries.extend(
            _summaries(rows, name, metric, sha256(path), spec_sha256)
        )
    return summaries, spec_sha256


def render_summary_csv(summaries: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries)
    return stream.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--structure",
        type=Path,
        default=ROOT / "artifacts/records/structure_v2_development.jsonl",
    )
    parser.add_argument(
        "--coupling-gap",
        type=Path,
        default=ROOT / "artifacts/records/corruption_coupling_gap_clean.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "artifacts/tables/suite_v2_development_results.csv",
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.out, args.manifest):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    if not args.spec.is_file():
        raise FileNotFoundError(args.spec)
    summaries, record_spec_sha256 = build_summaries(args.structure, args.coupling_gap)
    if sha256(args.spec) != record_spec_sha256:
        raise ValueError("summary inputs were generated under a different spec")
    with args.out.open("x", encoding="utf-8", newline="") as stream:
        stream.write(render_summary_csv(summaries))
    manifest = {
        "schema_version": SUMMARY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": "development_descriptive",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_command": sys.argv,
        "spec": {"path": str(args.spec), "sha256": record_spec_sha256},
        "inputs": {
            "structure": {
                "path": str(args.structure),
                "sha256": sha256(args.structure),
            },
            "coupling_gap": {
                "path": str(args.coupling_gap),
                "sha256": sha256(args.coupling_gap),
            },
        },
        "analysis_script": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "output": {
            "path": str(args.out),
            "sha256": sha256(args.out),
            "rows": len(summaries),
            "columns": FIELDNAMES,
        },
    }
    with args.manifest.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "manifest": str(args.manifest),
                "rows": len(summaries),
                "spec_sha256": record_spec_sha256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
