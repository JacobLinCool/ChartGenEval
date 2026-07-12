#!/usr/bin/env python3
"""Recompute canonical calibrated diagnostics from released raw chart features.

The released development and system-profile rows retain the raw features needed
to apply a scoring contract without rerunning third-party generators. This
transform removes every previously derived score and band position, recomputes
them with the bundled calibration, and records the exact scoring contract. It
never edits its input and refuses to overwrite an existing output.

Example::

    python experiments/rescore_release_profiles.py \
      raw_profile.jsonl \
      artifacts/records/system_profile_mapperatorinator.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chartgeneval.calibration import load_bundled_calibration, score_chart_features
from chartgeneval.metrics.common import SCORE_VERSION


DERIVED_FEATURES = {
    "overload_excess_nps",
    "density_spike_excess_nps",
    "pattern_chaos_excess_nll",
}


def rescore_record(record: dict, calibration: dict) -> dict:
    """Return one record under the sole current calibrated-score contract."""
    course = record.get("course")
    if not course:
        raise ValueError("record is missing a course")
    features = {
        key: value
        for key, value in record.items()
        if not key.endswith("_score")
        and not key.endswith("_band_sigma")
        and key not in DERIVED_FEATURES
        and key != "score_version"
    }
    scores = score_chart_features(features, calibration, course)
    return {**features, **scores, "score_version": SCORE_VERSION}


def rescore_file(source: Path, destination: Path) -> int:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    calibration = load_bundled_calibration()
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open() as input_stream, destination.open("x") as output_stream:
        for line_number, line in enumerate(input_stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            output_stream.write(
                json.dumps(rescore_record(record, calibration), sort_keys=True) + "\n"
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    count = rescore_file(args.source, args.destination)
    print(json.dumps({"source": str(args.source), "destination": str(args.destination), "rows": count}))


if __name__ == "__main__":
    main()
