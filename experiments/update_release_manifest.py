#!/usr/bin/env python3
"""Recompute the ChartGenEval release checksum manifest.

The updater preserves curated logical metadata on existing entries, adds newly
released artifact files, and refreshes byte counts, row counts, digests, and the
total.  It writes atomically and never includes ``artifacts/MANIFEST.json`` in
its own inventory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts/MANIFEST.json"
EXTERNAL_FILES = {
    "ARTIFACT_LICENSE.md",
    "experiments/suite_v2_development/SPEC.md",
    "src/chartgeneval/data/taiko_1000_parsed_clean_calibration_v2.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_count(path: Path) -> int | None:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as stream:
            return sum(1 for _ in csv.DictReader(stream))
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        return len(value) if isinstance(value, list) else None
    return None


def main() -> None:
    with MANIFEST.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    existing = {entry["path"]: entry for entry in manifest["files"]}

    artifact_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "artifacts").rglob("*")
        if path.is_file() and path != MANIFEST
    }
    inventory = EXTERNAL_FILES | artifact_files
    missing = inventory - existing.keys()
    stale = existing.keys() - inventory
    if stale:
        raise RuntimeError(f"manifest contains files outside the release: {sorted(stale)}")

    ordered_paths = [entry["path"] for entry in manifest["files"]]
    ordered_paths.extend(sorted(missing))
    refreshed = []
    for relative in ordered_paths:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        previous = existing.get(relative, {})
        entry = {
            "path": relative,
            "bytes": path.stat().st_size,
        }
        rows = _row_count(path)
        if rows is not None:
            entry["rows"] = rows
        for key in ("logical_rows", "logical_groups"):
            if key in previous:
                entry[key] = previous[key]
        entry["sha256"] = _sha256(path)
        refreshed.append(entry)

    manifest["files"] = refreshed
    manifest["total_bytes"] = sum(entry["bytes"] for entry in refreshed)
    temporary = MANIFEST.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    os.replace(temporary, MANIFEST)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST),
                "files": len(refreshed),
                "added": sorted(missing),
                "total_bytes": manifest["total_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
