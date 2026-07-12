"""External-system course mappings are explicit and fail closed."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from _system_conditions import (  # noqa: E402
    COURSE_MAPPING_VERSION,
    load_reference_course_maps,
    pick_reference_course,
)

ROOT = Path(__file__).resolve().parents[1]


def test_declared_reference_course_maps():
    maps = load_reference_course_maps()
    available = {"easy", "normal", "hard", "oni", "ura"}
    assert pick_reference_course("mapperatorinator", "normal", available, maps) == "normal"
    assert pick_reference_course("genelive", "beginner", available, maps) == "easy"
    assert pick_reference_course("genelive", "challenge", available, maps) == "ura"
    assert pick_reference_course("genelive", "challenge", available - {"ura"}, maps) == "oni"
    assert pick_reference_course("taikonation", None, available, maps) == "oni"


def test_reference_course_mapping_rejects_undeclared_course():
    maps = load_reference_course_maps()
    with pytest.raises(ValueError, match="no reference-course candidates"):
        pick_reference_course("genelive", "unknown", {"oni"}, maps)


@pytest.mark.parametrize(
    ("relative_path", "expected_rows", "expected_challenge_courses"),
    [
        (
            "artifacts/records/system_timing_clean.jsonl",
            1320,
            Counter({"oni": 60, "ura": 20}),
        ),
        (
            "artifacts/records/system_coupling_clean.jsonl",
            660,
            Counter({"oni": 30, "ura": 10}),
        ),
    ],
)
def test_released_system_records_use_declared_course_mapping(
    relative_path: str,
    expected_rows: int,
    expected_challenge_courses: Counter,
):
    maps = load_reference_course_maps()
    records = []
    with (ROOT / relative_path).open() as stream:
        for line in stream:
            if not line.strip():
                continue
            records.append(json.loads(line))

    available_by_sid = defaultdict(set)
    for row in records:
        if row["system"] == "official":
            available_by_sid[row["sid"]].add(row["course"])

    for row in records:
        assert row["course_mapping_version"] == COURSE_MAPPING_VERSION
        expected = pick_reference_course(
            row["system"],
            row["course"],
            available_by_sid[row["sid"]],
            maps,
        )
        assert row["grid_course"] == expected

    challenge_courses = Counter(
        row["grid_course"]
        for row in records
        if row["system"] == "genelive" and row["course"] == "challenge"
    )
    assert challenge_courses == expected_challenge_courses
    assert len(records) == expected_rows
