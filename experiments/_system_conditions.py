"""Validated system-to-reference-course mappings for external evaluation."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CONDITIONS = (
    Path(__file__).resolve().parents[1] / "artifacts" / "systems" / "conditions.json"
)
COURSE_MAPPING_VERSION = "system_reference_course_v1"


def load_reference_course_maps(path: str | Path = DEFAULT_CONDITIONS) -> dict[str, dict[str, tuple[str, ...]]]:
    """Load ordered, explicit reference-course candidates for every system."""
    data = json.loads(Path(path).read_text())
    version = data.get("reference_course_mapping_version")
    if version != COURSE_MAPPING_VERSION:
        raise ValueError(
            f"reference-course mapping version {version!r} does not match "
            f"the runner version {COURSE_MAPPING_VERSION!r}"
        )
    maps: dict[str, dict[str, tuple[str, ...]]] = {}
    for system in data.get("systems", []):
        system_id = system.get("id")
        raw_map = (system.get("configuration") or {}).get("reference_course_map")
        if not system_id or not isinstance(raw_map, dict) or not raw_map:
            raise ValueError(f"missing reference_course_map for system {system_id!r}")
        parsed: dict[str, tuple[str, ...]] = {}
        for native_course, candidates in raw_map.items():
            if not isinstance(candidates, list) or not candidates or not all(
                isinstance(candidate, str) and candidate for candidate in candidates
            ):
                raise ValueError(
                    f"invalid reference-course candidates for {system_id!r}/{native_course!r}"
                )
            parsed[native_course] = tuple(candidates)
        maps[system_id] = parsed
    return maps


def pick_reference_course(
    system: str,
    native_course: str | None,
    available_courses,
    course_maps: dict[str, dict[str, tuple[str, ...]]],
) -> str:
    """Resolve a declared course mapping; reject undeclared or unavailable mappings."""
    available = set(available_courses)
    if system == "official":
        candidates = (native_course,) if native_course else ()
    else:
        system_map = course_maps.get(system)
        if system_map is None:
            raise ValueError(f"no reference-course map declared for system {system!r}")
        candidates = system_map.get(native_course or "*") or system_map.get("*") or ()
    if not candidates:
        raise ValueError(
            f"no reference-course candidates declared for {system!r}/{native_course!r}"
        )
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(
        f"no declared reference course is available for {system!r}/{native_course!r}; "
        f"candidates={list(candidates)!r}, available={sorted(available)!r}"
    )
