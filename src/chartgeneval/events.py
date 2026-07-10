"""Event model and loaders for rhythm-game charts.

A chart, for evaluation purposes, is a time-ordered sequence of *hit* events.
Everything downstream (metrics, probes, calibration) consumes the same minimal
representation: a list of ``(time_seconds, class)`` tuples plus a tempo. Rolls,
balloons and end markers are intentionally dropped -- the metric suite scores
the played hit stream, not span notation.

Supported hit classes (Taiko no Tatsujin colour x size):

    don, ka, don_big, ka_big

Loaders
-------
``load_events_json``
    Native JSON format used by the package and its reproduction scripts. Two
    shapes are accepted:

    1. A flat list of events::

           [{"t": 0.5, "type": "don"}, {"t": 1.0, "type": "ka"}, ...]

    2. An object with a ``hits`` list (mirrors the sample dumps that generation
       systems emit)::

           {"bpm": 160.0, "course": "oni", "hits": [{"t": 0.5, "type": "don"}]}

    The object form may also carry ``bpm``, ``course`` and ``level`` metadata,
    returned alongside the events as a :class:`Chart`.

``load_taiko_parsed_course``
    Extracts hit events from one course of a ``taiko-1000-parsed`` dataset row
    (the ``segments``/``notes`` nested structure). This mirrors the extraction
    the golden pipeline used, so scores computed here reproduce the reference.

TJA / osu parsing are deliberately left as stubs (:func:`load_events_tja`,
:func:`load_events_osu`); they are format front-ends that were never part of the
evaluation golden path, and shipping a half-correct parser would be dishonest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

HIT_CLASSES: tuple[str, ...] = ("don", "ka", "don_big", "ka_big")
COURSES: tuple[str, ...] = ("easy", "normal", "hard", "oni", "ura")

# Note-type strings observed in taiko-1000-parsed mapped onto canonical classes.
# Only hit classes survive as events; spans/markers map to None (skipped from the
# event stream). The full set of *recognised* note types (below) still
# contributes its per-note BPM to the tempo median -- this matches the reference
# extraction, where a note's tempo counts toward the chart tempo even when the
# note itself is a roll/balloon/end marker rather than a hit.
NOTE_TYPE_MAP: dict[str, str | None] = {
    "Don": "don",
    "Ka": "ka",
    "DonBig": "don_big",
    "KaBig": "ka_big",
    "Roll": None,
    "RollBig": None,
    "Balloon": None,
    "BalloonAlt": None,
    "EndOf": None,
}

# Note types recognised by the reference parser; a recognised note's BPM is
# folded into the chart tempo median even if the note is not a hit.
_RECOGNISED_NOTE_TYPES: frozenset[str] = frozenset(
    {"Don", "Ka", "DonBig", "KaBig", "Roll", "RollBig", "Balloon", "BalloonAlt", "EndOf"}
)

Event = tuple[float, str]


@dataclass
class Chart:
    """A hit stream plus the context needed to score it."""

    events: list[Event]
    bpm: float = 0.0
    course: str | None = None
    level: int | None = None
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.events)


def sorted_hits(events: Iterable[Event]) -> list[Event]:
    """Return time-sorted ``(t, class)`` pairs, keeping only real hit classes."""
    out = [(float(t), str(c)) for t, c in events if str(c) in HIT_CLASSES]
    out.sort()
    return out


def _events_from_hits(hits: Iterable[dict]) -> list[Event]:
    events: list[Event] = []
    for h in hits or []:
        cls = h.get("type")
        if cls in HIT_CLASSES:
            events.append((float(h["t"]), cls))
    return sorted_hits(events)


def load_events_json(path: str | Path) -> Chart:
    """Load a chart from the native JSON event format.

    Accepts either a flat event list or an object with a ``hits`` array. See the
    module docstring for the exact shapes.
    """
    with open(path) as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return Chart(events=_events_from_hits(payload))

    if not isinstance(payload, dict):
        raise ValueError(f"unsupported JSON chart payload: {type(payload).__name__}")

    # Sample dumps nest the played chart under "gen"; unwrap it if present.
    body = payload.get("gen") if isinstance(payload.get("gen"), dict) else payload
    hits = body.get("hits")
    if hits is None and isinstance(payload.get("hits"), list):
        hits = payload["hits"]
    if hits is None:
        raise ValueError("JSON chart object has no 'hits' array")

    bpm = payload.get("bpm") or body.get("bpm") or 0.0
    return Chart(
        events=_events_from_hits(hits),
        bpm=float(bpm) if bpm else 0.0,
        course=payload.get("course") or body.get("course"),
        level=payload.get("level") or body.get("level"),
        meta={k: v for k, v in payload.items() if k not in ("hits", "gen")},
    )


def load_taiko_parsed_course(course_struct: dict | None) -> Chart | None:
    """Extract hit events from one course of a ``taiko-1000-parsed`` row.

    ``course_struct`` is the nested ``{"segments": [...], "level": int}`` object.
    Tempo is the median of the per-note BPM values (constant in the vast majority
    of charts). Returns ``None`` when the course carries no notes.
    """
    if not course_struct or not course_struct.get("segments"):
        return None
    times: list[float] = []
    classes: list[str] = []
    bpms: list[float] = []
    for seg in course_struct["segments"]:
        for n in seg.get("notes") or []:
            nt = n.get("note_type")
            if nt not in _RECOGNISED_NOTE_TYPES:
                continue
            # tempo median folds in every recognised note (hit or span/marker).
            if n.get("bpm"):
                bpms.append(float(n["bpm"]))
            canon = NOTE_TYPE_MAP.get(nt)
            if canon not in HIT_CLASSES:
                continue
            times.append(float(n["timestamp"]))
            classes.append(canon)
    if not times:
        return None
    events = sorted_hits(zip(times, classes))
    return Chart(
        events=events,
        bpm=float(np.median(bpms)) if bpms else 0.0,
        level=int(course_struct.get("level") or 0),
    )


def load_events_tja(path: str | Path) -> Chart:  # pragma: no cover - stub
    """STUB: parse a ``.tja`` file into a :class:`Chart`.

    Not implemented. TJA parsing (BPM/MEASURE/BRANCH/GOGO command handling and
    slot->time conversion) is a format front-end that was never part of the
    evaluation golden path. Convert your charts to the native JSON event format
    with an external TJA parser and use :func:`load_events_json`.
    """
    raise NotImplementedError(
        "TJA parsing is not implemented; convert to native JSON events "
        "(see load_events_json) using an external parser."
    )


def load_events_osu(path: str | Path) -> Chart:  # pragma: no cover - stub
    """STUB: parse an osu!taiko ``.osu`` file into a :class:`Chart`.

    Not implemented for the same reason as :func:`load_events_tja`.
    """
    raise NotImplementedError(
        "osu parsing is not implemented; convert to native JSON events "
        "(see load_events_json) using an external parser."
    )
