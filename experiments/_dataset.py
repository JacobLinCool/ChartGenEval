"""Shared dataset helpers for the reproduction scripts.

Loads charts from a taiko-1000-parsed(-clean) Hugging Face dataset row and builds
the training-split n-gram model + calibration rows. Requires the ``[data]``
extra (``datasets``) and a local HF token for the gated dataset.
"""

from __future__ import annotations

import re
import json
from collections import Counter
from pathlib import Path

from chartgeneval.events import COURSES, load_taiko_parsed_course
from chartgeneval.metrics.common import NGramModel, event_tokens

DEFAULT_DATASET = "JacobLinCool/taiko-1000-parsed-clean"
DEFAULT_DATASET_REVISION = "b72da4616d643018e81f372cea06ce51349285e0"
DEFAULT_CLEAN_SPLIT_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "splits"
    / "clean_split_manifest.json"
)


def _validate_revision(revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError("dataset revision must be a 40-character lowercase commit hash")


def iter_rows(dataset_id, split, limit=0, *, revision=DEFAULT_DATASET_REVISION):
    """Stream dataset rows, yielding ``(sid, row)``. Audio is not decoded."""
    from datasets import Audio, load_dataset

    _validate_revision(revision)
    ds = load_dataset(dataset_id, revision=revision, split=split, streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        yield f"{split}_{i:05d}", row


def iter_canonical_rows(
    dataset_id,
    split,
    limit=0,
    *,
    revision=DEFAULT_DATASET_REVISION,
    manifest_path=DEFAULT_CLEAN_SPLIT_MANIFEST,
):
    """Yield canonical rows in pinned stored order after filtering aliases.

    The clean-split manifest is treated as an identity multiset, not as an
    ordering oracle.  Stored order comes only from the pinned dataset revision;
    canonical indices are assigned after rows with ``canonical=False`` have
    been removed.
    """
    from datasets import Audio, load_dataset

    _validate_revision(revision)
    with Path(manifest_path).open(encoding="utf-8") as stream:
        payload = json.load(stream)
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError("clean split artifact has no manifest mapping")
    entries = [entry for entry in manifest.values() if entry.get("split") == split]
    if not entries:
        raise ValueError(f"clean split manifest has no entries for {split!r}")

    expected_identities = Counter(
        (
            int(entry["group_id"]),
            bool(entry["canonical"]),
            str(entry["audio_sha256"]),
        )
        for entry in entries
    )
    canonical_identities = {
        (int(entry["group_id"]), str(entry["audio_sha256"]))
        for entry in entries
        if bool(entry["canonical"])
    }
    if len(canonical_identities) != sum(bool(entry["canonical"]) for entry in entries):
        raise ValueError(f"duplicate canonical identity in clean split {split!r}")

    ds = load_dataset(dataset_id, revision=revision, split=split, streaming=True)
    if "audio" in ds.features:
        ds = ds.cast_column("audio", Audio(decode=False))
    remaining = expected_identities.copy()
    seen_groups = set()
    seen_audio = set()
    canonical_index = 0
    for raw_index, row in enumerate(ds):
        if not isinstance(row.get("canonical"), bool):
            raise RuntimeError(f"{split} raw row {raw_index} has no boolean canonical flag")
        identity = (
            int(row["group_id"]),
            row["canonical"],
            str(row["audio_sha256"]),
        )
        if remaining[identity] <= 0:
            raise RuntimeError(
                f"{split} raw row {raw_index} identity is absent or exhausted in the manifest"
            )
        remaining[identity] -= 1
        if not row["canonical"]:
            continue
        group_id, _canonical, audio_sha256 = identity
        if (group_id, audio_sha256) not in canonical_identities:
            raise RuntimeError(f"{split} raw row {raw_index} has an unknown canonical identity")
        if group_id in seen_groups or audio_sha256 in seen_audio:
            raise RuntimeError(f"duplicate canonical group/audio identity in {split!r}")
        seen_groups.add(group_id)
        seen_audio.add(audio_sha256)
        yield f"{split}_{canonical_index:05d}", row
        canonical_index += 1
        if limit and canonical_index >= limit:
            return

    if any(remaining.values()):
        raise RuntimeError(f"pinned {split!r} rows do not exhaust the clean split manifest")
    if canonical_index != len(canonical_identities):
        raise RuntimeError(f"canonical row-count drift in {split!r}")


def official_charts(row):
    """Yield ``(course, Chart)`` for every non-empty course in a row."""
    for course in COURSES:
        chart = load_taiko_parsed_course(row.get(course))
        if chart is not None and chart.events:
            chart.course = course
            yield course, chart


def build_train_lm(
    dataset_id,
    split="train",
    limit=0,
    order=3,
    alpha=0.05,
    verbose=True,
    *,
    revision=DEFAULT_DATASET_REVISION,
):
    """Fit the course-conditioned n-gram model on the training split."""
    lm = NGramModel(order=order, alpha=alpha)
    n_charts = 0
    for i, (sid, row) in enumerate(
        iter_rows(dataset_id, split, limit, revision=revision)
    ):
        for course, chart in official_charts(row):
            lm.add(course, event_tokens(chart.events, chart.bpm))
            n_charts += 1
        if verbose and i % 100 == 0:
            print(f"[lm {split}] songs={i + 1} charts={n_charts}", flush=True)
    return lm, n_charts


def build_calibration_rows(
    dataset_id,
    split="train",
    limit=0,
    verbose=True,
    *,
    revision=DEFAULT_DATASET_REVISION,
):
    """Collect ``{sid, course, events, bpm, level}`` rows for band building."""
    rows = []
    for i, (sid, row) in enumerate(
        iter_rows(dataset_id, split, limit, revision=revision)
    ):
        for course, chart in official_charts(row):
            rows.append(
                {
                    "sid": sid,
                    "course": course,
                    "events": chart.events,
                    "bpm": chart.bpm,
                    "level": chart.level,
                }
            )
        if verbose and i % 100 == 0:
            print(f"[calib {split}] songs={i + 1} charts={len(rows)}", flush=True)
    return rows
