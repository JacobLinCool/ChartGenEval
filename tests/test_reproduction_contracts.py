"""Fail-closed checks for the dataset-backed release runners."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from _dataset import iter_rows  # noqa: E402
from _coupling_feature_contract import sha256, verify_mel_file  # noqa: E402
from evaluate_charts import chart_identity  # noqa: E402
from summarize_suite_v2_development import (  # noqa: E402
    _load as load_suite_v2_records,
    _validate_panel as validate_suite_v2_panel,
)


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(EXPERIMENTS / script), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )


def test_chart_identity_preserves_full_stem_and_parses_course_suffix():
    assert chart_identity("song_part_a.json", "oni") == ("song_part_a", "oni")
    assert chart_identity("song_part_b.json", "oni") == ("song_part_b", "oni")
    assert chart_identity("test_00000_easy.json", "oni") == ("test_00000", "easy")


def test_dataset_iterator_rejects_moving_revision_before_network_access():
    with pytest.raises(ValueError, match="40-character lowercase commit hash"):
        next(iter_rows("example/dataset", "train", revision="main"))


@pytest.mark.parametrize("limit", ["0", "41"])
def test_development_probe_runner_rejects_out_of_panel_limit(limit, tmp_path):
    result = _run(
        "reproduce_probes.py",
        "--limit-songs",
        limit,
        "--out",
        str(tmp_path / "must-not-exist.jsonl"),
    )
    assert result.returncode != 0
    assert "development panel" in result.stderr
    assert not (tmp_path / "must-not-exist.jsonl").exists()


def test_removed_legacy_limit_is_not_accepted(tmp_path):
    result = _run(
        "reproduce_probes.py",
        "--limit",
        "1",
        "--out",
        str(tmp_path / "must-not-exist.jsonl"),
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --limit 1" in result.stderr


def test_calibration_runner_rejects_test_split_and_nan_alpha(tmp_path):
    bad_split = _run(
        "reproduce_calibration.py",
        "--split",
        "test",
        "--out",
        str(tmp_path / "bad-split.json"),
    )
    assert bad_split.returncode == 2
    assert "invalid choice" in bad_split.stderr

    bad_alpha = _run(
        "reproduce_calibration.py",
        "--alpha",
        "nan",
        "--out",
        str(tmp_path / "bad-alpha.json"),
    )
    assert bad_alpha.returncode != 0
    assert "finite and positive" in bad_alpha.stderr


def test_chart_runner_requires_system_identifier():
    result = _run("evaluate_charts.py", "--charts", "missing")
    assert result.returncode == 2
    assert "--system" in result.stderr


def test_suite_v2_panel_rejects_duplicate_condition_cell():
    rows = load_suite_v2_records(
        ROOT / "artifacts/records/structure_v2_development.jsonl"
    )
    target_index = next(
        index
        for index, row in enumerate(rows)
        if row["probe"] == "C8_bar_shuffle" and row["dose_index"] == 3
    )
    target = rows[target_index]
    duplicate = next(
        row
        for row in rows
        if row["sid"] == target["sid"]
        and row["course"] == target["course"]
        and row["probe"] == "C8_bar_shuffle"
        and row["dose_index"] == 2
    )
    rows[target_index] = dict(duplicate)
    with pytest.raises(ValueError, match="duplicate cell"):
        validate_suite_v2_panel(rows, "mutated")


def test_mel_contract_rejects_byte_tampering(tmp_path):
    path = tmp_path / "test_00000.mel.npy"
    original = np.zeros((128, 16), dtype=np.float16)
    np.save(path, original)
    row = {
        "sid": "test_00000",
        "mel_bytes": path.stat().st_size,
        "mel_sha256": sha256(path),
        "mel_shape": [128, 16],
        "mel_dtype": "float16",
    }
    verified = verify_mel_file(path, row)
    assert np.array_equal(verified, original)

    np.save(path, np.ones((128, 16), dtype=np.float16))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_mel_file(path, row)
