"""Release-bundle integrity tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = ROOT / "experiments" / "verify_release_artifacts.py"
DEFAULT_MANIFEST = ROOT / "artifacts" / "MANIFEST.json"


def test_release_artifact_manifest():
    subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_release_artifact_manifest_rejects_wrong_total_bytes(tmp_path):
    manifest = json.loads(DEFAULT_MANIFEST.read_text())
    manifest["total_bytes"] += 1
    bad_manifest = tmp_path / "MANIFEST.json"
    bad_manifest.write_text(json.dumps(manifest))

    result = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--manifest", str(bad_manifest)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "total_bytes" in result.stderr
