"""Contract tests for the untouched confirmatory workflow.

These tests never load the held-out panel.  They exercise only frozen metadata
and synthetic in-memory observations.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "experiments" / "confirmatory_holdout_v1"
sys.path.insert(0, str(WORKFLOW))

import analyze as confirmatory_analysis  # noqa: E402
import contract as confirmatory_contract  # noqa: E402
import run as confirmatory_run  # noqa: E402
from contract import derive_rng_seed, load_contract, runtime_dependency_contract  # noqa: E402


@pytest.mark.parametrize(
    ("name", "phase", "start", "stop"),
    [
        ("confirmatory.json", "confirmatory", 40, 120),
        ("smoke_synthetic.json", "smoke_synthetic", 0, 4),
        ("smoke_development.json", "smoke_development", 0, 2),
    ],
)
def test_frozen_configs_validate(name, phase, start, stop):
    config, contract, *_ = load_contract(WORKFLOW / "configs" / name)
    assert config["phase"] == phase
    assert config["dataset"]["canonical_song_index_start_inclusive"] == start
    assert config["dataset"]["canonical_song_index_stop_exclusive"] == stop
    assert config["dataset"]["canonical_filter"] is True
    assert contract["analysis"]["cluster_unit"] == "group_id"
    assert contract["clean_split_contract"]["test_source_rows"] == 140
    assert contract["clean_split_contract"]["test_canonical_groups"] == 120
    assert len(contract["replicate_base_seeds"]) == 5
    assert len(contract["primary_pairs"]) == 10
    assert contract["co_primary_claims"] == [
        {
            "claim_id": "C5__repetition_and_variety",
            "pair_ids": ["C5__repetition", "C5__surface_variety"],
            "combination_rule": "all_pairs_must_survive",
        }
    ]


def test_runtime_dependency_contract_accepts_frozen_environment(monkeypatch, tmp_path):
    expected_venv = tmp_path / ".venv"
    expected_venv.mkdir()
    (expected_venv / "pyvenv.cfg").write_text("version = 3.11.14\n")

    monkeypatch.setattr(confirmatory_contract, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(platform, "python_version", lambda: "3.11.14")
    monkeypatch.setattr(sys, "prefix", str(expected_venv))

    runtime = runtime_dependency_contract()
    assert runtime["venv_path"] == ".venv"
    assert runtime["python_major_minor"] == "3.11"
    assert len(runtime["pyvenv_cfg_sha256"]) == 64


@pytest.mark.parametrize("python_version", ["3.9.21", "3.13.9", "3.14.6"])
def test_runtime_dependency_contract_rejects_other_python_versions(
    monkeypatch, python_version
):
    monkeypatch.setattr(platform, "python_version", lambda: python_version)
    major_minor = ".".join(python_version.split(".")[:2])
    with pytest.raises(
        RuntimeError, match=rf"unsupported experiment Python {major_minor}"
    ):
        runtime_dependency_contract()


def test_replicate_seed_is_stable_and_changes_by_base_seed():
    first = derive_rng_seed(1103, "group_117", "oni", "C3_type_shuffle", 2)
    assert first == derive_rng_seed(1103, "group_117", "oni", "C3_type_shuffle", 2)
    assert first != derive_rng_seed(2207, "group_117", "oni", "C3_type_shuffle", 2)


def test_constant_response_is_zero_sensitivity():
    assert confirmatory_analysis._spearman_or_zero([0.0, 0.3, 0.6, 1.0], [1.0] * 4) == 0.0


def test_control_replicates_cannot_cancel_before_equivalence_check():
    _config, contract, *_ = load_contract(WORKFLOW / "configs" / "smoke_synthetic.json")
    metric = "density_adequacy_score"
    official_metrics = {name: None for name in contract["control_absolute_tolerances"]}
    official_metrics[metric] = 1.0
    observations = [
        {
            "group_id": 0,
            "course": "oni",
            "condition": "official",
            "dose_index": 0,
            "replicate_index": 0,
            "status": "success",
            "corruption_noop": False,
            "metrics": official_metrics,
        }
    ]
    # Mean is exactly the baseline, but every replicate violates the frozen
    # equivalence tolerance. A mean-before-abs bug would incorrectly pass.
    for replicate, value in enumerate((1.1, 0.9, 1.1, 0.9, 1.0), start=1):
        metrics = dict(official_metrics)
        metrics[metric] = value
        observations.append(
            {
                "group_id": 0,
                "course": "oni",
                "condition": "CTRL_identity",
                "dose_index": 0,
                "replicate_index": replicate,
                "status": "success",
                "corruption_noop": False,
                "metrics": metrics,
            }
        )
    diagnostics, passes, course_passes = confirmatory_analysis._control_diagnostics(
        observations, contract
    )
    target = next(
        row
        for row in diagnostics
        if row["control"] == "CTRL_identity" and row["metric"] == metric
    )
    assert target["max_absolute_delta"] == pytest.approx(0.1)
    assert target["passes"] is False
    assert passes[("CTRL_identity", metric)] is False
    assert course_passes[(0, "oni", "CTRL_identity", metric)] is False


def test_group_cluster_bootstrap_uses_group_values():
    summary = confirmatory_analysis._cluster_summary(
        {117: (-1.0, -2.0), 130: (0.0, 0.0)},
        draws=1000,
        seed=7,
        levels=(0.95,),
    )
    assert summary["n_group_clusters"] == 2
    assert summary["dose_statistic_mean"] == pytest.approx(-0.5)
    assert summary["target_minus_sham_mean"] == pytest.approx(-1.0)
    assert summary["group_ids"] == [117, 130]


def test_alias_interleaving_drives_canonical_index_and_true_manifest_id():
    rows = [
        (0, "test_row_00000", {"group_id": 900, "canonical": False, "audio_sha256": "alias-a"}),
        (1, "test_row_00001", {"group_id": 10, "canonical": True, "audio_sha256": "audio-a"}),
        (2, "test_row_00002", {"group_id": 901, "canonical": False, "audio_sha256": "alias-b"}),
        (3, "test_row_00003", {"group_id": 11, "canonical": True, "audio_sha256": "audio-b"}),
        # The development iterator must never consume this later row.
        (4, "test_row_00004", {"group_id": 12, "canonical": True, "audio_sha256": "audio-c"}),
    ]
    manifest_ids = {
        (10, "audio-a"): "real_manifest_sid_a",
        (11, "audio-b"): "real_manifest_sid_b",
        (12, "audio-c"): "real_manifest_sid_c",
    }
    scanned = [
        scan
        for scan, _row in confirmatory_run._iter_source_panel(
            rows,
            context_stop=2,
            scan_full_split=False,
            canonical_manifest_ids=manifest_ids,
        )
    ]
    assert [row["canonical_song_index"] for row in scanned] == [None, 0, None, 1]
    assert [row["manifest_source_row_id"] for row in scanned] == [
        None,
        "real_manifest_sid_a",
        None,
        "real_manifest_sid_b",
    ]
    assert [row["group_id"] for row in scanned if row["canonical"]] == [10, 11]
    assert all(
        row["action"] == "filtered_alias"
        for row in scanned
        if not row["canonical"]
    )


@pytest.mark.parametrize(
    "rows",
    [
        [
            (0, "synthetic_00000", {"group_id": 1, "canonical": True, "audio_sha256": "a"}),
            (1, "synthetic_00001", {"group_id": 1, "canonical": True, "audio_sha256": "b"}),
        ],
        [
            (0, "synthetic_00000", {"group_id": 1, "canonical": True, "audio_sha256": "a"}),
            (1, "synthetic_00001", {"group_id": 2, "canonical": True, "audio_sha256": "a"}),
        ],
    ],
)
def test_duplicate_canonical_group_or_audio_fails(rows):
    with pytest.raises(RuntimeError, match="duplicate canonical"):
        list(
            confirmatory_run._iter_source_panel(
                rows,
                context_stop=2,
                scan_full_split=True,
                canonical_manifest_ids=None,
            )
        )


def test_full_source_scan_contract_is_140_120_20():
    rows = []
    canonical_group = 0
    for source_index in range(140):
        canonical = source_index % 7 != 0
        group_id = canonical_group if canonical else 10_000 + source_index
        audio_sha256 = f"canonical-{canonical_group}" if canonical else f"alias-{source_index}"
        rows.append(
            (
                source_index,
                f"synthetic_{source_index:05d}",
                {
                    "group_id": group_id,
                    "canonical": canonical,
                    "audio_sha256": audio_sha256,
                },
            )
        )
        canonical_group += int(canonical)
    scanned = [
        scan
        for scan, _row in confirmatory_run._iter_source_panel(
            rows,
            context_stop=120,
            scan_full_split=True,
            canonical_manifest_ids=None,
        )
    ]
    counts = confirmatory_run._validate_source_scan_contract(
        scanned,
        {
            "source": "synthetic",
            "split": "synthetic",
            "scan_full_split": True,
            "expected_source_rows": 140,
            "expected_canonical_songs": 120,
            "expected_filtered_alias_rows": 20,
        },
        expected_canonical_set=None,
    )
    assert counts == {
        "scanned_source_rows": 140,
        "canonical_context_rows": 120,
        "filtered_alias_rows": 20,
    }


def test_sampling_context_constructor_accepts_only_canonical_rows():
    source_index, stored_id, row = next(
        confirmatory_run._synthetic_rows(1, "context_test")
    )
    charts = list(confirmatory_run._official_charts(row))
    context = confirmatory_run._sampling_context_record(
        source_row_index=source_index,
        stored_split_row_id=stored_id,
        manifest_source_row_id="real_manifest_sid",
        canonical_song_index=0,
        row=row,
        panel_group="smoke",
        charts=charts,
    )
    assert context["stored_split_row_id"] == stored_id
    assert context["manifest_source_row_id"] == "real_manifest_sid"
    assert context["canonical"] is True
    assert context["sid"] == f"group_{row['group_id']}"
    with pytest.raises(ValueError, match="canonical rows only"):
        confirmatory_run._sampling_context_record(
            source_row_index=source_index,
            stored_split_row_id=stored_id,
            manifest_source_row_id=None,
            canonical_song_index=0,
            row={**row, "canonical": False},
            panel_group="smoke",
            charts=charts,
        )


def test_minimum_72_is_applied_to_80_unique_group_clusters():
    _config, contract, *_ = load_contract(WORKFLOW / "configs" / "smoke_synthetic.json")
    minimum = contract["analysis"]["minimum_confirmatory_group_clusters"]
    summary = confirmatory_analysis._cluster_summary(
        {group_id: (-1.0, -1.0) for group_id in range(1_000, 1_080)},
        draws=10,
        seed=9,
        levels=(0.95,),
    )
    assert minimum == 72
    assert summary["n_group_clusters"] == 80
    assert summary["n_group_clusters"] >= minimum
