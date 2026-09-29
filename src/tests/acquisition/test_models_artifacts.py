from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from acquisition.models_artifacts import (
    AcquisitionPaths,
    AcquisitionResult,
    ArtifactRecord,
)
from workspace.data_layout import data_root
from workspace.roots import WorkspaceRoots


def test_artifact_record_as_dict_round_trips_all_fields():
    record = ArtifactRecord(
        country="gb",
        run_date="2026-01-01",
        artifact_type="source",
        source_name="companies-house",
        source_url="https://example.test/data.csv",
        format="csv",
        file_path="/data/gb/acquire/companies-house.csv",
        content_hash="abc123",
        row_count=42,
        chunk_index=1,
        note="note",
        source_snapshot_date="2025-12-31",
        publication_frequency="monthly",
        refresh_if_older_than_days=30,
        acquired_at_utc="2026-01-01T00:00:00Z",
        last_acquired_at_utc="2025-12-01T00:00:00Z",
        days_since_last_acquisition=31,
        freshness_gate_applied=True,
        freshness_gate_passed=False,
        freshness_gate_reason="stale",
    )

    assert record.as_dict() == {
        "country": "gb",
        "run_date": "2026-01-01",
        "artifact_type": "source",
        "source_name": "companies-house",
        "source_url": "https://example.test/data.csv",
        "format": "csv",
        "file_path": "/data/gb/acquire/companies-house.csv",
        "content_hash": "abc123",
        "row_count": 42,
        "chunk_index": 1,
        "note": "note",
        "source_snapshot_date": "2025-12-31",
        "publication_frequency": "monthly",
        "refresh_if_older_than_days": 30,
        "acquired_at_utc": "2026-01-01T00:00:00Z",
        "last_acquired_at_utc": "2025-12-01T00:00:00Z",
        "days_since_last_acquisition": 31,
        "freshness_gate_applied": True,
        "freshness_gate_passed": False,
        "freshness_gate_reason": "stale",
    }


def test_artifact_record_as_dict_defaults_optional_fields_to_none():
    record = ArtifactRecord(
        country="gb",
        run_date="2026-01-01",
        artifact_type="source",
        source_name="companies-house",
        source_url="https://example.test/data.csv",
        format="csv",
        file_path="/data/gb/acquire/companies-house.csv",
    )

    as_dict = record.as_dict()
    assert as_dict["content_hash"] is None
    assert as_dict["row_count"] is None
    assert as_dict["freshness_gate_applied"] is None


def test_artifact_record_is_frozen():
    record = ArtifactRecord(
        country="gb",
        run_date="2026-01-01",
        artifact_type="source",
        source_name="companies-house",
        source_url="https://example.test/data.csv",
        format="csv",
        file_path="/data/gb/acquire/companies-house.csv",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        record.country = "fr"  # type: ignore[misc]


def test_acquisition_paths_holds_each_stage_directory(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    paths = AcquisitionPaths(
        roots=workspace_roots,
        data_root=data_root(workspace_roots),
        acquire_dir=layer_fixture_dir("gb", layer="acquire"),
        prepare_dir=layer_fixture_dir("gb", layer="prepare"),
        source_dir=layer_fixture_dir("gb", layer="source"),
        canonical_dir=layer_fixture_dir("gb", layer="canonical"),
    )

    assert paths.roots == workspace_roots
    assert paths.acquire_dir == layer_fixture_dir("gb", layer="acquire")
    assert paths.canonical_dir == layer_fixture_dir("gb", layer="canonical")


def test_acquisition_result_defaults_artifact_count_to_zero():
    result = AcquisitionResult(country="gb", status="ok", message="done")

    assert result.artifact_count == 0


def test_acquisition_result_carries_explicit_artifact_count():
    result = AcquisitionResult(
        country="gb", status="ok", message="done", artifact_count=3
    )

    assert result.artifact_count == 3
