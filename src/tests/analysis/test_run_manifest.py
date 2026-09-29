from __future__ import annotations

import pytest

from analysis.run_manifest import (
    build_phase_entry,
    load_run_manifest,
    record_phase_run,
    resolve_run_manifest_path,
    validate_phase_entry,
)
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def _entry(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "created_utc": "2026-06-21T00:00:00+00:00",
        "phase": "phase0_schema_profile",
        "run_date": "2026-06-21",
        "systems": ["gb", "ie"],
        "active_systems": 2,
        "parameters": {"required_columns": ["country_tokens"]},
        "outputs": {"schema_profile_parquet": "profile/schema_profile.parquet"},
    }
    base.update(overrides)
    return base


def test_build_phase_entry_round_trips_mandatory_fields() -> None:
    entry = build_phase_entry(
        created_utc="2026-06-21T00:00:00+00:00",
        phase="phase0_schema_profile",
        run_date="2026-06-21",
        systems=["gb", "ie"],
        active_systems=2,
        parameters={"required_columns": ["country_tokens"]},
        outputs={"schema_profile_parquet": "profile/schema_profile.parquet"},
    )

    assert entry["phase"] == "phase0_schema_profile"
    assert entry["systems"] == ["gb", "ie"]
    assert "row_counts" not in entry
    assert "runtime_seconds" not in entry
    assert "notes" not in entry


def test_build_phase_entry_includes_optional_fields_when_given() -> None:
    entry = build_phase_entry(
        created_utc="2026-06-21T00:00:00+00:00",
        phase="phase1_token_metrics",
        run_date="2026-06-21",
        systems=["fr"],
        active_systems=1,
        parameters={"engine": "duckdb"},
        outputs={"country_stats": "metrics/token_stats_country.parquet"},
        row_counts={"country_rows": 10},
        runtime_seconds=1.25,
        notes=["stoplist thresholds unchanged from default"],
    )

    assert entry["row_counts"] == {"country_rows": 10}
    assert entry["runtime_seconds"] == 1.25
    assert entry["notes"] == ["stoplist thresholds unchanged from default"]


def test_validate_phase_entry_raises_on_missing_mandatory_field() -> None:
    entry = _entry()
    del entry["parameters"]

    with pytest.raises(ValueError, match="parameters"):
        validate_phase_entry(entry)


def test_validate_phase_entry_raises_on_unknown_field() -> None:
    entry = _entry(unexpected_field="x")

    with pytest.raises(ValueError, match="unexpected_field"):
        validate_phase_entry(entry)


def test_validate_phase_entry_raises_on_wrong_type() -> None:
    entry = _entry(active_systems="2")

    with pytest.raises(TypeError, match="active_systems"):
        validate_phase_entry(entry)

    entry = _entry(systems="gb")
    with pytest.raises(TypeError, match="systems"):
        validate_phase_entry(entry)


def test_resolve_run_manifest_path_names_the_manifest_file(
    workspace_roots: WorkspaceRoots,
) -> None:
    # Where the run root lands is `workspace.artifact_layout`'s contract and
    # is asserted there; this test's own subject is that the manifest is
    # named `run_manifest.json` directly under that run's own root.
    path = resolve_run_manifest_path(workspace_roots, "2026-06-21")

    assert path.name == "run_manifest.json"
    assert path.parent == analysis_report_run_dir(
        workspace_roots, "token_performance", "2026-06-21"
    )


def test_load_run_manifest_returns_empty_skeleton_when_absent(
    workspace_roots: WorkspaceRoots,
) -> None:
    manifest = load_run_manifest(workspace_roots, "2026-06-21")

    assert manifest == {"run_date": "2026-06-21", "phases": {}}


def test_record_phase_run_writes_and_round_trips(
    workspace_roots: WorkspaceRoots,
) -> None:
    entry = _entry()

    manifest_path = record_phase_run(workspace_roots, "2026-06-21", entry)

    assert manifest_path.exists()
    loaded = load_run_manifest(workspace_roots, "2026-06-21")
    assert loaded["phases"]["phase0_schema_profile"]["active_systems"] == 2


def test_record_phase_run_accumulates_multiple_phases_without_clobbering(
    workspace_roots: WorkspaceRoots,
) -> None:
    record_phase_run(
        workspace_roots, "2026-06-21", _entry(phase="phase0_schema_profile")
    )
    record_phase_run(
        workspace_roots,
        "2026-06-21",
        _entry(phase="phase1_token_metrics", active_systems=3),
    )

    loaded = load_run_manifest(workspace_roots, "2026-06-21")

    assert set(loaded["phases"].keys()) == {
        "phase0_schema_profile",
        "phase1_token_metrics",
    }
    assert loaded["phases"]["phase0_schema_profile"]["active_systems"] == 2
    assert loaded["phases"]["phase1_token_metrics"]["active_systems"] == 3


def test_record_phase_run_rerun_replaces_only_its_own_entry(
    workspace_roots: WorkspaceRoots,
) -> None:
    record_phase_run(
        workspace_roots, "2026-06-21", _entry(phase="phase0_schema_profile")
    )
    record_phase_run(
        workspace_roots,
        "2026-06-21",
        _entry(phase="phase1_token_metrics", active_systems=3),
    )

    # Rerun Phase 0 with a different system count -- must replace its own
    # entry, and must not disturb Phase 1's entry recorded above.
    record_phase_run(
        workspace_roots,
        "2026-06-21",
        _entry(phase="phase0_schema_profile", active_systems=5, systems=["gb"]),
    )

    loaded = load_run_manifest(workspace_roots, "2026-06-21")

    assert loaded["phases"]["phase0_schema_profile"]["active_systems"] == 5
    assert loaded["phases"]["phase0_schema_profile"]["systems"] == ["gb"]
    assert loaded["phases"]["phase1_token_metrics"]["active_systems"] == 3
