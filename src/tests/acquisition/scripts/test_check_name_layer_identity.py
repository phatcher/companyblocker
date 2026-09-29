from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts import check_name_layer_identity
from workspace.roots import WorkspaceRoots


def _write_names(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def test_check_system_name_layer_identity_conforms_with_unique_uri_and_source(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_names(
        layer_fixture_dir("gb", layer="canonical")
        / "2026-06-01"
        / "names"
        / "gb-001.parquet",
        pl.DataFrame(
            {
                "system_uri": ["name://a", "name://b"],
                "source_uri": ["gb://1", "gb://2"],
            }
        ),
    )

    status = check_name_layer_identity.check_system_name_layer_identity(
        workspace_roots, "gb"
    )

    assert status.snapshot == "2026-06-01"
    assert status.row_count == 2
    assert status.distinct_system_uri == 2
    assert status.null_source_uri == 0
    assert status.conforms is True


def test_check_system_name_layer_identity_flags_many_to_one_system_uri(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    # Pre-regeneration shape: every variant row shares its entity's own
    # system_uri, and there's no source_uri column at all.
    _write_names(
        layer_fixture_dir("fr", layer="canonical")
        / "2026-06-01"
        / "names"
        / "fr-001.parquet",
        pl.DataFrame({"system_uri": ["fr://1", "fr://1", "fr://2"]}),
    )

    status = check_name_layer_identity.check_system_name_layer_identity(
        workspace_roots, "fr"
    )

    assert status.row_count == 3
    assert status.distinct_system_uri == 2
    assert status.null_source_uri == 3
    assert status.conforms is False


def test_check_system_name_layer_identity_flags_null_source_uri(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_names(
        layer_fixture_dir("wikidata", layer="canonical")
        / "2026-07-16"
        / "names"
        / "wikidata-001.parquet",
        pl.DataFrame(
            {
                "system_uri": ["name://a", "name://b"],
                "source_uri": ["wikidata://1", None],
            }
        ),
    )

    status = check_name_layer_identity.check_system_name_layer_identity(
        workspace_roots, "wikidata"
    )

    assert status.null_source_uri == 1
    assert status.conforms is False


def test_check_system_name_layer_identity_missing_layer_is_non_conforming(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
) -> None:
    status = check_name_layer_identity.check_system_name_layer_identity(
        workspace_roots, "ie"
    )

    assert status.snapshot is None
    assert status.row_count == 0
    assert status.conforms is False


def test_check_system_name_layer_identity_picks_latest_snapshot(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_names(
        layer_fixture_dir("offeneregister", layer="canonical")
        / "2026-06-26"
        / "names"
        / "or-001.parquet",
        pl.DataFrame({"system_uri": ["a"], "source_uri": [None]}),
    )
    _write_names(
        layer_fixture_dir("offeneregister", layer="canonical")
        / "2026-08-25"
        / "names"
        / "or-001.parquet",
        pl.DataFrame({"system_uri": ["name://a"], "source_uri": ["or://1"]}),
    )

    status = check_name_layer_identity.check_system_name_layer_identity(
        workspace_roots, "offeneregister"
    )

    assert status.snapshot == "2026-08-25"
    assert status.conforms is True
