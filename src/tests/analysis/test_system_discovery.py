from __future__ import annotations

from pathlib import Path

import polars as pl

from analysis.system_discovery import (
    discover_systems_with_tokenized_parquet,
    filter_runnable_systems,
    is_system_runnable,
)
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots


def test_is_system_runnable_true_for_a_live_catalog_system() -> None:
    assert is_system_runnable("ie") is True


def test_is_system_runnable_false_for_a_stopped_catalog_system() -> None:
    # dbpedia's catalog entry is `status: stopped` -- see
    # src/acquisition/catalog/systems/dbpedia.json.
    assert is_system_runnable("dbpedia") is False


def test_is_system_runnable_true_for_a_system_absent_from_the_catalog() -> None:
    # An unknown code (no catalog entry) is not the same thing as a known,
    # excluded status -- auto-discovery should not silently drop a fixture
    # system the catalog has simply never heard of.
    assert is_system_runnable("not-a-real-system-code") is True


def test_filter_runnable_systems_drops_only_the_stopped_entry() -> None:
    assert filter_runnable_systems(["ie", "dbpedia", "gb"]) == ["ie", "gb"]


def _write_tokenized_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name_tokenized": ["acme"]}).write_parquet(path)


def test_discover_systems_with_tokenized_parquet_excludes_a_stopped_system(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_tokenized_parquet(
        system_layer_dir(workspace_roots, "ie", layer=TOKENIZED_LAYER_NAME)
        / "ie-001.parquet"
    )
    _write_tokenized_parquet(
        system_layer_dir(workspace_roots, "dbpedia", layer=TOKENIZED_LAYER_NAME)
        / "dbpedia-001.parquet"
    )

    discovered = discover_systems_with_tokenized_parquet(roots=workspace_roots)

    assert discovered == ["ie"]


def test_discover_systems_with_tokenized_parquet_finds_partitioned_output(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A real tokenized/ layer mirrors cleansed/'s own shape and is
    partitioned under jurisdiction_code=*/ for most systems -- discovery
    must find it there, not only in the flat top-level layout the other
    test in this file exercises.
    """
    _write_tokenized_parquet(
        layer_partition_dir(
            system_layer_dir(workspace_roots, "ie", layer=TOKENIZED_LAYER_NAME),
            value="ie",
        )
        / "part-00001.parquet"
    )

    discovered = discover_systems_with_tokenized_parquet(roots=workspace_roots)

    assert discovered == ["ie"]
