"""Direct tests for `workspace.duckdb_catalog`.

Each test builds one of `layer_layout`'s documented shapes on disk and
asserts the DuckDB-backed twin returns exactly what its pathlib original
returns -- proving the resolution rule (partitioned wins, `primary/`/`names/`
preferred over the pre-split top level, `chunks/` never returned) survives a
change of the engine that answers "which files match".
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import duckdb
import pytest

from workspace.duckdb_catalog import (
    partition_values_via_duckdb,
    resolve_name_files_via_duckdb,
    resolve_primary_files_via_duckdb,
)
from workspace.layer_layout import (
    partition_values,
    resolve_name_files,
    resolve_primary_files,
)


@pytest.fixture
def con() -> Generator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect()
    try:
        yield connection
    finally:
        connection.close()


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_partitioned_primary_family_matches(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "primary" / "jurisdiction_code=gb" / "part-00001.parquet")
    _touch(layer_dir / "primary" / "jurisdiction_code=ie" / "part-00001.parquet")

    expected = resolve_primary_files(layer_dir, system_code="acme")
    actual = resolve_primary_files_via_duckdb(con, layer_dir, system_code="acme")

    assert actual == expected
    assert len(actual) == 2


def test_flat_primary_family_matches(con: duckdb.DuckDBPyConnection, tmp_path: Path):
    layer_dir = tmp_path / "canonical" / "2026-01-01"
    _touch(layer_dir / "primary" / "acme-001.parquet")
    _touch(layer_dir / "primary" / "acme-002.parquet")
    # A companion family sitting alongside primary must not be picked up.
    _touch(layer_dir / "primary" / "acme-names-001.parquet")

    expected = resolve_primary_files(layer_dir, system_code="acme")
    actual = resolve_primary_files_via_duckdb(con, layer_dir, system_code="acme")

    assert actual == expected
    assert len(actual) == 2


def test_pre_family_split_fallback_matches(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
):
    # No primary/ subdirectory at all -- the layer's own top level is what a
    # layer written before the family split still looks like.
    layer_dir = tmp_path / "matched"
    _touch(layer_dir / "acme-001.parquet")

    expected = resolve_primary_files(layer_dir, system_code="acme")
    actual = resolve_primary_files_via_duckdb(con, layer_dir, system_code="acme")

    assert actual == expected
    assert len(actual) == 1


def test_chunks_staging_never_returned(con: duckdb.DuckDBPyConnection, tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "chunks" / "acme-000.parquet")

    expected = resolve_primary_files(layer_dir, system_code="acme")
    actual = resolve_primary_files_via_duckdb(con, layer_dir, system_code="acme")

    assert actual == expected == []


def test_missing_layer_returns_empty(con: duckdb.DuckDBPyConnection, tmp_path: Path):
    layer_dir = tmp_path / "does-not-exist"

    assert resolve_primary_files_via_duckdb(con, layer_dir, system_code="acme") == []
    assert resolve_name_files_via_duckdb(con, layer_dir, system_code="acme") == []
    assert partition_values_via_duckdb(con, layer_dir) == []


def test_names_family_matches(con: duckdb.DuckDBPyConnection, tmp_path: Path):
    layer_dir = tmp_path / "canonical" / "2026-01-01"
    _touch(layer_dir / "names" / "acme-names-001.parquet")
    _touch(layer_dir / "primary" / "acme-001.parquet")

    expected = resolve_name_files(layer_dir, system_code="acme")
    actual = resolve_name_files_via_duckdb(con, layer_dir, system_code="acme")

    assert actual == expected
    assert len(actual) == 1


def test_names_family_absent_is_empty_not_error(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
):
    layer_dir = tmp_path / "canonical" / "2026-01-01"
    _touch(layer_dir / "primary" / "acme-001.parquet")

    expected = resolve_name_files(layer_dir, system_code="acme")
    actual = resolve_name_files_via_duckdb(con, layer_dir, system_code="acme")

    assert actual == expected == []


def test_partition_values_matches(con: duckdb.DuckDBPyConnection, tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "primary" / "jurisdiction_code=gb" / "part-00001.parquet")
    _touch(layer_dir / "primary" / "jurisdiction_code=ie" / "part-00001.parquet")

    expected = partition_values(layer_dir)
    actual = partition_values_via_duckdb(con, layer_dir)

    assert actual == expected == ["gb", "ie"]


def test_partition_values_empty_directory_is_a_known_divergence(
    con: duckdb.DuckDBPyConnection, tmp_path: Path
):
    """DuckDB's `glob()` only ever matches files, never bare directories
    (see `partition_values_via_duckdb`'s docstring), so a partition
    directory with nothing written into it yet reads as present through
    pathlib and absent through DuckDB. Pinned here as the one place this
    prototype's resolution rule does not survive the backing swap, rather
    than left to be rediscovered as a surprise.
    """
    layer_dir = tmp_path / "cleansed"
    (layer_dir / "primary" / "jurisdiction_code=gb").mkdir(parents=True)

    assert partition_values(layer_dir) == ["gb"]
    assert partition_values_via_duckdb(con, layer_dir) == []
