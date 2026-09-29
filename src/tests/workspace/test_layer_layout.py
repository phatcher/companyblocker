"""Direct tests for `workspace.layer_layout`'s file-resolution contract.

`resolve_primary_files`, `resolve_name_files` and `partition_values` were
previously asserted only as the pathlib side of `test_duckdb_catalog.py`'s
twin comparisons, so their guarantee would leave with that file if it were
ever retired. This file pins each layout shape directly against the module
itself: partitioned vs. flat primary, the pre-family-split fallback,
`chunks/` staging never returned, a missing layer, the names family present
and absent, and the empty-partition-directory case where this module and its
DuckDB twin deliberately disagree. `resolve_partition_dir`, which has no
coverage anywhere else, is asserted here too.
"""

from __future__ import annotations

from pathlib import Path

from workspace.layer_layout import (
    partition_values,
    resolve_country_rows,
    resolve_name_files,
    resolve_partition_dir,
    resolve_primary_files,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_partitioned_primary_family(tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "primary" / "jurisdiction_code=gb" / "part-00001.parquet")
    _touch(layer_dir / "primary" / "jurisdiction_code=ie" / "part-00001.parquet")

    files = resolve_primary_files(layer_dir, system_code="acme")

    assert files == sorted(files)
    assert len(files) == 2
    assert all(path.suffix == ".parquet" for path in files)


def test_flat_primary_family_excludes_companion(tmp_path: Path):
    layer_dir = tmp_path / "canonical" / "2026-01-01"
    _touch(layer_dir / "primary" / "acme-001.parquet")
    _touch(layer_dir / "primary" / "acme-002.parquet")
    # A companion family sitting alongside primary must not be picked up.
    _touch(layer_dir / "primary" / "acme-names-001.parquet")

    files = resolve_primary_files(layer_dir, system_code="acme")

    assert [path.name for path in files] == ["acme-001.parquet", "acme-002.parquet"]


def test_partitioned_wins_over_flat(tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "primary" / "jurisdiction_code=gb" / "part-00001.parquet")
    _touch(layer_dir / "primary" / "acme-001.parquet")

    files = resolve_primary_files(layer_dir, system_code="acme")

    assert len(files) == 1
    assert files[0].parent.name == "jurisdiction_code=gb"


def test_pre_family_split_fallback(tmp_path: Path):
    # No primary/ subdirectory at all -- the layer's own top level is what a
    # layer written before the family split still looks like.
    layer_dir = tmp_path / "matched"
    _touch(layer_dir / "acme-001.parquet")

    files = resolve_primary_files(layer_dir, system_code="acme")

    assert [path.name for path in files] == ["acme-001.parquet"]


def test_chunks_staging_never_returned(tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "chunks" / "acme-000.parquet")

    assert resolve_primary_files(layer_dir, system_code="acme") == []


def test_missing_layer_returns_empty(tmp_path: Path):
    layer_dir = tmp_path / "does-not-exist"

    assert resolve_primary_files(layer_dir, system_code="acme") == []
    assert resolve_name_files(layer_dir, system_code="acme") == []
    assert partition_values(layer_dir) == []
    assert resolve_partition_dir(layer_dir, value="gb") is None


def test_names_family_present(tmp_path: Path):
    layer_dir = tmp_path / "canonical" / "2026-01-01"
    _touch(layer_dir / "names" / "acme-names-001.parquet")
    _touch(layer_dir / "primary" / "acme-001.parquet")

    files = resolve_name_files(layer_dir, system_code="acme")

    assert [path.name for path in files] == ["acme-names-001.parquet"]


def test_names_family_falls_back_to_pre_split_top_level(tmp_path: Path):
    layer_dir = tmp_path / "matched"
    _touch(layer_dir / "acme-names-001.parquet")

    files = resolve_name_files(layer_dir, system_code="acme")

    assert [path.name for path in files] == ["acme-names-001.parquet"]


def test_names_family_absent_is_empty_not_error(tmp_path: Path):
    layer_dir = tmp_path / "canonical" / "2026-01-01"
    _touch(layer_dir / "primary" / "acme-001.parquet")

    assert resolve_name_files(layer_dir, system_code="acme") == []


def test_partition_values_sorted_lowercased(tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "primary" / "jurisdiction_code=IE" / "part-00001.parquet")
    _touch(layer_dir / "primary" / "jurisdiction_code=gb" / "part-00001.parquet")

    assert partition_values(layer_dir) == ["gb", "ie"]


def test_partition_values_pre_split_fallback(tmp_path: Path):
    layer_dir = tmp_path / "matched"
    _touch(layer_dir / "jurisdiction_code=fr" / "part-00001.parquet")

    assert partition_values(layer_dir) == ["fr"]


def test_partition_values_empty_partition_directory_still_counts(tmp_path: Path):
    """Pinning this module's own rule directly: an empty partition directory
    (nothing written into it yet) still reads as present through pathlib,
    which is the divergence `test_duckdb_catalog.py`'s DuckDB-backed twin
    records against -- DuckDB's `glob()` only ever matches files, never bare
    directories, so it would report this partition as absent."""
    layer_dir = tmp_path / "cleansed"
    (layer_dir / "primary" / "jurisdiction_code=gb").mkdir(parents=True)

    assert partition_values(layer_dir) == ["gb"]


def test_resolve_partition_dir_prefers_primary_family(tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    partition_dir = layer_dir / "primary" / "jurisdiction_code=gb"
    partition_dir.mkdir(parents=True)

    assert resolve_partition_dir(layer_dir, value="gb") == partition_dir


def test_resolve_partition_dir_falls_back_to_pre_split_top_level(tmp_path: Path):
    layer_dir = tmp_path / "matched"
    partition_dir = layer_dir / "jurisdiction_code=gb"
    partition_dir.mkdir(parents=True)

    assert resolve_partition_dir(layer_dir, value="gb") == partition_dir


def test_resolve_partition_dir_missing_partition_is_none(tmp_path: Path):
    layer_dir = tmp_path / "cleansed"
    _touch(layer_dir / "primary" / "jurisdiction_code=gb" / "part-00001.parquet")

    assert resolve_partition_dir(layer_dir, value="ie") is None


def test_resolve_partition_dir_missing_layer_is_none(tmp_path: Path):
    assert resolve_partition_dir(tmp_path / "does-not-exist", value="gb") is None


def test_a_country_of_a_partitioned_dataset_is_its_partition_s_files(tmp_path: Path):
    primary = tmp_path / "primary"
    for country in ("gb", "ie"):
        (primary / f"jurisdiction_code={country}").mkdir(parents=True)
        (primary / f"jurisdiction_code={country}" / "part-0.parquet").touch()

    rows = resolve_country_rows(primary, country="GB")

    assert rows.files == (primary / "jurisdiction_code=gb" / "part-0.parquet",)
    assert rows.filter_column is None
    assert resolve_country_rows(primary, country="fr").files == ()


def test_a_country_of_a_chunked_dataset_is_every_file_filtered_on_the_column(
    tmp_path: Path,
):
    names = tmp_path / "names"
    names.mkdir()
    for chunk in ("gb-names-001.parquet", "gb-names-002.parquet"):
        (names / chunk).touch()

    rows = resolve_country_rows(names, country="gb")

    assert rows.files == (
        names / "gb-names-001.parquet",
        names / "gb-names-002.parquet",
    )
    assert rows.filter_column == "jurisdiction_code"
