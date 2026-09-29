from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.build_wikidata_jurisdiction_closure import discover_offending_qids


def test_discover_offending_qids_finds_family_split_partitions(
    tmp_path: Path, layer_fixture_dir
):
    """A real wikidata cleansed/ view partitioned under `primary/` must
    still be scanned for offending QIDs, not silently read as having none.
    """
    cleansed_dir = layer_fixture_dir("wikidata", layer="cleansed")
    partition_dir = cleansed_dir / "primary" / "jurisdiction_code=q999999"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["wikidata:1"]}).write_parquet(
        partition_dir / "part-00001.parquet"
    )

    assert discover_offending_qids(cleansed_dir) == ["Q999999"]


def test_discover_offending_qids_finds_legacy_top_level_partitions(
    tmp_path: Path, layer_fixture_dir
):
    cleansed_dir = layer_fixture_dir("wikidata", layer="cleansed")
    partition_dir = cleansed_dir / "jurisdiction_code=q999999"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["wikidata:1"]}).write_parquet(
        partition_dir / "part-00001.parquet"
    )

    assert discover_offending_qids(cleansed_dir) == ["Q999999"]


def test_discover_offending_qids_ignores_already_mapped_and_non_qid_values(
    tmp_path: Path,
    layer_fixture_dir,
):
    cleansed_dir = layer_fixture_dir("wikidata", layer="cleansed")
    primary_dir = cleansed_dir / "primary"
    for value in ("gb", "q999999"):
        partition_dir = primary_dir / f"jurisdiction_code={value}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"system_uri": ["wikidata:1"]}).write_parquet(
            partition_dir / "part-00001.parquet"
        )

    # "gb" is not QID-shaped (no leading digit-only suffix after "q") so it's
    # excluded; only the offending QID-shaped partition is returned.
    assert discover_offending_qids(cleansed_dir) == ["Q999999"]


def test_discover_offending_qids_empty_for_missing_dir(
    tmp_path: Path, layer_fixture_dir
):
    assert (
        discover_offending_qids(layer_fixture_dir("wikidata", layer="cleansed")) == []
    )
