from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl
import pytest

from acquisition.cleanser_orchestrate import (
    _migrate_legacy_flat_cleanse_chunks,
    materialize_cleansed_merge,
)


def test_migrate_legacy_flat_cleanse_chunks_moves_files_into_chunks_dir(
    tmp_path: Path,
    layer_fixture_dir,
):
    cleansed_dir = layer_fixture_dir("ie", layer="cleansed")
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["ie:1"]}).write_parquet(
        cleansed_dir / "ie-001.parquet"
    )
    pl.DataFrame({"system_uri": ["ie:2"]}).write_parquet(
        cleansed_dir / "ie-002.parquet"
    )
    (cleansed_dir / "empty_cleanser.parquet").write_bytes(b"x")

    _migrate_legacy_flat_cleanse_chunks(cleansed_dir=cleansed_dir)

    assert sorted(p.name for p in (cleansed_dir / "chunks").glob("*.parquet")) == [
        "ie-001.parquet",
        "ie-002.parquet",
    ]
    # empty_cleanser.parquet is a top-level diagnostic artifact, not a chunk.
    assert (cleansed_dir / "empty_cleanser.parquet").exists()
    assert not (cleansed_dir / "ie-001.parquet").exists()


def test_migrate_legacy_flat_cleanse_chunks_preserves_content(
    tmp_path: Path, layer_fixture_dir
):
    cleansed_dir = layer_fixture_dir("ie", layer="cleansed")
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({"system_uri": ["ie:1", "ie:2"], "name": ["a", "b"]})
    frame.write_parquet(cleansed_dir / "ie-001.parquet")

    _migrate_legacy_flat_cleanse_chunks(cleansed_dir=cleansed_dir)

    migrated = pl.read_parquet(cleansed_dir / "chunks" / "ie-001.parquet")
    assert migrated.equals(frame)


def test_migrate_legacy_flat_cleanse_chunks_is_a_noop_once_chunks_dir_exists(
    tmp_path: Path,
    layer_fixture_dir,
):
    cleansed_dir = layer_fixture_dir("ie", layer="cleansed")
    chunks_dir = cleansed_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["ie:1"]}).write_parquet(chunks_dir / "ie-001.parquet")
    # A flat file sitting alongside an already-existing chunks/ dir must not
    # be swept up -- chunks/ existing means migration already happened.
    pl.DataFrame({"system_uri": ["ie:2"]}).write_parquet(
        cleansed_dir / "ie-999.parquet"
    )

    _migrate_legacy_flat_cleanse_chunks(cleansed_dir=cleansed_dir)

    assert (cleansed_dir / "ie-999.parquet").exists()
    assert sorted(p.name for p in chunks_dir.glob("*.parquet")) == ["ie-001.parquet"]


def test_migrate_legacy_flat_cleanse_chunks_is_a_noop_when_already_partitioned(
    tmp_path: Path,
    layer_fixture_dir,
):
    cleansed_dir = layer_fixture_dir("gleif", layer="cleansed")
    partition_dir = cleansed_dir / "jurisdiction_code=gb"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gleif:1"]}).write_parquet(
        partition_dir / "part-00001.parquet"
    )
    stray_flat_file = cleansed_dir / "gleif-999.parquet"
    pl.DataFrame({"system_uri": ["gleif:2"]}).write_parquet(stray_flat_file)

    _migrate_legacy_flat_cleanse_chunks(cleansed_dir=cleansed_dir)

    assert stray_flat_file.exists()
    assert not (cleansed_dir / "chunks").exists()


def test_migrate_legacy_flat_cleanse_chunks_is_a_noop_when_already_family_split(
    tmp_path: Path,
    layer_fixture_dir,
):
    """A system already fully migrated to the `primary/`-family shape must
    not have a stray top-level flat file swept into `chunks/` --
    `is_partitioned_layer` has to recognise the *current* shape, not only
    the pre-split one the other noop test above covers.
    """
    cleansed_dir = layer_fixture_dir("gleif", layer="cleansed")
    partition_dir = cleansed_dir / "primary" / "jurisdiction_code=gb"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gleif:1"]}).write_parquet(
        partition_dir / "part-00001.parquet"
    )
    stray_flat_file = cleansed_dir / "gleif-999.parquet"
    pl.DataFrame({"system_uri": ["gleif:2"]}).write_parquet(stray_flat_file)

    _migrate_legacy_flat_cleanse_chunks(cleansed_dir=cleansed_dir)

    assert stray_flat_file.exists()
    assert not (cleansed_dir / "chunks").exists()


@pytest.mark.integration
def test_materialize_cleansed_merge_uses_hive_null_sentinel_for_missing_jurisdiction(
    tmp_path: Path,
    layer_fixture_dir,
):
    """A null jurisdiction_code must partition into a directory DuckDB's
    hive_partitioning=1 reader (used by match_ops.py) recognizes as NULL --
    not a bare jurisdiction_code=/ directory, whose empty-string-shaped name
    collides with the real (null) parquet column value and gets silently
    substituted for it on read."""
    cleansed_dir = layer_fixture_dir("wikidata", layer="cleansed")
    chunks_dir = cleansed_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["wikidata:1", "wikidata:2"],
            "name": ["Acme", "Acme UK"],
            "name_cleansed_basic": ["acme", "acme uk"],
            "name_cleansed": ["acme", "acme uk"],
            "jurisdiction_code": [None, "gb"],
        }
    ).write_parquet(chunks_dir / "wikidata-001.parquet")

    result_dir = materialize_cleansed_merge(
        source_dir=chunks_dir,
        output_dir=cleansed_dir,
        system="wikidata",
        name_col="name_cleansed",
        rows_per_file=1_000_000,
    )

    assert not (result_dir / "primary" / "jurisdiction_code=").exists()
    null_partition_dir = (
        result_dir / "primary" / "jurisdiction_code=__HIVE_DEFAULT_PARTITION__"
    )
    assert null_partition_dir.exists()

    written = pl.read_parquet(next(null_partition_dir.glob("*.parquet")))
    assert written["jurisdiction_code"].to_list() == [None]

    # Prove the actual downstream read path (match_ops.py's
    # hive_partitioning=1) now sees NULL rather than silently substituting
    # the partition-directory name for the real column value.
    con = duckdb.connect()
    rows = con.execute(
        "SELECT jurisdiction_code, system_uri FROM read_parquet(?, hive_partitioning=1) "
        "ORDER BY system_uri",
        [str(result_dir / "primary" / "jurisdiction_code=*/*.parquet")],
    ).fetchall()
    assert rows == [(None, "wikidata:1"), ("gb", "wikidata:2")]


def test_cleanse_leaves_the_previous_view_intact_when_a_run_fails(
    tmp_path: Path, monkeypatch
):
    """A full cleanse builds the new layer beside the live one and swaps it
    in at the end. The stage used to clear the live view first and then
    write partition files for minutes, so an interruption left a
    half-written mixture with nothing marking it partial.

    The failure is injected after the first of two files has been written,
    which is precisely the state the old flow could not survive.
    """
    from acquisition import cleanser_orchestrate
    from acquisition.cleanser_orchestrate import cleanse_canonical_view
    from workspace.layer_layout import LAYER_STAGING_SUFFIX, primary_family_dir

    canonical_dir = tmp_path / "canonical"
    for jurisdiction in ("gb", "ie"):
        partition_dir = primary_family_dir(canonical_dir) / (
            f"jurisdiction_code={jurisdiction}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "system_uri": [f"{jurisdiction}:1"],
                "jurisdiction_code": [jurisdiction],
                "name": ["EXAMPLE LTD"],
            }
        ).write_parquet(partition_dir / "part-00001.parquet")

    cleansed_dir = tmp_path / "cleansed"
    previous_partition = primary_family_dir(cleansed_dir) / "jurisdiction_code=gb"
    previous_partition.mkdir(parents=True, exist_ok=True)
    previous_file = previous_partition / "part-00001.parquet"
    pl.DataFrame({"marker": ["the previous view"]}).write_parquet(previous_file)

    real_write = cleanser_orchestrate._write_cleanse_output
    calls = {"n": 0}

    def _fail_on_second(lf, output_path, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("disk went away mid-run")
        return real_write(lf, output_path)

    monkeypatch.setattr(cleanser_orchestrate, "_write_cleanse_output", _fail_on_second)

    with pytest.raises(OSError, match="disk went away"):
        cleanse_canonical_view(canonical_dir, cleansed_dir, "gb", company_col="name")

    assert calls["n"] == 2, "the failure must land after a file was already written"
    assert previous_file.exists()
    assert pl.read_parquet(previous_file)["marker"].to_list() == ["the previous view"]
    staging = cleansed_dir.parent / f"{cleansed_dir.name}{LAYER_STAGING_SUFFIX}"
    assert not staging.exists()


def test_cleanse_canonical_view_normalises_cross_jurisdiction_suffixes(
    tmp_path: Path,
):
    """A registry's own jurisdiction must not narrow which legal-form suffix
    cleansing recognises. Passing no company_type_regex/company_type_mapping
    lets `CleanseConfig`'s own default resolve the full, multi-jurisdiction
    rule set, so a UK-partitioned row named with a German (`GmbH`), French
    (`SARL`) or Dutch (`BV`) suffix normalises by its own suffix instead of
    falling back to `default_private` -- on real `data/gb/cleansed/` output,
    803 rows across those three suffixes plus
    `Anstalt`/`Aktiengesellschaft`/`N.V.` were once all defaulted this way."""
    from acquisition.cleanser_orchestrate import cleanse_canonical_view
    from workspace.layer_layout import primary_family_dir

    canonical_dir = tmp_path / "canonical"
    partition_dir = primary_family_dir(canonical_dir) / "jurisdiction_code=gb"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb:1", "gb:2", "gb:3", "gb:4"],
            "jurisdiction_code": ["gb", "gb", "gb", "gb"],
            "name": ["ACME GMBH", "ACME SARL", "ACME BV", "ACME LTD"],
        }
    ).write_parquet(partition_dir / "part-00001.parquet")

    cleansed_dir = tmp_path / "cleansed"
    cleanse_canonical_view(canonical_dir, cleansed_dir, "gb", company_col="name")

    output_file = next(
        (primary_family_dir(cleansed_dir) / "jurisdiction_code=gb").glob("*.parquet")
    )
    result = pl.read_parquet(output_file).sort("system_uri")
    assert result["company_type_source"].to_list() == ["name_suffix"] * 4
    assert result["company_type"].to_list() == ["gmbh", "sarl", "bv", "ltd"]


@pytest.mark.integration
def test_swap_retries_a_blocked_rename_then_succeeds(tmp_path: Path, monkeypatch):
    """`data/` is shared across worktrees, so a concurrent session merely
    reading a layer blocks the rename on Windows. A short backoff clears the
    common case rather than losing the whole run to it."""
    from workspace import layer_layout
    from workspace.layer_layout import fresh_layer_staging_dir, swap_layer_into_place

    live = tmp_path / "cleansed"
    (live / "primary").mkdir(parents=True)
    (live / "primary" / "old.parquet").write_bytes(b"old")
    staging = fresh_layer_staging_dir(live)
    (staging / "primary").mkdir(parents=True)
    (staging / "primary" / "new.parquet").write_bytes(b"new")

    monkeypatch.setattr(layer_layout, "SWAP_RETRY_DELAYS_SECONDS", (0.0, 0.0))
    real_rename = Path.rename
    calls = {"n": 0}

    def _blocked_once(self, target):
        if self.name.endswith(layer_layout.LAYER_STAGING_SUFFIX):
            calls["n"] += 1
            if calls["n"] == 1:
                raise PermissionError(5, "Access is denied")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _blocked_once)
    swap_layer_into_place(staging=staging, live=live)

    assert calls["n"] == 2
    assert (live / "primary" / "new.parquet").exists()
    assert not (live / "primary" / "old.parquet").exists()
    assert not staging.exists()


@pytest.mark.integration
def test_swap_keeps_finished_output_when_it_cannot_be_moved(
    tmp_path: Path, monkeypatch
):
    """The swap is the last step, so a failure there means a complete layer
    was produced. Discarding it would throw the whole run away over a
    transient reader; the previous layer must also survive."""
    from workspace import layer_layout
    from workspace.layer_layout import (
        LayerSwapError,
        fresh_layer_staging_dir,
        swap_layer_into_place,
    )

    live = tmp_path / "cleansed"
    (live / "primary").mkdir(parents=True)
    (live / "primary" / "old.parquet").write_bytes(b"old")
    staging = fresh_layer_staging_dir(live)
    (staging / "primary").mkdir(parents=True)
    (staging / "primary" / "new.parquet").write_bytes(b"new")

    monkeypatch.setattr(layer_layout, "SWAP_RETRY_DELAYS_SECONDS", (0.0,))
    real_rename = Path.rename

    def _always_blocked(self, target):
        if self.name.endswith(layer_layout.LAYER_STAGING_SUFFIX):
            raise PermissionError(5, "Access is denied")
        return real_rename(self, target)

    monkeypatch.setattr(Path, "rename", _always_blocked)
    with pytest.raises(LayerSwapError, match="could not swap it in"):
        swap_layer_into_place(staging=staging, live=live)

    assert (staging / "primary" / "new.parquet").exists()
    assert (live / "primary" / "old.parquet").exists()
