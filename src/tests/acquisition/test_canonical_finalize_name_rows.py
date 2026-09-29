from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from workspace.roots import WorkspaceRoots


def _write_gleif_finalize_fixture(
    tmp_path: Path,
    layer_fixture_dir,
    *,
    name_shards: list[list[dict[str, str | None]]],
    entity_uris: list[str],
) -> tuple[Path, Path]:
    """Write a canonical-stage entity view plus one or more
    gleif-names-*.parquet shard files -- the on-disk inputs
    `finalize_canonical_name_rows` reads. `name_shards` is one row-list per
    shard file written."""

    from workspace.layer_layout import names_family_dir, primary_family_dir

    snapshot_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    partition_dir = primary_family_dir(snapshot_dir) / "jurisdiction_code=gb"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"system_uri": entity_uris, "jurisdiction_code": ["GB"] * len(entity_uris)}
    ).write_parquet(partition_dir / "part-00001.parquet")

    names_dir = names_family_dir(snapshot_dir)
    names_dir.mkdir(parents=True, exist_ok=True)
    for index, shard_rows in enumerate(name_shards, start=1):
        pl.DataFrame(shard_rows).write_parquet(
            names_dir / f"gleif-names-{index:03d}.parquet"
        )

    return snapshot_dir, names_dir


def test_finalize_canonical_name_rows_gives_each_row_its_own_identity(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """Identity composition happens where Canonical writes the names layer
    (its last write), not in a later stage that would have to rewrite
    already-written output."""

    from acquisition.canonical import finalize_canonical_name_rows
    from workspace.derived_uri import name_variant_uri, source_uri_of

    _snapshot_dir, names_dir = _write_gleif_finalize_fixture(
        tmp_path,
        layer_fixture_dir,
        name_shards=[
            [
                {
                    "system_uri": "gleif://A",
                    "LEI": "A",
                    "name": "Old Co",
                    "source_type": "PREVIOUS_LEGAL_NAME",
                    "language_code": None,
                    "derivation_note": "native OtherEntityName",
                    "name_type": "previous",
                }
            ]
        ],
        entity_uris=["gleif://A"],
    )

    written = finalize_canonical_name_rows(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    assert {path.name for path in written} == {"gleif-names-001.parquet"}
    result = pl.read_parquet(names_dir / "gleif-names-001.parquet")
    assert result.height == 1
    row = result.row(0, named=True)
    assert row["source_uri"] == "gleif://A"
    assert row["system_uri"].startswith("name://")
    assert row["system_uri"] != "gleif://A"

    expected = name_variant_uri(
        source_uri="gleif://A", name_type="previous", value="Old Co"
    )
    assert row["system_uri"] == expected
    # The row decomposes to the value of its own `source_uri` with no lookup.
    assert source_uri_of(row["system_uri"]) == row["source_uri"]


def test_finalize_canonical_name_rows_collapses_duplicate_triple_within_shard(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """The source captures the same entity's name twice within one shard
    file -- e.g. two native records for the same LEI both contributing a
    "previous name" of "Old Co". The pass collapses them to one row rather
    than raising, since both compose the same `name://` identity."""

    from acquisition.canonical import finalize_canonical_name_rows
    from workspace.derived_uri import name_variant_uri

    _snapshot_dir, _names_dir = _write_gleif_finalize_fixture(
        tmp_path,
        layer_fixture_dir,
        name_shards=[
            [
                {
                    "system_uri": "gleif://A",
                    "LEI": "A",
                    "name": "Old Co",
                    "source_type": "PREVIOUS_LEGAL_NAME",
                    "language_code": None,
                    "derivation_note": "native OtherEntityName",
                    "name_type": "previous",
                },
                {
                    "system_uri": "gleif://A",
                    "LEI": "A",
                    "name": "Old Co",
                    "source_type": "PREVIOUS_LEGAL_NAME",
                    "language_code": None,
                    "derivation_note": "native OtherEntityName (repeated)",
                    "name_type": "previous",
                },
            ]
        ],
        entity_uris=["gleif://A"],
    )

    written = finalize_canonical_name_rows(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    result = pl.read_parquet(written)
    assert result.height == 1
    expected = name_variant_uri(
        source_uri="gleif://A", name_type="previous", value="Old Co"
    )
    assert result.row(0, named=True)["system_uri"] == expected


def test_finalize_canonical_name_rows_collapses_duplicate_triple_across_shards(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """The within-shard collapse alone would miss this: the same
    (source_uri, name_type, name) triple appears once in each of two
    separate shard files, not twice in the same one -- e.g. the source
    emitted duplicate records for the same entity that landed in different
    shards. The collapse must see every shard together to catch it."""

    from acquisition.canonical import finalize_canonical_name_rows
    from workspace.derived_uri import name_variant_uri

    _snapshot_dir, _names_dir = _write_gleif_finalize_fixture(
        tmp_path,
        layer_fixture_dir,
        name_shards=[
            [
                {
                    "system_uri": "gleif://A",
                    "LEI": "A",
                    "name": "Old Co",
                    "source_type": "PREVIOUS_LEGAL_NAME",
                    "language_code": None,
                    "derivation_note": "native OtherEntityName",
                    "name_type": "previous",
                }
            ],
            [
                {
                    "system_uri": "gleif://A",
                    "LEI": "A",
                    "name": "Old Co",
                    "source_type": "PREVIOUS_LEGAL_NAME",
                    "language_code": None,
                    "derivation_note": "native OtherEntityName (second shard)",
                    "name_type": "previous",
                }
            ],
        ],
        entity_uris=["gleif://A"],
    )

    written = finalize_canonical_name_rows(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    combined = pl.concat(
        [pl.read_parquet(path) for path in written], how="vertical_relaxed"
    )
    assert combined.height == 1
    expected = name_variant_uri(
        source_uri="gleif://A", name_type="previous", value="Old Co"
    )
    assert combined.row(0, named=True)["system_uri"] == expected


def test_finalize_canonical_name_rows_consolidates_and_attaches_jurisdiction(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """The names family gets the same consolidation the entity view got,
    plus the `jurisdiction_code` column that makes "just grab the GB
    aliases" a filter rather than a hand-written join to the entity view.
    """
    from acquisition.canonical import finalize_canonical_name_rows
    from workspace.layer_layout import (
        names_family_dir,
        primary_family_dir,
    )

    snapshot_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    for jurisdiction, uri in [("gb", "gleif://A"), ("ie", "gleif://B")]:
        partition_dir = primary_family_dir(snapshot_dir) / (
            f"jurisdiction_code={jurisdiction}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {"system_uri": [uri], "jurisdiction_code": [jurisdiction.upper()]}
        ).write_parquet(partition_dir / "part-00001.parquet")

    names_dir = names_family_dir(snapshot_dir)
    names_dir.mkdir(parents=True, exist_ok=True)
    for index, (uri, name) in enumerate(
        [("gleif://A", "Alpha Ltd"), ("gleif://B", "Beta Ltd"), ("gleif://C", "Gone")],
        start=1,
    ):
        pl.DataFrame(
            {
                "system_uri": [uri],
                "name": [name],
                "name_type": ["primary"],
            }
        ).write_parquet(names_dir / f"gleif-names-{index:03d}.parquet")

    written = finalize_canonical_name_rows(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    # Three fragments in, one consolidated file out.
    assert [path.name for path in written] == ["gleif-names-001.parquet"]
    assert sorted(path.name for path in names_dir.glob("*.parquet")) == [
        "gleif-names-001.parquet"
    ]

    out = pl.read_parquet(written).sort("source_uri")
    # gleif://C's entity is absent from the canonical view, so its variant is
    # dropped: a source-level exclusion has to carry through to every family,
    # not inflate the sidecar with rows nothing downstream can match.
    assert out["source_uri"].to_list() == ["gleif://A", "gleif://B"]
    # The jurisdiction is the entity view's own value, verbatim.
    assert out["jurisdiction_code"].to_list() == ["GB", "IE"]
    # This is the layer's last write, so every row gets its own
    # identity here rather than a later stage supplying it. source_uri is
    # the entity's own former identity, a straight rename; system_uri is a
    # fresh per-row `name://` hash over (source_uri, name_type, name).
    from workspace.derived_uri import name_variant_uri

    assert out["system_uri"].to_list() == [
        name_variant_uri(
            source_uri="gleif://A", name_type="primary", value="Alpha Ltd"
        ),
        name_variant_uri(source_uri="gleif://B", name_type="primary", value="Beta Ltd"),
    ]


def test_finalize_canonical_name_rows_is_a_noop_without_a_sidecar(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import finalize_canonical_name_rows
    from workspace.layer_layout import primary_family_dir

    snapshot_dir = layer_fixture_dir("ie", layer="canonical") / "2026-06-15"
    partition_dir = primary_family_dir(snapshot_dir) / "jurisdiction_code=ie"
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["ie://1"], "jurisdiction_code": ["ie"]}).write_parquet(
        partition_dir / "part-00001.parquet"
    )

    assert (
        finalize_canonical_name_rows("ie", run_date="2026-06-15", roots=workspace_roots)
        == []
    )


def test_finalize_canonical_name_rows_rejects_a_sidecar_with_no_entity_view(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """Without the entity view there is no jurisdiction to join, and writing
    the sidecar with an all-null column would look like a successful run."""
    from acquisition.canonical import finalize_canonical_name_rows
    from workspace.layer_layout import names_family_dir

    snapshot_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    names_dir = names_family_dir(snapshot_dir)
    names_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"system_uri": ["gleif://A"], "name": ["Alpha"], "name_type": ["primary"]}
    ).write_parquet(names_dir / "gleif-names-001.parquet")

    with pytest.raises(FileNotFoundError, match="no canonical entity view"):
        finalize_canonical_name_rows(
            "gleif", run_date="2026-06-15", roots=workspace_roots
        )
