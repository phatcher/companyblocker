from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from acquisition import canonical
from acquisition.constants_status import STATUS_RESEARCH_REQUIRED
from workspace.roots import WorkspaceRoots


def _write_gleif_successor_chain_fixture(
    tmp_path: Path,
    layer_fixture_dir,
    *,
    successors: list[dict[str, str | None]],
    entities: list[dict[str, object]],
    names: list[dict[str, str | None]] | None = None,
) -> tuple[Path, Path]:
    """Write the three on-disk pieces canonicalize_system_successor_chain
    reads: Shard-stage successor edges, canonical entity rows, and the
    canonical names-file family (defaulting to one native LEGAL_NAME row per
    entity, mirroring what canonicalize_system_name_rows would have already
    produced this run).
    """
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)
    entity_by_uri = {row["system_uri"]: row for row in entities}
    successor_rows = [
        {
            "system_uri": row["system_uri"],
            "LEI": row.get("LEI", entity_by_uri[row["system_uri"]]["lei"]),
            "predecessor_name": row.get(
                "predecessor_name", entity_by_uri[row["system_uri"]]["name"]
            ),
            "successor_lei": row["successor_lei"],
            "successor_name": row.get("successor_name"),
        }
        for row in successors
    ]
    pl.DataFrame(successor_rows).write_parquet(
        source_dir / "gleif-successors-001.parquet"
    )

    canonical_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    entity_rows = {
        "system_uri": [row["system_uri"] for row in entities],
        "lei": [row["lei"] for row in entities],
        "name": [row["name"] for row in entities],
        "previous_names": [row.get("previous_names", []) for row in entities],
    }
    pl.DataFrame(
        entity_rows, schema_overrides={"previous_names": pl.List(pl.Utf8)}
    ).write_parquet(canonical_dir / "gleif-001.parquet")

    if names is None:
        names = [
            {
                "system_uri": str(row["system_uri"]),
                "id": str(row["lei"]),
                "name": str(row["name"]),
                "source_type": "LEGAL_NAME",
                "language_code": None,
                "derivation_note": "native LegalName",
                "name_type": "primary",
            }
            for row in entities
        ]
    pl.DataFrame(names).write_parquet(canonical_dir / "gleif-names-001.parquet")

    return source_dir, canonical_dir


def test_canonicalize_system_successor_chain_folds_direct_single_hop(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import canonicalize_system_successor_chain

    _source_dir, canonical_dir = _write_gleif_successor_chain_fixture(
        tmp_path,
        layer_fixture_dir,
        successors=[
            {"system_uri": "gleif://A", "successor_lei": "B", "successor_name": None}
        ],
        entities=[
            {"system_uri": "gleif://A", "lei": "A", "name": "Old Co"},
            {"system_uri": "gleif://B", "lei": "B", "name": "New Co"},
        ],
    )

    written = canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    entities_out = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    previous_by_uri = dict(
        zip(
            entities_out["system_uri"].to_list(),
            entities_out["previous_names"].to_list(),
            strict=True,
        )
    )
    assert previous_by_uri["gleif://B"] == ["Old Co"]
    assert previous_by_uri["gleif://A"] == []

    names_out = pl.read_parquet(canonical_dir / "gleif-names-001.parquet")
    chain_rows = names_out.filter(
        pl.col("source_type") == "SUCCESSOR_CHAIN_PREDECESSOR"
    )
    assert chain_rows.height == 1
    row = chain_rows.row(0, named=True)
    assert row["system_uri"] == "gleif://B"
    assert row["id"] == "B"
    assert row["name"] == "Old Co"
    assert row["name_type"] == "previous"
    assert row["derivation_note"] == "successor chain (predecessor LEI=A)"

    assert {path.name for path in written} == {
        "gleif-001.parquet",
        "gleif-names-001.parquet",
    }


def test_canonicalize_system_successor_chain_folds_into_a_partitioned_snapshot(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """This step rewrites canonical entity files in place, so it has to
    find them in the partitioned layout Canonical now writes -- across
    more than one partition, since a predecessor and its successor need not
    share a jurisdiction. A resolver that missed them would make this step a
    silent no-op, which no flat-layout fixture can detect.
    """
    from acquisition.canonical import canonicalize_system_successor_chain

    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif://A"],
            "LEI": ["A"],
            "predecessor_name": ["Old Co"],
            "successor_lei": ["B"],
            "successor_name": [None],
        },
        schema_overrides={"successor_name": pl.Utf8},
    ).write_parquet(source_dir / "gleif-successors-001.parquet")

    canonical_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    for jurisdiction, uri, lei, name in [
        ("gb", "gleif://A", "A", "Old Co"),
        ("ie", "gleif://B", "B", "New Co"),
    ]:
        partition_dir = canonical_dir / f"jurisdiction_code={jurisdiction}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "system_uri": [uri],
                "lei": [lei],
                "name": [name],
                "previous_names": [[]],
            },
            schema_overrides={"previous_names": pl.List(pl.Utf8)},
        ).write_parquet(partition_dir / "part-00001.parquet")

    pl.DataFrame(
        {
            "system_uri": ["gleif://A", "gleif://B"],
            "id": ["A", "B"],
            "name": ["Old Co", "New Co"],
            "source_type": ["LEGAL_NAME", "LEGAL_NAME"],
            "language_code": [None, None],
            "derivation_note": ["native LegalName"] * 2,
            "name_type": ["primary", "primary"],
        },
        schema_overrides={"language_code": pl.Utf8},
    ).write_parquet(canonical_dir / "gleif-names-001.parquet")

    written = canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    successor_out = pl.read_parquet(
        canonical_dir / "jurisdiction_code=ie" / "part-00001.parquet"
    )
    assert successor_out["previous_names"].to_list() == [["Old Co"]]
    assert {path.name for path in written} == {
        "part-00001.parquet",
        "gleif-names-001.parquet",
    }

    chain_rows = pl.read_parquet(canonical_dir / "gleif-names-001.parquet").filter(
        pl.col("source_type") == "SUCCESSOR_CHAIN_PREDECESSOR"
    )
    assert chain_rows.height == 1
    assert chain_rows.row(0, named=True)["system_uri"] == "gleif://B"


def test_canonicalize_system_successor_chain_multi_hop_folds_onto_terminal_and_intermediate(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_successor_chain

    _source_dir, canonical_dir = _write_gleif_successor_chain_fixture(
        tmp_path,
        layer_fixture_dir,
        successors=[
            {"system_uri": "gleif://A", "successor_lei": "B", "successor_name": None},
            {"system_uri": "gleif://B", "successor_lei": "C", "successor_name": None},
        ],
        entities=[
            {"system_uri": "gleif://A", "lei": "A", "name": "Diamond Bank"},
            {"system_uri": "gleif://B", "lei": "B", "name": "Interim Bank"},
            {"system_uri": "gleif://C", "lei": "C", "name": "Access Bank"},
        ],
    )

    canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    entities_out = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    previous_by_uri = dict(
        zip(
            entities_out["system_uri"].to_list(),
            entities_out["previous_names"].to_list(),
            strict=True,
        )
    )
    # B is only a direct target (of A) -- not itself a terminal for any chain.
    assert previous_by_uri["gleif://B"] == ["Diamond Bank"]
    # C is B's direct target and A's terminal -- both land here, no duplicate.
    assert set(previous_by_uri["gleif://C"]) == {"Interim Bank", "Diamond Bank"}
    assert len(previous_by_uri["gleif://C"]) == 2

    names_out = pl.read_parquet(canonical_dir / "gleif-names-001.parquet")
    chain_rows = names_out.filter(
        pl.col("source_type") == "SUCCESSOR_CHAIN_PREDECESSOR"
    )
    assert chain_rows.height == 3
    pairs = set(zip(chain_rows["system_uri"], chain_rows["name"], strict=True))
    assert pairs == {
        ("gleif://B", "Diamond Bank"),
        ("gleif://C", "Interim Bank"),
        ("gleif://C", "Diamond Bank"),
    }


def test_canonicalize_system_successor_chain_skips_pair_already_captured_natively(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_successor_chain

    _source_dir, canonical_dir = _write_gleif_successor_chain_fixture(
        tmp_path,
        layer_fixture_dir,
        successors=[
            {"system_uri": "gleif://A", "successor_lei": "C", "successor_name": None}
        ],
        entities=[
            {"system_uri": "gleif://A", "lei": "A", "name": "Diamond Bank"},
            {
                "system_uri": "gleif://C",
                "lei": "C",
                "name": "Access Bank",
                "previous_names": ["Diamond Bank"],
            },
        ],
        names=[
            {
                "system_uri": "gleif://A",
                "id": "A",
                "name": "Diamond Bank",
                "source_type": "LEGAL_NAME",
                "language_code": None,
                "derivation_note": "native LegalName",
                "name_type": "primary",
            },
            {
                "system_uri": "gleif://C",
                "id": "C",
                "name": "Access Bank",
                "source_type": "LEGAL_NAME",
                "language_code": None,
                "derivation_note": "native LegalName",
                "name_type": "primary",
            },
            {
                "system_uri": "gleif://C",
                "id": "C",
                "name": "Diamond Bank",
                "source_type": "PREVIOUS_LEGAL_NAME",
                "language_code": None,
                "derivation_note": "native OtherEntityName",
                "name_type": "previous",
            },
        ],
    )

    written = canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    assert written == []
    entities_out = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    previous_by_uri = dict(
        zip(
            entities_out["system_uri"].to_list(),
            entities_out["previous_names"].to_list(),
            strict=True,
        )
    )
    assert previous_by_uri["gleif://C"] == ["Diamond Bank"]
    names_out = pl.read_parquet(canonical_dir / "gleif-names-001.parquet")
    assert names_out.height == 3
    assert (names_out["source_type"] == "SUCCESSOR_CHAIN_PREDECESSOR").sum() == 0


def test_canonicalize_system_successor_chain_skips_cycle_without_hanging(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_successor_chain

    _source_dir, canonical_dir = _write_gleif_successor_chain_fixture(
        tmp_path,
        layer_fixture_dir,
        successors=[
            {"system_uri": "gleif://A", "successor_lei": "B", "successor_name": None},
            {"system_uri": "gleif://B", "successor_lei": "A", "successor_name": None},
        ],
        entities=[
            {"system_uri": "gleif://A", "lei": "A", "name": "Alpha"},
            {"system_uri": "gleif://B", "lei": "B", "name": "Beta"},
        ],
    )

    canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    entities_out = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    previous_by_uri = dict(
        zip(
            entities_out["system_uri"].to_list(),
            entities_out["previous_names"].to_list(),
            strict=True,
        )
    )
    # The mutual (cyclic) edge blocks the *terminal* walk, but each node's
    # direct-hop fold onto the other is unaffected by that and still lands.
    assert previous_by_uri["gleif://A"] == ["Beta"]
    assert previous_by_uri["gleif://B"] == ["Alpha"]


def test_canonicalize_system_successor_chain_handles_dangling_successor_lei(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_successor_chain

    _write_gleif_successor_chain_fixture(
        tmp_path,
        layer_fixture_dir,
        successors=[
            {
                "system_uri": "gleif://A",
                "successor_lei": "GHOST",
                "successor_name": None,
            }
        ],
        entities=[{"system_uri": "gleif://A", "lei": "A", "name": "Old Co"}],
    )

    written = canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    assert written == []


def test_canonicalize_system_successor_chain_is_noop_without_successor_shard_files(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_successor_chain

    canonical_dir = layer_fixture_dir("gleif", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"system_uri": ["gleif://A"], "lei": ["A"], "name": ["Old Co"]}
    ).write_parquet(canonical_dir / "gleif-001.parquet")
    pl.DataFrame(
        {
            "system_uri": ["gleif://A"],
            "id": ["A"],
            "name": ["Old Co"],
            "source_type": ["LEGAL_NAME"],
            "language_code": [None],
            "derivation_note": ["native LegalName"],
            "name_type": ["primary"],
        }
    ).write_parquet(canonical_dir / "gleif-names-001.parquet")

    assert (
        canonicalize_system_successor_chain(
            "gleif", run_date="2026-06-15", roots=workspace_roots
        )
        == []
    )


def test_canonicalize_system_successor_chain_research_policy_denies_stage(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import canonicalize_system_successor_chain

    denied_plan = SimpleNamespace(
        code="gleif",
        status=STATUS_RESEARCH_REQUIRED,
        notes="research pending",
        research=SimpleNamespace(
            allow_research_runtime=False, allowed_stages=("shard",)
        ),
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=denied_plan
    )

    with pytest.raises(
        RuntimeError, match="research execution policy denies stage 'canonical'"
    ):
        canonicalize_system_successor_chain(
            "gleif", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("gleif")


def test_canonicalize_system_successor_chain_blocked_status_rejected_even_with_allow_research(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import canonicalize_system_successor_chain

    blocked_plan = SimpleNamespace(
        code="gleif",
        status="blocked",
        notes="No access",
        research=SimpleNamespace(
            allow_research_runtime=True, allowed_stages=("canonical",)
        ),
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=blocked_plan
    )

    with pytest.raises(RuntimeError, match="gleif: blocked - No access"):
        canonicalize_system_successor_chain(
            "gleif", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("gleif")


def test_canonicalize_system_successor_chain_appends_into_existing_names_file(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """New chain-derived rows must land inside the names-file family that
    already exists (canonicalize_system_name_rows's output), not a new,
    separately-named file -- consumers glob the whole family together.
    """
    from acquisition.canonical import canonicalize_system_successor_chain

    _source_dir, canonical_dir = _write_gleif_successor_chain_fixture(
        tmp_path,
        layer_fixture_dir,
        successors=[
            {"system_uri": "gleif://A", "successor_lei": "B", "successor_name": None}
        ],
        entities=[
            {"system_uri": "gleif://A", "lei": "A", "name": "Old Co"},
            {"system_uri": "gleif://B", "lei": "B", "name": "New Co"},
        ],
    )
    names_files_before = {
        path.name for path in canonical_dir.glob("gleif-names-*.parquet")
    }

    canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    names_files_after = {
        path.name for path in canonical_dir.glob("gleif-names-*.parquet")
    }
    assert names_files_after == names_files_before == {"gleif-names-001.parquet"}
    assert pl.read_parquet(canonical_dir / "gleif-names-001.parquet").height == 3


def test_canonicalize_system_successor_chain_is_idempotent_across_repeated_runs(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_successor_chain

    _source_dir, canonical_dir = _write_gleif_successor_chain_fixture(
        tmp_path,
        layer_fixture_dir,
        successors=[
            {"system_uri": "gleif://A", "successor_lei": "B", "successor_name": None}
        ],
        entities=[
            {"system_uri": "gleif://A", "lei": "A", "name": "Old Co"},
            {"system_uri": "gleif://B", "lei": "B", "name": "New Co"},
        ],
    )

    canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )
    entities_after_first = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    names_after_first = pl.read_parquet(canonical_dir / "gleif-names-001.parquet")

    written_second = canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )
    entities_after_second = pl.read_parquet(canonical_dir / "gleif-001.parquet")
    names_after_second = pl.read_parquet(canonical_dir / "gleif-names-001.parquet")

    assert written_second == []
    assert entities_after_second.equals(entities_after_first)
    assert names_after_second.equals(names_after_first)
    chain_rows = names_after_second.filter(
        pl.col("source_type") == "SUCCESSOR_CHAIN_PREDECESSOR"
    )
    assert chain_rows.height == 1
