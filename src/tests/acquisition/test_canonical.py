from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from acquisition import canonical
from acquisition.canonical import (
    canonicalize_system_name_rows,
    canonicalize_system_primary_name_override,
    canonicalize_system_shards,
    canonicalize_system_successor_chain,
    derive_system_name_rows,
    finalize_canonical_name_rows,
)
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def test_canonical_stage_runs_gleif_shards_through_every_pass_in_pipeline_order(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """One synthetic GLEIF snapshot through the canonical stage's real call
    order (mirrors `scripts/process_companies.py`): derive name rows,
    canonicalize the entity shards, canonicalize the native name rows,
    promote a non-Latin primary name to its transliteration, fold a
    successor's predecessor name onto it, then finalize the name-variant
    sidecar. Proves the stage's modules -- `canonical_frame_transform`,
    `canonical_dedupe`, `canonical_sampling`, `canonical_utils`,
    `name_variant_uri`, `company_type_registry`, `workspace.layer_layout`
    and `workspace.data_file_naming` -- are wired together correctly on one
    real path; each function's own branches are covered directly in
    `test_canonical_shards.py`, `test_canonical_name_rows.py`,
    `test_canonical_primary_name_override.py`,
    `test_canonical_successor_chain.py` and
    `test_canonical_finalize_name_rows.py`.
    """
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)

    # The surviving entity: a live GLEIF record whose current primary name is
    # non-Latin script.
    pl.DataFrame(
        {
            "system_uri": ["gleif://529900T8BM49AURSDO55"],
            "LEI": ["529900T8BM49AURSDO55"],
            "Entity_LegalName": ["マルチチュード"],
            "Entity_LegalJurisdiction": ["GB"],
            "Entity_EntityStatus": ["ACTIVE"],
            "Entity_EntityCreationDate": ["2015-03-10"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": ["03856000"],
        }
    ).write_parquet(source_dir / "gleif-001.parquet")

    # GLEIF writes its native name-variant rows straight off the parsed XML
    # at Shard stage: one row in the source-native vocabulary the surviving
    # entity's legal name, one its preferred ASCII transliteration.
    pl.DataFrame(
        {
            "system_uri": ["gleif://529900T8BM49AURSDO55"] * 2,
            "LEI": ["529900T8BM49AURSDO55"] * 2,
            "name": ["マルチチュード", "MULTITUDE LTD"],
            "source_type": [
                "LEGAL_NAME",
                "PREFERRED_ASCII_TRANSLITERATED_LEGAL_NAME",
            ],
            "language_code": ["ja", None],
            "derivation_note": ["native LegalName", "native OtherEntityName"],
        }
    ).write_parquet(source_dir / "gleif-names-001.parquet")

    # A predecessor that merged into the surviving entity: GLEIF's
    # SuccessorLEI edge is captured at Shard stage, before the predecessor's
    # own entity row is dropped from the canonical view.
    pl.DataFrame(
        {
            "system_uri": ["gleif://OLDMULTITUDEPREDECESSOR0"],
            "LEI": ["OLDMULTITUDEPREDECESSOR0"],
            "predecessor_name": ["MULTITUDE HOLDINGS LTD"],
            "successor_lei": ["529900T8BM49AURSDO55"],
        }
    ).write_parquet(source_dir / "gleif-successors-001.parquet")

    assert (
        derive_system_name_rows("gleif", run_date="2026-06-17", roots=workspace_roots)
        == []
    )

    entity_paths = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    assert len(entity_paths) == 1
    entity_row = pl.read_parquet(entity_paths[0]).row(0, named=True)
    assert entity_row["jurisdiction_code"] == "GB"
    assert entity_row["company_number"] == "03856000"
    assert entity_row["name"] == "マルチチュード"

    name_paths = canonicalize_system_name_rows(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    assert len(name_paths) == 1
    assert sorted(pl.read_parquet(name_paths[0])["name_type"].to_list()) == [
        "primary",
        "transliteration_preferred",
    ]

    override_paths = canonicalize_system_primary_name_override(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    assert len(override_paths) == 1
    entity_row = pl.read_parquet(entity_paths[0]).row(0, named=True)
    assert entity_row["name"] == "MULTITUDE LTD"

    successor_paths = canonicalize_system_successor_chain(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    assert successor_paths
    entity_row = pl.read_parquet(entity_paths[0]).row(0, named=True)
    assert entity_row["previous_names"] == ["MULTITUDE HOLDINGS LTD"]

    finalized_paths = finalize_canonical_name_rows(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    assert finalized_paths
    finalized = pl.concat([pl.read_parquet(path) for path in finalized_paths])
    assert finalized["jurisdiction_code"].to_list() == ["GB"] * finalized.height
    assert finalized["system_uri"].n_unique() == finalized.height
    assert set(finalized["name_type"].to_list()) == {
        "primary",
        "transliteration_preferred",
        "previous",
    }


def test_a_systems_own_identifier_column_becomes_id_in_the_canonical_names() -> None:
    """Shard keeps each system's own spelling, so GLEIF's name rows arrive
    carrying `LEI`; canonical is one shape, so they leave carrying `id`, as
    every other system's already do."""
    gleif = pl.DataFrame({"system_uri": ["gleif://A"], "LEI": ["A"], "name": ["ACME"]})
    already = pl.DataFrame({"system_uri": ["gb://1"], "id": ["1"], "name": ["ACME"]})

    renamed = canonical._name_identifier_as_id(
        gleif, identifier_candidates=("LEI",)
    )
    untouched = canonical._name_identifier_as_id(
        already, identifier_candidates=("CompanyNumber",)
    )

    assert renamed.columns == ["system_uri", "id", "name"]
    assert renamed.get_column("id").to_list() == ["A"]
    assert untouched.equals(already)
