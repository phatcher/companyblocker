from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from acquisition import canonical
from acquisition.constants_status import STATUS_RESEARCH_REQUIRED
from acquisition.wikidata_name_rows import NAME_ROW_COLUMNS
from workspace.roots import WorkspaceRoots


def test_canonicalize_system_name_rows_maps_gleif_source_type_to_shared_vocabulary(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_name_rows

    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-15"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gleif://ABC123", "gleif://ABC123"],
            "LEI": ["ABC123", "ABC123"],
            "name": ["Alpha Ltd", "Alpha Old One"],
            "source_type": ["LEGAL_NAME", "PREVIOUS_LEGAL_NAME"],
            "language_code": ["en", None],
            "derivation_note": ["native LegalName", "native OtherEntityName"],
        }
    ).write_parquet(data_dir / "gleif-names-001.parquet")

    written = canonicalize_system_name_rows(
        "gleif", run_date="2026-06-15", roots=workspace_roots
    )

    assert [path.name for path in written] == ["gleif-names-001.parquet"]
    out = pl.read_parquet(written[0])
    assert out["name_type"].to_list() == ["primary", "previous"]
    # source_type and every other Shard-stage column survive unchanged.
    assert out["source_type"].to_list() == ["LEGAL_NAME", "PREVIOUS_LEGAL_NAME"]


def test_canonicalize_system_name_rows_fails_loudly_on_unmapped_source_type(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_name_rows

    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-15"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gleif://ABC123"],
            "LEI": ["ABC123"],
            "name": ["Alpha Mystery Name"],
            "source_type": ["SOME_FUTURE_TYPE_NOT_YET_MAPPED"],
            "language_code": [None],
            "derivation_note": ["native OtherEntityName"],
        }
    ).write_parquet(data_dir / "gleif-names-001.parquet")

    with pytest.raises(RuntimeError, match="name_variant_type_map"):
        canonicalize_system_name_rows(
            "gleif", run_date="2026-06-15", roots=workspace_roots
        )


def test_canonicalize_system_name_rows_is_a_noop_for_systems_without_type_map(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_name_rows

    # "ie" has neither a name_variant_type_map nor a registered deriver
    # -- "fr" no longer illustrates this
    # case since its loader registered both.
    data_dir = layer_fixture_dir("ie", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    assert (
        canonicalize_system_name_rows(
            "ie", run_date="2026-06-15", roots=workspace_roots
        )
        == []
    )


def test_canonicalize_system_name_rows_research_policy_denies_stage(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import canonicalize_system_name_rows

    denied_plan = SimpleNamespace(
        code="xx",
        status=STATUS_RESEARCH_REQUIRED,
        notes="research pending",
        name_variant_type_map=(("primary", "primary"),),
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
        canonicalize_system_name_rows(
            "xx", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("xx")


def test_canonicalize_system_name_rows_blocked_status_rejected_even_with_allow_research(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import canonicalize_system_name_rows

    blocked_plan = SimpleNamespace(
        code="xx",
        status="blocked",
        notes="No access",
        name_variant_type_map=(("primary", "primary"),),
        research=SimpleNamespace(
            allow_research_runtime=True, allowed_stages=("canonical",)
        ),
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=blocked_plan
    )

    with pytest.raises(RuntimeError, match="xx: blocked - No access"):
        canonicalize_system_name_rows(
            "xx", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("xx")


def _write_wikidata_shard_fixture(data_dir: Path) -> None:
    variant_dtype = pl.List(pl.Struct({"value": pl.Utf8, "language": pl.Utf8}))
    pl.DataFrame(
        {
            "system_uri": ["wikidata:Q8093"],
            "id": ["Q8093"],
            "label_en": ["Nintendo"],
            "aliases_en": [["NCL"]],
            "official_name_variants": [
                [
                    {"value": "任天堂株式会社", "language": "ja"},
                    {"value": "Nintendo Co., Ltd.", "language": "en"},
                ]
            ],
            "short_name_variants": [[{"value": "Nintendo", "language": "en"}]],
        },
        schema_overrides={
            "official_name_variants": variant_dtype,
            "short_name_variants": variant_dtype,
        },
    ).write_parquet(data_dir / "wikidata-001.parquet")


def test_derive_system_name_rows_writes_wikidata_names_files_alongside_shards(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import derive_system_name_rows

    data_dir = layer_fixture_dir("wikidata", layer="source") / "2026-07-16"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_wikidata_shard_fixture(data_dir)

    written = derive_system_name_rows(
        "wikidata", run_date="2026-07-16", roots=workspace_roots, allow_research=True
    )

    assert [path.name for path in written] == ["wikidata-names-001.parquet"]
    assert written[0].parent == data_dir
    out = pl.read_parquet(written[0])
    assert list(out.columns) == list(NAME_ROW_COLUMNS)
    assert out.height == 5


def test_derive_system_name_rows_ignores_sidecar_and_replaces_stale_names_file(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import derive_system_name_rows

    data_dir = layer_fixture_dir("wikidata", layer="source") / "2026-07-16"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_wikidata_shard_fixture(data_dir)
    pl.DataFrame(
        {"system_uri": ["wikidata:Q8093"], "description_en": ["junk"]}
    ).write_parquet(data_dir / "wikidata-sidecar-001.parquet")
    pl.DataFrame({"stale": ["junk-row"]}).write_parquet(
        data_dir / "wikidata-names-001.parquet"
    )

    written = derive_system_name_rows(
        "wikidata", run_date="2026-07-16", roots=workspace_roots, allow_research=True
    )

    assert [path.name for path in written] == ["wikidata-names-001.parquet"]
    out = pl.read_parquet(written[0])
    assert "stale" not in out.columns
    assert out.height == 5


def test_derive_system_name_rows_is_a_noop_for_systems_without_a_registered_deriver(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import derive_system_name_rows

    # "ie" has no registered deriver --
    # "fr" no longer illustrates this case since its loader registered one.
    data_dir = layer_fixture_dir("ie", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    assert (
        derive_system_name_rows("ie", run_date="2026-06-01", roots=workspace_roots)
        == []
    )


def test_derive_system_name_rows_research_policy_denies_stage(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import derive_system_name_rows

    denied_plan = SimpleNamespace(
        code="gb",
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
        derive_system_name_rows(
            "gb", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("gb")


def test_derive_system_name_rows_blocked_status_rejected_even_with_allow_research(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import derive_system_name_rows

    blocked_plan = SimpleNamespace(
        code="gb",
        status="blocked",
        notes="No access",
        research=SimpleNamespace(
            allow_research_runtime=True, allowed_stages=("canonical",)
        ),
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=blocked_plan
    )

    with pytest.raises(RuntimeError, match="gb: blocked - No access"):
        derive_system_name_rows(
            "gb", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("gb")


def test_canonicalize_system_name_rows_maps_wikidata_source_type_to_shared_vocabulary(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_name_rows

    data_dir = layer_fixture_dir("wikidata", layer="source") / "2026-07-16"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["wikidata:Q8093"] * 4,
            "id": ["Q8093"] * 4,
            "name": ["Nintendo", "NCL", "Nintendo Co., Ltd.", "Nintendo"],
            "source_type": ["LABEL_EN", "ALIAS_EN", "OFFICIAL_NAME", "SHORT_NAME"],
            "language_code": ["en", "en", "en", "en"],
            "derivation_note": ["note"] * 4,
        }
    ).write_parquet(data_dir / "wikidata-names-001.parquet")

    written = canonicalize_system_name_rows(
        "wikidata", run_date="2026-07-16", roots=workspace_roots, allow_research=True
    )

    out = pl.read_parquet(written[0])
    assert out["name_type"].to_list() == ["label", "alias", "official", "short"]
    assert out["id"].to_list() == ["Q8093"] * 4
    assert out["language_code"].to_list() == ["en", "en", "en", "en"]


def test_derive_and_canonicalize_name_rows_round_trip_for_wikidata(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    from acquisition.canonical import (
        canonicalize_system_name_rows,
        derive_system_name_rows,
    )

    data_dir = layer_fixture_dir("wikidata", layer="source") / "2026-07-16"
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_wikidata_shard_fixture(data_dir)

    derived = derive_system_name_rows(
        "wikidata", run_date="2026-07-16", roots=workspace_roots, allow_research=True
    )
    assert derived

    canonicalized = canonicalize_system_name_rows(
        "wikidata", run_date="2026-07-16", roots=workspace_roots, allow_research=True
    )
    out = pl.read_parquet(canonicalized[0])
    assert out.height == 5
    assert set(out["name_type"].to_list()) == {"label", "alias", "official", "short"}


def test_derive_system_name_rows_writes_gb_names_files_from_sidecar_alongside_shards(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import derive_system_name_rows

    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb://12345678"],
            "CompanyNumber": ["12345678"],
            "CompanyName": ["Alpha Ltd"],
        }
    ).write_parquet(data_dir / "gb-001.parquet")
    pl.DataFrame(
        {
            "system_uri": ["gb://12345678"],
            "PreviousName_1.CompanyName": ["Alpha Old One Ltd"],
            "PreviousName_1.CONDATE": ["01/01/2020"],
            "PreviousName_2.CompanyName": ["Alpha Older One Ltd"],
            "PreviousName_2.CONDATE": ["01/01/2010"],
        }
    ).write_parquet(data_dir / "gb-sidecar-001.parquet")

    written = derive_system_name_rows(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )

    assert [path.name for path in written] == ["gb-names-001.parquet"]
    assert written[0].parent == data_dir
    out = pl.read_parquet(written[0])
    assert list(out.columns) == list(NAME_ROW_COLUMNS)
    assert out.height == 2
    assert out["id"].to_list() == ["12345678", "12345678"]
    assert out["name"].to_list() == ["Alpha Old One Ltd", "Alpha Older One Ltd"]
    assert out["source_type"].to_list() == ["PREVIOUS_NAME", "PREVIOUS_NAME"]


def test_canonicalize_system_name_rows_maps_gb_source_type_to_shared_vocabulary(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_name_rows

    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb://12345678"],
            "id": ["12345678"],
            "name": ["Alpha Old One Ltd"],
            "source_type": ["PREVIOUS_NAME"],
            "language_code": [None],
            "derivation_note": ["Companies House PreviousName_N sidecar field"],
        }
    ).write_parquet(data_dir / "gb-names-001.parquet")

    written = canonicalize_system_name_rows(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )

    assert [path.name for path in written] == ["gb-names-001.parquet"]
    out = pl.read_parquet(written[0])
    assert out["name_type"].to_list() == ["previous"]


def test_derive_system_name_rows_writes_fr_names_files_from_sidecar_alongside_shards(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import derive_system_name_rows

    data_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr://552081317"],
            "siren": ["552081317"],
            "denominationUniteLegale": ["Alpha SA"],
        }
    ).write_parquet(data_dir / "fr-001.parquet")
    pl.DataFrame(
        {
            "system_uri": ["fr://552081317"],
            "sigleUniteLegale": ["ALP"],
            "denominationUsuelle1UniteLegale": ["[ND]"],
        }
    ).write_parquet(data_dir / "fr-sidecar-001.parquet")

    written = derive_system_name_rows(
        "fr", run_date="2026-06-15", roots=workspace_roots
    )

    assert [path.name for path in written] == ["fr-names-001.parquet"]
    assert written[0].parent == data_dir
    out = pl.read_parquet(written[0])
    assert list(out.columns) == list(NAME_ROW_COLUMNS)
    assert out.height == 1
    assert out["id"].to_list() == ["552081317"]
    assert out["name"].to_list() == ["ALP"]
    assert out["source_type"].to_list() == ["ACRONYM"]


def test_canonicalize_system_name_rows_maps_fr_source_type_to_shared_vocabulary(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    from acquisition.canonical import canonicalize_system_name_rows

    data_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["fr://552081317"] * 2,
            "id": ["552081317"] * 2,
            "name": ["ALP", "Alpha Trading"],
            "source_type": ["ACRONYM", "USUAL_NAME"],
            "language_code": [None, None],
            "derivation_note": [
                "SIRENE sigleUniteLegale",
                "SIRENE denominationUsuelleNUniteLegale",
            ],
        }
    ).write_parquet(data_dir / "fr-names-001.parquet")

    written = canonicalize_system_name_rows(
        "fr", run_date="2026-06-15", roots=workspace_roots
    )

    assert [path.name for path in written] == ["fr-names-001.parquet"]
    out = pl.read_parquet(written[0])
    assert out["name_type"].to_list() == ["acronym", "usual_name"]
