from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from acquisition import canonical
from acquisition.canonical import CANONICAL_OUTPUT_COLUMNS, canonicalize_system_shards
from acquisition.constants_status import STATUS_RESEARCH_REQUIRED
from acquisition.models_plan import SourceResource
from workspace.data_layout import data_root
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration

CANONICAL_PREFIX_COLUMNS = list(CANONICAL_OUTPUT_COLUMNS[:7])


def _assert_canonical_prefix_columns(df: pl.DataFrame) -> None:
    assert df.columns[:7] == CANONICAL_PREFIX_COLUMNS


def test_canonicalize_country_shards_maps_gb_contract_only_columns(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "CompanyNumber": ["12345678"],
            "CompanyName": ["Example Limited"],
            "system_uri": ["gb://12345678"],
            "CompanyCategory": ["Private Limited Company"],
            "CompanyStatus": ["Active"],
            "IncorporationDate": ["2020-01-02"],
            "RegAddress.AddressLine1": ["1 High Street"],
            "RegAddress.PostTown": ["London"],
            "RegAddress.PostCode": ["SW1A 1AA"],
            "RegAddress.Country": ["ENGLAND"],
            "URI": [
                "https://find-and-update.company-information.service.gov.uk/company/12345678"
            ],
            "extra_raw_column": ["keep-me"],
        }
    )
    source.write_parquet(data_dir / "gb-001.parquet")

    written = canonicalize_system_shards(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )

    assert len(written) == 1
    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)

    _assert_canonical_prefix_columns(out)

    assert row["jurisdiction_code"] == "gb"
    assert row["company_number"] == "12345678"
    assert row["system_uri"] == "gb://12345678"
    assert row["name"] == "Example Limited"
    assert row["company_type"] == "Limited"
    assert row["current_status"] == "Active"
    assert row["incorporation_date"] == "2020-01-02"
    assert row["registered_address_in_full"] == "1 High Street, London, SW1A 1AA"
    assert row["registry_url"].startswith(
        "https://find-and-update.company-information.service.gov.uk/"
    )
    assert row["metadata_generated_utc"] is not None
    assert "CompanyName" not in out.columns
    assert "RegAddress.Country" not in out.columns
    assert "extra_raw_column" not in out.columns


def test_canonicalize_country_shards_maps_fr_dates_and_status(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    get_company_type_exclusions = mocker.patch.object(
        canonical,
        "get_company_type_exclusions",
        return_value=set(),
    )

    data_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["fr://552100554"],
            "siren": ["552100554"],
            "denominationUniteLegale": ["EXEMPLE SARL"],
            "categorieJuridiqueUniteLegale": ["5499"],
            "etatAdministratifUniteLegale": ["C"],
            "dateCreationUniteLegale": ["20091231"],
            "dateCessationUniteLegale": ["20220105"],
        }
    )
    source.write_parquet(data_dir / "fr-001.parquet")

    written = canonicalize_system_shards(
        "fr", run_date="2026-06-15", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)

    _assert_canonical_prefix_columns(out)

    assert row["jurisdiction_code"] == "fr"
    assert row["company_number"] == "552100554"
    assert row["name"] == "EXEMPLE SARL"
    assert (
        row["company_type"]
        == "Société à responsabilité limitée (sans autre indication)"
    )
    assert row["incorporation_date"] == "2009-12-31"
    assert row["dissolution_date"] == "2022-01-05"
    assert row["current_status"] == "C"
    get_company_type_exclusions.assert_called_once_with("fr")


def test_canonicalize_ie_applies_source_alias_without_duplicate_column(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("ie", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "company_name": ["Example Ireland Limited"],
            "company_num": ["123456"],
            "status": ["Active"],
            "system_uri": ["ie://123456"],
        }
    )
    source.write_parquet(data_dir / "ie-001.parquet")

    written = canonicalize_system_shards(
        "ie", run_date="2026-06-01", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)

    assert row["company_number"] == "123456"
    assert row["system_uri"] == "ie://123456"
    assert "company_num" not in out.columns


def test_canonicalize_ie_removes_chunks_scratch_dir_after_run(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("ie", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "company_name": ["Example Ireland Limited"],
            "company_num": ["123456"],
            "status": ["Active"],
            "system_uri": ["ie://123456"],
        }
    ).write_parquet(data_dir / "ie-001.parquet")

    canonicalize_system_shards("ie", run_date="2026-06-01", roots=workspace_roots)

    output_dir = layer_fixture_dir("ie", layer="canonical") / "2026-06-01"
    assert not (output_dir / "chunks").exists()


def test_canonicalize_ie_cleans_stale_chunks_scratch_dir_before_run(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("ie", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "company_name": ["Example Ireland Limited"],
            "company_num": ["123456"],
            "status": ["Active"],
            "system_uri": ["ie://123456"],
        }
    ).write_parquet(data_dir / "ie-001.parquet")

    output_dir = layer_fixture_dir("ie", layer="canonical") / "2026-06-01"
    stale_chunks_dir = output_dir / "chunks"
    stale_chunks_dir.mkdir(parents=True, exist_ok=True)
    (stale_chunks_dir / "leftover.parquet").write_bytes(b"not a real parquet file")

    written = canonicalize_system_shards(
        "ie", run_date="2026-06-01", roots=workspace_roots
    )

    assert not stale_chunks_dir.exists()
    assert len(written) == 1


def test_canonicalize_ie_writes_partitioned_output_when_partition_by_set(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("ie", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "company_name": ["Example Ireland Limited"],
            "company_num": ["123456"],
            "status": ["Active"],
            "system_uri": ["ie://123456"],
        }
    ).write_parquet(data_dir / "ie-001.parquet")

    mocker.patch.object(
        SourceResource,
        "resolve_output_partition_by",
        return_value="jurisdiction_code",
    )

    written = canonicalize_system_shards(
        "ie", run_date="2026-06-01", roots=workspace_roots
    )

    assert len(written) == 1
    relative = written[0].relative_to(
        layer_fixture_dir("ie", layer="canonical") / "2026-06-01"
    )
    assert relative.as_posix() == "primary/jurisdiction_code=ie/part-00001.parquet"


@pytest.mark.parametrize(
    "company_name,company_category,expected_company_type",
    [
        (
            "Example Fund",
            "Investment Company with Variable Capital",
            "Investment Company",
        ),
        ("Example Limited", "Private Limited Company", "Limited"),
    ],
)
def test_canonicalize_country_shards_maps_gb_category_to_canonical_label(
    tmp_path: Path,
    layer_fixture_dir,
    company_name: str,
    company_category: str,
    expected_company_type: str,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gb://87654321"],
            "CompanyNumber": ["87654321"],
            "CompanyName": [company_name],
            "CompanyCategory": [company_category],
            "CompanyStatus": ["Active"],
        }
    )
    source.write_parquet(data_dir / "gb-001.parquet")

    written = canonicalize_system_shards(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)

    _assert_canonical_prefix_columns(out)
    assert row["company_type"] == expected_company_type


def test_canonicalize_country_shards_maps_fr_person_full_name(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    get_company_type_exclusions = mocker.patch.object(
        canonical,
        "get_company_type_exclusions",
        return_value=set(),
    )

    data_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["fr://123456789"],
            "siren": ["123456789"],
            "denominationUniteLegale": [None],
            "prenom1UniteLegale": ["JEAN"],
            "nomUniteLegale": ["DUPONT"],
            "categorieJuridiqueUniteLegale": ["1000"],
            "etatAdministratifUniteLegale": ["A"],
            "dateCreationUniteLegale": ["20200101"],
        }
    )
    source.write_parquet(data_dir / "fr-001.parquet")

    written = canonicalize_system_shards(
        "fr", run_date="2026-06-15", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)

    _assert_canonical_prefix_columns(out)

    assert row["company_number"] == "123456789"
    assert row["name"] == "JEAN DUPONT"
    assert "CompanyName" not in out.columns
    get_company_type_exclusions.assert_called_once_with("fr")


def test_canonicalize_country_shards_fails_when_no_shards(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    (data_root(workspace_roots) / "ie").mkdir(parents=True, exist_ok=True)

    try:
        canonicalize_system_shards("ie", run_date="2026-06-15", roots=workspace_roots)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as exc:
        assert "No sharded files found" in str(exc)


def test_canonicalize_country_shards_ignores_sidecar_shards(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gb://12345678"],
            "CompanyNumber": ["12345678"],
            "CompanyName": ["Example Limited"],
            "CompanyCategory": ["Private Limited Company"],
            "CompanyStatus": ["Active"],
        }
    ).write_parquet(data_dir / "gb-001.parquet")

    # Sidecar shards share the run folder but should not be canonicalized as inputs.
    pl.DataFrame({"irrelevant": ["x"]}).write_parquet(
        data_dir / "gb-sidecar-001.parquet"
    )

    written = canonicalize_system_shards(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )

    assert len(written) == 1
    out = pl.read_parquet(written[0])
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["company_number"] == "12345678"


def test_canonicalize_country_shards_ignores_name_variant_shards(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gb://12345678"],
            "CompanyNumber": ["12345678"],
            "CompanyName": ["Example Limited"],
            "CompanyCategory": ["Private Limited Company"],
            "CompanyStatus": ["Active"],
        }
    ).write_parquet(data_dir / "gb-001.parquet")

    # Name-variant shards share the run folder but must not be canonicalized
    # as if they were per-entity rows -- their schema is entirely different.
    pl.DataFrame({"name": ["x"], "source_type": ["LEGAL_NAME"]}).write_parquet(
        data_dir / "gb-names-001.parquet"
    )

    written = canonicalize_system_shards(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )

    assert len(written) == 1
    out = pl.read_parquet(written[0])
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["company_number"] == "12345678"


def test_canonicalize_country_shards_chunking_uses_filtered_rows(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    shard_one = pl.DataFrame(
        {
            "system_uri": ["fr://1", "fr://2"],
            "siren": ["1", "2"],
            "denominationUniteLegale": ["ALPHA", "BETA"],
            "categorieJuridiqueUniteLegale": ["KEEP", "DROP"],
            "etatAdministratifUniteLegale": ["A", "A"],
            "dateCreationUniteLegale": ["20200101", "20200101"],
        }
    )
    shard_two = pl.DataFrame(
        {
            "system_uri": ["fr://3", "fr://4"],
            "siren": ["3", "4"],
            "denominationUniteLegale": ["GAMMA", "DELTA"],
            "categorieJuridiqueUniteLegale": ["KEEP", "KEEP"],
            "etatAdministratifUniteLegale": ["A", "A"],
            "dateCreationUniteLegale": ["20200101", "20200101"],
        }
    )
    shard_one.write_parquet(data_dir / "fr-001.parquet")
    shard_two.write_parquet(data_dir / "fr-002.parquet")

    get_system_company_type_mapping = mocker.patch.object(
        canonical,
        "get_system_company_type_mapping",
        return_value={},
    )
    get_company_type_exclusions = mocker.patch.object(
        canonical,
        "get_company_type_exclusions",
        return_value={"DROP"},
    )

    written = canonicalize_system_shards(
        "fr", run_date="2026-06-15", roots=workspace_roots, partition_rows_per_file=2
    )

    # Four source rows, one excluded by company type: the excluded row must
    # not occupy a slot in the 2-rows-per-file partitioned output, so three
    # kept rows fill one full file and start a second.
    assert [
        path.relative_to(tmp_path).as_posix().split("2026-06-01/")[1]
        for path in written
    ] == [
        "primary/jurisdiction_code=fr/part-00001.parquet",
        "primary/jurisdiction_code=fr/part-00002.parquet",
    ]
    assert [pl.read_parquet(path).height for path in written] == [2, 1]
    assert get_system_company_type_mapping.call_count == 2
    get_system_company_type_mapping.assert_any_call("fr")
    get_company_type_exclusions.assert_called_once_with("fr")


def test_canonicalize_system_shards_supports_country_as_system(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gb://00000001"],
            "CompanyNumber": ["00000001"],
            "CompanyName": ["EXAMPLE LTD"],
            "CompanyStatus": ["Active"],
            "CompanyCategory": ["Investment Company with Variable Capital"],
            "IncorporationDate": ["2020-01-01"],
        }
    )
    source.write_parquet(data_dir / "gb-001.parquet")

    written = canonicalize_system_shards(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["jurisdiction_code"] == "gb"
    assert row["company_type"] == "Investment Company"


def test_canonicalize_stopped_dbpedia_rejects_even_with_allow_research(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """DBpedia's catalog status is `stopped` (acquisition deliberately
    discontinued, see catalog/systems/dbpedia.json), not `research_required`
    -- `allow_research` only ever unlocks `research_required`, never
    `stopped`, so this must still reject regardless. Superseded a prior
    version of this test asserting the opposite (that dbpedia canonicalized
    successfully under allow_research), back when dbpedia's status was still
    `research_required`."""
    data_dir = layer_fixture_dir("dbpedia", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "CompanyName": ["Acme Corp"],
            "dbpedia_uri": ["http://dbpedia.org/resource/Acme_Corp"],
        }
    ).write_parquet(data_dir / "dbpedia-001.parquet")

    with pytest.raises(RuntimeError, match="stopped"):
        canonicalize_system_shards(
            "dbpedia",
            run_date="2026-06-26",
            roots=workspace_roots,
            allow_research=True,
        )


def test_canonicalize_frame_maps_wikidata_country_qid_to_iso2_and_registry_url():
    frame = pl.DataFrame(
        {
            "id": ["Q100"],
            "label_en": ["Acme Corp"],
            "entity_type": ["item"],
            "inception": ["+2000-01-01T00:00:00Z"],
            "dissolved": [None],
            "country": [["Q145"]],
            "company_number_gb": ["123456789"],
        }
    )

    out = canonical._canonicalize_frame(
        frame,
        country="wikidata",
        source_company_type_col=None,
        exclude_company_type_values=set(),
        canonical_source_column_aliases=None,
        system_field_candidates={
            "name": ["label_en"],
            "company_type": ["entity_type"],
            "current_status": ["entity_type"],
            "incorporation_date": ["inception"],
            "dissolution_date": ["dissolved"],
            "registry_url": ["registry_url", "id"],
            "registered_address_in_full": ["label_en"],
            "inactive": ["inactive"],
            "branch": ["branch"],
            "branch_status": ["branch_status"],
        },
    )

    row = out.row(0, named=True)
    assert row["jurisdiction_code"] == "GB"
    assert row["company_number"] == "123456789"
    assert row["registry_url"] == "https://www.wikidata.org/wiki/Q100"
    assert row["registered_address_in_full"] is None
    assert row["company_type"] is None
    assert row["current_status"] == "Active"


def test_canonicalize_offeneregister_derives_company_number_from_register_fields(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("offeneregister", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "name": ["Example GmbH"],
            "register_art": ["HRB"],
            "register_number": ["12345"],
            "system_uri": ["offeneregister://hrb/12345"],
        }
    )
    source.write_parquet(data_dir / "offeneregister-001.parquet")

    written = canonicalize_system_shards(
        "offeneregister", run_date="2026-06-01", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)

    assert row["company_number"] == "HRB 12345"
    assert row["name"] == "Example GmbH"


def test_canonicalize_offeneregister_scopes_company_number_by_resolved_court(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("offeneregister", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "name": ["Example GmbH"],
            "register_art": ["HRB"],
            "register_number": ["150148"],
            # R1101 is a mapped court in the xjustiz crosswalk (RA000234).
            "company_number": ["R1101_HRB150148"],
            "system_uri": ["offeneregister://R1101_HRB150148"],
        }
    )
    source.write_parquet(data_dir / "offeneregister-001.parquet")

    written = canonicalize_system_shards(
        "offeneregister", run_date="2026-06-01", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] == "RA000234|HRB150148"


def test_canonicalize_offeneregister_uses_current_court_for_merged_history(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("offeneregister", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "name": ["Example GmbH"],
            "register_art": ["HRB"],
            "register_number": ["18423"],
            # Former court H1101 then current court R1101 (Amtsgericht merger
            # history); the most recent (last) court segment should win.
            "company_number": ["H1101_R1101_HRB18423"],
            "system_uri": ["offeneregister://H1101_R1101_HRB18423"],
        }
    )
    source.write_parquet(data_dir / "offeneregister-001.parquet")

    written = canonicalize_system_shards(
        "offeneregister", run_date="2026-06-01", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] == "RA000234|HRB18423"


def test_canonicalize_offeneregister_preserves_court_merger_disambiguation_suffix(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("offeneregister", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "name": ["Example GmbH"],
            "register_art": ["HRB"],
            # The separately shard-derived register_number drops the trailing
            # court-merger disambiguation suffix ("RZ") that only the raw
            # company_number retains -- without it, distinct companies sharing
            # the bare number 1162 at this court would collide.
            "register_number": ["1162"],
            "company_number": ["X1721R_HRB1162RZ"],
            "system_uri": ["offeneregister://X1721R_HRB1162RZ"],
        }
    )
    source.write_parquet(data_dir / "offeneregister-001.parquet")

    written = canonicalize_system_shards(
        "offeneregister", run_date="2026-06-01", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] == "RA000290|HRB1162RZ"


def test_canonicalize_offeneregister_nulls_company_number_for_placeholder_register_id(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("offeneregister", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "name": ["Example GmbH"],
            "register_art": ["HRB"],
            "register_number": ["200223"],
            # R1101 resolves cleanly, but 200223 falls in the confirmed
            # offeneregister.de placeholder-number band (2026-08-25 finding).
            "company_number": ["R1101_HRB200223"],
            "system_uri": ["offeneregister://R1101_HRB200223"],
        }
    )
    source.write_parquet(data_dir / "offeneregister-001.parquet")

    written = canonicalize_system_shards(
        "offeneregister", run_date="2026-06-01", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] is None


def test_canonicalize_offeneregister_nulls_company_number_for_unmapped_court(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("offeneregister", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "name": ["Example GmbH"],
            "register_art": ["HRB"],
            "register_number": ["999"],
            # ZZ9999 has a court-code shape but is not in the xjustiz crosswalk.
            "company_number": ["ZZ9999_HRB999"],
            "system_uri": ["offeneregister://ZZ9999_HRB999"],
        }
    )
    source.write_parquet(data_dir / "offeneregister-001.parquet")

    written = canonicalize_system_shards(
        "offeneregister", run_date="2026-06-01", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] is None


def test_canonicalize_frame_wikidata_prefers_english_legal_name_and_merges_alternative_names():
    frame = pl.DataFrame(
        {
            "id": ["Q102"],
            "label_en": ["Acme Label"],
            "official_name": [["Acme Corporation", "Societe Acme"]],
            "official_name_en": [["Acme Corporation"]],
            "short_name": [["ACME"]],
            "short_name_en": [["ACME"]],
            "aliases_en": [["Acme", "Acme Corporation"]],
            "entity_type": ["item"],
            "inception": ["+2000-01-01T00:00:00Z"],
            "dissolved": [None],
            "country": [["Q145"]],
            "company_number": ["123456789"],
        }
    )

    out = canonical._canonicalize_frame(
        frame,
        country="wikidata",
        source_company_type_col=None,
        exclude_company_type_values=set(),
        canonical_source_column_aliases=None,
        system_field_candidates={
            "name": ["label_en"],
            "company_number": ["company_number"],
            "company_type": ["entity_type"],
            "current_status": ["entity_type"],
            "incorporation_date": ["inception"],
            "dissolution_date": ["dissolved"],
            "registry_url": ["registry_url", "id"],
            "registered_address_in_full": ["label_en"],
            "inactive": ["inactive"],
            "branch": ["branch"],
            "branch_status": ["branch_status"],
        },
    )

    row = out.row(0, named=True)
    assert row["name"] == "Acme Corporation"
    assert row["alternative_names"] == ["Societe Acme", "ACME"]


def test_canonicalize_frame_maps_wikidata_jurisdiction_qid_when_country_missing():
    frame = pl.DataFrame(
        {
            "id": ["Q101"],
            "label_en": ["Beta Corp"],
            "entity_type": ["item"],
            "inception": ["+2010-05-01T00:00:00Z"],
            "dissolved": [None],
            "jurisdiction": [["Q183"]],
            "company_number": ["987654321"],
        }
    )

    out = canonical._canonicalize_frame(
        frame,
        country="wikidata",
        source_company_type_col=None,
        exclude_company_type_values=set(),
        canonical_source_column_aliases=None,
        system_field_candidates={
            "name": ["label_en"],
            "company_number": ["company_number"],
            "company_type": ["entity_type"],
            "current_status": ["entity_type"],
            "incorporation_date": ["inception"],
            "dissolution_date": ["dissolved"],
            "registry_url": ["registry_url", "id"],
            "registered_address_in_full": ["label_en"],
            "inactive": ["inactive"],
            "branch": ["branch"],
            "branch_status": ["branch_status"],
        },
    )

    row = out.row(0, named=True)
    assert row["jurisdiction_code"] == "DE"


def test_canonicalize_frame_sets_wikidata_inactive_and_status_from_dissolution_date():
    frame = pl.DataFrame(
        {
            "id": ["Q999"],
            "label_en": ["Gamma Corp"],
            "entity_type": ["item"],
            "inception": ["+2001-01-01T00:00:00Z"],
            "dissolved": ["+2020-01-01T00:00:00Z"],
            "country": [["Q145"]],
            "company_number": ["123"],
        }
    )

    out = canonical._canonicalize_frame(
        frame,
        country="wikidata",
        source_company_type_col=None,
        exclude_company_type_values=set(),
        canonical_source_column_aliases=None,
        system_field_candidates={
            "name": ["label_en"],
            "company_number": ["company_number"],
            "company_type": ["entity_type"],
            "current_status": ["entity_type"],
            "incorporation_date": ["inception"],
            "dissolution_date": ["dissolved"],
            "registry_url": ["registry_url", "id"],
            "registered_address_in_full": ["label_en"],
            "inactive": ["inactive"],
            "branch": ["branch"],
            "branch_status": ["branch_status"],
        },
    )

    row = out.row(0, named=True)
    assert row["dissolution_date"] == "2020-01-01"
    assert row["inactive"] is True
    assert row["current_status"] == "Dissolved"
    assert row["company_type"] is None


def test_canonicalize_frame_maps_wikidata_lei_from_p1278_claim():
    """Wikidata's P1278 (LEI) claim reaches canonical `lei`.

    Previously the canonical `lei` column was hardcoded to only ever
    populate for `country == "gleif"`, so Wikidata's genuine (if sparse --
    ~5.7% of company rows) `lei` shard column was silently dropped even
    though the field existed end to end otherwise.

    Also covers the companion `company_number` fix: it no longer leaks the
    `lei` value it used to be a bare copy of -- with no jurisdiction-scoped
    company-number claim in this fixture, `company_number` resolves to null
    for both rows even though `lei` is genuinely populated for the first.
    """
    frame = pl.DataFrame(
        {
            "id": ["Q100", "Q200"],
            "label_en": ["Acme Corp", "Beta Corp"],
            "entity_type": ["item", "item"],
            "inception": [None, None],
            "dissolved": [None, None],
            "country": [["Q145"], ["Q145"]],
            "lei": [["529900T8BM49AURSDO55"], []],
        }
    )

    out = canonical._canonicalize_frame(
        frame,
        country="wikidata",
        source_company_type_col=None,
        exclude_company_type_values=set(),
        canonical_source_column_aliases=None,
        system_field_candidates={
            "name": ["label_en"],
            "company_type": ["entity_type"],
            "current_status": ["entity_type"],
            "incorporation_date": ["inception"],
            "dissolution_date": ["dissolved"],
            "registry_url": ["registry_url", "id"],
            "registered_address_in_full": ["label_en"],
            "inactive": ["inactive"],
            "branch": ["branch"],
            "branch_status": ["branch_status"],
        },
    )

    assert out["lei"].to_list() == ["529900T8BM49AURSDO55", None]
    assert out["company_number"].to_list() == [None, None]


def test_canonicalize_frame_selects_wikidata_company_number_per_jurisdiction():
    """Wikidata's `company_number` is picked per resolved jurisdiction.

    GB/FR/DE each read their own claim-derived shard column
    (`company_number_gb`/`_fr`/`_de`); a jurisdiction with no confirmed
    source property (here a fourth, unmapped jurisdiction) resolves to null
    rather than guessing from whichever column happens to be populated.
    """
    frame = pl.DataFrame(
        {
            "id": ["Q1", "Q2", "Q3", "Q4"],
            "label_en": ["GB Co", "FR Co", "DE Co", "US Co"],
            "entity_type": ["item", "item", "item", "item"],
            "inception": [None, None, None, None],
            "dissolved": [None, None, None, None],
            "country": [["Q145"], ["Q142"], ["Q183"], ["Q30"]],
            "company_number_gb": ["01234567", None, None, None],
            "company_number_fr": [None, "552100554", None, None],
            "company_number_de": [None, None, "HRB123456", None],
        }
    )

    out = canonical._canonicalize_frame(
        frame,
        country="wikidata",
        source_company_type_col=None,
        exclude_company_type_values=set(),
        canonical_source_column_aliases=None,
        system_field_candidates={
            "name": ["label_en"],
            "company_type": ["entity_type"],
            "current_status": ["entity_type"],
            "incorporation_date": ["inception"],
            "dissolution_date": ["dissolved"],
            "registry_url": ["registry_url", "id"],
            "registered_address_in_full": ["label_en"],
            "inactive": ["inactive"],
            "branch": ["branch"],
            "branch_status": ["branch_status"],
        },
    )

    assert out["jurisdiction_code"].to_list() == ["GB", "FR", "DE", "US"]
    assert out["company_number"].to_list() == [
        "01234567",
        "552100554",
        "HRB123456",
        None,
    ]


def test_canonicalize_offeneregister_shards_in_allow_research_mode(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("offeneregister", layer="source") / "2026-06-26"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["offeneregister://de/HRB12345"],
            "register_art": ["HRB"],
            "register_number": ["12345"],
            "company_type": ["HRB"],
            "current_status": ["active"],
            "jurisdiction_code": ["de"],
            "name": ["Beispiel GmbH"],
            "registered_address": ["Musterstrasse 1, 50667 Koeln"],
            "retrieved_at": ["2026-06-26"],
        },
    ).write_parquet(data_dir / "offeneregister-001.parquet")

    written = canonicalize_system_shards(
        "offeneregister",
        run_date="2026-06-26",
        roots=workspace_roots,
        allow_research=True,
    )

    assert len(written) == 1
    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)
    _assert_canonical_prefix_columns(out)
    assert row["jurisdiction_code"] == "de"
    assert row["company_number"] == "HRB 12345"
    assert row["name"] == "Beispiel GmbH"
    assert row["company_type"] == "HRB"
    assert row["current_status"] == "active"
    assert row["registered_address_in_full"] == "Musterstrasse 1, 50667 Koeln"
    assert row["registry_url"] is None
    assert row["metadata_generated_utc"] is not None


def test_canonicalize_system_shards_maps_gleif_fields(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gleif://529900T8BM49AURSDO55"],
            "LEI": ["529900T8BM49AURSDO55"],
            "Entity_LegalName": ["EXAMPLE HOLDINGS LIMITED"],
            "Entity_LegalJurisdiction": ["GB"],
            "Entity_EntityStatus": ["ACTIVE"],
            "Entity_EntityCreationDate": ["2015-03-10"],
            "Entity_LegalAddress_FirstAddressLine": ["1 HIGH STREET"],
            "Entity_LegalAddress_City": ["LONDON"],
            "Entity_LegalAddress_PostalCode": ["SW1A 1AA"],
            "Entity_LegalAddress_Country": ["GB"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": ["03856000"],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    assert len(written) == 1

    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)

    assert row["jurisdiction_code"] == "GB"
    assert row["company_number"] == "03856000"
    assert row["name"] == "EXAMPLE HOLDINGS LIMITED"
    assert "CompanyName" not in out.columns
    assert row["current_status"] == "ACTIVE"
    assert row["incorporation_date"] == "2015-03-10"
    assert row["registered_address_in_full"] == "1 HIGH STREET, LONDON, SW1A 1AA, GB"
    assert row["registry_url"] == "https://search.gleif.org/#/record/03856000"
    assert row["metadata_generated_utc"] is not None


def test_canonicalize_system_shards_nulls_gleif_company_number_for_wrong_authority(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gleif://529900T8BM49AURSDO55"],
            "LEI": ["529900T8BM49AURSDO55"],
            "Entity_LegalName": ["EXAMPLE FUND SERVICES LIMITED"],
            "Entity_LegalJurisdiction": ["GB"],
            "Entity_RegistrationAuthority_RegistrationAuthorityID": ["RA000589"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": ["1141019"],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] is None


def test_canonicalize_system_shards_keeps_gleif_company_number_for_valid_authority(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": [
                "gleif://529900T8BM49AURSDO55",
                "gleif://213800Q7JGGZ17BFJK36",
                "gleif://097900BFF10000026455",
            ],
            "LEI": [
                "529900T8BM49AURSDO55",
                "213800Q7JGGZ17BFJK36",
                "097900BFF10000026455",
            ],
            "Entity_LegalName": [
                "EXAMPLE SCOTLAND LTD",
                "EXAMPLE FRANCE SAS",
                "EXAMPLE SHORT LTD",
            ],
            "Entity_LegalJurisdiction": ["GB", "FR", "GB"],
            "Entity_RegistrationAuthority_RegistrationAuthorityID": [
                "RA000587",
                "RA000189",
                "RA000585",
            ],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": [
                "SC386824",
                "418841748",
                "8230688",
            ],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    # Every written file, not just the first: these three rows span the GB
    # and FR jurisdiction partitions.
    out = pl.read_parquet(written).sort("lei")
    by_lei = {row["lei"]: row["company_number"] for row in out.iter_rows(named=True)}

    assert by_lei["213800Q7JGGZ17BFJK36"] == "418841748"
    assert by_lei["097900BFF10000026455"] == "08230688"
    assert by_lei["529900T8BM49AURSDO55"] == "SC386824"


def test_canonicalize_system_shards_gleif_company_number_unrestricted_for_unmapped_jurisdiction(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gleif://391200Y51UPND1Q04O86"],
            "LEI": ["391200Y51UPND1Q04O86"],
            "Entity_LegalName": ["EXAMPLE BV"],
            "Entity_LegalJurisdiction": ["NL"],
            "Entity_RegistrationAuthority_RegistrationAuthorityID": ["RA000463"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": ["12345678"],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] == "12345678"


def test_canonicalize_system_shards_keeps_gleif_company_number_for_valid_de_handelsregister_authority(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gleif://391200Y51UPND1Q04O86"],
            "LEI": ["391200Y51UPND1Q04O86"],
            "Entity_LegalName": ["EXAMPLE GMBH"],
            "Entity_LegalJurisdiction": ["DE"],
            "Entity_RegistrationAuthority_RegistrationAuthorityID": ["RA000304"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": [
                "HRB 125713"
            ],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    # DE company numbers are RA-code-scoped: bare Handelsregister numbers restart
    # per Amtsgericht and are not unique on their own (see
    # GLEIF_COMPANY_NUMBER_AUTHORITY_SCOPED_JURISDICTIONS).
    assert row["company_number"] == "RA000304|HRB125713"


def test_canonicalize_system_shards_nulls_gleif_company_number_for_de_non_commercial_authority(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": [
                "gleif://529900EXAMPLE0000001",
                "gleif://529900EXAMPLE0000002",
            ],
            "LEI": ["529900EXAMPLE0000001", "529900EXAMPLE0000002"],
            "Entity_LegalName": ["EXAMPLE FOUNDATION", "EXAMPLE BANK AG"],
            "Entity_LegalJurisdiction": ["DE", "DE"],
            # RA000704: Bavaria Foundations Directory (not a company register).
            # RA000373: BaFin (financial supervisory authority, not a company register).
            "Entity_RegistrationAuthority_RegistrationAuthorityID": [
                "RA000704",
                "RA000373",
            ],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": [
                "12345",
                "67890",
            ],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])

    assert out["company_number"].to_list() == [None, None]


def test_canonicalize_system_shards_nulls_gleif_company_number_for_de_unternehmensregister(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": ["gleif://529900EXAMPLE0000003"],
            "LEI": ["529900EXAMPLE0000003"],
            "Entity_LegalName": ["EXAMPLE AG"],
            "Entity_LegalJurisdiction": ["DE"],
            # RA000372: Bundesanzeiger's Unternehmensregister. GLEIF's own comment
            # on this code says the local court name must be appended to the ID,
            # so its format isn't directly comparable to offeneregister's.
            "Entity_RegistrationAuthority_RegistrationAuthorityID": ["RA000372"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": [
                "Munich HRB 125713"
            ],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    row = pl.read_parquet(written[0]).row(0, named=True)

    assert row["company_number"] is None


def test_canonicalize_system_shards_drops_gleif_rows_with_successor_lei(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    data_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    data_dir.mkdir(parents=True, exist_ok=True)

    source = pl.DataFrame(
        {
            "system_uri": [
                "gleif://OLDLEI0000000000001",
                "gleif://NEWLEI0000000000002",
            ],
            "LEI": ["OLDLEI0000000000001", "NEWLEI0000000000002"],
            "Entity_LegalName": ["OLD NAME LTD", "NEW NAME LTD"],
            "Entity_LegalJurisdiction": ["GB", "GB"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": [
                "01234567",
                "07654321",
            ],
            "Entity_SuccessorEntity_SuccessorLEI": ["NEWLEI0000000000002", None],
        }
    )
    source.write_parquet(data_dir / "gleif-001.parquet")

    written = canonicalize_system_shards(
        "gleif", run_date="2026-06-17", roots=workspace_roots
    )
    out = pl.read_parquet(written[0])

    assert out["lei"].to_list() == ["NEWLEI0000000000002"]


def test_canonicalize_system_shards_uses_latest_available_snapshot_when_run_date_omitted(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    old_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-16"
    new_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    old_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": ["gleif://OLD000000000000000001"],
            "LEI": ["OLD000000000000000001"],
            "Entity_LegalName": ["OLD LTD"],
            "Entity_LegalJurisdiction": ["GB"],
            "Entity_EntityStatus": ["ACTIVE"],
            "Entity_EntityCreationDate": ["2015-03-10"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": ["01234567"],
        }
    ).write_parquet(old_dir / "gleif-001.parquet")

    pl.DataFrame(
        {
            "system_uri": ["gleif://NEW000000000000000001"],
            "LEI": ["NEW000000000000000001"],
            "Entity_LegalName": ["NEW LTD"],
            "Entity_LegalJurisdiction": ["IE"],
            "Entity_EntityStatus": ["ACTIVE"],
            "Entity_EntityCreationDate": ["2016-04-11"],
            "Entity_RegistrationAuthority_RegistrationAuthorityEntityID": ["09876543"],
        }
    ).write_parquet(new_dir / "gleif-001.parquet")

    written = canonicalize_system_shards("gleif", run_date=None, roots=workspace_roots)

    assert len(written) == 1
    assert "data/gleif/canonical/2026-06-17" in written[0].as_posix()

    out = pl.read_parquet(written[0])
    row = out.row(0, named=True)
    assert row["company_number"] == "09876543"
    assert row["name"] == "NEW LTD"
    assert row["metadata_generated_utc"] is not None


def test_canonicalize_system_shards_writes_sample_parquet(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "system_uri": [f"gb://{index:08d}" for index in range(1, 21)],
            "CompanyNumber": [f"{index:08d}" for index in range(1, 21)],
            "CompanyName": [f"EXAMPLE {index} LIMITED" for index in range(1, 21)],
            "CompanyCategory": ["Private Limited Company"] * 20,
            "CompanyStatus": ["Active"] * 20,
        }
    ).write_parquet(data_dir / "gb-001.parquet")

    written = canonicalize_system_shards(
        "gb", run_date="2026-06-15", roots=workspace_roots, partition_rows_per_file=5
    )

    assert len(written) == 4
    sample_path = (
        layer_fixture_dir("gb", layer="canonical") / "2026-06-01" / "sample.parquet"
    )
    assert sample_path.exists()

    sample = pl.read_parquet(sample_path)
    total_rows = sum(pl.read_parquet(path).height for path in written)
    assert sample.height > 0
    assert sample.height <= total_rows


def test_canonicalize_system_shards_rejects_non_positive_chunk_size(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    with pytest.raises(ValueError, match="chunk_size must be greater than zero"):
        canonicalize_system_shards(
            "gb", run_date="2026-06-15", roots=workspace_roots, chunk_size=0
        )


def test_canonicalize_system_shards_rejects_system_without_field_mapping(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    unsupported_plan = SimpleNamespace(
        code="zz",
        status="live",
        notes="",
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=unsupported_plan
    )

    with pytest.raises(RuntimeError, match="No system_field_candidates configured"):
        canonicalize_system_shards("zz", run_date="2026-06-15", roots=workspace_roots)
    get_system_plan.assert_called_once_with("zz")


def test_canonicalize_system_shards_research_policy_denies_stage(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    denied_plan = SimpleNamespace(
        code="dbpedia",
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
        canonicalize_system_shards(
            "dbpedia", run_date="2026-06-26", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("dbpedia")


def test_canonicalize_system_shards_supported_status_passes_gate(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    """A `supported`-status plan must clear the runnable-status gate exactly
    like `live`, without needing `allow_research`. It fails immediately after
    for an unrelated, distinguishable reason (no `system_field_candidates`
    configured) -- proving the gate itself let it through rather than the
    call happening to succeed for some other reason.
    """
    supported_plan = SimpleNamespace(
        code="zz",
        status="supported",
        notes="",
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=supported_plan
    )

    with pytest.raises(RuntimeError, match="No system_field_candidates configured"):
        canonicalize_system_shards("zz", run_date="2026-06-15", roots=workspace_roots)
    get_system_plan.assert_called_once_with("zz")


def test_canonicalize_system_shards_blocked_status_rejected_even_with_allow_research(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    """`blocked` is never added to the runnable-status set, unlike
    `research_required` -- so `allow_research=True` (even paired with a
    research policy that would otherwise permit the `canonical` stage) must
    not let a blocked plan through.
    """
    blocked_plan = SimpleNamespace(
        code="gb",
        status="blocked",
        notes="Not yet available",
        research=SimpleNamespace(
            allow_research_runtime=True, allowed_stages=("canonical",)
        ),
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=blocked_plan
    )

    with pytest.raises(RuntimeError, match="gb: blocked - Not yet available"):
        canonicalize_system_shards(
            "gb", run_date="2026-06-15", roots=workspace_roots, allow_research=True
        )
    get_system_plan.assert_called_once_with("gb")


def test_compute_z_sample_size_returns_zero_for_non_positive_population() -> None:
    assert canonical._compute_z_sample_size(population_size=0) == 0


def test_write_canonical_sample_file_noop_for_empty_or_zero_rows(tmp_path: Path):
    output_dir = tmp_path / "canonical"
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical._write_canonical_sample_file(output_dir=output_dir, written_paths=[])
    assert not (output_dir / canonical.SAMPLE_FILE_NAME).exists()

    zero_path = output_dir / "gb-001.parquet"
    pl.DataFrame({"name": []}, schema={"name": pl.Utf8}).write_parquet(zero_path)
    canonical._write_canonical_sample_file(
        output_dir=output_dir, written_paths=[zero_path]
    )
    assert not (output_dir / canonical.SAMPLE_FILE_NAME).exists()


# The research-stage policy gate every canonicalize_* entry point runs is now
# one shared helper (acquisition.constants_status.require_runnable_plan); its
# own behaviour is covered in test_constants_status.py rather than re-tested
# per calling module.


def test_canonicalize_frame_falls_back_when_company_type_mapping_missing(
    mocker,
) -> None:
    frame = pl.DataFrame(
        {
            "CompanyNumber": ["001"],
            "CompanyName": ["ACME LTD"],
            "CompanyCategory": ["Private Limited Company"],
            "CompanyStatus": ["Active"],
        }
    )

    get_system_company_type_mapping = mocker.patch.object(
        canonical,
        "get_system_company_type_mapping",
        side_effect=KeyError("missing"),
    )

    out = canonical._canonicalize_frame(
        frame,
        country="gb",
        source_company_type_col=None,
        exclude_company_type_values=set(),
        system_field_candidates={
            "name": ["CompanyName"],
            "company_number": ["CompanyNumber"],
            "company_type": ["CompanyCategory"],
            "current_status": ["CompanyStatus"],
            "incorporation_date": ["IncorporationDate"],
            "dissolution_date": ["DissolutionDate"],
            "registry_url": ["URI"],
            "registered_address_in_full": ["registered_address_in_full"],
            "inactive": ["inactive"],
            "branch": ["branch"],
            "branch_status": ["branch_status"],
        },
    )

    row = out.row(0, named=True)
    assert row["company_type"] == "Private Limited Company"
    get_system_company_type_mapping.assert_called_once_with("gb")


def test_canonicalize_system_shards_raises_for_non_runnable_plan_status(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    blocked_plan = SimpleNamespace(
        code="gb", status="blocked", notes="Not yet available"
    )
    get_system_plan = mocker.patch.object(
        canonical, "get_system_plan", return_value=blocked_plan
    )

    with pytest.raises(RuntimeError, match="gb: blocked - Not yet available"):
        canonicalize_system_shards("gb", run_date="2026-06-15", roots=workspace_roots)
    get_system_plan.assert_called_once_with("gb")


def test_canonicalize_system_shards_removes_stale_and_sample_outputs(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb://00000001"],
            "CompanyNumber": ["00000001"],
            "CompanyName": ["EXAMPLE LTD"],
            "CompanyCategory": ["Private Limited Company"],
            "CompanyStatus": ["Active"],
        }
    ).write_parquet(data_dir / "gb-001.parquet")

    output_dir = layer_fixture_dir("gb", layer="canonical") / "2026-06-01"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_file = output_dir / "gb-999.parquet"
    stale_file.write_text("stale", encoding="utf-8")
    sample_file = output_dir / canonical.SAMPLE_FILE_NAME
    sample_file.write_text("stale-sample", encoding="utf-8")

    written = canonicalize_system_shards(
        "gb", run_date="2026-06-15", roots=workspace_roots
    )

    assert len(written) == 1
    assert not stale_file.exists()
    assert sample_file.exists()


def test_canonicalize_system_shards_stale_cleanup_spares_companion_name_files(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """Regression test: the entity pass's stale-output cleanup must only
    ever remove its own per-entity gb-NNN.parquet files, never a
    gb-names-*.parquet (or other companion-suffixed) file sitting in the
    same canonical output directory. Before this fix the cleanup glob was
    unfiltered and matched both, which was harmless when this pass always
    ran strictly before canonicalize_system_name_rows wrote anything, but
    became a genuine cross-thread race once the two started running
    concurrently."""
    data_dir = layer_fixture_dir("gb", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gb://00000001"],
            "CompanyNumber": ["00000001"],
            "CompanyName": ["EXAMPLE LTD"],
            "CompanyCategory": ["Private Limited Company"],
            "CompanyStatus": ["Active"],
        }
    ).write_parquet(data_dir / "gb-001.parquet")

    output_dir = layer_fixture_dir("gb", layer="canonical") / "2026-06-01"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_entity_file = output_dir / "gb-999.parquet"
    stale_entity_file.write_text("stale", encoding="utf-8")
    companion_names_file = output_dir / "gb-names-001.parquet"
    companion_names_file.write_text("real names output", encoding="utf-8")

    canonicalize_system_shards("gb", run_date="2026-06-15", roots=workspace_roots)

    assert not stale_entity_file.exists()
    assert companion_names_file.exists()
    assert companion_names_file.read_text(encoding="utf-8") == "real names output"


def test_canonicalize_system_shards_skips_empty_canonical_chunks(
    tmp_path: Path, layer_fixture_dir, mocker, workspace_roots: WorkspaceRoots
):
    data_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"
    data_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "siren": ["1"],
            "denominationUniteLegale": ["ALPHA"],
            "categorieJuridiqueUniteLegale": ["DROP"],
            "etatAdministratifUniteLegale": ["A"],
            "dateCreationUniteLegale": ["20200101"],
        }
    ).write_parquet(data_dir / "fr-001.parquet")

    get_company_type_exclusions = mocker.patch.object(
        canonical,
        "get_company_type_exclusions",
        return_value={"DROP"},
    )
    written = canonicalize_system_shards(
        "fr", run_date="2026-06-15", roots=workspace_roots, chunk_size=1
    )

    assert written == []
    get_company_type_exclusions.assert_called_once_with("fr")
