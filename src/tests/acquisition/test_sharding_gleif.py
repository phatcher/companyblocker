from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

import acquisition.sharding_gleif as gleif_sharding
from acquisition import sharding
from acquisition.registry import get_system_plan
from acquisition.sharding import shard_system_source
from workspace.roots import WorkspaceRoots


@pytest.fixture(autouse=True)
def clear_gleif_schema_cache():
    # Hold the cached functions themselves: a test that monkeypatches the
    # module attribute is unpatched only after this fixture's teardown, now
    # that the suite-wide autouse fixture in conftest.py depends on
    # monkeypatch and so is set up before, and torn down after, this one.
    schema_paths = gleif_sharding._gleif_company_schema_paths
    schema_indexes = gleif_sharding._gleif_company_schema_indexes
    schema_paths.cache_clear()
    schema_indexes.cache_clear()
    yield
    schema_paths.cache_clear()
    schema_indexes.cache_clear()


def test_gleif_text_helpers_cover_none_and_passthrough_cases():
    assert gleif_sharding._clean_xml_text(None) is None
    assert gleif_sharding._clean_xml_text("  value  ") == "value"
    assert gleif_sharding._clean_xml_text("   ") is None
    assert gleif_sharding._join_text(None) is None
    assert gleif_sharding._local_name("tag") == "tag"
    assert gleif_sharding._qname_local(None) is None
    assert gleif_sharding._qname_local("lei:Tag") == "Tag"


def test_gleif_append_value_deduplicates_existing_values():
    values: dict[str, str | None] = {"name": "Alpha"}

    gleif_sharding._append_value(values, "name", "Alpha")

    assert values == {"name": "Alpha"}


def test_gleif_company_schema_paths_raises_when_schema_file_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gleif_sharding, "_GLEIF_LEI_CDF_XSD", Path("missing-schema.xsd")
    )

    with pytest.raises(FileNotFoundError, match="Missing LEI-CDF schema file"):
        gleif_sharding._gleif_company_schema_paths()


def test_gleif_extract_record_preserves_lei_when_schema_paths_omit_it(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        gleif_sharding, "_gleif_company_schema_paths", lambda: frozenset()
    )
    lei_record = ET.fromstring(
        """
        <lei:LEIRecord xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
            <lei:LEI>ABC123</lei:LEI>
        </lei:LEIRecord>
        """
    )

    assert gleif_sharding._extract_gleif_record(lei_record) == {"LEI": "ABC123"}


def test_gleif_extract_name_rows_keeps_same_type_names_distinctly_paired():
    """Two OtherEntityName entries sharing a type must not desync like the
    generic pipe-joined flattening does: `_append_value` dedups each of the
    name/type columns independently, so two same-typed-but-different-text
    names would produce a 2-segment name column against a 1-segment (deduped)
    type column if read via that path. Reading elements directly avoids the
    issue entirely -- there is no pairing step to get wrong.
    """
    lei_record = ET.fromstring(
        """
        <lei:LEIRecord xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
            <lei:LEI>ABC123</lei:LEI>
            <lei:Entity>
                <lei:LegalName xml:lang="en">Alpha Ltd</lei:LegalName>
                <lei:OtherEntityNames>
                    <lei:OtherEntityName type="PREVIOUS_LEGAL_NAME">Alpha Old One</lei:OtherEntityName>
                    <lei:OtherEntityName type="PREVIOUS_LEGAL_NAME">Alpha Old Two</lei:OtherEntityName>
                </lei:OtherEntityNames>
                <lei:TransliteratedOtherEntityNames>
                    <lei:TransliteratedOtherEntityName type="AUTO_ASCII_TRANSLITERATED_LEGAL_NAME">ALPHA LTD</lei:TransliteratedOtherEntityName>
                </lei:TransliteratedOtherEntityNames>
            </lei:Entity>
        </lei:LEIRecord>
        """
    )

    rows = gleif_sharding._extract_gleif_name_rows(lei_record, lei="ABC123")

    assert rows == [
        {
            "LEI": "ABC123",
            "name": "Alpha Ltd",
            "source_type": "LEGAL_NAME",
            "language_code": "en",
            "derivation_note": "native LegalName",
        },
        {
            "LEI": "ABC123",
            "name": "Alpha Old One",
            "source_type": "PREVIOUS_LEGAL_NAME",
            "language_code": None,
            "derivation_note": "native OtherEntityName",
        },
        {
            "LEI": "ABC123",
            "name": "Alpha Old Two",
            "source_type": "PREVIOUS_LEGAL_NAME",
            "language_code": None,
            "derivation_note": "native OtherEntityName",
        },
        {
            "LEI": "ABC123",
            "name": "ALPHA LTD",
            "source_type": "AUTO_ASCII_TRANSLITERATED_LEGAL_NAME",
            "language_code": None,
            "derivation_note": "native TransliteratedOtherEntityName",
        },
    ]


def test_gleif_extract_name_rows_returns_empty_list_when_entity_missing():
    lei_record = ET.fromstring(
        """
        <lei:LEIRecord xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
            <lei:LEI>ABC123</lei:LEI>
        </lei:LEIRecord>
        """
    )

    assert gleif_sharding._extract_gleif_name_rows(lei_record, lei="ABC123") == []


def test_gleif_extract_successor_rows_pairs_multiple_successors_distinctly():
    """SuccessorEntity is maxOccurs unbounded (an entity can split into
    several successors), so this must not desync the same way multiple
    same-typed OtherEntityName entries would if read via the generic
    flattened columns instead of directly from the XML tree.
    """
    lei_record = ET.fromstring(
        """
        <lei:LEIRecord xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
            <lei:LEI>ABC123</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Alpha Ltd</lei:LegalName>
                <lei:SuccessorEntity>
                    <lei:SuccessorLEI>SUCC001</lei:SuccessorLEI>
                </lei:SuccessorEntity>
                <lei:SuccessorEntity>
                    <lei:SuccessorLEI>SUCC002</lei:SuccessorLEI>
                </lei:SuccessorEntity>
            </lei:Entity>
        </lei:LEIRecord>
        """
    )

    rows = gleif_sharding._extract_gleif_successor_rows(lei_record, lei="ABC123")

    assert rows == [
        {
            "LEI": "ABC123",
            "predecessor_name": "Alpha Ltd",
            "successor_lei": "SUCC001",
            "successor_name": None,
        },
        {
            "LEI": "ABC123",
            "predecessor_name": "Alpha Ltd",
            "successor_lei": "SUCC002",
            "successor_name": None,
        },
    ]


def test_gleif_extract_successor_rows_captures_name_only_choice_branch():
    """SuccessorEntity is a choice between SuccessorLEI and
    SuccessorEntityName -- a successor can be named without an LEI. Both
    are captured at this layer so nothing from the XML is silently dropped,
    even though downstream chain-walking only acts on LEI-identified rows.
    """
    lei_record = ET.fromstring(
        """
        <lei:LEIRecord xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
            <lei:LEI>ABC123</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Alpha Ltd</lei:LegalName>
                <lei:SuccessorEntity>
                    <lei:SuccessorEntityName>Unregistered Successor Co</lei:SuccessorEntityName>
                </lei:SuccessorEntity>
            </lei:Entity>
        </lei:LEIRecord>
        """
    )

    rows = gleif_sharding._extract_gleif_successor_rows(lei_record, lei="ABC123")

    assert rows == [
        {
            "LEI": "ABC123",
            "predecessor_name": "Alpha Ltd",
            "successor_lei": None,
            "successor_name": "Unregistered Successor Co",
        }
    ]


def test_gleif_extract_successor_rows_returns_empty_list_when_entity_missing():
    lei_record = ET.fromstring(
        """
        <lei:LEIRecord xmlns:lei="http://www.gleif.org/data/schema/leidata/2016">
            <lei:LEI>ABC123</lei:LEI>
        </lei:LEIRecord>
        """
    )

    assert gleif_sharding._extract_gleif_successor_rows(lei_record, lei="ABC123") == []


def test_gleif_write_xml_zip_chunked_returns_none_when_archive_contains_only_csv(
    tmp_path: Path,
):
    source_path = tmp_path / "gleif.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("gleif.csv", "LEI,EntityName\nABC123,Alpha Ltd\n")

    assert (
        gleif_sharding.write_gleif_xml_zip_chunked(
            source_path, tmp_path, "gleif", chunk_size=10
        )
        is None
    )


def test_gleif_write_xml_zip_chunked_rejects_unexpected_root_element(tmp_path: Path):
    source_path = tmp_path / "gleif.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("gleif.xml", "<root><item /></root>")

    with pytest.raises(ValueError, match="Unexpected XML root"):
        gleif_sharding.write_gleif_xml_zip_chunked(
            source_path, tmp_path, "gleif", chunk_size=10
        )


def test_gleif_shard_system_source_splits_zip_csv(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "gleif-lei-cdf-2026-06-17.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "gleif.csv",
            "lei,entityName\nABC123,Alpha Ltd\nDEF456,Beta Ltd\n",
        )

    written_paths = shard_system_source(
        "gleif", roots=workspace_roots, run_date="2026-06-17", chunk_size=1
    )

    assert all(path.parent == source_dir for path in written_paths)
    assert [path.name for path in written_paths] == [
        "gleif-001.parquet",
        "gleif-002.parquet",
    ]
    assert [pl.read_parquet(path).height for path in written_paths] == [1, 1]


def test_gleif_shard_system_source_splits_zip_xml(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "gleif-lei-cdf-2026-06-17.zip"
    xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<lei:LEIData xmlns:lei=\"http://www.gleif.org/data/schema/leidata/2016\">
    <lei:LEIRecords>
        <lei:LEIRecord>
            <lei:LEI>ABC123</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Alpha Ltd</lei:LegalName>
                <lei:LegalJurisdiction>GB</lei:LegalJurisdiction>
                <lei:EntityStatus>ACTIVE</lei:EntityStatus>
                <lei:LegalAddress>
                    <lei:FirstAddressLine>1 Main Street</lei:FirstAddressLine>
                    <lei:City>London</lei:City>
                    <lei:PostalCode>SW1A 1AA</lei:PostalCode>
                    <lei:Country>GB</lei:Country>
                </lei:LegalAddress>
            </lei:Entity>
            <lei:Registration>
                <lei:InitialRegistrationDate>2015-03-10T00:00:00Z</lei:InitialRegistrationDate>
            </lei:Registration>
        </lei:LEIRecord>
        <lei:LEIRecord>
            <lei:LEI>DEF456</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Beta Ltd</lei:LegalName>
                <lei:LegalJurisdiction>IE</lei:LegalJurisdiction>
                <lei:EntityStatus>ACTIVE</lei:EntityStatus>
                <lei:LegalAddress>
                    <lei:FirstAddressLine>2 River Road</lei:FirstAddressLine>
                    <lei:City>Dublin</lei:City>
                    <lei:PostalCode>D02</lei:PostalCode>
                    <lei:Country>IE</lei:Country>
                </lei:LegalAddress>
            </lei:Entity>
            <lei:Registration>
                <lei:InitialRegistrationDate>2018-11-01T00:00:00Z</lei:InitialRegistrationDate>
            </lei:Registration>
        </lei:LEIRecord>
    </lei:LEIRecords>
</lei:LEIData>
"""
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("gleif.xml", xml_payload)

    written_paths = shard_system_source(
        "gleif", roots=workspace_roots, run_date="2026-06-17", chunk_size=1
    )

    assert [path.name for path in written_paths] == [
        "gleif-001.parquet",
        "gleif-002.parquet",
    ]
    first = pl.read_parquet(written_paths[0])
    second = pl.read_parquet(written_paths[1])
    assert first.height == 1
    assert second.height == 1
    assert first["LEI"].to_list() == ["ABC123"]
    assert first["Entity_LegalName"].to_list() == ["Alpha Ltd"]
    assert second["LEI"].to_list() == ["DEF456"]
    assert second["Entity_LegalJurisdiction"].to_list() == ["IE"]


def test_gleif_shard_system_source_filters_by_supported_countries_list(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "gleif-lei-cdf-2026-06-17.zip"
    xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<lei:LEIData xmlns:lei=\"http://www.gleif.org/data/schema/leidata/2016\">
    <lei:LEIRecords>
        <lei:LEIRecord>
            <lei:LEI>GB111</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Alpha Ltd</lei:LegalName>
                <lei:LegalJurisdiction>GB</lei:LegalJurisdiction>
            </lei:Entity>
        </lei:LEIRecord>
        <lei:LEIRecord>
            <lei:LEI>IE222</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Beta Ltd</lei:LegalName>
                <lei:LegalJurisdiction>IE</lei:LegalJurisdiction>
            </lei:Entity>
        </lei:LEIRecord>
    </lei:LEIRecords>
</lei:LEIData>
"""
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("gleif.xml", xml_payload)

    base_plan = get_system_plan("gleif")
    filtered_plan = replace(base_plan, supported_countries=("gb",))
    monkeypatch.setattr(sharding, "get_system_plan", lambda _: filtered_plan)

    written_paths = shard_system_source(
        "gleif", roots=workspace_roots, run_date="2026-06-17", chunk_size=10
    )

    out = pl.read_parquet(written_paths[0])
    assert out.height == 1
    assert out["LEI"].to_list() == ["GB111"]


def test_gleif_resolve_supported_country_codes_mixes_selector_and_explicit_values() -> (
    None
):
    base_plan = get_system_plan("gleif")
    mixed_plan = replace(base_plan, supported_countries=("live", "ie", "us"))

    resolved = sharding._resolve_supported_country_codes(mixed_plan)

    assert resolved is not None
    assert "US" in resolved
    assert "IE" in resolved
    assert len(resolved) == len(set(resolved))


def test_gleif_shard_system_source_harmonizes_sparse_xml_chunk_schemas(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "gleif-lei-cdf-2026-06-17.zip"
    xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<lei:LEIData xmlns:lei=\"http://www.gleif.org/data/schema/leidata/2016\">
    <lei:LEIRecords>
        <lei:LEIRecord>
            <lei:LEI>AAA111</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Alpha Ltd</lei:LegalName>
                <lei:LegalAddress>
                    <lei:FirstAddressLine>1 Main Street</lei:FirstAddressLine>
                    <lei:City>London</lei:City>
                    <lei:Country>GB</lei:Country>
                </lei:LegalAddress>
            </lei:Entity>
            <lei:Registration>
                <lei:InitialRegistrationDate>2015-03-10T00:00:00Z</lei:InitialRegistrationDate>
            </lei:Registration>
        </lei:LEIRecord>
        <lei:LEIRecord>
            <lei:LEI>BBB222</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Beta Ltd</lei:LegalName>
                <lei:LegalAddress>
                    <lei:FirstAddressLine>2 River Road</lei:FirstAddressLine>
                    <lei:AdditionalAddressLine>Floor 3</lei:AdditionalAddressLine>
                    <lei:AdditionalAddressLine>West Wing</lei:AdditionalAddressLine>
                    <lei:City>Dublin</lei:City>
                    <lei:Country>IE</lei:Country>
                </lei:LegalAddress>
            </lei:Entity>
            <lei:Registration>
                <lei:InitialRegistrationDate>2018-11-01T00:00:00Z</lei:InitialRegistrationDate>
                <lei:ValidationSources>FULLY_CORROBORATED</lei:ValidationSources>
            </lei:Registration>
        </lei:LEIRecord>
    </lei:LEIRecords>
</lei:LEIData>
"""
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("gleif.xml", xml_payload)

    written_paths = shard_system_source(
        "gleif", roots=workspace_roots, run_date="2026-06-17", chunk_size=1
    )

    assert [path.name for path in written_paths] == [
        "gleif-001.parquet",
        "gleif-002.parquet",
    ]
    first_schema = tuple(pl.read_parquet_schema(written_paths[0]).keys())
    second_schema = tuple(pl.read_parquet_schema(written_paths[1]).keys())
    assert first_schema == second_schema


def test_gleif_shard_system_source_flattens_company_fields_and_excludes_officers(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "gleif-lei-cdf-2026-06-17.zip"
    xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<lei:LEIData xmlns:lei=\"http://www.gleif.org/data/schema/leidata/2016\">
    <lei:LEIRecords>
        <lei:LEIRecord>
            <lei:LEI>ABC123</lei:LEI>
            <lei:Entity>
                <lei:LegalName xml:lang=\"en\">Alpha Ltd</lei:LegalName>
                <lei:LegalJurisdiction>GB</lei:LegalJurisdiction>
                <lei:EntityStatus>ACTIVE</lei:EntityStatus>
                <lei:OtherEntityNames>
                    <lei:OtherEntityName>Alpha Trading</lei:OtherEntityName>
                    <lei:OtherEntityName>Alpha Holdings</lei:OtherEntityName>
                </lei:OtherEntityNames>
                <lei:LegalAddress>
                    <lei:FirstAddressLine>1 Main Street</lei:FirstAddressLine>
                    <lei:AdditionalAddressLine>Floor 2</lei:AdditionalAddressLine>
                    <lei:AdditionalAddressLine>West Wing</lei:AdditionalAddressLine>
                    <lei:City>London</lei:City>
                    <lei:PostalCode>SW1A 1AA</lei:PostalCode>
                    <lei:Country>GB</lei:Country>
                </lei:LegalAddress>
                <lei:Officers>
                    <lei:Officer>
                        <lei:FirstName>Jane</lei:FirstName>
                        <lei:LastName>Doe</lei:LastName>
                    </lei:Officer>
                </lei:Officers>
            </lei:Entity>
            <lei:Registration>
                <lei:InitialRegistrationDate>2015-03-10T00:00:00Z</lei:InitialRegistrationDate>
                <lei:ValidationSources>FULLY_CORROBORATED</lei:ValidationSources>
            </lei:Registration>
        </lei:LEIRecord>
    </lei:LEIRecords>
</lei:LEIData>
"""
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("gleif.xml", xml_payload)

    written_paths = shard_system_source(
        "gleif", roots=workspace_roots, run_date="2026-06-17", chunk_size=10
    )

    assert [path.name for path in written_paths] == ["gleif-001.parquet"]

    out = pl.read_parquet(written_paths[0])
    row = out.row(0, named=True)
    assert row["LEI"] == "ABC123"
    assert row["Entity_LegalName"] == "Alpha Ltd"
    assert row["Entity_LegalName__lang"] == "en"
    assert row["Entity_LegalJurisdiction"] == "GB"
    assert row["Entity_EntityStatus"] == "ACTIVE"
    assert (
        row["Entity_OtherEntityNames_OtherEntityName"]
        == "Alpha Trading | Alpha Holdings"
    )
    assert row["Entity_LegalAddress_AdditionalAddressLine"] == "Floor 2 | West Wing"
    assert row["Registration_ValidationSources"] == "FULLY_CORROBORATED"
    assert not any("Officer" in name for name in out.columns)

    names_path = source_dir / "gleif-names-001.parquet"
    assert names_path.exists()
    names_out = pl.read_parquet(names_path)
    assert (
        names_out["system_uri"].to_list() == [out.row(0, named=True)["system_uri"]] * 3
    )
    assert names_out["name"].to_list() == [
        "Alpha Ltd",
        "Alpha Trading",
        "Alpha Holdings",
    ]
    assert names_out["source_type"].to_list() == ["LEGAL_NAME", None, None]
    assert names_out["derivation_note"].to_list() == [
        "native LegalName",
        "native OtherEntityName",
        "native OtherEntityName",
    ]


def test_gleif_shard_system_source_ignores_non_schema_extension_fields(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "gleif-lei-cdf-2026-06-17.zip"
    xml_payload = """<?xml version=\"1.0\" encoding=\"UTF-8\"?>
<lei:LEIData xmlns:lei=\"http://www.gleif.org/data/schema/leidata/2016\" xmlns:ext=\"http://example.com/ext\">
    <lei:LEIRecords>
        <lei:LEIRecord>
            <lei:LEI>ABC123</lei:LEI>
            <lei:Entity>
                <lei:LegalName>Alpha Ltd</lei:LegalName>
                <lei:EntityStatus>ACTIVE</lei:EntityStatus>
            </lei:Entity>
            <lei:Registration>
                <lei:InitialRegistrationDate>2015-03-10T00:00:00Z</lei:InitialRegistrationDate>
                <lei:LastUpdateDate>2016-03-10T00:00:00Z</lei:LastUpdateDate>
                <lei:RegistrationStatus>ISSUED</lei:RegistrationStatus>
                <lei:NextRenewalDate>2017-03-10T00:00:00Z</lei:NextRenewalDate>
                <lei:ManagingLOU>12345678901234567890</lei:ManagingLOU>
            </lei:Registration>
            <lei:Extension>
                <ext:CustomField>SHOULD_NOT_APPEAR</ext:CustomField>
            </lei:Extension>
        </lei:LEIRecord>
    </lei:LEIRecords>
</lei:LEIData>
"""

    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr("gleif.xml", xml_payload)

    written_paths = shard_system_source(
        "gleif", roots=workspace_roots, run_date="2026-06-17", chunk_size=10
    )
    out = pl.read_parquet(written_paths[0])

    assert out["LEI"].to_list() == ["ABC123"]
    assert out["Entity_LegalName"].to_list() == ["Alpha Ltd"]
    assert not any("CustomField" in name for name in out.columns)


def test_gleif_shard_system_source_uses_latest_available_snapshot_when_run_date_omitted(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("gleif", layer="source") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "gleif-lei-cdf-2026-06-01.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "gleif.csv",
            "lei,entityName\nABC123,Alpha Ltd\nDEF456,Beta Ltd\n",
        )

    written_paths = shard_system_source(
        "gleif", roots=workspace_roots, run_date=None, chunk_size=1
    )

    assert all(path.parent == source_dir for path in written_paths)
    assert [path.name for path in written_paths] == [
        "gleif-001.parquet",
        "gleif-002.parquet",
    ]
    assert [pl.read_parquet(path).height for path in written_paths] == [1, 1]
