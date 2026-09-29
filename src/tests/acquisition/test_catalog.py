import subprocess
import sys
from pathlib import Path

import pytest

from acquisition.catalog import load_catalog_plans
from acquisition.catalog.validator import validate_catalog_schema
from acquisition.models import SourceResource
from acquisition.registry import get_system_plan


def test_catalog_loader_rejects_missing_required_field(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "missing required field 'code'" in str(exc)
    else:
        raise AssertionError("Expected missing required field to be rejected")


def test_catalog_loader_rejects_invalid_resources_type(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","resources":{}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except TypeError as exc:
        assert "resources must be a list" in str(exc)
    else:
        raise AssertionError("Expected invalid resources type to be rejected")


def test_catalog_loader_rejects_invalid_registry_kind(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"invalid","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "Invalid registry kind" in str(exc)
    else:
        raise AssertionError("Expected invalid registry kind to be rejected")


def test_catalog_loader_rejects_non_string_required_field(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":123}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except TypeError as exc:
        assert "field 'notes' must be a string" in str(exc)
    else:
        raise AssertionError("Expected non-string required field to be rejected")


def test_catalog_loader_rejects_non_string_company_type_column(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","company_type_column":123}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "company_type_column" in str(exc)
    else:
        raise AssertionError("Expected non-string company_type_column to be rejected")


def test_catalog_loader_rejects_non_list_and_tokens(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","and_tokens":123}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "and_tokens" in str(exc)
    else:
        raise AssertionError("Expected non-list and_tokens to be rejected")


def test_catalog_loader_rejects_non_list_personal_owner_markers(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","personal_owner_markers":123}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "personal_owner_markers" in str(exc)
    else:
        raise AssertionError("Expected non-list personal_owner_markers to be rejected")


def test_catalog_loader_rejects_non_object_resource_entry(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","resources":["bad"]}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except TypeError as exc:
        assert "resources[0] must be an object" in str(exc)
    else:
        raise AssertionError("Expected non-object resource entry to be rejected")


def test_catalog_loader_rejects_invalid_research_type(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"research_required","access_mode":"search_portal","info_url":"https://example.com","notes":"x","research":"invalid"}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "research must be an object" in str(exc)
    else:
        raise AssertionError("Expected invalid research type to be rejected")


def test_catalog_loader_rejects_status_requiring_research_without_research_metadata(
    tmp_path,
):
    """A missing `research` block for `research_required` is only ever caught
    downstream, by `SystemPlan.__post_init__` -- confirm the diagnostic still
    surfaces through the full `load_catalog_plans` path, not just direct
    `SystemPlan(...)` construction."""
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"research_required","access_mode":"search_portal","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "xx: status 'research_required' requires research metadata" in str(exc)
    else:
        raise AssertionError(
            "Expected research_required status without research metadata to be rejected"
        )


def test_catalog_loader_rejects_canonical_derived_fields_non_object_spec(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_derived_fields":{"company_number":"not-an-object"}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except TypeError as exc:
        assert (
            "canonical_derived_fields['company_number'] must be an object, got str"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-object canonical_derived_fields spec to be rejected"
        )


def test_catalog_loader_rejects_canonical_derived_fields_wrong_type_value(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_derived_fields":{"company_number":{"type":"merge","parts":["a"]}}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "canonical_derived_fields['company_number'].type must be 'concat'"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-'concat' canonical_derived_fields type to be rejected"
        )


def test_catalog_loader_rejects_canonical_derived_fields_empty_parts(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_derived_fields":{"company_number":{"type":"concat","parts":[]}}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "canonical_derived_fields['company_number'].parts must be a non-empty list"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected empty canonical_derived_fields parts to be rejected"
        )


def test_catalog_loader_rejects_canonical_derived_fields_non_string_part(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_derived_fields":{"company_number":{"type":"concat","parts":[123]}}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "canonical_derived_fields['company_number'].parts[0] must be a non-empty string, got int"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-string canonical_derived_fields part to be rejected"
        )


def test_catalog_loader_rejects_canonical_derived_fields_non_string_separator(
    tmp_path,
):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_derived_fields":{"company_number":{"type":"concat","parts":["a"],"separator":5}}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except TypeError as exc:
        assert (
            "canonical_derived_fields['company_number'].separator must be a string, got int"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-string canonical_derived_fields separator to be rejected"
        )


def test_catalog_loader_rejects_system_field_candidates_non_list_value(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","system_field_candidates":{"name":"CompanyName"}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "system_field_candidates['name'] must be a non-empty list, got str"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-list system_field_candidates value to be rejected"
        )


def test_catalog_loader_rejects_system_field_candidates_empty_list_value(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","system_field_candidates":{"name":[]}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "system_field_candidates['name'] must be a non-empty list, got list"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected empty-list system_field_candidates value to be rejected"
        )


def test_catalog_loader_rejects_name_variant_type_map_non_string_value(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","name_variant_type_map":{"primary":123}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "name_variant_type_map['primary'] must be a non-empty string, got int"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-string name_variant_type_map value to be rejected"
        )


def test_catalog_loader_rejects_map_field_blank_key(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_source_column_aliases":{" ":"target"}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "canonical_source_column_aliases keys must be non-empty strings" in str(
            exc
        )
    else:
        raise AssertionError("Expected blank map key to be rejected")


def test_catalog_loader_rejects_supported_countries_blank_item(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","supported_countries":[""]}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "supported_countries[0] must be a non-empty string" in str(exc)
    else:
        raise AssertionError("Expected blank supported_countries item to be rejected")


def test_catalog_loader_rejects_primary_name_override_type_priority_non_string_item(
    tmp_path,
):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","primary_name_override_type_priority":[5]}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "primary_name_override_type_priority[0] must be a non-empty string, got int"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-string primary_name_override_type_priority item to be rejected"
        )


def test_catalog_loader_rejects_resource_read_defaults_extension_key_without_dot(
    tmp_path,
):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","resources":[{"name":"r1","url":"https://example.com/f","file_format":"csv","read_defaults":{"by_extension":{"csv":{"reader_mode":"line_iter"}}}}]}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "r1: read_defaults.by_extension keys must be dot-prefixed extensions"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected non-dot-prefixed read_defaults extension key to be rejected"
        )


def test_catalog_loader_rejects_resource_projection_defaults_invalid_engine(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","resources":[{"name":"r1","url":"https://example.com/f","file_format":"csv","projection_defaults":{"engine":"bogus"}}]}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "r1: projection_defaults.engine must be 'python' or "
            "'wikisieve' when provided" in str(exc)
        )
    else:
        raise AssertionError(
            "Expected invalid projection_defaults engine to be rejected"
        )


def test_catalog_loader_rejects_resource_output_defaults_invalid_chunk_size(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","resources":[{"name":"r1","url":"https://example.com/f","file_format":"csv","output_defaults":{"chunk_size":-5}}]}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "r1: output_defaults.chunk_size must be a positive integer or null"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected invalid output_defaults chunk_size to be rejected"
        )


def test_catalog_loader_rejects_file_name_mismatch(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "wrong-name.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"Bad","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "file name 'wrong-name' must match code 'xx'" in str(exc)
    else:
        raise AssertionError("Expected file-name/code mismatch to be rejected")


def test_catalog_loader_rejects_non_object_catalog_file(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "bad.json").write_text(
        '["not-an-object"]',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except TypeError as exc:
        assert "Invalid catalog file format" in str(exc)
    else:
        raise AssertionError("Expected non-object catalog file to be rejected")


def test_catalog_loader_accepts_matching_file_name_and_code(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "aa.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"aa","display_name":"A","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    country, system = load_catalog_plans(systems_dir)
    assert "aa" in country
    assert system == {}


def test_catalog_loader_returns_empty_registries_when_systems_dir_is_missing(tmp_path):
    country, system = load_catalog_plans(tmp_path / "missing")

    assert country == {}
    assert system == {}


def test_catalog_loader_places_system_registry_entries_in_system_bucket(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "gleif.json").write_text(
        '{"registry":"system","all_systems_target":false,"code":"gleif","display_name":"GLEIF","region":"global","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    country, system = load_catalog_plans(systems_dir)

    assert country == {}
    assert set(system) == {"gleif"}


def test_catalog_loader_returns_expected_seeded_codes():
    country, system = load_catalog_plans()

    for code in ("dk", "ee", "fi", "fr", "gb", "ie", "nl"):
        assert code in country

    assert "wikidata" in system
    assert "edgar" in system
    assert "gleif" in system
    assert "dbpedia" in system
    assert "offeneregister" in system


def test_registry_uses_catalog_definitions_for_seeded_systems():
    gb = get_system_plan("gb")
    assert gb.resources[0].name == "basic_company_data_as_one_file"

    nl = get_system_plan("nl")
    assert nl.status == "research_required"
    assert nl.research is not None

    wikidata = get_system_plan("wikidata")
    assert wikidata.status == "live"
    assert wikidata.resources[0].adapter == "direct_download"
    assert wikidata.resources[0].read_defaults is not None
    assert wikidata.resources[0].projection_defaults is not None
    assert wikidata.resources[0].projection_defaults["engine"] == "wikisieve"

    edgar = get_system_plan("edgar")
    assert edgar.status == "research_required"
    assert edgar.resources == ()

    offeneregister = get_system_plan("offeneregister")
    assert offeneregister.status == "live"
    assert offeneregister.resources[0].adapter == "direct_download"
    assert offeneregister.canonical_derived_fields == (
        ("company_number", ("register_art", "register_number"), " "),
    )

    gleif = get_system_plan("gleif")
    assert gleif.resources[0].adapter == "gleif_latest_concatenated"


def test_catalog_loader_loads_all_committed_catalog_files():
    country, system = load_catalog_plans()
    assert len(country) == 29
    # +1 for "perturbed": the single catalog-registered pseudo-system materialized
    # perturbation output resolves against, rather than one entry per profile.
    assert len(system) == 6


def test_catalog_schema_validator_accepts_committed_catalog_files():
    files = validate_catalog_schema()
    assert len(files) == 35


def test_catalog_schema_validator_rejects_invalid_enum_value(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"invalid_mode","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    except ValueError as exc:
        assert "Schema validation failed" in str(exc)
        assert "access_mode" in str(exc)
    else:
        raise AssertionError("Expected schema validation to fail on invalid enum")


def test_catalog_schema_validator_accepts_null_research(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","research":null}',
        encoding="utf-8",
    )

    files = validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    assert len(files) == 1


def test_catalog_schema_validator_accepts_supported_countries_selector(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"system","all_systems_target":false,"code":"xx","display_name":"X","region":"global","status":"live","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","supported_countries":["live"]}',
        encoding="utf-8",
    )

    files = validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    assert len(files) == 1


def test_catalog_schema_validator_accepts_supported_countries_list(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"system","all_systems_target":false,"code":"xx","display_name":"X","region":"global","status":"live","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","supported_countries":["gb","ie"]}',
        encoding="utf-8",
    )

    files = validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    assert len(files) == 1


def test_catalog_schema_validator_accepts_research_execution_policy_fields(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"system","all_systems_target":false,"code":"xx","display_name":"X","region":"global","status":"research_required","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","research":{"pass_date":"2026-06-26","basis":"test","priority":"P2","allow_research_runtime":true,"allowed_stages":["acquire","shard"]}}',
        encoding="utf-8",
    )

    files = validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    assert len(files) == 1


def test_source_resource_resolve_read_options_by_extension():
    resource = SourceResource(
        name="wikidata_entities_latest_all",
        url="https://example.com/latest-all.json.gz",
        file_format="jsonl",
        read_defaults={
            "by_extension": {
                ".gz": {"reader_mode": "line_iter"},
                ".bz2": {"reader_mode": "chunked_indexed", "chunk_size_mb": 16},
            },
            "runtime": {"indexed_bzip2_parallelization": "auto"},
        },
    )

    gz_options = resource.resolve_read_options_for_path(Path("wikidata-all.json.gz"))
    bz2_options = resource.resolve_read_options_for_path(Path("wikidata-all.json.bz2"))

    assert gz_options["reader_mode"] == "line_iter"
    assert gz_options["indexed_bzip2_parallelization"] == "auto"
    assert bz2_options["reader_mode"] == "chunked_indexed"
    assert bz2_options["chunk_size_bytes"] == 16 * 1024 * 1024


def test_source_resource_accepts_projection_defaults():
    resource = SourceResource(
        name="wikidata_entities_latest_all",
        url="https://example.com/latest-all.json.gz",
        file_format="jsonl",
        projection_defaults={
            "engine": "wikisieve",
            "binary_path": "tools/bin/wikisieve.exe",
            "output_mode": "jsonl",
        },
    )

    assert resource.projection_defaults is not None
    assert resource.projection_defaults["engine"] == "wikisieve"


def test_catalog_loader_accepts_system_uri_identifier_candidates(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","system_uri_identifier_candidates":["company_number","CompanyNumber"]}',
        encoding="utf-8",
    )

    country, system = load_catalog_plans(systems_dir)

    assert "xx" in country
    assert system == {}
    assert country["xx"].system_uri_identifier_candidates == (
        "company_number",
        "CompanyNumber",
    )


def test_catalog_loader_accepts_system_field_candidates(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","system_field_candidates":{"name":["CompanyName"],"company_number":["CompanyNumber"],"company_type":["CompanyType"],"current_status":["status"],"incorporation_date":["incorporation_date"],"dissolution_date":["dissolution_date"],"registry_url":["registry_url"],"registered_address_in_full":["registered_address_in_full"],"inactive":["inactive"],"branch":["branch"],"branch_status":["branch_status"]}}',
        encoding="utf-8",
    )

    country, system = load_catalog_plans(systems_dir)

    assert "xx" in country
    assert system == {}
    assert country["xx"].system_field_candidates is not None
    assert dict(country["xx"].system_field_candidates)["name"] == ("CompanyName",)


def test_catalog_loader_accepts_canonical_source_column_aliases(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_source_column_aliases":{"company_num":"company_number"}}',
        encoding="utf-8",
    )

    country, system = load_catalog_plans(systems_dir)

    assert "xx" in country
    assert system == {}
    assert country["xx"].canonical_source_column_aliases == (
        ("company_num", "company_number"),
    )


def test_catalog_loader_accepts_canonical_input_columns(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_input_columns":["official_name","aliases_en"]}',
        encoding="utf-8",
    )

    country, system = load_catalog_plans(systems_dir)

    assert "xx" in country
    assert system == {}
    assert country["xx"].canonical_input_columns == ("official_name", "aliases_en")


def test_catalog_loader_accepts_canonical_derived_fields(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"system","all_systems_target":false,"code":"xx","display_name":"X","region":"global","status":"live","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_derived_fields":{"company_number":{"type":"concat","parts":["register_art","register_number"],"separator":" "}}}',
        encoding="utf-8",
    )

    country, system = load_catalog_plans(systems_dir)

    assert country == {}
    assert "xx" in system
    assert system["xx"].canonical_derived_fields == (
        ("company_number", ("register_art", "register_number"), " "),
    )


def test_catalog_schema_validator_rejects_blank_system_uri_identifier_candidate(
    tmp_path,
):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","system_uri_identifier_candidates":[""]}',
        encoding="utf-8",
    )

    try:
        validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    except ValueError as exc:
        assert "Schema validation failed" in str(exc)
        assert "system_uri_identifier_candidates" in str(exc)
    else:
        raise AssertionError(
            "Expected schema validation to fail on blank system_uri identifier candidate"
        )


def test_catalog_schema_validator_rejects_empty_system_field_candidates_list(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","system_field_candidates":{"name":[]}}',
        encoding="utf-8",
    )

    try:
        validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    except ValueError as exc:
        assert "Schema validation failed" in str(exc)
        assert "system_field_candidates" in str(exc)
    else:
        raise AssertionError(
            "Expected schema validation to fail on empty system_field_candidates values"
        )


def test_catalog_schema_validator_rejects_blank_canonical_source_alias_target(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":false,"code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x","canonical_source_column_aliases":{"company_num":""}}',
        encoding="utf-8",
    )

    try:
        validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    except ValueError as exc:
        assert "Schema validation failed" in str(exc)
        assert "canonical_source_column_aliases" in str(exc)
    else:
        raise AssertionError(
            "Expected schema validation to fail on blank canonical_source_column_aliases target"
        )


@pytest.mark.integration
def test_validate_catalog_script_runs_successfully():
    result = subprocess.run(
        [sys.executable, "scripts/validate_catalog.py"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Validated" in result.stdout


def test_catalog_loader_rejects_missing_all_systems_target(tmp_path):
    """The flag is required rather than defaulted: a system cannot be
    onboarded without someone deciding whether `--systems all` targets it, and
    an absent flag meaning "excluded" would put that rule back in whoever
    remembers it."""
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert "missing required field 'all_systems_target'" in str(exc)
    else:
        raise AssertionError("Expected missing all_systems_target to be rejected")


def test_catalog_loader_rejects_non_boolean_all_systems_target(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":"yes","code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except TypeError as exc:
        assert "field 'all_systems_target' must be a boolean, got str" in str(exc)
    else:
        raise AssertionError("Expected non-boolean all_systems_target to be rejected")


def test_catalog_loader_rejects_all_systems_target_with_non_runnable_status(tmp_path):
    """The flag and the status cannot drift apart silently: a system
    `--systems all` sweeps must carry a status a run can actually proceed
    against."""
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    (systems_dir / "xx.json").write_text(
        '{"registry":"country","all_systems_target":true,"code":"xx","display_name":"X","region":"eu","status":"research_required","access_mode":"search_portal","info_url":"https://example.com","notes":"x","research":{"basis":"test"}}',
        encoding="utf-8",
    )

    try:
        load_catalog_plans(systems_dir)
    except ValueError as exc:
        assert (
            "xx: all_systems_target is set but status 'research_required' is not runnable"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Expected all_systems_target with a non-runnable status to be rejected"
        )


def test_catalog_schema_validator_rejects_missing_all_systems_target(tmp_path):
    systems_dir = tmp_path / "systems"
    systems_dir.mkdir(parents=True, exist_ok=True)
    schema_path = Path("src/acquisition/catalog/systems.schema.json")

    (systems_dir / "xx.json").write_text(
        '{"registry":"country","code":"xx","display_name":"X","region":"eu","status":"supported","access_mode":"bulk_download","info_url":"https://example.com","notes":"x"}',
        encoding="utf-8",
    )

    try:
        validate_catalog_schema(systems_dir=systems_dir, schema_path=schema_path)
    except ValueError as exc:
        assert "Schema validation failed" in str(exc)
        assert "all_systems_target" in str(exc)
    else:
        raise AssertionError(
            "Expected schema validation to fail on a missing all_systems_target"
        )


def test_committed_catalog_declares_all_membership_independently_of_status():
    """Membership is the catalog's own declaration, not a reading of `status`
    . `perturbed` is `live` and excluded -- `company_perturbation`
     generates it rather than acquiring it -- so the two answers differ, which
     is the whole point of the flag."""
    country, system = load_catalog_plans()
    plans = {**country, **system}

    flagged = {code for code, plan in plans.items() if plan.all_systems_target}
    assert flagged == {"fr", "gb", "gleif", "ie", "offeneregister", "wikidata"}

    assert plans["perturbed"].status == "live"
    assert plans["perturbed"].all_systems_target is False
    assert plans["wikidata"].status == "live"
    assert plans["wikidata"].all_systems_target is True

    for code, plan in plans.items():
        if plan.all_systems_target:
            assert plan.status in {"live", "supported"}, code
