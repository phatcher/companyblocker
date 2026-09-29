from __future__ import annotations

import json
from pathlib import Path

import pytest

from acquisition import company_type_mappings
from acquisition.company_type_mappings import get_company_type_mapping
from acquisition.registry import COUNTRY_REGISTRY, get_system_company_type_mapping


@pytest.fixture(autouse=True)
def clear_mapping_cache():
    get_company_type_mapping.cache_clear()
    yield
    get_company_type_mapping.cache_clear()


def test_supported_country_mapping_files_load_for_configured_columns():
    for country_code, plan in COUNTRY_REGISTRY.items():
        if not plan.company_type_column:
            continue
        mapping = get_company_type_mapping(country_code)
        assert isinstance(mapping, dict)


def test_get_company_type_mapping_normalizes_country_and_preserves_string_pairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_mappings, "_MAPPINGS_DIR", tmp_path)
    (tmp_path / "gb.json").write_text(
        json.dumps({" LTD ": "Limited", "PLC": "Public Limited Company"}),
        encoding="utf-8",
    )

    assert get_company_type_mapping(" GB ") == {
        " LTD ": "Limited",
        "PLC": "Public Limited Company",
    }


def test_get_company_type_mapping_returns_empty_dict_when_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_mappings, "_MAPPINGS_DIR", tmp_path)

    assert get_company_type_mapping("nl") == {}


def test_get_company_type_mapping_raises_for_blank_country():
    with pytest.raises(ValueError, match="Country code is required"):
        get_company_type_mapping("   ")


def test_get_company_type_mapping_raises_for_non_object_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_mappings, "_MAPPINGS_DIR", tmp_path)
    (tmp_path / "ie.json").write_text(
        json.dumps(["Limited"]),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must contain an object"):
        get_company_type_mapping("ie")


def test_get_company_type_mapping_raises_for_non_string_key_or_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_mappings, "_MAPPINGS_DIR", tmp_path)
    (tmp_path / "fr.json").write_text(
        json.dumps({"SARL": 123}),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must be string to string"):
        get_company_type_mapping("fr")


def test_supported_country_mapping_files_exist_for_configured_columns():
    for country_code, plan in COUNTRY_REGISTRY.items():
        if not plan.company_type_column:
            continue
        mapping = get_company_type_mapping(country_code)
        assert mapping, f"Expected non-empty mapping for {country_code}"


@pytest.mark.parametrize(
    "system,source_value,expected",
    [
        ("fr", "1000", "Entrepreneur individuel"),
        ("gb", "Private Limited Company", "Limited"),
        ("ie", "1119", "Private Limited Shares"),
    ],
)
def test_country_mapping_resolution_matches_registry_contract(
    system, source_value, expected
):
    mapping = get_system_company_type_mapping(system)
    assert mapping.get(source_value) == expected
