from __future__ import annotations

import json
from pathlib import Path

import pytest

from acquisition import company_type_excludes
from acquisition.company_type_excludes import get_company_type_exclusions
from acquisition.registry import COUNTRY_REGISTRY


@pytest.fixture(autouse=True)
def clear_exclusions_cache():
    get_company_type_exclusions.cache_clear()
    yield
    get_company_type_exclusions.cache_clear()


def test_supported_country_exclusion_files_load_for_configured_columns():
    for country_code, plan in COUNTRY_REGISTRY.items():
        if not plan.company_type_column:
            continue
        exclusions = get_company_type_exclusions(country_code)
        assert isinstance(exclusions, set)


def test_get_company_type_exclusions_normalizes_country_and_discards_blank_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_excludes, "_EXCLUDES_DIR", tmp_path)
    (tmp_path / "gb-exclude.json").write_text(
        json.dumps([" LTD ", "", "LTD", "PLC", "  "]),
        encoding="utf-8",
    )

    assert get_company_type_exclusions(" GB ") == {"LTD", "PLC"}


def test_get_company_type_exclusions_returns_empty_set_when_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_excludes, "_EXCLUDES_DIR", tmp_path)

    assert get_company_type_exclusions("nl") == set()


def test_get_company_type_exclusions_raises_for_blank_country():
    with pytest.raises(ValueError, match="Country code is required"):
        get_company_type_exclusions("   ")


def test_get_company_type_exclusions_raises_for_non_list_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_excludes, "_EXCLUDES_DIR", tmp_path)
    (tmp_path / "ie-exclude.json").write_text(
        json.dumps({"value": "LIMITED"}),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must contain a JSON array"):
        get_company_type_exclusions("ie")


def test_get_company_type_exclusions_raises_for_non_string_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(company_type_excludes, "_EXCLUDES_DIR", tmp_path)
    (tmp_path / "fr-exclude.json").write_text(
        json.dumps(["SARL", 123]),
        encoding="utf-8",
    )

    with pytest.raises(TypeError, match="must be strings"):
        get_company_type_exclusions("fr")
