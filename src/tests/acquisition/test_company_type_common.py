from __future__ import annotations

import json
from pathlib import Path

import pytest

from acquisition.company_type_common import (
    load_country_scoped_json,
    load_country_scoped_payload,
    load_optional_json,
    normalize_company_type_country,
    normalize_string_map_payload,
    normalize_string_set_payload,
)


def test_normalize_company_type_country_strips_and_lowercases():
    assert normalize_company_type_country(" GB ", error_message="x") == "gb"


def test_normalize_company_type_country_raises_for_blank_country():
    with pytest.raises(ValueError, match="blank"):
        normalize_company_type_country("   ", error_message="blank country code")


def test_load_optional_json_returns_none_when_file_missing(tmp_path: Path):
    assert load_optional_json(tmp_path / "missing.json") is None


def test_load_optional_json_reads_existing_file(tmp_path: Path):
    path = tmp_path / "present.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    assert load_optional_json(path) == {"a": 1}


def test_load_country_scoped_json_builds_path_from_normalized_country(
    tmp_path: Path,
):
    (tmp_path / "gb.excludes.json").write_text(json.dumps(["Foo"]), encoding="utf-8")

    path, raw = load_country_scoped_json(
        country=" GB ",
        base_dir=tmp_path,
        filename_suffix=".excludes.json",
        error_message="x",
    )

    assert path == tmp_path / "gb.excludes.json"
    assert raw == ["Foo"]


def test_load_country_scoped_json_returns_none_payload_when_file_absent(
    tmp_path: Path,
):
    path, raw = load_country_scoped_json(
        country="fr",
        base_dir=tmp_path,
        filename_suffix=".json",
        error_message="x",
    )

    assert path == tmp_path / "fr.json"
    assert raw is None


def test_load_country_scoped_payload_applies_normalizer_to_raw_and_path(
    tmp_path: Path,
):
    (tmp_path / "de.json").write_text(json.dumps(["a", "b"]), encoding="utf-8")

    seen: dict[str, object] = {}

    def normalize_payload(raw: object, path: Path) -> str:
        seen["raw"] = raw
        seen["path"] = path
        return "normalized"

    result = load_country_scoped_payload(
        country="de",
        base_dir=tmp_path,
        filename_suffix=".json",
        error_message="x",
        normalize_payload=normalize_payload,
    )

    assert result == "normalized"
    assert seen == {"raw": ["a", "b"], "path": tmp_path / "de.json"}


def test_normalize_string_set_payload_strips_and_drops_blank_values(tmp_path: Path):
    result = normalize_string_set_payload(
        [" Foo ", "", "  ", "Bar"],
        path=tmp_path / "x.json",
        invalid_container_message="container",
        invalid_item_message="item",
    )

    assert result == {"Foo", "Bar"}


def test_normalize_string_set_payload_returns_empty_set_for_none(tmp_path: Path):
    assert (
        normalize_string_set_payload(
            None,
            path=tmp_path / "x.json",
            invalid_container_message="container",
            invalid_item_message="item",
        )
        == set()
    )


def test_normalize_string_set_payload_raises_for_non_list_container(tmp_path: Path):
    with pytest.raises(TypeError, match="container"):
        normalize_string_set_payload(
            {"a": 1},
            path=tmp_path / "x.json",
            invalid_container_message="container",
            invalid_item_message="item",
        )


def test_normalize_string_set_payload_raises_for_non_string_item(tmp_path: Path):
    with pytest.raises(TypeError, match="item"):
        normalize_string_set_payload(
            ["ok", 123],
            path=tmp_path / "x.json",
            invalid_container_message="container",
            invalid_item_message="item",
        )


def test_normalize_string_map_payload_preserves_string_pairs(tmp_path: Path):
    result = normalize_string_map_payload(
        {"LTD": "Limited"},
        path=tmp_path / "x.json",
        invalid_container_message="container",
        invalid_item_message="item",
    )

    assert result == {"LTD": "Limited"}


def test_normalize_string_map_payload_returns_empty_dict_for_none(tmp_path: Path):
    assert (
        normalize_string_map_payload(
            None,
            path=tmp_path / "x.json",
            invalid_container_message="container",
            invalid_item_message="item",
        )
        == {}
    )


def test_normalize_string_map_payload_raises_for_non_dict_container(tmp_path: Path):
    with pytest.raises(TypeError, match="container"):
        normalize_string_map_payload(
            ["a"],
            path=tmp_path / "x.json",
            invalid_container_message="container",
            invalid_item_message="item",
        )


def test_normalize_string_map_payload_raises_for_non_string_key_or_value(
    tmp_path: Path,
):
    with pytest.raises(TypeError, match="item"):
        normalize_string_map_payload(
            {"LTD": 1},
            path=tmp_path / "x.json",
            invalid_container_message="container",
            invalid_item_message="item",
        )
