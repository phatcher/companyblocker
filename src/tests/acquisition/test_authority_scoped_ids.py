from __future__ import annotations

import json
from pathlib import Path

import pytest

from acquisition import authority_scoped_ids
from acquisition.authority_scoped_ids import get_crosswalk, resolve_authority_code


@pytest.fixture(autouse=True)
def clear_crosswalk_cache():
    authority_scoped_ids._load_crosswalk.cache_clear()
    yield
    authority_scoped_ids._load_crosswalk.cache_clear()


def test_get_crosswalk_loads_scheme_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(authority_scoped_ids, "_RESOURCES_DIR", tmp_path)
    (tmp_path / "xjustiz.json").write_text(
        json.dumps({"K1101R": "RA000197"}), encoding="utf-8"
    )

    assert get_crosswalk("xjustiz") == {"K1101R": "RA000197"}


def test_get_crosswalk_returns_empty_dict_when_scheme_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(authority_scoped_ids, "_RESOURCES_DIR", tmp_path)

    assert get_crosswalk("unknown-scheme") == {}


def test_resolve_authority_code_returns_mapped_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(authority_scoped_ids, "_RESOURCES_DIR", tmp_path)
    (tmp_path / "xjustiz.json").write_text(
        json.dumps({"K1101R": "RA000197"}), encoding="utf-8"
    )

    assert resolve_authority_code("xjustiz", "K1101R") == "RA000197"


def test_resolve_authority_code_returns_none_for_unmapped_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(authority_scoped_ids, "_RESOURCES_DIR", tmp_path)
    (tmp_path / "xjustiz.json").write_text(
        json.dumps({"K1101R": "RA000197"}), encoding="utf-8"
    )

    assert resolve_authority_code("xjustiz", "UNKNOWN") is None


@pytest.mark.parametrize("empty_value", [None, ""])
def test_resolve_authority_code_returns_none_without_hitting_crosswalk_for_empty_input(
    empty_value: str | None, monkeypatch: pytest.MonkeyPatch
):
    def _fail_if_called(scheme: str) -> dict[str, str]:
        raise AssertionError("get_crosswalk should not be called for empty input")

    monkeypatch.setattr(authority_scoped_ids, "get_crosswalk", _fail_if_called)

    assert resolve_authority_code("xjustiz", empty_value) is None
