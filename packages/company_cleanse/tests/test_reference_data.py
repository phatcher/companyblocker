from __future__ import annotations

import json

import pytest
from company_cleanse import reference_data


@pytest.fixture(autouse=True)
def _clear_cache():
    """`get_iso20275_entity_legal_forms` is `@cache`d, so tests that swap out the
    resource directory must clear it first, or a real result cached by an earlier
    test would still be returned."""
    reference_data.get_iso20275_entity_legal_forms.cache_clear()
    yield
    reference_data.get_iso20275_entity_legal_forms.cache_clear()


def test_get_iso20275_entity_legal_forms_loads_the_bundled_reference_data():
    legal_forms = reference_data.get_iso20275_entity_legal_forms()

    assert isinstance(legal_forms, dict)
    assert legal_forms
    sample = next(iter(legal_forms.values()))
    assert "country_code" in sample
    assert "entity_legal_form_name" in sample


def test_get_iso20275_entity_legal_forms_caches_the_result():
    first = reference_data.get_iso20275_entity_legal_forms()
    second = reference_data.get_iso20275_entity_legal_forms()

    assert first is second


def test_get_iso20275_entity_legal_forms_returns_empty_dict_when_resource_is_missing(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(reference_data, "_RESOURCES_DIR", tmp_path)

    assert reference_data.get_iso20275_entity_legal_forms() == {}


def test_get_iso20275_entity_legal_forms_rejects_non_object_json(tmp_path, monkeypatch):
    monkeypatch.setattr(reference_data, "_RESOURCES_DIR", tmp_path)
    (tmp_path / "entity_legal_forms_iso20275.json").write_text(
        json.dumps(["not", "an", "object"]), encoding="utf-8"
    )

    with pytest.raises(TypeError):
        reference_data.get_iso20275_entity_legal_forms()
