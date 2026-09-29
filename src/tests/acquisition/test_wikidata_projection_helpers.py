from __future__ import annotations

import json

from acquisition.io_eta_helpers import _json_loads
from acquisition.wikidata_projection_helpers import (
    _extract_claim_entity_ids,
    _extract_claim_text_values,
    _extract_claim_text_values_by_language,
    _extract_claim_text_values_with_english,
    _extract_en_aliases,
    _extract_en_description,
    _extract_en_label,
    _extract_instance_of_fast,
    _extract_projection_claims,
    _extract_sitelinks_count,
    _extract_time_claim,
    _get_langstring_field,
    _is_company_like,
    _project_wikidata_company_record_line,
)


def _claim(property_id: str, *values: object) -> dict[str, object]:
    return {
        "claims": {
            property_id: [
                {"mainsnak": {"datavalue": {"value": value}}} for value in values
            ]
        }
    }


def test_extract_claim_entity_ids_reads_ids_from_claims():
    entity = _claim("P17", {"id": "Q145"}, {"id": "Q142"})

    assert _extract_claim_entity_ids(entity, "P17") == ["Q145", "Q142"]


def test_extract_claim_entity_ids_returns_empty_list_for_missing_property():
    assert _extract_claim_entity_ids({"claims": {}}, "P17") == []
    assert _extract_claim_entity_ids({}, "P17") == []


def test_extract_claim_entity_ids_skips_malformed_claim_shapes():
    entity = {
        "claims": {
            "P17": [
                "not-a-dict",
                {"mainsnak": "not-a-dict"},
                {"mainsnak": {"datavalue": "not-a-dict"}},
                {"mainsnak": {"datavalue": {"value": None}}},
                {"mainsnak": {"datavalue": {"value": {"id": "Q145"}}}},
            ]
        }
    }

    assert _extract_claim_entity_ids(entity, "P17") == ["Q145"]


def test_extract_time_claim_returns_first_time_value():
    entity = _claim("P571", {"time": "+2001-01-01T00:00:00Z"})

    assert _extract_time_claim(entity, "P571") == "+2001-01-01T00:00:00Z"


def test_extract_time_claim_returns_none_when_absent():
    assert _extract_time_claim({"claims": {}}, "P571") is None


def test_extract_claim_text_values_handles_plain_strings_and_langstrings():
    entity = _claim("P856", "https://example.test", {"text": "Acme", "language": "en"})

    assert _extract_claim_text_values(entity, "P856") == [
        "https://example.test",
        "Acme",
    ]


def test_extract_claim_text_values_by_language_filters_to_requested_language():
    entity = _claim(
        "P1448",
        {"text": "Acme GmbH", "language": "de"},
        {"text": "Acme Ltd", "language": "en"},
    )

    assert _extract_claim_text_values_by_language(entity, "P1448", language="de") == [
        "Acme GmbH"
    ]


def test_extract_claim_text_values_with_english_returns_all_english_and_tagged():
    entity = _claim(
        "P1448",
        {"text": "Acme GmbH", "language": "de"},
        {"text": "Acme Ltd", "language": "en"},
        "Acme",
    )

    all_values, english_values, tagged_values = _extract_claim_text_values_with_english(
        entity, "P1448"
    )

    assert all_values == ["Acme GmbH", "Acme Ltd", "Acme"]
    assert english_values == ["Acme Ltd"]
    assert tagged_values == [
        {"value": "Acme GmbH", "language": "de"},
        {"value": "Acme Ltd", "language": "en"},
        {"value": "Acme", "language": None},
    ]


def test_extract_instance_of_fast_delegates_to_p31():
    entity = _claim("P31", {"id": "Q783794"})

    assert _extract_instance_of_fast(entity) == ["Q783794"]


def test_get_langstring_field_returns_none_for_missing_or_wrong_type_field():
    assert _get_langstring_field({}, "labels") is None
    assert _get_langstring_field({"labels": "not-a-dict"}, "labels") is None


def test_get_langstring_field_returns_language_entry():
    entity = {"labels": {"en": {"value": "Acme"}}}

    assert _get_langstring_field(entity, "labels", "en") == {"value": "Acme"}


def test_extract_en_label_description_and_aliases():
    entity = {
        "labels": {"en": {"value": "Acme"}},
        "descriptions": {"en": {"value": "a company"}},
        "aliases": {"en": [{"value": "Acme Corp"}, {"value": ""}, "not-a-dict"]},
    }

    assert _extract_en_label(entity) == "Acme"
    assert _extract_en_description(entity) == "a company"
    assert _extract_en_aliases(entity) == ["Acme Corp"]


def test_extract_en_label_returns_none_when_absent():
    assert _extract_en_label({}) is None
    assert _extract_en_description({}) is None
    assert _extract_en_aliases({}) == []


def test_extract_sitelinks_count_counts_entries():
    assert _extract_sitelinks_count({"sitelinks": {"enwiki": {}, "dewiki": {}}}) == 2
    assert _extract_sitelinks_count({}) == 0


def test_extract_projection_claims_populates_every_declared_key_and_instance_of():
    entity = {
        "claims": {
            "P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q145"}}}}],
            "P1278": [{"mainsnak": {"datavalue": {"value": "LEI123"}}}],
            "P1448": [
                {
                    "mainsnak": {
                        "datavalue": {"value": {"text": "Acme", "language": "en"}}
                    }
                }
            ],
        }
    }

    claims, instance_of = _extract_projection_claims(entity)

    assert claims["country"] == ["Q145"]
    assert claims["lei"] == ["LEI123"]
    assert claims["official_name"] == ["Acme"]
    assert claims["official_name_en"] == ["Acme"]
    assert claims["jurisdiction"] == []
    assert claims["inception"] is None
    assert instance_of == []  # extract_projection_claims never populates instance_of


def test_extract_projection_claims_returns_defaults_without_claims_key():
    claims, instance_of = _extract_projection_claims({"id": "Q1"})

    assert claims["country"] == []
    assert claims["lei"] == []
    assert instance_of == []


def test_is_company_like_matches_any_known_qid():
    assert _is_company_like(["Q999999", "Q783794"]) is True
    assert _is_company_like(["Q999999"]) is False
    assert _is_company_like([]) is False


def _company_entity(**overrides: object) -> dict[str, object]:
    entity: dict[str, object] = {
        "id": "Q1",
        "type": "item",
        "modified": "2026-01-01T00:00:00Z",
        "labels": {"en": {"value": "Acme"}},
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }
    entity.update(overrides)
    return entity


def test_project_wikidata_company_record_line_projects_a_company_entity():
    line = json.dumps(_company_entity()).encode("utf-8")

    projected = _project_wikidata_company_record_line(line)

    assert projected is not None
    record = _json_loads(projected)
    assert record["id"] == "Q1"
    assert record["label_en"] == "Acme"
    assert record["instance_of"] == ["Q783794"]
    assert "company_number" not in record


def test_project_wikidata_company_record_line_returns_none_for_missing_id():
    line = json.dumps({"claims": {}}).encode("utf-8")

    assert _project_wikidata_company_record_line(line) is None


def test_project_wikidata_company_record_line_returns_none_for_unparseable_line():
    assert _project_wikidata_company_record_line(b"not-json") is None


def test_project_wikidata_company_record_line_without_legal_form_qids_trusts_pass_one():
    # legal_form_qids omitted: no instance_of/legal_form filtering happens here,
    # even for an entity that isn't company-like on its own.
    entity = _company_entity()
    entity["claims"] = {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]}
    line = json.dumps(entity).encode("utf-8")

    projected = _project_wikidata_company_record_line(line)

    assert projected is not None


def test_project_wikidata_company_record_line_with_legal_form_qids_filters_non_company():
    entity = _company_entity()
    entity["claims"] = {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}]}
    line = json.dumps(entity).encode("utf-8")

    assert (
        _project_wikidata_company_record_line(line, legal_form_qids=frozenset()) is None
    )


def test_project_wikidata_company_record_line_admits_legal_form_match_when_not_company_like():
    entity = _company_entity()
    entity["claims"] = {
        "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
        "P1454": [{"mainsnak": {"datavalue": {"value": {"id": "Q783772"}}}}],
    }
    line = json.dumps(entity).encode("utf-8")

    projected = _project_wikidata_company_record_line(
        line, legal_form_qids=frozenset({"Q783772"})
    )

    assert projected is not None
    record = _json_loads(projected)
    assert record["matched_company_type_qids"] == ["Q783772"]
