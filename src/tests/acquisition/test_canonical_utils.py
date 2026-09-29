from __future__ import annotations

import re

import polars as pl
import pytest

from acquisition import canonical_utils
from acquisition.canonical_system_config import _WIKIDATA_COUNTRY_QID_TO_ISO2
from acquisition.canonical_utils import (
    coalesce_date_iso,
    coalesce_name_list,
    coalesce_text,
    dedupe_names,
    has_latin_char,
    normalize_name_list,
    resolve_wikidata_country_iso2_expr,
    select_first_non_empty,
    select_wikidata_name_bundle,
    to_name_list_or_none,
    to_text_or_none,
)


def test_has_latin_char_detects_latin_and_rejects_non_latin_scripts() -> None:
    assert has_latin_char("Alpha Ltd") is True
    assert has_latin_char("マルチチュード") is False
    assert has_latin_char('מאלטיטיוד בע"מ') is False
    # Mixed script: any Latin character is enough to count as True.
    assert has_latin_char("マルチチュード Ltd") is True


def test_select_first_non_empty_returns_first_truthy_candidate_in_order() -> None:
    assert select_first_non_empty([None, "", "second", "third"]) == "second"
    assert select_first_non_empty([]) is None
    assert select_first_non_empty([None, None]) is None


def test_to_name_list_or_none_handles_string_list_and_struct_values() -> None:
    assert to_name_list_or_none(None) is None
    assert to_name_list_or_none("  Alpha  ") == ["Alpha"]
    assert to_name_list_or_none("   ") is None
    assert to_name_list_or_none([" Alpha ", "", "Beta"]) == ["Alpha", "Beta"]
    assert to_name_list_or_none(
        [
            {"company_name": " Legacy One "},
            {"company_name": ""},
            {"not_company_name": "ignored"},
        ]
    ) == ["Legacy One"]


def test_normalize_name_list_returns_empty_list_when_missing() -> None:
    assert normalize_name_list(None) == []
    assert normalize_name_list("Acme") == ["Acme"]


def test_dedupe_names_dedupes_case_insensitively_and_preserves_first_seen() -> None:
    values = ["Acme", "Acme", " ACME ", "", "Beta", "Beta"]
    assert dedupe_names(values) == ["Acme", "Beta"]


def test_select_wikidata_name_bundle_prefers_english_legal_and_builds_alternatives() -> (
    None
):
    payload = {
        "label_en": "Acme Label",
        "official_name_en": ["Acme Corporation"],
        "official_name": ["Acme Corporation", "Societe Acme"],
        "short_name_en": ["ACME"],
        "short_name": ["ACME"],
        "aliases_en": ["Acme", "Acme Corporation"],
    }

    bundle = select_wikidata_name_bundle(payload)

    assert bundle["name"] == "Acme Corporation"
    assert bundle["alternative_names"] == ["Societe Acme", "ACME"]


def test_select_wikidata_name_bundle_dedupes_alternatives_case_insensitively() -> None:
    payload = {
        "label_en": "Air New Zealand",
        "official_name_en": ["Air New Zealand Limited"],
        "official_name": ["AIR NEW ZEALAND LIMITED"],
        "short_name_en": [],
        "short_name": ["ANZ"],
        "aliases_en": ["Air New Zealand Limited", "ANZ", "AIZ", "Air New Zealand Ltd"],
    }

    bundle = select_wikidata_name_bundle(payload)

    assert bundle["name"] == "Air New Zealand Limited"
    assert bundle["alternative_names"] == ["ANZ", "AIZ", "Air New Zealand Ltd"]


def test_select_wikidata_name_bundle_prefers_latin_official_name_when_no_english_legal_name() -> (
    None
):
    payload = {
        "label_en": "Maserati",
        "official_name_en": [],
        "official_name": ["マセラティ", "Maserati S.p.A."],
        "short_name_en": [],
        "short_name": [],
        "aliases_en": [],
    }

    bundle = select_wikidata_name_bundle(payload)

    assert bundle["name"] == "Maserati S.p.A."
    assert bundle["alternative_names"] == ["マセラティ"]


def test_select_wikidata_name_bundle_falls_back_to_label_en() -> None:
    payload = {
        "label_en": "Fallback Label",
        "official_name_en": [],
        "official_name": [],
        "short_name_en": [],
        "short_name": [],
        "aliases_en": ["Alias One"],
    }

    bundle = select_wikidata_name_bundle(payload)

    assert bundle["name"] == "Fallback Label"
    assert bundle["alternative_names"] == ["Alias One"]


def test_resolve_wikidata_country_iso2_expr_maps_country_then_jurisdiction() -> None:
    frame = pl.DataFrame(
        {
            "country": [["Q145"], None, ["Q999999"]],
            "jurisdiction": [None, ["Q183"], None],
        }
    )

    out = frame.select(
        resolve_wikidata_country_iso2_expr(set(frame.columns)).alias(
            "jurisdiction_code"
        )
    )
    assert out["jurisdiction_code"].to_list() == ["GB", "DE", None]


def test_resolve_wikidata_country_iso2_expr_returns_none_for_unmapped_qid() -> None:
    """An unmapped QID must resolve to None, never leak through as a raw QID string."""
    frame = pl.DataFrame(
        {
            "country": [["Q999999"]],
        }
    )

    out = frame.select(
        resolve_wikidata_country_iso2_expr(set(frame.columns)).alias(
            "jurisdiction_code"
        )
    )
    assert out["jurisdiction_code"].to_list() == [None]


def test_resolve_wikidata_country_iso2_expr_resolves_via_qid_closure_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A QID absent from the static ISO2 table but present in the QID
    closure cache (e.g. a sub-national entity resolved to its country via
    the Wikidata Query Service) must resolve through it."""
    monkeypatch.setattr(
        canonical_utils,
        "_load_wikidata_qid_country_closure",
        lambda: {"Q99999": "Q145"},
    )
    frame = pl.DataFrame({"country": [["Q99999"]]})

    out = frame.select(
        resolve_wikidata_country_iso2_expr(set(frame.columns)).alias(
            "jurisdiction_code"
        )
    )
    assert out["jurisdiction_code"].to_list() == ["GB"]


def test_resolve_wikidata_country_iso2_expr_returns_none_when_closure_target_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A QID resolved by the closure cache to a country QID that's itself
    not in the static ISO2 table (e.g. a historical/defunct state) must
    still resolve to None, never leak either QID through."""
    monkeypatch.setattr(
        canonical_utils,
        "_load_wikidata_qid_country_closure",
        lambda: {
            "Q99999": "Q12548"
        },  # Q12548: Holy Roman Empire, not in the ISO2 table
    )
    frame = pl.DataFrame({"country": [["Q99999"]]})

    out = frame.select(
        resolve_wikidata_country_iso2_expr(set(frame.columns)).alias(
            "jurisdiction_code"
        )
    )
    assert out["jurisdiction_code"].to_list() == [None]


_QID_SHAPE = re.compile(r"^Q\d+$", re.IGNORECASE)


def test_resolve_wikidata_country_iso2_expr_never_returns_a_qid_shaped_value() -> None:
    """Regression guard: resolving every QID this repo currently knows about
    -- every key in the static ISO2 table, every key and every value in the
    real checked-in QID closure cache -- must never produce a QID-shaped
    output. A leaked QID is strictly worse than a missing value, so this
    can't silently reappear."""
    candidate_qids = (
        set(_WIKIDATA_COUNTRY_QID_TO_ISO2)
        | set(canonical_utils._load_wikidata_qid_country_closure())
        | set(canonical_utils._load_wikidata_qid_country_closure().values())
    )
    assert candidate_qids, "expected at least one real QID to check against"

    frame = pl.DataFrame({"country": [[qid] for qid in sorted(candidate_qids)]})
    out = frame.select(
        resolve_wikidata_country_iso2_expr(set(frame.columns)).alias(
            "jurisdiction_code"
        )
    )
    for value in out["jurisdiction_code"].to_list():
        assert value is None or not _QID_SHAPE.match(value), (
            f"jurisdiction_code resolved to a raw QID-shaped value: {value!r}"
        )


def test_resolve_wikidata_country_iso2_expr_matches_country_names_case_insensitively() -> (
    None
):
    frame = pl.DataFrame(
        {
            "Country": [["Q145"], None],
            "Jurisdiction": [None, ["Q183"]],
        }
    )

    out = frame.select(
        resolve_wikidata_country_iso2_expr(set(frame.columns)).alias(
            "jurisdiction_code"
        )
    )
    assert out["jurisdiction_code"].to_list() == ["GB", "DE"]


def test_coalesce_text_uses_candidate_order_and_normalizes_empty() -> None:
    frame = pl.DataFrame(
        {
            "a": ["  ", None, "Primary"],
            "b": ["Fallback", "  Alt  ", None],
        }
    )

    out = frame.select(coalesce_text(set(frame.columns), ["a", "b"]).alias("value"))
    assert out["value"].to_list() == ["Fallback", "Alt", "Primary"]


def test_coalesce_text_handles_list_typed_values() -> None:
    frame = pl.DataFrame(
        {
            "company_number": [None, "", None],
            "lei": [["  5493001KJTIIGC8Y1R12  "], [""], ["  ", "549300ABC"]],
        }
    )

    out = frame.select(
        coalesce_text(set(frame.columns), ["company_number", "lei"]).alias("value")
    )
    assert out["value"].to_list() == ["5493001KJTIIGC8Y1R12", None, "549300ABC"]


def test_coalesce_text_matches_candidates_case_insensitively() -> None:
    frame = pl.DataFrame(
        {
            "CompanyName": ["Example Limited", None],
            "Fallback": [None, "Fallback Ltd"],
        }
    )

    out = frame.select(
        coalesce_text(set(frame.columns), ["companyname", "fallback"]).alias("value")
    )
    assert out["value"].to_list() == ["Example Limited", "Fallback Ltd"]


def test_to_text_or_none_handles_polars_series_from_list_column() -> None:
    nested = pl.Series("", ["Y4DQT2X5DDRE70QFR206"])
    assert to_text_or_none(nested) == "Y4DQT2X5DDRE70QFR206"


def test_coalesce_date_iso_accepts_supported_formats() -> None:
    frame = pl.DataFrame(
        {
            "d1": ["2020-01-02", None, None, None],
            "d2": [None, "2020-01-03T00:00:00Z", None, None],
            "d3": [None, None, "04/01/2020", None],
            "d4": [None, None, None, "20200105"],
        }
    )

    out = frame.select(
        coalesce_date_iso(set(frame.columns), ["d1", "d2", "d3", "d4"]).alias("date")
    )
    assert out["date"].to_list() == [
        "2020-01-02",
        "2020-01-03",
        "2020-01-04",
        "2020-01-05",
    ]


def test_coalesce_name_list_handles_strings_and_company_name_structs() -> None:
    frame = pl.DataFrame(
        {
            "names_a": ["Legacy A", None, ""],
            "names_b": ["Alt 1", "Only B", "Trailing"],
        }
    )

    out = frame.select(
        coalesce_name_list(set(frame.columns), ["names_a", "names_b"]).alias("names")
    )
    assert out["names"].to_list() == [["Legacy A"], ["Only B"], ["Trailing"]]
