import json
from importlib.resources import files

import pytest
from company_cleanse import get_geographic_terms
from company_cleanse.geographic_terms import (
    GEOGRAPHIC_TERM_KIND_CITY,
    GEOGRAPHIC_TERM_KIND_COUNTRY,
    GEOGRAPHIC_TERM_KIND_REGION,
    GEOGRAPHIC_TERM_KINDS,
    _require_geographic_terms_shape,
    derive_iso20275_geographic_terms,
    resolve_geographic_term_kinds,
)


def _curated_payload() -> dict:
    resource_path = files("company_cleanse.resources").joinpath("geographic_terms.json")
    return json.loads(resource_path.read_text(encoding="utf-8"))


def test_country_tier_carries_iso20275_country_names():
    countries = get_geographic_terms(GEOGRAPHIC_TERM_KIND_COUNTRY)

    assert "France" in countries
    assert "Ireland" in countries
    assert "Germany" in countries


def test_country_tier_carries_the_curated_colloquial_forms():
    countries = get_geographic_terms(GEOGRAPHIC_TERM_KIND_COUNTRY)

    # These are exactly the forms ISO 3166's formal long names cannot supply.
    assert "UK" in countries
    assert "USA" in countries
    assert "Britain" in countries
    assert "Deutschland" in countries


def test_iso_codes_are_excluded_at_both_lengths():
    countries = get_geographic_terms(GEOGRAPHIC_TERM_KIND_COUNTRY)
    folded = {term.casefold() for term in countries}

    # Alpha-2 collides with ordinary words; alpha-3 is no safer.
    for code in ("gb", "de", "fr", "it", "in", "is", "at", "be", "us"):
        assert code not in folded
    for code in ("and", "are", "can", "per", "deu", "fra", "gbr"):
        assert code not in folded

    # ...but the unambiguous short forms that happen to look like codes stay.
    assert "usa" in folded
    assert "uk" in folded
    assert "uae" in folded


def test_region_tier_is_sub_national_and_disjoint_from_country_tier():
    countries = get_geographic_terms(GEOGRAPHIC_TERM_KIND_COUNTRY)
    regions = get_geographic_terms(GEOGRAPHIC_TERM_KIND_REGION)

    assert "Alberta" in regions
    assert "British Columbia" in regions
    assert "Flemish Region" in regions
    assert "Scotland" in regions
    assert "England" in regions

    folded_countries = {term.casefold() for term in countries}
    folded_regions = {term.casefold() for term in regions}
    assert not folded_countries & folded_regions

    # `Canada` and `Guernsey` appear in ISO 20275's jurisdiction column too; they
    # stay countries rather than being duplicated into the region tier.
    assert "canada" in folded_countries
    assert "canada" not in folded_regions


def test_city_tier_is_wired_but_empty():
    assert get_geographic_terms(GEOGRAPHIC_TERM_KIND_CITY) == ()
    assert GEOGRAPHIC_TERM_KIND_CITY in _curated_payload()["kinds"]


def test_country_tier_matches_the_derived_base_list_plus_the_curated_layer():
    derived = derive_iso20275_geographic_terms()
    curated_countries = {
        entry["term"]
        for entry in _curated_payload()["terms"]
        if entry["kind"] == GEOGRAPHIC_TERM_KIND_COUNTRY
    }

    expected = set(derived[GEOGRAPHIC_TERM_KIND_COUNTRY]) | curated_countries
    assert set(get_geographic_terms(GEOGRAPHIC_TERM_KIND_COUNTRY)) == expected


def test_derived_counts_match_the_source_resource():
    derived = derive_iso20275_geographic_terms()

    # 129 countries in entity_legal_forms_iso20275.json, plus the truncated leading
    # segment of each formal name that carries a parenthetical or comma qualifier.
    assert len(derived[GEOGRAPHIC_TERM_KIND_COUNTRY]) >= 129
    assert "Korea (Republic of)" in derived[GEOGRAPHIC_TERM_KIND_COUNTRY]
    assert "Korea" in derived[GEOGRAPHIC_TERM_KIND_COUNTRY]

    # 73 sub-national jurisdictions, less the `Rhode Island`/`Rhode island` casing
    # duplicate that collapses to one term.
    assert len(derived[GEOGRAPHIC_TERM_KIND_REGION]) == 72
    assert derived[GEOGRAPHIC_TERM_KIND_CITY] == ()


def test_terms_are_returned_read_only_and_sorted():
    for kind in GEOGRAPHIC_TERM_KINDS:
        terms = get_geographic_terms(kind)
        assert isinstance(terms, tuple)
        assert list(terms) == sorted(terms)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown geographic term kind"):
        get_geographic_terms("continent")


def test_resolve_kinds_defaults_to_country_and_always_includes_it():
    assert resolve_geographic_term_kinds(None) == (GEOGRAPHIC_TERM_KIND_COUNTRY,)
    assert resolve_geographic_term_kinds(()) == (GEOGRAPHIC_TERM_KIND_COUNTRY,)
    assert resolve_geographic_term_kinds(("region",)) == ("country", "region")
    assert resolve_geographic_term_kinds(("city", "region")) == (
        "country",
        "region",
        "city",
    )


def test_curated_resource_records_its_exclusions():
    payload = _curated_payload()

    assert "provenance" in payload
    exclusions = payload["exclusions"]
    assert "iso_alpha_2" in exclusions
    assert "iso_alpha_3" in exclusions
    assert "US" in exclusions


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ([], TypeError),
        ({"kinds": ["country"]}, TypeError),
        ({"kinds": ["planet"], "terms": []}, ValueError),
        ({"kinds": ["country"], "terms": [{"kind": "country"}]}, ValueError),
        ({"kinds": ["country"], "terms": [{"term": "UK", "kind": "moon"}]}, ValueError),
    ],
)
def test_curated_resource_shape_is_validated(payload, expected_error):
    with pytest.raises(expected_error):
        _require_geographic_terms_shape(payload, source_label="test payload")
