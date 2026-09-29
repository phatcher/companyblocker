from __future__ import annotations

import re

import pytest
from company_cleanse.rules import (
    CANONICAL_COMPANY_TYPE_PATTERN,
    UnknownCompanyTypeCountryError,
    _build_company_type_mapping,
    _build_company_type_suffix_trie,
    _load_company_type_rules,
    _match_company_type_from_suffix_surface,
    get_company_type_rules,
    get_company_type_rules_for_country,
)


def test_packaged_company_type_rules_load():
    rules = _load_company_type_rules()

    assert rules
    assert any(
        source == "Limited" and canonical == "Ltd" and country == "gb"
        for source, canonical, country in rules
    )


def test_packaged_company_type_rules_canonical_format_guard():
    rules = _load_company_type_rules()
    invalid = [
        (source, canonical)
        for source, canonical, _country in rules
        if not CANONICAL_COMPANY_TYPE_PATTERN.fullmatch(canonical)
    ]
    assert not invalid


def test_canonical_company_type_pattern_allows_ampersand_separator_token():
    assert CANONICAL_COMPANY_TYPE_PATTERN.fullmatch("GmbH & Co KG")
    assert not CANONICAL_COMPANY_TYPE_PATTERN.fullmatch("GmbH &")


def test_get_company_type_rules_returns_regex_and_copy_of_mapping():
    regex, mapping = get_company_type_rules()

    assert regex
    assert mapping["limited"] == "ltd"

    mapping["limited"] = "changed"
    _, fresh_mapping = get_company_type_rules()
    assert fresh_mapping["limited"] == "ltd"


def test_get_company_type_rules_includes_new_german_variant_mappings():
    _regex, mapping = get_company_type_rules()

    assert mapping["mbh"] == "gmbh"
    assert mapping["partgmbb"] == "partg"


def test_suffix_trie_matcher_prefers_longest_match_on_clean_surface():
    custom_mapping = {
        "KG": "KG",
        "CO KG": "CO KG",
        "GMBH CO KG": "GMBH CO KG",
    }
    trie, max_tokens = _build_company_type_suffix_trie(custom_mapping)

    assert (
        _match_company_type_from_suffix_surface(
            "ALPHA BETA GMBH CO KG", trie, max_tokens
        )
        == "GMBH CO KG"
    )
    assert (
        _match_company_type_from_suffix_surface("ALPHA BETA CO KG", trie, max_tokens)
        == "CO KG"
    )
    assert (
        _match_company_type_from_suffix_surface("ALPHA BETA KG", trie, max_tokens)
        == "KG"
    )


def test_suffix_trie_matcher_matches_regex_on_clean_surfaces():
    regex, mapping = get_company_type_rules()
    trie, max_tokens = _build_company_type_suffix_trie(mapping)

    clean_surfaces = [
        "acme limited",
        "beta services ltd",
        "gamma holdings plc",
        "alpha beta kg",
        "alpha beta co kg",
        "theta gmbh co kg",
        "omega limited liability partnership",
        "xi community interest company",
    ]

    regex_matches = []
    for surface in clean_surfaces:
        match = re.search(regex, surface, re.IGNORECASE)
        regex_matches.append(match.group(1) if match else None)

    trie_matches = [
        _match_company_type_from_suffix_surface(surface, trie, max_tokens)
        for surface in clean_surfaces
    ]

    assert trie_matches == regex_matches


def test_build_company_type_mapping_rejects_unknown_canonical_value():
    with pytest.raises(ValueError, match="Unknown canonical company type"):
        _build_company_type_mapping(
            [("Limited", "NotARealCanonical")],
            {"Ltd"},
        )


def test_build_company_type_mapping_rejects_conflicting_rules():
    with pytest.raises(ValueError, match="Conflicting company type mappings detected"):
        _build_company_type_mapping(
            [("Limited", "Ltd"), ("Limited", "PLC")],
            {"Ltd", "PLC"},
        )


def test_build_company_type_suffix_trie_ignores_empty_variant_keys():
    trie, max_tokens = _build_company_type_suffix_trie({"": "IGNORED", "LTD": "LTD"})
    assert max_tokens == 1
    assert (
        _match_company_type_from_suffix_surface("ACME LTD", trie, max_tokens) == "LTD"
    )


def test_suffix_trie_matcher_returns_none_for_empty_or_non_token_surface():
    trie, max_tokens = _build_company_type_suffix_trie({"ltd": "ltd"})
    assert _match_company_type_from_suffix_surface("   ", trie, max_tokens) is None
    assert _match_company_type_from_suffix_surface("ACME LTD", trie, 0) is None


def test_get_company_type_rules_for_country_is_scoped_to_that_country():
    _regex, gb_mapping = get_company_type_rules_for_country("gb")

    assert gb_mapping
    assert gb_mapping.get("limited") == "ltd"
    assert gb_mapping.get("plc") == "plc"
    # Australia-only ISO 20275 classification labels must not leak into a gb-scoped mapping.
    assert "public company limited by guarantee" not in gb_mapping
    assert "public company limited by shares" not in gb_mapping
    # Nor should another country's canonicals gb doesn't have.
    assert "gmbh" not in gb_mapping.values()


def test_get_company_type_rules_for_country_raises_for_unknown_country():
    with pytest.raises(UnknownCompanyTypeCountryError, match="'zz'"):
        get_company_type_rules_for_country("zz")


def test_get_company_type_rules_for_country_scoped_mapping_is_a_copy():
    _regex, mapping = get_company_type_rules_for_country("gb")
    mapping["limited"] = "changed"

    _regex, fresh_mapping = get_company_type_rules_for_country("gb")
    assert fresh_mapping["limited"] == "ltd"
