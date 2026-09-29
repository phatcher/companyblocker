from __future__ import annotations

from acquisition.country_counts import coerce_country_counts


def test_coerce_country_counts_normalizes_keys_and_coerces_values():
    raw = {" GB ": "3", "FR": 2, "de": 1.9}

    assert coerce_country_counts(raw) == {"gb": 3, "fr": 2, "de": 1}


def test_coerce_country_counts_drops_blank_keys_and_uncoercible_values():
    raw = {"": 5, "   ": 4, "gb": "not-a-number", "fr": None, "ie": "7"}

    assert coerce_country_counts(raw) == {"ie": 7}


def test_coerce_country_counts_returns_empty_dict_for_non_dict_input():
    assert coerce_country_counts(["gb", "fr"]) == {}
    assert coerce_country_counts(None) == {}
    assert coerce_country_counts("gb") == {}


def test_coerce_country_counts_coerces_non_string_keys():
    assert coerce_country_counts({1: "2", 2.5: "3"}) == {"1": 2, "2.5": 3}
