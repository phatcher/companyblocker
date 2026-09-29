import pytest
from company_cleanse.normalize import (
    DEFAULT_NORMALIZATION_OPERATIONS,
    NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS,
    SUPPORTED_NORMALIZATION_OPERATIONS,
    geographic_tiers_for_operation,
    normalize_operations,
    normalize_tokens,
    parse_normalization_profile,
)


def test_the_operation_is_supported_but_off_by_default():
    assert (
        NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS in SUPPORTED_NORMALIZATION_OPERATIONS
    )
    assert (
        NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS not in DEFAULT_NORMALIZATION_OPERATIONS
    )
    assert NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS not in parse_normalization_profile(
        "default"
    )


def test_the_default_chain_leaves_a_geographic_term_alone():
    assert normalize_tokens("Acme Systems Ireland") == "acme systems ireland"
    assert normalize_tokens("Siemens (UK) Ltd") == "siemens uk ltd"


@pytest.mark.parametrize("profile", ["default|geographic", "default|geographic_terms"])
def test_the_alias_and_the_canonical_spelling_resolve_identically(profile):
    operations = parse_normalization_profile(profile)

    assert operations[0] == NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS
    assert operations == parse_normalization_profile("default|geographic_terms")


def test_the_operation_runs_ahead_of_punctuation_so_brackets_are_still_visible():
    operations = parse_normalization_profile("default|geographic")

    assert operations[0] == NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS
    assert operations[1] == "punctuation"
    # Once punctuation has run, `(UK)` is just `uk` in trailing position behind
    # `ltd`, and the bracketed arm has nothing left to recognize.
    assert (
        normalize_tokens("Siemens (UK) Ltd", normalization_profile="default|geographic")
        == "siemens ltd"
    )
    assert (
        normalize_tokens(
            "Siemens (UK) Ltd",
            normalization_profile="punctuation|singlespace|geographic|lowercase",
        )
        == "siemens uk ltd"
    )


def test_tier_modifiers_are_folded_into_the_operation_in_tier_order():
    assert parse_normalization_profile("default|geographic|+region")[0] == (
        "geographic_terms+region"
    )
    assert parse_normalization_profile("default|geographic|+city|+region")[0] == (
        "geographic_terms+region+city"
    )
    # `country` is the default tier, so naming it changes nothing.
    assert parse_normalization_profile("default|geographic|+country")[0] == (
        NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS
    )


def test_tiers_read_back_off_a_resolved_operation():
    assert geographic_tiers_for_operation("geographic_terms") == ("country",)
    assert geographic_tiers_for_operation("geographic_terms+region") == (
        "country",
        "region",
    )
    assert geographic_tiers_for_operation("geographic_terms+region+city") == (
        "country",
        "region",
        "city",
    )


@pytest.mark.parametrize(
    ("profile", "expected_message"),
    [
        ("default|+region", "without selecting"),
        ("default|punctuation|+region", "without selecting"),
        ("punctuation|lowercase|+city", "without selecting"),
        ("default|+", "empty"),
        ("default|geographic|+moon", "Unknown geographic term kind"),
    ],
)
def test_a_tier_modifier_without_the_operation_is_rejected(profile, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        parse_normalization_profile(profile)


def test_a_tier_modifier_on_another_operation_is_rejected():
    with pytest.raises(ValueError, match="only 'geographic_terms' supports"):
        normalize_operations(["punctuation+region"])


def test_directly_supplied_operations_canonicalize_the_same_way():
    assert normalize_operations(["geographic", "lowercase"]) == (
        "geographic_terms",
        "lowercase",
    )
    assert normalize_operations(["geographic_terms+city"]) == ("geographic_terms+city",)


@pytest.mark.parametrize(
    ("profile", "name", "expected"),
    [
        ("default|geographic", "Siemens (UK) Ltd", "siemens ltd"),
        ("default|geographic", "Acme Systems Ireland", "acme systems"),
        ("default|geographic", "Acme Systems Alberta", "acme systems alberta"),
        ("default|geographic|+region", "Acme Systems Alberta", "acme systems"),
        # `+city` is wired but strips nothing until city data exists.
        ("default|geographic|+city", "Acme Systems London", "acme systems london"),
        # The names the positional rule protects survive on every profile.
        ("default|geographic|+region|+city", "Air France", "air france"),
        ("default|geographic|+region|+city", "Bank of Ireland", "bank of ireland"),
        ("default|geographic|+region|+city", "China Mobile", "china mobile"),
        # The ordering constraint: a legal form still holds the trailing position.
        ("default|geographic", "Oracle Ireland Ltd", "oracle ireland ltd"),
    ],
)
def test_end_to_end_normalization_through_a_profile(profile, name, expected):
    assert normalize_tokens(name, normalization_profile=profile) == expected


def test_tiers_are_part_of_the_normalization_cache_key():
    # Two profiles differing only by tier must not collide in the operation cache.
    name = "Acme Systems Alberta"

    assert normalize_tokens(name, normalization_profile="default|geographic") == (
        "acme systems alberta"
    )
    assert (
        normalize_tokens(name, normalization_profile="default|geographic|+region")
        == "acme systems"
    )
    assert normalize_tokens(name, normalization_profile="default|geographic") == (
        "acme systems alberta"
    )
