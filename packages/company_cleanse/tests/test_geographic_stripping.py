import pytest
from company_cleanse.geographic_terms import strip_geographic_terms


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Siemens (UK)", "Siemens"),
        ("Siemens [UK]", "Siemens"),
        ("Siemens (UK) Ltd", "Siemens Ltd"),
        ("Acme Systems (Ireland) Ltd", "Acme Systems Ltd"),
        ("Acme Systems Ireland", "Acme Systems"),
        ("Acme Systems, Ireland", "Acme Systems"),
        ("Acme Holdings Deutschland", "Acme Holdings"),
        ("Acme Trading United Arab Emirates", "Acme Trading"),
    ],
)
def test_trailing_and_bracketed_terms_are_stripped(name, expected):
    assert strip_geographic_terms(name) == expected


@pytest.mark.parametrize(
    "name",
    [
        # The four names the positional rule exists to protect.
        "Air France",
        "Bank of Ireland",
        "Bank of America",
        "China Mobile",
        # A geographic term in the middle is never a trailing marker.
        "Ireland Systems Group",
        # Stripping must never consume the whole name.
        "Ireland",
        "France",
        # A punctuation-only preceding token is as much a fragment as a connector.
        "Acme & Ireland",
    ],
)
def test_an_integral_geographic_term_survives(name):
    assert strip_geographic_terms(name) == name


def test_a_legal_form_in_the_trailing_position_blocks_the_trailing_arm():
    # The documented ordering constraint: `Ltd` occupies the trailing position, so
    # this only resolves once legal-form stripping has run (the short-name stage).
    assert strip_geographic_terms("Oracle Ireland Ltd") == "Oracle Ireland Ltd"
    assert strip_geographic_terms("Oracle Ireland Ltd", min_remaining_tokens=1) == (
        "Oracle Ireland Ltd"
    )
    assert strip_geographic_terms("Oracle Ireland", min_remaining_tokens=1) == "Oracle"


def test_min_remaining_tokens_is_what_keeps_a_two_token_name_intact():
    # `Air France` and `Oracle Ireland` are structurally identical; only a caller
    # that has already removed a legal form may lower the bar to one token.
    assert strip_geographic_terms("Air France") == "Air France"
    assert strip_geographic_terms("Air France", min_remaining_tokens=1) == "Air"


@pytest.mark.parametrize(
    "name",
    [
        "Bank of New England",
        "Bank of New South Wales",
        "Acme New England",
    ],
)
def test_a_preceding_place_qualifier_refuses_the_strip(name):
    assert strip_geographic_terms(name, kinds=("region",)) == name


def test_multi_word_terms_are_matched_longest_first():
    # `New York` is consumed whole rather than read as `York` behind a `New`.
    assert strip_geographic_terms("Acme Systems New York", kinds=("region",)) == (
        "Acme Systems"
    )
    assert strip_geographic_terms(
        "Acme Systems British Columbia", kinds=("region",)
    ) == ("Acme Systems")


def test_tiers_are_additive_and_country_is_the_default():
    name = "Acme Systems Alberta"

    assert strip_geographic_terms(name) == name
    assert strip_geographic_terms(name, kinds=("country",)) == name
    assert strip_geographic_terms(name, kinds=("region",)) == "Acme Systems"
    assert strip_geographic_terms(name, kinds=("region", "city")) == "Acme Systems"

    # Selecting a further tier never turns the country tier off.
    assert strip_geographic_terms("Acme Systems Ireland", kinds=("region",)) == (
        "Acme Systems"
    )


def test_the_city_tier_strips_nothing_until_city_data_exists():
    assert strip_geographic_terms("Acme Systems London", kinds=("city",)) == (
        "Acme Systems London"
    )
    assert strip_geographic_terms("Acme Systems Berlin", kinds=("city",)) == (
        "Acme Systems Berlin"
    )


def test_matching_is_diacritic_and_case_insensitive():
    assert strip_geographic_terms("Acme Holdings TÜRKIYE") == "Acme Holdings"
    assert strip_geographic_terms("Acme Holdings türkiye") == "Acme Holdings"
    assert strip_geographic_terms("Acme Holdings deutschland") == "Acme Holdings"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_empty_input_is_returned_unchanged(value):
    assert strip_geographic_terms(value) == value


def test_a_wholly_bracketed_name_is_left_alone():
    assert strip_geographic_terms("(Ireland)") == "(Ireland)"
