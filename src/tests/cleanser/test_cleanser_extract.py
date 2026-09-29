import pytest
from company_cleanse.extract import (
    _derive_acronym,
    _derive_lead_token_decisions,
    _ensure_non_acronym_short_name_in_cleansed,
    _ensure_quoted_name_in_cleansed,
    _extract_leading_delimited_tokens,
    _extract_leading_quoted_tokens,
    _extract_leading_token_struct,
    _extract_quoted_type_parenthesized_parts,
    _initials_excluding_company_type,
    _is_acronym_like_token,
    _is_valid_short_name_token,
    _recombine_quoted_name_short_name,
    _replace_company_type_with_canonical,
    _trim_quoted_name_company_type_suffix,
)
from company_cleanse.rules import COMPANY_TYPE_MAPPING


@pytest.mark.parametrize(
    "value,expected",
    [
        ("(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD", "misg"),
        ("(A C R) UK LIMITED", "acr"),
        ("(A.C_R.) UK LIMITED", "acr"),
        ("MISG UK LIMITED", None),
        ("(!!!) UK LIMITED", ""),
        ('"FELDA EUROPE"', None),
        ('"FELDA EUROPE" S.R.L.', "felda europe"),
    ],
)
def test_extract_leading_delimited_tokens_cases(value, expected):
    assert _extract_leading_delimited_tokens(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ('"ABC" SERVICES LTD', "abc"),
        ("'TRIPLE D' SERVICES LTD", "triple d"),
        ('"FELDA EUROPE"', None),
        ("MISG UK LIMITED", None),
        (None, None),
    ],
)
def test_extract_leading_quoted_tokens_cases(value, expected):
    assert _extract_leading_quoted_tokens(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, {"lead_token": None, "lead_quoted_token": None}),
        (
            "(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD",
            {"lead_token": "misg", "lead_quoted_token": None},
        ),
        (
            '"TRIPLE D" SERVICES LTD',
            {"lead_token": "triple d", "lead_quoted_token": "triple d"},
        ),
        (
            "MEDICAL INDUSTRY SUPPORT GROUP LTD",
            {"lead_token": None, "lead_quoted_token": None},
        ),
    ],
)
def test_extract_leading_token_struct_cases(value, expected):
    assert _extract_leading_token_struct(value) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("MISG", True),
        ("A1", True),
        ("A B C", True),
        ("TRIPLE D", False),
        ("A", False),
        ("!!!", False),
        (None, False),
    ],
)
def test_is_acronym_like_token_cases(value, expected):
    assert _is_acronym_like_token(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("MISG", True),
        ("A B", True),
        ("A", False),
        ("!!!", False),
        (None, False),
    ],
)
def test_is_valid_short_name_token_cases(value, expected):
    assert _is_valid_short_name_token(value) is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        (
            '"VARIG" S.A. (VIACAO AEREA RIO GRANDENSE)',
            ("varig", "sa", "VIACAO AEREA RIO GRANDENSE"),
        ),
        (
            '"VARIG" (VIACAO AEREA RIO GRANDENSE)',
            ("varig", None, "VIACAO AEREA RIO GRANDENSE"),
        ),
        (
            "V.A.FLETCHER (OLD PARK HOUSE FARM PARTNERSHIP)",
            (None, None, None),
        ),
        (
            '"VARIG" XYZTYPE (VIACAO AEREA RIO GRANDENSE)',
            (None, None, None),
        ),
    ],
)
def test_extract_quoted_type_parenthesized_parts_cases(value, expected):
    acronym, canonical_type, full_name = _extract_quoted_type_parenthesized_parts(
        value,
        COMPANY_TYPE_MAPPING,
    )
    assert (acronym, canonical_type, full_name) == expected


@pytest.mark.parametrize(
    "stripped_name,special_short_name,expected_short,expected_quoted",
    [
        ("(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD", None, "misg", None),
        ('"TRIPLE D" SERVICES LTD', None, None, "triple d"),
        ('"K" LINE SHIPPING LTD', None, None, "k"),
        ('"A B C" SERVICES LTD', None, "abc", "abc"),
        ('"ABC" SERVICES LTD', None, "abc", "abc"),
        ('"FELDA EUROPE"', None, None, None),
        ("(THE) MIDDLE EAST TIMES", None, None, None),
        ("(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD", "OVERRIDE", "OVERRIDE", None),
        (None, None, None, None),
    ],
)
def test_derive_lead_token_decisions_cases(
    stripped_name, special_short_name, expected_short, expected_quoted
):
    result = _derive_lead_token_decisions(stripped_name, special_short_name)
    assert result["short_name"] == expected_short
    assert result["quoted_name"] == expected_quoted


@pytest.mark.parametrize(
    "normalized_name,extracted_company_type,canonical_company_type,expected",
    [
        (None, "LIMITED", "LTD", None),
        ("ACME LIMITED", "LIMITED", None, "ACME LIMITED"),
        ("ACME LIMITED", "LIMITED", "private", "ACME LIMITED"),
        ("ACME ENTITY", "ENTITY", "OVERSEAS ENTITY", "ACME ENTITY"),
        ("ACME", None, "LTD", "ACME LTD"),
        ("ACME LIMITED", "LIMITED", "LTD", "ACME LTD"),
        (
            "ACME PUBLIC LIMITED COMPANY",
            "PUBLIC   LIMITED COMPANY",
            "PLC",
            "ACME PUBLIC LIMITED COMPANY PLC",
        ),
    ],
)
def test_replace_company_type_with_canonical_cases(
    normalized_name,
    extracted_company_type,
    canonical_company_type,
    expected,
):
    assert (
        _replace_company_type_with_canonical(
            normalized_name,
            extracted_company_type,
            canonical_company_type,
        )
        == expected
    )


@pytest.mark.parametrize(
    "companyname,company_type,expected",
    [
        (None, "LTD", None),
        ("", "LTD", None),
        ("MEDICAL INDUSTRY SUPPORT GROUP LTD", "LTD", "misg"),
        ("THETA HOLDINGS SP ZOO", "SP ZOO", "th"),
        ("ACME PRIVATE", "private", "ap"),
        ("ACME HOLDINGS LIMITED", "PLC", "ahl"),
    ],
)
def test_initials_excluding_company_type_cases(companyname, company_type, expected):
    assert _initials_excluding_company_type(companyname, company_type) == expected


@pytest.mark.parametrize(
    "short_name,companyname,company_type,expected",
    [
        ("MISG", "MEDICAL INDUSTRY SUPPORT GROUP LTD", "LTD", "misg"),
        ("MIG", "MEDICAL INDUSTRY SUPPORT GROUP LTD", "LTD", None),
        (None, "MEDICAL INDUSTRY SUPPORT GROUP LTD", "LTD", None),
        ("TH", "THETA HOLDINGS SP ZOO", "SP ZOO", "th"),
        ("misg", "medical industry support group ltd", "ltd", "misg"),
    ],
)
def test_derive_acronym_cases(short_name, companyname, company_type, expected):
    assert _derive_acronym(short_name, companyname, company_type) == expected


@pytest.mark.parametrize(
    "short_name,acronym,cleansed_name,expected",
    [
        # Domain note: company-type tokens are not meaningful short names.
        # The current helper still prefixes them when no acronym is present;
        # keep one implementation-facing case until that behavior is changed.
        pytest.param(
            None, None, "alpha care ltd", "alpha care ltd", id="no-short-name-no-change"
        ),
        pytest.param(
            None,
            "MISG",
            "medical industry support group ltd",
            "medical industry support group ltd",
            id="acronym-like-but-not-company-type-no-prefix",
        ),
        pytest.param(
            "LTD",
            "LTD",
            "alpha care ltd",
            "alpha care ltd",
            id="short-name-blocked-when-acronym-present",
        ),
        pytest.param(
            "LTD",
            None,
            "alpha care ltd",
            "ltd alpha care ltd",
            id="implementation-note-company-type-prefix-added",
        ),
        pytest.param("LTD", None, None, None, id="no-cleansed-name"),
    ],
)
def test_ensure_non_acronym_short_name_in_cleansed_cases(
    short_name, acronym, cleansed_name, expected
):
    assert (
        _ensure_non_acronym_short_name_in_cleansed(short_name, acronym, cleansed_name)
        == expected
    )


@pytest.mark.parametrize(
    "quoted_name,acronym,cleansed_name,expected",
    [
        (None, None, "SERVICES LTD", "SERVICES LTD"),
        ("", None, "SERVICES LTD", "SERVICES LTD"),
        ("ABC", "ABC", "SERVICES LTD", "SERVICES LTD"),
        ("ABC", None, "SERVICES LTD", "abc SERVICES LTD"),
        ("TRIPLE D", None, "TRIPLE D SERVICES LTD", "TRIPLE D SERVICES LTD"),
        ("TRIPLE D", None, "SERVICES LTD", "triple d SERVICES LTD"),
        ("TRIPLE D", None, None, None),
    ],
)
def test_ensure_quoted_name_in_cleansed_cases(
    quoted_name, acronym, cleansed_name, expected
):
    assert (
        _ensure_quoted_name_in_cleansed(quoted_name, acronym, cleansed_name) == expected
    )


@pytest.mark.parametrize(
    "quoted_name,company_type,expected",
    [
        ("abc limited", "ltd", "abc"),
        ("abc ltd", "ltd", "abc"),
        ("abc", "ltd", "abc"),
        ("ltd", "ltd", None),
        (None, "ltd", None),
    ],
)
def test_trim_quoted_name_company_type_suffix_cases(
    quoted_name, company_type, expected
):
    assert (
        _trim_quoted_name_company_type_suffix(
            quoted_name, company_type, COMPANY_TYPE_MAPPING
        )
        == expected
    )


@pytest.mark.parametrize(
    "quoted_name,short_name,company_type,cleansed_name,expected",
    [
        (
            "triple d",
            "properties",
            "ltd",
            "triple d properties ltd",
            "triple d properties",
        ),
        (
            "belle vue",
            "enterprises",
            "ltd",
            "belle vue enterprises ltd",
            "belle vue enterprises",
        ),
        (
            "1st rate",
            "1st rate psychology",
            "ltd",
            "1st rate psychology services ltd",
            "1st rate psychology",
        ),
        (
            "beechbank court",
            "beechbank court",
            "ltd",
            "beechbank court management company ltd",
            "beechbank court",
        ),
    ],
)
def test_recombine_quoted_name_short_name_cases(
    quoted_name, short_name, company_type, cleansed_name, expected
):
    assert (
        _recombine_quoted_name_short_name(
            quoted_name, short_name, company_type, cleansed_name
        )
        == expected
    )
