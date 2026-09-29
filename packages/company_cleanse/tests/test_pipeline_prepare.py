import polars as pl
from company_cleanse.pipeline import (
    _step_build_name_body,
    _step_derive_lead_token_fields,
    _step_extract_special_parts,
    _step_prepare_stripped_name,
)
from company_cleanse.rules import COMPANY_TYPE_MAPPING
from polars.testing import assert_frame_equal


def test_step_prepare_stripped_name_trims_without_eager_lowercasing():
    source = pl.DataFrame(
        {
            "CompanyName": ['  "Abc" ltd  ', "  (the) Middle East Times ltd  "],
        }
    )

    actual = (
        _step_prepare_stripped_name(source.lazy(), "CompanyName")
        .select(["_stripped_name", "_stripped_name_pre_owner"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "_stripped_name": ['"Abc" ltd', "(the) Middle East Times ltd"],
            "_stripped_name_pre_owner": ['"Abc" ltd', "(the) Middle East Times ltd"],
        }
    )

    assert_frame_equal(actual, expected)


def test_step_extract_special_parts_from_quoted_parenthesized_pattern():
    source = pl.DataFrame(
        {
            "_stripped_name": ['"VARIG" S.A. (VIACAO AEREA RIO GRANDENSE)'],
        }
    )

    actual = (
        _step_extract_special_parts(source.lazy(), COMPANY_TYPE_MAPPING)
        .select(["_special_short_name", "_special_company_type", "_special_full_name"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "_special_short_name": ["varig"],
            "_special_company_type": ["sa"],
            "_special_full_name": ["VIACAO AEREA RIO GRANDENSE"],
        }
    )

    assert_frame_equal(actual, expected)


def test_step_derive_lead_token_fields_projects_semantic_columns():
    source = pl.DataFrame(
        {
            "_stripped_name": [
                "(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD",
                '"TRIPLE D" SERVICES LTD',
            ],
            "_special_short_name": [None, None],
        }
    )

    actual = (
        _step_derive_lead_token_fields(source.lazy())
        .select(["_lead_token", "_lead_quoted_token", "short_name", "quoted_name"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "_lead_token": ["misg", "triple d"],
            "_lead_quoted_token": [None, "triple d"],
            "short_name": ["misg", None],
            "quoted_name": [None, "triple d"],
        }
    )

    assert_frame_equal(actual, expected)


def test_step_build_name_body_handles_the_and_special_full_name():
    source = pl.DataFrame(
        {
            "_special_full_name": [None, None, "OVERRIDE FULL NAME"],
            "_lead_token": ["the", "abc", None],
            "_lead_quoted_token_valid": [False, False, False],
            "_stripped_name": [
                "(the) middle east times ltd",
                '"abc" services ltd',
                "IGNORED INPUT",
            ],
        }
    )

    actual = (
        _step_build_name_body(source.lazy())
        .select(["_name_body", "_name_body_reordered"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "_name_body": [
                "the middle east times ltd",
                '"abc" services ltd',
                "OVERRIDE FULL NAME",
            ],
            "_name_body_reordered": [
                "the middle east times ltd",
                '"abc" services ltd',
                "OVERRIDE FULL NAME",
            ],
        }
    )

    assert_frame_equal(actual, expected)


def test_step_build_name_body_reorders_trailing_the_parenthetical():
    # UK Companies House convention: a name filed as "X (The)" is displayed as
    # "The X" -- this is a distinct code path from a *leading* "(the)", which
    # test_step_build_name_body_handles_the_and_special_full_name already covers.
    source = pl.DataFrame(
        {
            "_special_full_name": pl.Series([None, None, None], dtype=pl.Utf8),
            "_lead_token": pl.Series([None, None, None], dtype=pl.Utf8),
            "_lead_quoted_token_valid": [False, False, False],
            "_stripped_name": [
                "Filbert Ltd (The)",
                "Filbert Ltd (the)",
                "Filbert Ltd (Not The)",
            ],
        }
    )

    actual = (
        _step_build_name_body(source.lazy())
        .select(["_name_body", "_name_body_reordered"])
        .collect()
    )

    expected = pl.DataFrame(
        {
            "_name_body": [
                "Filbert Ltd (The)",
                "Filbert Ltd (the)",
                "Filbert Ltd (Not The)",
            ],
            "_name_body_reordered": [
                "the Filbert Ltd",
                "the Filbert Ltd",
                # Only a trailing parenthetical containing exactly "the" reorders --
                # anything else in the parens is left as ordinary trailing text.
                "Filbert Ltd (Not The)",
            ],
        }
    )

    assert_frame_equal(actual, expected)
