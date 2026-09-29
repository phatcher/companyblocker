"""Direct tests for `company_tokenize.name_preprocessing`.

What is pinned is what a caller relies on: the name column is left alone, the
new column holds the text a tokenizer or vectorizer reads, the profile string
says which parts run, and a profile that spells something out wrongly is
refused.
"""

from __future__ import annotations

import polars as pl
import pytest
from company_tokenize.name_preprocessing import (
    NAME_PREPROCESSED_COLUMN,
    NamePreprocessing,
    name_preprocessing,
    parse_name_preprocessing_profile,
)

_NAMES = [
    "Acme Holdings Limited",
    "Murphy & Sons Teoranta",
    "Acme Holdings Limited",
    None,
]


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"name": _NAMES}, schema={"name": pl.Utf8})


def test_the_default_removes_the_company_type_and_keeps_noise_words() -> None:
    out = name_preprocessing(_frame(), name_col="name")

    assert out.get_column("name").to_list() == _NAMES
    assert out.get_column(NAME_PREPROCESSED_COLUMN).to_list() == [
        "acme holdings",
        "murphy & sons",
        "acme holdings",
        None,
    ]


def test_noise_words_are_removed_only_when_the_profile_names_a_level() -> None:
    out = name_preprocessing(
        _frame(), name_col="name", profile="default|+noise_words:aggressive"
    )

    assert out.get_column(NAME_PREPROCESSED_COLUMN).to_list()[0] == "acme"


def test_no_profile_opts_out_of_the_form_a_tokenizer_was_trained_on() -> None:
    """A run may keep capitals upstream or score the raw name; a tokenizer
    handed that answers in unknown tokens. Removing nothing still normalises."""
    frame = pl.DataFrame({"name": ["Müller & Söhne GmbH", "ACME, Holdings Ltd."]})

    out = name_preprocessing(frame, name_col="name", profile="default|-company_type")

    processed = out.get_column(NAME_PREPROCESSED_COLUMN).to_list()
    assert processed[0].startswith("muller") and processed[0].endswith("gmbh")
    assert processed[1] == "acme holdings ltd"
    assert out.get_column("name").to_list() == frame.get_column("name").to_list()


def test_a_profile_is_read_into_the_parts_it_asks_for() -> None:
    assert parse_name_preprocessing_profile("default") == NamePreprocessing()
    assert parse_name_preprocessing_profile(
        "default | -company_type | +noise_words:strict"
    ) == NamePreprocessing(remove_company_type=False, noise_words_level="strict")


@pytest.mark.parametrize(
    "profile",
    [
        "",
        "-company_type",
        "default|+noise_words",
        "default|+noise_words:",
        "default|lowercase",
    ],
)
def test_a_profile_that_does_not_spell_out_what_it_means_is_refused(
    profile: str,
) -> None:
    with pytest.raises(ValueError, match="name preprocessing|names its level"):
        parse_name_preprocessing_profile(profile)
