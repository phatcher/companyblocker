from __future__ import annotations

import re

import pytest

from scripts import (
    acquire_companies,
    generate_noise_words,
    process_companies,
    train_tokenizer,
)


def _normalized_help(parser_builder) -> str:
    help_text = " ".join(parser_builder().format_help().split())
    return re.sub(r"(?<=\w)- (?=\w)", "-", help_text)


@pytest.mark.parametrize(
    ("parser_builder", "expected_fragment"),
    [
        (
            acquire_companies.build_parser,
            "One or more system codes. Countries are valid system codes (for example 'gb'), and non-country systems are handled the same way. Use 'all' to target every system the catalog flags with all_systems_target. Comma-separated values are accepted.",
        ),
        (
            process_companies.build_parser,
            "One or more system codes to process. Countries are valid system codes (for example 'gb'), and non-country systems are handled the same way. Use 'all' to target every system the catalog flags with all_systems_target. Comma-separated values are accepted.",
        ),
        (
            train_tokenizer.build_parser,
            "Use explicit system codes (for example 'fr gb ie') for per-system country training. Use '--systems global' to train a single global tokenizer across every system in '--systems all' with cleansed data. Use mixed values (for example 'fr ie global') to train country tokenizers and then a global tokenizer in one pass.",
        ),
        (
            generate_noise_words.build_parser,
            "System code to target (for example 'ie' or 'global').",
        ),
    ],
)
def test_script_help_includes_common_systems_text(parser_builder, expected_fragment):
    help_text = _normalized_help(parser_builder)
    assert expected_fragment in help_text


@pytest.mark.parametrize(
    ("parser_builder", "expected_fragment"),
    [
        (
            acquire_companies.build_parser,
            "Run date folder in YYYY-MM-DD format. Defaults to today's date.",
        ),
        (
            process_companies.build_parser,
            "Acquisition/canonical run date in YYYY-MM-DD format. If omitted, latest available snapshots are used where supported.",
        ),
    ],
)
def test_script_help_includes_run_date_text(parser_builder, expected_fragment):
    help_text = _normalized_help(parser_builder)
    assert expected_fragment in help_text


@pytest.mark.parametrize(
    "parser_builder", [acquire_companies.build_parser, process_companies.build_parser]
)
def test_script_help_includes_supported_codes_epilog(parser_builder):
    help_text = _normalized_help(parser_builder)
    assert "Country codes in --systems all: fr, gb, ie" in help_text
    assert (
        "Non-country systems in --systems all: gleif, offeneregister, wikidata"
        in help_text
    )
