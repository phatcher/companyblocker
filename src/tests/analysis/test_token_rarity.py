from __future__ import annotations

import polars as pl
import pytest

from analysis.token_rarity import (
    attach_language_frequency,
    resolve_system_language,
    summarize_language_rarity,
)


def test_resolve_system_language_known_and_unknown_systems():
    assert resolve_system_language("gb") == "en"
    assert resolve_system_language("FR") == "fr"
    assert resolve_system_language("global") is None
    assert resolve_system_language("unknown") is None


def test_attach_language_frequency_flags_unseen_tokens():
    token_stats = pl.DataFrame(
        {
            "token": ["the", "zzqxplkxxnope"],
            "document_frequency": [5, 1],
            "document_frequency_pct": [0.5, 0.1],
            "idf": [1.0, 3.0],
        }
    )

    result = attach_language_frequency(token_stats, language="en")

    assert result.columns == [
        "token",
        "document_frequency",
        "document_frequency_pct",
        "idf",
        "zipf_frequency",
        "unseen_in_wordfreq",
    ]
    the_row = result.filter(pl.col("token") == "the")
    assert the_row.get_column("zipf_frequency").item() > 0.0
    assert the_row.get_column("unseen_in_wordfreq").item() is False

    nonsense_row = result.filter(pl.col("token") == "zzqxplkxxnope")
    assert nonsense_row.get_column("zipf_frequency").item() == 0.0
    assert nonsense_row.get_column("unseen_in_wordfreq").item() is True


def test_attach_language_frequency_requires_token_column():
    with pytest.raises(ValueError, match="missing required column"):
        attach_language_frequency(pl.DataFrame({"other": ["x"]}), language="en")


def test_summarize_language_rarity_reports_full_and_rare_subsets():
    token_stats = pl.DataFrame(
        {
            "token": ["the", "company", "zzqxplkxxnope"],
            "document_frequency": [5, 3, 1],
            "document_frequency_pct": [0.5, 0.3, 0.1],
            "idf": [1.0, 1.5, 3.0],
        }
    )
    stats_with_zipf = attach_language_frequency(token_stats, language="en")

    summary = summarize_language_rarity(
        stats_with_zipf,
        system="gb",
        language="en",
        rare_max_document_frequency=1,
        rare_max_document_frequency_pct=None,
    )

    assert summary["system"] == "gb"
    assert summary["language"] == "en"
    assert summary["total_tokens"] == 3
    assert summary["rare_tokens"] == 1
    assert summary["rare_unseen_in_wordfreq_pct"] == pytest.approx(1.0)
    assert 0.0 <= summary["unseen_in_wordfreq_pct"] <= 1.0


def test_summarize_language_rarity_handles_empty_frame():
    empty = pl.DataFrame(
        schema={
            "token": pl.Utf8,
            "document_frequency": pl.Int64,
            "document_frequency_pct": pl.Float64,
            "idf": pl.Float64,
            "zipf_frequency": pl.Float64,
            "unseen_in_wordfreq": pl.Boolean,
        }
    )

    summary = summarize_language_rarity(
        empty,
        system="gb",
        language="en",
        rare_max_document_frequency=1,
        rare_max_document_frequency_pct=None,
    )

    assert summary["total_tokens"] == 0
    assert summary["rare_tokens"] == 0


def test_summarize_language_rarity_validates_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        summarize_language_rarity(
            pl.DataFrame({"token": ["a"]}),
            system="gb",
            language="en",
            rare_max_document_frequency=1,
            rare_max_document_frequency_pct=None,
        )
