import polars as pl
import pytest
from company_tokenize.noise_export import (
    DEFAULT_MAX_NOISE_WORD_CANDIDATES,
    build_noise_word_candidates_payload,
    select_noise_word_candidates,
    summarize_noise_word_candidates,
    validate_noise_word_candidates_payload,
)


def _make_stats(rows: list[tuple[str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "token": [row[0] for row in rows],
            "document_frequency_pct": [row[1] for row in rows],
            "idf": [row[2] for row in rows],
        },
        schema={
            "token": pl.Utf8,
            "document_frequency_pct": pl.Float64,
            "idf": pl.Float64,
        },
    )


def test_select_noise_word_candidates_ranks_by_document_frequency_then_idf():
    stats = _make_stats(
        [
            ("ltd", 0.90, 1.1),
            ("services", 0.05, 4.1),
            ("management", 0.03, 4.5),
            ("acme", 0.001, 8.9),
        ]
    )

    selected = select_noise_word_candidates(stats, max_candidates=2, min_token_length=2)

    assert selected.get_column("token").to_list() == ["ltd", "services"]


def test_select_noise_word_candidates_filters_short_tokens():
    stats = _make_stats(
        [
            ("de", 0.5, 3.0),
            ("company", 0.2, 4.0),
        ]
    )

    selected = select_noise_word_candidates(
        stats, max_candidates=10, min_token_length=3
    )

    assert selected.get_column("token").to_list() == ["company"]


def test_select_noise_word_candidates_raises_on_missing_columns():
    stats = pl.DataFrame({"token": ["a"], "idf": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        select_noise_word_candidates(stats)


def test_select_noise_word_candidates_raises_on_invalid_bounds():
    stats = _make_stats([("a", 0.5, 1.0)])
    with pytest.raises(ValueError, match="max_candidates"):
        select_noise_word_candidates(stats, max_candidates=-1)
    with pytest.raises(ValueError, match="min_token_length"):
        select_noise_word_candidates(stats, min_token_length=0)


def test_select_noise_word_candidates_returns_empty_when_max_candidates_zero():
    stats = _make_stats([("ltd", 0.9, 1.1)])
    selected = select_noise_word_candidates(stats, max_candidates=0)
    assert selected.height == 0


def test_default_max_noise_word_candidates_is_two_hundred():
    # Regression guard on the documented, data-derived default -- see the
    # module docstring in company_tokenize.noise_export for the reasoning.
    assert DEFAULT_MAX_NOISE_WORD_CANDIDATES == 200


def test_build_noise_word_candidates_payload_keeps_continuous_score_bounded():
    rows = [(f"token{i}", 1.0 - (i * 0.001), 1.0 + i) for i in range(10)]
    stats = _make_stats(rows)

    payload = build_noise_word_candidates_payload(
        stats,
        system="IE",
        source_stats_path="artifacts/tokenizers/ie/token_tfidf_stats.parquet",
        generated_utc="2026-08-30T00:00:00Z",
        max_candidates=5,
        min_token_length=2,
    )

    assert payload["system"] == "ie"
    assert payload["tfidf_use_case"] == "corpus"
    assert payload["tfidf_namespace"] == "corpus_tfidf"
    assert payload["selection"]["max_candidates"] == 5
    assert payload["selection"]["candidate_count"] == 5
    assert payload["selection"]["source_token_count"] == 10
    assert len(payload["candidates"]) == 5
    # Continuous score preserved, not thresholded to a binary set.
    for candidate in payload["candidates"]:
        assert set(candidate) == {"token", "document_frequency_pct", "idf"}
        assert isinstance(candidate["idf"], float)
        assert isinstance(candidate["document_frequency_pct"], float)


def test_build_noise_word_candidates_payload_handles_empty_stats():
    stats = _make_stats([])
    payload = build_noise_word_candidates_payload(
        stats,
        system="ie",
        source_stats_path="artifacts/tokenizers/ie/token_tfidf_stats.parquet",
        generated_utc="2026-08-30T00:00:00Z",
    )
    assert payload["candidates"] == []
    assert payload["selection"]["candidate_count"] == 0
    assert payload["selection"]["idf_cutoff"] is None
    assert payload["selection"]["document_frequency_pct_cutoff"] is None


def test_validate_noise_word_candidates_payload_accepts_well_formed_payload():
    payload = {
        "system": "ie",
        "candidates": [
            {"token": "ltd", "document_frequency_pct": 0.9, "idf": 1.1},
        ],
    }
    assert validate_noise_word_candidates_payload(payload) is payload


@pytest.mark.parametrize(
    "payload,match",
    [
        ("not a dict", "must be an object"),
        ({}, "'system' string"),
        ({"system": ""}, "'system' string"),
        ({"system": "ie"}, "'candidates' list"),
        ({"system": "ie", "candidates": "not a list"}, "'candidates' list"),
        (
            {"system": "ie", "candidates": ["not a dict"]},
            "candidates\\[0\\] must be an object",
        ),
        (
            {
                "system": "ie",
                "candidates": [{"document_frequency_pct": 0.1, "idf": 1.0}],
            },
            "'token' string",
        ),
        (
            {"system": "ie", "candidates": [{"token": "a", "idf": 1.0}]},
            "'document_frequency_pct'",
        ),
        (
            {
                "system": "ie",
                "candidates": [
                    {"token": "a", "document_frequency_pct": 1.5, "idf": 1.0}
                ],
            },
            "must be between 0 and 1",
        ),
        (
            {
                "system": "ie",
                "candidates": [
                    {"token": "a", "document_frequency_pct": 0.5, "idf": "not numeric"}
                ],
            },
            "'idf'",
        ),
    ],
)
def test_validate_noise_word_candidates_payload_rejects_malformed_payloads(
    payload, match
):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_noise_word_candidates_payload(payload)


def test_summarize_noise_word_candidates_reports_count_and_idf_range():
    payload = {
        "system": "ie",
        "candidates": [
            {"token": "ltd", "document_frequency_pct": 0.9, "idf": 1.1},
            {"token": "services", "document_frequency_pct": 0.05, "idf": 4.1},
        ],
    }
    summary = summarize_noise_word_candidates(payload)
    assert summary == {"candidate_count": 2, "idf_min": 1.1, "idf_max": 4.1}


def test_summarize_noise_word_candidates_handles_missing_candidates():
    summary = summarize_noise_word_candidates({"system": "ie"})
    assert summary == {"candidate_count": 0, "idf_min": None, "idf_max": None}
