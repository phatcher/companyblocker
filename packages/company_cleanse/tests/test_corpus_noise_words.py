import pytest
from company_cleanse.corpus_noise_words import (
    DEFAULT_SHORT_NAME_NOISE_WORD_MAX_TOKENS,
    get_corpus_noise_words,
    validate_short_name_noise_word_candidates_payload,
)


def test_get_corpus_noise_words_selects_jurisdiction_specific_system():
    gb_words = get_corpus_noise_words("gb")
    fr_words = get_corpus_noise_words("fr")

    assert "ltd" in gb_words
    assert gb_words != fr_words


def test_get_corpus_noise_words_maps_country_code_to_promoted_system():
    # "de" has no promoted system of its own -- it maps to the offeneregister
    # (German company registry) system, per the closed jurisdiction table.
    de_words = get_corpus_noise_words("de")
    assert "gmbh" in de_words


@pytest.mark.parametrize("jurisdiction_code", [None, "", "  ", "zz", "es"])
def test_get_corpus_noise_words_falls_back_to_global_for_unknown_jurisdiction(
    jurisdiction_code,
):
    global_words = get_corpus_noise_words("global")
    assert get_corpus_noise_words(jurisdiction_code) == global_words


def test_get_corpus_noise_words_respects_max_tokens_cutoff():
    bounded = get_corpus_noise_words("gb", max_tokens=5)
    assert len(bounded) <= 5

    unbounded = get_corpus_noise_words(
        "gb", max_tokens=DEFAULT_SHORT_NAME_NOISE_WORD_MAX_TOKENS
    )
    assert set(bounded).issubset(set(unbounded))


def test_get_corpus_noise_words_is_never_flattened_across_jurisdictions():
    # No token from "fr" should leak into "gb"'s list purely by construction --
    # each jurisdiction's list is its own promoted, ranked slice.
    gb_words = set(get_corpus_noise_words("gb"))
    fr_words = set(get_corpus_noise_words("fr"))
    assert gb_words - fr_words


def test_validate_short_name_noise_word_candidates_payload_accepts_real_packaged_shape():
    import json
    from importlib.resources import files

    payload = json.loads(
        files("company_cleanse.resources")
        .joinpath("short_name_noise_word_candidates.json")
        .read_text(encoding="utf-8")
    )
    assert validate_short_name_noise_word_candidates_payload(payload) is payload


@pytest.mark.parametrize(
    "payload,match",
    [
        ([], "must be an object"),
        ({}, "systems"),
        (
            {"systems": {"gb": {"candidates": [{"document_frequency_pct": 0.1}]}}},
            "token",
        ),
        ({"systems": {"gb": {"candidates": []}}}, "global"),
    ],
)
def test_validate_short_name_noise_word_candidates_payload_rejects_bad_shapes(
    payload, match
):
    with pytest.raises((TypeError, ValueError), match=match):
        validate_short_name_noise_word_candidates_payload(payload)
