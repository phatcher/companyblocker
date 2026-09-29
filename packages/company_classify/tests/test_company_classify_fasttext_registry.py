"""Contract tests for the packaged pretrained fastText checkpoint registry.

Offline throughout: the registry is a packaged JSON resource and nothing here downloads a
checkpoint.
"""

from __future__ import annotations

import pytest
from company_classify.fasttext_registry import (
    _SLUG_PATTERN,
    DEFAULT_FASTTEXT_CHECKPOINT_SLUG,
    FastTextCheckpointEntry,
    _parse_entry,
    fasttext_checkpoints_for_language,
    list_fasttext_checkpoints,
    load_fasttext_checkpoint_registry,
    resolve_fasttext_checkpoint_entry,
    resolve_fasttext_slug_for_jurisdictions,
)


def _raw_entry(**overrides) -> dict:
    entry = {
        "slug": "xx-example-300",
        "source_url": "https://example.invalid/cc.xx.300.bin.gz",
        "languages": ["xx"],
        "jurisdictions": ["xx"],
        "coverage": "monolingual",
        "dimension": 300,
        "checksum": None,
        "notes": "",
    }
    entry.update(overrides)
    return entry


def test_registry_loads_and_is_non_empty():
    registry = load_fasttext_checkpoint_registry()

    assert registry
    assert all(
        isinstance(entry, FastTextCheckpointEntry) for entry in registry.values()
    )
    assert all(slug == entry.slug for slug, entry in registry.items())


def test_registry_slugs_match_the_shared_slug_pattern():
    for entry in list_fasttext_checkpoints():
        assert _SLUG_PATTERN.fullmatch(entry.slug)


def test_registry_covers_english_and_the_repo_non_english_jurisdictions():
    for language in ("en", "fr", "de"):
        matches = fasttext_checkpoints_for_language(language)
        assert matches, f"no registered fastText checkpoint for '{language}'"
        assert all(language in entry.languages for entry in matches)


def test_default_checkpoint_is_registered_and_english():
    entry = resolve_fasttext_checkpoint_entry(DEFAULT_FASTTEXT_CHECKPOINT_SLUG)
    assert "en" in entry.languages
    assert entry.coverage == "monolingual"


def test_resolve_checkpoint_entry_raises_for_an_unregistered_slug():
    with pytest.raises(KeyError):
        resolve_fasttext_checkpoint_entry("not-a-registered-slug")


@pytest.mark.parametrize(
    ("jurisdictions", "expected_slug"),
    [
        (("fr",), "fr-cc-300"),
        (("de",), "de-cc-300"),
        (("gb",), DEFAULT_FASTTEXT_CHECKPOINT_SLUG),
        (("ie",), DEFAULT_FASTTEXT_CHECKPOINT_SLUG),
    ],
)
def test_resolve_slug_for_jurisdictions_picks_the_matched_entry(
    jurisdictions, expected_slug
):
    assert resolve_fasttext_slug_for_jurisdictions(jurisdictions) == expected_slug


@pytest.mark.parametrize(
    "jurisdictions",
    [None, (), ("xx",), ("fr", "de")],
)
def test_resolve_slug_for_jurisdictions_falls_back_to_the_default(jurisdictions):
    assert (
        resolve_fasttext_slug_for_jurisdictions(jurisdictions)
        == DEFAULT_FASTTEXT_CHECKPOINT_SLUG
    )


def test_parse_entry_rejects_a_slug_that_looks_like_a_url_path():
    with pytest.raises(ValueError, match="invalid slug"):
        _parse_entry(_raw_entry(slug="org/model"), index=1)


def test_parse_entry_rejects_missing_required_key():
    raw = _raw_entry()
    del raw["source_url"]

    with pytest.raises(ValueError, match="missing required key"):
        _parse_entry(raw, index=2)


def test_parse_entry_rejects_entry_without_languages():
    with pytest.raises(ValueError, match="declares no languages"):
        _parse_entry(_raw_entry(languages=[]), index=3)


def test_parse_entry_rejects_unknown_coverage():
    with pytest.raises(ValueError, match="coverage"):
        _parse_entry(_raw_entry(coverage="bilingual"), index=4)


def test_parse_entry_accepts_a_missing_checksum_as_none():
    raw = _raw_entry()
    del raw["checksum"]

    entry = _parse_entry(raw, index=5)

    assert entry.checksum is None
