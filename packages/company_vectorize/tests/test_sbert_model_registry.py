"""Contract tests for the packaged sentence-embedding model registry.

Offline throughout: the registry is a packaged JSON resource, and nothing here
loads a checkpoint. Which checkpoints are genuine sentence-embedding models
was established when they were added (see `sbert_pooling_gate.py`'s module
docstring); what these tests hold is the *shape* of the registry and the
guarantee that an unregistered checkpoint identifier still passes through
untouched.
"""

from __future__ import annotations

import pytest
from company_vectorize.sbert_model_registry import (
    _SLUG_PATTERN,
    SbertModelEntry,
    _parse_entry,
    is_registered_checkpoint,
    list_sbert_models,
    load_sbert_model_registry,
    resolve_sbert_model_name,
    sbert_models_for_language,
)
from company_vectorize.sbert_strategy import DEFAULT_SBERT_MODEL_NAME


def _raw_entry(**overrides) -> dict:
    entry = {
        "slug": "xx-example-model",
        "checkpoint": "example-org/example-model",
        "languages": ["xx"],
        "jurisdictions": ["xx"],
        "coverage": "monolingual",
        "notes": "",
    }
    entry.update(overrides)
    return entry


def test_registry_loads_and_is_non_empty():
    registry = load_sbert_model_registry()

    assert registry
    assert all(isinstance(entry, SbertModelEntry) for entry in registry.values())
    assert all(slug == entry.slug for slug, entry in registry.items())


def test_registry_slugs_cannot_be_confused_with_checkpoint_identifiers():
    # A checkpoint identifier is "org/model" or a filesystem path; a slug is
    # neither, which is what makes resolve_sbert_model_name() unambiguous.
    for entry in list_sbert_models():
        assert _SLUG_PATTERN.fullmatch(entry.slug)
        assert "/" not in entry.slug
        assert "\\" not in entry.slug


def test_registry_covers_the_strategy_default_checkpoint():
    # The default must stay reachable and pre-verified, so an untouched run
    # behaves exactly as it did before the registry existed.
    checkpoints = {entry.checkpoint for entry in list_sbert_models()}
    assert DEFAULT_SBERT_MODEL_NAME in checkpoints
    assert is_registered_checkpoint(DEFAULT_SBERT_MODEL_NAME)


def test_resolve_slug_returns_the_registered_checkpoint():
    entry = list_sbert_models()[0]

    assert resolve_sbert_model_name(entry.slug) == entry.checkpoint


@pytest.mark.parametrize(
    "model",
    [
        "sentence-transformers/all-mpnet-base-v2",
        "some-org/an-unlisted-model",
        "/local/path/to/trained-encoder",
        "C:\\local\\path\\to\\trained-encoder",
    ],
)
def test_resolve_passes_unregistered_identifiers_through_unchanged(model: str):
    assert resolve_sbert_model_name(model) == model


def test_resolve_is_idempotent():
    entry = list_sbert_models()[0]

    once = resolve_sbert_model_name(entry.slug)

    assert resolve_sbert_model_name(once) == once


def test_registry_offers_a_model_per_repo_language():
    for language in ("en", "fr", "de"):
        matches = sbert_models_for_language(language)
        assert matches, f"no registered sentence-embedding model for '{language}'"
        assert all(language in entry.languages for entry in matches)


def test_multilingual_entries_declare_their_coverage():
    for entry in list_sbert_models():
        assert entry.coverage in {"monolingual", "multilingual"}
        if entry.coverage == "monolingual":
            assert len(entry.languages) == 1


def test_parse_entry_rejects_a_slug_that_looks_like_a_checkpoint():
    with pytest.raises(ValueError, match="invalid slug"):
        _parse_entry(_raw_entry(slug="sentence-transformers/all-MiniLM-L6-v2"), index=1)


def test_parse_entry_rejects_missing_required_key():
    raw = _raw_entry()
    del raw["checkpoint"]

    with pytest.raises(ValueError, match="missing required key"):
        _parse_entry(raw, index=3)


def test_parse_entry_rejects_entry_without_languages():
    with pytest.raises(ValueError, match="declares no languages"):
        _parse_entry(_raw_entry(languages=[]), index=2)


def test_parse_entry_rejects_unknown_coverage():
    with pytest.raises(ValueError, match="coverage"):
        _parse_entry(_raw_entry(coverage="bilingual"), index=4)
