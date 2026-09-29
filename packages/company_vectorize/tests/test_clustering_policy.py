"""Policy-boundary tests for `clustering_policy.py`.

Exercise the representation -> class/kwargs *selection* in isolation from
construction: assert the returned class objects and `BuildSettingsPolicy`
kwargs directly, without instantiating any strategy or settings object.
`test_clustering_factory.py` covers the construction/wiring side (that the
public factory functions actually build the right instance) as a separate
concern.
"""

import pytest
from company_vectorize.clustering_contract import (
    EncoderTargetIndexBuildSettings,
    SbertTargetIndexBuildSettings,
    SentencepieceTargetIndexBuildSettings,
    TfidfTargetIndexBuildSettings,
    WordpieceTargetIndexBuildSettings,
)
from company_vectorize.clustering_policy import (
    DEFAULT_SBERT_MODEL_NAME,
    resolve_build_settings_policy,
    resolve_sbert_model_for_jurisdictions,
    resolve_strategy_class,
)
from company_vectorize.encoder_strategy import EncoderClusteringStrategy
from company_vectorize.sbert_model_registry import list_sbert_models
from company_vectorize.sbert_strategy import SbertClusteringStrategy
from company_vectorize.sentencepiece_strategy import SentencepieceClusteringStrategy
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy
from company_vectorize.wordpiece_cluster import WordpieceClusteringStrategy


@pytest.mark.parametrize(
    ("representation", "expected_class"),
    [
        ("tfidf", TfidfClusteringStrategy),
        ("wordpiece", WordpieceClusteringStrategy),
        ("sentencepiece", SentencepieceClusteringStrategy),
        ("sbert", SbertClusteringStrategy),
        ("encoder", EncoderClusteringStrategy),
    ],
)
def test_resolve_strategy_class_returns_class_not_instance(
    representation, expected_class
):
    resolved = resolve_strategy_class(representation)

    assert resolved is expected_class
    assert isinstance(resolved, type)


def test_resolve_strategy_class_rejects_unknown_representation():
    with pytest.raises(ValueError, match="Unsupported representation"):
        resolve_strategy_class("unknown")


def test_resolve_build_settings_policy_for_tfidf_carries_ngram_kwargs():
    policy = resolve_build_settings_policy(
        "tfidf", tfidf_ngram_min=2, tfidf_ngram_max=4
    )

    assert policy.settings_type is TfidfTargetIndexBuildSettings
    assert policy.kwargs == {"ngram_min": 2, "ngram_max": 4, "analyzer": "char_wb"}


def test_resolve_build_settings_policy_coerces_ngram_kwargs_to_int():
    policy = resolve_build_settings_policy(
        "tfidf", tfidf_ngram_min="1", tfidf_ngram_max="2"
    )

    assert policy.kwargs == {"ngram_min": 1, "ngram_max": 2, "analyzer": "char_wb"}


def test_resolve_build_settings_policy_for_tfidf_carries_the_analyzer():
    policy = resolve_build_settings_policy(
        "tfidf", tfidf_ngram_min=2, tfidf_ngram_max=3, tfidf_analyzer="word"
    )

    assert policy.kwargs["analyzer"] == "word"


@pytest.mark.parametrize(
    ("representation", "expected_settings_type"),
    [
        ("wordpiece", WordpieceTargetIndexBuildSettings),
        ("sentencepiece", SentencepieceTargetIndexBuildSettings),
    ],
)
def test_resolve_build_settings_policy_for_token_list_representations_has_no_kwargs(
    representation, expected_settings_type
):
    policy = resolve_build_settings_policy(
        representation, tfidf_ngram_min=1, tfidf_ngram_max=1
    )

    assert policy.settings_type is expected_settings_type
    assert policy.kwargs == {}


def test_resolve_build_settings_policy_for_sbert_defaults_model_name():
    policy = resolve_build_settings_policy(
        "sbert", tfidf_ngram_min=1, tfidf_ngram_max=1
    )

    assert policy.settings_type is SbertTargetIndexBuildSettings
    assert policy.kwargs == {
        "model_name": DEFAULT_SBERT_MODEL_NAME,
        "force_untrained_pooling": False,
    }


def test_resolve_build_settings_policy_for_sbert_forwards_custom_model_name():
    policy = resolve_build_settings_policy(
        "sbert",
        tfidf_ngram_min=1,
        tfidf_ngram_max=1,
        sbert_model_name="local/path/to/trained-encoder",
    )

    assert policy.kwargs == {
        "model_name": "local/path/to/trained-encoder",
        "force_untrained_pooling": False,
    }


def test_resolve_build_settings_policy_rejects_unknown_representation():
    with pytest.raises(ValueError, match="Unsupported representation"):
        resolve_build_settings_policy("unknown", tfidf_ngram_min=1, tfidf_ngram_max=1)


class _FakeEncoder:
    def embed(self, name: str):
        return [0.0]


def test_resolve_build_settings_policy_for_encoder_carries_the_handle_and_name():
    fake_encoder = _FakeEncoder()
    policy = resolve_build_settings_policy(
        "encoder",
        tfidf_ngram_min=1,
        tfidf_ngram_max=1,
        encoder=fake_encoder,
        encoder_name="fake-encoder-v1",
    )

    assert policy.settings_type is EncoderTargetIndexBuildSettings
    assert policy.kwargs == {"encoder": fake_encoder, "encoder_name": "fake-encoder-v1"}


def test_resolve_build_settings_policy_for_encoder_requires_an_encoder():
    with pytest.raises(ValueError, match="requires encoder="):
        resolve_build_settings_policy("encoder", tfidf_ngram_min=1, tfidf_ngram_max=1)


@pytest.mark.parametrize(
    ("jurisdictions", "expected_slug"),
    [
        (("fr",), "fr-sentence-camembert-base"),
        (("de",), "de-gbert-large-paraphrase-cosine"),
    ],
)
def test_resolve_sbert_model_for_jurisdictions_picks_the_matched_monolingual_entry(
    jurisdictions, expected_slug
):
    expected_checkpoint = next(
        entry.checkpoint for entry in list_sbert_models() if entry.slug == expected_slug
    )

    assert resolve_sbert_model_for_jurisdictions(jurisdictions) == expected_checkpoint


@pytest.mark.parametrize(
    "jurisdictions",
    [None, (), ("gb",), ("ie",), ("xx",), ("fr", "de")],
)
def test_resolve_sbert_model_for_jurisdictions_falls_back_to_the_default(
    jurisdictions,
):
    # "gb"/"ie" resolve to the same checkpoint as the default anyway (the
    # registry's own English monolingual entry), "xx" matches nothing, and
    # ("fr", "de") together match no single monolingual entry's coverage --
    # all fall back rather than failing the run.
    assert resolve_sbert_model_for_jurisdictions(jurisdictions) == (
        DEFAULT_SBERT_MODEL_NAME
    )
