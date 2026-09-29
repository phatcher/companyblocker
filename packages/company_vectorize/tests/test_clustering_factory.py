import pytest
from company_vectorize import (
    clustering_factory,
    resolve_clustering_strategy,
    resolve_target_index_build_settings,
)
from company_vectorize.clustering_contract import (
    EncoderTargetIndexBuildSettings,
    SbertTargetIndexBuildSettings,
)
from company_vectorize.clustering_policy import BuildSettingsPolicy
from company_vectorize.encoder_strategy import EncoderClusteringStrategy
from company_vectorize.sbert_strategy import (
    DEFAULT_SBERT_MODEL_NAME,
    SbertClusteringStrategy,
)
from company_vectorize.sentencepiece_strategy import SentencepieceClusteringStrategy
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy
from company_vectorize.wordpiece_cluster import WordpieceClusteringStrategy


def test_resolve_clustering_strategy_for_tfidf():
    strategy = resolve_clustering_strategy("tfidf")
    assert isinstance(strategy, TfidfClusteringStrategy)


def test_resolve_clustering_strategy_for_wordpiece():
    strategy = resolve_clustering_strategy("wordpiece")
    assert isinstance(strategy, WordpieceClusteringStrategy)
    assert strategy.representation == "wordpiece"


def test_resolve_clustering_strategy_for_sentencepiece():
    strategy = resolve_clustering_strategy("sentencepiece")
    assert isinstance(strategy, SentencepieceClusteringStrategy)
    assert strategy.representation == "sentencepiece"


def test_resolve_clustering_strategy_for_sbert():
    strategy = resolve_clustering_strategy("sbert")
    assert isinstance(strategy, SbertClusteringStrategy)
    assert strategy.representation == "sbert"


def test_resolve_clustering_strategy_for_encoder():
    strategy = resolve_clustering_strategy("encoder")
    assert isinstance(strategy, EncoderClusteringStrategy)
    assert strategy.representation == "encoder"


def test_resolve_clustering_strategy_rejects_unknown_representation():
    with pytest.raises(ValueError, match="Unsupported representation"):
        resolve_clustering_strategy("unknown")


def test_resolve_target_index_build_settings_for_sbert_default_model():
    settings = resolve_target_index_build_settings(
        "sbert", tfidf_ngram_min=1, tfidf_ngram_max=1
    )
    assert isinstance(settings, SbertTargetIndexBuildSettings)
    assert settings.model_name == DEFAULT_SBERT_MODEL_NAME


def test_resolve_target_index_build_settings_for_sbert_custom_model():
    settings = resolve_target_index_build_settings(
        "sbert",
        tfidf_ngram_min=1,
        tfidf_ngram_max=1,
        sbert_model_name="local/path/to/trained-encoder",
    )
    assert settings.model_name == "local/path/to/trained-encoder"
    assert settings.force_untrained_pooling is False


def test_resolve_target_index_build_settings_for_sbert_resolves_a_registry_slug():
    # A slug is resolved here, so what reaches
    # SbertTargetIndexBuildSettings.model_name is always loadable directly.
    settings = resolve_target_index_build_settings(
        "sbert",
        tfidf_ngram_min=1,
        tfidf_ngram_max=1,
        sbert_model_name="fr-sentence-camembert-base",
    )
    assert settings.model_name == "dangvantuan/sentence-camembert-base"


def test_resolve_target_index_build_settings_for_sbert_threads_force_override():
    settings = resolve_target_index_build_settings(
        "sbert",
        tfidf_ngram_min=1,
        tfidf_ngram_max=1,
        sbert_model_name="some-org/unlisted-model",
        sbert_force_untrained_pooling=True,
    )
    assert settings.model_name == "some-org/unlisted-model"
    assert settings.force_untrained_pooling is True


def test_resolve_target_index_build_settings_for_encoder():
    class _FakeEncoder:
        def embed(self, name: str):
            return [0.0]

    fake_encoder = _FakeEncoder()
    settings = resolve_target_index_build_settings(
        "encoder",
        tfidf_ngram_min=1,
        tfidf_ngram_max=1,
        encoder=fake_encoder,
        encoder_name="fake-encoder-v1",
    )
    assert isinstance(settings, EncoderTargetIndexBuildSettings)
    assert settings.encoder is fake_encoder
    assert settings.encoder_name == "fake-encoder-v1"


def test_resolve_target_index_build_settings_for_encoder_requires_an_encoder():
    with pytest.raises(ValueError, match="requires encoder="):
        resolve_target_index_build_settings(
            "encoder", tfidf_ngram_min=1, tfidf_ngram_max=1
        )


# --- Construction/wiring boundary -------------------------------------------
#
# The tests above exercise the factory functions end-to-end (representation
# name in, real instance out), which also incidentally exercises
# `clustering_policy.py`'s selection logic (covered directly and in isolation
# by `test_clustering_policy.py`). These tests isolate the *other* half:
# given a policy result, does the factory construct exactly what the policy
# says to build, and nothing else -- verified by substituting a fake policy
# result via monkeypatch rather than relying on a real representation name.


class _FakeStrategy:
    def __init__(self) -> None:
        self.constructed = True


def test_resolve_clustering_strategy_instantiates_whatever_policy_selects(monkeypatch):
    monkeypatch.setattr(
        clustering_factory,
        "resolve_strategy_class",
        lambda representation: _FakeStrategy,
    )

    strategy = resolve_clustering_strategy("anything")

    assert isinstance(strategy, _FakeStrategy)
    assert strategy.constructed is True


def test_resolve_target_index_build_settings_builds_settings_type_with_policy_kwargs(
    monkeypatch,
):
    captured_calls: list[tuple[str, dict[str, object]]] = []

    def _fake_resolve_build_settings_policy(representation, **kwargs):
        captured_calls.append((representation, kwargs))
        return BuildSettingsPolicy(
            SbertTargetIndexBuildSettings, {"model_name": "fake-model"}
        )

    monkeypatch.setattr(
        clustering_factory,
        "resolve_build_settings_policy",
        _fake_resolve_build_settings_policy,
    )

    settings = resolve_target_index_build_settings(
        "anything", tfidf_ngram_min=9, tfidf_ngram_max=9
    )

    assert isinstance(settings, SbertTargetIndexBuildSettings)
    assert settings.model_name == "fake-model"
    assert captured_calls == [
        (
            "anything",
            {
                "tfidf_ngram_min": 9,
                "tfidf_ngram_max": 9,
                "tfidf_analyzer": "char_wb",
                "sbert_model_name": DEFAULT_SBERT_MODEL_NAME,
                "sbert_force_untrained_pooling": False,
                "encoder": None,
                "encoder_name": "",
            },
        )
    ]


def test_resolve_target_index_build_settings_never_calls_settings_constructor_with_extra_kwargs(
    monkeypatch,
):
    """Construction uses exactly the kwargs policy hands back -- no implicit
    extras (e.g. `representation`) leak into the settings constructor call."""

    class _RecordingSettingsType:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(
        clustering_factory,
        "resolve_build_settings_policy",
        lambda representation, **kwargs: BuildSettingsPolicy(
            _RecordingSettingsType, {"only": "this"}
        ),
    )

    settings = resolve_target_index_build_settings(
        "anything", tfidf_ngram_min=1, tfidf_ngram_max=1
    )

    assert settings.kwargs == {"only": "this"}
