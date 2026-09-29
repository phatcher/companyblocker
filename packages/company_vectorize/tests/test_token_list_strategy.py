"""Unit coverage for `TokenListClusteringStrategy` itself.

`wordpiece_cluster.py`/`sentencepiece_strategy.py` subclass this and are
exercised by `test_wordpiece_cluster.py`/`test_sentencepiece_strategy.py`,
but neither test imports `token_list_strategy` by its own path, so this
module's own decisions -- the `build_settings` type guard, the TF-IDF-over-
tokens vectorizer configuration, the list-shaped source-scoring projection,
and wiring `representation` through to the dense-vocabulary gate on both
`build_backend_index()` and `score_source_chunk()` -- are asserted here
against a minimal stub subclass instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest
from company_vectorize.clustering_contract import TargetIndexBuildSettings
from company_vectorize.dense_vocabulary_gate import DenseVocabularyScaleError
from company_vectorize.token_list_strategy import TokenListClusteringStrategy


@dataclass(frozen=True)
class _StubSettings(TargetIndexBuildSettings):
    """Concrete settings type for a strategy that carries no tuning of its own."""


@dataclass(frozen=True)
class _OtherSettings(TargetIndexBuildSettings):
    """A distinct settings type, used only to trip the type-mismatch guard."""


class _SparseStubStrategy(TokenListClusteringStrategy):
    """A representation outside `DENSE_VOCABULARY_REPRESENTATIONS`."""

    representation = "stub-tokens"
    settings_type = _StubSettings


class _DenseStubStrategy(TokenListClusteringStrategy):
    """A representation the dense-vocabulary gate actually watches."""

    representation = "wordpiece"
    settings_type = _StubSettings


def _target_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target_uri": ["t1", "t2"],
            "tokens": [["ac", "me", "ltd"], ["ac", "me", "holdings"]],
        }
    )


def test_build_target_index_rejects_a_mismatched_settings_type():
    strategy = _SparseStubStrategy()

    with pytest.raises(TypeError, match="stub-tokens"):
        strategy.build_target_index(
            _target_frame(),
            target_id_col="target_uri",
            text_col="tokens",
            build_settings=_OtherSettings(),
        )


def test_build_target_index_fits_a_tfidf_vectorizer_over_token_lists():
    strategy = _SparseStubStrategy()

    index = strategy.build_target_index(
        _target_frame(),
        target_id_col="target_uri",
        text_col="tokens",
        build_settings=_StubSettings(),
    )

    assert index.target_ids == ["t1", "t2"]
    assert index.target_matrix.shape[0] == 2
    assert hasattr(index.vectorizer, "idf_")
    # The vectorizer treats each element of the list column as a whole token
    # rather than re-tokenizing text, so "ac" and "me" are distinct features.
    assert "ac" in index.vectorizer.vocabulary_
    assert "me" in index.vectorizer.vocabulary_


def test_build_source_scoring_projection_treats_the_column_as_token_lists():
    strategy = _SparseStubStrategy()
    frame = pl.DataFrame(
        {"tokens": [["a", "b"], [], None]},
        schema={"tokens": pl.List(pl.Utf8)},
    )
    text_value_expr, value_col, nonempty_predicate = (
        strategy._build_source_scoring_projection("tokens")
    )

    projected = frame.select(text_value_expr.alias(value_col))
    assert projected.get_column(value_col).to_list() == [["a", "b"], [], []]

    kept = projected.select(nonempty_predicate.alias("keep")).get_column("keep")
    assert kept.to_list() == [True, False, False]


def test_ensure_backend_supported_gates_a_dense_representation():
    strategy = _DenseStubStrategy()
    index = strategy.build_target_index(
        _target_frame(),
        target_id_col="target_uri",
        text_col="tokens",
        build_settings=_StubSettings(),
    )

    with pytest.raises(DenseVocabularyScaleError, match="wordpiece"):
        strategy.build_backend_index(
            index,
            backend="sklearn",
            top_k=2,
            max_candidates_per_source=None,
            backend_options={"max_rows": 1},
        )

    with pytest.raises(DenseVocabularyScaleError, match="wordpiece"):
        strategy.score_source_chunk(
            _target_frame().rename({"target_uri": "source_uri"}),
            target_index=index,
            source_id_col="source_uri",
            text_col="tokens",
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
            nn_index=None,
            backend="sklearn",
            backend_index=None,
            backend_options={"max_rows": 1},
        )


def test_rebuilt_index_scores_a_source_chunk_identically_to_the_original():
    strategy = _SparseStubStrategy()
    settings = _StubSettings()
    index = strategy.build_target_index(
        _target_frame(),
        target_id_col="target_uri",
        text_col="tokens",
        build_settings=settings,
    )
    source = pl.DataFrame(
        {
            "source_uri": ["s1", "s2"],
            "tokens": [["ac", "me", "ltd"], ["unrelated", "words"]],
        }
    )

    def _score(built_index):
        backend_index = strategy.build_backend_index(
            built_index,
            backend="sklearn",
            top_k=2,
            max_candidates_per_source=None,
            backend_options=None,
        )
        return strategy.score_source_chunk(
            source,
            target_index=built_index,
            source_id_col="source_uri",
            text_col="tokens",
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
            nn_index=None,
            backend="sklearn",
            backend_index=backend_index,
            backend_options=None,
        ).sort(["source_id", "rank"])

    parts = strategy.target_index_storable_parts(index)
    rebuilt = strategy.rebuild_target_index(parts, build_settings=settings)

    assert rebuilt.target_ids == index.target_ids
    assert _score(rebuilt).equals(_score(index))


def test_rebuild_rejects_a_mismatched_settings_type():
    strategy = _SparseStubStrategy()
    index = strategy.build_target_index(
        _target_frame(),
        target_id_col="target_uri",
        text_col="tokens",
        build_settings=_StubSettings(),
    )
    parts = strategy.target_index_storable_parts(index)

    with pytest.raises(TypeError, match="stub-tokens"):
        strategy.rebuild_target_index(parts, build_settings=_OtherSettings())


def test_ensure_backend_supported_leaves_a_non_gated_representation_alone():
    strategy = _SparseStubStrategy()
    index = strategy.build_target_index(
        _target_frame(),
        target_id_col="target_uri",
        text_col="tokens",
        build_settings=_StubSettings(),
    )

    backend_index = strategy.build_backend_index(
        index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options={"max_rows": 1},
    )

    assert backend_index is not None
