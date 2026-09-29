from __future__ import annotations

import polars as pl
import pytest
from company_vectorize.clustering_contract import (
    TfidfTargetIndexBuildSettings,
    WordpieceTargetIndexBuildSettings,
)
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy


def _frame() -> pl.DataFrame:
    return pl.DataFrame({"id": ["a", "b"], "text": ["acme", "acme systems"]})


def _build(settings: TfidfTargetIndexBuildSettings):
    return TfidfClusteringStrategy().build_target_index(
        _frame(), target_id_col="id", text_col="text", build_settings=settings
    )


def test_default_analyzer_is_character_ngrams_so_a_one_word_name_has_features():
    index = _build(TfidfTargetIndexBuildSettings(ngram_min=2, ngram_max=3))

    assert index.vectorizer.analyzer == "char_wb"
    assert index.target_matrix[0].nnz > 0


def test_word_analyzer_leaves_a_one_word_name_without_features_under_the_default_range():
    index = _build(
        TfidfTargetIndexBuildSettings(ngram_min=2, ngram_max=3, analyzer="word")
    )

    assert index.vectorizer.analyzer == "word"
    assert index.target_matrix[0].nnz == 0
    assert index.target_matrix[1].nnz > 0


def test_unknown_analyzer_is_refused():
    with pytest.raises(ValueError, match="analyzer must be one of"):
        _build(TfidfTargetIndexBuildSettings(ngram_min=2, ngram_max=3, analyzer="byte"))


def test_rebuilt_index_scores_a_source_chunk_identically_to_the_original():
    settings = TfidfTargetIndexBuildSettings(ngram_min=2, ngram_max=3)
    strategy = TfidfClusteringStrategy()
    index = _build(settings)
    source = pl.DataFrame({"id": ["s1", "s2"], "text": ["acme systems", "unrelated"]})

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
            source_id_col="id",
            text_col="text",
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


def test_rebuild_rejects_wrong_settings_type():
    strategy = TfidfClusteringStrategy()
    index = _build(TfidfTargetIndexBuildSettings(ngram_min=2, ngram_max=3))
    parts = strategy.target_index_storable_parts(index)

    with pytest.raises(TypeError, match="tfidf strategy requires"):
        strategy.rebuild_target_index(
            parts, build_settings=WordpieceTargetIndexBuildSettings()
        )
