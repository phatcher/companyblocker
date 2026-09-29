import polars as pl
import pytest
from company_vectorize import (
    WordpieceTargetIndexBuildSettings,
    resolve_target_index_build_settings,
)
from company_vectorize.dense_vocabulary_gate import DenseVocabularyScaleError
from company_vectorize.sentencepiece_strategy import SentencepieceClusteringStrategy


def test_sentencepiece_strategy_builds_weighted_sparse_index():
    strategy = SentencepieceClusteringStrategy()
    target = pl.DataFrame(
        {
            "system_uri": ["t1", "t2"],
            "cluster_tokens": [["▁ac", "me", "▁ltd"], ["▁ac", "me", "▁holdings"]],
        }
    )

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="cluster_tokens",
        build_settings=resolve_target_index_build_settings(
            "sentencepiece",
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
        ),
    )

    assert index.target_ids == ["t1", "t2"]
    assert index.target_matrix.shape[0] == 2
    # TfidfVectorizer exposes learned inverse-document frequencies.
    assert hasattr(index.vectorizer, "idf_")
    assert len(index.vectorizer.idf_) > 0


def test_sentencepiece_strategy_scores_source_chunk():
    strategy = SentencepieceClusteringStrategy()
    target = pl.DataFrame(
        {
            "system_uri": ["t1", "t2"],
            "cluster_tokens": [["▁ac", "me", "▁ltd"], ["▁be", "ta", "▁plc"]],
        }
    )
    source = pl.DataFrame(
        {
            "system_uri": ["s1"],
            "cluster_tokens": [["▁ac", "me", "▁ltd"]],
        }
    )

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="cluster_tokens",
        build_settings=resolve_target_index_build_settings(
            "sentencepiece",
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
        ),
    )
    backend_index = strategy.build_backend_index(
        index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options=None,
    )

    rows = strategy.score_source_chunk(
        source,
        target_index=index,
        source_id_col="system_uri",
        text_col="cluster_tokens",
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=backend_index,
        backend_options=None,
    )

    assert rows.height >= 1
    top = rows.sort(["source_id", "rank"]).row(0, named=True)
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t1"


def test_sentencepiece_strategy_rejects_wrong_settings_type():
    strategy = SentencepieceClusteringStrategy()
    target = pl.DataFrame(
        {"system_uri": ["t1"], "cluster_tokens": [["▁ac", "me"]]},
    )

    with pytest.raises(TypeError, match="sentencepiece strategy requires"):
        strategy.build_target_index(
            target,
            target_id_col="system_uri",
            text_col="cluster_tokens",
            build_settings=WordpieceTargetIndexBuildSettings(),
        )


def test_sentencepiece_strategy_is_gated_like_wordpiece():
    # The dense-vocabulary gate sits on the shared TokenListClusteringStrategy,
    # not either tokenizer: sentencepiece has to be refused on the same terms.
    strategy = SentencepieceClusteringStrategy()
    target = pl.DataFrame(
        {
            "system_uri": ["t1", "t2"],
            "cluster_tokens": [["▁ac", "me", "▁ltd"], ["▁be", "ta", "▁plc"]],
        }
    )
    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="cluster_tokens",
        build_settings=resolve_target_index_build_settings(
            "sentencepiece",
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
        ),
    )

    with pytest.raises(DenseVocabularyScaleError, match="sentencepiece"):
        strategy.build_backend_index(
            index,
            backend="sklearn",
            top_k=2,
            max_candidates_per_source=None,
            backend_options={"max_rows": 1},
        )
