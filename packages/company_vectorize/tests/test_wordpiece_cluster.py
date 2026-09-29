import polars as pl
import pytest
from company_vectorize import resolve_target_index_build_settings
from company_vectorize.dense_vocabulary_gate import DenseVocabularyScaleError
from company_vectorize.wordpiece_cluster import WordpieceClusteringStrategy


def test_wordpiece_strategy_builds_weighted_sparse_index():
    strategy = WordpieceClusteringStrategy()
    target = pl.DataFrame(
        {
            "system_uri": ["t1", "t2"],
            "cluster_tokens": [["ac", "##me", "ltd"], ["ac", "##me", "holdings"]],
        }
    )

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="cluster_tokens",
        build_settings=resolve_target_index_build_settings(
            "wordpiece",
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
        ),
    )

    assert index.target_ids == ["t1", "t2"]
    assert index.target_matrix.shape[0] == 2
    # TfidfVectorizer exposes learned inverse-document frequencies.
    assert hasattr(index.vectorizer, "idf_")
    assert len(index.vectorizer.idf_) > 0


def test_wordpiece_strategy_scores_source_chunk():
    strategy = WordpieceClusteringStrategy()
    target = pl.DataFrame(
        {
            "system_uri": ["t1", "t2"],
            "cluster_tokens": [["ac", "##me", "ltd"], ["be", "##ta", "plc"]],
        }
    )
    source = pl.DataFrame(
        {
            "system_uri": ["s1"],
            "cluster_tokens": [["ac", "##me", "ltd"]],
        }
    )

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="cluster_tokens",
        build_settings=resolve_target_index_build_settings(
            "wordpiece",
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


def _two_row_index(strategy):
    target = pl.DataFrame(
        {
            "system_uri": ["t1", "t2"],
            "cluster_tokens": [["ac", "##me", "ltd"], ["be", "##ta", "plc"]],
        }
    )
    return strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="cluster_tokens",
        build_settings=resolve_target_index_build_settings(
            "wordpiece",
            tfidf_ngram_min=1,
            tfidf_ngram_max=2,
        ),
    )


def _source_chunk():
    return pl.DataFrame(
        {"system_uri": ["s1"], "cluster_tokens": [["ac", "##me", "ltd"]]}
    )


@pytest.mark.parametrize("backend", ["sklearn", "sparse_dot_topn"])
def test_build_backend_index_refuses_exhaustive_backend_past_max_rows(backend):
    # The dense-vocabulary gate. max_rows is dropped to 1 rather than the
    # target side grown to gb's real 5.7M rows: the gate's decision is over the
    # row count, so the threshold is what has to move for a test to reach it.
    strategy = WordpieceClusteringStrategy()
    index = _two_row_index(strategy)

    with pytest.raises(DenseVocabularyScaleError, match="wordpiece"):
        strategy.build_backend_index(
            index,
            backend=backend,
            top_k=2,
            max_candidates_per_source=None,
            backend_options={"max_rows": 1},
        )


def test_score_source_chunk_refuses_independently_of_build_backend_index():
    # score_source_chunk() is reachable with a caller-supplied nn_index and no
    # backend_index, so the gate cannot live in build_backend_index() alone.
    strategy = WordpieceClusteringStrategy()
    index = _two_row_index(strategy)

    with pytest.raises(DenseVocabularyScaleError):
        strategy.score_source_chunk(
            _source_chunk(),
            target_index=index,
            source_id_col="system_uri",
            text_col="cluster_tokens",
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
            nn_index=None,
            backend="sklearn",
            backend_index=None,
            backend_options={"max_rows": 1},
        )


def test_force_runs_the_gated_combination_end_to_end():
    strategy = WordpieceClusteringStrategy()
    index = _two_row_index(strategy)
    options = {"max_rows": 1, "force": True}

    backend_index = strategy.build_backend_index(
        index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options=options,
    )
    rows = strategy.score_source_chunk(
        _source_chunk(),
        target_index=index,
        source_id_col="system_uri",
        text_col="cluster_tokens",
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=backend_index,
        backend_options=options,
    )

    assert rows.sort(["source_id", "rank"]).row(0, named=True)["target_id"] == "t1"


def test_small_target_side_needs_no_force():
    # The default max_rows already covers a two-row target, so the everyday
    # small-corpus case must not have to know the gate exists.
    strategy = WordpieceClusteringStrategy()
    index = _two_row_index(strategy)

    backend_index = strategy.build_backend_index(
        index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options=None,
    )

    assert backend_index is not None


def test_svd_rerank_is_not_gated_at_any_scale():
    strategy = WordpieceClusteringStrategy()
    index = _two_row_index(strategy)

    backend_index = strategy.build_backend_index(
        index,
        backend="svd_rerank",
        top_k=2,
        max_candidates_per_source=None,
        backend_options={"max_rows": 1},
    )

    assert backend_index is not None
