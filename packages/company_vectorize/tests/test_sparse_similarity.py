import polars as pl
from company_vectorize import resolve_target_index_build_settings
from company_vectorize.sparse_similarity import (
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy


def _build_target_index():
    strategy = TfidfClusteringStrategy()
    target = pl.DataFrame(
        {
            "system_uri": ["t1", "t2", "t3"],
            "name": ["acme limited", "acme holdings", "beta partners"],
        }
    )
    return strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=resolve_target_index_build_settings(
            "tfidf", tfidf_ngram_min=1, tfidf_ngram_max=1
        ),
    )


def test_svd_rerank_backend_index_builds_and_scores():
    target_index = _build_target_index()

    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="svd_rerank",
        top_k=2,
        max_candidates_per_source=None,
        backend_options={"svd_candidates": 3, "svd_dimensions": 2},
    )

    assert backend_index is not None
    assert backend_index.backend == "svd_rerank"
    assert backend_index.svd_rerank is not None
    assert backend_index.svd_rerank.candidate_count == 3

    rows = score_source_with_backend(
        ["s1"],
        target_index.vectorizer.transform(["acme limited"]),
        target_index=target_index,
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="svd_rerank",
        backend_index=backend_index,
    )

    assert rows
    top = min(rows, key=lambda row: row["rank"])
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t1"


def test_svd_rerank_backend_falls_back_to_sklearn_without_index():
    target_index = _build_target_index()

    rows = score_source_with_backend(
        ["s1"],
        target_index.vectorizer.transform(["acme limited"]),
        target_index=target_index,
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="svd_rerank",
        backend_index=None,
    )

    assert rows
    top = min(rows, key=lambda row: row["rank"])
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t1"
