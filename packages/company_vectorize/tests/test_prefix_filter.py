import numpy as np
import polars as pl
import pytest
from company_vectorize import resolve_target_index_build_settings
from company_vectorize.clustering_contract import TargetClusteringIndex
from company_vectorize.prefix_filter import (
    PrefixFilterIndex,
    build_prefix_filter_index,
    prefix_filter_candidates,
)
from company_vectorize.sparse_similarity import (
    build_target_nearest_neighbors,
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy
from scipy.sparse import csr_matrix


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
        # Whole words, so the vocabulary the assertions name ("beta") is a
        # feature; the default character analyzer would split it.
        build_settings=resolve_target_index_build_settings(
            "tfidf", tfidf_ngram_min=1, tfidf_ngram_max=1, tfidf_analyzer="word"
        ),
    )


def _row(matrix: csr_matrix, index: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = matrix.tocsr()
    start, end = matrix.indptr[index], matrix.indptr[index + 1]
    return matrix.indices[start:end], matrix.data[start:end]


def _pairs(rows: list[dict[str, object]]) -> dict[tuple[str, str], float]:
    pairs: dict[tuple[str, str], float] = {}
    for row in rows:
        similarity = row["similarity"]
        assert isinstance(similarity, float)
        pairs[(str(row["source_id"]), str(row["target_id"]))] = similarity
    return pairs


def _assert_same_pairs(
    actual: dict[tuple[str, str], float], expected: dict[tuple[str, str], float]
) -> None:
    """Same candidate pairs, scores equal to float rounding: the exhaustive
    path reads cosine off sklearn's distance, the filtered path off a direct
    dot product."""
    assert set(actual) == set(expected)
    for pair, score in expected.items():
        assert actual[pair] == pytest.approx(score, abs=1e-6), pair


def test_build_prefix_filter_index_rejects_bad_min_similarity():
    target_index = _build_target_index()
    with pytest.raises(ValueError):
        build_prefix_filter_index(target_index.target_matrix, min_similarity=-0.1)
    with pytest.raises(ValueError):
        build_prefix_filter_index(target_index.target_matrix, min_similarity=1.5)


def test_build_prefix_filter_index_rejects_dense_matrix():
    dense = np.ones((3, 4))
    with pytest.raises(ValueError):
        build_prefix_filter_index(dense, min_similarity=0.5)  # type: ignore[arg-type]


def test_build_prefix_filter_index_none_for_empty_target():
    empty = csr_matrix((0, 0))
    assert build_prefix_filter_index(empty, min_similarity=0.5) is None


def test_prefix_filter_index_records_its_min_similarity():
    target_index = _build_target_index()
    prefix_index = build_prefix_filter_index(
        target_index.target_matrix, min_similarity=0.6
    )
    assert prefix_index is not None
    assert prefix_index.min_similarity == 0.6


def test_prefix_filter_candidates_finds_shared_rare_token():
    target_index = _build_target_index()
    prefix_index = build_prefix_filter_index(
        target_index.target_matrix, min_similarity=0.5
    )
    assert prefix_index is not None

    # "beta" is a rare token unique to t3; a source row containing it should
    # surface t3 as a candidate.
    beta_col = target_index.vectorizer.vocabulary_["beta"]
    candidates = prefix_filter_candidates(
        np.asarray([beta_col], dtype=np.int64), np.asarray([1.0]), prefix_index
    )
    t3_row = target_index.target_ids.index("t3")
    assert t3_row in candidates.tolist()


def test_prefix_filter_candidates_empty_for_empty_row():
    target_index = _build_target_index()
    prefix_index = build_prefix_filter_index(
        target_index.target_matrix, min_similarity=0.5
    )
    assert prefix_index is not None

    candidates = prefix_filter_candidates(
        np.asarray([], dtype=np.int64), np.asarray([]), prefix_index
    )
    assert candidates.size == 0


def test_prefix_ends_where_the_remaining_norm_drops_below_the_threshold():
    """A heavy rare term alone reaches the threshold, so the light common
    terms after it are outside the prefix. The token-Jaccard bound this
    replaced would have kept two of the three columns."""
    weights = np.asarray([0.99, 0.1, 0.1])
    weights = weights / np.linalg.norm(weights)
    # One extra row per common column so column 0 ranks rarest.
    target = csr_matrix(
        np.vstack(
            [
                weights,
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    )
    prefix_index = build_prefix_filter_index(target, min_similarity=0.75)
    assert prefix_index is not None

    assert prefix_index.inverted_index[0].tolist() == [0]
    assert prefix_index.inverted_index[1].tolist() == [1]
    assert prefix_index.inverted_index[2].tolist() == [2]


def test_filtered_scan_keeps_a_pair_the_jaccard_bound_dropped():
    """Rows where the two bounds disagree: t1 shares one common column with
    the source, so its token-Jaccard prefix (three of four columns) left that
    column out and the pair was never scored, though its cosine clears the
    threshold. The cosine bound keeps every column of t1 because the norm
    remaining after each stays above the threshold, so the filtered scan
    returns exactly the exhaustive scan's candidates."""
    target = csr_matrix(
        np.asarray(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.0, 0.0, 0.6, 0.8],
                [0.0, 0.6, 0.0, 0.8],
            ]
        )
    )
    target_index = TargetClusteringIndex(
        target_ids=["t1", "t2", "t3"], vectorizer=None, target_matrix=target
    )
    source = csr_matrix(np.asarray([[0.0, 0.0, 0.0, 1.0]]))
    min_similarity = 0.4

    filtered_index = build_target_similarity_backend_index(
        target_index,
        backend="sklearn",
        top_k=5,
        backend_options={"prefix_filter": True},
        min_similarity=min_similarity,
    )
    assert filtered_index is not None and filtered_index.prefix_filter is not None
    filtered = score_source_with_backend(
        ["s1"],
        source,
        target_index=target_index,
        top_k=5,
        min_similarity=min_similarity,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=filtered_index,
    )
    exhaustive = score_source_with_backend(
        ["s1"],
        source,
        target_index=target_index,
        top_k=5,
        min_similarity=min_similarity,
        max_candidates_per_source=None,
        nn_index=build_target_nearest_neighbors(target_index, top_k=5),
        backend="sklearn",
        backend_index=None,
    )

    assert _pairs(filtered)[("s1", "t1")] == pytest.approx(0.5)
    _assert_same_pairs(_pairs(filtered), _pairs(exhaustive))


def test_filtered_scan_equals_exhaustive_scan_on_a_fixture():
    strategy = TfidfClusteringStrategy()
    names = [
        "acme limited",
        "acme holdings",
        "acme holding company",
        "beta partners",
        "beta partner group",
        "gamma industries",
        "gamma industrial supplies",
        "delta logistics",
        "delta logistic services",
        "epsilon trading",
    ]
    target = pl.DataFrame(
        {"system_uri": [f"t{i}" for i in range(len(names))], "name": names}
    )
    target_index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=resolve_target_index_build_settings(
            "tfidf", tfidf_ngram_min=2, tfidf_ngram_max=3
        ),
    )
    sources = [
        "acme ltd",
        "beta partners llp",
        "gamma industries plc",
        "delta logistics",
        "zeta unrelated",
    ]
    src_ids = [f"s{i}" for i in range(len(sources))]
    src_matrix = target_index.vectorizer.transform(sources)
    top_k = 10  # Wider than any candidate set, so no top-k tie can differ.

    for min_similarity in (0.0, 0.3, 0.6):
        filtered_index = build_target_similarity_backend_index(
            target_index,
            backend="sklearn",
            top_k=top_k,
            backend_options={"prefix_filter": True},
            min_similarity=min_similarity,
        )
        filtered = score_source_with_backend(
            src_ids,
            src_matrix,
            target_index=target_index,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=None,
            nn_index=None,
            backend="sklearn",
            backend_index=filtered_index,
        )
        exhaustive = score_source_with_backend(
            src_ids,
            src_matrix,
            target_index=target_index,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=None,
            nn_index=build_target_nearest_neighbors(target_index, top_k=top_k),
            backend="sklearn",
            backend_index=None,
        )
        # Pairs at zero cosine can fill an exhaustive top-k but never share a
        # column, so they are the one set the filter cannot return.
        expected = {
            pair: score for pair, score in _pairs(exhaustive).items() if score > 0.0
        }
        _assert_same_pairs(_pairs(filtered), expected)


def test_sklearn_backend_prefix_filter_needs_min_similarity():
    target_index = _build_target_index()
    with pytest.raises(ValueError, match="min_similarity"):
        build_target_similarity_backend_index(
            target_index,
            backend="sklearn",
            top_k=2,
            max_candidates_per_source=None,
            backend_options={"prefix_filter": True},
        )


def test_filtered_scan_refuses_a_threshold_below_the_index_was_built_for():
    target_index = _build_target_index()
    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="sklearn",
        top_k=2,
        backend_options={"prefix_filter": True},
        min_similarity=0.5,
    )
    with pytest.raises(ValueError, match="sized for"):
        score_source_with_backend(
            ["s1"],
            target_index.vectorizer.transform(["acme limited"]),
            target_index=target_index,
            top_k=2,
            min_similarity=0.2,
            max_candidates_per_source=None,
            nn_index=None,
            backend="sklearn",
            backend_index=backend_index,
        )


def test_sklearn_backend_prefix_filter_scores_and_matches_unfiltered_top_hit():
    target_index = _build_target_index()

    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options={"prefix_filter": True},
        min_similarity=0.0,
    )
    assert backend_index is not None
    assert backend_index.prefix_filter is not None

    rows = score_source_with_backend(
        ["s1"],
        target_index.vectorizer.transform(["acme limited"]),
        target_index=target_index,
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=backend_index,
    )

    assert rows
    top = min(rows, key=lambda row: row["rank"])
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t1"


def test_sklearn_backend_prefix_filter_off_without_the_option():
    target_index = _build_target_index()

    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options=None,
        min_similarity=0.5,
    )
    assert backend_index is not None
    assert backend_index.prefix_filter is None


def test_sklearn_backend_prefix_filter_rejects_dense_representation():
    dense_target_index = TargetClusteringIndex(
        target_ids=["t1", "t2"],
        vectorizer=None,
        target_matrix=np.ones((2, 4)),
    )
    with pytest.raises(ValueError):
        build_target_similarity_backend_index(
            dense_target_index,
            backend="sklearn",
            top_k=2,
            max_candidates_per_source=None,
            backend_options={"prefix_filter": True},
            min_similarity=0.5,
        )


def test_prefix_filter_candidates_dense_fallback_matches_naive_union():
    """Regression for a real MemoryError hit on `gleif -> gb` WordPiece data:
    when a row's prefix columns' inverted lists are large relative to the
    target corpus, `prefix_filter_candidates` must still return the same
    result as the naive `np.unique(np.concatenate(...))` path -- just via
    the bounded dense-mask fallback instead.
    """
    n_targets = 40
    inverted_index = {
        0: np.arange(0, 35, dtype=np.int64),
        1: np.arange(3, 38, dtype=np.int64),
        2: np.asarray([7], dtype=np.int64),
    }
    token_rank = np.asarray([0, 1, 2], dtype=np.int64)
    prefix_index = PrefixFilterIndex(
        token_rank=token_rank,
        inverted_index=inverted_index,
        min_similarity=0.5,
        n_targets=n_targets,
    )

    # Weights 0.7, 0.7, 0.14 leave a remainder norm of 0.14 after the second
    # column, below the threshold, so the prefix is columns 0 and 1 only.
    # Their candidate lists (35 entries each) sum to 70, over n_targets=40,
    # forcing the dense-mask fallback path.
    candidates = prefix_filter_candidates(
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0.7, 0.7, 0.14]),
        prefix_index,
    )
    expected = np.unique(np.concatenate([inverted_index[0], inverted_index[1]]))
    assert candidates.tolist() == expected.tolist()
