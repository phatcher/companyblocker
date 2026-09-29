"""Unit coverage for the `"dense_brute"` chunked exhaustive backend.

The correctness bar is direct agreement with `sklearn`'s brute cosine
`NearestNeighbors`, since `"dense_brute"` exists as an exact, bounded-memory
alternative to it, not an approximation. The memory-bound test mirrors
`test_prefix_filter_memory.py`'s `tracemalloc`-based approach: peak
allocation from the scan must stay flat as the target grows, since a block's
size is fixed independent of target size.
"""

from __future__ import annotations

import gc
import tracemalloc
from typing import cast

import numpy as np
import pytest
from company_vectorize.clustering_contract import TargetClusteringIndex
from company_vectorize.dense_brute_similarity import (
    DEFAULT_DENSE_BRUTE_BLOCK_BYTES,
    DEFAULT_DENSE_BRUTE_STORAGE_DTYPE,
    TargetDenseBruteIndex,
    build_target_dense_brute_index,
    score_source_with_dense_brute_backend,
)
from company_vectorize.sparse_similarity import (
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


def _target_index(n_targets: int, dim: int, *, seed: int = 0) -> TargetClusteringIndex:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_targets, dim)).astype(np.float32)
    target_ids = [f"t{i}" for i in range(n_targets)]
    return TargetClusteringIndex(
        target_ids=target_ids, vectorizer=None, target_matrix=matrix
    )


def _source_matrix(n_sources: int, dim: int, *, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_sources, dim)).astype(np.float32)


def _sklearn_reference(
    target_index: TargetClusteringIndex, src_matrix: np.ndarray, *, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=top_k)
    nn.fit(target_index.target_matrix)
    distances, indices = nn.kneighbors(src_matrix)
    return indices, 1.0 - distances


def _score_dense_brute(
    target_index: TargetClusteringIndex,
    src_matrix: np.ndarray,
    *,
    top_k: int,
    min_similarity: float = 0.0,
    max_candidates_per_source: int | None = None,
    backend_options: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    src_ids = [f"s{i}" for i in range(src_matrix.shape[0])]
    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="dense_brute",
        top_k=top_k,
        max_candidates_per_source=max_candidates_per_source,
        backend_options=backend_options,
    )
    return score_source_with_backend(
        src_ids,
        src_matrix,
        target_index=target_index,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        nn_index=None,
        backend="dense_brute",
        backend_index=backend_index,
    )


def _rows_by_source(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    return by_source


def test_defaults_are_a_conservative_bounded_footprint():
    assert DEFAULT_DENSE_BRUTE_STORAGE_DTYPE == "float16"
    assert DEFAULT_DENSE_BRUTE_BLOCK_BYTES == 256 * 1024 * 1024


def test_sparse_target_matrix_is_refused():
    dense_index = _target_index(10, 4)
    sparse_index = TargetClusteringIndex(
        target_ids=dense_index.target_ids,
        vectorizer=None,
        target_matrix=csr_matrix(dense_index.target_matrix),
    )
    with pytest.raises(ValueError, match="dense target matrix"):
        build_target_dense_brute_index(sparse_index)


def test_empty_target_returns_none_index():
    empty_index = TargetClusteringIndex(
        target_ids=[], vectorizer=None, target_matrix=np.empty((0, 4), dtype=np.float32)
    )
    assert build_target_dense_brute_index(empty_index) is None


def test_none_index_scores_to_no_rows():
    rows = score_source_with_dense_brute_backend(
        ["s1"],
        np.zeros((1, 4), dtype=np.float32),
        target_index=_target_index(5, 4),
        dense_brute_index=None,
        top_k=3,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )
    assert rows == []


def test_empty_source_chunk_scores_to_no_rows():
    target_index = _target_index(5, 4)
    dense_brute_index = build_target_dense_brute_index(target_index)
    rows = score_source_with_dense_brute_backend(
        [],
        np.empty((0, 4), dtype=np.float32),
        target_index=target_index,
        dense_brute_index=dense_brute_index,
        top_k=3,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )
    assert rows == []


def test_index_with_zero_targets_scores_to_no_rows():
    # Not reachable through build_target_dense_brute_index() (it returns None
    # for an empty target, per test_empty_target_returns_none_index above);
    # this exercises score_source_with_dense_brute_backend()'s own defensive
    # floor directly, for a caller that built one by hand.
    empty_target_index = TargetClusteringIndex(
        target_ids=[], vectorizer=None, target_matrix=np.empty((0, 4), dtype=np.float32)
    )
    dense_brute_index = TargetDenseBruteIndex(
        target_matrix=np.empty((0, 4), dtype=np.float32),
        block_rows=1,
        storage_dtype=np.dtype(np.float32),
    )
    rows = score_source_with_dense_brute_backend(
        ["s1"],
        np.zeros((1, 4), dtype=np.float32),
        target_index=empty_target_index,
        dense_brute_index=dense_brute_index,
        top_k=3,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )
    assert rows == []


@pytest.mark.parametrize("block_bytes", [64, 4096, 1 << 30])
def test_dense_brute_matches_sklearn_exactly_at_float32(block_bytes):
    target_index = _target_index(300, 12)
    src_matrix = _source_matrix(15, 12)
    top_k = 5

    rows = _score_dense_brute(
        target_index,
        src_matrix,
        top_k=top_k,
        backend_options={"storage_dtype": "float32", "block_bytes": block_bytes},
    )
    expected_indices, expected_scores = _sklearn_reference(
        target_index, src_matrix, top_k=top_k
    )

    by_source = _rows_by_source(rows)
    for src_index in range(src_matrix.shape[0]):
        got = sorted(by_source[f"s{src_index}"], key=lambda row: int(row["rank"]))
        got_targets = [row["target_id"] for row in got]
        got_scores = [float(row["similarity"]) for row in got]
        expected_targets = [
            target_index.target_ids[j] for j in expected_indices[src_index]
        ]
        assert got_targets == expected_targets
        assert got_scores == pytest.approx(list(expected_scores[src_index]), abs=1e-4)


@pytest.mark.parametrize(
    ("top_k", "min_similarity", "max_candidates_per_source"),
    [
        (5, 0.0, None),
        (5, 0.0, 2),
        (10, 0.5, None),
        (1, 0.0, None),
    ],
)
def test_dense_brute_honours_every_pruning_parameter(
    top_k, min_similarity, max_candidates_per_source
):
    target_index = _target_index(200, 8)
    src_matrix = _source_matrix(10, 8)

    dense_brute_rows = _score_dense_brute(
        target_index,
        src_matrix,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        backend_options={"storage_dtype": "float32"},
    )

    src_ids = [f"s{i}" for i in range(src_matrix.shape[0])]
    sklearn_rows = score_source_with_backend(
        src_ids,
        src_matrix,
        target_index=target_index,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        nn_index=None,
        backend="sklearn",
        backend_index=None,
    )

    def _key(row: dict[str, object]) -> tuple[str, int]:
        return (str(row["source_id"]), cast(int, row["rank"]))

    dense_brute_rows = sorted(dense_brute_rows, key=_key)
    sklearn_rows = sorted(sklearn_rows, key=_key)

    assert [row["source_id"] for row in dense_brute_rows] == [
        row["source_id"] for row in sklearn_rows
    ]
    assert [row["target_id"] for row in dense_brute_rows] == [
        row["target_id"] for row in sklearn_rows
    ]
    for got, expected in zip(dense_brute_rows, sklearn_rows, strict=True):
        assert got["similarity"] == pytest.approx(expected["similarity"], abs=1e-4)


def test_float16_storage_completes_and_ranks_its_own_top_match():
    # float16 is lossy (module docstring), so this only asserts the backend
    # completes and still finds a source row's near-exact duplicate target as
    # its top match, not bit-for-bit agreement with the float32/sklearn path.
    target_index = _target_index(150, 16)
    src_matrix = target_index.target_matrix[:5].copy()

    rows = _score_dense_brute(
        target_index,
        src_matrix,
        top_k=3,
        backend_options={"storage_dtype": "float16"},
    )
    by_source = _rows_by_source(rows)
    for src_index in range(5):
        top = min(by_source[f"s{src_index}"], key=lambda row: int(row["rank"]))
        assert top["target_id"] == target_index.target_ids[src_index]


def test_memmap_target_gives_the_same_result_as_in_memory(tmp_path):
    target_index = _target_index(120, 10)
    src_matrix = _source_matrix(8, 10)
    top_k = 4

    in_memory_rows = _score_dense_brute(
        target_index,
        src_matrix,
        top_k=top_k,
        backend_options={"storage_dtype": "float32"},
    )

    path = tmp_path / "target.dat"
    memmap = np.memmap(
        path, dtype=np.float32, mode="w+", shape=target_index.target_matrix.shape
    )
    memmap[:] = target_index.target_matrix
    memmap.flush()
    memmap_index = TargetClusteringIndex(
        target_ids=target_index.target_ids, vectorizer=None, target_matrix=memmap
    )
    backend_index = build_target_similarity_backend_index(
        memmap_index,
        backend="dense_brute",
        top_k=top_k,
        max_candidates_per_source=None,
        backend_options={"storage_dtype": "float32", "block_bytes": 2048},
    )
    assert isinstance(backend_index.dense_brute.target_matrix, np.memmap)
    memmap_rows = score_source_with_backend(
        [f"s{i}" for i in range(src_matrix.shape[0])],
        src_matrix,
        target_index=memmap_index,
        top_k=top_k,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="dense_brute",
        backend_index=backend_index,
    )
    del memmap, backend_index, memmap_index
    gc.collect()

    def _key(row: dict[str, object]) -> tuple[str, int]:
        return (str(row["source_id"]), cast(int, row["rank"]))

    in_memory_rows = sorted(in_memory_rows, key=_key)
    memmap_rows = sorted(memmap_rows, key=_key)
    assert [
        (row["source_id"], row["target_id"], row["rank"]) for row in in_memory_rows
    ] == [(row["source_id"], row["target_id"], row["rank"]) for row in memmap_rows]
    for got, expected in zip(memmap_rows, in_memory_rows, strict=True):
        assert got["similarity"] == pytest.approx(expected["similarity"], abs=1e-4)


def _peak_bytes(fn) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


@pytest.mark.integration
def test_peak_allocation_is_bounded_by_the_block_not_the_target():
    dim = 384
    block_bytes = 256 * 1024  # small enough to force many blocks either way
    small_target = _target_index(5_000, dim, seed=3)
    large_target = _target_index(50_000, dim, seed=3)
    src_matrix = _source_matrix(20, dim, seed=4)

    small_peak = _peak_bytes(
        lambda: _score_dense_brute(
            small_target,
            src_matrix,
            top_k=5,
            backend_options={"storage_dtype": "float32", "block_bytes": block_bytes},
        )
    )
    large_peak = _peak_bytes(
        lambda: _score_dense_brute(
            large_target,
            src_matrix,
            top_k=5,
            backend_options={"storage_dtype": "float32", "block_bytes": block_bytes},
        )
    )

    # 10x the target rows; a bound tied to the block rather than the target
    # keeps peak allocation close to flat. Allow 2x for the genuinely linear
    # parts (more target ids, more rows in the final result).
    assert large_peak <= small_peak * 2, (
        f"peak allocation scaled with target size: {small_peak} bytes for "
        f"5,000 targets, {large_peak} bytes for 50,000 targets"
    )


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"storage_dtype": "float64"}, "storage_dtype"),
        ({"block_bytes": 0}, "block_bytes"),
        ({"block_bytes": -1}, "block_bytes"),
        ({"block_bytes": "lots"}, "block_bytes"),
        ({"block_bytes": True}, "block_bytes"),
    ],
)
def test_unreadable_backend_options_are_rejected(options, match):
    with pytest.raises(ValueError, match=match):
        build_target_dense_brute_index(_target_index(10, 4), backend_options=options)
