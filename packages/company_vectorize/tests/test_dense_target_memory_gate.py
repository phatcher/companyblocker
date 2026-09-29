"""Unit coverage for the `"sklearn"`-on-dense-target memory refusal.

Everything here is a pure decision over `(target_rows, dimensions,
available_bytes)`, mirroring `test_dense_vocabulary_gate.py`'s approach: the
real `fr`-scale case is exercised by passing that row count and dimension
directly, no corpus is read.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from company_vectorize.clustering_contract import TargetClusteringIndex
from company_vectorize.dense_target_memory_gate import (
    BYTES_PER_ELEMENT,
    RESIDENT_COPIES,
    SklearnDenseTargetMemoryError,
    ensure_sklearn_dense_target_index_memory_supported,
    ensure_sklearn_dense_target_memory_supported,
    estimate_sklearn_dense_target_memory_bytes,
)
from scipy.sparse import csr_matrix

# fr's target-side row count, used verbatim so the refusal is
# checked at the scale it was written about.
FR_TARGET_ROWS = 12_900_000
SIXTY_FOUR_GB = 64 * 1024**3


def test_estimate_matches_the_documented_arithmetic():
    assert estimate_sklearn_dense_target_memory_bytes(1, 1) == (
        BYTES_PER_ELEMENT * RESIDENT_COPIES
    )
    assert estimate_sklearn_dense_target_memory_bytes(100, 10) == 100 * 10 * 4 * 2


def test_fr_scale_at_768_dimensions_is_refused_on_a_64gb_machine():
    with pytest.raises(SklearnDenseTargetMemoryError):
        ensure_sklearn_dense_target_memory_supported(
            target_rows=FR_TARGET_ROWS,
            dimensions=768,
            available_bytes=SIXTY_FOUR_GB,
        )


def test_refusal_message_states_both_figures_and_names_dense_brute():
    with pytest.raises(SklearnDenseTargetMemoryError) as excinfo:
        ensure_sklearn_dense_target_memory_supported(
            target_rows=FR_TARGET_ROWS,
            dimensions=768,
            available_bytes=SIXTY_FOUR_GB,
        )

    message = str(excinfo.value)
    needed = estimate_sklearn_dense_target_memory_bytes(FR_TARGET_ROWS, 768)
    assert f"{needed:,}" in message
    assert f"{SIXTY_FOUR_GB:,}" in message
    assert "dense_brute" in message
    # Only the fix is named, nothing else -- unlike the dense-vocabulary
    # gate's message, this one carries no representation/backend menu.
    assert "svd_rerank" not in message
    assert "tfidf" not in message


def test_a_small_dense_target_is_accepted():
    ensure_sklearn_dense_target_memory_supported(
        target_rows=1_000, dimensions=384, available_bytes=SIXTY_FOUR_GB
    )


def test_estimate_exactly_at_available_bytes_is_accepted():
    needed = estimate_sklearn_dense_target_memory_bytes(1_000, 384)
    ensure_sklearn_dense_target_memory_supported(
        target_rows=1_000, dimensions=384, available_bytes=needed
    )


def test_one_byte_short_is_refused():
    needed = estimate_sklearn_dense_target_memory_bytes(1_000, 384)
    with pytest.raises(SklearnDenseTargetMemoryError):
        ensure_sklearn_dense_target_memory_supported(
            target_rows=1_000, dimensions=384, available_bytes=needed - 1
        )


# -- ensure_sklearn_dense_target_index_memory_supported: the self-contained --
# -- wrapper `src/blocking/workflow.py` calls unconditionally -----------------


class _SizedIds:
    """A `len()`-only stand-in for `target_ids` at `fr` scale.

    Materializing 12.9M real id strings would cost real time and memory this
    decision (a pure function of `len(target_ids)` and `target_matrix.shape`)
    never reads past its length.
    """

    def __init__(self, count: int) -> None:
        self._count = count

    def __len__(self) -> int:
        return self._count


class _FakeDenseMatrix:
    """A `.ndim`/`.shape`-only stand-in for a dense target matrix at `fr` scale.

    `ensure_sklearn_dense_target_index_memory_supported()` only reads those
    two attributes plus what `scipy.sparse.issparse()` needs to say "not
    sparse"; a real `(12_900_000, 768)` float32 array would actually commit
    ~40GB, which this decision's own inputs never require allocating.
    """

    ndim = 2

    def __init__(self, n_targets: int, dim: int) -> None:
        self.shape = (n_targets, dim)


def _dense_target_index(n_targets: int, dim: int) -> TargetClusteringIndex:
    return TargetClusteringIndex(
        target_ids=cast(list[str], _SizedIds(n_targets)),
        vectorizer=None,
        target_matrix=cast(np.ndarray, _FakeDenseMatrix(n_targets, dim)),
    )


def test_index_wrapper_refuses_a_too_large_dense_sklearn_target():
    target_index = _dense_target_index(FR_TARGET_ROWS, 768)
    with pytest.raises(SklearnDenseTargetMemoryError):
        ensure_sklearn_dense_target_index_memory_supported(
            target_index, backend="sklearn", available_bytes=SIXTY_FOUR_GB
        )


def test_index_wrapper_is_a_noop_for_any_backend_but_sklearn():
    target_index = _dense_target_index(FR_TARGET_ROWS, 768)
    ensure_sklearn_dense_target_index_memory_supported(
        target_index, backend="dense_brute", available_bytes=0
    )
    ensure_sklearn_dense_target_index_memory_supported(
        target_index, backend="kmeans", available_bytes=0
    )


def test_index_wrapper_is_a_noop_for_a_sparse_target():
    sparse_index = TargetClusteringIndex(
        target_ids=["t1", "t2"],
        vectorizer=None,
        target_matrix=csr_matrix(np.zeros((2, 4), dtype=np.float32)),
    )
    ensure_sklearn_dense_target_index_memory_supported(
        sparse_index, backend="sklearn", available_bytes=0
    )


def test_index_wrapper_accepts_a_small_dense_sklearn_target():
    target_index = _dense_target_index(1_000, 384)
    ensure_sklearn_dense_target_index_memory_supported(
        target_index, backend="sklearn", available_bytes=SIXTY_FOUR_GB
    )
