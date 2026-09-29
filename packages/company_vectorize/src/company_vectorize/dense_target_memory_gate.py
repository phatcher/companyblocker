"""Memory-based refusal for the `"sklearn"` backend against a dense target matrix.

`"sklearn"`'s brute cosine search (`sparse_similarity.py`'s
`build_target_nearest_neighbors`) holds the whole target matrix resident and
takes its own normalized copy at query time -- fine for a sparse
representation, whose vectors are typically a few dozen nonzeros each
regardless of vocabulary size, but not for a dense one: a 12.9M-row target is
about 20GB at 384 dimensions and about 40GB at 768, before the doubling for
that normalized copy, against a 64GB machine. Unlike
`dense_vocabulary_gate.py`'s refusal, this is not a row-count threshold --
row count alone says nothing about a dense matrix's actual footprint, which
scales with dimensionality too -- so the check is a direct byte estimate
against the memory the caller reports as available, not a fixed ceiling.

This module knows nothing about `psutil`: the available-bytes figure is
passed in by the caller (`src/blocking/workflow.py`, which already depends on
`psutil`), keeping this package's own dependency surface unchanged.
"""

from __future__ import annotations

from scipy.sparse import issparse

from .clustering_contract import TargetClusteringIndex

# `sklearn`'s brute cosine search stores the target as (at least) float32 and
# takes one more normalized copy of it at query time -- see
# `sparse_similarity.py`'s `build_target_nearest_neighbors`. Both copies are
# counted; this is a deliberately conservative estimate, not a measured
# ceiling.
BYTES_PER_ELEMENT = 4
RESIDENT_COPIES = 2


class SklearnDenseTargetMemoryError(ValueError):
    """`sklearn` was asked to score a dense target matrix larger than available memory."""


def estimate_sklearn_dense_target_memory_bytes(
    target_rows: int, dimensions: int
) -> int:
    """Estimated bytes `sklearn`'s brute cosine search needs for this dense target.

    `target_rows * dimensions * BYTES_PER_ELEMENT * RESIDENT_COPIES`: the
    resident target matrix itself, plus the normalized copy `sklearn`'s
    cosine metric computes at query time, both at float32.
    """
    return target_rows * dimensions * BYTES_PER_ELEMENT * RESIDENT_COPIES


def ensure_sklearn_dense_target_memory_supported(
    *, target_rows: int, dimensions: int, available_bytes: int
) -> None:
    """Refuse `sklearn` against a dense target too large for `available_bytes`.

    Args:
        target_rows: How many rows the dense target matrix carries.
        dimensions: The target matrix's feature width.
        available_bytes: The machine's available memory, as the caller
            measured it (`psutil.virtual_memory().available` in
            `src/blocking/workflow.py`).

    Raises:
        SklearnDenseTargetMemoryError: The estimate from
            `estimate_sklearn_dense_target_memory_bytes()` exceeds
            `available_bytes`.
    """
    needed = estimate_sklearn_dense_target_memory_bytes(target_rows, dimensions)
    if needed <= available_bytes:
        return

    raise SklearnDenseTargetMemoryError(
        f"'sklearn' needs an estimated {needed:,} bytes to hold a dense "
        f"{target_rows:,}-row, {dimensions:,}-dimension target matrix "
        "resident plus its own normalized copy, but only "
        f"{available_bytes:,} bytes are available. Use the 'dense_brute' "
        "backend instead: it scans the target a block at a time in bounded "
        "memory."
    )


def ensure_sklearn_dense_target_index_memory_supported(
    target_index: TargetClusteringIndex, *, backend: str, available_bytes: int
) -> None:
    """`ensure_sklearn_dense_target_memory_supported()`, applied to a built index.

    A no-op unless `backend == "sklearn"` and `target_index.target_matrix` is
    dense -- the same self-contained-applicability shape
    `dense_vocabulary_gate.ensure_dense_vocabulary_scale_supported()` uses, so
    a caller can call this once, unconditionally, right after building the
    target index (`src/blocking/workflow.py`'s `_score_country`) without
    deciding backend/matrix-type applicability itself.
    """
    if backend != "sklearn" or issparse(target_index.target_matrix):
        return

    target_matrix = target_index.target_matrix
    dimensions = int(target_matrix.shape[1]) if target_matrix.ndim == 2 else 0
    ensure_sklearn_dense_target_memory_supported(
        target_rows=len(target_index.target_ids),
        dimensions=dimensions,
        available_bytes=available_bytes,
    )
