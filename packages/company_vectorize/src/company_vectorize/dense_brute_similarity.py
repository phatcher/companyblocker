from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import normalize

from .clustering_contract import TargetClusteringIndex
from .ranking import append_ranked_match_block

# An exact, CPU-only exhaustive backend for a *dense* target matrix
# (the "sbert" representation, and any future dense embedding), scanning the
# target a block at a time with a running top-k rather than the "sklearn"
# backend's whole-matrix-resident `NearestNeighbors` fit plus its own
# normalized copy. That combination is unaffordable at `fr` scale: about 20GB
# for its 12.9M targets at 384 dimensions, about 40GB at 768, before doubling
# for sklearn's own normalized copy, against a 64GB machine.
#
# Memory bound: the target is held in `storage_dtype` (`"float16"` by default,
# half of `"float32"`'s footprint), and each block is converted to float32 --
# numpy has no usable float16 multiply, confirmed directly: a 1,000 x 50,000
# multiply at 384 dimensions took 126s in float16 against 0.11s in float32,
# and 0.23s converting each block to float32 first. Peak allocation from the
# scan itself is bounded by one block plus the source chunk and the running
# top-k, independent of how large the target is; the one-time cost of storing
# the whole target at `storage_dtype` is a separate, deliberate, one-time
# build-time choice, not part of that per-block bound.
#
# CPU only: the GPU gain measured on one block (0.046s on a GTX 1080 Ti versus
# 0.11s for a CPU float32 multiply) is not enough to widen this item's scope,
# and this backend is meant as the exact reference a faster dense index is
# checked against.

DEFAULT_DENSE_BRUTE_STORAGE_DTYPE = "float16"
DEFAULT_DENSE_BRUTE_BLOCK_BYTES = 256 * 1024 * 1024

STORAGE_DTYPE_OPTION = "storage_dtype"
BLOCK_BYTES_OPTION = "block_bytes"

_STORAGE_DTYPES: dict[str, np.dtype] = {
    "float16": np.dtype(np.float16),
    "float32": np.dtype(np.float32),
}

# Every block is converted to this dtype for the one multiply that scores it
# against the source chunk -- float16 has no usable multiply (see module
# docstring), regardless of `storage_dtype`.
_SCAN_DTYPE = np.dtype(np.float32)


def _resolve_storage_dtype(options: dict[str, object]) -> np.dtype:
    raw = options.get(STORAGE_DTYPE_OPTION, DEFAULT_DENSE_BRUTE_STORAGE_DTYPE)
    name = str(raw).strip().lower()
    dtype = _STORAGE_DTYPES.get(name)
    if dtype is None:
        raise ValueError(
            f"dense_brute backend option '{STORAGE_DTYPE_OPTION}' must be one "
            f"of {sorted(_STORAGE_DTYPES)}, got {raw!r}."
        )
    return dtype


def _resolve_block_bytes(options: dict[str, object]) -> int:
    raw = options.get(BLOCK_BYTES_OPTION, DEFAULT_DENSE_BRUTE_BLOCK_BYTES)
    block_bytes = -1
    if isinstance(raw, bool):
        raw = None
    elif isinstance(raw, (int, float, str)):
        try:
            block_bytes = int(raw)
        except (TypeError, ValueError):
            block_bytes = -1
    if block_bytes <= 0:
        raise ValueError(
            f"dense_brute backend option '{BLOCK_BYTES_OPTION}' must be a "
            f"positive integer byte count, got {raw!r}."
        )
    return block_bytes


@dataclass(frozen=True)
class TargetDenseBruteIndex:
    """Fitted state for the `"dense_brute"` similarity backend.

    Built by `build_target_dense_brute_index()` and stored on
    `TargetSimilarityBackendIndex.dense_brute`; not normally constructed
    directly.

    Attributes:
        target_matrix: The target embedding matrix held at `storage_dtype`
            (a plain `numpy.ndarray` or a `numpy.memmap`, whichever
            `TargetClusteringIndex.target_matrix` was), aligned by position
            with `TargetClusteringIndex.target_ids`.
        block_rows: How many target rows one scan block holds, resolved from
            the `"block_bytes"` backend option against `target_matrix`'s
            feature width at scan dtype (float32).
        storage_dtype: The resolved `"storage_dtype"` backend option
            (`"float16"` or `"float32"`) `target_matrix` is held at.
    """

    target_matrix: np.ndarray
    block_rows: int
    storage_dtype: np.dtype


def build_target_dense_brute_index(
    target_index: TargetClusteringIndex,
    *,
    backend_options: dict[str, object] | None = None,
) -> TargetDenseBruteIndex | None:
    """Build the `"dense_brute"` similarity-backend index for `target_index`.

    Returns `None` for an empty target (no rows), matching the other backend
    indexes' `None`-on-empty-target convention.

    Raises:
        ValueError: `target_index.target_matrix` is sparse -- this backend is
            for dense representations only (see `_score_source_with_sklearn`
            /the `"sklearn"` backend for sparse ones) -- or either backend
            option cannot be read (`_resolve_storage_dtype`/
            `_resolve_block_bytes`).
    """
    n_targets = len(target_index.target_ids)
    if n_targets == 0:
        return None
    if issparse(target_index.target_matrix):
        raise ValueError(
            "dense_brute backend requires a dense target matrix (for example "
            "the 'sbert' representation); sparse representations like "
            "tfidf/wordpiece/sentencepiece are not supported by this backend "
            "-- their sparsity is scored exactly by the 'sklearn' backend "
            "instead, which gets free rejections from feature overlap that a "
            "dense embedding has no equivalent of."
        )

    options = backend_options or {}
    storage_dtype = _resolve_storage_dtype(options)
    block_bytes = _resolve_block_bytes(options)

    # `copy=False`, called directly on `target_matrix` rather than through
    # `np.asarray()` first: a target already at `storage_dtype` (including a
    # `numpy.memmap`) is returned unchanged -- the same object, the same
    # `numpy.memmap` subclass, backed by the same mapped file -- rather than
    # materialized. `np.asarray()` defaults to `subok=False` and would strip
    # a memmap down to a plain, fully-materializing `ndarray` view even when
    # no copy is otherwise needed, defeating exactly the bound this backend
    # exists to keep.
    stored = target_index.target_matrix.astype(storage_dtype, copy=False)
    n_features = int(stored.shape[1]) if stored.ndim == 2 else 0
    bytes_per_row_at_scan_dtype = max(1, n_features * _SCAN_DTYPE.itemsize)
    block_rows = max(1, min(n_targets, block_bytes // bytes_per_row_at_scan_dtype))

    return TargetDenseBruteIndex(
        target_matrix=stored, block_rows=int(block_rows), storage_dtype=storage_dtype
    )


def score_source_with_dense_brute_backend(
    src_ids: list[str],
    src_matrix,
    *,
    target_index: TargetClusteringIndex,
    dense_brute_index: TargetDenseBruteIndex | None,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
) -> list[dict[str, object]]:
    """Score one source chunk against `dense_brute_index`'s target, one block at a time.

    Cosine similarity, computed exactly: both the source chunk and each
    target block are L2-normalized (`sklearn.preprocessing.normalize`, which
    leaves an all-zero row as zero rather than dividing by zero) before the
    one multiply that scores them, the same metric `"sklearn"`'s
    `NearestNeighbors(metric="cosine")` computes, regardless of whether the
    representation that produced them already normalizes its own output.

    A running top-`k` is merged across blocks (`k = min(top_k,
    max_candidates_per_source, n_targets)`), so peak allocation from the scan
    is the current block, the source chunk, and this `(n_src, k)` running
    state -- never the whole target. `min_similarity`/`max_candidates_per_source`
    are applied by `append_ranked_matches()` once the running top-k is final,
    identically to every other backend in this package.

    Returns `[]` if `dense_brute_index` is `None` (no target index built, or
    an empty target), the source chunk is empty, or
    `max(top_k, max_candidates_per_source)` resolves to `<= 0`.
    """
    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    if dense_brute_index is None or max_per_source <= 0:
        return []

    src = np.asarray(src_matrix, dtype=_SCAN_DTYPE)
    if src.ndim != 2 or src.shape[0] == 0:
        return []
    src = normalize(src, norm="l2", copy=False)

    n_src = src.shape[0]
    n_targets = len(target_index.target_ids)
    k = min(max_per_source, n_targets)
    if k <= 0:
        return []

    best_scores = np.full((n_src, k), -np.inf, dtype=_SCAN_DTYPE)
    best_indices = np.full((n_src, k), -1, dtype=np.int64)

    stored = dense_brute_index.target_matrix
    block_rows = dense_brute_index.block_rows
    for start in range(0, n_targets, block_rows):
        end = min(start + block_rows, n_targets)
        block = np.asarray(stored[start:end], dtype=_SCAN_DTYPE)
        block = normalize(block, norm="l2", copy=False)
        sims = src @ block.T

        block_width = end - start
        block_k = min(k, block_width)
        if block_k < block_width:
            local_top = np.argpartition(sims, -block_k, axis=1)[:, -block_k:]
        else:
            local_top = np.tile(np.arange(block_width), (n_src, 1))
        local_scores = np.take_along_axis(sims, local_top, axis=1)
        global_indices = local_top.astype(np.int64) + start

        merged_scores = np.concatenate([best_scores, local_scores], axis=1)
        merged_indices = np.concatenate([best_indices, global_indices], axis=1)
        keep = np.argpartition(merged_scores, -k, axis=1)[:, -k:]
        best_scores = np.take_along_axis(merged_scores, keep, axis=1)
        best_indices = np.take_along_axis(merged_indices, keep, axis=1)

    rows: list[dict[str, object]] = []
    # A slot the running top-k never filled carries index -1; scoring it
    # `-inf` drops it, as skipping it by mask did a row at a time.
    scores = np.where(best_indices >= 0, best_scores.astype(np.float64), -np.inf)
    append_ranked_match_block(
        rows,
        src_ids=src_ids,
        target_ids=target_index.target_ids,
        candidate_indices=np.where(best_indices >= 0, best_indices, 0),
        candidate_scores=scores,
        min_similarity=min_similarity,
        max_per_source=max_per_source,
    )
    return rows
