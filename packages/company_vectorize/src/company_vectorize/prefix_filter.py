"""Prefix filtering for the `"sklearn"` backend: an exact pre-scoring gate on cosine.

Papadakis et al. 2020 (arXiv:1905.06167, section 5, the filtering family) is the
survey. The bound is L2AP's (Anastasiu and Karypis, "L2AP: Fast Cosine
Similarity Search with Prefix L-2 Norm Bounds", ICDE 2014), the successor to
All-Pairs' maximum-weight bound (Bayardo, Ma and Srikant, "Scaling Up All
Pairs Similarity Search", WWW 2007). Distinct from comparison cleaning in
`src/blocking`, which prunes candidate pairs after they are scored: this prunes
before, so a pair that cannot reach the run's `min_similarity` is never handed
to the similarity call at all.

Mechanism. Every vocabulary column gets one global rank, rarest first by target
document frequency, ties by column index. A row's terms are sorted by that rank,
and its prefix is the leading run of terms up to the first position where the
L2 norm of the terms still to come drops below `min_similarity`. A source row
is scored only against the target rows whose prefix shares a column with its
own, found through an inverted index built once over the target prefixes.

Why that is exact. Every row is a unit vector (the vectorizers this backend
accepts L2-normalize), so cosine is the dot product, and by Cauchy-Schwarz the
dot product over any subset of one row's terms is at most that subset's norm.
Two rows with disjoint prefixes share terms only outside their prefixes, and
because one global order sizes both, every shared term lies in the remainder of
whichever row's prefix ends earlier. That remainder's norm is below
`min_similarity`, so the pair scores below it and the scan would have dropped
it anyway. The bound reads nothing from the other side, which is what lets the
target index be built before any source row is seen. All-Pairs' bound sizes a
prefix from the other collection's largest weight per term instead, which the
streamed source side cannot supply at build time, and on tf-idf character
n-grams it is the looser of the two, since a short name gives a common n-gram a
large maximum weight.

The prefix is sized against `min_similarity` less `PREFIX_BOUND_SLACK`, so float
rounding in the scored dot product cannot readmit a pair the bound excluded at
equality. A row whose own norm is below that threshold gets an empty prefix,
which is right: against a unit vector it cannot reach the threshold. At
`min_similarity` zero every term is in the prefix and the gate admits any pair
sharing a column, which are the only pairs with a nonzero cosine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, issparse

PREFIX_BOUND_SLACK = 1e-6
"""Margin below `min_similarity` the prefix bound is sized against."""

PREFIX_FILTER_REPRESENTATIONS: frozenset[str] = frozenset(
    {"tfidf", "wordpiece", "sentencepiece"}
)
"""The representations whose rows are sparse unit vectors, which the bound needs."""


@dataclass(frozen=True)
class PrefixFilterIndex:
    """Fitted prefix-filter state for the `"sklearn"` similarity backend.

    Built by `build_prefix_filter_index()` and stored on
    `TargetSimilarityBackendIndex.prefix_filter`; not normally constructed
    directly.

    Attributes:
        token_rank: Rank per vocabulary column (`target_matrix.shape[1]`
            entries), ascending document frequency -- `token_rank[col] == 0`
            is the globally rarest token, ties broken by column index for
            determinism.
        inverted_index: Vocabulary column index -> sorted `np.ndarray` of
            target row indices whose prefix contains that column. Only
            columns that appear in at least one target row's prefix have an
            entry.
        min_similarity: The cosine threshold the prefixes were sized for. A
            scan at a lower threshold is not exact under this index.
        n_targets: `target_matrix.shape[0]` -- the fixed upper bound
            `prefix_filter_candidates()`'s dense fallback path allocates
            against (see its docstring).
    """

    token_rank: np.ndarray
    inverted_index: dict[int, np.ndarray]
    min_similarity: float
    n_targets: int


def _validate_min_similarity(min_similarity: float) -> float:
    value = float(min_similarity)
    if not (0.0 <= value <= 1.0):
        raise ValueError("min_similarity must be between 0 and 1")
    return value


def _prefix_bound_squared(min_similarity: float) -> float:
    """Squared remainder norm at or above which a term is still in the prefix.

    Zero when the slack swallows the threshold, so every term is kept.
    """
    bound = min_similarity - PREFIX_BOUND_SLACK
    return bound * bound if bound > 0.0 else 0.0


def _prefix_columns_for_row(
    col_indices: np.ndarray,
    col_values: np.ndarray,
    token_rank: np.ndarray,
    min_similarity: float,
) -> np.ndarray:
    """Return the leading rarity-sorted prefix of one row's columns.

    `col_indices` is the row's nonzero vocabulary columns in any order and
    `col_values` their weights in the same order.
    """
    col_indices = np.asarray(col_indices)
    if col_indices.size == 0:
        return col_indices
    order = np.argsort(token_rank[col_indices], kind="stable")
    sorted_cols = col_indices[order]
    weights_sq = np.square(np.asarray(col_values, dtype=np.float64)[order])
    remaining_sq = np.cumsum(weights_sq[::-1])[::-1]
    keep = remaining_sq >= _prefix_bound_squared(min_similarity)
    return sorted_cols[: int(np.count_nonzero(keep))]


def build_prefix_filter_index(
    target_matrix: csr_matrix, *, min_similarity: float
) -> PrefixFilterIndex | None:
    """Build the prefix-filter index for `target_matrix`, sized for `min_similarity`.

    Returns `None` for an empty target (no rows/columns) -- callers should
    treat that the same as "no prefix filtering available", matching the
    other backend indexes' `None`-on-empty-target convention.

    Raises:
        ValueError: `min_similarity` is outside `[0.0, 1.0]`, or `target_matrix`
            is not sparse (prefix filtering is a sparse/token-overlap
            technique; it has no defined meaning over a dense embedding --
            see `SbertClusteringStrategy`'s docstring for the representations
            this applies to).
    """
    min_similarity = _validate_min_similarity(min_similarity)
    if not issparse(target_matrix):
        raise ValueError(
            "prefix filtering requires a sparse target matrix (tfidf/wordpiece/"
            "sentencepiece representations); it has no defined meaning over a "
            "dense embedding like 'sbert'."
        )

    target_matrix = target_matrix.tocsr()
    n_targets, n_features = target_matrix.shape
    if n_targets == 0 or n_features == 0:
        return None

    indptr = target_matrix.indptr
    indices = target_matrix.indices
    doc_freq = np.bincount(indices, minlength=n_features)

    # Rank by ascending document frequency (rarest first); tie-break by
    # column index so the ordering is deterministic across runs.
    order = np.lexsort((np.arange(n_features), doc_freq))
    token_rank = np.empty(n_features, dtype=np.int64)
    token_rank[order] = np.arange(n_features)

    nnz = int(indices.size)
    if nnz == 0:
        return PrefixFilterIndex(
            token_rank=token_rank,
            inverted_index={},
            min_similarity=min_similarity,
            n_targets=n_targets,
        )

    # Every nonzero, ordered by row then by rank within the row. Rows keep
    # their CSR extents, so `indptr` still delimits each row after the sort.
    row_of = np.repeat(np.arange(n_targets), np.diff(indptr))
    perm = np.lexsort((token_rank[indices], row_of))
    cols = indices[perm]
    rows = row_of[perm]
    weights_sq = np.square(np.asarray(target_matrix.data, dtype=np.float64)[perm])

    # Remainder norm at each position: the row's total less what precedes
    # the position within the row.
    exclusive = np.cumsum(weights_sq) - weights_sq
    row_start = exclusive[np.minimum(indptr[:-1], nnz - 1)]
    row_total = np.bincount(rows, weights=weights_sq, minlength=n_targets)
    remaining_sq = row_total[rows] - (exclusive - row_start[rows])
    keep = remaining_sq >= _prefix_bound_squared(min_similarity)

    kept_cols = cols[keep]
    kept_rows = rows[keep]
    by_col = np.argsort(kept_cols, kind="stable")
    kept_cols = kept_cols[by_col]
    kept_rows = kept_rows[by_col]
    columns, starts = np.unique(kept_cols, return_index=True)
    inverted_index = {
        int(col): np.asarray(row_list, dtype=np.int64)
        for col, row_list in zip(columns, np.split(kept_rows, starts[1:]))
    }
    return PrefixFilterIndex(
        token_rank=token_rank,
        inverted_index=inverted_index,
        min_similarity=min_similarity,
        n_targets=n_targets,
    )


def prefix_filter_candidates(
    col_indices: np.ndarray, col_values: np.ndarray, prefix_index: PrefixFilterIndex
) -> np.ndarray:
    """Return the target row indices sharing a prefix token with one source row.

    The source row is given as its nonzero vocabulary columns, `col_indices`,
    and their weights, `col_values`, in the same order; its prefix is sized
    by the index's own `min_similarity`.

    An empty result means no target prefix shares a column with this row's
    prefix, so no target can reach the threshold against it.

    This assumes a row's rarest tokens still have a small-relative-to-target
    document frequency -- true for TF-IDF's large sparse n-gram vocabulary,
    but not guaranteed for a small, dense subword vocabulary (WordPiece/
    SentencePiece): with only a few thousand possible tokens spread across
    millions of target rows, even a row's globally-rarest token can still
    match a large fraction of all targets. Concatenating several such large
    per-column candidate lists before deduping can transiently need many
    times the corpus size in memory -- confirmed against real `gleif -> gb`
    WordPiece data, where this raised `MemoryError`. Below, once the
    prospective concatenated size exceeds `n_targets` (the true deduped
    union can never exceed it, so anything past that point is pure
    duplication), switch to a boolean membership mask over the target rows
    instead -- a hard `n_targets`-byte memory ceiling regardless of how
    large or how many candidate lists there are, at the cost of paying that
    full `n_targets`-sized allocation even when the real union is small.
    """
    prefix_cols = _prefix_columns_for_row(
        col_indices, col_values, prefix_index.token_rank, prefix_index.min_similarity
    )
    candidate_sets = [
        prefix_index.inverted_index[int(col)]
        for col in prefix_cols
        if int(col) in prefix_index.inverted_index
    ]
    if not candidate_sets:
        return np.empty(0, dtype=np.int64)

    total_size = sum(arr.size for arr in candidate_sets)
    if total_size <= prefix_index.n_targets:
        return np.unique(np.concatenate(candidate_sets))

    mask = np.zeros(prefix_index.n_targets, dtype=bool)
    for arr in candidate_sets:
        mask[arr] = True
    return np.flatnonzero(mask)
