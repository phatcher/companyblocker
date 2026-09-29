from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.sparse import csr_matrix, issparse

from .clustering_contract import TargetClusteringIndex
from .ranking import append_ranked_matches

# "lsh" is a sub-linear candidate-generation backend for the *sparse*
# representations only ("tfidf"/"wordpiece"/"sentencepiece"): dense cosine
# embeddings ("sbert") get a dense ANN index (HNSW, hnsw_similarity.py) instead,
# and centroid/density partitioning is the third sibling
# (partition_similarity.py's "kmeans"/"hdbscan"). All three sit behind the same
# TargetSimilarityBackendIndex selector alongside the exhaustive backends in
# sparse_similarity.py.
#
# Family chosen: MinHash + LSH banding (Broder 1997; Leskovec/Rajaraman/Ullman,
# "Mining of Massive Datasets" ch. 3) over the *set of nonzero vocabulary
# columns* of a row -- i.e. which TF-IDF/token-list terms a name contains, not
# their weights. Every sparse representation this package builds
# (tfidf_strategy.py, token_list_strategy.py for wordpiece/sentencepiece) is
# already a `scipy.sparse.csr_matrix` over a fitted vectorizer's vocabulary
# columns, so this one implementation covers all three representation-agnostic:
# it never looks at the vectorizer, only at `target_matrix`'s sparsity pattern.
# MinHash's "signature" is also literally the vocabulary this item's Why asks
# for ("define the LSH signature and bucket-index contract").
#
# This backend estimates JACCARD similarity via signature agreement, not the
# cosine similarity every other backend in this package reports -- the two
# metrics disagree on TF-IDF-weighted rows in general. That is accepted here:
# this backend is candidate generation only (per the item's scope note),
# producing an approximate score good enough to rank/threshold its own
# shortlist. Reranking that shortlist with an exact cosine score is out of
# scope for this module.
#
# Determinism: every hash function used here is pure integer arithmetic over
# vocabulary-column indices (always ints) seeded through
# `numpy.random.default_rng`, never Python's `hash()`, whose str/bytes hashing
# is salted per-process by `PYTHONHASHSEED`. The same `(target_matrix, params)`
# therefore produces byte-identical signatures and bucket assignments on any
# process or machine -- see `test_lsh_similarity.py`'s
# `test_signatures_are_identical_across_processes` for a literal
# separate-process assertion of this.

DEFAULT_LSH_NUM_PERM = 64
DEFAULT_LSH_NUM_BANDS = 16
DEFAULT_LSH_SEED = 42

# 32-bit Mersenne prime: the modulus for the `a * col + b mod prime` universal
# hash family used to build minhash signatures. Column indices seen in
# practice (vocabulary size) and the `[1, prime)` hash coefficients both stay
# well under 2**32, so `a * col` stays inside int64 (`numpy`'s signature dtype)
# with no overflow, for any vocabulary this package is expected to fit in
# memory as a `csr_matrix` at all.
_MINHASH_PRIME = 2_147_483_647

# Multiplier/modulus for folding one LSH band's minhash values into a single
# bucket key. Computed with Python's arbitrary-precision ints (never a fixed
# numpy dtype): folding several `_MINHASH_PRIME`-sized values is the one place
# a fixed-width accumulator could silently overflow and corrupt bucket
# assignment, so this step never uses one.
_BAND_KEY_MULTIPLIER = 1_000_003
_BAND_KEY_MODULUS = 2**61 - 1


@dataclass(frozen=True)
class MinHashLshParams:
    """Explicit, injectable parameters for the `"lsh"` similarity backend.

    Resolved from `backend_options` by `_resolve_lsh_params()`; every value
    defaults per this dataclass's field defaults rather than a magic number
    buried in the hashing code.

    Attributes:
        num_perm: Minhash signature length -- the number of independent
            `(a, b)` hash functions each row is hashed through. Higher values
            estimate Jaccard similarity more precisely (see
            `score_source_with_lsh_backend()`'s agreement-fraction estimator)
            at the cost of more hashing work per row.
        num_bands: Number of LSH bands the `num_perm`-length signature is
            split into for bucketing. Must divide `num_perm` evenly (each band
            gets `num_perm // num_bands` signature values). More bands (so
            fewer values per band) raise recall -- a pair needs to agree on
            just one band to become a candidate -- at the cost of more/larger
            buckets and looser candidate sets, the standard LSH banding
            recall/precision trade-off.
        seed: Seed for `numpy.random.default_rng()`, generating this index's
            `(a, b)` hash coefficient pairs deterministically. The same seed
            plus the same target/source matrices reproduces the same
            signatures and bucket assignment on any process or machine.
    """

    num_perm: int = DEFAULT_LSH_NUM_PERM
    num_bands: int = DEFAULT_LSH_NUM_BANDS
    seed: int = DEFAULT_LSH_SEED


def _resolve_int_option(options: dict[str, object], key: str, default: int) -> int:
    raw = options.get(key, default)
    return int(raw) if isinstance(raw, (int, float, str)) else default


def _resolve_lsh_params(backend_options: dict[str, object] | None) -> MinHashLshParams:
    options = backend_options or {}
    num_perm = _resolve_int_option(options, "num_perm", DEFAULT_LSH_NUM_PERM)
    num_bands = _resolve_int_option(options, "num_bands", DEFAULT_LSH_NUM_BANDS)
    seed = _resolve_int_option(options, "seed", DEFAULT_LSH_SEED)

    if num_perm <= 0:
        raise ValueError("num_perm must be greater than zero")
    if num_bands <= 0:
        raise ValueError("num_bands must be greater than zero")
    if num_perm % num_bands != 0:
        raise ValueError(
            f"num_perm ({num_perm}) must be evenly divisible by num_bands "
            f"({num_bands}), so every band gets the same number of signature "
            "values."
        )
    return MinHashLshParams(num_perm=num_perm, num_bands=num_bands, seed=seed)


def _hash_coefficients(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic `(a, b)` universal-hash coefficient pairs, one per minhash function.

    `numpy.random.default_rng(seed)` (PCG64) reproduces the same stream for a
    given integer seed on any process or platform -- unlike Python's built-in
    `hash()`, whose string/bytes hashing is salted per-process by
    `PYTHONHASHSEED` even when the seed itself is fixed.
    """
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MINHASH_PRIME, size=num_perm, dtype=np.int64)
    b = rng.integers(0, _MINHASH_PRIME, size=num_perm, dtype=np.int64)
    return a, b


def _minhash_signature_matrix(
    matrix: csr_matrix, *, a: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """Row-wise minhash signatures for every row of `matrix`.

    Each row is treated as the *set* of its nonzero vocabulary-column indices
    (which terms it contains), not their TF-IDF weights -- see the module
    docstring. A row with no nonzero columns (should not reach this backend;
    callers filter empty rows before scoring) gets a signature of all
    `_MINHASH_PRIME` sentinels, an impossible real hash value, so it never
    matches any bucket.

    Returns an `(n_rows, num_perm)` `int64` array.
    """
    n_rows = matrix.shape[0]
    num_perm = a.shape[0]
    indptr = matrix.indptr
    indices = matrix.indices
    signatures = np.full((n_rows, num_perm), _MINHASH_PRIME, dtype=np.int64)

    for row in range(n_rows):
        start, end = indptr[row], indptr[row + 1]
        cols = indices[start:end]
        if cols.size == 0:
            continue
        hashed = (
            a[np.newaxis, :] * cols[:, np.newaxis].astype(np.int64) + b[np.newaxis, :]
        ) % _MINHASH_PRIME
        signatures[row] = hashed.min(axis=0)

    return signatures


def _band_bucket_keys(signature_row: np.ndarray, *, num_bands: int) -> list[int]:
    """One deterministic bucket key per LSH band for a single row's signature."""
    band_width = signature_row.shape[0] // num_bands
    keys: list[int] = []
    for band in range(num_bands):
        start = band * band_width
        end = start + band_width
        key = 0
        for value in signature_row[start:end]:
            key = (key * _BAND_KEY_MULTIPLIER + int(value)) % _BAND_KEY_MODULUS
        keys.append(key)
    return keys


def _require_sparse(matrix: object, *, which: str) -> None:
    if not issparse(matrix):
        raise ValueError(
            f"lsh backend requires a sparse {which} matrix (the 'tfidf'/"
            "'wordpiece'/'sentencepiece' representations): it hashes "
            "vocabulary-column nonzero patterns, which a dense embedding like "
            "'sbert' does not have. Use the 'hnsw' backend for dense cosine "
            "embeddings instead."
        )


@dataclass(frozen=True)
class TargetLshIndex:
    """Fitted state for the `"lsh"` similarity backend.

    Built by `build_target_lsh_index()` and stored on
    `TargetSimilarityBackendIndex.lsh`; not normally constructed directly.

    Attributes:
        params: The resolved `MinHashLshParams` this index was built with.
            `score_source_with_lsh_backend()` reads hashing/banding parameters
            from here rather than re-resolving `backend_options`, so a source
            row can never be hashed with different parameters than the target
            index it is being matched against.
        target_signatures: `(n_targets, params.num_perm)` minhash signature
            matrix, aligned by position with `TargetClusteringIndex.target_ids`.
        hash_coefficients: The `(a, b)` coefficient pair used to build
            `target_signatures`, reused unchanged to hash source rows into the
            same hash space.
        buckets: One dict per band -- band index -> {bucket key -> sorted
            `np.ndarray` of target row indices sharing that band's key}.
    """

    params: MinHashLshParams
    target_signatures: np.ndarray
    hash_coefficients: tuple[np.ndarray, np.ndarray]
    buckets: list[dict[int, np.ndarray]]


def build_target_lsh_index(
    target_index: TargetClusteringIndex,
    *,
    backend_options: dict[str, object] | None = None,
) -> TargetLshIndex | None:
    """Build the `"lsh"` similarity-backend index for `target_index`.

    Returns `None` for an empty target (no rows), matching the other backend
    indexes' `None`-on-empty-target convention.

    Raises:
        ValueError: `target_index.target_matrix` is not sparse, or the
            resolved `MinHashLshParams` are invalid (see `_resolve_lsh_params`).
    """
    n_targets = len(target_index.target_ids)
    if n_targets == 0:
        return None
    _require_sparse(target_index.target_matrix, which="target")

    params = _resolve_lsh_params(backend_options)
    a, b = _hash_coefficients(params.num_perm, params.seed)
    # `_require_sparse()` above already confirmed this at runtime; `cast()`
    # narrows `TargetClusteringIndex.target_matrix`'s declared
    # `csr_matrix | np.ndarray` for mypy, which cannot see through that check.
    target_matrix = cast(csr_matrix, target_index.target_matrix).tocsr()
    signatures = _minhash_signature_matrix(target_matrix, a=a, b=b)

    raw_buckets: list[dict[int, list[int]]] = [{} for _ in range(params.num_bands)]
    for row in range(n_targets):
        for band, key in enumerate(
            _band_bucket_keys(signatures[row], num_bands=params.num_bands)
        ):
            raw_buckets[band].setdefault(key, []).append(row)

    buckets: list[dict[int, np.ndarray]] = [
        {key: np.asarray(rows, dtype=np.int64) for key, rows in band.items()}
        for band in raw_buckets
    ]
    return TargetLshIndex(
        params=params,
        target_signatures=signatures,
        hash_coefficients=(a, b),
        buckets=buckets,
    )


def score_source_with_lsh_backend(
    src_ids: list[str],
    src_matrix: csr_matrix,
    *,
    target_index: TargetClusteringIndex,
    lsh_index: TargetLshIndex | None,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
) -> list[dict[str, object]]:
    """Score one source chunk against `lsh_index`'s target buckets.

    Each source row is hashed with `lsh_index`'s own `(a, b)` coefficients and
    band parameters, then matched against every band's bucket it falls into;
    the union of matched target rows across bands is its candidate set. Each
    candidate is scored by minhash signature agreement (the fraction of
    `params.num_perm` hash functions the source and candidate signatures agree
    on), an unbiased estimator of Jaccard similarity between the two rows'
    vocabulary-column sets -- an approximate score for ranking/thresholding
    this backend's own shortlist, not the exact cosine score the other
    backends report. Reranking this shortlist with an exact score is a
    separate concern, out of scope here.

    Returns `[]` if `lsh_index` is `None` (no target index built, or an empty
    target) or `max(top_k, max_candidates_per_source)` resolves to `<= 0`;
    matches the other backends' behavior for those cases.
    """
    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    if lsh_index is None or max_per_source <= 0:
        return []
    _require_sparse(src_matrix, which="source")

    src_matrix = src_matrix.tocsr()
    a, b = lsh_index.hash_coefficients
    src_signatures = _minhash_signature_matrix(src_matrix, a=a, b=b)
    num_bands = lsh_index.params.num_bands

    rows: list[dict[str, object]] = []
    for src_index, src_id in enumerate(src_ids):
        signature_row = src_signatures[src_index]
        candidate_set: set[int] = set()
        for band, key in enumerate(
            _band_bucket_keys(signature_row, num_bands=num_bands)
        ):
            bucket = lsh_index.buckets[band].get(key)
            if bucket is not None:
                candidate_set.update(int(idx) for idx in bucket)
        if not candidate_set:
            continue

        candidate_indices = np.fromiter(
            candidate_set, dtype=np.int64, count=len(candidate_set)
        )
        candidate_signatures = lsh_index.target_signatures[candidate_indices]
        agreement = (candidate_signatures == signature_row[np.newaxis, :]).mean(axis=1)
        append_ranked_matches(
            rows,
            src_id=src_id,
            target_ids=target_index.target_ids,
            candidate_indices=candidate_indices,
            candidate_scores=agreement.astype(np.float64),
            min_similarity=min_similarity,
            max_per_source=max_per_source,
        )
    return rows
