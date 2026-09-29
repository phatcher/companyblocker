from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .clustering_contract import SimilarityBackendIndex, TargetClusteringIndex
from .dense_brute_similarity import (
    TargetDenseBruteIndex,
    build_target_dense_brute_index,
    score_source_with_dense_brute_backend,
)
from .hnsw_similarity import (
    TargetHnswIndex,
    build_target_hnsw_index,
    score_source_with_hnsw_backend,
)
from .lsh_similarity import (
    TargetLshIndex,
    build_target_lsh_index,
    score_source_with_lsh_backend,
)
from .partition_similarity import (
    TargetPartitionIndex,
    build_target_partition_index,
    score_source_with_partition_backend,
)
from .prefix_filter import (
    PrefixFilterIndex,
    build_prefix_filter_index,
    prefix_filter_candidates,
)
from .ranking import append_ranked_match_block, append_ranked_matches

# "svd_rerank" reduces the sparse TF-IDF/token matrix to a dense, low-dimensional
# projection (TruncatedSVD) before running the neighbor search, then reranks the
# shortlist with an exact sparse cosine score. This is a real constant-factor
# speedup (a 128-dim dense comparison is far cheaper than a sparse comparison over
# the full vocabulary) but NOT a sub-linear approximate-nearest-neighbor index:
# sklearn's NearestNeighbors has no ball_tree/kd_tree support for metric="cosine",
# so algorithm="auto" always resolves to "brute" here (confirmed directly:
# NearestNeighbors(metric="cosine", algorithm="auto")._fit_method == "brute"),
# meaning every query still scans all targets, just in the cheaper reduced space.
# It also trades away recall the plain "sklearn" backend doesn't: a true nearest
# neighbor whose reduced-space rank falls outside svd_candidates is permanently
# missed by the rerank step. A real sub-linear index (LSH/HNSW/IVF) is the `lsh`
# and `hnsw` backends' job; don't read "svd_rerank" as that.


@dataclass(frozen=True)
class TargetSvdRerankIndex:
    """Fitted state for the `"svd_rerank"` similarity backend.

    Built by `build_target_similarity_backend_index()` and stored on
    `TargetSimilarityBackendIndex.svd_rerank`; not normally constructed
    directly.

    Attributes:
        neighbor_index: `sklearn.neighbors.NearestNeighbors` (cosine) fitted
            over `projected_target`, used to shortlist candidates in the
            reduced SVD space.
        svd: The fitted `TruncatedSVD` projector, reused via `.transform()`
            to project source vectors into the same reduced space.
        projected_target: L2-normalized dense target matrix in the reduced
            SVD space (`n_targets x n_components`).
        candidate_count: Number of candidates retrieved per source row from
            `neighbor_index` before exact reranking -- the resolved
            `svd_candidates` backend option, clamped to `[1, n_targets]`.
    """

    neighbor_index: NearestNeighbors
    svd: TruncatedSVD
    projected_target: np.ndarray
    candidate_count: int


@dataclass(frozen=True)
class TargetSimilarityBackendIndex(SimilarityBackendIndex):
    """Concrete `SimilarityBackendIndex` used by this module's backends.

    Attributes:
        sklearn_nn: Fitted `NearestNeighbors` index when
            `backend == "sklearn"`; `None` otherwise. Narrows
            `SimilarityBackendIndex.sklearn_nn` to a concrete type.
        svd_rerank: Fitted `TargetSvdRerankIndex` when
            `backend == "svd_rerank"`; `None` otherwise. Narrows
            `SimilarityBackendIndex.svd_rerank` to a concrete type.
        partition: Fitted `TargetPartitionIndex` (`partition_similarity.py`)
            when `backend` is `"kmeans"` or `"hdbscan"`; `None` otherwise.
        prefix_filter: Fitted `PrefixFilterIndex` (`prefix_filter.py`) when
            `backend == "sklearn"` and the `"prefix_filter"` backend option
            was on, sized for the `min_similarity` the index was built with;
            `None` otherwise (including when `backend == "sklearn"` but the
            option was off -- the unfiltered brute-force path).
        lsh: Fitted `TargetLshIndex` (`lsh_similarity.py`) when
            `backend == "lsh"`; `None` otherwise.
        dense_brute: Fitted `TargetDenseBruteIndex` (`dense_brute_similarity.py`)
            when `backend == "dense_brute"`; `None` otherwise.
        hnsw: Fitted `TargetHnswIndex` (`hnsw_similarity.py`) when
            `backend == "hnsw"`; `None` otherwise.
    """

    sklearn_nn: NearestNeighbors | None = None
    svd_rerank: TargetSvdRerankIndex | None = None
    partition: TargetPartitionIndex | None = None
    prefix_filter: PrefixFilterIndex | None = None
    lsh: TargetLshIndex | None = None
    dense_brute: TargetDenseBruteIndex | None = None
    hnsw: TargetHnswIndex | None = None


_SPARSE_DOT_TOPN_WARNED = False

# Working-set budget for one batched candidate-pair multiply in
# `_score_source_with_sklearn_prefix_filtered`. Peak allocation of that multiply
# is linear in (candidate pairs x mean nonzeros per row of the two matrices), at
# a measured ~24 bytes per pair-nonzero -- 24.2-26.0 bytes across source/target
# row densities of 3-60 nonzeros and 50,000-1,000,000 pairs, i.e. flat in both
# dimensions. `_PAIR_SCORE_BYTES_PER_NONZERO` rounds that up for margin; the
# budget is converted to a pair count per call against the matrices' own
# densities rather than fixed, because a fixed pair count would re-make the
# assumption this whole path got wrong: that row density stays low.
_PAIR_SCORE_BYTE_BUDGET = 256 * 1024 * 1024
_PAIR_SCORE_BYTES_PER_NONZERO = 32
_MIN_PAIRS_PER_SCORE_BATCH = 1024


def _mean_row_nonzeros(matrix: csr_matrix) -> float:
    n_rows = int(matrix.shape[0])
    if n_rows == 0:
        return 0.0
    return float(matrix.nnz) / n_rows


def _resolve_pair_batch_size(src_matrix: csr_matrix, target_matrix: csr_matrix) -> int:
    """Candidate pairs to score per batched multiply for these two matrices.

    Sized so one batch's working set stays near `_PAIR_SCORE_BYTE_BUDGET`
    given the matrices' actual mean row densities, floored at
    `_MIN_PAIRS_PER_SCORE_BATCH` so a pathologically dense pair of matrices
    still makes progress a batch at a time rather than degenerating into a
    per-pair loop.
    """
    per_pair_bytes = _PAIR_SCORE_BYTES_PER_NONZERO * max(
        1.0, _mean_row_nonzeros(src_matrix) + _mean_row_nonzeros(target_matrix)
    )
    return max(
        _MIN_PAIRS_PER_SCORE_BATCH, int(_PAIR_SCORE_BYTE_BUDGET / per_pair_bytes)
    )


def _score_candidate_pairs(
    src_matrix: csr_matrix,
    target_matrix: csr_matrix,
    pair_src: np.ndarray,
    pair_tgt: np.ndarray,
    *,
    pair_batch_size: int,
) -> np.ndarray:
    """Exact cosine similarity for each `(pair_src[i], pair_tgt[i])` row pair.

    Both matrices are L2-normalized by the vectorizers this backend accepts,
    so the row-wise dot product already equals cosine similarity. Evaluated
    in slices of at most `pair_batch_size` pairs: the caller already bounds
    how many pairs it accumulates before calling, but a *single* source row's
    candidate set can exceed the budget on its own (see
    `prefix_filter_candidates()`'s dense-mask fallback), and that row still
    has to be scored in one call to keep its ranking exact.
    """
    total = int(pair_src.size)
    if total <= pair_batch_size:
        return np.asarray(
            src_matrix[pair_src].multiply(target_matrix[pair_tgt]).sum(axis=1)
        ).ravel()

    scores = np.empty(total, dtype=np.float64)
    for start in range(0, total, pair_batch_size):
        end = min(start + pair_batch_size, total)
        scores[start:end] = np.asarray(
            src_matrix[pair_src[start:end]]
            .multiply(target_matrix[pair_tgt[start:end]])
            .sum(axis=1)
        ).ravel()
    return scores


def _flush_prefix_filtered_batch(
    rows: list[dict[str, object]],
    batch: list[tuple[int, np.ndarray]],
    *,
    src_ids: list[str],
    src_matrix: csr_matrix,
    target_index: TargetClusteringIndex,
    pair_batch_size: int,
    min_similarity: float,
    max_per_source: int,
) -> None:
    """Score one accumulated sub-batch and append its ranked matches to `rows`.

    The sub-batch holds `(source row index, candidate target rows)` entries. `rows` is
    mutated in place, and `batch` is left for the caller to clear.
    """
    if not batch:
        return

    pair_src = np.concatenate(
        [
            np.full(candidates.size, src_index, dtype=np.int64)
            for src_index, candidates in batch
        ]
    )
    pair_tgt = np.concatenate([candidates for _, candidates in batch])
    pair_scores = _score_candidate_pairs(
        src_matrix,
        target_index.target_matrix,
        pair_src,
        pair_tgt,
        pair_batch_size=pair_batch_size,
    )

    offset = 0
    for src_index, candidates in batch:
        count = int(candidates.size)
        append_ranked_matches(
            rows,
            src_id=src_ids[src_index],
            target_ids=target_index.target_ids,
            candidate_indices=candidates,
            candidate_scores=np.asarray(
                pair_scores[offset : offset + count], dtype=np.float64
            ),
            min_similarity=min_similarity,
            max_per_source=max_per_source,
        )
        offset += count


def build_target_nearest_neighbors(
    target_index: TargetClusteringIndex,
    *,
    top_k: int,
    max_candidates_per_source: int | None = None,
) -> NearestNeighbors | None:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    n_targets = len(target_index.target_ids)
    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    if n_targets == 0 or max_per_source <= 0:
        return None

    n_neighbors = min(max_per_source, n_targets)
    nn = NearestNeighbors(
        metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1
    )
    nn.fit(target_index.target_matrix)
    return nn


DEFAULT_SVD_RERANK_CANDIDATES = 200
DEFAULT_SVD_RERANK_DIMENSIONS = 128


def build_target_similarity_backend_index(
    target_index: TargetClusteringIndex,
    *,
    backend: str,
    top_k: int,
    max_candidates_per_source: int | None = None,
    backend_options: dict[str, object] | None = None,
    min_similarity: float | None = None,
) -> TargetSimilarityBackendIndex | None:
    if backend == "sklearn":
        sklearn_nn = build_target_nearest_neighbors(
            target_index,
            top_k=top_k,
            max_candidates_per_source=max_candidates_per_source,
        )
        prefix_filter_index = None
        options = backend_options or {}
        if bool(options.get("prefix_filter", False)):
            if min_similarity is None:
                raise ValueError(
                    "prefix filtering sizes its prefixes by the run's "
                    "min_similarity; pass min_similarity to build the index."
                )
            prefix_filter_index = build_prefix_filter_index(
                target_index.target_matrix, min_similarity=min_similarity
            )
        return TargetSimilarityBackendIndex(
            backend=backend,
            sklearn_nn=sklearn_nn,
            prefix_filter=prefix_filter_index,
        )

    if backend == "sparse_dot_topn":
        return TargetSimilarityBackendIndex(backend=backend)

    if backend == "svd_rerank":
        options = backend_options or {}
        raw_svd_candidates = options.get(
            "svd_candidates", DEFAULT_SVD_RERANK_CANDIDATES
        )
        raw_svd_dimensions = options.get(
            "svd_dimensions", DEFAULT_SVD_RERANK_DIMENSIONS
        )
        svd_candidates = (
            int(raw_svd_candidates)
            if isinstance(raw_svd_candidates, (int, float, str))
            else DEFAULT_SVD_RERANK_CANDIDATES
        )
        svd_dimensions = (
            int(raw_svd_dimensions)
            if isinstance(raw_svd_dimensions, (int, float, str))
            else DEFAULT_SVD_RERANK_DIMENSIONS
        )
        n_targets = len(target_index.target_ids)
        if n_targets == 0:
            return TargetSimilarityBackendIndex(backend=backend)

        svd_candidates = max(1, min(svd_candidates, n_targets))
        n_features = int(target_index.target_matrix.shape[1])
        max_components = max(1, min(n_features - 1, n_targets - 1))
        n_components = max(1, min(svd_dimensions, max_components))

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        projected_target = svd.fit_transform(target_index.target_matrix)
        projected_target = normalize(projected_target, norm="l2", copy=False)

        neighbor_index = NearestNeighbors(
            metric="cosine",
            algorithm="auto",
            n_neighbors=svd_candidates,
            n_jobs=-1,
        )
        neighbor_index.fit(projected_target)
        return TargetSimilarityBackendIndex(
            backend=backend,
            svd_rerank=TargetSvdRerankIndex(
                neighbor_index=neighbor_index,
                svd=svd,
                projected_target=projected_target,
                candidate_count=svd_candidates,
            ),
        )

    if backend in ("kmeans", "hdbscan"):
        partition = build_target_partition_index(
            target_index, backend=backend, backend_options=backend_options
        )
        return TargetSimilarityBackendIndex(backend=backend, partition=partition)

    if backend == "lsh":
        lsh = build_target_lsh_index(target_index, backend_options=backend_options)
        return TargetSimilarityBackendIndex(backend=backend, lsh=lsh)

    if backend == "dense_brute":
        dense_brute = build_target_dense_brute_index(
            target_index, backend_options=backend_options
        )
        return TargetSimilarityBackendIndex(backend=backend, dense_brute=dense_brute)

    if backend == "hnsw":
        hnsw = build_target_hnsw_index(target_index, backend_options=backend_options)
        return TargetSimilarityBackendIndex(backend=backend, hnsw=hnsw)

    raise ValueError(f"Unsupported similarity backend: {backend}")


def score_source_with_backend(
    src_ids: list[str],
    src_matrix: csr_matrix,
    *,
    target_index: TargetClusteringIndex,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
    nn_index: NearestNeighbors | None,
    backend: str,
    backend_index: TargetSimilarityBackendIndex | None,
) -> list[dict[str, object]]:
    if backend == "sklearn":
        return _score_source_with_sklearn(
            src_ids,
            src_matrix,
            target_index=target_index,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
            nn_index=(
                backend_index.sklearn_nn if backend_index is not None else nn_index
            ),
            prefix_filter_index=(
                backend_index.prefix_filter if backend_index is not None else None
            ),
        )

    if backend == "sparse_dot_topn":
        return _score_source_with_sparse_dot_topn(
            src_ids,
            src_matrix,
            target_index=target_index,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
        )

    if backend == "svd_rerank":
        return _score_source_with_svd_rerank(
            src_ids,
            src_matrix,
            target_index=target_index,
            backend_index=backend_index,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
        )

    if backend in ("kmeans", "hdbscan"):
        if backend_index is None:
            raise ValueError(
                f"'{backend}' backend requires build_backend_index() to be called "
                "first (unlike 'sklearn'/'svd_rerank', there is no cheap on-the-fly "
                "fallback for a missing partition index)."
            )
        return score_source_with_partition_backend(
            src_ids,
            src_matrix,
            target_index=target_index,
            partition_index=backend_index.partition,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
        )

    if backend == "lsh":
        if backend_index is None:
            raise ValueError(
                "'lsh' backend requires build_backend_index() to be called first "
                "(like 'kmeans'/'hdbscan', there is no cheap on-the-fly fallback "
                "for a missing bucket index)."
            )
        return score_source_with_lsh_backend(
            src_ids,
            src_matrix,
            target_index=target_index,
            lsh_index=backend_index.lsh,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
        )

    if backend == "dense_brute":
        if backend_index is None:
            raise ValueError(
                "'dense_brute' backend requires build_backend_index() to be "
                "called first (like 'kmeans'/'hdbscan'/'lsh', there is no "
                "cheap on-the-fly fallback for a missing block-scan index)."
            )
        return score_source_with_dense_brute_backend(
            src_ids,
            src_matrix,
            target_index=target_index,
            dense_brute_index=backend_index.dense_brute,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
        )

    if backend == "hnsw":
        if backend_index is None:
            raise ValueError(
                "'hnsw' backend requires build_backend_index() to be called "
                "first (like 'kmeans'/'hdbscan'/'lsh'/'dense_brute', there is "
                "no cheap on-the-fly fallback for a missing usearch index)."
            )
        return score_source_with_hnsw_backend(
            src_ids,
            src_matrix,
            target_index=target_index,
            hnsw_index=backend_index.hnsw,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
        )

    raise ValueError(f"Unsupported similarity backend: {backend}")


def _score_source_with_sparse_dot_topn(
    src_ids: list[str],
    src_matrix: csr_matrix,
    *,
    target_index: TargetClusteringIndex,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
) -> list[dict[str, object]]:
    global _SPARSE_DOT_TOPN_WARNED

    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    if max_per_source <= 0:
        return []

    try:
        from sparse_dot_topn import awesome_cossim_topn

        sim = awesome_cossim_topn(
            src_matrix, target_index.target_matrix.T, max_per_source, min_similarity
        )
        rows: list[dict[str, object]] = []
        for row_idx, src_id in enumerate(src_ids):
            start = sim.indptr[row_idx]
            end = sim.indptr[row_idx + 1]
            if start == end:
                continue
            append_ranked_matches(
                rows,
                src_id=src_id,
                target_ids=target_index.target_ids,
                candidate_indices=sim.indices[start:end],
                candidate_scores=sim.data[start:end],
                min_similarity=min_similarity,
                max_per_source=max_per_source,
            )
        return rows
    except ModuleNotFoundError:
        if not _SPARSE_DOT_TOPN_WARNED:
            warnings.warn(
                "sparse_dot_topn backend requested but package is not installed; falling back to sklearn backend.",
                RuntimeWarning,
                stacklevel=2,
            )
            _SPARSE_DOT_TOPN_WARNED = True

    return _score_source_with_sklearn(
        src_ids,
        src_matrix,
        target_index=target_index,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        nn_index=None,
    )


def _score_source_with_sklearn(
    src_ids: list[str],
    src_matrix: csr_matrix,
    *,
    target_index: TargetClusteringIndex,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
    nn_index: NearestNeighbors | None,
    prefix_filter_index: PrefixFilterIndex | None = None,
) -> list[dict[str, object]]:
    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    n_targets = len(target_index.target_ids)
    if n_targets == 0 or max_per_source <= 0:
        return []

    if prefix_filter_index is not None:
        return _score_source_with_sklearn_prefix_filtered(
            src_ids,
            src_matrix,
            target_index=target_index,
            prefix_filter_index=prefix_filter_index,
            min_similarity=min_similarity,
            max_per_source=max_per_source,
        )

    if nn_index is None:
        n_neighbors = min(max_per_source, n_targets)
        nn_index = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=n_neighbors, n_jobs=-1
        )
        nn_index.fit(target_index.target_matrix)

    distances, indices = nn_index.kneighbors(src_matrix, return_distance=True)
    rows: list[dict[str, object]] = []
    append_ranked_match_block(
        rows,
        src_ids=src_ids,
        target_ids=target_index.target_ids,
        candidate_indices=np.asarray(indices),
        candidate_scores=1.0 - np.asarray(distances, dtype=np.float64),
        min_similarity=min_similarity,
        max_per_source=max_per_source,
    )
    return rows


def _score_source_with_sklearn_prefix_filtered(
    src_ids: list[str],
    src_matrix: csr_matrix,
    *,
    target_index: TargetClusteringIndex,
    prefix_filter_index: PrefixFilterIndex,
    min_similarity: float,
    max_per_source: int,
) -> list[dict[str, object]]:
    """Prefix-filtered variant of `_score_source_with_sklearn`.

    Only (source row, target row) pairs sharing a prefix token are ever handed a
    similarity computation, rather than every target row via
    `NearestNeighbors.kneighbors`; the gate is the L2AP prefix filter in
    `prefix_filter.py`, exact for the cosine scored here at the threshold the
    index was built for. Similarity is computed exactly, as sparse
    row-wise dot products, over that pruned candidate set -- both matrices
    are L2-normalized by the `"tfidf"`/`"wordpiece"`/`"sentencepiece"`
    vectorizers, so the dot product already equals cosine similarity, the
    same equivalence `_score_source_with_svd_rerank`'s exact-rerank step
    relies on.

    Candidate pairs are gathered across *several* source rows and scored in
    one batched sparse multiply, not one `.dot()` call per source row: an
    earlier per-row-loop version measurably lost to `kneighbors`' single
    vectorized BLAS matmul at every scale tried (500-20,000 target rows)
    because per-row Python/scipy call overhead dominated the FLOPs actually
    saved by pruning.

    Those sub-batches are bounded (`_resolve_pair_batch_size()`), rather than
    the whole `src_ids` chunk being gathered before the first multiply as it
    was originally. Under a large sparse TF-IDF vocabulary a row's candidate
    set is small and a whole chunk fits in one batch, which is the case those
    benchmarks covered; under a small dense subword vocabulary (WordPiece/
    SentencePiece) at real target scale, `prefix_filter_candidates()`'s
    dense-mask fallback can return close to `n_targets` candidates for a
    *single* row, so gathering a whole chunk first made peak memory scale
    with `source_chunk_size x n_targets` -- observed driving RSS from ~12GB
    to ~36GB with no scoring progress on a real `gleif -> gb` WordPiece run.
    Peak now scales with the batch budget plus one source row's
    candidate set, independent of how many source rows the chunk holds.
    """
    if min_similarity < prefix_filter_index.min_similarity:
        raise ValueError(
            "prefix filter index was sized for min_similarity="
            f"{prefix_filter_index.min_similarity}, above this scan's "
            f"{min_similarity}; the filter is exact only at or above the "
            "threshold it was built for."
        )
    src_matrix = src_matrix.tocsr()
    indptr = src_matrix.indptr
    indices = src_matrix.indices
    data = src_matrix.data
    pair_batch_size = _resolve_pair_batch_size(src_matrix, target_index.target_matrix)

    rows: list[dict[str, object]] = []
    batch: list[tuple[int, np.ndarray]] = []
    batch_pairs = 0
    for src_index in range(len(src_ids)):
        start, end = indptr[src_index], indptr[src_index + 1]
        candidates = prefix_filter_candidates(
            indices[start:end], data[start:end], prefix_filter_index
        )
        if candidates.size == 0:
            continue

        batch.append((src_index, candidates))
        batch_pairs += int(candidates.size)
        if batch_pairs >= pair_batch_size:
            _flush_prefix_filtered_batch(
                rows,
                batch,
                src_ids=src_ids,
                src_matrix=src_matrix,
                target_index=target_index,
                pair_batch_size=pair_batch_size,
                min_similarity=min_similarity,
                max_per_source=max_per_source,
            )
            batch = []
            batch_pairs = 0

    _flush_prefix_filtered_batch(
        rows,
        batch,
        src_ids=src_ids,
        src_matrix=src_matrix,
        target_index=target_index,
        pair_batch_size=pair_batch_size,
        min_similarity=min_similarity,
        max_per_source=max_per_source,
    )
    return rows


def _score_source_with_svd_rerank(
    src_ids: list[str],
    src_matrix: csr_matrix,
    *,
    target_index: TargetClusteringIndex,
    backend_index: TargetSimilarityBackendIndex | None,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
) -> list[dict[str, object]]:
    svd_state = backend_index.svd_rerank if backend_index is not None else None
    if svd_state is None:
        return _score_source_with_sklearn(
            src_ids,
            src_matrix,
            target_index=target_index,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
            nn_index=None,
        )

    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    if max_per_source <= 0:
        return []

    projected_source = svd_state.svd.transform(src_matrix)
    projected_source = normalize(projected_source, norm="l2", copy=False)
    _, candidate_indices = svd_state.neighbor_index.kneighbors(
        projected_source, return_distance=True
    )

    rows: list[dict[str, object]] = []
    # The shortlist is one block, but each row is rescored against its own
    # candidates, so the exact scores are gathered row by row and ranked as
    # one block.
    indices = np.asarray(candidate_indices, dtype=np.int64)
    if indices.size:
        exact_scores = np.empty(indices.shape, dtype=np.float64)
        for src_index in range(indices.shape[0]):
            candidate_matrix = target_index.target_matrix[indices[src_index]]
            exact_scores[src_index] = (
                src_matrix[src_index].dot(candidate_matrix.T).toarray().ravel()
            )
        append_ranked_match_block(
            rows,
            src_ids=src_ids,
            target_ids=target_index.target_ids,
            candidate_indices=indices,
            candidate_scores=exact_scores,
            min_similarity=min_similarity,
            max_per_source=max_per_source,
        )

    return rows
