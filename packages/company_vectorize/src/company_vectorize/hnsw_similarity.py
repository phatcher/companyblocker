from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import issparse
from usearch.index import Index

from .clustering_contract import TargetClusteringIndex
from .ranking import append_ranked_match_block

# A real sub-linear (approximate nearest-neighbour) index for a *dense*
# target matrix, on `usearch`, chosen over `hnswlib`/`faiss` for storage
# width: float16/int8 in-process, against `hnswlib`'s float32-only and
# `faiss`'s much larger dependency footprint. "dense_brute" stays the exact, CPU-bound reference
# this backend's recall is measured against (dense_brute_similarity.py); this
# backend trades a small, bounded amount of recall for a search cost that no
# longer scales linearly with the target's row count.
#
# Cosine similarity via usearch's `metric="cos"`, the same metric every other
# backend in this package reports (see dense_brute_similarity.py's docstring
# and _score_source_with_sklearn's `1.0 - distance` conversion, which this
# module mirrors): usearch's cosine distance is `1 - cosine_similarity`, and
# an all-zero row scores a well-defined distance (confirmed directly) rather
# than dividing by zero, so no separate normalization step is needed before
# handing vectors to usearch.
#
# Keys are the target row's *position* (0..n_targets-1), never `target_id`
# itself: usearch keys are integers, and every other backend in this package
# already threads candidate row positions through `target_ids` at ranking
# time (`append_ranked_matches`), so this backend does the same rather than
# inventing a second id scheme.
#
# Build concurrency is capped, not left at usearch's own `threads=0` ("every
# core"): measured directly on a 32-logical-core machine, `Index.add()` at
# `threads=0` gave a fixed-seed 3,000-row/32-dim fixture a mean recall of
# 0.95-0.97 against `dense_brute` with real dips to 0.949 across repeated
# builds of the *same* data (concurrent graph insertion is order-sensitive,
# so more concurrent inserters means more graph-quality variance, not just
# more speed) -- too close to this item's own recall bar to be a reliable
# default. `threads=8` on the same fixture measured mean 0.984, min 0.979
# across 30 builds: comfortably clear, and still multi-threaded, so a
# `gb`/`fr`-scale build keeps most of the wall-clock benefit `usearch` was
# chosen for. Since the thread count shapes the graph, it is a build setting
# like the graph's own: a run records it, and a saved index is keyed on it.
DEFAULT_HNSW_BUILD_THREADS = 8

DEFAULT_HNSW_STORAGE_DTYPE = "float16"
DEFAULT_HNSW_CONNECTIVITY = 16
DEFAULT_HNSW_EXPANSION_ADD = 128
DEFAULT_HNSW_EXPANSION_SEARCH = 64

STORAGE_DTYPE_OPTION = "storage_dtype"
CONNECTIVITY_OPTION = "connectivity"
EXPANSION_ADD_OPTION = "expansion_add"
EXPANSION_SEARCH_OPTION = "expansion_search"
BUILD_THREADS_OPTION = "build_threads"
INDEX_DIR_OPTION = "index_dir"

# The backend options that change what gets *built* -- the saved graph a
# reopened index must have been built with to be reused. `expansion_search`
# is deliberately excluded: it is a search-time knob usearch lets a caller
# change on an already-built (or already-reopened) `Index` directly (see
# `build_target_hnsw_index`), never baked into the saved graph, so a change
# to it alone must not force a rebuild. A caller computing a cache directory
# keyed on the graph's build settings should
# hash `hnsw_build_settings()`'s output, not the whole `backend_options`
# mapping, for exactly that reason.
_BUILD_SETTING_OPTIONS: tuple[str, ...] = (
    STORAGE_DTYPE_OPTION,
    CONNECTIVITY_OPTION,
    EXPANSION_ADD_OPTION,
    BUILD_THREADS_OPTION,
)

_USEARCH_SCALAR_KINDS: dict[str, str] = {"float16": "f16", "int8": "i8"}
HNSW_STORAGE_DTYPES: tuple[str, ...] = tuple(_USEARCH_SCALAR_KINDS)

_METRIC = "cos"
_INDEX_FILENAME = "hnsw_index.usearch"
_MANIFEST_FILENAME = "hnsw_manifest.json"


def _resolve_storage_dtype(options: dict[str, object]) -> str:
    raw = options.get(STORAGE_DTYPE_OPTION, DEFAULT_HNSW_STORAGE_DTYPE)
    name = str(raw).strip().lower()
    if name not in _USEARCH_SCALAR_KINDS:
        raise ValueError(
            f"hnsw backend option '{STORAGE_DTYPE_OPTION}' must be one of "
            f"{sorted(_USEARCH_SCALAR_KINDS)}, got {raw!r}."
        )
    return name


def _resolve_positive_int_option(
    options: dict[str, object], key: str, default: int
) -> int:
    raw = options.get(key, default)
    value = None
    if isinstance(raw, bool):
        value = None
    elif isinstance(raw, (int, float, str)):
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = None
    if value is None or value <= 0:
        raise ValueError(
            f"hnsw backend option '{key}' must be a positive integer, got {raw!r}."
        )
    return value


@dataclass(frozen=True)
class HnswBuildParams:
    """Resolved `hnsw` backend options that determine the built graph.

    Attributes:
        storage_dtype: `"float16"` or `"int8"` -- the `usearch` scalar kind
            the index stores vectors at.
        connectivity: `usearch.index.Index(connectivity=...)`: graph fan-out
            per node.
        expansion_add: `usearch.index.Index(expansion_add=...)`: candidate
            list size while building.
        build_threads: `usearch.index.Index.add(threads=...)`: concurrent
            inserters, which change the graph built.
    """

    storage_dtype: str
    connectivity: int
    expansion_add: int
    build_threads: int


def _resolve_build_params(options: dict[str, object]) -> HnswBuildParams:
    return HnswBuildParams(
        storage_dtype=_resolve_storage_dtype(options),
        connectivity=_resolve_positive_int_option(
            options, CONNECTIVITY_OPTION, DEFAULT_HNSW_CONNECTIVITY
        ),
        expansion_add=_resolve_positive_int_option(
            options, EXPANSION_ADD_OPTION, DEFAULT_HNSW_EXPANSION_ADD
        ),
        build_threads=_resolve_positive_int_option(
            options, BUILD_THREADS_OPTION, DEFAULT_HNSW_BUILD_THREADS
        ),
    )


def hnsw_build_settings(backend_options: dict[str, object] | None) -> dict[str, object]:
    """The subset of `backend_options` a caller should key a saved-index cache directory on.

    That subset is everything that changes the built graph, and nothing that
    only changes how it is searched (`expansion_search`) or how many results
    a query asks for (`top_k`/`max_candidates_per_source`).

    Resolves each option through the same validation `build_target_hnsw_index`
    applies, so an invalid option is rejected here, before any directory is
    resolved from it, rather than surfacing later as a confusing rebuild.
    """
    params = _resolve_build_params(backend_options or {})
    return {
        "storage_dtype": params.storage_dtype,
        "connectivity": params.connectivity,
        "expansion_add": params.expansion_add,
        "build_threads": params.build_threads,
    }


def _resolve_index_dir(options: dict[str, object]) -> Path | None:
    raw = options.get(INDEX_DIR_OPTION)
    if raw is None:
        return None
    if isinstance(raw, (str, os.PathLike)):
        return Path(raw)
    raise ValueError(
        f"hnsw backend option '{INDEX_DIR_OPTION}' must be a path-like value, "
        f"got {raw!r}."
    )


def _require_dense_target(matrix: object) -> None:
    if issparse(matrix):
        raise ValueError(
            "hnsw backend requires a dense target matrix (for example the "
            "'sbert' representation); sparse representations like "
            "tfidf/wordpiece/sentencepiece are not supported by this backend "
            "-- their sparsity is scored exactly by the 'sklearn' backend, or "
            "approximately by the 'lsh' backend, instead."
        )


@dataclass(frozen=True)
class TargetHnswIndex:
    """Fitted state for the `"hnsw"` similarity backend.

    Built by `build_target_hnsw_index()` and stored on
    `TargetSimilarityBackendIndex.hnsw`; not normally constructed directly.

    Attributes:
        index: The built (or reopened, memory-mapped) `usearch.index.Index`.
        connectivity: Resolved `"connectivity"` backend option the graph was
            built with.
        expansion_add: Resolved `"expansion_add"` backend option the graph
            was built with.
        expansion_search: Resolved `"expansion_search"` backend option,
            applied to `index` whether it was just built or reopened -- see
            `build_target_hnsw_index`'s docstring for why this one option is
            always applied fresh rather than trusted from a saved index.
        storage_dtype: Resolved `"storage_dtype"` backend option (`"float16"`
            or `"int8"`) `index` stores vectors at.
        reopened: Whether `index` was reopened memory-mapped from a saved,
            matching index rather than built fresh this call.
    """

    index: Index
    connectivity: int
    expansion_add: int
    expansion_search: int
    storage_dtype: str
    reopened: bool


def _manifest_payload(
    *, params: HnswBuildParams, ndim: int, count: int
) -> dict[str, object]:
    return {
        "storage_dtype": params.storage_dtype,
        "connectivity": params.connectivity,
        "expansion_add": params.expansion_add,
        "build_threads": params.build_threads,
        "metric": _METRIC,
        "ndim": ndim,
        "count": count,
    }


def _read_manifest(manifest_path: Path) -> dict[str, object] | None:
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _try_reopen(
    *, index_path: Path, manifest_path: Path, expected_manifest: dict[str, object]
) -> Index | None:
    if not index_path.is_file() or not manifest_path.is_file():
        return None
    if _read_manifest(manifest_path) != expected_manifest:
        return None
    return Index.restore(index_path, view=True)


def _atomic_save(
    index: Index, index_path: Path, manifest_path: Path, manifest: dict[str, object]
) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = index_path.with_name(f".{index_path.name}.tmp-{os.getpid()}")
    try:
        index.save(temporary_path)
        os.replace(temporary_path, index_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    # The manifest is written last, once the index file itself is fully on
    # disk under its final name: a reader that sees a manifest sees a
    # complete, matching index file, the same completion-marker-last
    # convention `workspace.artifact_archive.commit_artifact_write` uses
    # (this package may not depend on that module -- see its README's
    # Boundaries section -- so this is a self-contained copy of the same
    # idea, not a call to it).
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")


def build_target_hnsw_index(
    target_index: TargetClusteringIndex,
    *,
    backend_options: dict[str, object] | None = None,
) -> TargetHnswIndex | None:
    """Build (or reopen) the `"hnsw"` similarity-backend index for `target_index`.

    Returns `None` for an empty target (no rows), matching the other
    backends' `None`-on-empty-target convention.

    When `backend_options["index_dir"]` is given and already holds a graph
    built from the same `hnsw_build_settings()` (a matching manifest plus the
    index file it describes), that graph is reopened memory-mapped through
    `usearch.index.Index.restore(..., view=True)` rather than rebuilt -- the
    expensive step at scale. Otherwise a fresh
    graph is built and, when `index_dir` is given, saved there for a later
    call to reopen. `expansion_search` is applied to `index` either way,
    fresh from this call's own resolved option: it is a search-time knob
    (`hnsw_build_settings()`'s docstring), and `usearch` does not preserve it
    across a save/restore round-trip (confirmed directly -- a restored index
    reports `usearch`'s own default regardless of what it was built with).

    Raises:
        ValueError: `target_index.target_matrix` is sparse, or a backend
            option cannot be read (`hnsw_build_settings`/
            `_resolve_positive_int_option`).
    """
    n_targets = len(target_index.target_ids)
    if n_targets == 0:
        return None
    _require_dense_target(target_index.target_matrix)

    options = backend_options or {}
    params = _resolve_build_params(options)
    expansion_search = _resolve_positive_int_option(
        options, EXPANSION_SEARCH_OPTION, DEFAULT_HNSW_EXPANSION_SEARCH
    )

    target_matrix = np.asarray(target_index.target_matrix, dtype=np.float32, order="C")
    ndim = int(target_matrix.shape[1]) if target_matrix.ndim == 2 else 0
    manifest = _manifest_payload(params=params, ndim=ndim, count=n_targets)

    index_dir = _resolve_index_dir(options)
    index_path: Path | None = None
    manifest_path: Path | None = None
    if index_dir is not None:
        index_path = index_dir / _INDEX_FILENAME
        manifest_path = index_dir / _MANIFEST_FILENAME
        reopened_index = _try_reopen(
            index_path=index_path,
            manifest_path=manifest_path,
            expected_manifest=manifest,
        )
        if reopened_index is not None:
            # `expansion_search` is a mutable runtime attribute (confirmed
            # directly: set then read back the new value) despite the
            # installed `usearch` type stub declaring the property
            # read-only, hence the ignore.
            reopened_index.expansion_search = expansion_search  # type: ignore[misc]
            return TargetHnswIndex(
                index=reopened_index,
                connectivity=params.connectivity,
                expansion_add=params.expansion_add,
                expansion_search=expansion_search,
                storage_dtype=params.storage_dtype,
                reopened=True,
            )

    index = Index(
        ndim=ndim,
        metric=_METRIC,
        dtype=_USEARCH_SCALAR_KINDS[params.storage_dtype],
        connectivity=params.connectivity,
        expansion_add=params.expansion_add,
        expansion_search=expansion_search,
    )
    keys = np.arange(n_targets, dtype=np.int64)
    index.add(keys, target_matrix, threads=params.build_threads)

    if index_path is not None and manifest_path is not None:
        _atomic_save(index, index_path, manifest_path, manifest)

    return TargetHnswIndex(
        index=index,
        connectivity=params.connectivity,
        expansion_add=params.expansion_add,
        expansion_search=expansion_search,
        storage_dtype=params.storage_dtype,
        reopened=False,
    )


def _as_matches_matrix(matches, *, n_src: int, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a `usearch` search result to `(n_src, k)` keys/distances.

    `Index.search()` returns a `Matches` object (1-D `keys`/`distances`) for
    a single query row -- including a `(1, ndim)` array, confirmed directly
    -- and a `BatchMatches` object (2-D, `NaN`-distance-padded to `k` when
    fewer than `k` candidates exist) for more than one. Both are reshaped to
    `(n_src, k)` here, `NaN`-padding the single-row case the same way
    `BatchMatches` already pads the multi-row case, so callers read one shape
    regardless of `n_src`.
    """
    keys = np.asarray(matches.keys)
    distances = np.asarray(matches.distances)
    if n_src == 1:
        keys = keys.reshape(1, -1)
        distances = distances.reshape(1, -1)
    width = keys.shape[1]
    if width < k:
        pad = k - width
        keys = np.concatenate([keys, np.zeros((n_src, pad), dtype=keys.dtype)], axis=1)
        distances = np.concatenate(
            [distances, np.full((n_src, pad), np.nan, dtype=distances.dtype)], axis=1
        )
    return keys, distances


def score_source_with_hnsw_backend(
    src_ids: list[str],
    src_matrix,
    *,
    target_index: TargetClusteringIndex,
    hnsw_index: TargetHnswIndex | None,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
) -> list[dict[str, object]]:
    """Score one source chunk against `hnsw_index`'s target, `usearch.Index.search()`.

    Cosine similarity is `1.0 - distance`, `usearch`'s own `"cos"` metric
    (module docstring), applied through `append_ranked_matches()` exactly as
    every other backend in this package applies it, so `min_similarity`,
    `top_k` and `max_candidates_per_source` mean the same thing on `"hnsw"`
    as on `"dense_brute"`/`"sklearn"`/every other backend.

    Returns `[]` if `hnsw_index` is `None` (no target index built, or an
    empty target), the source chunk is empty, or
    `max(top_k, max_candidates_per_source)` resolves to `<= 0`.
    """
    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )
    if hnsw_index is None or max_per_source <= 0:
        return []

    src = np.asarray(src_matrix, dtype=np.float32)
    if src.ndim != 2 or src.shape[0] == 0:
        return []

    n_targets = len(target_index.target_ids)
    k = min(max_per_source, n_targets)
    if k <= 0:
        return []

    n_src = src.shape[0]
    matches = hnsw_index.index.search(src, k)
    keys, distances = _as_matches_matrix(matches, n_src=n_src, k=k)

    rows: list[dict[str, object]] = []
    # A search that returned fewer than `k` neighbours pads with `nan`;
    # scoring those `-inf` drops them, as the per-row mask did.
    valid = ~np.isnan(distances)
    append_ranked_match_block(
        rows,
        src_ids=src_ids,
        target_ids=target_index.target_ids,
        candidate_indices=np.where(valid, keys, 0).astype(np.int64),
        candidate_scores=np.where(valid, 1.0 - distances, -np.inf).astype(np.float64),
        min_similarity=min_similarity,
        max_per_source=max_per_source,
    )
    return rows
