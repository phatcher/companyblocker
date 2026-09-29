"""Source-chunk scoring orchestration.

Validates scoring inputs, runs a built target index's vectorizer and backend against one
source chunk, and holds the shared `BaseSimilarityScoringStrategy` base every concrete
`ClusteringStrategy` subclasses.

Split out of a single, larger module that also used to own
target-index *building* -- that concern now lives in
`target_index_lifecycle.py`, which this module does not import from and does
not depend on (both halves depend only on `clustering_contract.py`'s shared
`TargetClusteringIndex` type). `BaseSimilarityScoringStrategy.build_backend_index()`
still delegates to `sparse_similarity.py`'s backend-index construction, which
is a distinct concern from either half of this split: it builds the ranking
backend (NN search, prefix filter, ...) used *during* scoring, not the target
vectorization index itself.
"""

from __future__ import annotations

import time

import polars as pl

from .clustering_contract import TargetClusteringIndex
from .sparse_similarity import (
    TargetSimilarityBackendIndex,
    build_target_similarity_backend_index,
    score_source_with_backend,
)

SCORING_STEP_VECTORIZE_SOURCE = "vectorize_source"
SCORING_STEP_BACKEND = "backend"
"""The two steps of scoring a source chunk a caller can ask to have timed: turning
the chunk into vectors through the target's vectorizer, and the similarity
backend's work on those vectors, candidate finding, scoring and ranking together."""

_SIMILARITY_SCHEMA = {
    "source_id": pl.Utf8,
    "target_id": pl.Utf8,
    "similarity": pl.Float64,
    "rank": pl.Int32,
}


def empty_similarity_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {"source_id": [], "target_id": [], "similarity": [], "rank": []},
        schema=_SIMILARITY_SCHEMA,
    )


def validate_source_scoring_inputs(
    source_frame: pl.DataFrame,
    *,
    source_id_col: str,
    text_col: str,
    top_k: int,
    min_similarity: float,
) -> None:
    if source_id_col not in source_frame.columns:
        raise ValueError(f"source_frame is missing required column '{source_id_col}'.")
    if text_col not in source_frame.columns:
        raise ValueError(f"source_frame is missing required column '{text_col}'.")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if not (0.0 <= min_similarity <= 1.0):
        raise ValueError("min_similarity must be between 0 and 1")


def _score_rows_with_backend(
    source_nonempty: pl.DataFrame,
    *,
    value_col: str,
    target_index: TargetClusteringIndex,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
    nn_index,
    backend: str,
    backend_index: TargetSimilarityBackendIndex | None,
    timings: dict[str, float] | None = None,
) -> list[dict[str, object]]:
    src_ids = source_nonempty.get_column("source_id").to_list()
    src_values = source_nonempty.get_column(value_col).to_list()
    clock = time.perf_counter()
    src_matrix = target_index.vectorizer.transform(src_values)
    vectorized = time.perf_counter()
    rows = score_source_with_backend(
        src_ids,
        src_matrix,
        target_index=target_index,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        nn_index=nn_index,
        backend=backend,
        backend_index=backend_index,
    )
    if timings is not None:
        scored = time.perf_counter()
        timings[SCORING_STEP_VECTORIZE_SOURCE] = (
            timings.get(SCORING_STEP_VECTORIZE_SOURCE, 0.0) + vectorized - clock
        )
        timings[SCORING_STEP_BACKEND] = (
            timings.get(SCORING_STEP_BACKEND, 0.0) + scored - vectorized
        )
    return rows


def score_source_chunk_from_frame(
    source_frame: pl.DataFrame,
    *,
    target_index: TargetClusteringIndex,
    source_id_col: str,
    text_col: str,
    text_value_expr: pl.Expr,
    value_col: str,
    nonempty_predicate: pl.Expr,
    top_k: int,
    min_similarity: float,
    max_candidates_per_source: int | None,
    nn_index,
    backend: str,
    backend_index: TargetSimilarityBackendIndex | None,
    timings: dict[str, float] | None = None,
) -> pl.DataFrame:
    validate_source_scoring_inputs(
        source_frame,
        source_id_col=source_id_col,
        text_col=text_col,
        top_k=top_k,
        min_similarity=min_similarity,
    )
    if not target_index.target_ids:
        return empty_similarity_frame()

    source = source_frame.select(
        pl.col(source_id_col).cast(pl.Utf8).alias("source_id"),
        text_value_expr.alias(value_col),
    )
    source_nonempty = source.filter(nonempty_predicate)
    if source_nonempty.height == 0:
        return empty_similarity_frame()

    rows = _score_rows_with_backend(
        source_nonempty,
        value_col=value_col,
        target_index=target_index,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        nn_index=nn_index,
        backend=backend,
        backend_index=backend_index,
        timings=timings,
    )
    if not rows:
        return empty_similarity_frame()
    return pl.DataFrame(rows, schema=_SIMILARITY_SCHEMA)


class BaseSimilarityScoringStrategy:
    """Shared scoring/backend-selection base for every `ClusteringStrategy`.

    `storage_irrelevant_build_settings_fields` defaults to empty here: a
    subclass whose build settings include a field that does not affect
    `target_index_storable_parts()`'s output overrides it (see
    `SbertClusteringStrategy.force_untrained_pooling`).
    """

    storage_irrelevant_build_settings_fields: frozenset[str] = frozenset()

    def _ensure_backend_supported(
        self,
        target_index: TargetClusteringIndex,
        *,
        backend: str,
        backend_options: dict[str, object] | None,
    ) -> None:
        """Refuse a backend this strategy cannot score with; accepts all by default.

        Called from both `build_backend_index()` and `score_source_chunk()`, because
        the latter is reachable on its own: the contract lets a caller pass a
        pre-built `nn_index` and no `backend_index` at all, which would otherwise
        walk straight past a check made only at build time.
        """
        _ = target_index, backend, backend_options

    def build_backend_index(
        self,
        target_index: TargetClusteringIndex,
        *,
        backend: str,
        top_k: int,
        max_candidates_per_source: int | None,
        backend_options: dict[str, object] | None,
        min_similarity: float | None = None,
    ) -> TargetSimilarityBackendIndex | None:
        self._ensure_backend_supported(
            target_index, backend=backend, backend_options=backend_options
        )
        return build_target_similarity_backend_index(
            target_index,
            backend=backend,
            top_k=top_k,
            max_candidates_per_source=max_candidates_per_source,
            backend_options=backend_options,
            min_similarity=min_similarity,
        )

    def _build_source_scoring_projection(
        self, text_col: str
    ) -> tuple[pl.Expr, str, pl.Expr]:
        """Default projection for plain-text strategies such as sbert and tfidf.

        Casts to string, treats null as empty and strips whitespace. Strategies scoring on
        a different column shape, token lists for instance, override this.
        """
        return (
            pl.col(text_col)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars(),
            "_text",
            pl.col("_text") != "",
        )

    def score_source_chunk(
        self,
        source_frame: pl.DataFrame,
        *,
        target_index: TargetClusteringIndex,
        source_id_col: str,
        text_col: str,
        top_k: int,
        min_similarity: float,
        max_candidates_per_source: int | None,
        nn_index,
        backend: str,
        backend_index: TargetSimilarityBackendIndex | None,
        backend_options: dict[str, object] | None,
        timings: dict[str, float] | None = None,
    ) -> pl.DataFrame:
        self._ensure_backend_supported(
            target_index, backend=backend, backend_options=backend_options
        )
        text_value_expr, value_col, nonempty_predicate = (
            self._build_source_scoring_projection(text_col)
        )
        return score_source_chunk_from_frame(
            source_frame,
            target_index=target_index,
            source_id_col=source_id_col,
            text_col=text_col,
            text_value_expr=text_value_expr,
            value_col=value_col,
            nonempty_predicate=nonempty_predicate,
            top_k=top_k,
            min_similarity=min_similarity,
            max_candidates_per_source=max_candidates_per_source,
            nn_index=nn_index,
            backend=backend,
            backend_index=backend_index,
            timings=timings,
        )
