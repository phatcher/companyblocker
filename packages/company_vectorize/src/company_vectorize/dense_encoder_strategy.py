from __future__ import annotations

from typing import Any, cast

import polars as pl

from .clustering_contract import (
    TargetClusteringIndex,
    TargetIndexBuildSettings,
    TargetIndexStorableParts,
)
from .similarity_strategy_helpers import BaseSimilarityScoringStrategy
from .target_index_lifecycle import (
    build_target_index_from_frame,
    validate_target_index_inputs,
)

# A dense target matrix, one embedding row per name, has no sparsity to
# exploit: `sparse_dot_topn` requires real CSR input, and `svd_rerank`'s
# exact-rerank step calls `.toarray()` on the dot product, which only a scipy
# sparse matrix has. Both would misbehave deep in `sparse_similarity.py`, so
# they are refused here with a message saying which backends do fit.
UNSUPPORTED_DENSE_BACKENDS = {
    "sparse_dot_topn": (
        "requires scipy sparse input; this strategy produces dense embeddings. "
        "Use the sklearn backend, or a dense backend built for a dense target "
        "(dense_brute, hnsw)."
    ),
    "svd_rerank": (
        "its exact-rerank step assumes a sparse target matrix; this strategy "
        "produces dense embeddings. Use the sklearn backend, or a dense backend "
        "built for a dense target (dense_brute, hnsw)."
    ),
}


class DenseEncoderClusteringStrategy(BaseSimilarityScoringStrategy):
    """Shared dense-embedding strategy: the target index is an encoder's matrix.

    An encoder embeds each name, one row per target. `sbert` and `encoder` differ only in where the encoder comes from, a
    checkpoint the strategy loads or a handle the caller already holds, so a
    subclass names its representation and settings type and implements
    `_build_encoder`, which returns an object in the sklearn-vectorizer shape
    (`fit`/`fit_transform`/`transform`) that `build_target_index_from_frame()`
    and the scoring helpers expect.

    The encoder itself is never stored: `target_index_storable_parts()` keeps
    the target ids and the matrix, and `rebuild_target_index()` builds the
    encoder again from the build settings and assigns the stored matrix
    straight through, so a memory-mapped array (`numpy.load(..., mmap_mode=...)`)
    comes back with no copy.
    """

    representation: str
    settings_type: type[TargetIndexBuildSettings]

    def _build_encoder(self, build_settings: Any) -> Any:
        raise NotImplementedError

    def _require_settings(self, build_settings: object) -> None:
        if not isinstance(build_settings, self.settings_type):
            raise TypeError(
                f"{self.representation} strategy requires {self.settings_type.__name__}"
            )

    def build_target_index(
        self,
        target_frame: pl.DataFrame,
        *,
        target_id_col: str,
        text_col: str,
        build_settings,
    ) -> TargetClusteringIndex:
        self._require_settings(build_settings)
        validate_target_index_inputs(
            target_frame,
            target_id_col=target_id_col,
            text_col=text_col,
        )
        return build_target_index_from_frame(
            target_frame,
            target_id_col=target_id_col,
            text_value_expr=pl.col(text_col)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars(),
            value_col="_text",
            nonempty_predicate=pl.col("_text") != "",
            vectorizer=self._build_encoder(build_settings),
            empty_fit_placeholder="placeholder",
        )

    def target_index_storable_parts(
        self, target_index: TargetClusteringIndex
    ) -> TargetIndexStorableParts:
        return {
            "target_ids": pl.DataFrame({"target_id": target_index.target_ids}),
            "target_matrix": target_index.target_matrix,
        }

    def rebuild_target_index(
        self,
        storable_parts: TargetIndexStorableParts,
        *,
        build_settings,
    ) -> TargetClusteringIndex:
        self._require_settings(build_settings)
        target_ids_frame = cast(pl.DataFrame, storable_parts["target_ids"])
        return TargetClusteringIndex(
            target_ids=target_ids_frame.get_column("target_id").to_list(),
            vectorizer=self._build_encoder(build_settings),
            target_matrix=storable_parts["target_matrix"],
        )

    def _ensure_backend_supported(
        self,
        target_index: TargetClusteringIndex,
        *,
        backend: str,
        backend_options: dict[str, object] | None,
    ) -> None:
        _ = target_index, backend_options
        reason = UNSUPPORTED_DENSE_BACKENDS.get(backend)
        if reason is not None:
            raise ValueError(
                f"{self.representation} strategy does not support the "
                f"'{backend}' backend: {reason}"
            )
