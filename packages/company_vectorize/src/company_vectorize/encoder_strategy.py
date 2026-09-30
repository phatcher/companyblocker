from __future__ import annotations

import numpy as np

from .clustering_contract import EncoderHandle, EncoderTargetIndexBuildSettings
from .dense_encoder_strategy import DenseEncoderClusteringStrategy


class _EncoderVectorizerAdapter:
    """Adapt an `EncoderHandle` (`.embed(name) -> vector`) to the sklearn-vectorizer shape.

    `build_target_index_from_frame()` and the shared scoring helpers expect a
    `fit`/`fit_transform`/`transform` vectorizer -- the same shape
    `sbert_strategy._SentenceTransformerEncoder` adapts a `SentenceTransformer`
    checkpoint to. This adapter is what lets `EncoderClusteringStrategy` reuse
    them unchanged for a handle whose native interface embeds one name at a
    time rather than a batch. It holds no state beyond the handle itself and
    the embedding width learned from the first real `embed()` call, needed
    only to shape an empty target's placeholder matrix correctly.
    """

    def __init__(self, encoder: EncoderHandle) -> None:
        self._encoder = encoder
        self._dimension: int | None = None

    def _embed_one(self, text: str) -> np.ndarray:
        vector = np.asarray(self._encoder.embed(text), dtype=np.float32)
        if self._dimension is None:
            self._dimension = vector.shape[-1]
        return vector

    def fit(self, texts: list[str]) -> _EncoderVectorizerAdapter:
        if texts:
            self._embed_one(texts[0])
        return self

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimension = self._dimension or 0
            return np.empty((0, dimension), dtype=np.float32)
        return np.stack([self._embed_one(text) for text in texts])


class EncoderClusteringStrategy(DenseEncoderClusteringStrategy):
    """Dense representation over any already-loaded encoder handle's vectors.

    Generalizes `SbertClusteringStrategy`'s shape to an encoder this package
    never loads itself: `EncoderTargetIndexBuildSettings.encoder` is already
    a ready-to-use handle, not a checkpoint name or path this strategy
    resolves and loads on its own -- that is what distinguishes it from
    `"sbert"`. Every learned family that produces such a handle (through a
    training pipeline's own persisted-model archive) reuses this one
    strategy rather than a bespoke one per family.

    `EncoderTargetIndexBuildSettings.encoder_name` is carried through
    unchanged, not interpreted, so a caller composing a run's production
    record or a strategy-comparison row can name the model that produced the
    vectors without this strategy inspecting `encoder` itself.

    The caller already holds `build_settings.encoder`, so a rebuilt index
    rewraps that same handle rather than this strategy serializing it. The
    dense backends' restrictions are `dense_encoder_strategy`'s.
    """

    representation = "encoder"
    settings_type = EncoderTargetIndexBuildSettings
    storage_irrelevant_build_settings_fields = frozenset({"encoder_name"})
    """`encoder_name` only labels which model produced the vectors; it plays
    no part in what gets embedded, so it does not belong in a cache key built
    from `target_index_storable_parts()`'s output plus its build settings."""

    def _build_encoder(
        self, build_settings: EncoderTargetIndexBuildSettings
    ) -> _EncoderVectorizerAdapter:
        return _EncoderVectorizerAdapter(build_settings.encoder)
