from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from .clustering_contract import SbertTargetIndexBuildSettings
from .dense_encoder_strategy import DenseEncoderClusteringStrategy
from .sbert_model_registry import resolve_sbert_model_name
from .sbert_pooling_gate import ensure_sentence_embedding_checkpoint

DEFAULT_SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _import_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The 'sbert' representation requires the optional "
            "'sentence-transformers' dependency (install the company_vectorize "
            "'sbert' extra to use it)."
        ) from exc
    return SentenceTransformer


class _Encoder(Protocol):
    """The sklearn-vectorizer shape expected of any `TargetClusteringIndex.vectorizer`.

    Required by `build_target_index_from_frame()` and the scoring helpers. This strategy
    injects a factory producing one, `SbertClusteringStrategy.encoder_factory`, instead of
    loading `sentence_transformers` directly, so tests can supply a fake implementation
    without monkeypatching module internals.
    """

    def fit(self, texts: list[str]) -> _Encoder: ...

    def fit_transform(self, texts: list[str]) -> np.ndarray: ...

    def transform(self, texts: list[str]) -> np.ndarray: ...


EncoderFactory = Callable[[SbertTargetIndexBuildSettings], _Encoder]


class _SentenceTransformerEncoder:
    """Adapt a `SentenceTransformer` checkpoint to the `_Encoder` shape.

    That shape is `fit`/`fit_transform`/`transform`, what the shared
    `build_target_index_from_frame()` and scoring helpers expect, so this strategy reuses
    them unchanged instead of duplicating index-building logic. This is the default
    `encoder_factory` for `SbertClusteringStrategy`.

    Owns the two checks that only make sense against a real checkpoint, in
    the order that produces the most useful failure: the optional dependency
    must be installed, then the checkpoint must be a declared
    sentence-embedding model (`sbert_pooling_gate.py`) -- both before
    anything is downloaded.
    """

    def __init__(self, build_settings: SbertTargetIndexBuildSettings) -> None:
        checkpoint = resolve_sbert_model_name(build_settings.model_name)
        sentence_transformer_cls = _import_sentence_transformer()
        ensure_sentence_embedding_checkpoint(
            checkpoint, force=build_settings.force_untrained_pooling
        )
        self._model = sentence_transformer_cls(checkpoint)

    def fit(self, texts: list[str]) -> _SentenceTransformerEncoder:
        _ = texts
        return self

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        if not texts:
            dimension = self._model.get_sentence_embedding_dimension()
            return np.empty((0, dimension), dtype=np.float32)
        embeddings = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


class SbertClusteringStrategy(DenseEncoderClusteringStrategy):
    """Dense sentence-transformer embedding strategy.

    Pretrained by default (`DEFAULT_SBERT_MODEL_NAME`); pass a registry slug
    (see `sbert_model_registry.py`), another hub checkpoint, or a local
    checkpoint path in `SbertTargetIndexBuildSettings.model_name` to use a
    different encoder, including a custom-trained one -- all the same code
    path, just a different checkpoint.

    A loaded checkpoint is not a storable array or frame, so a rebuilt index
    reloads the encoder from the same checkpoint identity. The dense backends'
    restrictions are `dense_encoder_strategy`'s.

    Args:
        encoder_factory: Builds the `_Encoder` used to embed text, given the
            call's `SbertTargetIndexBuildSettings`. Defaults to
            `_SentenceTransformerEncoder` (a real `sentence_transformers`
            checkpoint); tests inject a fake factory here instead of
            monkeypatching module internals. An injected factory owns its own
            checkpoint handling -- slug resolution and the pooling gate belong
            to the real loader, not to a fake encoder that never loads
            anything.
    """

    representation = "sbert"
    settings_type = SbertTargetIndexBuildSettings
    storage_irrelevant_build_settings_fields = frozenset({"force_untrained_pooling"})
    """`force_untrained_pooling` only changes whether loading the checkpoint
    raises (`sbert_pooling_gate.py`); it plays no part in what gets embedded,
    so it does not belong in a cache key built from `target_index_storable_parts()`'s
    output plus its build settings."""

    def __init__(
        self, encoder_factory: EncoderFactory = _SentenceTransformerEncoder
    ) -> None:
        self._encoder_factory = encoder_factory

    def _build_encoder(self, build_settings: SbertTargetIndexBuildSettings) -> _Encoder:
        return self._encoder_factory(build_settings)
