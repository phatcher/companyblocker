"""Public factory entry points for constructing clustering strategy objects.

Construction/wiring only: `clustering_policy.py` owns *which* concrete type
a representation name maps to; this module calls into that policy and
performs the actual instantiation. Kept as a separate module (rather than
folded into `clustering_policy.py`) so the two concerns stay independently
testable -- see `clustering_policy.py`'s module docstring for the full
rationale. Public contract (`resolve_clustering_strategy()`/
`resolve_target_index_build_settings()`, both re-exported from
`company_vectorize.__init__`) is unchanged by this split.
"""

from __future__ import annotations

from .clustering_contract import (
    ClusteringStrategy,
    EncoderHandle,
    TargetIndexBuildSettings,
)
from .clustering_policy import (
    DEFAULT_SBERT_MODEL_NAME,
    resolve_build_settings_policy,
    resolve_strategy_class,
)


def resolve_clustering_strategy(representation: str) -> ClusteringStrategy:
    strategy_class = resolve_strategy_class(representation)
    return strategy_class()


def resolve_target_index_build_settings(
    representation: str,
    *,
    tfidf_ngram_min: int,
    tfidf_ngram_max: int,
    tfidf_analyzer: str = "char_wb",
    sbert_model_name: str = DEFAULT_SBERT_MODEL_NAME,
    sbert_force_untrained_pooling: bool = False,
    encoder: EncoderHandle | None = None,
    encoder_name: str = "",
) -> TargetIndexBuildSettings:
    policy = resolve_build_settings_policy(
        representation,
        tfidf_ngram_min=tfidf_ngram_min,
        tfidf_ngram_max=tfidf_ngram_max,
        tfidf_analyzer=tfidf_analyzer,
        sbert_model_name=sbert_model_name,
        sbert_force_untrained_pooling=sbert_force_untrained_pooling,
        encoder=encoder,
        encoder_name=encoder_name,
    )
    return policy.settings_type(**policy.kwargs)
