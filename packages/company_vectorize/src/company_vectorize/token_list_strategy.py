from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from .clustering_contract import (
    TargetClusteringIndex,
    TargetIndexBuildSettings,
    TargetIndexStorableParts,
)
from .dense_vocabulary_gate import ensure_dense_vocabulary_scale_supported
from .similarity_strategy_helpers import BaseSimilarityScoringStrategy
from .target_index_lifecycle import (
    build_target_index_from_frame,
    rebuild_target_index_from_tfidf_vectorizer_parts,
    tfidf_vectorizer_target_index_storable_parts,
    tfidf_vectorizer_vocabulary_from_storable_parts,
    validate_target_index_inputs,
)


def _identity_tokens(tokens: list[str]) -> list[str]:
    return tokens


class TokenListClusteringStrategy(BaseSimilarityScoringStrategy):
    """Shared TF-IDF-over-emitted-tokens strategy for tokenizer-backed representations.

    WordPiece and SentencePiece both emit a list of string pieces per name, so the
    target-index and scoring logic is representation-agnostic and only the settings type
    and name differ per subclass.

    The dense subword vocabulary both representations produce is also what
    makes them the gated side of the dense-vocabulary compatibility check: the
    exhaustive `"sklearn"`/`"sparse_dot_topn"` backends are refused past
    `max_rows` target rows unless forced. Because that is a property of the token
    vocabulary rather than of either tokenizer, the check belongs here rather
    than on the two subclasses -- see `dense_vocabulary_gate.py`.
    """

    representation: str
    settings_type: type[TargetIndexBuildSettings]

    def _build_vectorizer(
        self, *, vocabulary: dict[str, int] | None = None
    ) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=False,
            analyzer=_identity_tokens,
            token_pattern=None,
            preprocessor=None,
            tokenizer=None,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
            vocabulary=vocabulary,
        )

    def build_target_index(
        self,
        target_frame: pl.DataFrame,
        *,
        target_id_col: str,
        text_col: str,
        build_settings,
    ) -> TargetClusteringIndex:
        if not isinstance(build_settings, self.settings_type):
            raise TypeError(
                f"{self.representation} strategy requires {self.settings_type.__name__}"
            )
        validate_target_index_inputs(
            target_frame,
            target_id_col=target_id_col,
            text_col=text_col,
        )
        vectorizer = self._build_vectorizer()
        return build_target_index_from_frame(
            target_frame,
            target_id_col=target_id_col,
            text_value_expr=pl.col(text_col)
            .cast(pl.List(pl.Utf8), strict=False)
            .fill_null([]),
            value_col="_tokens",
            nonempty_predicate=pl.col("_tokens").list.len() > 0,
            vectorizer=vectorizer,
            empty_fit_placeholder=["placeholder"],
        )

    def target_index_storable_parts(
        self, target_index: TargetClusteringIndex
    ) -> TargetIndexStorableParts:
        return tfidf_vectorizer_target_index_storable_parts(target_index)

    def rebuild_target_index(
        self,
        storable_parts: TargetIndexStorableParts,
        *,
        build_settings,
    ) -> TargetClusteringIndex:
        if not isinstance(build_settings, self.settings_type):
            raise TypeError(
                f"{self.representation} strategy requires {self.settings_type.__name__}"
            )
        vocabulary = tfidf_vectorizer_vocabulary_from_storable_parts(storable_parts)
        vectorizer = self._build_vectorizer(vocabulary=vocabulary)
        return rebuild_target_index_from_tfidf_vectorizer_parts(
            storable_parts, vectorizer=vectorizer
        )

    def _ensure_backend_supported(
        self,
        target_index: TargetClusteringIndex,
        *,
        backend: str,
        backend_options: dict[str, object] | None,
    ) -> None:
        ensure_dense_vocabulary_scale_supported(
            representation=self.representation,
            backend=backend,
            target_rows=len(target_index.target_ids),
            backend_options=backend_options,
        )

    def _build_source_scoring_projection(
        self, text_col: str
    ) -> tuple[pl.Expr, str, pl.Expr]:
        return (
            pl.col(text_col).cast(pl.List(pl.Utf8), strict=False).fill_null([]),
            "_tokens",
            pl.col("_tokens").list.len() > 0,
        )
