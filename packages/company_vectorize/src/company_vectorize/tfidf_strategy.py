from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer

from .clustering_contract import (
    TFIDF_ANALYZERS,
    TargetClusteringIndex,
    TargetIndexStorableParts,
    TfidfTargetIndexBuildSettings,
)
from .similarity_strategy_helpers import BaseSimilarityScoringStrategy
from .target_index_lifecycle import (
    build_target_index_from_frame,
    rebuild_target_index_from_tfidf_vectorizer_parts,
    tfidf_vectorizer_target_index_storable_parts,
    tfidf_vectorizer_vocabulary_from_storable_parts,
    validate_target_index_inputs,
)


class TfidfClusteringStrategy(BaseSimilarityScoringStrategy):
    representation = "tfidf"

    def _build_vectorizer(
        self,
        build_settings: TfidfTargetIndexBuildSettings,
        *,
        vocabulary: dict[str, int] | None = None,
    ) -> TfidfVectorizer:
        return TfidfVectorizer(
            lowercase=True,
            analyzer=str(build_settings.analyzer),
            ngram_range=(int(build_settings.ngram_min), int(build_settings.ngram_max)),
            min_df=1,
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
        if not isinstance(build_settings, TfidfTargetIndexBuildSettings):
            raise TypeError("tfidf strategy requires TfidfTargetIndexBuildSettings")
        ngram_min = int(build_settings.ngram_min)
        ngram_max = int(build_settings.ngram_max)
        analyzer = str(build_settings.analyzer)
        if analyzer not in TFIDF_ANALYZERS:
            raise ValueError(
                f"analyzer must be one of {sorted(TFIDF_ANALYZERS)}, got {analyzer!r}"
            )
        if target_id_col not in target_frame.columns:
            raise ValueError(
                f"target_frame is missing required column '{target_id_col}'."
            )
        if text_col not in target_frame.columns:
            raise ValueError(f"target_frame is missing required column '{text_col}'.")
        if ngram_min <= 0:
            raise ValueError("ngram_min must be greater than zero")
        if ngram_max <= 0:
            raise ValueError("ngram_max must be greater than zero")
        if ngram_max < ngram_min:
            raise ValueError("ngram_max must be greater than or equal to ngram_min")
        validate_target_index_inputs(
            target_frame,
            target_id_col=target_id_col,
            text_col=text_col,
        )

        vectorizer = self._build_vectorizer(build_settings)
        return build_target_index_from_frame(
            target_frame,
            target_id_col=target_id_col,
            text_value_expr=pl.col(text_col)
            .cast(pl.Utf8, strict=False)
            .fill_null("")
            .str.strip_chars(),
            value_col="_text",
            nonempty_predicate=pl.col("_text") != "",
            vectorizer=vectorizer,
            empty_fit_placeholder="placeholder",
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
        if not isinstance(build_settings, TfidfTargetIndexBuildSettings):
            raise TypeError("tfidf strategy requires TfidfTargetIndexBuildSettings")
        vocabulary = tfidf_vectorizer_vocabulary_from_storable_parts(storable_parts)
        vectorizer = self._build_vectorizer(build_settings, vocabulary=vocabulary)
        return rebuild_target_index_from_tfidf_vectorizer_parts(
            storable_parts, vectorizer=vectorizer
        )
