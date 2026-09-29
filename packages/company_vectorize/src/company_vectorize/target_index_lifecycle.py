"""Target-side index lifecycle.

Validates inputs and builds a fitted `TargetClusteringIndex` from a target
frame, and holds the persist-and-rehydrate helpers shared by every strategy
whose vectorizer is a fitted `sklearn`-style `TfidfVectorizer`
(`TfidfClusteringStrategy` over raw text, `TokenListClusteringStrategy` over
pre-tokenized lists): both need the fitted vocabulary and idf weights, not a
pickled vectorizer, to rebuild one that scores identically.
`SbertClusteringStrategy` needs none of this -- its vectorizer is a loaded
encoder rebuilt straight from its build settings, not from stored state.

Split out of `similarity_strategy_helpers.py` to separate this
"build a target index once" concern from that module's "score a source chunk
against an already-built index" concern -- the same policy/construction-style
boundary `clustering_policy.py`/`clustering_factory.py` drew for factory
wiring. Every concrete `ClusteringStrategy.build_target_index()`
implementation (`TfidfClusteringStrategy`, `TokenListClusteringStrategy`,
`SbertClusteringStrategy`) calls into this module; none of it depends on
`similarity_strategy_helpers.py`, and nothing in that module depends on this
one -- there is no import cycle between the two halves of the split.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

from .clustering_contract import TargetClusteringIndex, TargetIndexStorableParts


def validate_target_index_inputs(
    target_frame: pl.DataFrame,
    *,
    target_id_col: str,
    text_col: str,
) -> None:
    if target_id_col not in target_frame.columns:
        raise ValueError(f"target_frame is missing required column '{target_id_col}'.")
    if text_col not in target_frame.columns:
        raise ValueError(f"target_frame is missing required column '{text_col}'.")


def _build_index_from_nonempty_target(
    target_nonempty: pl.DataFrame,
    *,
    value_col: str,
    vectorizer,
) -> TargetClusteringIndex:
    target_ids = target_nonempty.get_column("target_id").to_list()
    target_values = target_nonempty.get_column(value_col).to_list()
    target_matrix = vectorizer.fit_transform(target_values)
    return TargetClusteringIndex(
        target_ids=target_ids, vectorizer=vectorizer, target_matrix=target_matrix
    )


def build_target_index_from_frame(
    target_frame: pl.DataFrame,
    *,
    target_id_col: str,
    text_value_expr: pl.Expr,
    value_col: str,
    nonempty_predicate: pl.Expr,
    vectorizer,
    empty_fit_placeholder,
) -> TargetClusteringIndex:
    target = target_frame.select(
        pl.col(target_id_col).cast(pl.Utf8).alias("target_id"),
        text_value_expr.alias(value_col),
    )
    target_nonempty = target.filter(nonempty_predicate)
    if target_nonempty.height == 0:
        vectorizer.fit([empty_fit_placeholder])
        empty = vectorizer.transform([])
        return TargetClusteringIndex(
            target_ids=[], vectorizer=vectorizer, target_matrix=empty
        )
    return _build_index_from_nonempty_target(
        target_nonempty,
        value_col=value_col,
        vectorizer=vectorizer,
    )


def tfidf_vectorizer_target_index_storable_parts(
    target_index: TargetClusteringIndex,
) -> TargetIndexStorableParts:
    """Storable parts for a target index built from a fitted `TfidfVectorizer`.

    Applies to any `TargetClusteringIndex` whose vectorizer is
    `sklearn`-style: `vocabulary_`/`idf_`-bearing, with `.transform()`.
    Shared by `TfidfClusteringStrategy.target_index_storable_parts()` and
    `TokenListClusteringStrategy.target_index_storable_parts()`: the fitted
    vocabulary and idf weights, plus the already-built target matrix broken
    into its CSR arrays, are enough to rebuild a vectorizer that scores
    identically -- see `rebuild_target_index_from_tfidf_vectorizer_parts()`.
    Neither strategy needs to pickle the vectorizer itself. `target_matrix`
    is always a `csr_matrix` here: both callers fit it via `TfidfVectorizer`,
    which never returns another sparse format.
    """
    matrix = cast(csr_matrix, target_index.target_matrix)
    vocabulary = target_index.vectorizer.vocabulary_
    idf_weights = np.asarray(target_index.vectorizer.idf_, dtype=np.float64)
    return {
        "target_ids": pl.DataFrame({"target_id": target_index.target_ids}),
        "vocabulary": pl.DataFrame(
            {"token": list(vocabulary.keys()), "index": list(vocabulary.values())}
        ),
        "idf_weights": idf_weights,
        "target_matrix_data": np.asarray(matrix.data),
        "target_matrix_indices": np.asarray(matrix.indices),
        "target_matrix_indptr": np.asarray(matrix.indptr),
    }


def tfidf_vectorizer_vocabulary_from_storable_parts(
    storable_parts: TargetIndexStorableParts,
) -> dict[str, int]:
    """The `vocabulary_` mapping a fresh `TfidfVectorizer` needs at construction.

    A strategy calls this before constructing the vectorizer it passes to
    `rebuild_target_index_from_tfidf_vectorizer_parts()`: `sklearn` fixes a
    `TfidfVectorizer`'s vocabulary via its `vocabulary=` constructor
    argument, not after the fact.
    """
    vocab_frame = cast(pl.DataFrame, storable_parts["vocabulary"])
    return dict(
        zip(
            vocab_frame.get_column("token").to_list(),
            (int(index) for index in vocab_frame.get_column("index").to_list()),
            strict=True,
        )
    )


def rebuild_target_index_from_tfidf_vectorizer_parts(
    storable_parts: TargetIndexStorableParts,
    *,
    vectorizer,
) -> TargetClusteringIndex:
    """Rebuild the `TargetClusteringIndex` behind a stored-parts result.

    Args:
        storable_parts: A `tfidf_vectorizer_target_index_storable_parts()` result.
        vectorizer: A freshly constructed vectorizer, of the same type and
            settings as the one that result came from, with `vocabulary=`
            already set from
            `tfidf_vectorizer_vocabulary_from_storable_parts(storable_parts)`.
            This function sets its `idf_` from `storable_parts` and returns
            it as the rebuilt index's `vectorizer`.
    """
    vectorizer.idf_ = storable_parts["idf_weights"]
    target_ids_frame = cast(pl.DataFrame, storable_parts["target_ids"])
    target_ids = target_ids_frame.get_column("target_id").to_list()
    target_matrix = csr_matrix(
        (
            storable_parts["target_matrix_data"],
            storable_parts["target_matrix_indices"],
            storable_parts["target_matrix_indptr"],
        ),
        shape=(len(target_ids), len(vectorizer.vocabulary_)),
    )
    return TargetClusteringIndex(
        target_ids=target_ids, vectorizer=vectorizer, target_matrix=target_matrix
    )
