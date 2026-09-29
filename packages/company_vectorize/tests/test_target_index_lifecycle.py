"""Regression coverage for `target_index_lifecycle.py`, exercised directly
against pure helper functions rather than through a concrete
`ClusteringStrategy` subclass -- this module's job is only "validate a target
frame, then fit a vectorizer over it," with no scoring involved.
"""

from typing import ClassVar

import numpy as np
import polars as pl
import pytest
from company_vectorize.clustering_contract import TargetClusteringIndex
from company_vectorize.target_index_lifecycle import (
    _build_index_from_nonempty_target,
    build_target_index_from_frame,
    validate_target_index_inputs,
)


def test_validate_target_index_inputs_passes_for_valid_frame():
    frame = pl.DataFrame({"target_id": ["t1"], "name": ["Acme Ltd"]})
    validate_target_index_inputs(frame, target_id_col="target_id", text_col="name")


def test_validate_target_index_inputs_rejects_missing_id_col():
    frame = pl.DataFrame({"name": ["Acme Ltd"]})
    with pytest.raises(ValueError, match="missing required column 'target_id'"):
        validate_target_index_inputs(frame, target_id_col="target_id", text_col="name")


def test_validate_target_index_inputs_rejects_missing_text_col():
    frame = pl.DataFrame({"target_id": ["t1"]})
    with pytest.raises(ValueError, match="missing required column 'name'"):
        validate_target_index_inputs(frame, target_id_col="target_id", text_col="name")


class _FakeVectorizer:
    """Deterministic stand-in for an sklearn-style vectorizer: a fixed
    lookup table by input value, no real vectorization math involved --
    only the index-lifecycle plumbing (which rows get fitted, what an empty
    target produces) is under test here.
    """

    _VECTORS: ClassVar[dict[str, list[float]]] = {
        "acme ltd": [1.0, 0.0],
        "beta plc": [0.0, 1.0],
        "placeholder": [1.0, 1.0],
    }

    def fit(self, values: list[str]) -> "_FakeVectorizer":
        _ = values
        return self

    def fit_transform(self, values: list[str]) -> np.ndarray:
        return np.array([self._VECTORS[v] for v in values], dtype=np.float32)

    def transform(self, values: list[str]) -> np.ndarray:
        if not values:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([self._VECTORS[v] for v in values], dtype=np.float32)


def test_build_index_from_nonempty_target_fits_and_aligns_ids():
    target_nonempty = pl.DataFrame(
        {"target_id": ["t1", "t2"], "_text": ["acme ltd", "beta plc"]}
    )

    index = _build_index_from_nonempty_target(
        target_nonempty, value_col="_text", vectorizer=_FakeVectorizer()
    )

    assert isinstance(index, TargetClusteringIndex)
    assert index.target_ids == ["t1", "t2"]
    assert index.target_matrix.shape == (2, 2)
    np.testing.assert_array_equal(index.target_matrix[0], [1.0, 0.0])
    np.testing.assert_array_equal(index.target_matrix[1], [0.0, 1.0])


def test_build_target_index_from_frame_filters_and_projects():
    target_frame = pl.DataFrame(
        {
            "system_uri": ["t1", "t2", "t3"],
            "name": ["Acme Ltd", None, "Beta Plc"],
        }
    )

    index = build_target_index_from_frame(
        target_frame,
        target_id_col="system_uri",
        text_value_expr=pl.col("name")
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .str.to_lowercase(),
        value_col="_text",
        nonempty_predicate=pl.col("_text") != "",
        vectorizer=_FakeVectorizer(),
        empty_fit_placeholder="placeholder",
    )

    # The null-name row (t2) is dropped by nonempty_predicate before fitting.
    assert index.target_ids == ["t1", "t3"]
    assert index.target_matrix.shape == (2, 2)


def test_build_target_index_from_frame_handles_all_empty_target():
    target_frame = pl.DataFrame({"system_uri": ["t1"], "name": [None]})

    index = build_target_index_from_frame(
        target_frame,
        target_id_col="system_uri",
        text_value_expr=pl.col("name").cast(pl.Utf8, strict=False).fill_null(""),
        value_col="_text",
        nonempty_predicate=pl.col("_text") != "",
        vectorizer=_FakeVectorizer(),
        empty_fit_placeholder="placeholder",
    )

    assert index.target_ids == []
    assert index.target_matrix.shape == (0, 2)
