"""Regression coverage for `similarity_strategy_helpers.py`'s scoring
orchestration half, exercised directly against a hand-built
`TargetClusteringIndex` -- no target-frame building
(`target_index_lifecycle.py`) involved, proving the scoring and
index-building halves are independently testable.
"""

import numpy as np
import polars as pl
import pytest
from company_vectorize.clustering_contract import TargetClusteringIndex
from company_vectorize.similarity_strategy_helpers import (
    SCORING_STEP_BACKEND,
    SCORING_STEP_VECTORIZE_SOURCE,
    BaseSimilarityScoringStrategy,
    _score_rows_with_backend,
    empty_similarity_frame,
    score_source_chunk_from_frame,
    validate_source_scoring_inputs,
)


class _FakeVectorizer:
    """`.transform()`-only stand-in: scoring never calls `.fit()`/
    `.fit_transform()`, only `.transform()` on the already-fitted target
    vectorizer -- this fake enforces that by not implementing either.
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    def transform(self, values: list[str]) -> np.ndarray:
        return np.array([self._vectors[v] for v in values], dtype=np.float32)


def _hand_built_target_index() -> TargetClusteringIndex:
    vectors = {
        "acme ltd": [1.0, 0.0],
        "acme limited": [0.98, 0.02],
        "omega plc": [0.0, 1.0],
    }
    return TargetClusteringIndex(
        target_ids=["t-acme", "t-omega"],
        vectorizer=_FakeVectorizer(vectors),
        target_matrix=np.array(
            [vectors["acme ltd"], vectors["omega plc"]], dtype=np.float32
        ),
    )


def test_validate_source_scoring_inputs_passes_for_valid_args():
    frame = pl.DataFrame({"source_id": ["s1"], "name": ["Acme"]})
    validate_source_scoring_inputs(
        frame, source_id_col="source_id", text_col="name", top_k=5, min_similarity=0.5
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "source_id_col": "missing",
                "text_col": "name",
                "top_k": 5,
                "min_similarity": 0.5,
            },
            "missing required column 'missing'",
        ),
        (
            {
                "source_id_col": "source_id",
                "text_col": "missing",
                "top_k": 5,
                "min_similarity": 0.5,
            },
            "missing required column 'missing'",
        ),
        (
            {
                "source_id_col": "source_id",
                "text_col": "name",
                "top_k": 0,
                "min_similarity": 0.5,
            },
            "top_k must be greater than zero",
        ),
        (
            {
                "source_id_col": "source_id",
                "text_col": "name",
                "top_k": 5,
                "min_similarity": 1.5,
            },
            "min_similarity must be between 0 and 1",
        ),
    ],
)
def test_validate_source_scoring_inputs_rejects_invalid_args(kwargs, match):
    frame = pl.DataFrame({"source_id": ["s1"], "name": ["Acme"]})
    with pytest.raises(ValueError, match=match):
        validate_source_scoring_inputs(frame, **kwargs)


def test_empty_similarity_frame_has_expected_schema_and_zero_rows():
    frame = empty_similarity_frame()
    assert frame.height == 0
    assert frame.columns == ["source_id", "target_id", "similarity", "rank"]


def test_score_rows_with_backend_ranks_nearest_target_first():
    target_index = _hand_built_target_index()
    source_nonempty = pl.DataFrame({"source_id": ["s1"], "_text": ["acme limited"]})

    rows = _score_rows_with_backend(
        source_nonempty,
        value_col="_text",
        target_index=target_index,
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=None,
    )

    assert rows
    top = min(rows, key=lambda r: r["rank"])
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t-acme"


def test_score_source_chunk_from_frame_returns_empty_for_empty_target_index():
    empty_target_index = TargetClusteringIndex(
        target_ids=[], vectorizer=_FakeVectorizer({}), target_matrix=np.empty((0, 2))
    )
    source_frame = pl.DataFrame({"source_id": ["s1"], "name": ["Acme"]})

    result = score_source_chunk_from_frame(
        source_frame,
        target_index=empty_target_index,
        source_id_col="source_id",
        text_col="name",
        text_value_expr=pl.col("name"),
        value_col="_text",
        nonempty_predicate=pl.col("_text") != "",
        top_k=5,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=None,
    )

    assert result.height == 0
    assert result.columns == ["source_id", "target_id", "similarity", "rank"]


def test_score_source_chunk_from_frame_scores_nonempty_rows():
    target_index = _hand_built_target_index()
    source_frame = pl.DataFrame(
        {"source_id": ["s1", "s2"], "name": ["acme limited", "omega plc"]}
    )

    result = score_source_chunk_from_frame(
        source_frame,
        target_index=target_index,
        source_id_col="source_id",
        text_col="name",
        text_value_expr=pl.col("name"),
        value_col="_text",
        nonempty_predicate=pl.col("_text") != "",
        top_k=1,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=None,
    )

    assert result.height == 2
    by_source = {row["source_id"]: row["target_id"] for row in result.to_dicts()}
    assert by_source["s1"] == "t-acme"
    assert by_source["s2"] == "t-omega"


def test_score_source_chunk_from_frame_adds_each_steps_seconds_to_timings():
    target_index = _hand_built_target_index()
    source_frame = pl.DataFrame({"source_id": ["s1"], "name": ["acme limited"]})
    timings: dict[str, float] = {}

    for _ in range(2):
        score_source_chunk_from_frame(
            source_frame,
            target_index=target_index,
            source_id_col="source_id",
            text_col="name",
            text_value_expr=pl.col("name"),
            value_col="_text",
            nonempty_predicate=pl.col("_text") != "",
            top_k=1,
            min_similarity=0.0,
            max_candidates_per_source=None,
            nn_index=None,
            backend="sklearn",
            backend_index=None,
            timings=timings,
        )

    assert set(timings) == {SCORING_STEP_VECTORIZE_SOURCE, SCORING_STEP_BACKEND}
    assert all(seconds >= 0.0 for seconds in timings.values())


def test_score_source_chunk_from_frame_is_deterministic_across_repeated_calls():
    """Identical input scored twice must produce identical output.

    Proves the candidate-pairing protocol (schema, backend dispatch, ranking)
    is reproducible end to end, not merely "same set of rows in some order" --
    row order and rank assignment must match exactly, since downstream
    consumers (`src/blocking/workflow.py`) treat this frame as stable output.
    """
    target_index = _hand_built_target_index()
    source_frame = pl.DataFrame(
        {"source_id": ["s1", "s2"], "name": ["acme limited", "omega plc"]}
    )
    kwargs = {
        "target_index": target_index,
        "source_id_col": "source_id",
        "text_col": "name",
        "text_value_expr": pl.col("name"),
        "value_col": "_text",
        "nonempty_predicate": pl.col("_text") != "",
        "top_k": 2,
        "min_similarity": 0.0,
        "max_candidates_per_source": None,
        "nn_index": None,
        "backend": "sklearn",
        "backend_index": None,
    }

    first = score_source_chunk_from_frame(source_frame, **kwargs)
    second = score_source_chunk_from_frame(source_frame, **kwargs)

    assert first.equals(second)


def test_tied_similarity_candidates_rank_deterministically_across_repeated_calls():
    """Two targets equidistant from the source must still rank the same way
    every time the identical input is scored, even though `append_ranked_matches`
    makes no promise about *which* tied candidate ranks first.
    """
    vectors = {
        "acme": [1.0, 0.0],
        "acme co": [1.0, 0.0],
        "acme corp": [1.0, 0.0],
    }
    target_index = TargetClusteringIndex(
        target_ids=["t-a", "t-b", "t-c"],
        vectorizer=_FakeVectorizer(vectors),
        target_matrix=np.array(
            [vectors["acme"], vectors["acme co"], vectors["acme corp"]],
            dtype=np.float32,
        ),
    )
    source_frame = pl.DataFrame({"source_id": ["s1"], "name": ["acme"]})
    kwargs = {
        "target_index": target_index,
        "source_id_col": "source_id",
        "text_col": "name",
        "text_value_expr": pl.col("name"),
        "value_col": "_text",
        "nonempty_predicate": pl.col("_text") != "",
        "top_k": 3,
        "min_similarity": 0.0,
        "max_candidates_per_source": None,
        "nn_index": None,
        "backend": "sklearn",
        "backend_index": None,
    }

    results = [score_source_chunk_from_frame(source_frame, **kwargs) for _ in range(5)]

    assert all(r.height == 3 for r in results)
    first = results[0]
    for other in results[1:]:
        assert first.equals(other)


class _RecordingStrategy(BaseSimilarityScoringStrategy):
    """Concrete strategy under test: only overrides the source projection to
    prove `BaseSimilarityScoringStrategy.score_source_chunk()` wires the
    projection into `score_source_chunk_from_frame()` correctly, without
    pulling in any real strategy's target-index-building logic.
    """


def test_base_similarity_scoring_strategy_uses_default_projection():
    target_index = _hand_built_target_index()
    strategy = _RecordingStrategy()
    source_frame = pl.DataFrame({"source_id": ["s1"], "name": [" acme limited "]})

    result = strategy.score_source_chunk(
        source_frame,
        target_index=target_index,
        source_id_col="source_id",
        text_col="name",
        top_k=1,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=None,
        backend_options=None,
    )

    assert result.height == 1
    assert result.row(0, named=True)["target_id"] == "t-acme"
