"""Unit coverage for `ranking.append_ranked_matches`.

This is the shared ranking/thresholding seam every similarity backend calls
(`sparse_similarity.py`'s sklearn/sparse_dot_topn/svd_rerank,
`partition_similarity.py`'s kmeans/hdbscan), so its own contract -- sort
descending, drop below `min_similarity`, cap at `max_per_source`, assign
1-based rank in kept order -- is asserted here directly rather than through
whichever backend test happens to exercise it.
"""

from __future__ import annotations

import numpy as np
from company_vectorize.ranking import (
    append_ranked_match_block,
    append_ranked_matches,
)


def test_appends_rows_sorted_by_descending_score_with_rank():
    rows: list[dict[str, object]] = []

    append_ranked_matches(
        rows,
        src_id="s1",
        target_ids=["t0", "t1", "t2"],
        candidate_indices=np.array([0, 1, 2]),
        candidate_scores=np.array([0.2, 0.9, 0.5]),
        min_similarity=0.0,
        max_per_source=10,
    )

    assert [row["target_id"] for row in rows] == ["t1", "t2", "t0"]
    assert [row["rank"] for row in rows] == [1, 2, 3]
    assert [row["similarity"] for row in rows] == [0.9, 0.5, 0.2]
    assert all(row["source_id"] == "s1" for row in rows)


def test_filters_matches_below_min_similarity():
    rows: list[dict[str, object]] = []

    append_ranked_matches(
        rows,
        src_id="s1",
        target_ids=["t0", "t1"],
        candidate_indices=np.array([0, 1]),
        candidate_scores=np.array([0.1, 0.8]),
        min_similarity=0.5,
        max_per_source=10,
    )

    assert [row["target_id"] for row in rows] == ["t1"]


def test_stops_at_max_per_source():
    rows: list[dict[str, object]] = []

    append_ranked_matches(
        rows,
        src_id="s1",
        target_ids=["t0", "t1", "t2"],
        candidate_indices=np.array([0, 1, 2]),
        candidate_scores=np.array([0.9, 0.8, 0.7]),
        min_similarity=0.0,
        max_per_source=2,
    )

    assert [row["target_id"] for row in rows] == ["t0", "t1"]
    assert [row["rank"] for row in rows] == [1, 2]


def test_noop_when_max_per_source_is_not_positive():
    rows: list[dict[str, object]] = []

    append_ranked_matches(
        rows,
        src_id="s1",
        target_ids=["t0"],
        candidate_indices=np.array([0]),
        candidate_scores=np.array([0.9]),
        min_similarity=0.0,
        max_per_source=0,
    )

    assert rows == []


def test_noop_when_candidates_are_empty():
    rows: list[dict[str, object]] = []

    append_ranked_matches(
        rows,
        src_id="s1",
        target_ids=[],
        candidate_indices=np.array([]),
        candidate_scores=np.array([]),
        min_similarity=0.0,
        max_per_source=10,
    )

    assert rows == []


def test_appends_to_rows_already_populated_by_a_prior_source():
    rows: list[dict[str, object]] = [
        {"source_id": "s0", "target_id": "t9", "similarity": 0.4, "rank": 1}
    ]

    append_ranked_matches(
        rows,
        src_id="s1",
        target_ids=["t0"],
        candidate_indices=np.array([0]),
        candidate_scores=np.array([0.9]),
        min_similarity=0.0,
        max_per_source=10,
    )

    assert len(rows) == 2
    assert rows[0]["source_id"] == "s0"
    assert rows[1] == {
        "source_id": "s1",
        "target_id": "t0",
        "similarity": 0.9,
        "rank": 1,
    }


def test_the_block_form_gives_what_a_row_at_a_time_gives():
    """`append_ranked_match_block` is the same ranking over many rows at
    once: the rows it appends are what the loop it replaces appended, ties,
    thresholds and caps included."""
    rng = np.random.default_rng(7)
    target_ids = [f"t{index}" for index in range(12)]
    # Repeated scores so ties are exercised, and a row of them all below the
    # threshold so an empty row is too.
    scores = np.round(rng.random((6, 5)), 2)
    scores[3] = 0.1
    indices = rng.integers(0, len(target_ids), size=(6, 5))
    src_ids = [f"s{index}" for index in range(6)]

    for min_similarity, max_per_source in ((0.0, 5), (0.5, 3), (0.99, 5)):
        one_at_a_time: list[dict[str, object]] = []
        for position, src_id in enumerate(src_ids):
            append_ranked_matches(
                one_at_a_time,
                src_id=src_id,
                target_ids=target_ids,
                candidate_indices=indices[position],
                candidate_scores=scores[position],
                min_similarity=min_similarity,
                max_per_source=max_per_source,
            )
        as_a_block: list[dict[str, object]] = []
        append_ranked_match_block(
            as_a_block,
            src_ids=src_ids,
            target_ids=target_ids,
            candidate_indices=indices,
            candidate_scores=scores,
            min_similarity=min_similarity,
            max_per_source=max_per_source,
        )

        assert as_a_block == one_at_a_time


def test_a_padded_candidate_slot_is_dropped():
    """A backend whose search gave a row fewer candidates than the block is
    wide pads the rest, and a padded slot never becomes a row."""
    rows: list[dict[str, object]] = []

    append_ranked_match_block(
        rows,
        src_ids=["s0"],
        target_ids=["t0", "t1"],
        candidate_indices=np.array([[1, 0]]),
        candidate_scores=np.array([[0.8, -np.inf]]),
        min_similarity=0.0,
        max_per_source=2,
    )

    assert rows == [
        {"source_id": "s0", "target_id": "t1", "similarity": 0.8, "rank": 1}
    ]
