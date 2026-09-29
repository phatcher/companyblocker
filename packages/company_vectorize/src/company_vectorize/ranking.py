from __future__ import annotations

import numpy as np


def append_ranked_matches(
    rows: list[dict[str, object]],
    *,
    src_id: str,
    target_ids: list[str],
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    min_similarity: float,
    max_per_source: int,
) -> None:
    """Rank one source row's candidates by score and append the kept ones to `rows`.

    `rows` is mutated in place. Shared by every similarity backend, namely
    `sparse_similarity.py`'s sklearn/sparse_dot_topn/svd_rerank and
    `partition_similarity.py`'s kmeans/hdbscan, so ranking and thresholding semantics stay
    identical across backends.
    """
    if max_per_source <= 0 or candidate_indices.size == 0 or candidate_scores.size == 0:
        return

    order = np.argsort(candidate_scores)[::-1]
    rank = 0
    for pos in order:
        score = float(candidate_scores[pos])
        if score < min_similarity:
            continue
        rank += 1
        rows.append(
            {
                "source_id": src_id,
                "target_id": target_ids[int(candidate_indices[pos])],
                "similarity": score,
                "rank": rank,
            }
        )
        if rank >= max_per_source:
            break


def append_ranked_match_block(
    rows: list[dict[str, object]],
    *,
    src_ids: list[str],
    target_ids: list[str],
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    min_similarity: float,
    max_per_source: int,
) -> None:
    """`append_ranked_matches` over a whole block of source rows at once.

    `candidate_indices` and `candidate_scores` are `(len(src_ids), k)`: row
    `i` holds source `src_ids[i]`'s candidates, as a backend's own search
    returned them. The ranking is the same as a row at a time -- each row's
    candidates ordered by score descending, those below `min_similarity`
    dropped, ranks numbered from one over what is kept, and the first
    `max_per_source` taken -- and rows are appended in source order, so a
    caller reads the same list either way. The ordering of two candidates of
    one row at the same score is `numpy.argsort`'s, as it is there.

    The work per row is numpy's rather than Python's: a source row costs a
    few array slices instead of a loop over its candidates, which is what a
    backend scoring hundreds of thousands of rows pays.
    """
    if max_per_source <= 0 or candidate_indices.size == 0:
        return

    scores = np.asarray(candidate_scores, dtype=np.float64)
    indices = np.asarray(candidate_indices)
    if scores.ndim != 2 or indices.shape != scores.shape:
        raise ValueError(
            "candidate indices and scores are one block of shape "
            f"(sources, k); got {indices.shape} and {scores.shape}"
        )

    order = np.argsort(scores, axis=1)[:, ::-1]
    ordered_scores = np.take_along_axis(scores, order, axis=1)
    ordered_indices = np.take_along_axis(indices, order, axis=1)

    kept = ordered_scores >= min_similarity
    # A rank counts only the candidates kept before it, and nothing past the
    # cap survives, which is the row-at-a-time loop's `rank` and its break.
    ranks = np.cumsum(kept, axis=1)
    kept &= ranks <= max_per_source

    row_positions, column_positions = np.nonzero(kept)
    rows.extend(
        {
            "source_id": src_ids[int(row_position)],
            "target_id": target_ids[int(ordered_indices[row_position, column])],
            "similarity": float(ordered_scores[row_position, column]),
            "rank": int(ranks[row_position, column]),
        }
        for row_position, column in zip(row_positions, column_positions, strict=True)
    )
