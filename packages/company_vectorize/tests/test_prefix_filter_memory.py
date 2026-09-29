"""Allocation-bound regression tests for the prefix-filtered `"sklearn"`
scoring path (`_score_source_with_sklearn_prefix_filtered`).

The bug these lock in was invisible to correctness tests: the function
returned the right answer, it just gathered every source row's candidate set
for the whole chunk before the first batched multiply, so peak allocation
scaled with `source_chunk_size x candidates-per-row`. Under a large sparse
TF-IDF vocabulary that is cheap, because a row's candidate set is small;
under a small dense subword vocabulary (WordPiece/SentencePiece) at real
target scale a single row's candidate set approaches `n_targets`, and a real
`gleif -> gb` run drove RSS from ~12GB to ~36GB with no scoring progress.

The fixture below reproduces that *shape* rather than that scale: a handful
of tokens shared by most rows, so each source row's prefix filter returns a
large fraction of all targets. Assertions are on `tracemalloc` peaks, which
observe NumPy/SciPy array allocations, and are written as ratios between two
runs of the same code rather than absolute byte thresholds, so they do not
depend on the machine.
"""

from __future__ import annotations

import gc
import random
import tracemalloc
from collections.abc import Callable
from typing import Any

import polars as pl
import pytest
from company_vectorize import resolve_target_index_build_settings, sparse_similarity
from company_vectorize.sparse_similarity import (
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy

# A deliberately tiny vocabulary shared by every row -- the dense-vocabulary
# shape WordPiece/SentencePiece produce, where "the row's rarest token" is
# still common enough to match a large fraction of the corpus.
_DENSE_VOCABULARY = [
    "alpha",
    "bravo",
    "charlie",
    "delta",
    "echo",
    "foxtrot",
    "golf",
    "hotel",
    "india",
    "juliet",
    "kilo",
    "lima",
]
_TOKENS_PER_ROW = 4


def _dense_vocabulary_name(rng: random.Random) -> str:
    return " ".join(rng.sample(_DENSE_VOCABULARY, _TOKENS_PER_ROW))


def _build_dense_vocabulary_case(n_targets: int, n_sources: int, seed: int):
    rng = random.Random(seed)
    target_ids = [f"t{i}" for i in range(n_targets)]
    target_names = [_dense_vocabulary_name(rng) for _ in range(n_targets)]
    source_ids = [f"s{i}" for i in range(n_sources)]
    source_names = [_dense_vocabulary_name(rng) for _ in range(n_sources)]

    strategy = TfidfClusteringStrategy()
    target_index = strategy.build_target_index(
        pl.DataFrame({"system_uri": target_ids, "name": target_names}),
        target_id_col="system_uri",
        text_col="name",
        build_settings=resolve_target_index_build_settings(
            "tfidf", tfidf_ngram_min=1, tfidf_ngram_max=1
        ),
    )
    src_matrix = target_index.vectorizer.transform(source_names)
    return target_index, source_ids, src_matrix


def _score(target_index, source_ids, src_matrix, *, max_per_source: int = 1):
    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="sklearn",
        top_k=max_per_source,
        max_candidates_per_source=None,
        backend_options={"prefix_filter": True},
        min_similarity=0.0,
    )
    assert backend_index is not None and backend_index.prefix_filter is not None
    return score_source_with_backend(
        source_ids,
        src_matrix,
        target_index=target_index,
        top_k=max_per_source,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=backend_index,
    )


def _peak_bytes(fn: Callable[[], Any]) -> tuple[Any, int]:
    gc.collect()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def _set_byte_budget(monkeypatch: pytest.MonkeyPatch, budget: int) -> None:
    monkeypatch.setattr(sparse_similarity, "_PAIR_SCORE_BYTE_BUDGET", budget)


def test_dense_vocabulary_source_row_matches_most_targets() -> None:
    """Guards the fixture itself: if the prefix filter stops returning a large
    fraction of all targets per source row, the memory assertions below stop
    testing anything.
    """
    from company_vectorize.prefix_filter import (
        build_prefix_filter_index,
        prefix_filter_candidates,
    )

    n_targets = 600
    target_index, _, src_matrix = _build_dense_vocabulary_case(n_targets, 1, seed=7)
    prefix_index = build_prefix_filter_index(
        target_index.target_matrix, min_similarity=0.0
    )
    assert prefix_index is not None

    row = src_matrix.tocsr()
    candidates = prefix_filter_candidates(
        row.indices[row.indptr[0] : row.indptr[1]],
        row.data[row.indptr[0] : row.indptr[1]],
        prefix_index,
    )
    assert candidates.size > n_targets // 2


@pytest.mark.integration
def test_peak_allocation_does_not_scale_with_source_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression proper: peak allocation must stay flat as the source
    chunk grows, since sub-batches are flushed at a fixed budget. The pre-fix
    gather-the-whole-chunk shape scaled peak linearly with the source row
    count and fails this outright.
    """
    _set_byte_budget(monkeypatch, 2 * 1024 * 1024)

    small_case = _build_dense_vocabulary_case(600, 100, seed=11)
    large_case = _build_dense_vocabulary_case(600, 800, seed=11)

    _, small_peak = _peak_bytes(lambda: _score(*small_case))
    _, large_peak = _peak_bytes(lambda: _score(*large_case))

    # 8x the source rows, so 8x the candidate pairs. Allowing 2x covers the
    # genuinely linear parts (the result rows, the source matrix itself);
    # unbounded accumulation lands near 8x.
    assert large_peak <= small_peak * 2, (
        f"peak allocation scaled with source chunk size: "
        f"{small_peak} bytes for 100 rows, {large_peak} bytes for 800 rows"
    )


@pytest.mark.integration
def test_bounded_batch_budget_lowers_peak_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget small enough to force several sub-batches must measurably
    lower peak allocation against a budget large enough to hold the whole
    chunk (which is exactly the pre-fix behaviour).
    """
    case = _build_dense_vocabulary_case(600, 800, seed=13)

    _set_byte_budget(monkeypatch, 1 << 40)
    _, unbounded_peak = _peak_bytes(lambda: _score(*case))

    _set_byte_budget(monkeypatch, 2 * 1024 * 1024)
    _, bounded_peak = _peak_bytes(lambda: _score(*case))

    assert bounded_peak * 4 <= unbounded_peak, (
        f"batching did not bound peak allocation: {bounded_peak} bytes bounded "
        f"vs {unbounded_peak} bytes unbounded"
    )


@pytest.mark.integration
def test_sub_batching_does_not_change_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bounding the batch must not perturb ranking: identical rows, in the
    same order, whether the chunk is scored in one multiply or many.
    """
    case = _build_dense_vocabulary_case(400, 250, seed=17)

    _set_byte_budget(monkeypatch, 1 << 40)
    unbounded_rows = _score(*case, max_per_source=5)

    _set_byte_budget(monkeypatch, 2 * 1024 * 1024)
    bounded_rows = _score(*case, max_per_source=5)

    assert bounded_rows == unbounded_rows
    assert unbounded_rows


def test_single_source_row_exceeding_batch_budget_is_scored_in_one_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single row whose candidate set alone exceeds the batch budget is
    sliced inside the multiply but still ranked once, so its top-k is exact
    and its ranks are not restarted per slice.
    """
    case = _build_dense_vocabulary_case(400, 3, seed=19)

    _set_byte_budget(monkeypatch, 1 << 40)
    unbounded_rows = _score(*case, max_per_source=5)

    # Floor and budget both driven below one row's candidate count, so
    # `_score_candidate_pairs` has to slice within a single source row.
    monkeypatch.setattr(sparse_similarity, "_MIN_PAIRS_PER_SCORE_BATCH", 1)
    _set_byte_budget(monkeypatch, 4096)
    sliced_rows = _score(*case, max_per_source=5)

    assert sliced_rows == unbounded_rows
    ranks_per_source: dict[str, list[int]] = {}
    for row in sliced_rows:
        ranks_per_source.setdefault(str(row["source_id"]), []).append(int(row["rank"]))
    assert ranks_per_source
    for ranks in ranks_per_source.values():
        assert ranks == sorted(ranks)
        assert ranks == list(range(1, len(ranks) + 1))
