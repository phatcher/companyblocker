"""Unit coverage for the `"hnsw"` sub-linear (usearch ANN) backend.

Shaped like `test_dense_brute_similarity.py`: generated vectors only, never
`sentence_transformers`, which crashes when imported inside pytest on this
machine. `"hnsw"` is approximate, so its correctness bar is recall against
`"dense_brute"`'s exact reference at a fixed seed, not bit-for-bit agreement.
"""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
from company_vectorize.clustering_contract import TargetClusteringIndex
from company_vectorize.dense_brute_similarity import (
    build_target_dense_brute_index,
    score_source_with_dense_brute_backend,
)
from company_vectorize.hnsw_similarity import (
    DEFAULT_HNSW_BUILD_THREADS,
    DEFAULT_HNSW_CONNECTIVITY,
    DEFAULT_HNSW_EXPANSION_ADD,
    DEFAULT_HNSW_EXPANSION_SEARCH,
    DEFAULT_HNSW_STORAGE_DTYPE,
    HNSW_STORAGE_DTYPES,
    TargetHnswIndex,
    build_target_hnsw_index,
    hnsw_build_settings,
    score_source_with_hnsw_backend,
)
from company_vectorize.sparse_similarity import (
    build_target_similarity_backend_index,
    score_source_with_backend,
)
from scipy.sparse import csr_matrix


def _target_index(n_targets: int, dim: int, *, seed: int = 0) -> TargetClusteringIndex:
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n_targets, dim)).astype(np.float32)
    target_ids = [f"t{i}" for i in range(n_targets)]
    return TargetClusteringIndex(
        target_ids=target_ids, vectorizer=None, target_matrix=matrix
    )


def _source_matrix(n_sources: int, dim: int, *, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_sources, dim)).astype(np.float32)


def _score_hnsw(
    target_index: TargetClusteringIndex,
    src_matrix: np.ndarray,
    *,
    top_k: int,
    min_similarity: float = 0.0,
    max_candidates_per_source: int | None = None,
    backend_options: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    src_ids = [f"s{i}" for i in range(src_matrix.shape[0])]
    backend_index = build_target_similarity_backend_index(
        target_index,
        backend="hnsw",
        top_k=top_k,
        max_candidates_per_source=max_candidates_per_source,
        backend_options=backend_options,
    )
    return score_source_with_backend(
        src_ids,
        src_matrix,
        target_index=target_index,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        nn_index=None,
        backend="hnsw",
        backend_index=backend_index,
    )


def _score_dense_brute(
    target_index: TargetClusteringIndex, src_matrix: np.ndarray, *, top_k: int
) -> list[dict[str, object]]:
    src_ids = [f"s{i}" for i in range(src_matrix.shape[0])]
    dense_brute_index = build_target_dense_brute_index(
        target_index, backend_options={"storage_dtype": "float32"}
    )
    return score_source_with_dense_brute_backend(
        src_ids,
        src_matrix,
        target_index=target_index,
        dense_brute_index=dense_brute_index,
        top_k=top_k,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )


def _rows_by_source(
    rows: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_source: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_source.setdefault(str(row["source_id"]), []).append(row)
    return by_source


def _key(row: dict[str, object]) -> tuple[str, int]:
    return (str(row["source_id"]), cast(int, row["rank"]))


def test_defaults_are_standard_hnsw_knobs():
    assert DEFAULT_HNSW_STORAGE_DTYPE == "float16"
    assert DEFAULT_HNSW_CONNECTIVITY == 16
    assert DEFAULT_HNSW_EXPANSION_ADD == 128
    assert DEFAULT_HNSW_EXPANSION_SEARCH == 64
    assert set(HNSW_STORAGE_DTYPES) == {"float16", "int8"}


def test_sparse_target_matrix_is_refused():
    dense_index = _target_index(10, 4)
    sparse_index = TargetClusteringIndex(
        target_ids=dense_index.target_ids,
        vectorizer=None,
        target_matrix=csr_matrix(dense_index.target_matrix),
    )
    with pytest.raises(ValueError, match="dense target matrix"):
        build_target_hnsw_index(sparse_index)


def test_empty_target_returns_none_index():
    empty_index = TargetClusteringIndex(
        target_ids=[], vectorizer=None, target_matrix=np.empty((0, 4), dtype=np.float32)
    )
    assert build_target_hnsw_index(empty_index) is None


def test_none_index_scores_to_no_rows():
    rows = score_source_with_hnsw_backend(
        ["s1"],
        np.zeros((1, 4), dtype=np.float32),
        target_index=_target_index(5, 4),
        hnsw_index=None,
        top_k=3,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )
    assert rows == []


def test_empty_source_chunk_scores_to_no_rows():
    target_index = _target_index(5, 4)
    hnsw_index = build_target_hnsw_index(target_index)
    rows = score_source_with_hnsw_backend(
        [],
        np.empty((0, 4), dtype=np.float32),
        target_index=target_index,
        hnsw_index=hnsw_index,
        top_k=3,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )
    assert rows == []


@pytest.mark.parametrize(
    ("option", "value", "match"),
    [
        ("storage_dtype", "float32", "storage_dtype"),
        ("connectivity", 0, "connectivity"),
        ("connectivity", -1, "connectivity"),
        ("connectivity", "lots", "connectivity"),
        ("connectivity", True, "connectivity"),
        ("expansion_add", 0, "expansion_add"),
        ("expansion_search", 0, "expansion_search"),
    ],
)
def test_unreadable_backend_options_are_rejected(option, value, match):
    with pytest.raises(ValueError, match=match):
        build_target_hnsw_index(_target_index(10, 4), backend_options={option: value})


def test_hnsw_build_settings_excludes_search_time_options():
    # expansion_search (and any non-build key like index_dir/top_k) must not
    # change the build-settings key: a cache directory keyed on it would
    # rebuild for a change that only affects search, not the saved graph
    # (module docstring).
    settings = hnsw_build_settings(
        {"expansion_search": 999, "index_dir": "/some/path", "connectivity": 8}
    )
    assert settings == {
        "storage_dtype": "float16",
        "connectivity": 8,
        "expansion_add": DEFAULT_HNSW_EXPANSION_ADD,
        "build_threads": DEFAULT_HNSW_BUILD_THREADS,
    }


def test_recall_against_dense_brute_meets_the_bar():
    # The bar: recall >= 0.95 at default settings, fixed seed.
    target_index = _target_index(3000, 32, seed=7)
    src_matrix = _source_matrix(100, 32, seed=8)
    top_k = 10

    hnsw_rows = _score_hnsw(target_index, src_matrix, top_k=top_k)
    exact_rows = _score_dense_brute(target_index, src_matrix, top_k=top_k)

    hnsw_by_source = _rows_by_source(hnsw_rows)
    exact_by_source = _rows_by_source(exact_rows)

    recalls = []
    for src_index in range(src_matrix.shape[0]):
        src_id = f"s{src_index}"
        hnsw_targets = {row["target_id"] for row in hnsw_by_source.get(src_id, [])}
        exact_targets = {row["target_id"] for row in exact_by_source.get(src_id, [])}
        if not exact_targets:
            continue
        recalls.append(len(hnsw_targets & exact_targets) / len(exact_targets))

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.95, (
        f"mean recall {mean_recall} against dense_brute at top_k={top_k}"
    )


@pytest.mark.parametrize(
    ("top_k", "min_similarity", "max_candidates_per_source"),
    [
        (5, 0.0, None),
        (5, 0.0, 2),
        (10, -1.0, None),
        (1, 0.0, None),
    ],
)
def test_hnsw_honours_every_pruning_parameter(
    top_k, min_similarity, max_candidates_per_source
):
    # Structural assertions on hnsw's own output, not agreement with an
    # exact backend's answer set: an approximate index can legitimately find
    # a slightly different top-k than an exact scan even at generous
    # expansion settings (recall against dense_brute is covered separately,
    # above, with a margin measured directly), so this test isolates
    # ranking/thresholding/capping -- append_ranked_matches()'s contract,
    # shared by every backend -- from ANN recall.
    target_index = _target_index(300, 8, seed=3)
    src_matrix = _source_matrix(10, 8, seed=4)
    max_per_source = (
        top_k
        if max_candidates_per_source is None
        else min(top_k, max_candidates_per_source)
    )

    hnsw_rows = _score_hnsw(
        target_index,
        src_matrix,
        top_k=top_k,
        min_similarity=min_similarity,
        max_candidates_per_source=max_candidates_per_source,
        backend_options={"expansion_add": 256, "expansion_search": 256},
    )

    by_source = _rows_by_source(hnsw_rows)
    for src_index in range(src_matrix.shape[0]):
        rows = sorted(
            by_source.get(f"s{src_index}", []), key=lambda row: int(row["rank"])
        )
        assert len(rows) <= max_per_source
        assert [int(row["rank"]) for row in rows] == list(range(1, len(rows) + 1))
        similarities = [float(row["similarity"]) for row in rows]
        assert similarities == sorted(similarities, reverse=True)
        assert all(score >= min_similarity for score in similarities)
        assert len({row["target_id"] for row in rows}) == len(rows)


@pytest.mark.parametrize("storage_dtype", ["float16", "int8"])
def test_both_storage_widths_complete_and_rank_a_near_duplicate_top(storage_dtype):
    # Both widths are lossy (module docstring), so this only asserts the
    # backend completes and still finds a source row's near-exact duplicate
    # target as its top match, not bit-for-bit agreement with float32.
    target_index = _target_index(150, 16, seed=5)
    src_matrix = target_index.target_matrix[:5].copy()

    rows = _score_hnsw(
        target_index,
        src_matrix,
        top_k=3,
        backend_options={"storage_dtype": storage_dtype},
    )
    by_source = _rows_by_source(rows)
    for src_index in range(5):
        top = min(by_source[f"s{src_index}"], key=lambda row: int(row["rank"]))
        assert top["target_id"] == target_index.target_ids[src_index]


def test_memmap_target_gives_the_same_result_as_in_memory(tmp_path):
    target_index = _target_index(120, 10, seed=6)
    src_matrix = _source_matrix(8, 10, seed=9)
    top_k = 4
    options = {"expansion_add": 256, "expansion_search": 256}

    in_memory_rows = _score_hnsw(
        target_index, src_matrix, top_k=top_k, backend_options=options
    )

    path = tmp_path / "target.dat"
    memmap = np.memmap(
        path, dtype=np.float32, mode="w+", shape=target_index.target_matrix.shape
    )
    memmap[:] = target_index.target_matrix
    memmap.flush()
    memmap_index = TargetClusteringIndex(
        target_ids=target_index.target_ids, vectorizer=None, target_matrix=memmap
    )
    memmap_rows = _score_hnsw(
        memmap_index, src_matrix, top_k=top_k, backend_options=options
    )

    in_memory_rows = sorted(in_memory_rows, key=_key)
    memmap_rows = sorted(memmap_rows, key=_key)
    assert [(row["source_id"], row["target_id"]) for row in in_memory_rows] == [
        (row["source_id"], row["target_id"]) for row in memmap_rows
    ]


def test_saved_index_reopens_with_identical_neighbours(tmp_path):
    target_index = _target_index(400, 12, seed=11)
    src_matrix = _source_matrix(20, 12, seed=12)
    top_k = 5
    index_dir = tmp_path / "hnsw-index"

    built_index = build_target_hnsw_index(
        target_index, backend_options={"index_dir": str(index_dir)}
    )
    assert isinstance(built_index, TargetHnswIndex)
    assert built_index.reopened is False
    built_rows = score_source_with_hnsw_backend(
        [f"s{i}" for i in range(src_matrix.shape[0])],
        src_matrix,
        target_index=target_index,
        hnsw_index=built_index,
        top_k=top_k,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )

    reopened_index = build_target_hnsw_index(
        target_index, backend_options={"index_dir": str(index_dir)}
    )
    assert isinstance(reopened_index, TargetHnswIndex)
    assert reopened_index.reopened is True
    reopened_rows = score_source_with_hnsw_backend(
        [f"s{i}" for i in range(src_matrix.shape[0])],
        src_matrix,
        target_index=target_index,
        hnsw_index=reopened_index,
        top_k=top_k,
        min_similarity=0.0,
        max_candidates_per_source=None,
    )

    assert sorted(built_rows, key=_key) == sorted(reopened_rows, key=_key)


def test_reopen_ignores_saved_expansion_search_and_applies_the_new_value(tmp_path):
    # expansion_search is a search-time knob, not baked into the saved graph
    # (module docstring): a second call with a *different* expansion_search
    # still reopens (reopened=True) rather than rebuilding.
    target_index = _target_index(80, 8, seed=13)
    index_dir = tmp_path / "hnsw-index"

    def _build(expansion_search):
        options = {"index_dir": str(index_dir), "expansion_search": expansion_search}
        return build_target_hnsw_index(target_index, backend_options=options)

    first = _build(64)
    assert first.reopened is False
    assert first.expansion_search == 64

    second = _build(32)
    assert second.reopened is True
    assert second.expansion_search == 32
    assert second.index.expansion_search == 32


def test_manifest_records_the_build_settings(tmp_path):
    target_index = _target_index(30, 6, seed=14)
    index_dir = tmp_path / "hnsw-index"
    build_target_hnsw_index(
        target_index,
        backend_options={
            "index_dir": str(index_dir),
            "storage_dtype": "int8",
            "connectivity": 24,
            "expansion_add": 96,
            "build_threads": 2,
        },
    )
    manifest = json.loads(
        (index_dir / "hnsw_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == {
        "storage_dtype": "int8",
        "connectivity": 24,
        "expansion_add": 96,
        "build_threads": 2,
        "metric": "cos",
        "ndim": 6,
        "count": 30,
    }
