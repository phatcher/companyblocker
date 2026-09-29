from __future__ import annotations

import importlib.metadata
import json
from typing import ClassVar

import numpy as np
import polars as pl
from company_vectorize.clustering_contract import (
    SbertTargetIndexBuildSettings,
    TfidfTargetIndexBuildSettings,
)
from company_vectorize.sbert_strategy import SbertClusteringStrategy
from company_vectorize.tfidf_strategy import TfidfClusteringStrategy

from validation import target_index_cache as cache
from workspace.roots import WorkspaceRoots


class _FakeSbertEncoder:
    """Deterministic stand-in for a real sentence-transformer encoder,
    injected through `SbertClusteringStrategy(encoder_factory=...)` the same
    way `test_runner.py`'s `_RecordingFixedVectorEncoder` is: the optional
    `sentence-transformers` dependency is not installed here, and importing
    it inside pytest crashes the process on this machine."""

    _VECTORS: ClassVar[dict[str, list[float]]] = {
        "acme": [1.0, 0.0],
        "beta": [0.0, 1.0],
        "gamma": [0.5, 0.5],
    }

    def __init__(self) -> None:
        self.seen_texts: list[str] = []

    def fit(self, texts: list[str]) -> _FakeSbertEncoder:
        _ = texts
        return self

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        self.seen_texts.extend(texts)
        return np.array([self._VECTORS[text] for text in texts], dtype=np.float32)


def test_library_versions_reads_metadata_without_importing(monkeypatch) -> None:
    """Reads each version through `importlib.metadata.version()`: present
    for a library whose distribution metadata resolves, absent for one
    whose metadata is not found, and never imports either library to get
    there."""

    def _fake_version(name: str) -> str:
        if name == "sentence_transformers":
            return "9.9.9"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cache.importlib.metadata, "version", _fake_version)

    result = json.loads(cache._library_versions())

    assert result == {"sentence_transformers": "9.9.9"}


def _frame(rows: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame({"system_uri": list(rows.keys()), "name": list(rows.values())})


def _sbert_settings(population_key: str) -> dict[str, object]:
    return cache.build_target_index_cache_settings(
        population_key=population_key,
        representation="sbert",
        text_view="name",
        tfidf_ngram_min=None,
        tfidf_ngram_max=None,
        tfidf_analyzer=None,
        similarity_backend="sklearn",
        backend_options=None,
        token_cache_key=None,
        tokenizer="wordpiece",
        tokenizer_scope="country",
        strategy="SbertClusteringStrategy",
    )


def test_resolve_target_index_sbert_hit_encodes_nothing(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A second run against the same target under the same model encodes
    nothing on the target side."""
    encoder = _FakeSbertEncoder()
    strategy = SbertClusteringStrategy(encoder_factory=lambda _settings: encoder)
    build_settings = SbertTargetIndexBuildSettings(model_name="fake-model")
    frame = _frame({"t:1": "acme", "t:2": "beta"})
    settings = _sbert_settings("pop-1")

    index_first, hit_first, _ = cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=settings,
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=frame,
        target_id_col="system_uri",
        text_col="name",
    )
    assert hit_first is False
    assert sorted(encoder.seen_texts) == ["acme", "beta"]
    assert set(index_first.target_ids) == {"t:1", "t:2"}

    encoder.seen_texts.clear()
    index_second, hit_second, _ = cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=settings,
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=frame,
        target_id_col="system_uri",
        text_col="name",
    )
    assert hit_second is True
    assert encoder.seen_texts == []
    assert set(index_second.target_ids) == {"t:1", "t:2"}
    stored = dict(zip(index_second.target_ids, index_second.target_matrix, strict=True))
    assert index_second.target_matrix.dtype == np.float32
    np.testing.assert_array_equal(stored["t:1"], [1.0, 0.0])
    np.testing.assert_array_equal(stored["t:2"], [0.0, 1.0])


def test_resolve_target_index_sbert_growth_encodes_only_new_rows(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A run whose target gained rows encodes only those rows -- the row
    store survives the growth rather than invalidating wholesale."""
    encoder = _FakeSbertEncoder()
    strategy = SbertClusteringStrategy(encoder_factory=lambda _settings: encoder)
    build_settings = SbertTargetIndexBuildSettings(model_name="fake-model")

    cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=_sbert_settings("pop-1"),
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=_frame({"t:1": "acme", "t:2": "beta"}),
        target_id_col="system_uri",
        text_col="name",
    )
    encoder.seen_texts.clear()

    index_second, hit_second, _ = cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=_sbert_settings("pop-2"),
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=_frame({"t:1": "acme", "t:2": "beta", "t:3": "gamma"}),
        target_id_col="system_uri",
        text_col="name",
    )

    assert hit_second is False
    assert encoder.seen_texts == ["gamma"]
    assert set(index_second.target_ids) == {"t:1", "t:2", "t:3"}


def test_resolve_target_index_sbert_reuses_embedding_across_target_systems(
    workspace_roots: WorkspaceRoots,
) -> None:
    """The row store is keyed on model identity alone, never on
    `target_system`/`country`: the same `system_uri` text encoded once under
    one target is reused when a different target's cache resolves it."""
    encoder = _FakeSbertEncoder()
    strategy = SbertClusteringStrategy(encoder_factory=lambda _settings: encoder)
    build_settings = SbertTargetIndexBuildSettings(model_name="fake-model")

    cache.resolve_target_index(
        roots=workspace_roots,
        target_system="gb",
        country="gb",
        settings=_sbert_settings("pop-1"),
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=_frame({"shared:1": "acme"}),
        target_id_col="system_uri",
        text_col="name",
    )
    encoder.seen_texts.clear()

    _, hit, _ = cache.resolve_target_index(
        roots=workspace_roots,
        target_system="ie",
        country="ie",
        settings=_sbert_settings("pop-2"),
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=_frame({"shared:1": "acme"}),
        target_id_col="system_uri",
        text_col="name",
    )

    assert hit is True
    assert encoder.seen_texts == []


def _tfidf_settings(population_key: str) -> dict[str, object]:
    return cache.build_target_index_cache_settings(
        population_key=population_key,
        representation="tfidf",
        text_view="name",
        tfidf_ngram_min=1,
        tfidf_ngram_max=2,
        tfidf_analyzer="char_wb",
        similarity_backend="sklearn",
        backend_options=None,
        token_cache_key=None,
        tokenizer="wordpiece",
        tokenizer_scope="country",
        strategy="TfidfClusteringStrategy",
    )


def test_resolve_target_index_snapshot_reuses_cache_on_hit(
    workspace_roots: WorkspaceRoots, mocker
) -> None:
    strategy = TfidfClusteringStrategy()
    build_spy = mocker.spy(strategy, "build_target_index")
    build_settings = TfidfTargetIndexBuildSettings(ngram_min=1, ngram_max=2)
    frame = _frame({"t:1": "acme limited", "t:2": "beta holdings"})
    settings = _tfidf_settings("pop-x")

    index_first, hit_first, _ = cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=settings,
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=frame,
        target_id_col="system_uri",
        text_col="name",
    )
    assert hit_first is False
    assert build_spy.call_count == 1

    index_second, hit_second, _ = cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=settings,
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=frame,
        target_id_col="system_uri",
        text_col="name",
    )
    assert hit_second is True
    assert build_spy.call_count == 1
    assert index_second.target_ids == index_first.target_ids


def test_resolve_target_index_snapshot_key_changes_with_population(
    workspace_roots: WorkspaceRoots, mocker
) -> None:
    """A cleanse-layer re-run that changes the population key (a real
    content change, not a file touch) invalidates the snapshot; one that
    leaves it unchanged does not -- covered by the hit test above."""
    strategy = TfidfClusteringStrategy()
    build_spy = mocker.spy(strategy, "build_target_index")
    build_settings = TfidfTargetIndexBuildSettings(ngram_min=1, ngram_max=2)
    frame = _frame({"t:1": "acme limited", "t:2": "beta holdings"})

    cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=_tfidf_settings("pop-a"),
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=frame,
        target_id_col="system_uri",
        text_col="name",
    )
    cache.resolve_target_index(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=_tfidf_settings("pop-b"),
        clustering_strategy=strategy,
        build_settings=build_settings,
        target_frame=frame,
        target_id_col="system_uri",
        text_col="name",
    )

    assert build_spy.call_count == 2


def _neighbor_edges_frame(country: str = "gb") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target_id_a": ["t:1"],
            "target_id_b": ["t:2"],
            "similarity": [0.9],
            "country": [country],
        }
    )


def test_resolve_target_neighbor_edges_hit_calls_compute_once(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A second call under the same settings reuses the stored edges and
    never calls `compute` again: no probe on a cache hit."""
    calls = 0

    def _compute() -> pl.DataFrame:
        nonlocal calls
        calls += 1
        return _neighbor_edges_frame()

    settings = {
        "index_key": "idx-1",
        "similarity_backend": "sklearn",
        "backend_options": {},
        "target_neighbor_min_similarity": 0.6,
        "target_neighbor_max_per_target": 2,
    }

    edges_first, hit_first, _ = cache.resolve_target_neighbor_edges(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=settings,
        compute=_compute,
    )
    assert hit_first is False
    assert calls == 1
    assert edges_first.to_dicts() == _neighbor_edges_frame().to_dicts()

    edges_second, hit_second, _ = cache.resolve_target_neighbor_edges(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=settings,
        compute=_compute,
    )
    assert hit_second is True
    assert calls == 1
    assert edges_second.to_dicts() == edges_first.to_dicts()


def test_resolve_target_neighbor_edges_key_changes_with_settings(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A different threshold, cap, backend, or target index key resolves a
    different candidate, so a change to any of them re-probes rather than
    silently reusing edges computed under a different setting."""
    calls = 0

    def _compute() -> pl.DataFrame:
        nonlocal calls
        calls += 1
        return _neighbor_edges_frame()

    base_settings = {
        "index_key": "idx-1",
        "similarity_backend": "sklearn",
        "backend_options": {},
        "target_neighbor_min_similarity": 0.6,
        "target_neighbor_max_per_target": 2,
    }
    cache.resolve_target_neighbor_edges(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings=base_settings,
        compute=_compute,
    )
    cache.resolve_target_neighbor_edges(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        settings={**base_settings, "target_neighbor_max_per_target": 3},
        compute=_compute,
    )

    assert calls == 2


def test_resolve_hnsw_index_dir_changes_with_index_key_and_build_settings(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A different target index or a different hnsw build setting resolves a
    different directory, so a change to either never reuses a graph built
    under a different one; a search-time-only difference (not part of
    `hnsw_build_settings`) is not this function's concern -- it resolves
    whatever mapping its caller passes, unchanged."""
    base = cache.resolve_hnsw_index_dir(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        index_key="idx-1",
        hnsw_build_settings={"storage_dtype": "float16", "connectivity": 16},
    )
    same_again = cache.resolve_hnsw_index_dir(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        index_key="idx-1",
        hnsw_build_settings={"storage_dtype": "float16", "connectivity": 16},
    )
    different_index_key = cache.resolve_hnsw_index_dir(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        index_key="idx-2",
        hnsw_build_settings={"storage_dtype": "float16", "connectivity": 16},
    )
    different_build_settings = cache.resolve_hnsw_index_dir(
        roots=workspace_roots,
        target_system="t",
        country="gb",
        index_key="idx-1",
        hnsw_build_settings={"storage_dtype": "int8", "connectivity": 16},
    )

    assert base == same_again
    assert base != different_index_key
    assert base != different_build_settings
    assert cache.HNSW_INDEX_STORE_FACET in base.parts
