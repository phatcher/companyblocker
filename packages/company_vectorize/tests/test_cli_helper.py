from __future__ import annotations

import inspect
from typing import cast

from company_vectorize._cli_helper import SETTINGS
from company_vectorize.clustering_policy import resolve_build_settings_policy
from company_vectorize.dense_vocabulary_gate import (
    DENSE_VOCABULARY_REPRESENTATIONS,
    EXHAUSTIVE_SPARSE_BACKENDS,
)
from company_vectorize.prefix_filter import PREFIX_FILTER_REPRESENTATIONS

_TYPES = {"str", "int", "float", "bool"}
_DIMENSIONS = {"representation", "text_view", "backend"}


def _by_name() -> dict[str, dict[str, object]]:
    return {str(setting["name"]): setting for setting in SETTINGS}


def test_every_setting_is_a_well_formed_declaration_with_a_unique_name() -> None:
    names = [setting["name"] for setting in SETTINGS]

    assert len(names) == len(set(names))
    for setting in SETTINGS:
        assert {"name", "type", "default", "help"} <= set(setting)
        assert setting["type"] in _TYPES
        assert set(cast(dict[str, object], setting.get("applies") or {})) <= _DIMENSIONS


def test_tfidf_defaults_are_the_build_policy_defaults() -> None:
    settings = _by_name()
    parameters = inspect.signature(resolve_build_settings_policy).parameters

    assert settings["tfidf_analyzer"]["default"] == parameters["tfidf_analyzer"].default
    assert settings["tfidf_ngram_min"]["applies"] == {"representation": ("tfidf",)}


def test_the_row_gate_settings_apply_exactly_where_the_gate_does() -> None:
    settings = _by_name()

    for name in ("max_rows", "force"):
        applies = cast(dict[str, tuple[str, ...]], settings[name]["applies"])
        assert set(applies["representation"]) == DENSE_VOCABULARY_REPRESENTATIONS
        assert set(applies["backend"]) == EXHAUSTIVE_SPARSE_BACKENDS


def test_each_backend_option_applies_to_its_own_backend_only() -> None:
    settings = _by_name()

    # The prefix filter is scoped by representation as well: its bound needs
    # the sparse unit-vector rows, so a dense `sbert` run never receives it.
    prefix_applies = cast(
        dict[str, tuple[str, ...]], settings["prefix_filter"]["applies"]
    )
    assert prefix_applies["backend"] == ("sklearn",)
    assert set(prefix_applies["representation"]) == PREFIX_FILTER_REPRESENTATIONS
    assert settings["svd_candidates"]["applies"] == {"backend": ("svd_rerank",)}
    assert settings["kmeans_clusters"]["applies"] == {"backend": ("kmeans",)}
    assert settings["max_passes"]["applies"] == {"backend": ("kmeans",)}
    assert settings["min_cluster_size"]["applies"] == {"backend": ("hdbscan",)}
    assert settings["block_bytes"]["applies"] == {"backend": ("dense_brute",)}


def test_a_setting_two_backends_share_is_declared_once_with_one_default() -> None:
    """One idea, one setting: the seed serves kmeans and lsh and the storage
    precision dense_brute and hnsw, so each declared default must be both
    backends' own."""
    from company_vectorize.dense_brute_similarity import (
        DEFAULT_DENSE_BRUTE_STORAGE_DTYPE,
    )
    from company_vectorize.hnsw_similarity import DEFAULT_HNSW_STORAGE_DTYPE
    from company_vectorize.lsh_similarity import DEFAULT_LSH_SEED
    from company_vectorize.partition_similarity import DEFAULT_KMEANS_SEED

    settings = _by_name()

    assert settings["backend_seed"]["applies"] == {"backend": ("kmeans", "lsh")}
    assert settings["backend_seed"]["option"] == "seed"
    assert settings["backend_seed"]["default"] == DEFAULT_KMEANS_SEED == 42
    assert DEFAULT_LSH_SEED == DEFAULT_KMEANS_SEED
    assert settings["storage_dtype"]["applies"] == {"backend": ("dense_brute", "hnsw")}
    assert DEFAULT_HNSW_STORAGE_DTYPE == DEFAULT_DENSE_BRUTE_STORAGE_DTYPE
    assert settings["storage_dtype"]["default"] == DEFAULT_HNSW_STORAGE_DTYPE


def test_every_backend_setting_names_the_option_it_fills() -> None:
    """A setting scoped to a backend is one of that backend's options, and the
    run records it under that key."""
    for setting in SETTINGS:
        applies = cast(dict[str, object], setting.get("applies") or {})
        if "backend" in applies and setting["name"] not in ("max_rows", "force"):
            assert setting.get("option"), setting["name"]


def test_dense_brute_is_a_selectable_similarity_backend() -> None:
    from company_vectorize._cli_helper import SIMILARITY_BACKENDS

    settings = _by_name()
    assert "dense_brute" in SIMILARITY_BACKENDS
    choices = cast(tuple[str, ...], settings["similarity_backend"]["choices"])
    assert "dense_brute" in choices


def test_hnsw_is_a_selectable_similarity_backend_with_its_own_options() -> None:
    from company_vectorize._cli_helper import SIMILARITY_BACKENDS

    settings = _by_name()
    assert "hnsw" in SIMILARITY_BACKENDS
    choices = cast(tuple[str, ...], settings["similarity_backend"]["choices"])
    assert "hnsw" in choices

    for name in (
        "connectivity",
        "expansion_add",
        "expansion_search",
        "build_threads",
    ):
        assert settings[name]["applies"] == {"backend": ("hnsw",)}
