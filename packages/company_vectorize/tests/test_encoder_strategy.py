from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
import pytest
from company_vectorize.clustering_contract import (
    EncoderTargetIndexBuildSettings,
    WordpieceTargetIndexBuildSettings,
)
from company_vectorize.encoder_strategy import EncoderClusteringStrategy


class _FakeEncoder:
    """Bare `EncoderHandle`: only the one required method, no persistence shape.

    Deterministic vectors keyed by name, mirroring `_FakeEncoder` in
    `test_sbert_strategy.py` -- only the strategy's shape/plumbing is under
    test here, not real embedding quality.
    """

    _VECTORS: ClassVar[dict[str, list[float]]] = {
        "Acme Ltd": [1.0, 0.0],
        "Beta Plc": [0.0, 1.0],
        "placeholder": [1.0, 1.0],
    }

    def embed(self, name: str) -> np.ndarray:
        return np.array(self._VECTORS[name], dtype=np.float32)


@dataclass(frozen=True)
class _FakeModelProvenance:
    """Stands in for `company_classify.persistence.ModelProvenance`."""

    training_source: str
    run_id: str


@dataclass(frozen=True)
class _FakePersistedModel:
    """Stands in for `company_classify.persistence.PersistedModel`.

    A caller loading a real persisted `company_classify` pair-classifier
    encoder (`persistence.load_pair_classifier_model()`) gets back exactly
    this shape: the fitted model plus its provenance manifest. This double
    exercises that same caller-side pattern -- unwrap `.model` for
    `EncoderTargetIndexBuildSettings.encoder`, read `.provenance` for
    `encoder_name` -- without this package depending on `company_classify`.
    """

    model: _FakeEncoder
    provenance: _FakeModelProvenance
    model_class: str


def _fake_persisted_classifier_encoder() -> _FakePersistedModel:
    return _FakePersistedModel(
        model=_FakeEncoder(),
        provenance=_FakeModelProvenance(
            training_source="gleif_gb_pairs", run_id="run-42"
        ),
        model_class="company_classify.encoder_models.PooledSubwordContrastiveEncoder",
    )


def _target_frame() -> pl.DataFrame:
    return pl.DataFrame({"system_uri": ["t1", "t2"], "name": ["Acme Ltd", "Beta Plc"]})


def test_encoder_strategy_builds_dense_target_index():
    strategy = EncoderClusteringStrategy()
    target = _target_frame()

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=EncoderTargetIndexBuildSettings(
            encoder=_FakeEncoder(), encoder_name="fake-encoder-v1"
        ),
    )

    assert index.target_ids == ["t1", "t2"]
    assert isinstance(index.target_matrix, np.ndarray)
    assert index.target_matrix.shape == (2, 2)


def test_encoder_strategy_scores_source_chunk():
    strategy = EncoderClusteringStrategy()
    target = _target_frame()
    source = pl.DataFrame({"system_uri": ["s1"], "name": ["Acme Ltd"]})

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=EncoderTargetIndexBuildSettings(
            encoder=_FakeEncoder(), encoder_name="fake-encoder-v1"
        ),
    )
    backend_index = strategy.build_backend_index(
        index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options=None,
    )

    rows = strategy.score_source_chunk(
        source,
        target_index=index,
        source_id_col="system_uri",
        text_col="name",
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=backend_index,
        backend_options=None,
    )

    assert rows.height >= 1
    top = rows.sort(["source_id", "rank"]).row(0, named=True)
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t1"


def test_encoder_strategy_completes_a_run_over_a_persisted_classifier_package_encoder():
    """The `"Done when"` clause's second case: a persisted classifier-package
    encoder (not a bare fake) completes a build/score run, and its provenance
    carries an identifier through to `encoder_name`."""
    persisted = _fake_persisted_classifier_encoder()
    strategy = EncoderClusteringStrategy()
    target = _target_frame()
    source = pl.DataFrame({"system_uri": ["s1"], "name": ["Beta Plc"]})

    settings = EncoderTargetIndexBuildSettings(
        encoder=persisted.model,
        encoder_name=f"{persisted.model_class}:{persisted.provenance.run_id}",
    )
    index = strategy.build_target_index(
        target, target_id_col="system_uri", text_col="name", build_settings=settings
    )
    backend_index = strategy.build_backend_index(
        index,
        backend="sklearn",
        top_k=2,
        max_candidates_per_source=None,
        backend_options=None,
    )

    rows = strategy.score_source_chunk(
        source,
        target_index=index,
        source_id_col="system_uri",
        text_col="name",
        top_k=2,
        min_similarity=0.0,
        max_candidates_per_source=None,
        nn_index=None,
        backend="sklearn",
        backend_index=backend_index,
        backend_options=None,
    )

    assert settings.encoder_name == (
        "company_classify.encoder_models.PooledSubwordContrastiveEncoder:run-42"
    )
    top = rows.sort(["source_id", "rank"]).row(0, named=True)
    assert top["source_id"] == "s1"
    assert top["target_id"] == "t2"


def test_rebuilt_index_scores_a_source_chunk_identically_to_the_original():
    strategy = EncoderClusteringStrategy()
    settings = EncoderTargetIndexBuildSettings(
        encoder=_FakeEncoder(), encoder_name="fake-encoder-v1"
    )
    target = _target_frame()
    source = pl.DataFrame({"system_uri": ["s1"], "name": ["Acme Ltd"]})
    index = strategy.build_target_index(
        target, target_id_col="system_uri", text_col="name", build_settings=settings
    )

    def _score(built_index):
        backend_index = strategy.build_backend_index(
            built_index,
            backend="sklearn",
            top_k=2,
            max_candidates_per_source=None,
            backend_options=None,
        )
        return strategy.score_source_chunk(
            source,
            target_index=built_index,
            source_id_col="system_uri",
            text_col="name",
            top_k=2,
            min_similarity=0.0,
            max_candidates_per_source=None,
            nn_index=None,
            backend="sklearn",
            backend_index=backend_index,
            backend_options=None,
        ).sort(["source_id", "rank"])

    parts = strategy.target_index_storable_parts(index)
    rebuilt = strategy.rebuild_target_index(parts, build_settings=settings)

    assert rebuilt.target_ids == index.target_ids
    assert _score(rebuilt).equals(_score(index))


def test_rebuild_accepts_a_memory_mapped_matrix_without_copying(tmp_path):
    strategy = EncoderClusteringStrategy()
    settings = EncoderTargetIndexBuildSettings(
        encoder=_FakeEncoder(), encoder_name="fake-encoder-v1"
    )
    target = _target_frame()
    index = strategy.build_target_index(
        target, target_id_col="system_uri", text_col="name", build_settings=settings
    )
    parts = strategy.target_index_storable_parts(index)

    matrix_path = tmp_path / "target_matrix.npy"
    np.save(matrix_path, parts["target_matrix"])
    memmapped = np.load(matrix_path, mmap_mode="r")
    parts = dict(parts)
    parts["target_matrix"] = memmapped

    rebuilt = strategy.rebuild_target_index(parts, build_settings=settings)

    assert isinstance(rebuilt.target_matrix, np.memmap)
    assert np.shares_memory(rebuilt.target_matrix, memmapped)


def test_rebuild_rejects_wrong_settings_type():
    strategy = EncoderClusteringStrategy()
    index = strategy.build_target_index(
        pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]}),
        target_id_col="system_uri",
        text_col="name",
        build_settings=EncoderTargetIndexBuildSettings(
            encoder=_FakeEncoder(), encoder_name="fake-encoder-v1"
        ),
    )
    parts = strategy.target_index_storable_parts(index)

    with pytest.raises(TypeError, match="encoder strategy requires"):
        strategy.rebuild_target_index(
            parts, build_settings=WordpieceTargetIndexBuildSettings()
        )


def test_encoder_strategy_rejects_wrong_settings_type():
    strategy = EncoderClusteringStrategy()
    target = pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]})

    with pytest.raises(TypeError, match="encoder strategy requires"):
        strategy.build_target_index(
            target,
            target_id_col="system_uri",
            text_col="name",
            build_settings=WordpieceTargetIndexBuildSettings(),
        )


@pytest.mark.parametrize("backend", ["sparse_dot_topn", "svd_rerank"])
def test_encoder_strategy_rejects_unsupported_backends(backend):
    strategy = EncoderClusteringStrategy()
    target = pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]})
    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=EncoderTargetIndexBuildSettings(
            encoder=_FakeEncoder(), encoder_name="fake-encoder-v1"
        ),
    )

    with pytest.raises(ValueError, match=f"does not support the '{backend}' backend"):
        strategy.build_backend_index(
            index,
            backend=backend,
            top_k=2,
            max_candidates_per_source=None,
            backend_options=None,
        )


def test_encoder_strategy_builds_an_empty_target_index_placeholder():
    strategy = EncoderClusteringStrategy()
    target = pl.DataFrame(
        {"system_uri": [], "name": []},
        schema={
            "system_uri": pl.Utf8,
            "name": pl.Utf8,
        },
    )

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=EncoderTargetIndexBuildSettings(
            encoder=_FakeEncoder(), encoder_name="fake-encoder-v1"
        ),
    )

    assert index.target_ids == []
    assert index.target_matrix.shape == (0, 2)
