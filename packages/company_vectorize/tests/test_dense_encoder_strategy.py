from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import polars as pl
import pytest
from company_vectorize.clustering_contract import (
    TargetIndexBuildSettings,
    WordpieceTargetIndexBuildSettings,
)
from company_vectorize.dense_encoder_strategy import DenseEncoderClusteringStrategy


@dataclass(frozen=True)
class _FakeSettings(TargetIndexBuildSettings):
    pass


class _FakeEncoder:
    _VECTORS: ClassVar[dict[str, list[float]]] = {
        "Acme Ltd": [1.0, 0.0],
        "Beta Plc": [0.0, 1.0],
        "placeholder": [1.0, 1.0],
    }

    def fit(self, texts: list[str]) -> "_FakeEncoder":
        return self

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.array([self._VECTORS[text] for text in texts], dtype=np.float32)


class _FakeDenseStrategy(DenseEncoderClusteringStrategy):
    representation = "fake"
    settings_type = _FakeSettings

    def __init__(self) -> None:
        self.encoders_built = 0

    def _build_encoder(self, build_settings: _FakeSettings) -> _FakeEncoder:
        self.encoders_built += 1
        return _FakeEncoder()


_TARGET = pl.DataFrame({"system_uri": ["t1", "t2"], "name": ["Acme Ltd", " Beta Plc "]})


def _index(strategy: _FakeDenseStrategy):
    return strategy.build_target_index(
        _TARGET,
        target_id_col="system_uri",
        text_col="name",
        build_settings=_FakeSettings(),
    )


def test_the_target_index_is_the_encoders_matrix_over_stripped_names() -> None:
    index = _index(_FakeDenseStrategy())

    assert index.target_ids == ["t1", "t2"]
    assert index.target_matrix.tolist() == [[1.0, 0.0], [0.0, 1.0]]


def test_a_rebuilt_index_builds_the_encoder_again_and_keeps_the_stored_matrix() -> None:
    strategy = _FakeDenseStrategy()
    parts = strategy.target_index_storable_parts(_index(strategy))

    rebuilt = strategy.rebuild_target_index(parts, build_settings=_FakeSettings())

    assert set(parts) == {"target_ids", "target_matrix"}
    assert rebuilt.target_ids == ["t1", "t2"]
    assert rebuilt.target_matrix is parts["target_matrix"]
    assert strategy.encoders_built == 2


def test_another_representations_settings_are_refused_by_name() -> None:
    strategy = _FakeDenseStrategy()

    with pytest.raises(TypeError, match="fake strategy requires _FakeSettings"):
        strategy.build_target_index(
            _TARGET,
            target_id_col="system_uri",
            text_col="name",
            build_settings=WordpieceTargetIndexBuildSettings(),
        )


@pytest.mark.parametrize("backend", ["sparse_dot_topn", "svd_rerank"])
def test_a_sparse_only_backend_is_refused_naming_the_representation(
    backend: str,
) -> None:
    strategy = _FakeDenseStrategy()

    with pytest.raises(
        ValueError, match=f"fake strategy does not support the '{backend}' backend"
    ):
        strategy.build_backend_index(
            _index(strategy),
            backend=backend,
            top_k=2,
            max_candidates_per_source=None,
            backend_options=None,
        )
