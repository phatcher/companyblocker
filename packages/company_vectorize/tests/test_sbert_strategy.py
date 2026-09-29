import builtins
import sys
from typing import ClassVar

import numpy as np
import polars as pl
import pytest
from company_vectorize import sbert_strategy
from company_vectorize.clustering_contract import (
    SbertTargetIndexBuildSettings,
    WordpieceTargetIndexBuildSettings,
)
from company_vectorize.sbert_pooling_gate import UntrainedPoolingError
from company_vectorize.sbert_strategy import SbertClusteringStrategy


class _FakeEncoder:
    """Deterministic stand-in for `_SentenceTransformerEncoder`, injected via
    `SbertClusteringStrategy(encoder_factory=...)` so these tests don't need
    the optional `sentence-transformers` dependency installed or a real model
    download -- only the strategy's shape/plumbing is under test here, not
    actual semantic embedding quality.
    """

    _VECTORS: ClassVar[dict[str, list[float]]] = {
        "Acme Ltd": [1.0, 0.0],
        "Beta Plc": [0.0, 1.0],
        "placeholder": [1.0, 1.0],
    }

    def fit(self, texts: list[str]) -> "_FakeEncoder":
        _ = texts
        return self

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.transform(texts)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.array([self._VECTORS[text] for text in texts], dtype=np.float32)


def _fake_strategy() -> SbertClusteringStrategy:
    return SbertClusteringStrategy(encoder_factory=lambda model_name: _FakeEncoder())


def test_sbert_strategy_builds_dense_target_index():
    strategy = _fake_strategy()
    target = pl.DataFrame(
        {"system_uri": ["t1", "t2"], "name": ["Acme Ltd", "Beta Plc"]}
    )

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=SbertTargetIndexBuildSettings(model_name="fake"),
    )

    assert index.target_ids == ["t1", "t2"]
    assert isinstance(index.target_matrix, np.ndarray)
    assert index.target_matrix.shape == (2, 2)


def test_sbert_strategy_scores_source_chunk():
    strategy = _fake_strategy()
    target = pl.DataFrame(
        {"system_uri": ["t1", "t2"], "name": ["Acme Ltd", "Beta Plc"]}
    )
    source = pl.DataFrame({"system_uri": ["s1"], "name": ["Acme Ltd"]})

    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=SbertTargetIndexBuildSettings(model_name="fake"),
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


def test_rebuilt_index_scores_a_source_chunk_identically_to_the_original():
    strategy = _fake_strategy()
    settings = SbertTargetIndexBuildSettings(model_name="fake")
    target = pl.DataFrame(
        {"system_uri": ["t1", "t2"], "name": ["Acme Ltd", "Beta Plc"]}
    )
    source = pl.DataFrame({"system_uri": ["s1"], "name": ["Acme Ltd"]})
    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=settings,
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
    strategy = _fake_strategy()
    settings = SbertTargetIndexBuildSettings(model_name="fake")
    target = pl.DataFrame(
        {"system_uri": ["t1", "t2"], "name": ["Acme Ltd", "Beta Plc"]}
    )
    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=settings,
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
    strategy = _fake_strategy()
    index = strategy.build_target_index(
        pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]}),
        target_id_col="system_uri",
        text_col="name",
        build_settings=SbertTargetIndexBuildSettings(model_name="fake"),
    )
    parts = strategy.target_index_storable_parts(index)

    with pytest.raises(TypeError, match="sbert strategy requires"):
        strategy.rebuild_target_index(
            parts, build_settings=WordpieceTargetIndexBuildSettings()
        )


def test_sbert_strategy_rejects_wrong_settings_type():
    strategy = _fake_strategy()
    target = pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]})

    with pytest.raises(TypeError, match="sbert strategy requires"):
        strategy.build_target_index(
            target,
            target_id_col="system_uri",
            text_col="name",
            build_settings=WordpieceTargetIndexBuildSettings(),
        )


@pytest.mark.parametrize("backend", ["sparse_dot_topn", "svd_rerank"])
def test_sbert_strategy_rejects_unsupported_backends(backend):
    strategy = _fake_strategy()
    target = pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]})
    index = strategy.build_target_index(
        target,
        target_id_col="system_uri",
        text_col="name",
        build_settings=SbertTargetIndexBuildSettings(model_name="fake"),
    )

    with pytest.raises(ValueError, match=f"does not support the '{backend}' backend"):
        strategy.build_backend_index(
            index,
            backend=backend,
            top_k=2,
            max_candidates_per_source=None,
            backend_options=None,
        )


def test_encoder_factory_receives_the_build_settings():
    seen: list[SbertTargetIndexBuildSettings] = []

    def _factory(build_settings):
        seen.append(build_settings)
        return _FakeEncoder()

    strategy = SbertClusteringStrategy(encoder_factory=_factory)
    settings = SbertTargetIndexBuildSettings(model_name="fake")
    strategy.build_target_index(
        pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]}),
        target_id_col="system_uri",
        text_col="name",
        build_settings=settings,
    )

    assert seen == [settings]


class _FakeSentenceTransformer:
    """Stands in for `sentence_transformers.SentenceTransformer` itself.

    The two checks the real loader owns -- the optional dependency and the
    sentence-embedding pooling gate -- live between
    `_import_sentence_transformer()` and constructing the model, so that seam
    is the only place a test can observe them without
    `sentence-transformers` installed.
    """

    constructed_with: ClassVar[list[str]] = []

    def __init__(self, checkpoint: str) -> None:
        type(self).constructed_with.append(checkpoint)

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, normalize_embeddings: bool = False):
        _ = normalize_embeddings
        return np.zeros((len(texts), 2), dtype=np.float32)


@pytest.fixture
def fake_sentence_transformer(monkeypatch):
    _FakeSentenceTransformer.constructed_with = []
    monkeypatch.setattr(
        sbert_strategy,
        "_import_sentence_transformer",
        lambda: _FakeSentenceTransformer,
    )
    return _FakeSentenceTransformer


def test_default_encoder_factory_resolves_a_registry_slug(fake_sentence_transformer):
    sbert_strategy._SentenceTransformerEncoder(
        SbertTargetIndexBuildSettings(model_name="fr-sentence-camembert-base")
    )

    assert fake_sentence_transformer.constructed_with == [
        "dangvantuan/sentence-camembert-base"
    ]


def test_default_encoder_factory_rejects_a_checkpoint_without_pooling(
    tmp_path, fake_sentence_transformer
):
    # A NER-shaped checkpoint: weights and config, no sentence-transformers
    # configuration. sentence_transformers would wrap it in untrained mean
    # pooling and return plausible-looking embeddings.
    checkpoint = tmp_path / "camembert-ner"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"architectures": ["CamembertForTokenClassification"]}', encoding="utf-8"
    )

    with pytest.raises(UntrainedPoolingError, match="untrained mean pooling"):
        sbert_strategy._SentenceTransformerEncoder(
            SbertTargetIndexBuildSettings(model_name=str(checkpoint))
        )

    # Rejected before the checkpoint is loaded at all, not after.
    assert fake_sentence_transformer.constructed_with == []


def test_default_encoder_factory_loads_unpooled_checkpoint_under_force(
    tmp_path, fake_sentence_transformer
):
    checkpoint = tmp_path / "camembert-ner"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        '{"architectures": ["CamembertForTokenClassification"]}', encoding="utf-8"
    )

    sbert_strategy._SentenceTransformerEncoder(
        SbertTargetIndexBuildSettings(
            model_name=str(checkpoint), force_untrained_pooling=True
        )
    )

    assert fake_sentence_transformer.constructed_with == [str(checkpoint)]


def test_default_encoder_factory_raises_helpful_error_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    # The absence is simulated rather than relied upon. This test used to rest on
    # sentence-transformers genuinely not being installed, which stopped being true
    # once the `sbert` extra was added to the workspace: an environment-dependent
    # premise means the failure path is exercised only where the extra is absent, and
    # the test fails rather than skips everywhere else.
    real_import = builtins.__import__

    def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    strategy = SbertClusteringStrategy()
    target = pl.DataFrame({"system_uri": ["t1"], "name": ["Acme Ltd"]})

    with pytest.raises(ModuleNotFoundError, match="sentence-transformers"):
        strategy.build_target_index(
            target,
            target_id_col="system_uri",
            text_col="name",
            build_settings=SbertTargetIndexBuildSettings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            ),
        )
