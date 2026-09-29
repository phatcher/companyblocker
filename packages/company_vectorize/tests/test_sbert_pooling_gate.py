"""Tests for the default-fail gate on checkpoints with no declared
sentence-embedding configuration.

Offline: the local-checkpoint cases build real on-disk checkpoint layouts in
`tmp_path` -- the same files `sentence_transformers` itself reads to decide
whether it must fall back to wrapping a bare transformer in untrained mean
pooling -- so the real probe runs against real directory shapes. The hub cases
inject a status probe rather than reaching the network.
"""

from __future__ import annotations

import json

import pytest
from company_vectorize.sbert_pooling_gate import (
    UntrainedPoolingError,
    checkpoint_pooling_status,
    ensure_sentence_embedding_checkpoint,
)
from company_vectorize.sbert_strategy import DEFAULT_SBERT_MODEL_NAME


def _write_bare_transformer_checkpoint(directory, *, architecture: str):
    """A checkpoint with model weights and config but no sentence-transformers
    configuration -- the shape of a NER or masked-LM checkpoint.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps({"architectures": [architecture], "model_type": "camembert"}),
        encoding="utf-8",
    )
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    return directory


def _write_sentence_transformer_checkpoint(directory):
    _write_bare_transformer_checkpoint(directory, architecture="CamembertModel")
    (directory / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "", "type": "...models.Transformer"},
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "...models.Pooling",
                },
            ]
        ),
        encoding="utf-8",
    )
    pooling = directory / "1_Pooling"
    pooling.mkdir(exist_ok=True)
    (pooling / "config.json").write_text(
        json.dumps({"word_embedding_dimension": 768, "pooling_mode_mean_tokens": True}),
        encoding="utf-8",
    )
    return directory


def test_local_sentence_transformer_checkpoint_is_declared(tmp_path):
    checkpoint = _write_sentence_transformer_checkpoint(tmp_path / "encoder")

    assert checkpoint_pooling_status(str(checkpoint)) == "declared"
    ensure_sentence_embedding_checkpoint(str(checkpoint))


def test_local_ner_checkpoint_is_rejected_by_default(tmp_path):
    # The `Jean-Baptiste/camembert-ner` shape: real weights, real config, no
    # sentence-transformers configuration anywhere. sentence_transformers
    # would load this happily under untrained mean pooling.
    checkpoint = _write_bare_transformer_checkpoint(
        tmp_path / "camembert-ner", architecture="CamembertForTokenClassification"
    )

    assert checkpoint_pooling_status(str(checkpoint)) == "absent"
    with pytest.raises(UntrainedPoolingError, match="untrained mean pooling"):
        ensure_sentence_embedding_checkpoint(str(checkpoint))


def test_local_ner_checkpoint_loads_under_explicit_force(tmp_path):
    checkpoint = _write_bare_transformer_checkpoint(
        tmp_path / "camembert-ner", architecture="CamembertForTokenClassification"
    )

    ensure_sentence_embedding_checkpoint(str(checkpoint), force=True)


def test_older_layout_with_only_sentence_bert_config_is_accepted(tmp_path):
    # Checkpoints published before the modules.json layout (for example
    # `dangvantuan/sentence-camembert-large`) ship only this file; requiring
    # the modern pair would reject genuine sentence encoders.
    checkpoint = _write_bare_transformer_checkpoint(
        tmp_path / "older-encoder", architecture="CamembertModel"
    )
    (checkpoint / "sentence_bert_config.json").write_text(
        json.dumps({"max_seq_length": 128}), encoding="utf-8"
    )

    assert checkpoint_pooling_status(str(checkpoint)) == "declared"


def test_unverifiable_checkpoint_is_rejected_rather_than_assumed_good():
    with pytest.raises(UntrainedPoolingError, match="could not be verified"):
        ensure_sentence_embedding_checkpoint(
            "some-org/unreachable-model", status_probe=lambda _: "unknown"
        )


def test_hub_checkpoint_declaring_pooling_is_accepted():
    ensure_sentence_embedding_checkpoint(
        "some-org/a-real-sentence-encoder", status_probe=lambda _: "declared"
    )


def test_registered_checkpoint_is_not_reprobed():
    # Registry entries were verified when they were added, so the default
    # model must not pay a probe (and a network round trip) per run.
    def _probe(_model: str):
        raise AssertionError("a registered checkpoint should not be probed")

    ensure_sentence_embedding_checkpoint(DEFAULT_SBERT_MODEL_NAME, status_probe=_probe)
    ensure_sentence_embedding_checkpoint("en-all-minilm-l6-v2", status_probe=_probe)


def test_force_skips_the_probe_entirely():
    def _probe(_model: str):
        raise AssertionError("force should short-circuit before probing")

    ensure_sentence_embedding_checkpoint(
        "some-org/unlisted", force=True, status_probe=_probe
    )
