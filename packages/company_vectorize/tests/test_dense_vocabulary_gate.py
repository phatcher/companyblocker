"""Unit coverage for the dense-vocabulary/exhaustive-backend gate.

Everything here is a pure decision over `(representation, backend,
target_rows, backend_options)`, so the real-scale case -- that `gb`'s target
scale is refused -- is exercised by passing that row count directly. No
corpus is read.
"""

from __future__ import annotations

import itertools

import pytest
from company_vectorize.dense_vocabulary_gate import (
    DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    DENSE_VOCABULARY_REPRESENTATIONS,
    EXHAUSTIVE_SPARSE_BACKENDS,
    DenseVocabularyScaleError,
    ensure_dense_vocabulary_scale_supported,
    resolve_dense_vocabulary_gate_options,
)

# The `gb` target-side row count from the run that was killed (the gate's
# checklist), used verbatim so the acceptance criterion is checked at the scale
# it was written about rather than at an arbitrary large number.
GB_TARGET_ROWS = 5_698_275


def test_default_max_rows_is_the_conservative_benchmarked_floor():
    assert DEFAULT_DENSE_VOCABULARY_MAX_ROWS == 20_000


@pytest.mark.parametrize(
    ("representation", "backend"),
    sorted(
        itertools.product(
            sorted(DENSE_VOCABULARY_REPRESENTATIONS),
            sorted(EXHAUSTIVE_SPARSE_BACKENDS),
        )
    ),
)
def test_gated_combination_is_refused_at_real_target_scale(representation, backend):
    with pytest.raises(DenseVocabularyScaleError):
        ensure_dense_vocabulary_scale_supported(
            representation=representation,
            backend=backend,
            target_rows=GB_TARGET_ROWS,
        )


def test_refusal_message_names_every_way_out():
    with pytest.raises(DenseVocabularyScaleError) as excinfo:
        ensure_dense_vocabulary_scale_supported(
            representation="wordpiece",
            backend="sklearn",
            target_rows=GB_TARGET_ROWS,
        )

    message = str(excinfo.value)
    # The message is as much the deliverable as the check: someone who hits
    # this must learn what was refused, why, and what to do instead, without
    # reading the source.
    assert "wordpiece" in message
    assert "sklearn" in message
    assert "5,698,275" in message
    assert "20,000" in message
    assert "--force" in message
    assert "--max-rows" in message
    assert "svd_rerank" in message
    assert "tfidf" in message


def test_force_bypasses_the_gate_at_any_scale():
    ensure_dense_vocabulary_scale_supported(
        representation="wordpiece",
        backend="sklearn",
        target_rows=GB_TARGET_ROWS,
        backend_options={"force": True},
    )


def test_small_corpus_runs_without_force():
    ensure_dense_vocabulary_scale_supported(
        representation="wordpiece",
        backend="sklearn",
        target_rows=DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    )


def test_threshold_is_inclusive_and_one_row_over_is_refused():
    ensure_dense_vocabulary_scale_supported(
        representation="sentencepiece",
        backend="sparse_dot_topn",
        target_rows=1_000,
        backend_options={"max_rows": 1_000},
    )
    with pytest.raises(DenseVocabularyScaleError):
        ensure_dense_vocabulary_scale_supported(
            representation="sentencepiece",
            backend="sparse_dot_topn",
            target_rows=1_001,
            backend_options={"max_rows": 1_000},
        )


@pytest.mark.parametrize("representation", ["tfidf", "sbert"])
def test_sparse_or_dense_embedding_representations_are_not_gated(representation):
    ensure_dense_vocabulary_scale_supported(
        representation=representation,
        backend="sklearn",
        target_rows=GB_TARGET_ROWS,
    )


@pytest.mark.parametrize("backend", ["svd_rerank", "kmeans", "hdbscan"])
def test_non_exhaustive_backends_are_not_gated(backend):
    # svd_rerank's per-query cost is bounded by its fixed low-dimensional
    # projection, and kmeans/hdbscan route to one partition rather than
    # scanning; none of them pay for vocabulary density the way the gated
    # backends do.
    ensure_dense_vocabulary_scale_supported(
        representation="wordpiece",
        backend=backend,
        target_rows=GB_TARGET_ROWS,
    )


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("TRUE", True),
        ("yes", True),
        ("1", True),
        (1, True),
        ("false", False),
        ("FALSE", False),
        ("no", False),
        ("0", False),
        (0, False),
    ],
)
def test_force_option_is_parsed_from_cli_string_literals(raw, expected):
    force, _ = resolve_dense_vocabulary_gate_options({"force": raw})
    assert force is expected


def test_force_false_as_a_string_does_not_bypass_the_gate():
    # `--additional-args backend.force=false` arrives as the string "false",
    # which `bool()` would read as True and silently open the gate.
    with pytest.raises(DenseVocabularyScaleError):
        ensure_dense_vocabulary_scale_supported(
            representation="wordpiece",
            backend="sklearn",
            target_rows=GB_TARGET_ROWS,
            backend_options={"force": "false"},
        )


@pytest.mark.parametrize("raw", ["maybe", None, 1.5, [], {}])
def test_unreadable_force_option_is_rejected_rather_than_guessed(raw):
    with pytest.raises(DenseVocabularyScaleError, match="'force'"):
        ensure_dense_vocabulary_scale_supported(
            representation="wordpiece",
            backend="sklearn",
            target_rows=100,
            backend_options={"force": raw},
        )


@pytest.mark.parametrize(
    ("raw", "expected"), [(50_000, 50_000), ("50000", 50_000), (50_000.0, 50_000)]
)
def test_max_rows_option_is_parsed_from_cli_string_literals(raw, expected):
    _, max_rows = resolve_dense_vocabulary_gate_options({"max_rows": raw})
    assert max_rows == expected


@pytest.mark.parametrize("raw", [-1, "-1", "lots", None, True, []])
def test_unreadable_max_rows_option_is_rejected(raw):
    with pytest.raises(DenseVocabularyScaleError, match="'max_rows'"):
        resolve_dense_vocabulary_gate_options({"max_rows": raw})


def test_absent_options_resolve_to_the_documented_defaults():
    assert resolve_dense_vocabulary_gate_options(None) == (
        False,
        DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    )
    assert resolve_dense_vocabulary_gate_options({}) == (
        False,
        DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    )


def test_a_bad_max_rows_is_reported_even_when_force_is_set():
    with pytest.raises(DenseVocabularyScaleError, match="'max_rows'"):
        ensure_dense_vocabulary_scale_supported(
            representation="wordpiece",
            backend="sklearn",
            target_rows=GB_TARGET_ROWS,
            backend_options={"force": True, "max_rows": "lots"},
        )


def test_unrelated_backend_options_are_left_alone():
    force, max_rows = resolve_dense_vocabulary_gate_options(
        {"prefix_filter": True, "storage_dtype": "float32"}
    )
    assert force is False
    assert max_rows == DEFAULT_DENSE_VOCABULARY_MAX_ROWS
