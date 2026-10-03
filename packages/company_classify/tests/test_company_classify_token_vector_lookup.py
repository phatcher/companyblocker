"""Tests for the `TokenVectorLookup` interface and its pretrained fastText implementation.

`test_load_pretrained_fasttext_vectors_over_a_real_tiny_checkpoint` loads gensim's own bundled
test fixture (`gensim.test.utils.datapath("crime-and-punishment.bin")`, a few KB, shipped with
the installed package) through the real `load_facebook_vectors` -- no network access and no
multi-gigabyte checkpoint -- so the real load path is exercised once without needing the real
pretrained checkpoint this item's dispatch downloaded separately. Every other test injects a
fake `_KeyedVectorsLike` and never touches `gensim` at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from company_classify.token_vector_lookup import (
    MeanPooledTokenVectorEncoder,
    TokenVectorProvenance,
    default_name_tokenizer,
    load_pretrained_fasttext_vectors,
    mean_pool_name_vector,
)


class _FakeKeyedVectors:
    """A minimal `_KeyedVectorsLike` fake: fixed vectors for known tokens, a deterministic
    subword-style fallback for unknown ones, so `vector()` never raises."""

    vector_size = 3

    def __init__(self, vectors: dict[str, np.ndarray], counts: dict[str, int]) -> None:
        self._vectors = vectors
        self._counts = counts
        self.key_to_index = {token: index for index, token in enumerate(vectors)}

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._vectors:
            return self._vectors[key]
        # Deterministic "subword" fallback for an out-of-vocabulary token, standing in for
        # FastTextKeyedVectors' real character-n-gram composition.
        return np.full(self.vector_size, fill_value=0.1)

    def get_vecattr(self, key: str, attr: str):
        assert attr == "count"
        return self._counts.get(key)

    def has_index_for(self, key: str) -> bool:
        return key in self._vectors


def _fake_lookup() -> _FakeKeyedVectors:
    return _FakeKeyedVectors(
        vectors={
            "acme": np.array([1.0, 0.0, 0.0]),
            "holdings": np.array([0.0, 1.0, 0.0]),
        },
        counts={"acme": 10, "holdings": 4},
    )


def test_provenance_is_carried_through():
    provenance = TokenVectorProvenance(
        slug="en-cc-300", source="https://example.invalid", checksum="abc123"
    )
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"), provenance=provenance, loader=lambda _path: _fake_lookup()
    )

    assert lookup.provenance == provenance


def test_dimension_matches_the_underlying_vector_size():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )

    assert lookup.dimension == 3


def test_vector_returns_a_subword_composed_vector_for_an_unseen_token():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )

    in_vocab = lookup.vector("acme")
    out_of_vocab = lookup.vector("acmeco")

    assert np.array_equal(in_vocab, [1.0, 0.0, 0.0])
    assert not np.array_equal(out_of_vocab, np.zeros(3))


def test_count_is_zero_for_an_out_of_vocabulary_token():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )

    assert lookup.count("acme") == 10
    assert lookup.count("nowhere-in-the-checkpoint") == 0


def test_total_is_the_sum_of_in_vocabulary_counts():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )

    assert lookup.total == 14


def test_mean_pool_name_vector_averages_token_vectors():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )

    pooled = mean_pool_name_vector("Acme Holdings", lookup)

    assert np.allclose(pooled, [0.5, 0.5, 0.0])


def test_mean_pool_name_vector_returns_zeros_for_an_empty_name():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )

    assert np.array_equal(mean_pool_name_vector("   ", lookup), np.zeros(3))


def test_default_name_tokenizer_lowercases_and_splits_on_whitespace():
    assert default_name_tokenizer("Acme  Holdings LTD") == ["acme", "holdings", "ltd"]


def test_load_pretrained_fasttext_vectors_over_a_real_tiny_checkpoint():
    gensim_test_utils = pytest.importorskip("gensim.test.utils")
    checkpoint_path = Path(gensim_test_utils.datapath("crime-and-punishment.bin"))

    lookup = load_pretrained_fasttext_vectors(
        checkpoint_path,
        provenance=TokenVectorProvenance(
            slug="test-fixture", source=str(checkpoint_path), checksum=None
        ),
    )

    assert lookup.dimension > 0
    assert lookup.total > 0
    assert lookup.count("landlady") > 0
    assert lookup.vector("landlord").shape == (lookup.dimension,)


def test_mean_pooled_token_vector_encoder_embeds_a_name_as_its_pooled_vector():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )
    encoder = MeanPooledTokenVectorEncoder(lookup)

    assert np.allclose(encoder.embed("Acme Holdings"), [0.5, 0.5, 0.0])
    assert np.allclose(encoder.embed("acme"), mean_pool_name_vector("acme", lookup))
    assert np.array_equal(encoder.embed("   "), np.zeros(3))


def test_mean_pooled_token_vector_encoder_reads_names_through_its_tokenizer():
    lookup = load_pretrained_fasttext_vectors(
        Path("unused"),
        provenance=TokenVectorProvenance(slug="s", source="s", checksum=None),
        loader=lambda _path: _fake_lookup(),
    )
    encoder = MeanPooledTokenVectorEncoder(
        lookup, tokenize=lambda name: name.split("-")
    )

    assert np.allclose(encoder.embed("acme-holdings"), [0.5, 0.5, 0.0])
