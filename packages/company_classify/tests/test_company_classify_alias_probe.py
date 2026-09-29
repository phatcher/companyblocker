"""Tests for the nearest-neighbour alias-pair probe.

Purely numeric: `embed` is a fixture lookup table, never a real vector backend, so these
assert the probe's own retrieval arithmetic independent of any `TokenVectorLookup`
implementation.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
from company_classify.alias_probe import nearest_neighbour_alias_hit_rate
from company_classify.pairs import NameVariant


def _embed_from(table: dict[str, np.ndarray]):
    def embed(name: str) -> np.ndarray:
        return table[name]

    return embed


def test_every_anchor_finds_its_true_alias_nearest():
    variants = [
        NameVariant(system_uri="e1", name="acme corp"),
        NameVariant(system_uri="e1", name="acme corporation"),
        NameVariant(system_uri="e2", name="beta industries"),
        NameVariant(system_uri="e2", name="beta industries ltd"),
    ]
    table = {
        "acme corp": np.array([1.0, 0.0]),
        "acme corporation": np.array([0.9, 0.1]),
        "beta industries": np.array([0.0, 1.0]),
        "beta industries ltd": np.array([0.1, 0.9]),
    }

    result = nearest_neighbour_alias_hit_rate(variants, embed=_embed_from(table))

    assert result.hits == 4
    assert result.total == 4
    assert result.hit_rate == 1.0
    assert result.pool_size == 4


def test_a_miss_is_counted_when_the_nearest_name_belongs_to_another_entity():
    variants = [
        NameVariant(system_uri="e1", name="acme corp"),
        NameVariant(system_uri="e1", name="acme corporation"),
        NameVariant(system_uri="e2", name="not related at all"),
    ]
    table = {
        # "acme corp" and "not related at all" are made deliberately closer than
        # "acme corp" and its own true alias, so the anchor's nearest neighbour misses.
        "acme corp": np.array([1.0, 0.0]),
        "acme corporation": np.array([-1.0, 0.0]),
        "not related at all": np.array([0.99, 0.01]),
    }

    result = nearest_neighbour_alias_hit_rate(variants, embed=_embed_from(table))

    # "acme corp" -> nearest is "not related at all" (miss); "acme corporation" -> nearest
    # is "not related at all" too (miss); "not related at all" has no other e2 variant, so
    # it is not evaluated as an anchor at all.
    assert result.total == 2
    assert result.hits == 0
    assert result.hit_rate == 0.0


def test_an_entity_with_a_single_variant_is_not_evaluated_as_an_anchor():
    variants = [
        NameVariant(system_uri="e1", name="acme corp"),
        NameVariant(system_uri="e1", name="acme corporation"),
        NameVariant(system_uri="e2", name="lonely entity"),
    ]
    table = {
        "acme corp": np.array([1.0, 0.0]),
        "acme corporation": np.array([0.9, 0.1]),
        "lonely entity": np.array([0.0, 1.0]),
    }

    result = nearest_neighbour_alias_hit_rate(variants, embed=_embed_from(table))

    assert result.total == 2  # only the two "acme" variants have a true alias to find


def test_fewer_than_two_distinct_names_yields_an_empty_result():
    variants = [NameVariant(system_uri="e1", name="only one name")]

    result = nearest_neighbour_alias_hit_rate(
        variants, embed=_embed_from({"only one name": np.array([1.0, 0.0])})
    )

    assert result.total == 0
    assert result.hits == 0
    assert result.hit_rate == 0.0
    assert result.pool_size == 1


def test_top_k_widens_what_counts_as_a_hit():
    variants = [
        NameVariant(system_uri="e1", name="acme corp"),
        NameVariant(system_uri="e1", name="acme corporation"),
        NameVariant(system_uri="e2", name="closer distractor"),
    ]
    table = {
        "acme corp": np.array([1.0, 0.0]),
        # The distractor is nearer to "acme corp" than its true alias is, but the true
        # alias is still the second-nearest, so top_k=2 should count it as a hit.
        "closer distractor": np.array([0.99, 0.0]),
        "acme corporation": np.array([0.5, 0.5]),
    }

    top_1 = nearest_neighbour_alias_hit_rate(
        variants, embed=_embed_from(table), top_k=1
    )
    top_2 = nearest_neighbour_alias_hit_rate(
        variants, embed=_embed_from(table), top_k=2
    )

    assert top_1.hit_rate < top_2.hit_rate


def test_a_chunk_size_below_one_is_rejected():
    variants = [
        NameVariant(system_uri="e1", name="acme corp"),
        NameVariant(system_uri="e1", name="acme corporation"),
    ]
    table = {
        "acme corp": np.array([1.0, 0.0]),
        "acme corporation": np.array([0.9, 0.1]),
    }

    with pytest.raises(ValueError, match="chunk_size must be >= 1"):
        nearest_neighbour_alias_hit_rate(
            variants, embed=_embed_from(table), chunk_size=0
        )


def test_peak_memory_follows_chunk_size_rather_than_the_pool():
    """Scoring anchors in batches of `chunk_size` holds `chunk_size x pool_size` floats at a
    time (never one dense `pool_size x pool_size` matrix, which grows with the square of the
    pool regardless of how many anchors there are and once made a de-jurisdiction pool of
    46,586 distinct names alone need roughly 16 GiB): holding the pool fixed and shrinking or
    growing `chunk_size` should shrink or grow peak memory with it. The pool here (5,500
    distinct names, most belonging to single-variant "filler" entities that are never scored
    as anchors) is large enough that a `chunk_size` of a few hundred against it is measurably
    different from a `chunk_size` of two.
    """
    rng = np.random.default_rng(0)
    dimension = 32
    filler_count = 4500
    anchor_entity_count = 500

    table: dict[str, np.ndarray] = {}
    variants: list[NameVariant] = []
    for i in range(filler_count):
        name = f"filler company {i}"
        table[name] = rng.standard_normal(dimension)
        variants.append(NameVariant(system_uri=f"filler-{i}", name=name))
    for i in range(anchor_entity_count):
        name_a, name_b = f"anchor company {i} a", f"anchor company {i} b"
        table[name_a] = rng.standard_normal(dimension)
        table[name_b] = rng.standard_normal(dimension)
        variants.append(NameVariant(system_uri=f"anchor-{i}", name=name_a))
        variants.append(NameVariant(system_uri=f"anchor-{i}", name=name_b))
    embed = _embed_from(table)

    tracemalloc.start()
    try:
        nearest_neighbour_alias_hit_rate(variants, embed=embed, chunk_size=2)
        _, small_chunk_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

        nearest_neighbour_alias_hit_rate(
            variants, embed=embed, chunk_size=anchor_entity_count * 2
        )
        _, large_chunk_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert large_chunk_peak > small_chunk_peak * 10
