"""Nearest-neighbour alias-pair probe: how often an embedding space puts a name's own alias closest to it.

A blocking ingredient's job is candidate retrieval, not pairwise classification, so the
question this probe answers is the retrieval one directly: embed every name in a pool of real
observed variants (`NameVariant`, the same shape `RealAliasPairProducer` consumes), and for
each name that has at least one other variant of its own entity in the pool, check whether the
nearest other name in embedding space (by cosine similarity) is a true alias -- same
`system_uri`, not itself -- rather than some other entity's name. The fraction that are is the
ingredient's hit rate.

Independent of any specific `TokenVectorLookup` or vector backend: it takes an `embed`
callable and works over whatever vectors that produces, so the pretrained fastText ingredient
here and a domain-trained one added later both run it unchanged, and a hit rate recorded by
one is directly comparable to the other's.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .pairs import NameVariant

_EPSILON = 1e-8

_DEFAULT_CHUNK_SIZE = 500
"""Anchors scored against the pool in one batch. Peak memory for that batch is
`chunk_size x pool_size` floats (the similarity matrix and the same-shape index array
`np.argpartition` produces before it is sliced down to `top_k`), so this bounds peak memory
independent of pool size rather than letting it grow with the square of it -- see this
package's README for the resulting footprint at wikidata's GB/DE/FR pool sizes.
"""


@dataclass(frozen=True)
class AliasProbeResult:
    """One embedding space's nearest-neighbour hit rate over a pool of real alias variants.

    Attributes:
        hit_rate: `hits / total`, or `0.0` when `total` is `0` (no anchor had another variant
            of its own entity in the pool).
        hits: Anchors whose nearest other name in the pool (by cosine similarity) shared the
            anchor's `system_uri`.
        total: Anchors evaluated -- every variant belonging to an entity with at least two
            variants in the pool, since an entity with only one has no true alias to be
            nearest.
        top_k: How many of the nearest other names counted as a hit if any of them was a true
            alias; `1` (nearest-neighbour only) unless the caller widened it.
        pool_size: Distinct names embedded and searched over.
    """

    hit_rate: float
    hits: int
    total: int
    top_k: int
    pool_size: int


def nearest_neighbour_alias_hit_rate(
    variants: Sequence[NameVariant],
    *,
    embed: Callable[[str], np.ndarray],
    top_k: int = 1,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> AliasProbeResult:
    """Run the nearest-neighbour alias probe over `variants`.

    `embed` is called once per distinct name in `variants` (duplicates across variants embed
    once and are reused), so the cost is one embedding per distinct name plus one cosine
    similarity per (anchor, pool name) pair.

    Only anchors -- names of an entity with more than one distinct variant in the pool -- are
    scored against the pool; a name with no true alias to find is never a query row. Anchors
    are scored in batches of `chunk_size` rather than as one `pool_size x pool_size` matrix, so
    peak memory is `chunk_size x pool_size` floats, tunable independent of how large the pool
    is, rather than growing with the square of it. Each anchor's top-`k` is taken with a
    partition (`np.argpartition`), not a full sort of its similarity row.

    Returns an `AliasProbeResult` with `total=0`/`hit_rate=0.0` if fewer than two distinct
    names are present, or if no entity has more than one variant in the pool.

    Raises:
        ValueError: `chunk_size` is less than `1`.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    names_by_entity: dict[str, list[str]] = defaultdict(list)
    for variant in variants:
        names_by_entity[variant.system_uri].append(variant.name)

    distinct_names = sorted({variant.name for variant in variants})
    pool_size = len(distinct_names)
    if pool_size < 2:
        return AliasProbeResult(
            hit_rate=0.0, hits=0, total=0, top_k=top_k, pool_size=pool_size
        )

    name_index = {name: index for index, name in enumerate(distinct_names)}
    entity_by_name: dict[str, str] = {}
    for entity, names in names_by_entity.items():
        for name in names:
            entity_by_name[name] = entity

    anchor_pairs = [
        (entity, anchor)
        for entity, names in names_by_entity.items()
        if len(set(names)) >= 2
        for anchor in set(names)
    ]

    vectors = np.stack(
        [np.asarray(embed(name), dtype=np.float64) for name in distinct_names]
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + _EPSILON
    normalized = vectors / norms

    hits = 0
    total = 0
    for start in range(0, len(anchor_pairs), chunk_size):
        batch = anchor_pairs[start : start + chunk_size]
        batch_indices = np.array([name_index[anchor] for _, anchor in batch])

        similarity_chunk = normalized[batch_indices] @ normalized.T
        similarity_chunk[np.arange(len(batch)), batch_indices] = -np.inf

        k = min(top_k, pool_size)
        # The k largest per row, unsorted -- avoids sorting the whole row for a top-k of 1.
        top_indices = np.argpartition(similarity_chunk, pool_size - k, axis=1)[
            :, pool_size - k :
        ]

        for row, (entity, anchor) in enumerate(batch):
            total += 1
            if any(
                entity_by_name[distinct_names[index]] == entity
                for index in top_indices[row]
            ):
                hits += 1

    hit_rate = hits / total if total else 0.0
    return AliasProbeResult(
        hit_rate=hit_rate,
        hits=hits,
        total=total,
        top_k=top_k,
        pool_size=pool_size,
    )
