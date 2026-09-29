"""Negative-pair sampling shared by every pair producer.

Positives differ by producer (a perturbation, an observed alias, a derived short name);
negatives do not, so they are drawn here once rather than three times. How a negative is drawn
decides what a downstream classifier learns to discriminate, which is why the choice is an
explicit part of the contract instead of a per-producer detail:

- `NegativeStrategy.RANDOM`: a uniformly drawn different entity. Cheap, and mostly trivially
  separable, so it teaches a model little beyond "these names look nothing alike".
- `NegativeStrategy.HARD`: the lexically closest name among a sampled candidate pool, by
  character-trigram Jaccard similarity. Harder, and the discrimination that actually matters for
  blocking, where the confusable candidates are exactly the ones a blocker surfaces.

Two invariants hold whichever strategy is used. A negative's counterpart is drawn only from
entities in the *same* partition as the anchor, so no pair straddles the train/evaluation
boundary; and a candidate whose name is identical to the anchor's is rejected, since that pair's
label is unlearnable noise rather than a hard example (`pairs.find_ambiguous_pairs()` would
flag it).

Sampling is deterministic without being order-dependent: each negative's seed is composed from
the anchor pair's own content plus a within-anchor counter, never from the anchor's position in
a list, so a producer that emits its positives in a different order still draws the same
negatives. Seed composition follows the repository's documented rule for combining several
identifying values into one seed (see `docs/architecture/design-principles.md`, "Deterministic
seeding"): length-prefixed `blake2b` segments, so no segment's content can be confused with a
delimiter, then a scalar `random.Random` seeded from the digest.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from .entity_split import EntitySplitter, SplitName
from .pairs import NON_MATCH_LABEL, EntityRecord, PairRecord, PairSource

DEFAULT_NEGATIVE_SEED = 42
DEFAULT_HARD_POOL_SIZE = 32
_NGRAM_SIZE = 3
_DRAW_ATTEMPT_MULTIPLIER = 8
_SMALL_PARTITION_SIZE = 256


class NegativeStrategy(StrEnum):
    """How a non-match counterpart is chosen for an anchor name."""

    RANDOM = "random"
    HARD = "hard"


@dataclass(frozen=True)
class NegativeSamplingConfig:
    """Negative-sampling policy applied to a producer's positives.

    Attributes:
        strategy: Which counterpart-selection rule to apply.
        negatives_per_positive: Expected negatives generated per positive pair. A fractional
            value is honoured in expectation: the fractional part is a deterministic per-anchor
            coin flip, so `0.5` yields one negative for about half the positives.
        hard_pool_size: Candidates scored per negative under `NegativeStrategy.HARD`. Larger
            pools produce harder negatives at proportionally more similarity work.
        seed: Seed mixed into every draw.
    """

    strategy: NegativeStrategy = NegativeStrategy.RANDOM
    negatives_per_positive: float = 1.0
    hard_pool_size: int = DEFAULT_HARD_POOL_SIZE
    seed: int = DEFAULT_NEGATIVE_SEED

    def __post_init__(self) -> None:
        if self.negatives_per_positive < 0:
            raise ValueError("negatives_per_positive must not be negative")
        if self.hard_pool_size < 1:
            raise ValueError("hard_pool_size must be at least 1")


def compose_negative_seed(*segments: str) -> int:
    """Combine identifying values into one seed via length-prefixed `blake2b` segments."""
    digest = hashlib.blake2b()
    for segment in segments:
        encoded = segment.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return int.from_bytes(digest.digest(), "big")


def _character_ngrams(text: str) -> frozenset[str]:
    normalized = " ".join(text.casefold().split())
    if len(normalized) <= _NGRAM_SIZE:
        return frozenset({normalized})
    return frozenset(
        normalized[index : index + _NGRAM_SIZE]
        for index in range(len(normalized) - _NGRAM_SIZE + 1)
    )


def lexical_similarity(left: str, right: str) -> float:
    """Character-trigram Jaccard similarity in `[0, 1]`, used to rank hard negatives."""
    left_grams = _character_ngrams(left)
    right_grams = _character_ngrams(right)
    union = left_grams | right_grams
    if not union:
        return 0.0
    return len(left_grams & right_grams) / len(union)


def _candidates_by_split(
    pool: Sequence[EntityRecord], splitter: EntitySplitter
) -> dict[SplitName, list[EntityRecord]]:
    distinct: dict[str, EntityRecord] = {}
    for record in pool:
        if record.system_uri not in distinct and record.name.strip():
            distinct[record.system_uri] = record

    ordered = [distinct[system_uri] for system_uri in sorted(distinct)]
    grouped: dict[SplitName, list[EntityRecord]] = {name: [] for name in SplitName}
    if not ordered:
        return grouped

    splits = splitter.assign_many([record.system_uri for record in ordered])
    for record, split in zip(ordered, splits, strict=True):
        grouped[split].append(record)
    return grouped


def _draw_candidates(
    candidates: Sequence[EntityRecord],
    *,
    rng: random.Random,
    wanted: int,
    exclude_system_uri: str,
    exclude_name: str,
) -> list[EntityRecord]:
    """Draw up to `wanted` distinct eligible candidates from one partition.

    Small partitions are filtered and sampled exactly. Large ones use rejection sampling
    instead, which keeps the cost of each draw independent of partition size: filtering every
    entity in a partition once per anchor would make generation quadratic in entity count.
    """
    if not candidates or wanted < 1:
        return []

    normalized_anchor = exclude_name.casefold().strip()

    def eligible(candidate: EntityRecord) -> bool:
        return (
            candidate.system_uri != exclude_system_uri
            and candidate.name.casefold().strip() != normalized_anchor
        )

    if len(candidates) <= max(_SMALL_PARTITION_SIZE, wanted * 4):
        pool = [candidate for candidate in candidates if eligible(candidate)]
        if len(pool) <= wanted:
            return pool
        return rng.sample(pool, wanted)

    drawn: dict[str, EntityRecord] = {}
    for _ in range(_DRAW_ATTEMPT_MULTIPLIER * wanted):
        if len(drawn) == wanted:
            break
        candidate = candidates[rng.randrange(len(candidates))]
        if eligible(candidate):
            drawn.setdefault(candidate.system_uri, candidate)

    return list(drawn.values())


def _negative_count(rng: random.Random, negatives_per_positive: float) -> int:
    whole = int(negatives_per_positive)
    remainder = negatives_per_positive - whole
    return whole + (1 if remainder > 0 and rng.random() < remainder else 0)


def sample_negatives(
    positives: Sequence[PairRecord],
    *,
    pool: Sequence[EntityRecord],
    splitter: EntitySplitter,
    source: PairSource,
    config: NegativeSamplingConfig | None = None,
) -> list[PairRecord]:
    """Draw non-match pairs for each positive's anchor name, within the anchor's partition.

    A partition holding fewer than two entities simply yields no negatives for the positives in
    it: there is no second entity to pair against.
    """
    effective = config if config is not None else NegativeSamplingConfig()
    if not positives or effective.negatives_per_positive == 0:
        return []

    candidates_by_split = _candidates_by_split(pool, splitter)
    negatives: list[PairRecord] = []

    for positive in positives:
        candidates = candidates_by_split[positive.split]
        rng = random.Random(  # nosec B311 - reproducible synthetic-data generation, not security
            compose_negative_seed(
                str(effective.seed),
                source.value,
                effective.strategy.value,
                positive.split.value,
                positive.left_system_uri,
                positive.left_name,
                positive.right_name,
            )
        )
        wanted = _negative_count(rng, effective.negatives_per_positive)
        if wanted == 0:
            continue

        if effective.strategy is NegativeStrategy.HARD:
            chosen = _draw_hard(
                positive,
                candidates=candidates,
                rng=rng,
                wanted=wanted,
                pool_size=effective.hard_pool_size,
            )
        else:
            chosen = _draw_candidates(
                candidates,
                rng=rng,
                wanted=wanted,
                exclude_system_uri=positive.left_system_uri,
                exclude_name=positive.left_name,
            )

        negatives.extend(
            PairRecord(
                left_name=positive.left_name,
                right_name=candidate.name,
                label=NON_MATCH_LABEL,
                source=source,
                split=positive.split,
                left_system_uri=positive.left_system_uri,
                right_system_uri=candidate.system_uri,
                detail=f"negative={effective.strategy.value}",
            )
            for candidate in chosen
        )

    return negatives


def _draw_hard(
    positive: PairRecord,
    *,
    candidates: Sequence[EntityRecord],
    rng: random.Random,
    wanted: int,
    pool_size: int,
) -> list[EntityRecord]:
    sampled = _draw_candidates(
        candidates,
        rng=rng,
        wanted=max(pool_size, wanted),
        exclude_system_uri=positive.left_system_uri,
        exclude_name=positive.left_name,
    )
    ranked = sorted(
        sampled,
        key=lambda candidate: (
            -lexical_similarity(positive.left_name, candidate.name),
            candidate.name,
            candidate.system_uri,
        ),
    )
    return ranked[:wanted]
