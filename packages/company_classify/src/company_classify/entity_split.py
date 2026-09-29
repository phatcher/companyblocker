"""Deterministic, order-independent entity-level train/validation/test split.

Every pair produced for self-supervised training is tagged with the split its *entity*
(`system_uri`) belongs to, not the split its row happened to land in. `split.py`'s
`split_examples()` cannot serve that purpose: it shuffles an externally-assembled list, so its
answer for a given entity depends on what else was in the list and in what order. Three
independently-built producers (`pair_producers.py`) sharing only a seed value would therefore
disagree about which split an entity belongs to, and a name variant of an entity trained on
would leak into the evaluation partition.

The split here is instead a pure function of `(seed, system_uri)`:
`hash(seed, system_uri) % SPLIT_RESOLUTION`, mapped onto contiguous ratio bands. Any producer,
in any process, holding only the seed can compute any entity's membership without coordination.
This follows the repository's ratified vectorized-Polars seeding convention (see
`docs/architecture/design-principles.md`, "Deterministic seeding"), whose reference
implementation is `company_tokenize/optimize.py`'s `pl.col("row_nr").hash(seed=seed) %
split_resolution`; the only change here is hashing `system_uri` instead of a row index, so the
result is a property of the entity rather than of its position in a file.

Scalar and vectorized access go through the same Polars hash (`assign()` is `assign_many()` on a
one-element series), so a per-entity lookup and a whole-dataframe expression can never diverge.

K-fold needs nothing further: `with_seed()` returns the same splitter under a different seed (K
seeds, K independently reconstructable splits), and `fold()`/`fold_expr()` expose the same hash
partitioned K ways when a single hash is preferred over K seeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import polars as pl

SPLIT_RESOLUTION = 1_000_000
DEFAULT_SPLIT_SEED = 42
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VALIDATION_RATIO = 0.15
DEFAULT_ENTITY_COLUMN = "system_uri"


class SplitName(StrEnum):
    """Partition an entity belongs to under `EntitySplitter`."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class SplitRatios:
    """Fractions of entities assigned to each partition.

    Attributes:
        train: Fraction assigned to `SplitName.TRAIN`, in `(0, 1)`.
        validation: Fraction assigned to `SplitName.VALIDATION`, in `[0, 1)`.

    The test fraction is whatever is left over (`test`), and must be greater than zero. The
    defaults (70/15/15) match `split.py`'s `split_examples()` defaults, so pair-level and
    example-level work share one set of proportions.
    """

    train: float = DEFAULT_TRAIN_RATIO
    validation: float = DEFAULT_VALIDATION_RATIO

    def __post_init__(self) -> None:
        if not 0 < self.train < 1:
            raise ValueError("train must be in (0, 1)")
        if not 0 <= self.validation < 1:
            raise ValueError("validation must be in [0, 1)")
        if self.train + self.validation >= 1:
            raise ValueError("train + validation must be < 1")

    @property
    def test(self) -> float:
        return 1.0 - self.train - self.validation


@dataclass(frozen=True)
class EntitySplitter:
    """The one shared split every pair producer tags against.

    Attributes:
        seed: Seed mixed into the per-entity hash. Two splitters with the same seed and ratios
            agree on every entity, whatever order entities are presented in.
        ratios: Partition proportions.
    """

    seed: int = DEFAULT_SPLIT_SEED
    ratios: SplitRatios = field(default_factory=SplitRatios)

    def with_seed(self, seed: int) -> EntitySplitter:
        """Return the same split policy under a different seed: one fold of a K-fold run."""
        return EntitySplitter(seed=seed, ratios=self.ratios)

    def fraction_expr(self, column: str = DEFAULT_ENTITY_COLUMN) -> pl.Expr:
        """Polars expression yielding each entity's position in `[0, 1)`."""
        return (pl.col(column).hash(seed=self.seed) % SPLIT_RESOLUTION).cast(
            pl.Float64
        ) / float(SPLIT_RESOLUTION)

    def split_expr(self, column: str = DEFAULT_ENTITY_COLUMN) -> pl.Expr:
        """Polars expression yielding each entity's `SplitName` value as a string column."""
        fraction = self.fraction_expr(column)
        return (
            pl.when(fraction < self.ratios.train)
            .then(pl.lit(SplitName.TRAIN.value))
            .when(fraction < self.ratios.train + self.ratios.validation)
            .then(pl.lit(SplitName.VALIDATION.value))
            .otherwise(pl.lit(SplitName.TEST.value))
            .alias("split")
        )

    def fold_expr(self, *, folds: int, column: str = DEFAULT_ENTITY_COLUMN) -> pl.Expr:
        """Polars expression yielding each entity's fold index in `[0, folds)`."""
        if folds < 2:
            raise ValueError("folds must be at least 2")
        return (pl.col(column).hash(seed=self.seed) % folds).alias("fold")

    def assign_many(self, system_uris: Sequence[str]) -> list[SplitName]:
        """Assign a batch of entities, one `SplitName` per input position."""
        if not system_uris:
            return []
        frame = _entity_frame(system_uris)
        values = frame.select(self.split_expr()).to_series().to_list()
        return [SplitName(value) for value in values]

    def assign(self, system_uri: str) -> SplitName:
        """Assign one entity. Identical by construction to that entity's `assign_many()` slot."""
        return self.assign_many([system_uri])[0]

    def fold_many(self, system_uris: Sequence[str], *, folds: int) -> list[int]:
        """Assign a batch of entities to fold indices, one per input position."""
        if not system_uris:
            return []
        frame = _entity_frame(system_uris)
        values = frame.select(self.fold_expr(folds=folds)).to_series().to_list()
        return [int(value) for value in values]

    def fold(self, system_uri: str, *, folds: int) -> int:
        """Assign one entity to a fold index in `[0, folds)`."""
        return self.fold_many([system_uri], folds=folds)[0]

    def partition(self, system_uris: Sequence[str]) -> dict[SplitName, list[str]]:
        """Group distinct entities by partition, each group sorted for reproducible output."""
        distinct = sorted(set(system_uris))
        grouped: dict[SplitName, list[str]] = {name: [] for name in SplitName}
        for system_uri, split in zip(distinct, self.assign_many(distinct), strict=True):
            grouped[split].append(system_uri)
        return grouped


def _entity_frame(system_uris: Sequence[str]) -> pl.DataFrame:
    for system_uri in system_uris:
        if not isinstance(system_uri, str) or not system_uri.strip():
            raise ValueError(
                "system_uri values must be non-empty strings; "
                f"got {system_uri!r}. An entity with no identity cannot have a "
                "reproducible split membership."
            )
    return pl.DataFrame(
        {DEFAULT_ENTITY_COLUMN: pl.Series(list(system_uris), dtype=pl.String)}
    )
