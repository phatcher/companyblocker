"""A deterministic example-level train/validation/test split.

`entity_split.EntitySplitter` is the entity-level counterpart that pair generation uses.
"""

from __future__ import annotations

import random

from .schema import DatasetSplit, LabeledExample


def split_examples(
    examples: list[LabeledExample],
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetSplit:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")

    test_ratio = 1.0 - train_ratio - validation_ratio
    if test_ratio <= 0:
        raise ValueError("train_ratio + validation_ratio must be < 1")

    items = examples.copy()
    # random.Random(seed) is this repository's ratified scalar-Python seeding convention
    # (see docs/architecture.md's "Deterministic seeding" section).
    # This usage predates that ratification but was confirmed to already match it as-is --
    # no change needed here.
    random.Random(seed).shuffle(items)  # nosec B311 - deterministic split, not security

    n_total = len(items)
    n_train = int(n_total * train_ratio)
    n_validation = int(n_total * validation_ratio)

    train = items[:n_train]
    validation = items[n_train : n_train + n_validation]
    test = items[n_train + n_validation :]

    return DatasetSplit(train=train, validation=validation, test=test)
