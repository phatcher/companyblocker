"""Command-line settings this package's code reads, declared as plain data.

A script reads these to build its flags, name each default in its help, and
report which settings a run took and where each value came from. Nothing here
builds a parser: the package stays a library, and the declaration shape is a
documented convention rather than a type shared with other packages.
"""

from __future__ import annotations

from .entity_split import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VALIDATION_RATIO,
)
from .negatives import DEFAULT_HARD_POOL_SIZE, DEFAULT_NEGATIVE_SEED, NegativeStrategy

NEGATIVE_STRATEGIES: tuple[str, ...] = tuple(
    strategy.value for strategy in NegativeStrategy
)
"""The `NegativeSamplingConfig.strategy` choices."""

SETTINGS: tuple[dict[str, object], ...] = (
    {
        "name": "split_seed",
        "type": "int",
        "default": DEFAULT_SPLIT_SEED,
        "help": "Seed mixed into the per-entity train/validation/test split.",
    },
    {
        "name": "split_train_ratio",
        "type": "float",
        "default": DEFAULT_TRAIN_RATIO,
        "help": "Fraction of entities assigned to the train partition.",
    },
    {
        "name": "split_validation_ratio",
        "type": "float",
        "default": DEFAULT_VALIDATION_RATIO,
        "help": (
            "Fraction of entities assigned to the validation partition; "
            "the remainder goes to test."
        ),
    },
    {
        "name": "negative_strategy",
        "type": "str",
        "default": NegativeStrategy.RANDOM.value,
        "choices": NEGATIVE_STRATEGIES,
        "help": "How a negative pair's counterpart is chosen: random or hard.",
    },
    {
        "name": "negatives_per_positive",
        "type": "float",
        "default": 1.0,
        "help": "Expected negatives generated per positive pair.",
    },
    {
        "name": "negative_seed",
        "type": "int",
        "default": DEFAULT_NEGATIVE_SEED,
        "help": "Seed mixed into every negative draw.",
    },
    {
        "name": "hard_pool_size",
        "type": "int",
        "default": DEFAULT_HARD_POOL_SIZE,
        "help": "Candidates scored per negative under the hard negative strategy.",
    },
)
"""Every setting a pair-producer run built on this package's split and
negative-sampling contracts takes."""
