"""Operator families under the inverted contract (`sited_operator.py`).

Each module here declares where its family could act and how it makes one
change, and registers a `SitedOperator` per operator. How many changes are
attempted, whether each fires and which site it lands on belong to the runner,
not to any family here.

Importing this package registers every family in `sited_registry`.
"""

from __future__ import annotations

from . import (
    diacritic_punct,
    homoglyph,
    legal_suffix,
    low_salience,
    phonetic,
    typo,
    word_order,
)

__all__ = [
    "diacritic_punct",
    "homoglyph",
    "legal_suffix",
    "low_salience",
    "phonetic",
    "typo",
    "word_order",
]
