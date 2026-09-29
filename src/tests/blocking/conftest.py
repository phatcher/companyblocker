"""Helpers every blocking test shares.

`run_location`/`comparison_location` wrap a bare directory as the location the
writers take, for a test whose subject is what gets written rather than where
the tree puts it; where a run sits is `test_run_layout.py`'s own subject.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from blocking.run_layout import (
    BlockingComparisonLocation,
    BlockingPairing,
    BlockingRunLocation,
    BlockingSourceKind,
)

_PAIRING = BlockingPairing(
    target_system="gb", source_kind=BlockingSourceKind.DATA, source_segment="gleif"
)


@pytest.fixture
def run_location() -> Callable[[Path], BlockingRunLocation]:
    """A run's location at whatever directory the test hands it."""

    def _location(directory: Path) -> BlockingRunLocation:
        return BlockingRunLocation(
            directory=directory, pairing=_PAIRING, representation="tfidf"
        )

    return _location


@pytest.fixture
def comparison_location() -> Callable[[Path], BlockingComparisonLocation]:
    """A comparison report's location at whatever directory the test hands it."""

    def _location(directory: Path) -> BlockingComparisonLocation:
        return BlockingComparisonLocation(directory=directory, pairing=_PAIRING)

    return _location
