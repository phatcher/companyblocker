"""No test may match a broad-tier signature without a marker, beyond the baseline.

`tooling/check_test_tier_markers.py` is the measurement: a test matches when its own
source, or the source of a fixture it depends on transitively, starts a subprocess,
materialises a layer, or reads the real `data/` corpus, and it carries neither
`@pytest.mark.integration` nor `@pytest.mark.performance`. This is the ratchet half of
the rule: the 39 tests the measurement found the day it was settled are real
`@pytest.mark.integration` decorators in their own files now (a reviewable diff, not a
runtime reclassification), and `src/tests/baselines/test_tier_markers.txt` starts empty
-- a new offender fails the gate instead of quietly running in the wrong tier.
`docs/TESTSTYLE.md` states the rule.

The measurement needs a real fixture-resolved collection of the whole tree. A full run
has already made one, so the ratchet reads that; a run collecting only a subset has not,
and falls back to `measure()`, which starts its own `pytest --collect-only` in a
subprocess. That fallback is why this test is tagged `integration` too -- the ratchet's
own gate is exactly as broad as what it measures.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import check_test_tier_markers
import pytest
from baseline import check_baseline
from check_test_tier_markers import (
    EXEMPT_TEST_FILES,
    _collects_the_whole_tree,
    measure_in_session,
)


def test_exempt_files_are_the_classifiers_own_direct_tests() -> None:
    """The one exemption is the classifier's own sample-source tests, not a
    growing list -- a real offender is fixed, not exempted."""
    assert EXEMPT_TEST_FILES == {"src/tests/tooling/test_test_tier_classifier.py"}


@pytest.mark.parametrize(
    ("file_or_dir", "keyword", "expected"),
    [
        ([], "", True),
        (["src/tests/analysis"], "", False),
        ([], "zipf", False),
    ],
)
def test_only_an_unrestricted_run_counts_as_collecting_the_whole_tree(
    file_or_dir: list[str], keyword: str, expected: bool
) -> None:
    """A run given paths or a `-k` expression collects a subset, and a subset holding
    no unmarked match is not evidence that none exists."""
    config = SimpleNamespace(
        option=SimpleNamespace(file_or_dir=file_or_dir, keyword=keyword)
    )
    assert _collects_the_whole_tree(cast(pytest.Config, config)) is expected


def test_measure_in_session_reads_the_kept_collection_instead_of_starting_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The saving is the whole point: a run that already collected the tree must not
    spawn a second collection to be told what it just collected."""

    def fail_if_called() -> dict[str, int]:
        raise AssertionError("measure_in_session spawned a subprocess pass")

    monkeypatch.setattr(check_test_tier_markers, "measure", fail_if_called)
    monkeypatch.setattr(check_test_tier_markers, "_collected_items", [])

    assert measure_in_session() == {}


@pytest.mark.integration
def test_no_test_matches_a_broad_tier_signature_without_a_marker_beyond_the_baseline() -> (
    None
):
    current = measure_in_session()
    check_baseline("test_tier_markers", current)
