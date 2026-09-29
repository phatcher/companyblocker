from __future__ import annotations

import polars as pl
import pytest

from blocking.comparison import (
    REPEATABILITY_CHECK_COLUMNS,
    check_strategy_comparison_repeatability,
)
from blocking.contracts import BLOCKING_ARTIFACT_SCHEMAS

_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["strategy_comparison"]


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(_SCHEMA, None)
    row.update(
        {
            "label": "tfidf",
            "representation": "tfidf",
            "similarity_backend": "sklearn",
            "stage": "pruned",
            "source_system": "gleif",
            "target_system": "gb",
            "country": "gb",
            "population": "universe",
            "precision": 0.9,
            "recall": 0.8,
            "f1": 0.85,
            "reduction_ratio": 0.99,
            "recall_at_k": 0.8,
            "candidate_set_size_ratio": 0.01,
        }
    )
    row.update(overrides)
    return row


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=_SCHEMA)


def test_identical_reruns_match_on_every_column() -> None:
    first = _frame(_row())
    second = _frame(_row())

    report = check_strategy_comparison_repeatability(first, second)

    assert report.get_column("matches").to_list() == [True]
    for column in REPEATABILITY_CHECK_COLUMNS:
        assert report.get_column(f"{column}_match").to_list() == [True]


def test_a_drifted_metric_is_reported_as_a_mismatch() -> None:
    first = _frame(_row(recall_at_k=0.8))
    second = _frame(_row(recall_at_k=0.6))

    report = check_strategy_comparison_repeatability(first, second)
    row = report.row(0, named=True)

    assert row["matches"] is False
    assert row["recall_at_k_match"] is False
    assert row["precision_match"] is True


def test_both_null_counts_as_a_match_not_a_mismatch() -> None:
    # A run without ground truth: every compared column is null both times, and
    # so is its population, the same "not computed" fact reported twice, not a
    # disagreement. The null population is part of the join key, so this is
    # also what keeps two identical re-runs from splitting into unmatched rows.
    unscored = {
        "population": None,
        "precision": None,
        "recall": None,
        "recall_at_k": None,
    }
    first = _frame(_row(**unscored))
    second = _frame(_row(**unscored))

    report = check_strategy_comparison_repeatability(first, second)

    assert report.height == 1
    assert report.row(0, named=True)["matches"] is True


def test_a_key_present_in_only_one_frame_is_a_mismatch_on_every_column() -> None:
    first = _frame(_row(country="gb"))
    second = _frame(_row(country="ie"))

    report = check_strategy_comparison_repeatability(first, second)

    assert report.height == 2
    assert report.get_column("matches").to_list() == [False, False]


def test_a_value_present_on_only_one_side_is_a_mismatch() -> None:
    first = _frame(_row(recall_at_k=0.8))
    second = _frame(_row(recall_at_k=None))

    report = check_strategy_comparison_repeatability(first, second)

    assert report.row(0, named=True)["recall_at_k_match"] is False


def test_rejects_empty_columns() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        check_strategy_comparison_repeatability(
            _frame(_row()), _frame(_row()), columns=()
        )
