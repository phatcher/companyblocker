from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from blocking.comparison import combine_strategy_comparisons, summarize_runtime_scaling
from blocking.contracts import BLOCKING_ARTIFACT_SCHEMAS
from blocking.reporting import write_strategy_comparison_aggregate

_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["strategy_comparison"]

_ACCELERATED = "exact_name_filter=true;prefix_filter=true"
_BASELINE = "exact_name_filter=true"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(_SCHEMA, None)
    row.update(
        {
            "label": "tfidf",
            "representation": "tfidf",
            "similarity_backend": "sklearn",
            "is_exact_backend": True,
            "accelerator_settings": _BASELINE,
            "runtime_seconds": 100.0,
            "stage": "pruned",
            "source_system": "gleif",
            "target_system": "gb",
            "country": "gb",
            "target_rows": 5_698_275,
            "precision": 0.88,
            "recall": 0.96,
        }
    )
    row.update(overrides)
    return row


def _frame(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=_SCHEMA)


def _both_stages(**overrides: object) -> list[dict[str, object]]:
    return [_row(stage="pruned", **overrides), _row(stage="raw", **overrides)]


def test_combine_derives_pair_label_from_the_rows_own_systems() -> None:
    combined = combine_strategy_comparisons(
        [
            _frame(*_both_stages(target_system="gb", country="gb")),
            _frame(
                *_both_stages(target_system="ie", country="ie", target_rows=820_000)
            ),
        ]
    )

    assert combined.get_column("pair_label").unique().sort().to_list() == [
        "gleif__gb",
        "gleif__ie",
    ]
    assert combined.columns[0] == "pair_label"


def test_combine_backfills_accelerator_settings_for_a_file_predating_the_column() -> (
    None
):
    legacy = _frame(*_both_stages()).drop("accelerator_settings")

    combined = combine_strategy_comparisons([legacy])

    assert combined.get_column("accelerator_settings").null_count() == combined.height


def test_scaling_summary_carries_the_runs_phase_figures() -> None:
    # Two legs in one bucket: the index build and the scan are medians, the
    # memory figure is the larger peak, and a leg with no phases at all
    # leaves the median to the other.
    combined = combine_strategy_comparisons(
        [
            _frame(
                *_both_stages(
                    country="gb",
                    target_index_seconds=586.5,
                    scoring_seconds=120.0,
                    peak_rss_bytes=4_600_000_000,
                )
            ),
            _frame(
                *_both_stages(
                    label="tfidf-2",
                    country="gb",
                    target_index_seconds=600.5,
                    scoring_seconds=100.0,
                    peak_rss_bytes=3_000_000_000,
                )
            ),
            _frame(*_both_stages(label="tfidf-3", country="gb")),
        ]
    )

    summary = summarize_runtime_scaling(combined)

    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["median_target_index_seconds"] == 593.5
    assert row["median_scoring_seconds"] == 110.0
    assert row["max_peak_rss_bytes"] == 4_600_000_000


def test_combine_backfills_the_phase_figures_for_a_file_predating_them() -> None:
    legacy = _frame(*_both_stages()).drop(
        "target_index_seconds", "scoring_seconds", "peak_rss_bytes"
    )

    combined = combine_strategy_comparisons([legacy])

    assert combined.get_column("peak_rss_bytes").null_count() == combined.height


def test_a_legacy_null_never_groups_with_a_known_unaccelerated_run() -> None:
    # "Unknown" and "known to be default" are different facts; pooling them
    # would put a run whose accelerator settings nobody recorded into the
    # baseline bucket.
    legacy = _frame(*_both_stages(target_system="ie", country="ie")).drop(
        "accelerator_settings"
    )
    known = _frame(*_both_stages(accelerator_settings=_BASELINE))

    summary = summarize_runtime_scaling(combine_strategy_comparisons([legacy, known]))

    assert summary.height == 2
    assert summary.get_column("accelerator_settings").null_count() == 1


def test_combine_rejects_a_frame_missing_a_column_that_is_not_backfillable() -> None:
    broken = _frame(*_both_stages()).drop("target_rows")

    with pytest.raises(ValueError, match="target_rows"):
        combine_strategy_comparisons([broken])


def test_combine_rejects_no_comparisons() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        combine_strategy_comparisons([])


def test_summarize_keeps_an_accelerated_leg_out_of_its_own_baseline() -> None:
    # The aggregation's terminal condition: same representation, same backend, same
    # label, same scale -- only the accelerator differs.
    accelerated = _frame(
        *_both_stages(accelerator_settings=_ACCELERATED, runtime_seconds=30.0)
    )
    baseline = _frame(
        *_both_stages(accelerator_settings=_BASELINE, runtime_seconds=210.0)
    )

    summary = summarize_runtime_scaling(
        combine_strategy_comparisons([accelerated, baseline])
    )

    assert summary.height == 2
    by_settings = dict(
        zip(
            summary.get_column("accelerator_settings").to_list(),
            summary.get_column("median_runtime_seconds").to_list(),
            strict=True,
        )
    )
    assert by_settings == {_ACCELERATED: 30.0, _BASELINE: 210.0}


def test_summarize_buckets_target_rows_by_decade() -> None:
    combined = combine_strategy_comparisons(
        [
            _frame(
                *_both_stages(target_system="ie", country="ie", target_rows=820_000)
            ),
            _frame(
                *_both_stages(target_system="gb", country="gb", target_rows=5_698_275)
            ),
            _frame(
                *_both_stages(target_system="fr", country="fr", target_rows=12_900_000)
            ),
        ]
    )

    summary = summarize_runtime_scaling(combined)

    assert summary.get_column("target_rows_bucket").to_list() == [
        "1e5-1e6",
        "1e6-1e7",
        "1e7-1e8",
    ]
    assert summary.get_column("target_rows_bucket_start").to_list() == [
        100_000,
        1_000_000,
        10_000_000,
    ]


def test_summarize_counts_a_run_once_rather_than_once_per_emitted_row() -> None:
    # runtime_seconds is repeated onto every row a run emits (per country,
    # times the pruned/raw stage pair). Weighting by row count would make a
    # three-country run look like three measurements of the same number.
    combined = combine_strategy_comparisons(
        [
            _frame(
                *_both_stages(country="gb"),
                *_both_stages(country="ie", target_rows=1_200_000),
            )
        ]
    )
    assert combined.height == 4

    summary = summarize_runtime_scaling(combined)

    # Both countries sit in the same decade, so this is one bucket: two legs
    # out of four rows, not four.
    assert summary.get_column("run_legs").to_list() == [2]
    # ...and the runtime in it is one whole-run figure covering both, so it is
    # an upper bound on either country's time rather than a measurement of it.
    assert summary.get_column("runtime_covers_countries").to_list() == [2]


def test_summarize_reports_a_single_country_runtime_as_covering_one_country() -> None:
    summary = summarize_runtime_scaling(
        combine_strategy_comparisons([_frame(*_both_stages())])
    )

    assert summary.get_column("run_legs").to_list() == [1]
    assert summary.get_column("runtime_covers_countries").to_list() == [1]
    assert summary.get_column("min_runtime_seconds").to_list() == [100.0]
    assert summary.get_column("max_runtime_seconds").to_list() == [100.0]
    assert summary.get_column("median_precision").to_list() == [0.88]
    assert summary.get_column("median_recall").to_list() == [0.96]


def test_summarize_tolerates_a_run_with_no_scored_rows() -> None:
    # A run against a source without ground truth contributes one all-null
    # metric row per stage, target_rows included.
    combined = combine_strategy_comparisons(
        [_frame(*_both_stages(target_rows=None, precision=None, recall=None))]
    )

    summary = summarize_runtime_scaling(combined)

    assert summary.height == 1
    assert summary.get_column("target_rows_bucket").to_list() == [None]
    assert summary.get_column("target_rows_bucket_start").to_list() == [None]


def test_write_strategy_comparison_aggregate_writes_parquet_and_csv(
    tmp_path: Path, comparison_location
) -> None:
    combined = combine_strategy_comparisons([_frame(*_both_stages())])
    runtime_scaling = summarize_runtime_scaling(combined)

    written = write_strategy_comparison_aggregate(
        comparison_location(tmp_path),
        combined=combined,
        runtime_scaling=runtime_scaling,
    )

    assert sorted(path.name for path in written) == [
        "strategy_comparison_combined.csv",
        "strategy_comparison_combined.parquet",
        "strategy_comparison_runtime_scaling.csv",
        "strategy_comparison_runtime_scaling.parquet",
    ]
    assert all(path.exists() for path in written)
    reread = pl.read_parquet(tmp_path / "strategy_comparison_combined.parquet")
    assert reread.columns == combined.columns
