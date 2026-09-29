from __future__ import annotations

import polars as pl
import pytest
from company_classify.schema import MetricBundle

from analysis.promotion_report import (
    PROMOTION_REPORT_SCHEMA,
    PromotionCandidate,
    build_promotion_report,
    validate_promotion_report_schema,
    write_promotion_report,
)
from workspace.roots import WorkspaceRoots


def _metric_bundle(**overrides: float) -> MetricBundle:
    base: dict[str, float] = {
        "precision": 0.9,
        "recall": 0.8,
        "f1": 0.847,
        "pr_auc": 0.92,
        "recall_at_k": 0.75,
        "candidate_set_size_ratio": 0.4,
        "precision_at_k": 0.85,
        "duplication_rate": 0.05,
        "latency_per_1k_rows": 1.2,
    }
    base.update(overrides)
    return MetricBundle(**base)


def _run_metrics_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "source_system": "gleif",
        "target_system": "ie",
        "country": "ie",
        "source_rows": 1000,
        "elapsed_seconds": 12.5,
        "latency_seconds_per_1k_rows": 12.5,
        "candidate_edges": 4000,
        "candidate_distinct_targets": 3500,
        "candidate_duplication_ratio": 0.125,
    }
    base.update(overrides)
    return base


def _candidate(
    *,
    strategy: str = "tfidf",
    operating_point: float = 0.5,
    metric_bundle: MetricBundle | None = None,
    run_metrics_row: dict[str, object] | None = None,
) -> PromotionCandidate:
    return PromotionCandidate(
        strategy=strategy,
        operating_point=operating_point,
        metric_bundle=metric_bundle if metric_bundle is not None else _metric_bundle(),
        run_metrics_row=(
            run_metrics_row if run_metrics_row is not None else _run_metrics_row()
        ),
    )


def test_build_promotion_report_carries_both_sides_metrics() -> None:
    report = build_promotion_report(
        [_candidate()],
        promotion_metric="classifier_recall_at_k",
        promotion_threshold=0.7,
    )

    assert report.height == 1
    row = report.row(0, named=True)
    assert row["strategy"] == "tfidf"
    assert row["operating_point"] == 0.5
    assert row["classifier_recall_at_k"] == pytest.approx(0.75)
    assert row["blocking_candidate_duplication_ratio"] == pytest.approx(0.125)
    assert row["promotion_metric"] == "classifier_recall_at_k"
    assert row["promotion_threshold"] == pytest.approx(0.7)
    assert row["promotion_passed"] is True


def test_build_promotion_report_fails_below_threshold() -> None:
    report = build_promotion_report(
        [_candidate()],
        promotion_metric="classifier_recall_at_k",
        promotion_threshold=0.99,
    )

    assert report.row(0, named=True)["promotion_passed"] is False


def test_build_promotion_report_supports_lower_is_better_metric() -> None:
    report = build_promotion_report(
        [_candidate()],
        promotion_metric="blocking_latency_seconds_per_1k_rows",
        promotion_threshold=20.0,
        higher_is_better=False,
    )

    assert report.row(0, named=True)["promotion_passed"] is True


def test_build_promotion_report_null_judged_value_fails_rather_than_raises() -> None:
    candidate = _candidate(
        run_metrics_row=_run_metrics_row(candidate_duplication_ratio=None)
    )

    report = build_promotion_report(
        [candidate],
        promotion_metric="blocking_candidate_duplication_ratio",
        promotion_threshold=0.5,
    )

    assert report.row(0, named=True)["promotion_passed"] is False


def test_build_promotion_report_multiple_strategies_and_operating_points() -> None:
    candidates = [
        _candidate(strategy="tfidf", operating_point=0.5),
        _candidate(
            strategy="sentencepiece",
            operating_point=0.6,
            metric_bundle=_metric_bundle(recall_at_k=0.6),
        ),
    ]

    report = build_promotion_report(
        candidates,
        promotion_metric="classifier_recall_at_k",
        promotion_threshold=0.7,
    )

    assert report.height == 2
    assert set(report["strategy"].to_list()) == {"tfidf", "sentencepiece"}
    passed_by_strategy = dict(
        zip(report["strategy"].to_list(), report["promotion_passed"].to_list())
    )
    assert passed_by_strategy["tfidf"] is True
    assert passed_by_strategy["sentencepiece"] is False


def test_build_promotion_report_raises_on_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_promotion_report(
            [], promotion_metric="classifier_recall_at_k", promotion_threshold=0.5
        )


def test_build_promotion_report_raises_on_unknown_promotion_metric() -> None:
    with pytest.raises(ValueError, match="promotion_metric"):
        build_promotion_report(
            [_candidate()], promotion_metric="recall_at_k", promotion_threshold=0.5
        )


def test_build_promotion_report_raises_on_missing_run_metrics_column() -> None:
    candidate = _candidate(run_metrics_row={"source_system": "gleif"})

    with pytest.raises(ValueError, match="missing column"):
        build_promotion_report(
            [candidate],
            promotion_metric="classifier_recall_at_k",
            promotion_threshold=0.5,
        )


def test_validate_promotion_report_schema_raises_on_missing_column() -> None:
    report = build_promotion_report(
        [_candidate()],
        promotion_metric="classifier_recall_at_k",
        promotion_threshold=0.5,
    )
    dropped = report.drop("promotion_passed")

    with pytest.raises(ValueError, match="missing column"):
        validate_promotion_report_schema(dropped)


def test_validate_promotion_report_schema_raises_on_unknown_column() -> None:
    report = build_promotion_report(
        [_candidate()],
        promotion_metric="classifier_recall_at_k",
        promotion_threshold=0.5,
    )
    extra = report.with_columns(pl.lit("x").alias("unexpected"))

    with pytest.raises(ValueError, match="unrecognized column"):
        validate_promotion_report_schema(extra)


def test_write_promotion_report_round_trips(workspace_roots: WorkspaceRoots) -> None:
    report = build_promotion_report(
        [_candidate()],
        promotion_metric="classifier_recall_at_k",
        promotion_threshold=0.5,
    )

    output_path = write_promotion_report(workspace_roots, "2026-09-07", report)

    assert output_path.exists()
    loaded = pl.read_parquet(output_path)
    assert loaded.height == 1
    assert set(loaded.columns) == set(PROMOTION_REPORT_SCHEMA)
