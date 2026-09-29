"""Promotion report combining classifier and blocking metrics.

Reads `validation`'s `run_metrics` artifact
(`validation.contracts.ARTIFACT_SCHEMAS["run_metrics"]`, produced by
`validation.runner.compute_run_metrics`) and `company_classify`'s
`MetricBundle` (`company_classify.schema.MetricBundle`, returned by
`evaluate.compute_binary_metrics()`) side by side, one row per `(strategy,
operating_point)`. Neither area's output is changed or re-derived here: this
module is a reshape over what each already emits, the same spirit as
`blocking.comparison.build_strategy_comparison()` but across areas rather
than across blocking runs alone.

A package or area never reads another's outputs, so this cross-area
synthesis lives here rather than inside the classifier or validation. A
later acceptance-gate policy (dataset-profile thresholds)
is expected to consume this report's shape; this module only guarantees the
shape and a single caller-supplied threshold exist, not what policy chooses
that threshold.

`operating_point` is the classifier score threshold `metric_bundle` was
computed at (the `threshold` argument passed to `evaluate.
compute_binary_metrics()`). `MetricBundle` itself carries no threshold field
-- one bundle is scored at exactly one threshold, and the caller who chose it
is the only party that knows what it was, so `PromotionCandidate` carries it
alongside rather than this module trying to recover it after the fact.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import polars as pl
from company_classify.schema import MetricBundle

from validation.contracts import ARTIFACT_SCHEMAS
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

REPORT_NAME = "promotion_report"

# polars accepts a dtype class (e.g. pl.Utf8) or an instance (e.g. pl.Utf8())
# interchangeably in schema dicts; pl.DataType alone only covers the latter.
# Restated rather than imported, as in `residual_pictures.py`: a typing idiom
# no module owns, unlike `ARTIFACT_SCHEMAS` above, whose columns are imported
# from the module that computes them precisely so they cannot drift.
PolarsDType = type[pl.DataType] | pl.DataType

_RUN_METRICS_SCHEMA: dict[str, PolarsDType] = ARTIFACT_SCHEMAS["run_metrics"]

# 1:1 copy of `MetricBundle`'s own fields, in the dataclass's declared order.
# Prefixed with `classifier_`/`blocking_` on the assembled report so neither
# side's column names can collide -- both define their own
# latency-per-1k-rows figure, computed over different populations (the
# classifier's scoring pass versus a blocking run's candidate generation).
_CLASSIFIER_METRIC_FIELDS: tuple[str, ...] = (
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "recall_at_k",
    "candidate_set_size_ratio",
    "precision_at_k",
    "duplication_rate",
    "latency_per_1k_rows",
)

PROMOTION_REPORT_SCHEMA: dict[str, PolarsDType] = {
    "strategy": pl.Utf8,
    "operating_point": pl.Float64,
    **{f"classifier_{field}": pl.Float64 for field in _CLASSIFIER_METRIC_FIELDS},
    **{f"blocking_{column}": dtype for column, dtype in _RUN_METRICS_SCHEMA.items()},
    "promotion_metric": pl.Utf8,
    "promotion_threshold": pl.Float64,
    "promotion_passed": pl.Boolean,
}

# The metric columns a caller may judge a promotion decision against --
# everything on the report except the identifying and decision columns
# themselves.
_JUDGEABLE_COLUMNS: frozenset[str] = frozenset(PROMOTION_REPORT_SCHEMA) - {
    "strategy",
    "operating_point",
    "promotion_metric",
    "promotion_threshold",
    "promotion_passed",
}


@dataclass(frozen=True)
class PromotionCandidate:
    """One `(strategy, operating_point)` row to score into a promotion report.

    Attributes:
        strategy: Caller-assigned label for the blocking strategy this row
            measures, matching `blocking.comparison.StrategyRunEntry.label`'s
            convention -- a name (e.g. `"tfidf"`), not a config dump.
        operating_point: The classifier score threshold `metric_bundle` was
            computed at. See the module docstring for why this is supplied
            here rather than read off `metric_bundle`.
        metric_bundle: `company_classify.schema.MetricBundle` for this
            strategy at `operating_point`.
        run_metrics_row: One row of `validation`'s `run_metrics` artifact
            for the same strategy, keyed by
            `validation.contracts.ARTIFACT_SCHEMAS["run_metrics"]`'s own
            column names (`source_system`, `target_system`, `country`,
            `source_rows`, `elapsed_seconds`,
            `latency_seconds_per_1k_rows`, `candidate_edges`,
            `candidate_distinct_targets`, `candidate_duplication_ratio`).
    """

    strategy: str
    operating_point: float
    metric_bundle: MetricBundle
    run_metrics_row: Mapping[str, object]


def build_promotion_report(
    candidates: Sequence[PromotionCandidate],
    *,
    promotion_metric: str,
    promotion_threshold: float,
    higher_is_better: bool = True,
) -> pl.DataFrame:
    """Build one promotion report row per `PromotionCandidate`.

    `promotion_metric` must be one of this report's own metric columns (see
    `PROMOTION_REPORT_SCHEMA`), not a raw `MetricBundle`/`run_metrics` field
    name -- pass e.g. `"classifier_recall_at_k"` or
    `"blocking_candidate_duplication_ratio"`. A row's `promotion_passed` is
    `judged_value >= promotion_threshold` when `higher_is_better` (the
    default -- most of this report's metrics are "more is better"), or
    `judged_value <= promotion_threshold` for a cost-shaped metric such as a
    latency column. `promotion_passed` is `False`, not raised, when the
    judged value is null (the "not computed" convention this report's inputs
    already use).

    This function makes no promotion policy of its own: which metric to
    judge and what threshold counts as a pass are the caller's choice, one
    per report. A later acceptance-gate policy plugs in here without this
    schema changing.

    Raises `ValueError` if `candidates` is empty, if `promotion_metric` is
    not a column this report produces, or if a candidate's
    `run_metrics_row` is missing a column `validation`'s `run_metrics`
    schema requires.
    """
    if not candidates:
        raise ValueError("candidates must be non-empty to build a promotion report")

    if promotion_metric not in _JUDGEABLE_COLUMNS:
        raise ValueError(
            f"promotion_metric {promotion_metric!r} is not one of this report's "
            f"metric columns: {sorted(_JUDGEABLE_COLUMNS)}"
        )

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        missing = [
            column
            for column in _RUN_METRICS_SCHEMA
            if column not in candidate.run_metrics_row
        ]
        if missing:
            raise ValueError(
                f"run_metrics_row for strategy {candidate.strategy!r} is missing "
                f"column(s): {missing}"
            )

        row: dict[str, object] = {
            "strategy": candidate.strategy,
            "operating_point": float(candidate.operating_point),
        }
        for field in _CLASSIFIER_METRIC_FIELDS:
            row[f"classifier_{field}"] = getattr(candidate.metric_bundle, field)
        for column in _RUN_METRICS_SCHEMA:
            row[f"blocking_{column}"] = candidate.run_metrics_row[column]

        judged_value = row[promotion_metric]
        if judged_value is None:
            passed = False
        else:
            judged_number = cast(float, judged_value)
            passed = (
                judged_number >= promotion_threshold
                if higher_is_better
                else judged_number <= promotion_threshold
            )
        row["promotion_metric"] = promotion_metric
        row["promotion_threshold"] = float(promotion_threshold)
        row["promotion_passed"] = passed
        rows.append(row)

    report = pl.DataFrame(rows, schema=PROMOTION_REPORT_SCHEMA)
    validate_promotion_report_schema(report)
    return report


def validate_promotion_report_schema(frame: pl.DataFrame) -> None:
    """Validate `frame` carries exactly `PROMOTION_REPORT_SCHEMA`'s columns.

    Raises `ValueError` naming what is missing or unexpected, so a caller
    assembling a report by hand (rather than through `build_promotion_report`)
    fails loudly at build time instead of shipping a report a later reader
    cannot rely on.
    """
    missing = [
        column for column in PROMOTION_REPORT_SCHEMA if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Promotion report missing column(s): {missing}")

    unknown = [
        column for column in frame.columns if column not in PROMOTION_REPORT_SCHEMA
    ]
    if unknown:
        raise ValueError(f"Promotion report has unrecognized column(s): {unknown}")


def resolve_promotion_report_dir(roots: WorkspaceRoots, run_date: str) -> Path:
    """Where one dated promotion-report run keeps its output."""
    return analysis_report_run_dir(roots, REPORT_NAME, run_date)


def write_promotion_report(
    roots: WorkspaceRoots, run_date: str, report: pl.DataFrame
) -> Path:
    """Validate and persist `report` as `promotion_report.parquet`.

    Returns the path written, under
    `artifacts/analysis/promotion_report/runs/<run_date>/`.
    """
    validate_promotion_report_schema(report)
    output_dir = resolve_promotion_report_dir(roots, run_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "promotion_report.parquet"
    report.write_parquet(output_path)
    return output_path
