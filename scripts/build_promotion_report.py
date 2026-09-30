"""Build a promotion report row from a real validation run and a real
classifier metric bundle.

Combines one row of `validation`'s `run_metrics` artifact with one
`company_classify.schema.MetricBundle`, judges the caller's chosen metric
against the caller's chosen threshold, and writes the result under
`artifacts/analysis/promotion_report/runs/<run_date>/promotion_report.parquet`.

Never writes under `data/`. The `--run-metrics-parquet` input is read, not
produced here: generate it with a validation run
(`scripts/validate_clustering.py`). The
`--metric-bundle-json` input is a JSON object with `MetricBundle`'s own
field names (`precision`, `recall`, `f1`, `pr_auc`, `recall_at_k`,
`candidate_set_size_ratio`, `precision_at_k`, `duplication_rate`,
`latency_per_1k_rows`) -- `scripts/measure_pair_classifier_ceiling.py
--json-out` produces a superset report whose `pair_classifier.test_metrics`
key is exactly this shape.

Usage:
    uv run python scripts/build_promotion_report.py \\
        --strategy tfidf --operating-point 0.5 \\
        --run-metrics-parquet tmp/promotion_report/run_metrics.parquet \\
        --source gleif --target ie --country ie \\
        --metric-bundle-json tmp/promotion_report/wikidata_gb_ceiling.json \\
        --metric-bundle-json-path pair_classifier.test_metrics \\
        --promotion-metric classifier_recall_at_k --promotion-threshold 0.7 \\
        --date 2026-09-07
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_dry_run_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)
from company_classify.schema import MetricBundle

from analysis.promotion_report import (
    PromotionCandidate,
    build_promotion_report,
    resolve_promotion_report_dir,
    write_promotion_report,
)
from workspace.roots import WorkspaceRoots


def _load_metric_bundle(json_path: Path, dotted_path: str) -> MetricBundle:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    for key in dotted_path.split("."):
        payload = payload[key]
    return MetricBundle(
        **{field: payload[field] for field in MetricBundle.__dataclass_fields__}
    )


def _load_run_metrics_row(
    parquet_path: Path, *, source_system: str, target_system: str, country: str
) -> dict[str, object]:
    frame = pl.read_parquet(parquet_path)
    matched = frame.filter(
        (pl.col("source_system") == source_system)
        & (pl.col("target_system") == target_system)
        & (pl.col("country") == country)
    )
    if matched.height == 0:
        raise ValueError(
            f"No run_metrics row for {source_system}/{target_system}/{country} "
            f"in {parquet_path}"
        )
    return matched.row(0, named=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--operating-point", type=float, required=True)
    parser.add_argument("--run-metrics-parquet", required=True)
    parser.add_argument("--source", dest="source_system", required=True)
    parser.add_argument("--target", dest="target_system", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--metric-bundle-json", required=True)
    parser.add_argument(
        "--metric-bundle-json-path",
        default="pair_classifier.test_metrics",
        help=(
            "Dotted key path into --metric-bundle-json's object locating "
            "the MetricBundle-shaped dict."
        ),
    )
    parser.add_argument("--promotion-metric", required=True)
    parser.add_argument("--promotion-threshold", type=float, required=True)
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Judge promotion_metric as a cost (pass when <= threshold).",
    )
    parser.add_argument(
        "--date", dest="run_date", default=datetime.now(UTC).strftime("%Y-%m-%d")
    )
    add_workspace_roots_args(parser)
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve every input path and the output path and report the "
            "plan, without reading the metric bundle or run metrics, or "
            "writing anything."
        ),
    )
    return parser


def _resolve_input_path(roots: WorkspaceRoots, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (roots.checkout / path).resolve()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    run_metrics_path = _resolve_input_path(roots, args.run_metrics_parquet)
    metric_bundle_path = _resolve_input_path(roots, args.metric_bundle_json)

    if args.dry_run:
        output_dir = resolve_promotion_report_dir(roots, args.run_date)
        report_dry_run(
            "build_promotion_report",
            strategy=args.strategy,
            operating_point=args.operating_point,
            run_metrics_parquet=run_metrics_path,
            metric_bundle_json=metric_bundle_path,
            promotion_metric=args.promotion_metric,
            promotion_threshold=args.promotion_threshold,
        )
        report_output_plan(
            "[dry-run] build_promotion_report:",
            [PlannedOutput(output_dir / "promotion_report.parquet", OUTPUT_CLEAR)],
        )
        return 0

    metric_bundle = _load_metric_bundle(
        metric_bundle_path, args.metric_bundle_json_path
    )
    run_metrics_row = _load_run_metrics_row(
        run_metrics_path,
        source_system=args.source_system,
        target_system=args.target_system,
        country=args.country,
    )

    candidate = PromotionCandidate(
        strategy=args.strategy,
        operating_point=args.operating_point,
        metric_bundle=metric_bundle,
        run_metrics_row=run_metrics_row,
    )
    report = build_promotion_report(
        [candidate],
        promotion_metric=args.promotion_metric,
        promotion_threshold=args.promotion_threshold,
        higher_is_better=not args.lower_is_better,
    )

    output_path = write_promotion_report(roots, args.run_date, report)
    print(report)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
