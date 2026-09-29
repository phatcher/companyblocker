"""Break down GLEIF's unrecoverable no-matches by GLEIF's own entity category.

Rejoins `analyze_matches.py`'s not-recoverable no-matches on `LEI` against the raw GLEIF source
shard, since the canonical and cleansed layers do not carry the category (`GENERAL`, `FUND`,
`SOLE_PROPRIETOR` and so on). Needs `analyze_matches.py` run first for the same scenario and
date, and writes its breakdown parquet and chart into that scenario's `metrics/` and `viz/`.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
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

from analysis.match_metrics import resolve_match_analysis_run_root
from analysis.match_metrics_gleif import (
    compute_gleif_not_recoverable_by_entity_category,
)
from analysis.report_layout import metrics_dir, viz_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Break down GLEIF no-match recoverability by GLEIF entity category "
            "(GENERAL, FUND, SOLE_PROPRIETOR, etc.) by rejoining the "
            "scripts/analyze_matches.py no-match output onto the raw GLEIF "
            "source shard via LEI. Requires analyze_matches.py to have already "
            "been run for the same source/target/country/run-date."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--date",
        dest="run_date",
        required=True,
        help="Match-analysis run date (YYYY-MM-DD) to read from "
        "artifacts/analysis/match_analysis/runs/<date>/.",
    )
    parser.add_argument(
        "--target",
        dest="target_system",
        required=True,
        help="Target system code the GLEIF scenario was matched against, "
        "e.g. gb, fr, ie.",
    )
    parser.add_argument(
        "--country",
        default=None,
        help="Country / hive-partition filter. Defaults to --target.",
    )
    parser.add_argument(
        "--target-display",
        default="",
        help="Optional display label, if one was set for the match-analysis run.",
    )
    parser.add_argument(
        "--source-date",
        dest="gleif_source_run_date",
        default=None,
        help="GLEIF source snapshot date (YYYY-MM-DD) to read entity categories "
        "from. Defaults to the latest available snapshot under "
        "data/gleif/source/.",
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the match-analysis scenario and report where the "
            "breakdown and chart would be written, without reading or "
            "writing anything."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    country = args.country or args.target_system

    if args.dry_run:
        run_root = resolve_match_analysis_run_root(
            roots,
            run_date=args.run_date,
            source_system="gleif",
            target_system=args.target_system,
            country=country,
        )
        report_dry_run(
            "analyze_gleif_entity_category",
            run_date=args.run_date,
            target_system=args.target_system,
            country=country,
            target_display=args.target_display,
            gleif_source_run_date=args.gleif_source_run_date,
        )
        report_output_plan(
            "[dry-run] analyze_gleif_entity_category:",
            [
                PlannedOutput(
                    metrics_dir(run_root) / "gleif_entity_category_breakdown.parquet",
                    OUTPUT_CLEAR,
                ),
                PlannedOutput(
                    viz_dir(run_root) / "gleif_entity_category_not_recoverable.png",
                    OUTPUT_CLEAR,
                ),
            ],
        )
        return 0

    breakdown_df, _match_paths, breakdown_path, chart_path = (
        compute_gleif_not_recoverable_by_entity_category(
            roots,
            run_date=args.run_date,
            target_system=args.target_system,
            country=args.country,
            target_display=args.target_display,
            gleif_source_run_date=args.gleif_source_run_date,
        )
    )
    for row in breakdown_df.iter_rows(named=True):
        print(
            f"[gleif_entity_category] {row['entity_category']}: "
            f"not_recoverable={row['not_recoverable_rows']} "
            f"({row['pct_of_not_recoverable_total']}% of total, "
            f"{row['not_recoverable_rate_within_category_pct']}% of category no-matches)"
        )
    print(f"[gleif_entity_category] breakdown: {breakdown_path}")
    print(f"[gleif_entity_category] chart: {chart_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
