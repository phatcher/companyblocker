from __future__ import annotations

import argparse
from datetime import UTC, datetime

import _bootstrap  # noqa: F401
from cli_common import (
    OUTPUT_CLEAR,
    OUTPUT_EXTEND,
    PlannedOutput,
    add_declared_arguments,
    add_dry_run_arg,
    add_workspace_roots_args,
    declared_settings,
    report_output_plan,
    report_resolved_settings,
    resolve_declared_settings,
    resolve_workspace_roots_from_args,
    resolved_setting_values,
    run_reporting_argument_errors,
)

from analysis._cli_helper import SETTINGS as ANALYSIS_SETTINGS
from analysis.report_layout import metrics_dir, reports_dir, viz_dir
from analysis.token_metrics import run_phase1_token_analysis
from workspace.artifact_layout import analysis_report_run_dir

_DECLARATIONS = declared_settings(ANALYSIS_SETTINGS)
_SURFACE: dict[str, dict[str, object]] = {
    "top_n": {"flag": "--top-n"},
    "stoplist_min_coverage_ratio": {"flag": "--stoplist-min-coverage-ratio"},
    "stoplist_min_global_df_pct": {"flag": "--stoplist-min-global-df-pct"},
    "engine": {"flag": "--engine", "default": "duckdb"},
    "threads": {"flag": "--threads", "default": 0},
    "verbose_progress": {"flag": "--verbose", "default": True},
}


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 1 token analysis: country/global token metrics, statistical report, and country plots."
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--date",
        dest="run_date",
        default=datetime.now(UTC).date().isoformat(),
        help="Run date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--systems",
        default=None,
        help="Optional comma-separated system codes to include. If omitted, systems are discovered from data/*/tokenized/.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional parquet filename to process per system (for example 'sample.parquet').",
    )
    parser.add_argument(
        "--country-token-col",
        default="country_tokens",
        help="Column used for per-country token metrics.",
    )
    parser.add_argument(
        "--global-token-col",
        default="global_tokens",
        help="Column used for global token metrics.",
    )
    parser.add_argument(
        "--legal-form-col",
        default="company_type",
        help="Column containing legal-form values used for legal-form token filtering.",
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--top-n-plot",
        dest="top_n",
        type=int,
        help=argparse.SUPPRESS,
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run directory and report the settings and output "
            "paths that would apply, without reading any data or rendering "
            "anything."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    roots = resolve_workspace_roots_from_args(args)
    systems = _parse_csv_list(args.systems) if args.systems else None
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    values = resolved_setting_values(resolved)

    if args.dry_run:
        run_root = analysis_report_run_dir(roots, "token_performance", args.run_date)
        report_resolved_settings(
            "analyze_tokens",
            resolved,
            systems=systems or "discovered",
            country_token_col=args.country_token_col,
            global_token_col=args.global_token_col,
            legal_form_col=args.legal_form_col,
        )
        report_output_plan(
            "[dry-run] analyze_tokens:",
            [
                PlannedOutput(metrics_dir(run_root), OUTPUT_CLEAR),
                PlannedOutput(reports_dir(run_root), OUTPUT_CLEAR),
                PlannedOutput(viz_dir(run_root), OUTPUT_CLEAR),
                PlannedOutput(
                    run_root / "run_manifest.json",
                    OUTPUT_EXTEND,
                    note="this phase's entry is replaced; every other phase's entry is kept",
                ),
            ],
        )
        return 0

    paths = run_phase1_token_analysis(
        roots,
        run_date=args.run_date,
        systems=systems,
        country_token_column=args.country_token_col,
        global_token_column=args.global_token_col,
        legal_form_column=args.legal_form_col,
        input_file=args.input_file,
        top_n=values["top_n"],
        stoplist_min_coverage_ratio=values["stoplist_min_coverage_ratio"],
        stoplist_min_global_df_pct=values["stoplist_min_global_df_pct"],
        engine=values["engine"],
        progress=lambda message: print(f"[info] {message}"),
        verbose_progress=values["verbose_progress"],
        threads=values["threads"],
    )

    print(f"Country stats: {paths.country_stats_path}")
    print(f"Global stats: {paths.global_stats_path}")
    print(f"Report: {paths.report_path}")
    print(f"Country plot: {paths.top_tokens_plot_path}")
    print(
        f"Country plot (excluding legal form): {paths.top_tokens_ex_legal_form_plot_path}"
    )
    print(f"Country TF-IDF plot: {paths.top_tfidf_plot_path}")
    print(
        f"Country TF-IDF plot (excluding legal form): {paths.top_tfidf_ex_legal_form_plot_path}"
    )
    print(f"Stoplist candidates: {paths.stoplist_candidates_path}")
    print(f"Stoplist candidates CSV: {paths.stoplist_candidates_csv_path}")
    print(f"Overview plot: {paths.overview_plot_path}")
    print(f"Run manifest: {paths.run_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
