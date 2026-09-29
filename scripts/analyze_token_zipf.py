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
from analysis.token_zipf import run_token_zipf_analysis, run_trim_profile_sweep
from workspace.artifact_layout import analysis_report_run_dir

_DECLARATIONS = declared_settings(ANALYSIS_SETTINGS)
_SURFACE: dict[str, dict[str, object]] = {
    "wordfreq_top_n": {"flag": "--wordfreq-top-n"},
    "highlight_regressions": {"flag": "--highlight-regressions"},
    "compare_power_law": {"flag": "--power-law-compare"},
    "force": {"flag": "--force"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank-frequency (Zipf) divergence evidence for cleansed company "
            "names. Sweeps raw/basic/cleansed name-column tiers per system, computed "
            "directly from text (pre-tokenizer-trim, so results aren't confounded by "
            "tokenize-stage noise-word removal), "
            "reports type-level and occurrence-weighted OOV against wordfreq, and "
            "name-level hapax/OOV incidence, alongside a rank-frequency plot per "
            "system/tier overlaid against a general-language reference curve."
        )
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
        help=(
            "Optional comma-separated system codes. If omitted, discovered from "
            "data/<system>/cleansed/*.parquet."
        ),
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--trim-sweep",
        action="store_true",
        help=(
            "Sweep tokenize-stage noise-word trim settings instead of the name-column "
            "tier (holds name-column tier fixed at --name-col, default name_cleansed). "
            "Not auto-discovered -- --systems is required in this mode, since it "
            "multiplies run count by up to 5 settings per system."
        ),
    )
    parser.add_argument(
        "--name-col",
        default="name_cleansed",
        help="Name column held fixed for --trim-sweep mode (default: name_cleansed).",
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run directory and report the settings and output "
            "paths that would apply, without reading any data, fitting, or "
            "rendering anything."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    systems = (
        [item.strip() for item in args.systems.split(",") if item.strip()]
        if args.systems
        else None
    )
    roots = resolve_workspace_roots_from_args(args)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    values = resolved_setting_values(resolved)

    if args.trim_sweep and not systems:
        parser.error("--trim-sweep requires --systems (not auto-discovered).")

    if args.dry_run:
        run_name = f"{args.run_date}-trim-sweep" if args.trim_sweep else args.run_date
        run_root = analysis_report_run_dir(roots, "token_zipf", run_name)
        report_resolved_settings(
            "analyze_token_zipf",
            resolved,
            systems=systems or "discovered",
            trim_sweep=args.trim_sweep,
            name_col=args.name_col if args.trim_sweep else None,
        )
        report_output_plan(
            "[dry-run] analyze_token_zipf:",
            [
                PlannedOutput(
                    metrics_dir(run_root),
                    OUTPUT_CLEAR,
                    note="the summary is written whole; a second run on this "
                    "date replaces it with only this run's own systems",
                ),
                PlannedOutput(
                    viz_dir(run_root),
                    OUTPUT_EXTEND,
                    note="each named system/tier plot is overwritten; a prior "
                    "run's plot for a system dropped from --systems is kept",
                ),
                PlannedOutput(reports_dir(run_root), OUTPUT_CLEAR),
            ],
        )
        return 0

    if args.trim_sweep:
        assert systems is not None  # nosec B101 - type narrowing; parser.error above exits otherwise
        paths = run_trim_profile_sweep(
            roots,
            systems=systems,
            name_col=args.name_col,
            run_date=args.run_date,
            wordfreq_top_n=values["wordfreq_top_n"],
            progress=lambda message: print(f"[info] {message}"),
            force=values["force"],
        )
    else:
        paths = run_token_zipf_analysis(
            roots,
            systems=systems,
            run_date=args.run_date,
            wordfreq_top_n=values["wordfreq_top_n"],
            highlight_regressions=values["highlight_regressions"],
            compare_power_law=values["compare_power_law"],
            progress=lambda message: print(f"[info] {message}"),
            force=values["force"],
        )

    print(f"Summary: {paths.summary_path}")
    print(f"Report: {paths.report_path}")
    print(f"Plots: {paths.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
