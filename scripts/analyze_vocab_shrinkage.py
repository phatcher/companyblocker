from __future__ import annotations

import argparse
from datetime import UTC, datetime

import _bootstrap  # noqa: F401
from cli_common import (
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
from analysis.report_layout import metrics_dir, viz_dir
from analysis.vocab_shrinkage import run_vocab_shrinkage_analysis
from workspace.artifact_layout import analysis_report_run_dir

_DECLARATIONS = declared_settings(ANALYSIS_SETTINGS)
_SURFACE: dict[str, dict[str, object]] = {
    "v_min": {"flag": "--v-min"},
    "budget_points": {"flag": "--budget-points"},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pooled-tokenizer vocabulary-shrinkage forecast. For each "
            "named system and a log-spaced grid of vocabulary budgets V, "
            "compares that corpus's own top-V tokens (by raw-tier document "
            "frequency) against a count-weighted and an equal-weighted pooled "
            "top-V, and reports the occurrence mass of the corpus's own top-V "
            "the pool drops -- a curve per corpus per pool weight, plus a "
            "chosen-V overlap diagram (Venn/UpSet) of the per-corpus top-V "
            "sets, split into in-language and OOV panels by wordfreq "
            "membership. Reads an existing analysis.token_zipf run's raw-tier "
            "token stats via the same projection analysis.noise_layers "
            "uses; does not run token_zipf itself."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--systems",
        required=True,
        help="Comma-separated system codes to compare (not auto-discovered).",
    )
    parser.add_argument(
        "--date",
        dest="run_date",
        default=datetime.now(UTC).date().isoformat(),
        help="Run date in YYYY-MM-DD format for this module's own run tree.",
    )
    parser.add_argument(
        "--zipf-run-date",
        default=None,
        help=(
            "Run date of the existing analysis.token_zipf run to read raw-tier "
            "token stats from. Defaults to --date."
        ),
    )
    parser.add_argument(
        "--budgets",
        default=None,
        help=(
            "Explicit comma-separated V values, overriding --v-min/--v-max/"
            "--budget-points."
        ),
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--v-max",
        type=int,
        default=None,
        help=(
            "Largest budget in the generated log grid. Defaults to the largest "
            "named corpus's own raw-tier vocab size."
        ),
    )
    parser.add_argument(
        "--chosen-v",
        type=int,
        default=None,
        help=(
            "Budget V for the overlap diagram. Defaults to the largest budget "
            "in the effective grid."
        ),
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


def _parse_budgets(value: str | None) -> list[int] | None:
    if value is None:
        return None
    budgets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not budgets:
        raise SystemExit(f"--budgets expects at least one integer, got: {value!r}")
    return budgets


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    if not systems:
        parser.error("--systems must name at least one system.")
    roots = resolve_workspace_roots_from_args(args)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    values = resolved_setting_values(resolved)
    budgets = _parse_budgets(args.budgets)

    if args.dry_run:
        run_root = analysis_report_run_dir(roots, "vocab_shrinkage", args.run_date)
        report_resolved_settings(
            "analyze_vocab_shrinkage",
            resolved,
            systems=systems,
            budgets=budgets,
            v_max=args.v_max,
            chosen_v=args.chosen_v,
        )
        report_output_plan(
            "[dry-run] analyze_vocab_shrinkage:",
            [
                PlannedOutput(
                    metrics_dir(run_root),
                    OUTPUT_EXTEND,
                    note="each named weight's curve/membership file is "
                    "overwritten; a prior run's file for a system dropped "
                    "from --systems is kept",
                ),
                PlannedOutput(
                    viz_dir(run_root),
                    OUTPUT_EXTEND,
                    note="each named weight's chart is overwritten; a prior "
                    "run's chart for a system dropped from --systems is kept",
                ),
            ],
        )
        return 0

    paths = run_vocab_shrinkage_analysis(
        roots,
        systems=systems,
        run_date=args.run_date,
        zipf_run_date=args.zipf_run_date,
        budgets=budgets,
        v_min=values["v_min"],
        v_max=args.v_max,
        budget_points=values["budget_points"],
        chosen_v=args.chosen_v,
        progress=lambda message: print(f"[info] {message}"),
    )

    print(f"Metrics: {paths.metrics_dir}")
    print(f"Plots: {paths.viz_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
