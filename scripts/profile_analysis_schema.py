from __future__ import annotations

import argparse
from datetime import UTC, datetime

import _bootstrap  # noqa: F401
from cli_common import (
    OUTPUT_CLEAR,
    OUTPUT_EXTEND,
    PlannedOutput,
    add_dry_run_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from analysis.io_contract import (
    build_contract,
    profile_data_contract,
    write_schema_profile_artifacts,
)
from analysis.report_layout import reports_dir
from workspace.artifact_layout import analysis_report_run_dir


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase 0 schema profiling for token performance analysis using data/<system>/tokenized parquet outputs."
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--systems",
        default=None,
        help="Optional comma-separated systems to profile. Defaults to discovered tokenized systems.",
    )
    parser.add_argument(
        "--date",
        dest="run_date",
        default=None,
        help="Run date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional parquet filename to profile per system (for example 'sample.parquet').",
    )
    parser.add_argument(
        "--required-columns",
        default=None,
        help="Optional comma-separated required column override.",
    )
    parser.add_argument(
        "--optional-columns",
        default=None,
        help="Optional comma-separated optional column override.",
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run directory and report where the profile and "
            "report would be written, without scanning any data or writing "
            "anything."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    roots = resolve_workspace_roots_from_args(args)
    run_date = args.run_date or datetime.now(UTC).date().isoformat()

    systems = _parse_csv_list(args.systems)
    required = _parse_csv_list(args.required_columns)
    optional = _parse_csv_list(args.optional_columns)

    if args.dry_run:
        run_root = analysis_report_run_dir(roots, "token_performance", run_date)
        report_dry_run(
            "profile_analysis_schema",
            systems=systems or "discovered",
            input_file=args.input_file,
        )
        report_output_plan(
            "[dry-run] profile_analysis_schema:",
            [
                PlannedOutput(
                    run_root / "profile" / "schema_profile.parquet", OUTPUT_CLEAR
                ),
                PlannedOutput(
                    reports_dir(run_root) / "schema_profile.md", OUTPUT_CLEAR
                ),
                PlannedOutput(
                    run_root / "run_manifest.json",
                    OUTPUT_EXTEND,
                    note="this phase's entry is replaced; every other phase's entry is kept",
                ),
            ],
        )
        return 0

    contract = build_contract(
        required_columns=required or None, optional_columns=optional or None
    )
    profile_df, drift_notes = profile_data_contract(
        roots=roots,
        systems=systems or None,
        contract=contract,
        input_file=args.input_file,
    )
    parquet_path, markdown_path, manifest_path = write_schema_profile_artifacts(
        roots=roots,
        run_date=run_date,
        profile_df=profile_df,
        drift_notes=drift_notes,
        contract=contract,
    )

    required_failures = profile_df.filter(
        profile_df["required"] & ~profile_df["present"]
    ).height
    print(f"Profile rows: {profile_df.height}")
    print(f"Required failures: {required_failures}")
    print(f"Drift notes: {len(drift_notes)}")
    print(f"Parquet: {parquet_path}")
    print(f"Report: {markdown_path}")
    print(f"Run manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
