from __future__ import annotations

import argparse
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

from analysis.benchmark_loader import benchmark_loader_modes
from analysis.data_loader import DEFAULT_REQUIRED_COLUMNS, LOAD_MODES
from workspace.roots import WorkspaceRoots


def _parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark analysis data loading modes without changing core analysis logic."
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--systems",
        default="fr,gb,gleif,ie",
        help="Comma-separated systems to benchmark.",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional parquet filename to benchmark per system (for example 'sample.parquet').",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of {system}-*.parquet files when input-file is not set; use -1 for all system files.",
    )
    parser.add_argument(
        "--modes",
        default=",".join(LOAD_MODES),
        help="Comma-separated modes to benchmark (full_read,duckdb_pruned).",
    )
    parser.add_argument("--runs", type=int, default=1, help="Benchmark runs per mode.")
    parser.add_argument(
        "--required-columns",
        default=",".join(DEFAULT_REQUIRED_COLUMNS),
        help="Columns to project in duckdb_pruned mode.",
    )
    parser.add_argument(
        "--filter-non-empty-column",
        default="",
        help="Optional text column for non-empty filtering in duckdb_pruned mode; default is disabled.",
    )
    parser.add_argument(
        "--output-parquet",
        default=None,
        help="Optional output parquet path for benchmark summary, absolute or relative to the checkout.",
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Report which systems and modes would be benchmarked and where "
            "the summary would be written, without running any benchmark."
        ),
    )
    return parser


def _resolve_output_path(roots: WorkspaceRoots, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (roots.checkout / path).resolve()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    roots = resolve_workspace_roots_from_args(args)
    systems = _parse_csv_list(args.systems)
    modes = _parse_csv_list(args.modes)
    required_columns = _parse_csv_list(args.required_columns)
    filter_non_empty_column = args.filter_non_empty_column.strip() or None
    output_path = _resolve_output_path(roots, args.output_parquet)

    if args.dry_run:
        report_dry_run(
            "benchmark_analysis_loader",
            systems=systems,
            modes=modes,
            runs=args.runs,
            required_columns=required_columns,
            filter_non_empty_column=filter_non_empty_column,
        )
        if output_path is not None:
            report_output_plan(
                "[dry-run] benchmark_analysis_loader:",
                [PlannedOutput(output_path, OUTPUT_CLEAR)],
            )
        return 0

    rows = []
    for system in systems:
        summary = benchmark_loader_modes(
            roots=roots,
            system=system,
            input_file=args.input_file,
            max_files=args.max_files,
            modes=modes,
            runs=args.runs,
            required_columns=required_columns,
            filter_non_empty_column=filter_non_empty_column,
        )
        if summary.height > 0:
            rows.append(summary)

    combined = pl.concat(rows, how="vertical_relaxed") if rows else None

    if combined is None or combined.height == 0:
        print("No benchmark rows produced (check systems/input-file availability).")
        return 0

    print(combined)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.write_parquet(output_path)
        print(f"Wrote benchmark summary parquet: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
