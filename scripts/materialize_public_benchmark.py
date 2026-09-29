"""Materialize a public two-table entity-matching benchmark into the blocking
loader's shape.

Reads a benchmark root holding `tableA.csv`, `tableB.csv` and a
`matches.csv` (or labelled `train`/`valid`/`test` splits in its place --
`validation.public_benchmark_materializer.load_benchmark_matches`) and
writes two systems, `<benchmark-name>-a` (indexed) and `<benchmark-name>-b`
(queried), under `data/`. Run each straight through `scripts/run_blocking.py`
afterwards with `--source <benchmark-name>-b --target
<benchmark-name>-a --match-col source_uri` -- no loader change is needed,
since the written rows already carry the shape it expects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    add_dry_run_arg,
    add_root_arg,
    report_dry_run,
    run_reporting_argument_errors,
)

from validation.public_benchmark_materializer import (
    DEFAULT_ID_COLUMN,
    DEFAULT_JURISDICTION_CODE,
    PublicBenchmarkMaterializationConfig,
    materialize_public_benchmark,
)
from workspace.roots import default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a public two-table benchmark (tableA.csv/tableB.csv/"
            "matches.csv) into the blocking loader's shape."
        ),
    )
    parser.add_argument(
        "--benchmark-root",
        required=True,
        help="Directory holding tableA.csv, tableB.csv and matches.csv (or "
        "labelled train/valid/test splits).",
    )
    parser.add_argument(
        "--benchmark-name",
        required=True,
        help="Short name for the benchmark, e.g. 'abt-buy'. The two systems "
        "written are '<name>-a' (indexed) and '<name>-b' (queried).",
    )
    parser.add_argument(
        "--id-col",
        default=DEFAULT_ID_COLUMN,
        help=f"Id column shared by tableA.csv/tableB.csv (default: {DEFAULT_ID_COLUMN!r}).",
    )
    parser.add_argument(
        "--jurisdiction-code",
        default=DEFAULT_JURISDICTION_CODE,
        help="Placeholder jurisdiction_code partition value for both systems "
        f"(default: {DEFAULT_JURISDICTION_CODE!r}); neither benchmark carries one.",
    )
    add_root_arg(
        parser,
        help_text="Repository root containing data/ (default: current directory).",
    )
    add_dry_run_arg(parser)
    return parser


def run_materialization(args: argparse.Namespace) -> int:
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)
    benchmark_root = Path(args.benchmark_root).resolve()

    config = PublicBenchmarkMaterializationConfig(
        roots=roots,
        benchmark_root=benchmark_root,
        benchmark_name=str(args.benchmark_name).strip().lower(),
        id_col=str(args.id_col),
        jurisdiction_code=str(args.jurisdiction_code).strip().lower(),
    )

    if args.dry_run:
        report_dry_run(
            "materialize_public_benchmark",
            benchmark_root=str(config.benchmark_root),
            benchmark_name=config.benchmark_name,
            id_col=config.id_col,
            jurisdiction_code=config.jurisdiction_code,
            indexed_system=f"{config.benchmark_name}-a",
            queried_system=f"{config.benchmark_name}-b",
        )
        return 0

    result = materialize_public_benchmark(config)

    print(
        f"[materialize] indexed_system={result.indexed_system} "
        f"rows={result.indexed_rows}"
    )
    print(
        f"[materialize] queried_system={result.queried_system} "
        f"rows={result.queried_rows}"
    )
    print(
        f"[materialize] truth_pairs={result.truth_pairs} "
        f"multi_match_source_count={result.multi_match_source_count}"
    )
    print(f"[materialize] indexed_dir={result.indexed_dir}")
    print(f"[materialize] queried_dir={result.queried_dir}")
    print(
        "[materialize] run blocking with --source "
        f"{result.queried_system} --target {result.indexed_system} "
        "--match-col source_uri"
    )
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_materialization(args)


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
