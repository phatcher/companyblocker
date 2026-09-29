"""Concatenate every pair's `strategy_comparison.parquet` into one cross-country
scaling table, and summarise runtime against target-side scale.

`compare_blocking_strategies.py` writes one report per source/target pair, into
that pair's own `comparison/` directory beside its runs,
`artifacts/blocking/data/<target>/<kind>/<source>/comparison/`. Each
already carries `runtime_seconds` and `target_rows` per `(label, stage,
country)` row, so how a representation/backend scales with target size is
already in the data -- it is just spread across directories with nothing that
puts two countries beside each other. This is that missing step, and no more:
the concatenation and the grouping both live in `blocking.comparison`, and the
only work here is finding the files and writing what comes back.
"""

from __future__ import annotations

import argparse
import sys

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

from blocking.comparison import (
    combine_strategy_comparisons,
    summarize_runtime_scaling,
)
from blocking.reporting import write_strategy_comparison_aggregate
from blocking.run_layout import (
    BlockingComparisonLocation,
    ComparisonArtefact,
    blocking_pairing,
    iter_blocking_comparisons,
    resolve_combined_comparison_location,
)

_AGGREGATE_ARTEFACTS = (
    ComparisonArtefact.COMBINED,
    ComparisonArtefact.COMBINED_CSV,
    ComparisonArtefact.RUNTIME_SCALING,
    ComparisonArtefact.RUNTIME_SCALING_CSV,
)
"""What `write_strategy_comparison_aggregate` writes, for the dry run's plan."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concatenate every source/target pair's strategy_comparison "
        "report into one combined table, and summarise min/median/max runtime "
        "per representation/backend/accelerator against target_rows. Reads the "
        "pairings under --output-dir's blocking tree and writes the combined "
        "report beside that tree."
    )
    add_workspace_roots_args(parser)
    add_dry_run_arg(parser)
    return parser


def _pairing_label(*, source_system: str, target_system: str) -> str:
    """The `<target>/<kind>/<source>` a pairing's report is filed under."""
    return blocking_pairing(
        source_system=source_system, target_system=target_system
    ).label


def _load(location: BlockingComparisonLocation) -> pl.DataFrame:
    path = location.path(ComparisonArtefact.REPORT)
    frame = pl.read_parquet(path)
    print(f"[aggregate-comparisons] read {frame.height} rows from {path}", flush=True)

    if "accelerator_settings" not in frame.columns:
        print(
            f"[aggregate-comparisons] note: {path} was written before the "
            "comparison schema recorded accelerator settings; its rows carry a "
            "null accelerator_settings and are grouped apart from runs whose "
            "settings are known, rather than assumed to be defaults.",
            flush=True,
        )

    # The pair label the aggregate groups on is derived from each row's own
    # source_system/target_system, so a file that has been moved still reports
    # under the pair it describes. A disagreement with the directory it sits
    # in is still worth saying out loud -- it means someone copied a report.
    directory_label = location.pairing.label if location.pairing else None
    row_labels = {
        _pairing_label(source_system=source, target_system=target)
        for source, target in zip(
            frame.get_column("source_system").to_list(),
            frame.get_column("target_system").to_list(),
            strict=True,
        )
    }
    if directory_label is not None and row_labels != {directory_label}:
        print(
            f"[aggregate-comparisons] warning: {path} sits under "
            f"'{directory_label}' but its rows describe {sorted(row_labels)}; "
            "reporting under what the rows say.",
            flush=True,
        )

    return frame


def aggregate(args: argparse.Namespace) -> int:
    roots = resolve_workspace_roots_from_args(args)

    reports = sorted(
        iter_blocking_comparisons(roots),
        key=lambda location: location.directory,
    )
    if not reports:
        raise FileNotFoundError(
            "no pairing has a persisted comparison report -- run "
            "compare_blocking_strategies.py for at least one source/target pair "
            "first, since there is nothing to aggregate until one has persisted "
            "a report."
        )

    output_location = resolve_combined_comparison_location(roots)
    if args.dry_run:
        report_dry_run(
            "aggregate_strategy_comparisons",
            reports=[report.path(ComparisonArtefact.REPORT) for report in reports],
        )
        report_output_plan(
            "[dry-run]  ",
            [
                PlannedOutput(output_location.path(artefact), OUTPUT_CLEAR)
                for artefact in _AGGREGATE_ARTEFACTS
            ],
        )
        return 0

    combined = combine_strategy_comparisons([_load(report) for report in reports])
    runtime_scaling = summarize_runtime_scaling(combined)

    written = write_strategy_comparison_aggregate(
        output_location, combined=combined, runtime_scaling=runtime_scaling
    )

    pairs = combined.get_column("pair_label").n_unique()
    print(
        f"[aggregate-comparisons] combined {combined.height} rows across "
        f"{pairs} pair(s) into {runtime_scaling.height} scaling bucket(s)",
        flush=True,
    )
    for path in written:
        print(f"[aggregate-comparisons] wrote {path}", flush=True)

    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=240):
        print(runtime_scaling)

    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return aggregate(args)
    except (ValueError, FileNotFoundError) as exc:
        # Mirrors compare_blocking_strategies.py's handler: a missing input or
        # a comparison file with a column gone is meant to be read, not decoded
        # from a traceback.
        print(f"[aggregate-comparisons] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
