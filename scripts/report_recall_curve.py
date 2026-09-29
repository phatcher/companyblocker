"""Report a completed blocking run's recall-against-comparisons-spent
curve and its normalised area, and rank two or more such runs against
each other's shared budget, without re-running candidate generation.

The runs are one pairing's, narrowed by `--representation` and `--run-key`
when given, each already holding `matched_edges.parquet`,
`pair_truth_eval_detail.parquet` and `manifest.json` (written by
`run_blocking.py` for a run that had ground truth and name forms on both
sides). `recall_curve.parquet` and `recall_curve_summary.parquet` are
written alongside them. When the selection holds two or more runs, they are
also ranked on `--population`'s area over the smallest budget any of them
reached, printed as a table.

Usage:
    .venv/Scripts/python.exe scripts/report_recall_curve.py \
        --source gleif --target ie \
        --representation tfidf --representation sentencepiece \
        --population never
"""

from __future__ import annotations

import argparse
import sys

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

from blocking.run_layout import BlockingRunLocation, RunArtefact, select_blocking_runs
from blocking.workflow import RecallCurveReportResult, compute_recall_curve_for_run
from validation.contracts import POPULATION_UNIVERSE
from validation.recall_curve import rank_by_recall_area
from workspace.roots import WorkspaceRoots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report a completed blocking run's recall-against-comparisons-spent "
            "curve and normalised area, ranking two or more runs when selected."
        ),
    )
    add_workspace_roots_args(parser)
    parser.add_argument("--source", dest="source_system", required=True)
    parser.add_argument("--target", dest="target_system", required=True)
    parser.add_argument(
        "--representation",
        action="append",
        dest="representations",
        default=None,
        help="Report only runs made with this representation. Repeat to keep "
        "several; omit for every representation the pairing holds.",
    )
    parser.add_argument(
        "--run-key",
        action="append",
        dest="run_keys",
        default=None,
        help="Report only the run whose directory this key names, as "
        "run_blocking.py printed it. Repeat to keep several.",
    )
    parser.add_argument(
        "--population",
        default=POPULATION_UNIVERSE,
        help=(
            "The population two or more runs are ranked on ('universe' or a "
            "name-equality level such as 'never'). Ignored when one run is "
            "selected. Default: %(default)s."
        ),
    )
    add_dry_run_arg(parser)
    return parser


def select_runs(
    roots: WorkspaceRoots, args: argparse.Namespace
) -> list[BlockingRunLocation]:
    """Every finished run of the pairing the arguments keep."""
    return [
        run
        for run in select_blocking_runs(
            roots, source_system=args.source_system, target_system=args.target_system
        )
        if (args.representations is None or run.representation in args.representations)
        and (args.run_keys is None or run.directory.name in args.run_keys)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)

    runs = select_runs(roots, args)
    if not runs:
        print(
            f"[recall-curve] error: no finished {args.source_system} -> "
            f"{args.target_system} run matches representations={args.representations} "
            f"run_keys={args.run_keys}",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        report_dry_run(
            "report_recall_curve",
            run_dirs=[run.directory for run in runs],
            population=args.population,
        )
        report_output_plan(
            "[dry-run]  ",
            [
                PlannedOutput(run.path(artefact), OUTPUT_CLEAR)
                for run in runs
                for artefact in (
                    RunArtefact.RECALL_CURVE,
                    RunArtefact.RECALL_CURVE_SUMMARY,
                )
            ],
        )
        return 0

    reports: list[tuple[str, RecallCurveReportResult]] = []
    for run in runs:
        try:
            report = compute_recall_curve_for_run(run)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[recall-curve] error: {exc}", file=sys.stderr)
            return 1
        reports.append((f"{run.representation}/{run.directory.name}", report))

    for label, report in reports:
        summary_row = report.summary.row(0, named=True)
        print(
            f"[recall-curve] {label}: backend={summary_row['similarity_backend']} "
            f"recall_area_never={summary_row['recall_area_never']}"
        )
        for path in report.written_paths:
            print(f"[recall-curve] {path.name}: {path}")

    if len(reports) > 1:
        ranking = rank_by_recall_area(
            [
                (
                    label,
                    report.summary.row(0, named=True)["similarity_backend"],
                    report.curve,
                )
                for label, report in reports
            ],
            population=args.population,
        )
        print(f"[recall-curve] ranking on population={args.population}:")
        for row in ranking.iter_rows(named=True):
            print(
                f"[recall-curve]   {row['label']} "
                f"({row['similarity_backend']}): "
                f"recall_area={row['recall_area']} "
                f"shared_budget={row['shared_budget']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
