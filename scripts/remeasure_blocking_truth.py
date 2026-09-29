"""Re-measure an existing blocking run's ground truth through the shared
URI walk, without re-running candidate generation.

`src/blocking/workflow.py`'s `_build_source_truth_map` now resolves a
source row's cross-system match by walking its own `source_uri`/`match_uri`
chain (`workspace.match_resolution.resolve_cross_system_match`) rather than
reading a `match_uri` column that only happens to be present today because
blocking's source always loads from `matched/`. A run directory written
before that landed still carries `pair_truth_eval`/`pair_truth_eval_detail`
scored the old way. This script re-resolves them through the walk and
rewrites only those two artefacts (plus `raw_pair_truth_eval` when the run
has one) and a `summary.json` note recording that the truth was
re-resolved and when -- every candidate artefact the run already wrote
(`matched_edges.parquet`, `raw_matched_edges.parquet`,
`clusters.parquet`, ...) is read back unchanged and never rewritten, so a
run made before this landed is re-measured in minutes rather than
re-blocked from scratch.

The run is selected by the pairing and representation it was filed under,
and by the key `run_blocking.py` printed when those leave more than one: this
rewrites files, so a selection matching several runs is refused rather than
applied to all of them.

Usage:
    .venv/Scripts/python.exe scripts/remeasure_blocking_truth.py \
        --source gleif --target gb --representation tfidf \
        --top-k 20
"""

from __future__ import annotations

import argparse
import sys

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

from blocking.run_layout import BlockingRunLocation, RunArtefact, select_blocking_runs
from blocking.workflow import remeasure_pair_truth_eval
from workspace.roots import WorkspaceRoots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-resolve an existing blocking run's pair_truth_eval/"
            "pair_truth_eval_detail through the shared URI walk, leaving "
            "every candidate artefact in the run directory untouched."
        ),
    )
    add_workspace_roots_args(parser)
    parser.add_argument("--source", dest="source_system", required=True)
    parser.add_argument("--target", dest="target_system", required=True)
    parser.add_argument(
        "--representation",
        default=None,
        help="The representation the run was made with. Needed when the pairing "
        "holds runs of more than one.",
    )
    parser.add_argument(
        "--run-key",
        default=None,
        help="The key naming the run's directory, as run_blocking.py printed it. "
        "Needed when the pairing and representation hold more than one run.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=(
            "Per-source candidate window recall_at_k is measured over, matching "
            "the original run's own --top-k. A run directory carries no record "
            "of the top_k it was scored with, so this is not inferred: omit it "
            "to leave recall_at_k null on every re-measured row."
        ),
    )
    add_dry_run_arg(parser)
    return parser


def select_run(roots: WorkspaceRoots, args: argparse.Namespace) -> BlockingRunLocation:
    """The one finished run the arguments name, or a `ValueError` saying what
    matched instead."""
    runs = select_blocking_runs(
        roots,
        source_system=args.source_system,
        target_system=args.target_system,
        representation=args.representation,
        run_key=args.run_key,
    )
    if not runs:
        raise ValueError(
            f"no finished {args.source_system} -> {args.target_system} run matches "
            f"representation={args.representation} run_key={args.run_key}"
        )
    if len(runs) > 1:
        matches = ", ".join(
            f"{run.representation}/{run.directory.name}" for run in runs
        )
        raise ValueError(
            f"{len(runs)} runs match ({matches}); name one with --representation "
            "and --run-key, since re-measuring rewrites files"
        )
    return runs[0]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)

    try:
        run = select_run(roots, args)
        if args.dry_run:
            report_dry_run(
                "remeasure_blocking_truth",
                data_dir=roots.data,
                run_dir=run.directory,
                top_k=args.top_k,
            )
            report_output_plan(
                "[dry-run]  ",
                [
                    PlannedOutput(
                        run.path(RunArtefact.PAIR_TRUTH_EVAL_DETAIL), OUTPUT_CLEAR
                    ),
                    PlannedOutput(
                        run.path(RunArtefact.SUMMARY),
                        OUTPUT_EXTEND,
                        note="country blocks replaced, remeasure note added",
                    ),
                ],
            )
            return 0
        result = remeasure_pair_truth_eval(run, roots=roots, top_k=args.top_k)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[remeasure] error: {exc}", file=sys.stderr)
        return 1

    print(
        f"[remeasure] re-resolved truth for {run.directory} at {result.remeasured_at}"
    )
    for path in result.written_paths:
        print(f"[remeasure] {path.name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
