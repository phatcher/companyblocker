"""Draw a pairing's per-pair outcomes and write the figures and tables, without
running anything.

Reads what `report_strategy_comparison.py` wrote for the pairing, every run's
verdict on every truth pair, and writes under
`artifacts/analysis/pair_outcomes/runs/<date>/`, figures in `viz/` and tables
in `metrics/`, each named for its pairing and level, what
`notebooks/analyse_pair_outcomes.ipynb` shows inline with its default settings: the UpSet plot of which runs found which pairs, the
run overlap heatmaps, the name-similarity histogram against the random-pair
level, the source cluster sizes, and the tables behind them. A pairing with a
single run gets no UpSet or overlap plot, since neither has two runs to set
side by side.

With `--source` and `--target` omitted, every pairing whose comparison holds
per-pair outcomes is drawn.

Usage:
    .venv/Scripts/python.exe scripts/plot_pair_outcomes.py \
        --source gleif --target ie
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib.pyplot as plt
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_dry_run_arg,
    add_run_date_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from analysis.pair_outcomes import (
    cluster_sizes,
    count_by_reach,
    count_found_by,
    draw_cluster_sizes,
    draw_found_by_upset,
    draw_name_similarity,
    draw_run_overlap,
    drop_unanimous_pairs,
    latest_runs,
    level_metrics,
    pair_found_matrix,
    random_pair_level,
    run_columns,
    run_overlap,
    select_runs,
    with_name_similarity,
)
from analysis.report_layout import metrics_dir, viz_dir
from blocking.run_layout import (
    BlockingComparisonLocation,
    BlockingPairing,
    ComparisonArtefact,
    iter_blocking_comparisons,
    resolve_comparison_location,
)
from workspace.artifact_layout import (
    analysis_report_run_dir,
    analysis_report_runs_root,
)
from workspace.roots import WorkspaceRoots

REPORT_NAME = "pair_outcomes"
FOUND_BY_TABLE = "found_by.csv"
REACH_TABLE = "reach.csv"
LEVEL_METRICS_TABLE = "level_metrics.csv"
FOUND_BY_UPSET = "found_by_upset.png"
RUN_OVERLAP = "run_overlap.png"
NAME_SIMILARITY = "name_similarity.png"
CLUSTER_SIZES = "cluster_sizes.png"
_TABLES = (FOUND_BY_TABLE, REACH_TABLE, LEVEL_METRICS_TABLE)
_FIGURES = (FOUND_BY_UPSET, RUN_OVERLAP, NAME_SIMILARITY, CLUSTER_SIZES)
_DPI = 110


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a pairing's per-pair outcomes under "
            "artifacts/analysis/pair_outcomes/runs/<date>/, running nothing. "
            "Omit --source and --target together to draw every pairing whose "
            "comparison holds per-pair outcomes."
        ),
    )
    add_workspace_roots_args(parser)
    add_run_date_arg(
        parser,
        detail="Names the analysis run directory the output is written under.",
        default_behavior="today",
    )
    parser.add_argument("--source", dest="source_system", default=None)
    parser.add_argument("--target", dest="target_system", default=None)
    parser.add_argument(
        "--level",
        default="never",
        help="The name-equality level whose pairs are drawn (default: never, the "
        "residual pairs no exact join finds).",
    )
    parser.add_argument(
        "--latest-per",
        default="representation",
        help="Keep the most recently finished run per value of this run column "
        "(default: representation).",
    )
    parser.add_argument(
        "--counts",
        default="both",
        choices=("exact", "at_least", "both"),
        help="What each UpSet bar counts: the pairs found by exactly its runs, by "
        "at least its runs, or both (default).",
    )
    parser.add_argument(
        "--random-quantile",
        type=float,
        default=0.99,
        help="The quantile of unrelated names' similarity taken as the random-pair "
        "level (default: 0.99).",
    )
    add_dry_run_arg(parser)
    return parser


def output_path(
    roots: WorkspaceRoots,
    run_date: str,
    pairing: BlockingPairing,
    level: str,
    name: str,
) -> Path:
    """Where one pairing's figure or table for one level is written."""
    run_dir = analysis_report_run_dir(roots, REPORT_NAME, run_date)
    directory = metrics_dir(run_dir) if name in _TABLES else viz_dir(run_dir)
    return (
        directory / f"{pairing.source_segment}__{pairing.target_system}_{level}_{name}"
    )


def latest_output(
    roots: WorkspaceRoots, pairing: BlockingPairing, level: str, name: str
) -> Path | None:
    """The most recently dated run's copy of one pairing's figure or table, if any run drew it."""
    runs_root = analysis_report_runs_root(roots, REPORT_NAME)
    if not runs_root.is_dir():
        return None
    for run_dir in sorted(runs_root.iterdir(), reverse=True):
        path = output_path(roots, run_dir.name, pairing, level, name)
        if path.is_file():
            return path
    return None


def comparison_pairing(location: BlockingComparisonLocation) -> BlockingPairing:
    """The pairing a per-pairing comparison describes; the combined report has none."""
    if location.pairing is None:
        raise ValueError("the combined comparison report describes no single pairing")
    return location.pairing


def _population(level: str) -> str:
    return "residual pairs" if level == "never" else f"level {level!r} pairs"


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _draw_pairing(
    location: BlockingComparisonLocation,
    *,
    roots: WorkspaceRoots,
    run_date: str,
    level: str,
    latest_per: str,
    counts: str,
    random_quantile: float,
) -> list[Path]:
    """Write one pairing's figures and tables and return their paths."""
    runs = pl.read_parquet(location.path(ComparisonArtefact.PAIR_OUTCOME_RUNS))
    outcomes = pl.read_parquet(location.path(ComparisonArtefact.PAIR_OUTCOMES))
    source_candidates = pl.read_parquet(
        location.path(ComparisonArtefact.PAIR_OUTCOME_SOURCE_CANDIDATES)
    )
    chosen = latest_runs(
        select_runs(
            runs,
            labels=None,
            representation=None,
            tokenizer=None,
            similarity_backend=None,
        ),
        by=latest_per,
    )
    run_names = dict(zip(chosen["label"], chosen[latest_per]))
    matrix = pair_found_matrix(
        outcomes,
        chosen,
        name_equality=level,
        country=None,
        run_names=run_names,
        min_similarity={},
        max_rank=None,
    )
    random_level = random_pair_level(matrix, quantile=random_quantile)
    marked = with_name_similarity(matrix, random_level=random_level)
    pairing = comparison_pairing(location)
    heading = (
        f"{pairing.source_segment} -> {pairing.target_system}: {_population(level)}"
    )

    def out(name: str) -> Path:
        path = output_path(roots, run_date, pairing, level, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    written: list[Path] = []
    for name, frame in (
        (FOUND_BY_TABLE, count_found_by(matrix)),
        (REACH_TABLE, count_by_reach(marked)),
        (
            LEVEL_METRICS_TABLE,
            level_metrics(
                outcomes,
                chosen,
                source_candidates,
                name_equality=level,
                country=None,
                run_names=run_names,
                min_similarity={},
                max_rank=None,
                random_level=random_level,
                above_random_only=False,
            ),
        ),
    ):
        frame.write_csv(out(name))
        written.append(out(name))

    if len(run_columns(matrix)) >= 2:
        remainder = drop_unanimous_pairs(matrix, found_by_none=True, found_by_all=False)
        fig = plt.figure(figsize=(11, 5.5))
        draw_found_by_upset(fig, remainder, title=f"{heading} by source", counts=counts)
        written.append(_save(fig, out(FOUND_BY_UPSET)))

        overlap = run_overlap(matrix)
        fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
        draw_run_overlap(left, overlap, value="jaccard", title="Jaccard")
        draw_run_overlap(right, overlap, value="containment", title="containment")
        fig.suptitle(f"{heading} by source")
        fig.tight_layout()
        written.append(_save(fig, out(RUN_OVERLAP)))
    else:
        print(
            f"[pair-outcomes] {pairing.source_segment} -> {pairing.target_system}: "
            "one run only, so no UpSet or overlap plot"
        )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    draw_name_similarity(
        ax, marked, random_level=random_level, title=f"{heading} by name similarity"
    )
    written.append(_save(fig, out(NAME_SIMILARITY)))

    sizes = cluster_sizes(
        matrix,
        source_candidates,
        chosen,
        run_names=run_names,
        min_similarity={},
        max_rank=None,
        random_level=random_level,
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    draw_cluster_sizes(ax, sizes, title=f"{heading} by source cluster size")
    written.append(_save(fig, out(CLUSTER_SIZES)))
    return written


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.source_system is None) != (args.target_system is None):
        parser.error("--source and --target are given together or omitted together")
    roots = resolve_workspace_roots_from_args(args)
    run_date = args.run_date or datetime.now(UTC).date().isoformat()

    locations = (
        [
            resolve_comparison_location(
                roots,
                source_system=args.source_system,
                target_system=args.target_system,
            )
        ]
        if args.source_system is not None
        else [
            location
            for location in iter_blocking_comparisons(roots)
            if location.path(ComparisonArtefact.PAIR_OUTCOMES).is_file()
        ]
    )

    if args.dry_run:
        report_dry_run(
            "plot_pair_outcomes",
            pairings=[
                (pairing.source_segment, pairing.target_system)
                for pairing in map(comparison_pairing, locations)
            ],
            level=args.level,
            latest_per=args.latest_per,
            counts=args.counts,
        )
        report_output_plan(
            "[dry-run]  ",
            [
                PlannedOutput(
                    output_path(
                        roots, run_date, comparison_pairing(location), args.level, name
                    ),
                    OUTPUT_CLEAR,
                )
                for location in locations
                for name in (*_TABLES, *_FIGURES)
            ],
        )
        return 0

    drawn = 0
    for location in locations:
        pairing = comparison_pairing(location)
        if not location.path(ComparisonArtefact.PAIR_OUTCOMES).is_file():
            print(
                f"[pair-outcomes] {pairing.source_segment} -> {pairing.target_system}: "
                "no per-pair outcomes; build them with report_strategy_comparison.py",
                file=sys.stderr,
            )
            continue
        written = _draw_pairing(
            location,
            roots=roots,
            run_date=run_date,
            level=args.level,
            latest_per=args.latest_per,
            counts=args.counts,
            random_quantile=args.random_quantile,
        )
        print(
            f"[pair-outcomes] {pairing.source_segment} -> {pairing.target_system}: "
            f"{len(written)} file(s) under "
            f"{analysis_report_run_dir(roots, REPORT_NAME, run_date)}"
        )
        drawn += 1
    if drawn == 0:
        print(
            "[pair-outcomes] error: no pairing with per-pair outcomes", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
