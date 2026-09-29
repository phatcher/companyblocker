from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_dry_run_arg,
    add_run_date_arg,
    add_workspace_roots_args,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from analysis.cluster_shape_pictures import render_cluster_shape_pictures
from analysis.report_layout import viz_dir
from blocking.inspection import compute_cluster_size_distribution
from blocking.run_layout import BlockingRunLocation, RunArtefact, read_run_location
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

REPORT_NAME = "cluster_shape"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster count and node volume against cluster size, "
            "before and after the target-neighbour union, drawn from one "
            "blocking run's own clustering, either side of the "
            "target-neighbour union. "
            "See analysis.cluster_shape_pictures."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--run-dir",
        required=True,
        help=(
            "A blocking run's output directory (holds clusters.parquet and "
            "clusters_before_union.parquet), absolute or relative to the checkout."
        ),
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Where to write the picture, absolute or relative to the checkout. "
            "Defaults to artifacts/analysis/cluster_shape/runs/<date>/viz/"
            "<source>__<target>_<representation>_<run>_cluster_shape.png."
        ),
    )
    add_run_date_arg(
        parser,
        detail="Names the analysis run directory the picture is written under.",
        default_behavior="today",
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run directory and output path and report what would "
            "be written, without reading the run or drawing the picture."
        ),
    )
    return parser


def _resolve_run_dir(roots: WorkspaceRoots, run_dir: str) -> Path:
    path = Path(run_dir)
    return path if path.is_absolute() else (roots.checkout / path).resolve()


def default_output_path(
    roots: WorkspaceRoots, run_date: str, location: BlockingRunLocation
) -> Path:
    """Where the picture of one run goes, named for the run it was drawn from."""
    pairing = location.pairing
    stem = (
        f"{pairing.source_segment}__{pairing.target_system}_"
        f"{location.representation}_{location.directory.name}"
    )
    return (
        viz_dir(analysis_report_run_dir(roots, REPORT_NAME, run_date))
        / f"{stem}_cluster_shape.png"
    )


def _resolve_output_path(
    roots: WorkspaceRoots, run_date: str, run_dir: Path, output_path: str | None
) -> Path:
    if output_path is None:
        return default_output_path(roots, run_date, read_run_location(roots, run_dir))
    path = Path(output_path)
    return path if path.is_absolute() else (roots.checkout / path).resolve()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    roots = resolve_workspace_roots_from_args(args)
    run_date = args.run_date or datetime.now(UTC).date().isoformat()

    run_dir = _resolve_run_dir(roots, args.run_dir)
    output_path = _resolve_output_path(roots, run_date, run_dir, args.output_path)

    if args.dry_run:
        report_output_plan(
            "[dry-run] analyze_cluster_shape:",
            [PlannedOutput(output_path, OUTPUT_CLEAR)],
        )
        return 0

    location = read_run_location(roots, run_dir)
    clusters_path = location.path(RunArtefact.CLUSTERS)
    before_path = location.path(RunArtefact.CLUSTERS_BEFORE_UNION)
    if not clusters_path.exists() or not before_path.exists():
        print(
            f"[error] {run_dir} is missing {clusters_path.name} or "
            f"{before_path.name} -- is this a run with clustering recorded "
            "either side of the target-neighbour union?"
        )
        return 1

    before = compute_cluster_size_distribution(pl.read_parquet(before_path))
    after = compute_cluster_size_distribution(pl.read_parquet(clusters_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_cluster_shape_pictures(
        [("before union", before), ("after union", after)],
        output_path,
        suptitle=f"cluster shape: {run_dir.name}",
    )
    print(f"Picture: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
