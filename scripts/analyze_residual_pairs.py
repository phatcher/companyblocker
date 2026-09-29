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
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)

from analysis.report_layout import viz_dir
from analysis.residual_pictures import (
    build_residual_picture_frame,
    render_residual_pictures,
    resolve_residual_pairs_path,
)
from blocking.run_layout import blocking_pairing, iter_blocking_runs
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def discover_latest_run_dir(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    target_system: str,
    representation: str,
) -> Path | None:
    """Most recently written of this pairing's finished runs under
    `representation` that holds a per-pair truth artefact, by either name
    `resolve_residual_pairs_path` accepts (the artefact's current and
    pre-rename names) -- mirrors `notebooks/analyse_residual_pairs.ipynb`'s
    own discovery cell so the script and the notebook find the same run given
    the same source/target/representation. The runs come from
    `blocking.run_layout`, which owns where they sit. Returns `None` if no run
    has ever written either artefact for this pairing/representation.
    """
    pairing = blocking_pairing(source_system=source_system, target_system=target_system)
    dated: list[tuple[float, Path]] = []
    for location in iter_blocking_runs(roots):
        if location.pairing != pairing or location.representation != representation:
            continue
        try:
            artefact = resolve_residual_pairs_path(location.directory)
        except FileNotFoundError:
            continue
        dated.append((artefact.stat().st_mtime, location.directory))
    if not dated:
        return None
    return max(dated)[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Four pictures over the true blocking residual (the "
            "never-level truth pairs, still unequal after cleansing, a blocking run's residual-pairs "
            "artefact carries) -- a token-Jaccard histogram found against missed, "
            "cumulative share found by rank, cumulative share by source cluster "
            "size, and a hex layer over Jaccard against name-length difference "
            "coloured by missed fraction. See analysis.residual_pictures."
        )
    )
    add_workspace_roots_args(parser)
    parser.add_argument("--source", dest="source_system", required=True)
    parser.add_argument("--target", dest="target_system", required=True)
    parser.add_argument(
        "--country",
        default=None,
        help="Optional country filter. Omit to draw every country in the run.",
    )
    parser.add_argument(
        "--representations",
        default="tfidf",
        help=(
            "Comma-separated representation labels to overlay as series, each "
            "auto-discovered under artifacts/blocking/data/<target>/<kind>/<source>/"
            "<representation>/ (blocking.contracts.resolve_blocking_pairing_dir). "
            "Default: tfidf."
        ),
    )
    parser.add_argument(
        "--run-dirs",
        default=None,
        help=(
            "Optional comma-separated input run directories to read, one per "
            "--representations entry in the same order, overriding auto-discovery. "
            "Absolute, or relative to the checkout."
        ),
    )
    parser.add_argument(
        "--date",
        dest="run_date",
        default=datetime.now(UTC).date().isoformat(),
        help="Run date in YYYY-MM-DD format, used for the output path.",
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve each representation's run directory and the output "
            "path, and report the plan, without reading any run's data or "
            "drawing the pictures."
        ),
    )
    return parser


def resolve_output_path(
    roots: WorkspaceRoots, run_date: str, source_system: str, target_system: str
) -> Path:
    return (
        viz_dir(analysis_report_run_dir(roots, "residual_pictures", run_date))
        / f"{source_system}__{target_system}_residual_pictures.png"
    )


def _resolve_run_dirs(
    roots: WorkspaceRoots,
    *,
    representations: list[str],
    explicit_run_dirs: list[str] | None,
    source_system: str,
    target_system: str,
) -> list[Path | None]:
    if explicit_run_dirs is None:
        return [
            discover_latest_run_dir(
                roots,
                source_system=source_system,
                target_system=target_system,
                representation=representation,
            )
            for representation in representations
        ]
    resolved: list[Path | None] = []
    for raw_dir in explicit_run_dirs:
        run_dir = Path(raw_dir)
        resolved.append(
            run_dir if run_dir.is_absolute() else (roots.checkout / run_dir).resolve()
        )
    return resolved


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    roots = resolve_workspace_roots_from_args(args)

    representations = [
        item.strip() for item in args.representations.split(",") if item.strip()
    ]
    explicit_run_dirs = (
        [item.strip() for item in args.run_dirs.split(",") if item.strip()]
        if args.run_dirs
        else None
    )
    if explicit_run_dirs is not None and len(explicit_run_dirs) != len(representations):
        parser.error(
            "--run-dirs must name exactly one directory per --representations entry."
        )

    run_dirs = _resolve_run_dirs(
        roots,
        representations=representations,
        explicit_run_dirs=explicit_run_dirs,
        source_system=args.source_system,
        target_system=args.target_system,
    )
    output_path = resolve_output_path(
        roots, args.run_date, args.source_system, args.target_system
    )

    if args.dry_run:
        report_dry_run(
            "analyze_residual_pairs",
            representations=representations,
            run_dirs={
                representation: run_dir
                for representation, run_dir in zip(representations, run_dirs)
            },
        )
        report_output_plan(
            "[dry-run] analyze_residual_pairs:",
            [PlannedOutput(output_path, OUTPUT_CLEAR)],
        )
        return 0

    series: list[tuple[str, pl.DataFrame]] = []
    for representation, run_dir in zip(representations, run_dirs):
        if run_dir is None:
            print(f"[warn] no run found for representation={representation}, skipping")
            continue
        frame = build_residual_picture_frame(run_dir, country=args.country)
        if frame.height == 0:
            print(
                f"[warn] representation={representation} run={run_dir} has no "
                "never-level truth pairs, skipping"
            )
            continue
        # A country stands in for the series label when only one representation
        # is being drawn (the Decision's overlay convention).
        label = (
            representation
            if len(representations) > 1
            else (args.country or representation)
        )
        series.append((label, frame))
        print(f"representation={representation} run={run_dir} rows={frame.height:,}")

    if not series:
        print("No runs with never-level truth pairs found; nothing to draw.")
        return 1

    render_residual_pictures(
        series,
        output_path,
        suptitle=f"{args.source_system} -> {args.target_system}"
        + (f" ({args.country})" if args.country else ""),
    )
    print(f"Pictures: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
