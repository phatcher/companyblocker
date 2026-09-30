"""Copy each country's report figures from the artefact tree into
`docs/reports/<country>/`, under fixed names, drawing nothing.

Two families are copied, each from where its own script writes it:

* A pairing's residual-pair figures, the newest dated run's copy
  (`plot_pair_outcomes.py`), as `<source>_residual_<figure>.png`, the folder
  named by the pairing's target.
* A tokenizer optimize run's elbow curve (`plot_optimize_elbow.py`), as
  `tokenizer_elbow_<trainer>.png`, the folder named by the system whose corpus
  trained it; the newest curve per trainer when a system holds several runs.

A figure not yet drawn is named with the command that draws it, and nothing
is copied in its place.

Usage:
    .venv/Scripts/python.exe scripts/publish_reports.py [--country ie]
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_dry_run_arg,
    add_external_destination_arg,
    add_workspace_roots_args,
    report_dry_run,
    report_output_plan,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)
from plot_pair_outcomes import (
    CLUSTER_SIZES,
    FOUND_BY_UPSET,
    NAME_SIMILARITY,
    RUN_OVERLAP,
    comparison_pairing,
    latest_output,
)

from blocking.run_layout import ComparisonArtefact, iter_blocking_comparisons
from workspace.artifact_layout import tokenizer_artifact_root
from workspace.repository import docs_dir
from workspace.roots import WorkspaceRoots

REPORTS_DIR_NAME = "reports"
RESIDUAL_LEVEL = "never"
RESIDUAL_FIGURES = (FOUND_BY_UPSET, RUN_OVERLAP, NAME_SIMILARITY, CLUSTER_SIZES)
TOKENIZER_TRAINERS = ("wordpiece", "sentencepiece_bpe", "sentencepiece_unigram")
ELBOW_GLOB = "elbow_curve.*.png"


@dataclass(frozen=True)
class ReportFigure:
    """One figure to copy: where it is drawn, where it goes, and the command
    that draws it when it is not there yet."""

    country: str
    name: str
    source: Path | None
    draw_command: str

    def drawn_source(self) -> Path:
        """The drawn file, for a figure already filtered to the drawn set."""
        if self.source is None:
            raise ValueError(f"{self.country}/{self.name} has not been drawn")
        return self.source

    def destination(self, reports_dir: Path) -> Path:
        return reports_dir / self.country / self.name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy each country's report figures from the artefact tree into "
            "docs/reports/<country>/, drawing nothing."
        ),
    )
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--country",
        default=None,
        help="Copy only this country's figures, named as its folder is (ie, gb, "
        "offeneregister); omit for every country with a figure.",
    )
    add_external_destination_arg(
        parser,
        "--reports-dir",
        help_text="Where the per-country folders are written (default: docs/reports).",
    )
    add_dry_run_arg(parser)
    return parser


def residual_figures(roots: WorkspaceRoots) -> list[ReportFigure]:
    """Every pairing's residual-pair figures, drawn or not."""
    figures = []
    for location in iter_blocking_comparisons(roots):
        if not location.path(ComparisonArtefact.PAIR_OUTCOMES).is_file():
            continue
        pairing = comparison_pairing(location)
        source = pairing.source_segment
        target = pairing.target_system
        command = (
            "uv run python scripts/plot_pair_outcomes.py "
            f"--source {source} --target {target}"
        )
        for figure in RESIDUAL_FIGURES:
            figures.append(
                ReportFigure(
                    country=target,
                    name=f"{source}_residual_{figure}",
                    source=latest_output(roots, pairing, RESIDUAL_LEVEL, figure),
                    draw_command=command,
                )
            )
    return figures


def _elbow_command(system: str, trainer: str) -> str:
    backend, _, encoding = trainer.partition("_")
    command = (
        "uv run python scripts/plot_optimize_elbow.py "
        f"--systems {system} --tokenizer {backend}"
    )
    return f"{command} --encoding {encoding}" if encoding else command


def elbow_figures(roots: WorkspaceRoots) -> list[ReportFigure]:
    """Every system's tokenizer elbow curves, one per trainer that has an
    optimize run, the newest curve where a trainer has several."""
    work = tokenizer_artifact_root(roots)
    if not work.is_dir():
        return []
    figures = []
    for system_dir in sorted(path for path in work.iterdir() if path.is_dir()):
        for trainer in TOKENIZER_TRAINERS:
            optimize = system_dir / trainer / "optimize"
            if not optimize.is_dir():
                continue
            curves = sorted(
                optimize.rglob(ELBOW_GLOB), key=lambda path: path.stat().st_mtime
            )
            figures.append(
                ReportFigure(
                    country=system_dir.name,
                    name=f"tokenizer_elbow_{trainer}.png",
                    source=curves[-1] if curves else None,
                    draw_command=_elbow_command(system_dir.name, trainer),
                )
            )
    return figures


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    reports_dir = args.reports_dir or docs_dir(roots.checkout) / REPORTS_DIR_NAME
    figures = [
        figure
        for figure in (*residual_figures(roots), *elbow_figures(roots))
        if args.country is None or figure.country == args.country
    ]
    drawn = [figure for figure in figures if figure.source is not None]
    missing = [figure for figure in figures if figure.source is None]

    if args.dry_run:
        report_dry_run(
            "publish_reports",
            countries=sorted({figure.country for figure in figures}),
            reports_dir=reports_dir,
        )
        report_output_plan(
            "[dry-run]  ",
            [
                PlannedOutput(figure.destination(reports_dir), OUTPUT_CLEAR)
                for figure in drawn
            ],
        )
        for figure in missing:
            print(f"[dry-run]  not drawn: {figure.country}/{figure.name}")
        return 0

    for figure in drawn:
        destination = figure.destination(reports_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure.drawn_source(), destination)
        print(f"[reports] {destination}")
    for command in dict.fromkeys(figure.draw_command for figure in missing):
        names = ", ".join(
            f"{figure.country}/{figure.name}"
            for figure in missing
            if figure.draw_command == command
        )
        print(f"[reports] not drawn: {names}\n[reports]   draw with: {command}")
    if not drawn:
        print("[reports] error: no figure to copy")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
