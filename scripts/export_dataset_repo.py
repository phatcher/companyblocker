"""Copy the latest canonical, matched and prepare outputs into a dataset repo.

The layout written is the one the dataset card documents: each system's latest
canonical snapshot under `data/canonical/<system>/`, its matched layer under
`data/matched/<system>/`, and Wikidata's raw extract under
`data/prepare/wikidata/`. A system's destination is cleared before it is
copied, so a file the pipeline no longer writes does not survive there.

Nothing is committed or uploaded; the copy is all this does.

Usage:
    .venv/Scripts/python.exe scripts/export_dataset_repo.py --dest <repo dir>
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
import polars as pl
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

from workspace.data_layout import system_layer_dir
from workspace.roots import WorkspaceRoots

DEFAULT_SYSTEMS: tuple[str, ...] = (
    "fr",
    "gb",
    "gleif",
    "ie",
    "offeneregister",
    "wikidata",
)

# The dataset repository's own top-level directory, the layout its card
# documents. It is that repository's, not this workspace's `data/` root.
DATASET_REPO_DATA_DIR = "data"

# What a canonical snapshot contributes, by the path it takes in the dataset
# repo. `primary/` and `names/` are directories; the rest are file globs.
CANONICAL_DIRS: tuple[str, ...] = ("primary", "names")
CANONICAL_FILES: tuple[str, ...] = (
    "sample.parquet",
    "_dedupe_report.json",
    "*-duplicates-*.parquet",
)


@dataclass(frozen=True, slots=True)
class CopyPlan:
    """One source directory and where it lands in the dataset repo."""

    label: str
    source: Path
    destination: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy the latest canonical, matched and prepare outputs into a "
            "dataset repository, in the layout its card documents."
        ),
    )
    add_workspace_roots_args(parser)
    add_external_destination_arg(
        parser,
        "--dest",
        required=True,
        help_text="The dataset repository's own directory.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=list(DEFAULT_SYSTEMS),
        help="The systems to copy. Default: %(default)s.",
    )
    parser.add_argument(
        "--sample-format",
        nargs="*",
        choices=("jsonl", "csv"),
        default=["jsonl"],
        help="Text copies of each sample.parquet written beside it, for "
        "reading without parquet tooling. Pass none for parquet only. "
        "Default: %(default)s.",
    )
    add_dry_run_arg(parser)
    return parser


def latest_snapshot_dir(layer_dir: Path) -> Path | None:
    """The layer's latest dated snapshot, or `None` when it holds none."""
    dated = [
        directory
        for directory in sorted(layer_dir.glob("*"))
        if directory.is_dir() and not directory.name.startswith("_")
    ]
    return dated[-1] if dated else None


def plan_system(roots: WorkspaceRoots, *, system: str, dest: Path) -> list[CopyPlan]:
    """Every copy one system contributes, in the order they are made."""
    plans: list[CopyPlan] = []
    dest_data = dest / DATASET_REPO_DATA_DIR
    canonical = latest_snapshot_dir(system_layer_dir(roots, system, layer="canonical"))
    if canonical is not None:
        plans.append(
            CopyPlan(
                label=f"{system} canonical {canonical.name}",
                source=canonical,
                destination=dest_data / "canonical" / system,
            )
        )
    # `system_layer_dir` resolves the matched layer to its own `current`
    # directory, so nothing is appended here.
    matched = system_layer_dir(roots, system, layer="matched")
    if matched.is_dir():
        plans.append(
            CopyPlan(
                label=f"{system} matched",
                source=matched,
                destination=dest_data / "matched" / system,
            )
        )
    prepare = latest_snapshot_dir(system_layer_dir(roots, system, layer="prepare"))
    if prepare is not None and any(prepare.glob("*.jsonl")):
        plans.append(
            CopyPlan(
                label=f"{system} prepare {prepare.name}",
                source=prepare,
                destination=dest_data / "prepare" / system,
            )
        )
    return plans


def write_sample_text(sample: Path, *, formats: tuple[str, ...]) -> list[Path]:
    """`sample.parquet` written again beside itself as text, so it reads
    without parquet tooling.

    JSON Lines holds the list columns (`alternative_names`, `previous_names`)
    as lists. CSV has no list type, so those columns are written as JSON
    strings; every other column is the value parquet holds.
    """
    if not sample.is_file() or not formats:
        return []
    frame = pl.read_parquet(sample)
    written: list[Path] = []
    if "jsonl" in formats:
        path = sample.with_suffix(".jsonl")
        frame.write_ndjson(path)
        written.append(path)
    if "csv" in formats:
        path = sample.with_suffix(".csv")
        flattened = frame.with_columns(
            pl.col(column).cast(pl.List(pl.Utf8)).list.join("|")
            for column, dtype in frame.schema.items()
            if isinstance(dtype, pl.List)
        )
        flattened.write_csv(path)
        written.append(path)
    return written


def copy_canonical(plan: CopyPlan) -> tuple[int, int]:
    """A canonical snapshot's own files, leaving the pipeline's staging and
    working directories behind. Returns files copied and bytes."""
    files = 0
    written = 0
    for name in CANONICAL_DIRS:
        source = plan.source / name
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            target = plan.destination / name / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files += 1
            written += path.stat().st_size
    for pattern in CANONICAL_FILES:
        for path in sorted(plan.source.glob(pattern)):
            if not path.is_file():
                continue
            target = plan.destination / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            files += 1
            written += path.stat().st_size
    return files, written


def copy_tree(plan: CopyPlan) -> tuple[int, int]:
    """Every file under `plan.source`, as it stands."""
    files = 0
    written = 0
    for path in sorted(plan.source.rglob("*")):
        if not path.is_file():
            continue
        target = plan.destination / path.relative_to(plan.source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files += 1
        written += path.stat().st_size
    return files, written


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)
    dest = args.dest.resolve()
    if not dest.is_dir():
        print(f"[export] error: {dest} is not a directory")
        return 1

    plans = [
        plan
        for system in args.systems
        for plan in plan_system(roots, system=system, dest=dest)
    ]
    if not plans:
        print("[export] error: no system has a layer to copy")
        return 1

    if args.dry_run:
        report_dry_run("export_dataset_repo", dest=str(dest), systems=args.systems)
        report_output_plan(
            "[dry-run]  ",
            [PlannedOutput(plan.destination, OUTPUT_CLEAR) for plan in plans],
        )
        return 0

    total_files = 0
    total_bytes = 0
    for plan in plans:
        shutil.rmtree(plan.destination, ignore_errors=True)
        plan.destination.mkdir(parents=True, exist_ok=True)
        copier = copy_canonical if "canonical" in plan.label else copy_tree
        files, written = copier(plan)
        for path in write_sample_text(
            plan.destination / "sample.parquet",
            formats=tuple(args.sample_format or ()),
        ):
            files += 1
            written += path.stat().st_size
        total_files += files
        total_bytes += written
        print(
            f"[export] {plan.label}: {files} file(s), "
            f"{written / 2**20:,.0f} MiB -> {plan.destination}"
        )
    print(f"[export] {total_files} file(s), {total_bytes / 2**30:,.1f} GiB into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
