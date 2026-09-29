"""Report whether every system's canonical name layer carries per-row
identity: a `system_uri` unique to the row and a non-null `source_uri`.

`finalize_canonical_name_rows` (`acquisition.canonical`) composes both
columns as Canonical writes the names layer, and it runs for every system
with a names sidecar. But a layer written before that composition landed
(or regenerated into the wrong shape) still reports as fresh by mtime --
timestamps can't tell a wrongly-shaped layer from a correctly-shaped one,
since a wrongly-shaped regeneration is still newer than its input. This
gives conformance a repeatable command instead of a one-off query.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import add_root_arg, run_reporting_argument_errors

from acquisition.pipeline_selection import normalize_systems
from workspace.data_layout import CANONICAL_LAYER_NAME, data_root, system_layer_dir
from workspace.roots import WorkspaceRoots, default_workspace_roots

REPORT_TABLE_CONFIG: dict[str, Any] = {
    "tbl_cols": -1,
    "tbl_width_chars": 10000,
    "tbl_hide_column_data_types": True,
    "tbl_hide_dtype_separator": True,
    "tbl_hide_dataframe_shape": True,
}


@dataclass(frozen=True)
class NameLayerIdentityStatus:
    system: str
    snapshot: str | None
    row_count: int
    distinct_system_uri: int
    null_source_uri: int
    conforms: bool


def _latest_names_snapshot(roots: WorkspaceRoots, system: str) -> Path | None:
    canonical_root = system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME)
    if not canonical_root.exists():
        return None
    for candidate_dir in sorted(
        (path for path in canonical_root.glob("*") if path.is_dir()), reverse=True
    ):
        names_dir = candidate_dir / "names"
        if names_dir.exists() and any(names_dir.glob("*.parquet")):
            return names_dir
    return None


def check_system_name_layer_identity(
    roots: WorkspaceRoots, system: str
) -> NameLayerIdentityStatus:
    """Measure one system's current names layer against the per-row identity
    shape `finalize_canonical_name_rows` writes: a `system_uri` unique to
    the row and a non-null `source_uri`. Missing entirely (no names layer at
    all, e.g. a system with no sidecar) is reported as non-conforming with a
    `None` snapshot rather than skipped, so an absent layer can't be
    mistaken for a conforming one."""

    names_dir = _latest_names_snapshot(roots, system)
    if names_dir is None:
        return NameLayerIdentityStatus(
            system=system,
            snapshot=None,
            row_count=0,
            distinct_system_uri=0,
            null_source_uri=0,
            conforms=False,
        )

    frame = pl.read_parquet(names_dir / "*.parquet")
    row_count = frame.height
    distinct_system_uri = (
        frame["system_uri"].n_unique() if "system_uri" in frame.columns else 0
    )
    if "source_uri" in frame.columns:
        null_source_uri = frame["source_uri"].null_count()
    else:
        null_source_uri = row_count

    conforms = (
        row_count > 0 and distinct_system_uri == row_count and null_source_uri == 0
    )
    return NameLayerIdentityStatus(
        system=system,
        snapshot=names_dir.parent.name,
        row_count=row_count,
        distinct_system_uri=distinct_system_uri,
        null_source_uri=null_source_uri,
        conforms=conforms,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Report the per-row identity shape of every system's canonical name "
            "layer -- row count, distinct system_uri, null source_uri -- and mark "
            "each conforming or not."
        )
    )
    add_root_arg(parser, help_text="Project root containing data/.")
    parser.add_argument(
        "--systems",
        nargs="+",
        default=None,
        help=(
            "Systems to check. Space or comma separated; 'all' expands to all "
            "live systems. Defaults to every system directory found under data/."
        ),
    )
    return parser


def _discover_systems(roots: WorkspaceRoots) -> list[str]:
    root_data_dir = data_root(roots)
    if not root_data_dir.exists():
        return []
    return sorted(path.name for path in root_data_dir.iterdir() if path.is_dir())


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    systems = (
        normalize_systems(args.systems) if args.systems else _discover_systems(roots)
    )
    if not systems:
        print("No systems found under data/ and none were specified with --systems.")
        return 0

    statuses = [check_system_name_layer_identity(roots, system) for system in systems]

    print(f"Name-layer identity shape ({root})")
    print("conforms: distinct system_uri == row_count and source_uri has no nulls.")
    print()
    rows = [
        {
            "system": status.system,
            "snapshot": status.snapshot or "-",
            "row_count": status.row_count,
            "distinct_system_uri": status.distinct_system_uri,
            "null_source_uri": status.null_source_uri,
            "conforms": "yes" if status.conforms else "no",
        }
        for status in statuses
    ]
    with pl.Config(**REPORT_TABLE_CONFIG):
        print(pl.DataFrame(rows))

    return 0 if all(status.conforms for status in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
