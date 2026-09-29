"""Report a layer holding two layout generations of the same partition.

`workspace.layer_layout.resolve_partition_dir` prefers a layer's `primary/`
family and falls back to the layer's own top level, so a layer written before
the family split still reads. That fallback is right for a layer with exactly
one generation on disk and silently wrong for a layer that still has both: a
caller reading through the resolver sees only `primary/`'s copy, while a
hand-built glob over the layer's top level sees the superseded one, and
nothing today reports that the two can disagree. `data/gleif/matched/` is a
real instance -- both a pre-split `jurisdiction_code=<value>/` at the layer's
own top level and a post-split one under `primary/` exist for the same
values, left behind when the family split moved the write target and nothing
removed the old shape.

This walks every layer directory under the data root (`--data-dir`) and reports one
where a partition value exists both under `primary/` and at the layer's own
top level -- the exact shape `resolve_partition_dir` silently disambiguates.
It does not read row counts or compare content: `Path.is_dir()` on both
locations is the whole test, matching what `resolve_partition_dir` itself
checks.

`src/tests/baselines/stale_layout_generations.txt` pins today's known
instances so the check can be gated in CI without failing on a leftover this
item does not delete -- deleting the stale generation under
`data/gleif/matched/` is destructive against shared storage and is the
user's own run, not this script's. Never add a line to that file to absorb
a new instance; only a floor that already exists on `main` before this
change belongs there.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import add_workspace_roots_args, resolve_workspace_roots_from_args

from workspace.data_layout import data_root
from workspace.layer_layout import PARTITION_COLUMN, PRIMARY_FAMILY_DIR_NAME

# Directories that hold no further layer beneath them, so descending into them
# cannot uncover a second generation: a partition directory's own contents are
# parquet files and nothing else, and `chunks/` is write-in-progress staging,
# never a read target `resolve_partition_dir` considers.
_PRUNED_DIR_NAMES = frozenset({"chunks"})

_PARTITION_PREFIX = f"{PARTITION_COLUMN}="


@dataclass(frozen=True)
class StaleGeneration:
    """One layer directory holding both a `primary/` and a top-level copy of
    the same partition value(s)."""

    layer_dir: str
    """POSIX path to the layer directory, relative to the scan root."""

    values: tuple[str, ...]
    """Partition values present in both generations, sorted."""


def _partition_values_at(root: Path) -> dict[str, Path]:
    """Partition value -> directory, for every `jurisdiction_code=<value>`
    child directory directly under `root`. Empty when `root` does not exist
    or holds none."""
    if not root.is_dir():
        return {}
    return {
        path.name[len(_PARTITION_PREFIX) :].strip().lower(): path
        for path in root.glob(f"{_PARTITION_PREFIX}*")
        if path.is_dir() and path.name[len(_PARTITION_PREFIX) :].strip()
    }


def discover_stale_generations(root: Path) -> list[StaleGeneration]:
    """Every layer directory under `root` holding the same partition value
    both under `primary/` and at the layer's own top level.

    Walks the whole tree rather than a hand-listed set of layers, so a layer
    this item's author never looked at is still covered.
    """
    found: list[StaleGeneration] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _PRUNED_DIR_NAMES and not name.startswith(_PARTITION_PREFIX)
        ]
        current = Path(dirpath)
        primary_values = _partition_values_at(current / PRIMARY_FAMILY_DIR_NAME)
        if not primary_values:
            continue
        top_level_values = _partition_values_at(current)
        overlap = sorted(set(primary_values) & set(top_level_values))
        if overlap:
            found.append(
                StaleGeneration(
                    layer_dir=current.relative_to(root).as_posix(),
                    values=tuple(overlap),
                )
            )
    return found


def measure(root: Path) -> dict[str, int]:
    """Layer directory (relative POSIX path) -> count of doubled partition
    values, for `baseline.check_baseline`."""
    return {
        stale.layer_dir: len(stale.values) for stale in discover_stale_generations(root)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_workspace_roots_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    layer_data_root = data_root(resolve_workspace_roots_from_args(args))

    if not layer_data_root.exists():
        print(f"No data root found at {layer_data_root}.")
        return 0

    stale = discover_stale_generations(layer_data_root)
    if not stale:
        print(
            f"No layer under {layer_data_root} holds two layout generations of a partition."
        )
        return 0

    print(
        f"Layers holding two layout generations of the same partition ({layer_data_root}):"
    )
    for entry in stale:
        print(f"  data/{entry.layer_dir}: {', '.join(entry.values)}")
    print(
        f"\nEach value above exists both under <layer>/primary/{PARTITION_COLUMN}=<value>/ "
        f"and <layer>/{PARTITION_COLUMN}=<value>/. resolve_partition_dir (workspace.layer_layout) "
        "reads only the primary/ copy; a caller that globs the layer by hand can read the "
        "superseded one instead. Deleting the superseded copy is a destructive change to "
        "shared storage and is an operator's own call, not this script's."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
