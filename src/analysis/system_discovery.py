"""Which systems have tokenized parquet output, for schema profiling and token metrics alike.

Wraps `acquisition.parquet_discovery`. Auto-discovery drops a system whose catalog status is outside `acquisition.constants_status.RUNNABLE_SYSTEM_STATUSES` (`filter_runnable_systems`, `is_system_runnable`); a system named explicitly resolves whatever its status.
"""

from __future__ import annotations

from collections.abc import Iterable

from acquisition.constants_status import RUNNABLE_SYSTEM_STATUSES
from acquisition.plan_registry import get_system_plan
from workspace.data_layout import (
    TOKENIZED_LAYER_NAME,
    data_root,
    system_layer_dir,
)
from workspace.layer_layout import resolve_primary_files
from workspace.parquet_discovery import discover_child_dirs_with_parquet
from workspace.roots import WorkspaceRoots


def is_system_runnable(system: str) -> bool:
    """Whether `system`'s catalog status permits it into auto-discovery.

    A system with no catalog entry at all is treated as runnable -- this
    only excludes a system the catalog explicitly knows about and has
    classified outside `RUNNABLE_SYSTEM_STATUSES` (e.g. `stopped`), not one
    the catalog has simply never heard of (a test fixture system, say).
    Naming a system explicitly always resolves regardless of this check;
    it only gates auto-discovery.
    """
    try:
        plan = get_system_plan(system)
    except KeyError:
        return True
    return plan.status in RUNNABLE_SYSTEM_STATUSES


def filter_runnable_systems(systems: Iterable[str]) -> list[str]:
    """Drop systems whose catalog status is not in `RUNNABLE_SYSTEM_STATUSES`.

    Applied to auto-discovery results only -- a system named explicitly by a
    caller bypasses this filter entirely and still resolves, catalog status
    notwithstanding.
    """
    return [system for system in systems if is_system_runnable(system)]


def discover_systems_with_tokenized_parquet(
    *,
    roots: WorkspaceRoots,
    input_file: str | None = None,
) -> list[str]:
    """Takes the project roots, not a pre-composed `data/` path: every location
    below comes from `workspace.data_layout` rather than being assembled here,
    so this area states the layout in no place of its own."""
    data_root_dir = data_root(roots)
    if input_file is not None:
        # A scoped/sample run names its exact output file, so an existence
        # check for that one file at tokenized_dir's own top level is
        # correct regardless of whether the layer as a whole is partitioned.
        discovered = [
            system_dir.name
            for system_dir in discover_child_dirs_with_parquet(
                parent_dir=data_root_dir,
                subdir_name=TOKENIZED_LAYER_NAME,
                required_file=input_file,
            )
        ]
        return filter_runnable_systems(discovered)

    # The general case has to be layer-shape-aware: a real tokenized/ layer
    # mirrors cleansed/'s shape and is partitioned under
    # jurisdiction_code=*/ for most systems, which discover_child_dirs_with_
    # parquet's own flat "*.parquet" top-level check cannot see.
    discovered = [
        system_dir.name
        for system_dir in sorted(data_root_dir.iterdir())
        if system_dir.is_dir()
        and resolve_primary_files(
            system_layer_dir(roots, system_dir.name, layer=TOKENIZED_LAYER_NAME),
            system_code=system_dir.name,
        )
    ]
    return filter_runnable_systems(discovered)
