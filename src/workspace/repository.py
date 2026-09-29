"""Which checkout a file sits in, the one anchor every other root here takes.

Every other module in this package takes a project root and returns a path
beneath it -- `data_layout`'s `data/`, `artifact_layout`'s `artifacts/` -- and
none of them derives that root, deliberately: an anchor is passed, so a file
cannot be broken by being moved. This module is the one legitimate producer of
it, for the single caller that has nowhere to take it from.

**Scope.** Two callers. A gate: a test or check whose subject is the tree it
is running in rather than a location inside it -- every docstring's item IDs,
every layer path literal, the direct-test ratchet. Handing such a check a root
from outside is not a simplification but a defect, since a suite running inside
a worktree would then scan the main checkout. And the roots resolver's caller
(`roots.resolve_workspace_roots`), which needs the checkout the running code
sits in as the anchor every overridable root defaults beneath. Everything else
takes the resolved `WorkspaceRoots` and never calls `repository_root`.

It also names the checkout's tracked `docs/` (`docs_dir`), from the checkout the
resolved roots already carry.
"""

from __future__ import annotations

from pathlib import Path

ROOT_MARKER_NAME = "pyproject.toml"
DOCS_DIR_NAME = "docs"


def repository_root(path_inside: Path) -> Path:
    """The checkout `path_inside` sits in: its nearest ancestor holding
    `pyproject.toml`, itself included.

    Walks to the marker rather than counting `parents[N]`, so the answer does
    not depend on how deep the caller happens to sit and moving the caller
    cannot silently change what it resolves to. Nearest wins, so a worktree
    nested inside a checkout resolves to the worktree.

    Raises `FileNotFoundError` when no ancestor holds the marker, since a wrong
    root is worse than no root: the caller would scan something, and the
    something would be arbitrary.
    """
    start = Path(path_inside).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ROOT_MARKER_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"No {ROOT_MARKER_NAME} in {start} or any ancestor, so the checkout "
        "holding it cannot be identified."
    )


def docs_dir(checkout: Path) -> Path:
    """The checkout's tracked `docs/` directory, the one place a script writes
    into the repository rather than a workspace root: a published figure or a
    generated reference page. `checkout` is `WorkspaceRoots.checkout`."""
    return Path(checkout) / DOCS_DIR_NAME
