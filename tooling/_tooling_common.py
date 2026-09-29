"""What every tool here shares, kept in `tooling/` so no tool imports the code it polices.

`tooling/` is deliberately separate from `src/`: a gate or a planning tool has to run
whatever state the areas are in, so finding its own checkout must not depend on them.
`workspace.repository.repository_root` answers the same question for code under `src/`;
this is the tools' own copy of that one rule, not a second design.
"""

from __future__ import annotations

from pathlib import Path

ROOT_MARKER_NAME = "pyproject.toml"


def repository_root(path_inside: Path) -> Path:
    """The checkout `path_inside` sits in: its nearest ancestor holding `pyproject.toml`.

    Walks to the marker instead of counting `parents[N]`, so a tool that moves deeper
    still resolves the same root, and a tool run from a worktree resolves the worktree,
    not the main checkout. `tooling/` has no `pyproject.toml` of its own, so the nearest
    one is the repository's.

    Raises `FileNotFoundError` when no ancestor holds the marker: a tool that scanned an
    arbitrary directory would report on it as though it were the repository.
    """
    start = Path(path_inside).resolve()
    for candidate in (start, *start.parents):
        if (candidate / ROOT_MARKER_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        f"No {ROOT_MARKER_NAME} in {start} or any ancestor, so the checkout "
        "holding it cannot be identified."
    )
