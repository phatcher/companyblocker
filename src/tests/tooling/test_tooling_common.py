"""`_tooling_common.py`'s own contract: which directory a tool takes as its checkout.

Every tool under `tooling/` resolves its root through this one function, so its rule is
asserted here directly instead of through whichever tool happens to be under test.
"""

from __future__ import annotations

from pathlib import Path

import _tooling_common
import pytest
from _tooling_common import repository_root


def test_repository_root_is_the_nearest_ancestor_holding_the_marker(
    tmp_path: Path,
) -> None:
    """Nearest wins, so a tool inside a worktree nested in a checkout resolves the
    worktree, and depth below the marker does not change the answer."""
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    worktree = tmp_path / ".worktrees" / "AAA-01"
    deep = worktree / "tooling" / "nested"
    deep.mkdir(parents=True)
    (worktree / "pyproject.toml").write_text("", encoding="utf-8")

    assert repository_root(deep / "tool.py") == worktree.resolve()
    assert repository_root(tmp_path / "tooling" / "tool.py") == tmp_path.resolve()


def test_repository_root_raises_when_no_ancestor_holds_the_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrong root is worse than none: the tool would report on an arbitrary tree."""
    monkeypatch.setattr(_tooling_common, "ROOT_MARKER_NAME", "no-such-marker.toml")
    with pytest.raises(FileNotFoundError, match="no-such-marker.toml"):
        repository_root(tmp_path / "tool.py")


def test_this_checkout_resolves_to_the_directory_holding_tooling() -> None:
    """The real case every tool relies on: from a file in `tooling/`, the root is its parent."""
    tool = Path(_tooling_common.__file__)
    assert repository_root(tool) == tool.resolve().parents[1]
