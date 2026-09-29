"""Direct tests for `workspace.repository`.

This is the one function in the package that derives an anchor rather than
taking one, so what it must get right is pinned here independently of any
caller: the answer does not depend on the caller's depth, a nested worktree
resolves to itself, and no marker means no answer rather than a guess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace.repository import ROOT_MARKER_NAME, repository_root


def test_repository_root_walks_to_the_marker_from_any_depth(tmp_path: Path):
    """The answer is the checkout holding the marker, however deep the caller
    sits, so moving a caller cannot change what it resolves to."""
    (tmp_path / ROOT_MARKER_NAME).write_text("", encoding="utf-8")
    deep = tmp_path / "src" / "tests" / "tooling" / "scripts"
    deep.mkdir(parents=True)

    assert repository_root(deep) == tmp_path.resolve()
    assert repository_root(tmp_path) == tmp_path.resolve()


def test_repository_root_finds_the_nearest_checkout_not_the_outermost(
    tmp_path: Path,
):
    """A worktree nested inside a checkout resolves to itself: this is the
    case that makes a gate scan its own tree rather than the main one."""
    (tmp_path / ROOT_MARKER_NAME).write_text("", encoding="utf-8")
    worktree = tmp_path / ".worktrees" / "ITEM-01"
    (worktree / "src").mkdir(parents=True)
    (worktree / ROOT_MARKER_NAME).write_text("", encoding="utf-8")

    assert repository_root(worktree / "src") == worktree.resolve()


def test_repository_root_refuses_a_tree_with_no_marker(tmp_path: Path):
    """No root beats an arbitrary one: a gate handed the wrong tree would
    scan something, and the something would be silently wrong."""
    nowhere = tmp_path / "detached"
    nowhere.mkdir()

    with pytest.raises(FileNotFoundError, match=ROOT_MARKER_NAME):
        repository_root(nowhere)
