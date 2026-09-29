"""Pytest wiring shared by every test file under `src/tests/`.

`layer_fixture_dir` is the one helper the whole test tier composes a layout
fixture through: bound to `tmp_path`, it takes the same scope/`layer`
arguments `workspace.data_layout.system_layer_dir` does and re-roots the
result under this test's own sandbox instead of the real repository, so a
test writes the production expression and wraps it rather than restating
`"data" / <system> / <layer>` as a literal of its own. See
`src/workspace/README.md`'s layer contract for the shape it composes, and
`src/tests/pipeline/conftest.py`'s `pipeline_corpus` fixture for the same
"materialise under an anchor the test hands in" idea applied to a whole
corpus rather than one directory.

`repo_root` is its counterpart for the other kind of test: one that scans the
checkout itself rather than materialising a fixture in a sandbox.

`workspace_roots` is the resolved-roots value a test hands to any layout
function, anchored under `tmp_path` with every root at its default, and the
autouse `_clear_workspace_environment` fixture removes the resolver's
variables from every test's environment, so a developer's shell can never
point a test at shared storage.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from workspace.data_layout import system_layer_dir
from workspace.repository import repository_root
from workspace.roots import (
    ENVIRONMENT_VARIABLES,
    WorkspaceRoots,
    default_workspace_roots,
)


@pytest.fixture(autouse=True)
def _clear_workspace_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees a workspace root variable, whatever the invoking shell set."""
    for name in ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workspace_roots(tmp_path: Path) -> WorkspaceRoots:
    """Every root at its default beneath this test's own `tmp_path`: the value
    a test passes where production code passes its resolved roots."""
    return default_workspace_roots(tmp_path)


@pytest.fixture
def repo_root() -> Path:
    """The checkout these tests live in, for a test that scans the repository
    rather than materialising a fixture under `tmp_path`.

    Wraps `workspace.repository.repository_root`, which owns the walk and
    carries its own direct tests, so this fixture only supplies the starting
    point: this file's own location, never a caller's argument. A gate
    asserting something about the tree must resolve against the checkout it
    is running in, or a suite run inside a worktree scans the main checkout
    instead.

    Use `layer_fixture_dir` instead whenever the test is building its own
    fixture; a test wanting this one is the exception, not the default.
    """

    return repository_root(Path(__file__))


@pytest.fixture
def layer_fixture_dir(workspace_roots: WorkspaceRoots) -> Callable[..., Path]:
    """Call like `system_layer_dir` minus the `roots` argument, e.g.
    `layer_fixture_dir("gb", layer=CANONICAL_LAYER_NAME)`, to get a fixture's
    layer directory under this test's own `tmp_path` rather than real roots
    a test would otherwise have to invent."""

    def _compose(*scope: str, layer: str) -> Path:
        return system_layer_dir(workspace_roots, *scope, layer=layer)

    return _compose
