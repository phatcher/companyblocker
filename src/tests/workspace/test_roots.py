"""Direct tests for `workspace.roots`.

The resolver's contract is pinned here independently of any caller: the
priority order for each overridable root, relative values resolving against
the checkout and never the working directory, each refused overlap naming both
paths, and the source recorded beside every root. The last test is the gate
that keeps the environment read in one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from workspace.roots import (
    ENV_DATA_DIR,
    ENV_OUTPUT_DIR,
    ENV_TEMP_DIR,
    ENVIRONMENT_VARIABLES,
    RootOverlapError,
    RootSource,
    WorkspaceRoots,
    default_workspace_roots,
    resolve_workspace_roots,
    scratch_workspace,
)

_ROOTS = (
    ("data", "data_dir", ENV_DATA_DIR, "data"),
    ("artifacts", "output_dir", ENV_OUTPUT_DIR, "artifacts"),
    ("temp", "temp_dir", ENV_TEMP_DIR, "tmp"),
)


@pytest.mark.parametrize(("name", "flag", "variable", "default_name"), _ROOTS)
def test_flag_beats_environment_beats_default(
    tmp_path: Path, name: str, flag: str, variable: str, default_name: str
):
    """Each overridable root takes the flag first, the variable next, and the
    checkout's own directory last, with the source saying which won."""
    by_flag = tmp_path / "by-flag"
    by_env = tmp_path / "by-env"

    both = resolve_workspace_roots(
        tmp_path, **{flag: by_flag}, environ={variable: str(by_env)}
    )
    env_only = resolve_workspace_roots(tmp_path, environ={variable: str(by_env)})
    neither = resolve_workspace_roots(tmp_path, environ={})

    assert (getattr(both, name), both.sources[name]) == (
        by_flag.resolve(),
        RootSource.FLAG,
    )
    assert (getattr(env_only, name), env_only.sources[name]) == (
        by_env.resolve(),
        RootSource.ENVIRONMENT,
    )
    assert (getattr(neither, name), neither.sources[name]) == (
        tmp_path.resolve() / default_name,
        RootSource.DEFAULT,
    )


def test_relative_values_resolve_against_the_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A relative flag or variable is relative to the checkout, so where the
    shell happens to be cannot move a run's data or results."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    roots = resolve_workspace_roots(
        checkout, data_dir="runs/data", environ={ENV_OUTPUT_DIR: "runs/out"}
    )

    assert roots.data == checkout.resolve() / "runs" / "data"
    assert roots.artifacts == checkout.resolve() / "runs" / "out"


def test_checkout_and_config_are_fixed_by_the_code_location(tmp_path: Path):
    roots = resolve_workspace_roots(
        tmp_path, environ={ENV_DATA_DIR: str(tmp_path / "d")}
    )

    assert roots.checkout == tmp_path.resolve()
    assert roots.config == tmp_path.resolve() / "config"
    assert roots.sources["checkout"] is RootSource.CHECKOUT
    assert roots.sources["config"] is RootSource.CHECKOUT


def test_an_empty_flag_or_variable_counts_as_not_given(tmp_path: Path):
    roots = resolve_workspace_roots(tmp_path, data_dir="", environ={ENV_TEMP_DIR: ""})

    assert roots.sources["data"] is RootSource.DEFAULT
    assert roots.sources["temp"] is RootSource.DEFAULT


def test_default_roots_ignore_the_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`default_workspace_roots` is the no-override value, so a variable in
    the process cannot leak into a test fixture or an unmigrated caller."""
    monkeypatch.setenv(ENV_DATA_DIR, str(tmp_path / "leaked"))

    assert default_workspace_roots(tmp_path).data == tmp_path.resolve() / "data"


@pytest.mark.parametrize(
    ("kwargs", "inner", "outer"),
    [
        ({"data_dir": "artifacts/data"}, "data", "artifacts"),
        ({"output_dir": "data"}, "artifacts", "data"),
        ({"temp_dir": "data/scratch"}, "temp", "data"),
        ({"temp_dir": "artifacts"}, "temp", "artifacts"),
        ({"data_dir": "config/data"}, "data", "config"),
        ({"output_dir": "config"}, "artifacts", "config"),
    ],
)
def test_a_root_inside_another_is_refused_naming_both(
    tmp_path: Path, kwargs: dict[str, str], inner: str, outer: str
):
    """A file under two roots would belong to both, so the pair is refused
    at resolution rather than discovered by the first reader it confuses."""
    with pytest.raises(RootOverlapError) as raised:
        resolve_workspace_roots(tmp_path, environ={}, **kwargs)

    message = str(raised.value)
    assert f"The {inner} root" in message
    assert f"the {outer} root {tmp_path.resolve() / outer}" in message


def test_the_manifest_carries_every_root_with_its_source(tmp_path: Path):
    roots = resolve_workspace_roots(
        tmp_path, output_dir=tmp_path / "out", environ={ENV_TEMP_DIR: "scratch"}
    )

    manifest = roots.to_manifest()

    assert set(manifest) == {"checkout", "config", "data", "artifacts", "temp"}
    assert manifest["artifacts"] == {
        "path": str((tmp_path / "out").resolve()),
        "source": "flag",
    }
    assert manifest["temp"]["source"] == "environment"
    assert manifest["data"]["source"] == "default"
    assert manifest["config"]["source"] == "checkout"


def test_the_suite_sees_no_workspace_variable():
    """The autouse fixture in `src/tests/conftest.py` clears every variable
    the resolver reads, so a developer's shell can never point a test at
    shared storage."""
    assert not [name for name in ENVIRONMENT_VARIABLES if name in os.environ]


def test_workspace_roots_fixture_anchors_under_tmp_path(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert workspace_roots.checkout == tmp_path.resolve()
    assert workspace_roots.data == tmp_path.resolve() / "data"


_SCAN_ROOTS = ("src", "scripts", "tooling")
_RESOLVER = "src/workspace/roots.py"
_THIS_TEST = "src/tests/workspace/test_roots.py"


def test_only_the_resolver_names_a_workspace_variable(repo_root: Path):
    """The environment is read in `resolve_workspace_roots` alone. A file
    naming a `CBLOCKER_` variable elsewhere is reading it, or documenting it
    in a literal that will drift; both compose the name from `workspace.roots`
    instead."""
    offenders = []
    for scan_root in _SCAN_ROOTS:
        for path in sorted((repo_root / scan_root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(repo_root).as_posix()
            if rel in (_RESOLVER, _THIS_TEST):
                continue
            if "CBLOCKER_" in path.read_text(encoding="utf-8"):
                offenders.append(rel)
    assert offenders == []


def test_a_scratch_workspace_puts_data_and_temp_under_the_temp_root_and_removes_them(
    tmp_path: Path,
) -> None:
    roots = default_workspace_roots(tmp_path)

    with scratch_workspace(roots, prefix="corpus") as scratch:
        (scratch.data / "sample.parquet").parent.mkdir(parents=True)
        (scratch.data / "sample.parquet").write_text("", encoding="utf-8")
        scratch_dir = scratch.data.parent
        assert roots.temp in scratch.data.parents
        assert roots.temp in scratch.temp.parents
        assert scratch.artifacts == roots.artifacts

    assert not scratch_dir.exists()


def test_a_scratch_workspace_is_removed_when_the_work_fails(tmp_path: Path) -> None:
    roots = default_workspace_roots(tmp_path)

    with (
        pytest.raises(RuntimeError),
        scratch_workspace(roots, prefix="corpus") as scratch,
    ):
        scratch_dir = scratch.data.parent
        raise RuntimeError("boom")

    assert not scratch_dir.exists()
