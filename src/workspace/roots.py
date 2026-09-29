"""Where a run reads and writes: five roots, resolved once, and passed to every
layout function in this package in place of a project root.

One `--root` used to anchor both `data/` and `artifacts/` at whatever directory
a script was launched from, so a run's results landed wherever the shell
happened to be, nothing could move data and artifacts separately, and a
relocated run was invisible to every reader still resolving from the project
root. This module replaces that single anchor with a resolved value:

- **checkout**: the repository the code sits in, from the code's own location
  (`repository.repository_root`), never overridable.
- **config**: `<checkout>/config`, never overridable, so a run can never read
  configuration from a different checkout than its code.
- **data**, **artifacts**, **temp**: overridable by flag, then by environment
  variable, then defaulting to `<checkout>/data`, `<checkout>/artifacts` and
  `<checkout>/tmp`. A relative value resolves against the checkout, never the
  working directory.

`resolve_workspace_roots` is the one place the `CBLOCKER_*` variables are read,
and it takes the environment as an argument rather than reaching for
`os.environ`, so a test can hand it a mapping and the suite can clear the real
variables once for every test. Each root remembers where its value came from,
and `WorkspaceRoots.to_manifest` renders that for a run manifest, so a result
can say which roots produced it.

The temp root holds disposable scratch only. Staging for an atomic write, a
layer's `__writing__` directory or a chunk's `.tmp` file, stays beside its
destination, because a rename across filesystems is either refused or a
non-atomic copy; nothing here changes that.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

CONFIG_DIR_NAME = "config"

DATA_DIR_NAME = "data"

ARTIFACT_DIR_NAME = "artifacts"

TEMP_DIR_NAME = "tmp"

ENV_DATA_DIR = "CBLOCKER_DATA_DIR"

ENV_OUTPUT_DIR = "CBLOCKER_OUTPUT_DIR"

ENV_TEMP_DIR = "CBLOCKER_TEMP_DIR"

ENVIRONMENT_VARIABLES = (ENV_DATA_DIR, ENV_OUTPUT_DIR, ENV_TEMP_DIR)
"""Every variable the resolver reads, for the test suite to clear."""


class RootSource(str, Enum):
    """Where a root's value came from, recorded beside the path."""

    CHECKOUT = "checkout"
    """Fixed by the code's location: the checkout itself and `config/`."""

    FLAG = "flag"

    ENVIRONMENT = "environment"

    DEFAULT = "default"


class RootOverlapError(ValueError):
    """Two roots that would make one path mean two things."""


@dataclass(frozen=True)
class WorkspaceRoots:
    """The five resolved roots, absolute, with the source of each."""

    checkout: Path
    config: Path
    data: Path
    artifacts: Path
    temp: Path
    sources: Mapping[str, RootSource]

    def to_manifest(self) -> dict[str, dict[str, str]]:
        """Each root as `{"path": ..., "source": ...}`, for a run manifest."""
        return {
            name: {"path": str(getattr(self, name)), "source": self.sources[name].value}
            for name in ("checkout", "config", "data", "artifacts", "temp")
        }


def resolve_workspace_roots(
    checkout: Path,
    *,
    data_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    temp_dir: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str],
) -> WorkspaceRoots:
    """Resolve the roots for one run: flag, then environment, then default.

    `checkout` is where the code sits; `data_dir`, `output_dir` and `temp_dir`
    are the flag values, None when the flag was not given; `environ` is the
    environment to read the `CBLOCKER_*` variables from. Every returned path is
    absolute, a relative value having been resolved against the checkout.

    Raises `RootOverlapError`, naming both paths, when an overridable root lies
    inside another, inside `config/`, or equals either: a file under such a
    pair would be both a data file and an artifact, or a scratch file and a
    result, depending on who asked.
    """
    checkout_path = Path(checkout).resolve()
    config = checkout_path / CONFIG_DIR_NAME
    data, data_source = _resolve_one(
        checkout_path, data_dir, environ.get(ENV_DATA_DIR), DATA_DIR_NAME
    )
    artifacts, artifacts_source = _resolve_one(
        checkout_path, output_dir, environ.get(ENV_OUTPUT_DIR), ARTIFACT_DIR_NAME
    )
    temp, temp_source = _resolve_one(
        checkout_path, temp_dir, environ.get(ENV_TEMP_DIR), TEMP_DIR_NAME
    )
    roots = WorkspaceRoots(
        checkout=checkout_path,
        config=config,
        data=data,
        artifacts=artifacts,
        temp=temp,
        sources={
            "checkout": RootSource.CHECKOUT,
            "config": RootSource.CHECKOUT,
            "data": data_source,
            "artifacts": artifacts_source,
            "temp": temp_source,
        },
    )
    _refuse_overlaps(roots)
    return roots


def default_workspace_roots(checkout: Path) -> WorkspaceRoots:
    """The roots with no flag and no environment: every overridable root at
    its default under `checkout`. For a caller that has resolved nothing of
    its own, and for a test anchoring a fixture tree under `tmp_path`."""
    return resolve_workspace_roots(checkout, environ={})


@contextmanager
def scratch_workspace(
    roots: WorkspaceRoots, *, prefix: str
) -> Iterator[WorkspaceRoots]:
    """Roots for a corpus that stands in for the data: data and temp in a fresh
    scratch directory under `roots.temp`, artifacts the real ones, removed on exit.

    For a measurement that samples a corpus and scores it with the ordinary
    run code, so the sample is read where that code reads data and nothing it
    writes escapes the temp root except under artifacts.
    """
    roots.temp.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"{prefix}_", dir=roots.temp))
    try:
        yield resolve_workspace_roots(
            roots.checkout,
            data_dir=scratch / DATA_DIR_NAME,
            output_dir=roots.artifacts,
            temp_dir=scratch / TEMP_DIR_NAME,
            environ={},
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _resolve_one(
    checkout: Path,
    flag_value: str | os.PathLike[str] | None,
    environment_value: str | None,
    default_name: str,
) -> tuple[Path, RootSource]:
    if flag_value is not None and str(flag_value) != "":
        return (checkout / Path(flag_value)).resolve(), RootSource.FLAG
    if environment_value:
        return (checkout / Path(environment_value)).resolve(), RootSource.ENVIRONMENT
    return checkout / default_name, RootSource.DEFAULT


def _refuse_overlaps(roots: WorkspaceRoots) -> None:
    overridable = ("data", "artifacts", "temp")
    for name in overridable:
        _refuse_inside(roots, name, "config")
    for outer in overridable:
        for inner in overridable:
            if inner != outer:
                _refuse_inside(roots, inner, outer)


def _refuse_inside(roots: WorkspaceRoots, inner: str, outer: str) -> None:
    inner_path: Path = getattr(roots, inner)
    outer_path: Path = getattr(roots, outer)
    if inner_path == outer_path or outer_path in inner_path.parents:
        raise RootOverlapError(
            f"The {inner} root {inner_path} ({_describe(roots, inner)}) lies inside "
            f"the {outer} root {outer_path} ({_describe(roots, outer)}); "
            "a file there would belong to both."
        )


def _describe(roots: WorkspaceRoots, name: str) -> str:
    source = roots.sources[name]
    if source is RootSource.CHECKOUT:
        return "fixed by the checkout"
    return f"from {source.value}"
