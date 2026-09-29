"""Direct tests for `workspace.cleanse_inputs`.

`resolve_input_dir`'s own contract: which canonical run gets picked (latest
by directory-sort order, or the exact `run_date` named), and the two loud
failures -- no canonical run at all, and a named `run_date` with no entity
files in it. Only `Path.mkdir`/`Path.write_text` under `tmp_path`; no script
invocation and no real cleanse run, since none of that is this module's own
decision. Every fixture path is composed with `layer_fixture_dir`
(`src/tests/conftest.py`) rather than restated as a literal, so this file
writes the same expression `resolve_input_dir` itself does.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from workspace.cleanse_inputs import resolve_input_dir
from workspace.data_layout import CANONICAL_LAYER_NAME
from workspace.roots import WorkspaceRoots


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "uses_latest_canonical_when_run_date_not_set",
            "system": "gb",
            "run_date": None,
            "create_snapshot_dates": ["2026-06-14", "2026-06-15"],
            "expected_snapshot_date": "2026-06-15",
            "raises": None,
        },
        {
            "name": "errors_when_no_canonical_exists",
            "system": "ie",
            "run_date": None,
            "create_system_dir_only": True,
            "raises": FileNotFoundError,
        },
        {
            "name": "requested_run_date_requires_existing_snapshot",
            "system": "fr",
            "run_date": "2026-06-15",
            "raises": FileNotFoundError,
        },
        {
            "name": "uses_requested_run_date",
            "system": "gb",
            "run_date": "2026-06-15",
            "create_snapshot_dates": ["2026-06-15"],
            "expected_snapshot_date": "2026-06-15",
            "raises": None,
        },
    ],
    ids=lambda case: case["name"],
)
def test_resolve_input_dir_cases(
    workspace_roots: WorkspaceRoots, layer_fixture_dir: Callable[..., Path], case
):
    canonical_root = layer_fixture_dir(case["system"], layer=CANONICAL_LAYER_NAME)
    if case.get("create_system_dir_only"):
        canonical_root.parent.mkdir(parents=True, exist_ok=True)
    for snapshot_date in case.get("create_snapshot_dates", []):
        snapshot_dir = canonical_root / snapshot_date
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / f"{case['system']}-001.parquet").write_text(
            "x", encoding="utf-8"
        )

    expected_exception = case.get("raises")
    if expected_exception is not None:
        with pytest.raises(expected_exception):
            resolve_input_dir(
                roots=workspace_roots,
                run_date=case["run_date"],
                system=case["system"],
            )
        return

    selected = resolve_input_dir(
        roots=workspace_roots,
        run_date=case["run_date"],
        system=case["system"],
    )
    assert selected == canonical_root / case["expected_snapshot_date"]


def test_resolve_input_dir_errors_when_canonical_root_has_no_run_dirs(
    workspace_roots: WorkspaceRoots, layer_fixture_dir: Callable[..., Path]
):
    layer_fixture_dir("gb", layer=CANONICAL_LAYER_NAME).mkdir(
        parents=True, exist_ok=True
    )

    with pytest.raises(FileNotFoundError):
        resolve_input_dir(
            roots=workspace_roots,
            run_date=None,
            system="gb",
        )


def test_resolve_input_dir_errors_when_system_is_blank(workspace_roots: WorkspaceRoots):
    with pytest.raises(ValueError, match="System must be provided"):
        resolve_input_dir(
            roots=workspace_roots,
            run_date=None,
            system="   ",
        )
