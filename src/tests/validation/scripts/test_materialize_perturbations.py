"""Tests for `scripts/materialize_perturbations.py`'s argument resolution.

Scoped to the entry point's own contract: where the run's seed comes from, and
that the output it would write is named for the seed it resolved. The dry run
reaches the call the script exists to make and stops there, so none of this
touches a source system or writes a dataset.

What a resolved seed then means is `validation.perturbation_profiles`' own
contract and is asserted in `test_perturbation_profiles.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import materialize_perturbations
from validation.perturbation_profiles import PROFILE_FILENAME
from workspace.kind_layout import Kind
from workspace.reference import Side, locate, reference
from workspace.roots import WorkspaceRoots

AUTHORED_SEED = 20260910


def _author(roots: WorkspaceRoots, profile_id: str = "light-noise", **extra) -> None:
    directory = locate(
        roots,
        reference(Kind.PERTURBATION, Side.PROFILE, name=profile_id, version="v1"),
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PROFILE_FILENAME).write_text(
        json.dumps(
            {
                "profile_id": profile_id,
                "scenarios": [
                    {
                        "scenario_id": "one-typo",
                        "chain": [{"operator_id": "typo.keyboard_substitution"}],
                    }
                ],
                **extra,
            }
        ),
        encoding="utf-8",
    )


def _dry_run(root: Path, *extra: str) -> int:
    args = materialize_perturbations.build_parser().parse_args(
        [
            "--perturb",
            "light-noise",
            "--source",
            "gb",
            "--root",
            str(root),
            "--dry-run",
            *extra,
        ]
    )
    return materialize_perturbations.run_materialization(args)


def test_a_profile_that_names_a_seed_needs_no_seed_argument(
    tmp_path, workspace_roots, capsys
):
    _author(workspace_roots, default_seed=AUTHORED_SEED)

    assert _dry_run(tmp_path) == 0
    assert f"seed={AUTHORED_SEED!r}" in capsys.readouterr().out


def test_a_seed_argument_overrides_the_profiles_own(tmp_path, workspace_roots, capsys):
    _author(workspace_roots, default_seed=AUTHORED_SEED)

    _dry_run(tmp_path, "--seed", "7")

    assert "seed=7" in capsys.readouterr().out


def test_the_report_says_which_of_the_two_the_seed_came_from(
    tmp_path, workspace_roots, capsys
):
    """A command that names no seed still leaves a record of where its seed was found."""
    _author(workspace_roots, default_seed=AUTHORED_SEED)

    _dry_run(tmp_path)

    assert "seed_from='profile'" in capsys.readouterr().out


def test_a_seed_argument_is_reported_as_the_argument_it_was(
    tmp_path, workspace_roots, capsys
):
    _author(workspace_roots, default_seed=AUTHORED_SEED)

    _dry_run(tmp_path, "--seed", "7")

    assert "seed_from='argument'" in capsys.readouterr().out


def test_the_output_directory_is_named_for_the_seed_that_was_resolved(
    tmp_path, workspace_roots, capsys
):
    """The dataset a run would write is the one the resolved seed identifies."""
    _author(workspace_roots, default_seed=AUTHORED_SEED)

    _dry_run(tmp_path)

    assert str(AUTHORED_SEED) in capsys.readouterr().out


def test_a_profile_naming_no_seed_and_no_seed_argument_is_refused(
    tmp_path, workspace_roots
):
    _author(workspace_roots)

    with pytest.raises(materialize_perturbations.SeedNotResolvedError):
        _dry_run(tmp_path)


def test_a_profile_naming_no_seed_still_runs_with_a_seed_argument(
    tmp_path, workspace_roots, capsys
):
    _author(workspace_roots)

    assert _dry_run(tmp_path, "--seed", "7") == 0
    assert "seed=7" in capsys.readouterr().out


def test_the_report_names_the_profile_reference_it_resolved(
    tmp_path, workspace_roots, capsys
):
    _author(workspace_roots, default_seed=AUTHORED_SEED)

    _dry_run(tmp_path)

    assert "profile='perturbation://light-noise/v1'" in capsys.readouterr().out


def test_a_dry_run_writes_no_dataset(tmp_path, workspace_roots):
    _author(workspace_roots, default_seed=AUTHORED_SEED)

    _dry_run(tmp_path)

    assert not (tmp_path / "data").exists()
