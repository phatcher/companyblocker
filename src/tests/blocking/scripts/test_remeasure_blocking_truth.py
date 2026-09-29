from __future__ import annotations

from pathlib import Path

import pytest

from blocking.run_layout import RunArtefact, resolve_run_location
from scripts import remeasure_blocking_truth
from tests.production_records import write_record
from workspace.identity import RunKeys
from workspace.roots import WorkspaceRoots


def _roots_argv(roots: WorkspaceRoots) -> list[str]:
    return [
        "--data-dir",
        str(roots.data),
        "--output-dir",
        str(roots.artifacts),
        "--temp-dir",
        str(roots.temp),
    ]


def _write_run(
    roots: WorkspaceRoots, *, representation: str = "tfidf", settings: str = "s"
) -> Path:
    location = resolve_run_location(
        roots,
        source_system="gleif",
        target_system="gb",
        representation=representation,
        keys=RunKeys(
            settings=settings,
            source_population="p",
            target_population="q",
            truth=None,
            index="i",
        ),
    )
    location.directory.mkdir(parents=True)
    location.path(RunArtefact.SUMMARY).write_text("{}", encoding="utf-8")
    write_record(roots, location.directory)
    return location.directory


def test_dry_run_names_the_selected_run_and_what_it_would_rewrite(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _write_run(workspace_roots)
    summary_before = (run_dir / "summary.json").read_bytes()

    exit_code = remeasure_blocking_truth.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--top-k",
            "2",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert f"run_dir={run_dir!r}" in out
    assert "top_k=2" in out
    assert f"would clear {run_dir / 'pair_truth_eval_detail.parquet'}" in out
    assert f"would extend {run_dir / 'summary.json'}" in out
    assert (run_dir / "summary.json").read_bytes() == summary_before
    assert not (run_dir / "pair_truth_eval_detail.parquet").exists()


def test_a_selection_matching_several_runs_is_refused(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(workspace_roots, settings="a")
    _write_run(workspace_roots, settings="b")

    exit_code = remeasure_blocking_truth.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--dry-run",
        ]
    )

    assert exit_code == 1
    assert "2 runs match" in capsys.readouterr().err


def test_a_run_key_picks_one_run_out_of_several(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(workspace_roots, settings="a")
    chosen = _write_run(workspace_roots, settings="b")

    exit_code = remeasure_blocking_truth.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--run-key",
            chosen.name,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    assert f"run_dir={chosen!r}" in capsys.readouterr().out


def test_no_matching_run_is_reported_as_an_error(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = remeasure_blocking_truth.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
        ]
    )

    assert exit_code == 1
    assert "no finished gleif -> gb run" in capsys.readouterr().err
