from __future__ import annotations

from pathlib import Path

import pytest

from blocking.run_layout import RunArtefact, resolve_run_location
from scripts import report_recall_curve
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
    roots: WorkspaceRoots, *, representation: str, settings: str = "s"
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


def test_dry_run_names_every_selected_run_and_the_curves_it_would_write(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    tfidf = _write_run(workspace_roots, representation="tfidf")
    sentencepiece = _write_run(workspace_roots, representation="sentencepiece")
    _write_run(workspace_roots, representation="wordpiece")

    exit_code = report_recall_curve.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--representation",
            "tfidf",
            "--representation",
            "sentencepiece",
            "--population",
            "never",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "population='never'" in out
    for run_dir in (tfidf, sentencepiece):
        assert f"would clear {run_dir / 'recall_curve.parquet'}" in out
        assert f"would clear {run_dir / 'recall_curve_summary.parquet'}" in out
        assert not (run_dir / "recall_curve.parquet").exists()
    assert "wordpiece" not in out


def test_a_selection_matching_no_run_is_reported_as_an_error(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run(workspace_roots, representation="tfidf")

    exit_code = report_recall_curve.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--run-key",
            "not-a-key",
        ]
    )

    assert exit_code == 1
    assert "no finished gleif -> gb run" in capsys.readouterr().err
