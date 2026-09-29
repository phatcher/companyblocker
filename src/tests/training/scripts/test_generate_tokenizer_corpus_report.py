from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest
from company_tokenize import (
    compute_corpus_content_hash,
    resolve_optimize_canary_pointer_path,
    resolve_optimize_dir,
    resolve_optimize_sweep_paths,
)

from scripts import generate_tokenizer_corpus_report as generate_report_script
from tests.promoted_tokenizers import promoted_tokenizer_files
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.published_reports import tokenizer_reports_dir
from workspace.roots import WorkspaceRoots


def _promote_one(
    roots: WorkspaceRoots, *, system: str = "ie", profile: str = "promoted"
) -> None:
    """Lay out a stored candidate holding metrics and point `profile` at it."""
    files = promoted_tokenizer_files(roots, system=system, profile=profile)
    files.metrics.write_text(json.dumps({"fertility_distance": 0.01}), encoding="utf-8")


def _write_training_corpus(roots: WorkspaceRoots, *, system: str = "ie") -> Path:
    corpus_path = tokenizer_scope_dir(roots, system=system) / "training_corpus.parquet"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"{system}:{i}" for i in range(43)],
            "name": ["acme systems ltd", "beta holdings ltd", "gamma trading co"] * 14
            + ["delta group"],
        }
    ).write_parquet(corpus_path)
    return corpus_path


def _run(monkeypatch, tmp_path: Path, *arguments: str) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_tokenizer_corpus_report.py", "--root", str(tmp_path), *arguments],
    )
    return generate_report_script.main()


def _published_report(roots: WorkspaceRoots, system: str) -> Path:
    return tokenizer_reports_dir(roots, scope=system) / "report.md"


def test_main_publishes_the_promoted_candidates_report(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    _promote_one(workspace_roots)

    assert (
        _run(monkeypatch, tmp_path, "--system", "ie", "--tokenizer", "wordpiece") == 0
    )

    published = _published_report(workspace_roots, "ie").read_text(encoding="utf-8")
    assert "- Candidate: `tokenizer://ie/wordpiece/fixture`" in published


def test_main_errors_actionably_when_nothing_is_promoted(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    _promote_one(workspace_roots, profile="naive")

    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, "--system", "ie")

    assert not tokenizer_reports_dir(workspace_roots).exists()


def test_main_builds_without_publishing_with_the_off_switch(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    _promote_one(workspace_roots)

    assert _run(monkeypatch, tmp_path, "--system", "ie", "--no-publish") == 0

    assert not tokenizer_reports_dir(workspace_roots).exists()


def _write_run_log(optimize_dir: Path) -> None:
    rows = [
        {
            "min_frequency": 1,
            "seed": 1,
            "vocab_size_requested": vocab,
            "vocab_size_resolved": vocab,
            "fertility_distance": distance,
            "rejected": False,
        }
        for vocab, distance in ((32, 0.08), (64, 0.02))
    ]
    optimize_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(optimize_dir / "optimize_runs.parquet")


@pytest.mark.graphics
def test_main_renders_a_fresh_elbow_curve_before_building_the_report(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    # A real, complete run log can sit in the optimize run's folder and still
    # never reach the report as a curve if nobody ran plot_optimize_elbow.py
    # first, so the script renders it itself.
    _promote_one(workspace_roots)
    corpus_path = _write_training_corpus(workspace_roots)
    optimize_dir = resolve_optimize_dir(
        scope_directory=tokenizer_scope_dir(workspace_roots, system="ie"),
        trainer="wordpiece",
        tokenizer_encoding=None,
    )
    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps({"grid_hash": "testgrid"}), encoding="utf-8")
    run_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=compute_corpus_content_hash(corpus_path),
        grid_hash="testgrid",
    )
    _write_run_log(run_paths.sweep_dir)
    assert not (run_paths.sweep_dir / "elbow_curve.wordpiece.png").exists()

    assert _run(monkeypatch, tmp_path, "--system", "ie") == 0

    assert (run_paths.sweep_dir / "elbow_curve.wordpiece.png").exists()
    report_dir = tokenizer_reports_dir(workspace_roots, scope="ie")
    assert "elbow.wordpiece.png" in (report_dir / "report.md").read_text(
        encoding="utf-8"
    )
    assert (report_dir / "elbow.wordpiece.png").exists()


def test_main_system_all_reports_every_system_and_skips_one_with_nothing_promoted(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    _promote_one(workspace_roots, system="ie")
    _promote_one(workspace_roots, system="gb")
    _promote_one(workspace_roots, system="fr", profile="naive")

    assert _run(monkeypatch, tmp_path, "--system", "all") == 0

    assert _published_report(workspace_roots, "ie").exists()
    assert _published_report(workspace_roots, "gb").exists()
    assert not _published_report(workspace_roots, "fr").exists()
    assert "fr: skipped" in capsys.readouterr().out


def test_main_system_all_with_nothing_stored_reports_nothing_without_erroring(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    assert _run(monkeypatch, tmp_path, "--system", "all") == 0
    assert not tokenizer_reports_dir(workspace_roots).exists()
