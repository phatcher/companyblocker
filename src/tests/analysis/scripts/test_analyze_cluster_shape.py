from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

from blocking.run_layout import resolve_run_location
from scripts import analyze_cluster_shape
from workspace.artifact_layout import analysis_report_run_dir
from workspace.identity import RunKeys
from workspace.roots import default_workspace_roots


def _run_dir(tmp_path: Path) -> Path:
    """A run directory where blocking files one, since the script reads the
    pairing back from the tree's own levels."""
    return resolve_run_location(
        default_workspace_roots(tmp_path),
        source_system="gleif",
        target_system="gb",
        representation="tfidf",
        keys=RunKeys(
            settings="s",
            source_population="p",
            target_population="q",
            truth=None,
            index="i",
        ),
    ).directory


def _expected_output(tmp_path: Path, run_dir: Path) -> Path:
    """The analysis run's picture of this run, not a file inside the run."""
    run = analysis_report_run_dir(
        default_workspace_roots(tmp_path), "cluster_shape", "2026-09-29"
    )
    return run / "viz" / f"gleif__gb_tfidf_{run_dir.name}_cluster_shape.png"


def _write_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    # Before the union: one pair (s1/t1) and one singleton (t2, no source
    # ever reached it). After: the target-neighbour edge links t1/t2 into
    # one cluster of three.
    pl.DataFrame(
        {
            "cluster_id": ["c1", "c1"],
            "node_id": ["s1", "t1"],
            "node_name": ["Acme Systems", "Acme Sys"],
            "node_role": ["source", "target"],
        }
    ).write_parquet(run_dir / "clusters_before_union.parquet")
    pl.DataFrame(
        {
            "cluster_id": ["c1", "c1", "c1"],
            "node_id": ["s1", "t1", "t2"],
            "node_name": ["Acme Systems", "Acme Sys", "Acme Sys Ltd"],
            "node_role": ["source", "target", "target"],
        }
    ).write_parquet(run_dir / "clusters.parquet")


def test_main_returns_error_code_when_run_dir_missing_clusters(
    tmp_path: Path, monkeypatch
):
    run_dir = _run_dir(tmp_path)
    run_dir.mkdir(parents=True)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_cluster_shape.py",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--run-dir",
            str(run_dir),
        ],
    )

    exit_code = analyze_cluster_shape.main()

    assert exit_code == 1


def test_dry_run_reports_the_resolved_output_path_and_writes_nothing(
    tmp_path: Path, monkeypatch, capsys
):
    run_dir = _run_dir(tmp_path)
    _write_run(run_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_cluster_shape.py",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-09-29",
            "--dry-run",
        ],
    )

    exit_code = analyze_cluster_shape.main()

    assert exit_code == 0
    output_path = _expected_output(tmp_path, run_dir)
    out = capsys.readouterr().out
    assert str(output_path) in out
    assert not output_path.exists()


@pytest.mark.graphics
def test_main_runs_end_to_end_and_writes_the_picture(tmp_path: Path, monkeypatch):
    """Asserts the drawn picture itself, not just the entry point's wiring --
    a dry run does not reach the render, so this stays a real invocation."""
    run_dir = _run_dir(tmp_path)
    _write_run(run_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_cluster_shape.py",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--run-dir",
            str(run_dir),
            "--date",
            "2026-09-29",
        ],
    )

    exit_code = analyze_cluster_shape.main()

    assert exit_code == 0
    output_path = _expected_output(tmp_path, run_dir)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
