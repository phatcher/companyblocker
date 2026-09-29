from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

from blocking.run_layout import RunArtefact, resolve_run_location
from scripts import analyze_residual_pairs
from tests.production_records import write_record
from workspace.artifact_layout import analysis_report_run_dir
from workspace.identity import RunKeys
from workspace.roots import WorkspaceRoots


def _run_dir(roots: WorkspaceRoots, settings: str) -> Path:
    """A gleif -> gb tfidf run directory where blocking files one, told apart
    from another run by its settings key."""
    return resolve_run_location(
        roots,
        source_system="gleif",
        target_system="gb",
        representation="tfidf",
        keys=RunKeys(
            settings=settings,
            source_population="p",
            target_population="q",
            truth=None,
            index="i",
        ),
    ).directory


pytestmark = pytest.mark.integration

_RESIDUAL_PAIRS_COLUMNS = [
    "source_system",
    "target_system",
    "country",
    "source_id",
    "target_id",
    "source_name",
    "source_name_cleansed",
    "target_name",
    "target_name_cleansed",
    "name_equality",
    "is_truth_pair",
    "found",
    "similarity",
    "rank",
]


def _write_run(
    roots: WorkspaceRoots,
    run_dir: Path,
    *,
    residual_pairs_name: str = "residual_pairs.parquet",
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            "gleif",
            "gb",
            "gb",
            "s1",
            "t1",
            "Acme Systems",
            "acme systems",
            "Acme Sys",
            "acme sys",
            "never",
            True,
            True,
            0.9,
            1,
        ),
        (
            "gleif",
            "gb",
            "gb",
            "s2",
            "t2",
            "Beta Ltd",
            "beta",
            "Nothing Alike",
            "nothing alike",
            "never",
            True,
            False,
            None,
            None,
        ),
    ]
    pl.DataFrame(rows, schema=_RESIDUAL_PAIRS_COLUMNS, orient="row").write_parquet(
        run_dir / residual_pairs_name
    )
    pl.DataFrame(
        {
            "cluster_id": ["c1", "c1"],
            "node_id": ["s1", "t1"],
            "node_name": ["Acme Systems", "Acme Sys"],
            "node_role": ["source", "target"],
        }
    ).write_parquet(run_dir / RunArtefact.CLUSTERS.value)
    # A finished run, which is what discovery reports on: its production
    # record is what says the run completed.
    write_record(roots, run_dir)


def test_resolve_output_path_names_the_picture_by_pairing(
    workspace_roots: WorkspaceRoots,
):
    resolved = analyze_residual_pairs.resolve_output_path(
        workspace_roots, "2026-09-07", "gleif", "gb"
    )

    assert resolved.parent == (
        analysis_report_run_dir(workspace_roots, "residual_pictures", "2026-09-07")
        / "viz"
    )
    assert resolved.name == "gleif__gb_residual_pictures.png"


def test_discover_latest_run_dir_finds_the_most_recently_written_run(
    workspace_roots: WorkspaceRoots,
):
    older = _run_dir(workspace_roots, "older")
    newer = _run_dir(workspace_roots, "newer")
    _write_run(workspace_roots, older)
    _write_run(workspace_roots, newer)
    import os
    import time

    time.sleep(0.05)
    os.utime(newer / "residual_pairs.parquet", None)

    found = analyze_residual_pairs.discover_latest_run_dir(
        workspace_roots,
        source_system="gleif",
        target_system="gb",
        representation="tfidf",
    )

    assert found == newer


def test_discover_latest_run_dir_returns_none_when_nothing_exists(
    workspace_roots: WorkspaceRoots,
):
    found = analyze_residual_pairs.discover_latest_run_dir(
        workspace_roots,
        source_system="gleif",
        target_system="gb",
        representation="tfidf",
    )

    assert found is None


def test_dry_run_reports_the_discovered_run_and_output_path_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    run_dir = _run_dir(workspace_roots, "run1")
    _write_run(workspace_roots, run_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_residual_pairs.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--representations",
            "tfidf",
            "--date",
            "2026-09-07",
            "--dry-run",
        ],
    )

    exit_code = analyze_residual_pairs.main()

    assert exit_code == 0
    output_path = (
        analysis_report_run_dir(workspace_roots, "residual_pictures", "2026-09-07")
        / "viz"
        / "gleif__gb_residual_pictures.png"
    )
    out = capsys.readouterr().out
    assert run_dir.name in out
    assert "would clear" in out
    assert "gleif__gb_residual_pictures.png" in out
    assert not output_path.exists()


@pytest.mark.graphics
def test_main_runs_end_to_end_and_writes_the_picture(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    """Asserts the drawn picture itself, not just the entry point's wiring --
    a dry run does not reach the render, so this stays a real invocation."""
    run_dir = _run_dir(workspace_roots, "run1")
    _write_run(workspace_roots, run_dir)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_residual_pairs.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--country",
            "gb",
            "--representations",
            "tfidf",
            "--date",
            "2026-09-07",
        ],
    )

    exit_code = analyze_residual_pairs.main()

    assert exit_code == 0
    output_path = (
        analysis_report_run_dir(workspace_roots, "residual_pictures", "2026-09-07")
        / "viz"
        / "gleif__gb_residual_pictures.png"
    )
    assert output_path.exists()
    assert output_path.stat().st_size > 0


@pytest.mark.graphics
def test_main_prefers_the_renamed_artifact_when_present(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    """Reaches the real render to prove discovery, not just wiring: a dry
    run never reads a run's own files, so it cannot tell the renamed
    artifact apart from the pre-rename one."""
    run_dir = _run_dir(workspace_roots, "run1")
    _write_run(
        workspace_roots, run_dir, residual_pairs_name="pair_truth_eval_detail.parquet"
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_residual_pairs.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--representations",
            "tfidf",
            "--date",
            "2026-09-07",
        ],
    )

    exit_code = analyze_residual_pairs.main()

    assert exit_code == 0


def test_main_returns_error_code_when_no_run_found(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_residual_pairs.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--representations",
            "tfidf",
        ],
    )

    exit_code = analyze_residual_pairs.main()

    assert exit_code == 1
