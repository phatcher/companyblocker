from __future__ import annotations

import sys

import polars as pl
import pytest

from scripts import analyze_token_zipf
from workspace.artifact_layout import analysis_report_run_dir
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def test_dry_run_reports_the_resolved_settings_and_output_paths_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_token_zipf.py",
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie",
            "--date",
            "2026-08-26",
            "--wordfreq-top-n",
            "200",
            "--dry-run",
        ],
    )

    exit_code = analyze_token_zipf.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "token_zipf", "2026-08-26")
    out = capsys.readouterr().out
    assert "wordfreq_top_n=200" in out
    assert "systems=['ie']" in out
    assert "would clear" in out
    assert "would extend" in out
    assert not run_root.exists()


@pytest.mark.graphics
def test_main_runs_end_to_end_and_writes_expected_artifacts(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    """Asserts the written artifacts themselves, not just the entry point's
    wiring -- a dry run does not reach the real fit/render, so this stays a
    real invocation."""
    cleansed_dir = system_layer_dir(workspace_roots, "ie", layer=CLEANSED_LAYER_NAME)
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "name": ["I B M", "Acme Systems Ltd", "Beta Holdings Ltd"],
            "name_cleansed_basic": ["ibm", "acme systems ltd", "beta holdings ltd"],
            "name_cleansed": ["ibm", "acme systems", "beta holdings"],
        }
    ).write_parquet(cleansed_dir / "ie-001.parquet")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_token_zipf.py",
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie",
            "--date",
            "2026-08-26",
            "--wordfreq-top-n",
            "200",
        ],
    )

    exit_code = analyze_token_zipf.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "token_zipf", "2026-08-26")
    assert (run_root / "reports" / "token_zipf_summary.md").exists()
    assert (run_root / "metrics" / "token_zipf_summary.parquet").exists()
    assert (run_root / "viz" / "ie_raw_zipf.png").exists()
    assert (run_root / "viz" / "ie_basic_zipf.png").exists()
    assert (run_root / "viz" / "ie_cleansed_zipf.png").exists()
