from __future__ import annotations

import sys

from scripts import analyze_tokens
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_resolved_settings_and_output_paths_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_tokens.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--date",
            "2026-09-16",
            "--systems",
            "gb,ie",
            "--dry-run",
        ],
    )

    exit_code = analyze_tokens.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(
        workspace_roots, "token_performance", "2026-09-16"
    )
    out = capsys.readouterr().out
    assert "top_n=50" in out
    assert "engine='duckdb'" in out
    assert "systems=['gb', 'ie']" in out
    assert "would clear" in out
    assert "would extend" in out
    assert "run_manifest.json" in out
    assert not run_root.exists()
