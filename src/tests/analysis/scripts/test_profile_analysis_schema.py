from __future__ import annotations

import sys

from scripts import profile_analysis_schema
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_resolved_output_paths_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "profile_analysis_schema.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--date",
            "2026-09-16",
            "--systems",
            "gb,ie",
            "--dry-run",
        ],
    )

    exit_code = profile_analysis_schema.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(
        workspace_roots, "token_performance", "2026-09-16"
    )
    out = capsys.readouterr().out
    assert "systems=['gb', 'ie']" in out
    assert "would clear" in out
    assert "would extend" in out
    assert "schema_profile.parquet" in out
    assert "schema_profile.md" in out
    assert "run_manifest.json" in out
    assert not run_root.exists()
