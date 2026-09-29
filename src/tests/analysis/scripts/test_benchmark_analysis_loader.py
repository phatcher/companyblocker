from __future__ import annotations

import sys

from scripts import benchmark_analysis_loader
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_resolved_plan_and_writes_nothing(
    tmp_path, workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    output_path = tmp_path / "summary.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_analysis_loader.py",
            "--data-dir",
            str(workspace_roots.data),
            "--systems",
            "gb,ie",
            "--modes",
            "full_read",
            "--output-parquet",
            str(output_path),
            "--dry-run",
        ],
    )

    exit_code = benchmark_analysis_loader.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "systems=['gb', 'ie']" in out
    assert "modes=['full_read']" in out
    assert "would clear" in out
    assert "summary.parquet" in out
    assert not output_path.exists()
