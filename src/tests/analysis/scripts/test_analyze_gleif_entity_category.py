from __future__ import annotations

import sys

from scripts import analyze_gleif_entity_category
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_resolved_scenario_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_gleif_entity_category.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--date",
            "2026-09-16",
            "--target",
            "gb",
            "--dry-run",
        ],
    )

    exit_code = analyze_gleif_entity_category.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "match_analysis", "2026-09-16")
    scenario_dir = run_root / "gleif_to_gb_gb"
    out = capsys.readouterr().out
    assert "run_date='2026-09-16'" in out
    assert "target_system='gb'" in out
    assert "country='gb'" in out
    assert "would clear" in out
    assert "gleif_entity_category_breakdown.parquet" in out
    assert "gleif_entity_category_not_recoverable.png" in out
    assert not scenario_dir.exists()
