from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import analyze_matches
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_resolved_scenario_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_matches.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--date",
            "2026-09-16",
            "--source",
            "gleif",
            "--target",
            "gb",
            "--max-rows",
            "500",
            "--dry-run",
        ],
    )

    exit_code = analyze_matches.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "match_analysis", "2026-09-16")
    scenario_dir = run_root / "gleif_to_gb_gb"
    out = capsys.readouterr().out
    assert "max_rows=500" in out
    assert "would clear" in out
    assert "metrics" in out
    assert not scenario_dir.exists()


def test_dry_run_covers_every_scenario_in_a_scenarios_file(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    scenarios_path = tmp_path / "scenarios.json"
    scenarios_path.write_text(
        json.dumps(
            [
                {"source_system": "gleif", "target_system": "gb"},
                {"source_system": "gleif", "target_system": "fr"},
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_matches.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--date",
            "2026-09-16",
            "--scenarios-file",
            str(scenarios_path),
            "--dry-run",
        ],
    )

    exit_code = analyze_matches.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "analyze_matches[gleif_to_gb_gb]" in out
    assert "analyze_matches[gleif_to_fr_fr]" in out


def test_main_requires_source_and_target_system_without_a_scenarios_file(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_matches.py", "--output-dir", str(workspace_roots.artifacts)],
    )

    with pytest.raises(SystemExit):
        analyze_matches.main()
