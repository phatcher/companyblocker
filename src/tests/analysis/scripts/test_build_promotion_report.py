from __future__ import annotations

import sys

from scripts import build_promotion_report
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_resolved_paths_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_promotion_report.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--strategy",
            "tfidf",
            "--operating-point",
            "0.5",
            "--run-metrics-parquet",
            "tmp/run_metrics.parquet",
            "--source",
            "gleif",
            "--target",
            "ie",
            "--country",
            "ie",
            "--metric-bundle-json",
            "tmp/ceiling.json",
            "--promotion-metric",
            "classifier_recall_at_k",
            "--promotion-threshold",
            "0.7",
            "--date",
            "2026-09-16",
            "--dry-run",
        ],
    )

    exit_code = build_promotion_report.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "strategy='tfidf'" in out
    assert "would clear" in out
    assert "promotion_report.parquet" in out
    assert "run_metrics.parquet" in out
    assert "ceiling.json" in out
    run_root = (
        workspace_roots.artifacts
        / "analysis"
        / "promotion_report"
        / "runs"
        / "2026-09-16"
    )
    assert not run_root.exists()
