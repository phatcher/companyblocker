from __future__ import annotations

import sys

import polars as pl
import pytest

from scripts import analyze_vocab_shrinkage
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def _write_zipf_run(roots: WorkspaceRoots, run_date: str) -> None:
    metrics_dir = analysis_report_run_dir(roots, "token_zipf", run_date) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for system, raw_tokens in [
        ("ie", {"the": (10, 0.5), "zzqxplkxxnope": (6, 0.3), "eire": (4, 0.2)}),
        ("gb", {"the": (90, 0.6), "acme": (50, 0.3), "qqxzznopeworld": (15, 0.1)}),
    ]:
        pl.DataFrame(
            {
                "token": list(raw_tokens.keys()),
                "document_frequency": [v[0] for v in raw_tokens.values()],
                "document_frequency_pct": [v[1] for v in raw_tokens.values()],
            }
        ).write_parquet(metrics_dir / f"{system}_raw_token_stats.parquet")
    pl.DataFrame(
        {
            "system": ["ie", "gb"],
            "tier": ["raw", "raw"],
            "total_docs": [20, 150],
        }
    ).write_parquet(metrics_dir / "token_zipf_summary.parquet")


def test_dry_run_reports_the_resolved_settings_and_output_paths_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_vocab_shrinkage.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie,gb",
            "--date",
            "2026-09-05",
            "--dry-run",
        ],
    )

    exit_code = analyze_vocab_shrinkage.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "vocab_shrinkage", "2026-09-05")
    out = capsys.readouterr().out
    assert "v_min=100" in out
    assert "systems=['ie', 'gb']" in out
    assert "would extend" in out
    assert not run_root.exists()


@pytest.mark.graphics
def test_main_runs_end_to_end_and_writes_expected_artifacts(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    """Asserts the written artifacts themselves, not just the entry point's
    wiring -- a dry run does not reach the real projection/render, so this
    stays a real invocation."""
    _write_zipf_run(workspace_roots, "2026-09-05")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_vocab_shrinkage.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie,gb",
            "--date",
            "2026-09-05",
            "--budgets",
            "1,2,3",
        ],
    )

    exit_code = analyze_vocab_shrinkage.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "vocab_shrinkage", "2026-09-05")
    assert (run_root / "metrics" / "dropped_mass_curve.parquet").exists()
    assert (run_root / "metrics" / "top_v_membership.parquet").exists()
    assert (run_root / "viz" / "_dropped_mass_count.png").exists()
    assert (run_root / "viz" / "_dropped_mass_equal.png").exists()


def test_main_requires_at_least_one_system(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_vocab_shrinkage.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "",
        ],
    )

    with pytest.raises(SystemExit):
        analyze_vocab_shrinkage.main()


def test_main_rejects_malformed_budgets_value(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    _write_zipf_run(workspace_roots, "2026-09-05")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_vocab_shrinkage.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie",
            "--date",
            "2026-09-05",
            "--budgets",
            "",
        ],
    )

    with pytest.raises(SystemExit):
        analyze_vocab_shrinkage.main()
