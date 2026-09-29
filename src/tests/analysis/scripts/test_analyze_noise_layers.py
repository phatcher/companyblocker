from __future__ import annotations

import sys

import polars as pl
import pytest

from scripts import analyze_noise_layers
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def _write_zipf_run(roots: WorkspaceRoots, run_date: str) -> None:
    metrics_dir = analysis_report_run_dir(roots, "token_zipf", run_date) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for system, raw_tokens, cleansed_tokens in [
        ("ie", {"ltd": (10, 1.0), "acme": (5, 0.5)}, {"acme": (5, 0.5)}),
        ("gb", {"ltd": (8, 0.8), "acme": (3, 0.3)}, {"acme": (3, 0.3)}),
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
                "token": list(cleansed_tokens.keys()),
                "document_frequency": [v[0] for v in cleansed_tokens.values()],
                "document_frequency_pct": [v[1] for v in cleansed_tokens.values()],
            }
        ).write_parquet(metrics_dir / f"{system}_cleansed_token_stats.parquet")
    pl.DataFrame(
        {
            "system": ["ie", "ie", "gb", "gb"],
            "tier": ["raw", "cleansed", "raw", "cleansed"],
            "total_docs": [10, 10, 10, 10],
        }
    ).write_parquet(metrics_dir / "token_zipf_summary.parquet")


def test_dry_run_reports_the_resolved_settings_and_output_paths_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    _write_zipf_run(workspace_roots, "2026-09-05")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_noise_layers.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie,gb",
            "--date",
            "2026-09-05",
            "--dry-run",
        ],
    )

    exit_code = analyze_noise_layers.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "noise_layers", "2026-09-05")
    out = capsys.readouterr().out
    assert "cleanse_tier='cleansed'" in out
    assert "systems=['ie', 'gb']" in out
    assert "would extend" in out
    assert "metrics" in out
    assert "viz" in out
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
            "analyze_noise_layers.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie,gb",
            "--date",
            "2026-09-05",
        ],
    )

    exit_code = analyze_noise_layers.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "noise_layers", "2026-09-05")
    assert (run_root / "metrics" / "projected_tokens.parquet").exists()
    assert (run_root / "metrics" / "membership.parquet").exists()
    assert (run_root / "viz" / "_remaining_tokens_by_layer.png").exists()
    assert (run_root / "viz" / "_removed_mass_by_layer.png").exists()


def test_main_requires_at_least_one_system(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_noise_layers.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "",
        ],
    )

    with pytest.raises(SystemExit):
        analyze_noise_layers.main()


def test_main_rejects_a_malformed_extra_noise_words_value(
    workspace_roots: WorkspaceRoots, monkeypatch
):
    _write_zipf_run(workspace_roots, "2026-09-05")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_noise_layers.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--systems",
            "ie",
            "--date",
            "2026-09-05",
            "--extra-noise-words",
            "not-a-key-value-pair",
        ],
    )

    with pytest.raises(SystemExit):
        analyze_noise_layers.main()
