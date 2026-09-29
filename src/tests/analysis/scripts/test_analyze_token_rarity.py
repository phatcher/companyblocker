from __future__ import annotations

import sys

import polars as pl

from scripts import analyze_token_rarity
from workspace.artifact_layout import analysis_report_run_dir, tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def _write_token_stats(roots: WorkspaceRoots, system: str) -> None:
    target = tokenizer_scope_dir(roots, system=system)
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "token": ["the", "company", "zzqxplkxxnope"],
            "document_frequency": [5, 3, 1],
            "document_frequency_pct": [0.5, 0.3, 0.1],
            "idf": [1.0, 1.5, 3.0],
        }
    ).write_parquet(target / "token_tfidf_stats.parquet")


def test_dry_run_reports_the_resolved_settings_and_writes_nothing(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    _write_token_stats(workspace_roots, "gb")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_token_rarity.py",
            "--output-dir",
            str(workspace_roots.artifacts),
            "--date",
            "2026-01-01",
            "--systems",
            "gb",
            "--dry-run",
        ],
    )

    exit_code = analyze_token_rarity.main()

    assert exit_code == 0
    run_root = analysis_report_run_dir(workspace_roots, "token_rarity", "2026-01-01")
    out = capsys.readouterr().out
    assert "max_document_frequency=1" in out
    assert "systems=['gb']" in out
    assert "would extend" in out
    assert "would clear" in out
    assert not run_root.exists()


def test_run_analysis_produces_report_and_language_stats_for_mapped_system(
    workspace_roots: WorkspaceRoots,
):
    _write_token_stats(workspace_roots, "gb")

    report_path = analyze_token_rarity.run_analysis(
        roots=workspace_roots,
        run_date="2026-01-01",
        systems=["gb"],
        include_global=False,
        max_document_frequency=1,
        max_document_frequency_pct=None,
        min_token_length=1,
    )

    assert report_path.exists()
    report_text = report_path.read_text(encoding="utf-8")
    assert "gb" in report_text
    assert "en" in report_text

    run_root = analysis_report_run_dir(workspace_roots, "token_rarity", "2026-01-01")
    language_stats_path = run_root / "metrics" / "gb_token_language_stats.parquet"
    rare_tokens_path = run_root / "metrics" / "gb_rare_tokens.csv"
    summary_path = run_root / "metrics" / "token_rarity_summary.parquet"

    assert language_stats_path.exists()
    assert rare_tokens_path.exists()
    assert summary_path.exists()

    language_stats = pl.read_parquet(language_stats_path)
    assert "zipf_frequency" in language_stats.columns
    assert "unseen_in_wordfreq" in language_stats.columns

    rare_tokens = pl.read_csv(rare_tokens_path)
    assert rare_tokens.get_column("token").to_list() == ["zzqxplkxxnope"]


def test_run_analysis_skips_wordfreq_for_unmapped_system(
    workspace_roots: WorkspaceRoots,
):
    _write_token_stats(workspace_roots, "gleif")

    report_path = analyze_token_rarity.run_analysis(
        roots=workspace_roots,
        run_date="2026-01-01",
        systems=["gleif"],
        include_global=False,
        max_document_frequency=1,
        max_document_frequency_pct=None,
        min_token_length=1,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "gleif" in report_text
    assert "n/a" in report_text

    run_root = analysis_report_run_dir(workspace_roots, "token_rarity", "2026-01-01")
    assert not (run_root / "metrics" / "gleif_token_language_stats.parquet").exists()
    assert (run_root / "metrics" / "gleif_rare_tokens.csv").exists()


def test_run_analysis_excludes_global_unless_requested(
    workspace_roots: WorkspaceRoots,
):
    _write_token_stats(workspace_roots, "gb")
    _write_token_stats(workspace_roots, "global")

    report_path = analyze_token_rarity.run_analysis(
        roots=workspace_roots,
        run_date="2026-01-01",
        systems=None,
        include_global=False,
        max_document_frequency=1,
        max_document_frequency_pct=None,
        min_token_length=1,
    )

    report_text = report_path.read_text(encoding="utf-8")
    assert "gb" in report_text
    assert "global" not in report_text
