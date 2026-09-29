from __future__ import annotations

from pathlib import Path

from scripts import generate_iso20275_additions as module
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def test_dry_run_reports_the_planned_reports_without_reading_or_writing(
    tmp_path: Path, workspace_roots: WorkspaceRoots, capsys
) -> None:
    exit_code = module.main(
        [
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--temp-dir",
            str(workspace_roots.temp),
            "--countries",
            "au,us",
            "--date",
            "2026-09-29",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run] generate_iso20275_additions: would proceed with:" in out
    assert "iso20275_countries='au,us' (given)" in out
    assert "countries=['au', 'us']" in out
    run_dir = analysis_report_run_dir(
        workspace_roots, "iso20275_additions", "2026-09-29"
    )
    for subdir, name in (
        ("metrics", "iso20275_additions_by_country.json"),
        ("metrics", "iso20275_additions_by_country_abbr_only.json"),
        ("reports", "iso20275_decision_sheet_by_country.md"),
        ("reports", "iso20275_mapping_audit.md"),
    ):
        assert f"would clear {run_dir / subdir / name}" in out
    assert not run_dir.exists()


def test_dry_run_reports_every_country_when_none_are_named(
    tmp_path: Path, workspace_roots: WorkspaceRoots, capsys
) -> None:
    exit_code = module.main(
        [
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--temp-dir",
            str(workspace_roots.temp),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "iso20275_countries=None (default)" in out
    assert "countries='all'" in out


def test_display_policies_are_explicit():
    assert module.SOURCE_DISPLAY_POLICY == "preserve-punctuation"
    assert module.CANONICAL_DISPLAY_POLICY == "strip-punctuation"


def test_source_display_preserves_hyphen():
    assert (
        module._source_display_token("  Co-operative   Society ")
        == "Co-operative Society"
    )


def test_canonical_display_strips_hyphen():
    assert (
        module._canonical_display_without_punctuation("Co-operative") == "Co operative"
    )


def test_existing_source_normalization_blocks_hyphen_spacing_reproposal():
    rules = [{"source": "Co operative", "canonical": "Coop", "country": "au"}]
    iso_data = {
        "AU_COOP": {
            "status": "ACTV",
            "country_code": "au",
            "entity_legal_form_name": "Co-operative",
            "abbreviations_local": "Co-operative",
            "abbreviations_transliterated": "",
        }
    }

    additions_by_country, _ = module.build_country_data(
        rules, iso_data, include_countries={"au"}
    )

    assert additions_by_country["au"]["simple"] == []
    assert additions_by_country["au"]["decision_needed"] == []
