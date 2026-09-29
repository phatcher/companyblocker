from __future__ import annotations

import json
import os
from pathlib import Path

from blocking.run_layout import (
    ComparisonArtefact,
    resolve_comparison_location,
    resolve_pairing_dir,
)
from scripts import pipeline_status
from tests.production_records import write_record
from workspace.artifact_layout import analysis_report_run_dir
from workspace.records import RECORD_FILENAME
from workspace.roots import WorkspaceRoots


def _touch(path: Path, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _finish_run(roots: WorkspaceRoots, run_dir: Path, mtime: float) -> None:
    """The record that makes `run_dir` a finished run, dated like its files."""
    write_record(roots, run_dir)
    os.utime(run_dir / RECORD_FILENAME, (mtime, mtime))


def test_build_pipeline_status_table_flags_stage_older_than_upstream(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    base = 1_700_000_000.0
    _touch(
        layer_fixture_dir("gb", layer="acquire") / "2026-08-01" / "f.json", base + 100
    )
    # shard predates the (newer) acquire snapshot -> should be flagged stale.
    _touch(
        layer_fixture_dir("gb", layer="source") / "2026-08-01" / "gb-001.parquet", base
    )

    table = pipeline_status.build_pipeline_status_table(workspace_roots, ["gb"])
    row = table.row(0, named=True)

    assert row["system"] == "gb"
    assert "!" not in row["acquire"]
    assert row["shard"].endswith("!")
    assert row["canonical"] == "-"


def test_build_pipeline_status_table_reports_collapsed_cleansed_dir(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    # cleansed/cleansed_merge collapsed into one directory -- the
    # merged/partitioned view lives at cleansed/jurisdiction_code=*/ directly.
    base = 1_700_000_000.0
    _touch(
        layer_fixture_dir("gb", layer="cleansed")
        / "jurisdiction_code=gb"
        / "part-0.parquet",
        base,
    )

    table = pipeline_status.build_pipeline_status_table(workspace_roots, ["gb"])
    row = table.row(0, named=True)

    assert row["cleanse"] != "-"


def test_parse_scenario_name() -> None:
    assert pipeline_status._parse_scenario_name("gleif_to_gb_gb") == (
        "gleif",
        "gb",
        "gb",
    )
    assert pipeline_status._parse_scenario_name("not-a-scenario") is None


def test_build_match_analysis_status_flags_stale_run(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
) -> None:
    base = 1_700_000_000.0
    # Source system's matched output is newer than the recorded analysis run.
    _touch(
        layer_fixture_dir("gleif", layer="matched")
        / "jurisdiction_code=gb"
        / "p.parquet",
        base + 1000,
    )
    _touch(
        analysis_report_run_dir(workspace_roots, "match_analysis", "2026-07-01")
        / "gleif_to_gb_gb"
        / "metrics"
        / "summary.parquet",
        base,
    )

    statuses = pipeline_status.build_match_analysis_status(workspace_roots)

    assert len(statuses) == 1
    assert statuses[0].scenario == "gleif_to_gb_gb"
    assert statuses[0].latest_run_date == "2026-07-01"
    assert statuses[0].is_stale is True


def test_build_match_analysis_status_not_stale_when_run_is_newest(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
) -> None:
    base = 1_700_000_000.0
    _touch(
        layer_fixture_dir("gleif", layer="matched")
        / "jurisdiction_code=gb"
        / "p.parquet",
        base,
    )
    _touch(
        analysis_report_run_dir(workspace_roots, "match_analysis", "2026-07-01")
        / "gleif_to_gb_gb"
        / "metrics"
        / "summary.parquet",
        base + 1000,
    )

    statuses = pipeline_status.build_match_analysis_status(workspace_roots)

    assert statuses[0].is_stale is False


def test_build_match_analysis_status_empty_when_no_runs(
    tmp_path: Path, workspace_roots: WorkspaceRoots
) -> None:
    assert pipeline_status.build_match_analysis_status(workspace_roots) == []


def test_build_stage_refresh_command_derives_from_process_companies_choices() -> None:
    command = pipeline_status.build_stage_refresh_command(system="gb", stage="cleanse")

    assert command == (
        "uv run --no-sync python scripts/process_companies.py "
        "--systems gb --processes cleanse"
    )


def test_build_stage_refresh_command_rejects_unknown_stage() -> None:
    try:
        pipeline_status.build_stage_refresh_command(system="gb", stage="not-a-stage")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown stage")


def test_every_stage_order_entry_has_a_refresh_command() -> None:
    # STAGE_ORDER is derived from PIPELINE_STAGE_NAMES; every entry it
    # produces must also be one of process_companies.py's own --processes
    # choices, or a renamed/removed stage would silently break the refresh
    # commands this report prints.
    for stage in pipeline_status.STAGE_ORDER:
        pipeline_status.build_stage_refresh_command(system="gb", stage=stage)


def test_find_stale_stage_cells_matches_table_markers(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
) -> None:
    base = 1_700_000_000.0
    _touch(
        layer_fixture_dir("gb", layer="acquire") / "2026-08-01" / "f.json", base + 100
    )
    _touch(
        layer_fixture_dir("gb", layer="source") / "2026-08-01" / "gb-001.parquet", base
    )

    stale = pipeline_status.find_stale_stage_cells(workspace_roots, ["gb"])

    assert stale == [("gb", "shard")]


def test_build_match_analysis_refresh_command_parses_scenario_name() -> None:
    command = pipeline_status.build_match_analysis_refresh_command("gleif_to_gb_gb")

    assert command == (
        "uv run --no-sync python scripts/analyze_matches.py "
        "--source gleif --target gb --country gb"
    )


def test_build_match_analysis_refresh_command_none_for_unparseable_scenario() -> None:
    assert (
        pipeline_status.build_match_analysis_refresh_command("not-a-scenario") is None
    )


def _gleif_gb_run_dir(roots: WorkspaceRoots) -> Path:
    return (
        resolve_pairing_dir(roots, source_system="gleif", target_system="gb")
        / "tfidf"
        / "abc123def456"
    )


def test_build_blocking_run_status_flags_stale_run(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
) -> None:
    base = 1_700_000_000.0
    _touch(
        layer_fixture_dir("gleif", layer="matched")
        / "jurisdiction_code=gb"
        / "p.parquet",
        base + 1000,
    )
    run_dir = _gleif_gb_run_dir(workspace_roots)
    _touch(run_dir / "summary.json", base)
    _touch(run_dir / "clusters.parquet", base)
    _finish_run(workspace_roots, run_dir, base)

    statuses = pipeline_status.build_blocking_run_status(workspace_roots)

    assert len(statuses) == 1
    status = statuses[0]
    assert status.pair == "gb/data/gleif"
    assert status.source_system == "gleif"
    assert status.target_system == "gb"
    assert status.source_kind == "data"
    assert status.representation == "tfidf"
    assert status.run_id == "abc123def456"
    assert status.is_stale is True


def test_build_blocking_run_status_not_stale_when_run_is_newest(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
) -> None:
    base = 1_700_000_000.0
    _touch(
        layer_fixture_dir("gleif", layer="matched")
        / "jurisdiction_code=gb"
        / "p.parquet",
        base,
    )
    run_dir = _gleif_gb_run_dir(workspace_roots)
    _touch(run_dir / "summary.json", base + 1000)
    _finish_run(workspace_roots, run_dir, base + 1000)

    statuses = pipeline_status.build_blocking_run_status(workspace_roots)

    assert statuses[0].is_stale is False


def test_build_blocking_run_status_reads_a_perturbed_source_from_the_summary(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
) -> None:
    """A perturbed source is filed under a hyphenated segment the selector
    cannot be recovered from, so the refresh command's source comes from the
    run's own `summary.json`; the segment stands in only when it is silent."""
    run_dir = (
        resolve_pairing_dir(
            workspace_roots, source_system="perturbed://ie/p1/v1/1", target_system="ie"
        )
        / "tfidf"
        / "abc123def456"
    )
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text(
        json.dumps({"source_system": "perturbed://ie/p1/v1/1", "target_system": "ie"}),
        encoding="utf-8",
    )
    write_record(workspace_roots, run_dir)

    statuses = pipeline_status.build_blocking_run_status(workspace_roots)

    assert len(statuses) == 1
    assert statuses[0].pair == "ie/perturbed/ie_p1_v1_1"
    assert statuses[0].source_kind == "perturbed"
    assert statuses[0].source_system == "perturbed://ie/p1/v1/1"
    assert statuses[0].target_system == "ie"


def test_build_blocking_run_status_ignores_directories_that_are_not_runs(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
) -> None:
    # `compare_blocking_strategies.py`'s per-pairing `comparison` directory
    # holds a `strategy_comparison.parquet` but is no run's location, and must
    # not be counted as a run.
    comparison_location = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    _touch(comparison_location.path(ComparisonArtefact.REPORT), 1_700_000_000.0)

    assert pipeline_status.build_blocking_run_status(workspace_roots) == []


def test_build_blocking_run_status_empty_when_no_data_blocking(
    tmp_path: Path, workspace_roots: WorkspaceRoots
) -> None:
    assert pipeline_status.build_blocking_run_status(workspace_roots) == []


def test_build_blocking_refresh_command() -> None:
    command = pipeline_status.build_blocking_refresh_command(
        source_system="gleif", target_system="gb", representation="tfidf"
    )

    assert command == (
        "uv run --no-sync python scripts/run_blocking.py "
        "--source gleif --target gb --representation tfidf"
    )
