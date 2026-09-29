from __future__ import annotations

import sys
from pathlib import Path

from scripts import process_companies
from workspace.roots import WorkspaceRoots


def test_process_companies_passes_additional_args_to_shard_stage(
    tmp_path: Path, monkeypatch, mocker
):
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        return_value=[tmp_path / "dbpedia-source.parquet"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_companies.py",
            "--systems",
            "dbpedia",
            "--processes",
            "shard",
            "--root",
            str(tmp_path),
            "--additional-args",
            "shard.infer_links=1",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    shard_system_source.assert_called_once()
    assert shard_system_source.call_args.args[0] == "dbpedia"
    assert shard_system_source.call_args.kwargs.get("stage_options") == {
        "infer_links": True
    }


def test_process_companies_force_flag_reaches_shard_stage_options(
    tmp_path: Path, monkeypatch, mocker
):
    """`--force` alone (no `--additional-args shard.force=...`) must reach
    the shard stage's own `stage_options`, since that's the only path that
    lets the wikidata prepare sub-stage honour it."""
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        return_value=[tmp_path / "wikidata-source.parquet"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_companies.py",
            "--systems",
            "wikidata",
            "--processes",
            "shard",
            "--root",
            str(tmp_path),
            "--allow-research",
            "--force",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    shard_system_source.assert_called_once()
    assert shard_system_source.call_args.kwargs.get("stage_options") == {"force": True}


def test_process_companies_force_flag_merges_with_additional_args_for_shard_stage(
    tmp_path: Path, monkeypatch, mocker
):
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        return_value=[tmp_path / "dbpedia-source.parquet"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_companies.py",
            "--systems",
            "dbpedia",
            "--processes",
            "shard",
            "--root",
            str(tmp_path),
            "--additional-args",
            "shard.infer_links=1",
            "--force",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    shard_system_source.assert_called_once()
    assert shard_system_source.call_args.kwargs.get("stage_options") == {
        "infer_links": True,
        "force": True,
    }


def test_process_companies_passes_sidecar_flag_to_shard_stage(
    tmp_path: Path, monkeypatch, mocker
):
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        return_value=[tmp_path / "ie-source.parquet"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_companies.py",
            "--systems",
            "ie",
            "--processes",
            "shard",
            "--root",
            str(tmp_path),
            "--sidecar",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    shard_system_source.assert_called_once()
    assert shard_system_source.call_args.args[0] == "ie"
    assert shard_system_source.call_args.kwargs.get("sidecar") is True


def test_process_companies_passes_sidecar_false_to_shard_stage(
    tmp_path: Path, monkeypatch, mocker
):
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        return_value=[tmp_path / "ie-source.parquet"],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_companies.py",
            "--systems",
            "ie",
            "--processes",
            "shard",
            "--root",
            str(tmp_path),
            "--sidecar=false",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    shard_system_source.assert_called_once()
    assert shard_system_source.call_args.args[0] == "ie"
    assert shard_system_source.call_args.kwargs.get("sidecar") is False


def test_process_companies_passes_allow_research_to_shard_stage(tmp_path: Path, mocker):
    calls: list[tuple[str, str]] = []

    def fake_shard_system_source(system_code: str, **kwargs):
        assert kwargs["allow_research"] is True
        calls.append(("shard", system_code))
        return [tmp_path / f"{system_code}-source.parquet"]

    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        side_effect=fake_shard_system_source,
    )

    exit_code = process_companies.run_pipeline(
        systems=["dbpedia"],
        processes=["shard"],
        root=tmp_path,
        allow_research=True,
    )

    assert exit_code == 0
    assert calls == [("shard", "dbpedia")]
    shard_system_source.assert_called_once()


def test_is_stage_up_to_date_never_shortcuts_wikidata_shard(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    snapshot = "2026-07-16"
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / snapshot
    source_dir = layer_fixture_dir("wikidata", layer="source") / snapshot
    acquire_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)

    acquire_file = acquire_dir / "wikidata-all.json.gz"
    source_file = source_dir / "wikidata-001.parquet"
    acquire_file.write_bytes(b"acquire")
    source_file.write_bytes(b"source")

    # Arrange mtimes so generic freshness logic would otherwise report up-to-date.
    source_stat = source_file.stat().st_mtime
    acquire_stat = acquire_file.stat().st_mtime
    if source_stat < acquire_stat:
        source_file.touch()

    assert (
        process_companies._is_stage_up_to_date(
            stage="shard",
            system_code="wikidata",
            roots=workspace_roots,
            run_date=snapshot,
            input_file=None,
        )
        is False
    )
