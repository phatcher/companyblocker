from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts import process_companies
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_cleansed_parquet(roots: WorkspaceRoots, system: str) -> None:
    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"{system}:1"],
            "name": ["example ltd"],
            "name_cleansed_basic": ["example ltd"],
            "name_cleansed": ["example ltd"],
            "jurisdiction_code": [system],
        }
    ).write_parquet(cleansed_dir / f"{system}-001.parquet")


def test_process_companies_dry_run_calls_no_stage_work_functions(
    tmp_path: Path, mocker, capsys, workspace_roots: WorkspaceRoots
):
    _write_cleansed_parquet(workspace_roots, "gb")

    cleanse_canonical_view = mocker.patch.object(
        process_companies, "cleanse_canonical_view"
    )
    tokenize_name = mocker.patch.object(process_companies, "tokenize_name")
    shard_system_source = mocker.patch.object(process_companies, "shard_system_source")
    canonicalize_system_shards = mocker.patch.object(
        process_companies, "canonicalize_system_shards"
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["shard", "canonical", "cleanse", "tokenize"],
        root=tmp_path,
        dry_run=True,
    )

    assert exit_code == 0
    cleanse_canonical_view.assert_not_called()
    tokenize_name.assert_not_called()
    shard_system_source.assert_not_called()
    canonicalize_system_shards.assert_not_called()

    out = capsys.readouterr().out
    assert "[dry-run] No stage will actually execute" in out
    assert "[dry-run] [gb] Stage shard: would RUN" in out
    assert "[dry-run] [gb] Stage canonical: would RUN" in out
    assert "[dry-run] [gb] Stage cleanse:" in out
    assert "[dry-run] [gb] Stage tokenize: would RUN" in out


def test_process_companies_dry_run_writes_no_files(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    _write_cleansed_parquet(workspace_roots, "gb")
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    before_snapshot = {
        path: path.stat().st_mtime for path in cleansed_dir.rglob("*") if path.is_file()
    }

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse", "tokenize"],
        root=tmp_path,
        dry_run=True,
    )

    assert exit_code == 0
    after_snapshot = {
        path: path.stat().st_mtime for path in cleansed_dir.rglob("*") if path.is_file()
    }
    assert before_snapshot == after_snapshot
    assert not (layer_fixture_dir("gb", layer="tokenized")).exists()


def test_process_companies_dry_run_reports_up_to_date_skip_without_force(
    tmp_path: Path, layer_fixture_dir, capsys
):
    canonical_dir = layer_fixture_dir("gb", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gb:1"], "jurisdiction_code": ["gb"]}).write_parquet(
        canonical_dir / "gb-001.parquet"
    )
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gb:1"]}).write_parquet(
        cleansed_dir / "gb-001.parquet"
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse"],
        root=tmp_path,
        run_date="2026-06-15",
        dry_run=True,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run] [gb] Stage cleanse: would SKIP (output already up to date)" in out
    # Nothing should have been touched, including no chunks/ dir created.
    assert not (cleansed_dir / "chunks").exists()


def test_process_companies_dry_run_reports_up_to_date_skip_for_family_split_output(
    tmp_path: Path, layer_fixture_dir, capsys
):
    """A real cleansed/ view lives under `primary/`, not directly at
    cleansed_dir's own top level. `_is_stage_up_to_date` must resolve it
    there rather than reporting a fresh, current output as missing on every
    run.
    """
    canonical_dir = layer_fixture_dir("gb", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gb:1"], "jurisdiction_code": ["gb"]}).write_parquet(
        canonical_dir / "gb-001.parquet"
    )
    cleansed_partition_dir = (
        layer_fixture_dir("gb", layer="cleansed") / "primary" / "jurisdiction_code=gb"
    )
    cleansed_partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gb:1"]}).write_parquet(
        cleansed_partition_dir / "part-00001.parquet"
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse"],
        root=tmp_path,
        run_date="2026-06-15",
        dry_run=True,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run] [gb] Stage cleanse: would SKIP (output already up to date)" in out


def test_process_companies_dry_run_reports_flat_chunk_migration_and_force_cleanup(
    tmp_path: Path, layer_fixture_dir, capsys
):
    canonical_dir = layer_fixture_dir("gb", layer="canonical") / "2026-06-15"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gb:1"], "jurisdiction_code": ["gb"]}).write_parquet(
        canonical_dir / "gb-001.parquet"
    )
    # Flat cleansed/ files with no chunks/ directory yet.
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["gb:1"]}).write_parquet(
        cleansed_dir / "gb-001.parquet"
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse"],
        root=tmp_path,
        run_date="2026-06-15",
        force=True,
        dry_run=True,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert (
        f"would migrate {cleansed_dir} into {cleansed_dir / 'chunks'} "
        "[1 legacy flat file(s) at its top level]"
    ) in out
    assert (
        f"would clear {cleansed_dir} [merged view: 0 partition(s), 1 flat file(s)]"
        in out
    )

    # Nothing on disk should have actually moved.
    assert (cleansed_dir / "gb-001.parquet").exists()
    assert not (cleansed_dir / "chunks").exists()


def test_process_companies_dry_run_reports_wikidata_prepare_would_be_reused(
    tmp_path: Path, layer_fixture_dir, capsys
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-07-16"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    (acquire_dir / "wikidata-all.json.gz").write_bytes(b"x")

    prepare_dir = layer_fixture_dir("wikidata", layer="prepare") / "2026-07-16"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    (prepare_dir / "wikidata-companies.jsonl").write_bytes(b"{}\n" * 100)

    source_dir = layer_fixture_dir("wikidata", layer="source") / "2026-07-16"
    source_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["wikidata:1"]}).write_parquet(
        source_dir / "wikidata-001.parquet"
    )

    exit_code = process_companies.run_pipeline(
        systems=["wikidata"],
        processes=["shard"],
        root=tmp_path,
        dry_run=True,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[dry-run] [wikidata] Stage shard: would RUN" in out
    assert "wikidata prepare: found existing prepared data" in out
    assert "extraction would be SKIPPED and this reused" in out
    # dry-run must not touch anything, including the prepared JSONL.
    assert (prepare_dir / "wikidata-companies.jsonl").exists()


def test_process_companies_dry_run_reports_wikidata_prepare_force_would_reextract(
    tmp_path: Path, layer_fixture_dir, capsys
):
    """With --force, the dry-run report must say the expensive extraction
    WOULD run over the existing artifact, not that it would be skipped and
    reused -- the two are no longer the same outcome."""
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-07-16"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    (acquire_dir / "wikidata-all.json.gz").write_bytes(b"x")

    prepare_dir = layer_fixture_dir("wikidata", layer="prepare") / "2026-07-16"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    (prepare_dir / "wikidata-companies.jsonl").write_bytes(b"{}\n" * 100)

    source_dir = layer_fixture_dir("wikidata", layer="source") / "2026-07-16"
    source_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"system_uri": ["wikidata:1"]}).write_parquet(
        source_dir / "wikidata-001.parquet"
    )

    exit_code = process_companies.run_pipeline(
        systems=["wikidata"],
        processes=["shard"],
        root=tmp_path,
        force=True,
        dry_run=True,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "wikidata prepare: found existing prepared data" in out
    assert "--force is set" in out
    assert "WOULD RUN and overwrite it" in out
    assert "extraction would be SKIPPED and this reused" not in out
    # dry-run must not touch anything, including the prepared JSONL.
    assert (prepare_dir / "wikidata-companies.jsonl").exists()


def test_process_companies_dry_run_reports_wikidata_prepare_would_extract(
    tmp_path: Path, layer_fixture_dir, capsys
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-07-16"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    (acquire_dir / "wikidata-all.json.gz").write_bytes(b"x")

    exit_code = process_companies.run_pipeline(
        systems=["wikidata"],
        processes=["shard"],
        root=tmp_path,
        dry_run=True,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "no existing prepared data found" in out
    assert "WOULD trigger the expensive full-dump extraction" in out
