from __future__ import annotations

import bz2
import gzip
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from acquisition import sharding
from acquisition.sharding import shard_system_source
from acquisition.sharding_perf import ShardPerformance
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def test_shard_country_source_splits_france_parquet(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("fr", layer="acquire") / "2026-06-01"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    output_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"

    source_path = acquire_dir / "stockunitelegale.parquet"
    pl.DataFrame(
        {"CompanyName": ["Alpha", "Beta", "Gamma"], "value": [1, 2, 3]}
    ).write_parquet(source_path)

    written_paths = shard_system_source(
        "fr", roots=workspace_roots, run_date="2026-06-15", chunk_size=2
    )

    assert all(path.parent == output_dir for path in written_paths)
    assert [path.name for path in written_paths] == ["fr-001.parquet", "fr-002.parquet"]
    assert [pl.read_parquet(path).height for path in written_paths] == [2, 1]


def test_shard_country_source_splits_ireland_zip_csv(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    output_dir = layer_fixture_dir("ie", layer="source") / "2026-06-15"

    source_path = acquire_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,value\nDelta,4\nEcho,5\nFoxtrot,6\n",
        )

    written_paths = shard_system_source(
        "ie", roots=workspace_roots, run_date="2026-06-15", chunk_size=2
    )

    assert all(path.parent == output_dir for path in written_paths)
    assert [path.name for path in written_paths] == ["ie-001.parquet", "ie-002.parquet"]
    assert [pl.read_parquet(path).height for path in written_paths] == [2, 1]


def test_shard_country_source_emits_generic_telemetry_summary(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    source_path = acquire_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,value\nDelta,4\nEcho,5\nFoxtrot,6\n",
        )

    messages: list[str] = []
    shard_system_source(
        "ie",
        roots=workspace_roots,
        run_date="2026-06-15",
        chunk_size=2,
        progress=messages.append,
    )

    assert any("shard end-to-end throughput" in message for message in messages)
    assert any("shard phase summary" in message for message in messages)


def test_streaming_parquet_writer_records_telemetry(tmp_path: Path):
    messages: list[str] = []
    perf = ShardPerformance(prefix="ie", progress=messages.append)
    writer = sharding._StreamingParquetWriter(
        output_dir=tmp_path,
        prefix="ie",
        chunk_size=2,
        progress=messages.append,
        telemetry=perf,
    )

    # Telemetry is recorded per append() call (staging granularity), not per
    # final chunk_size-bounded output file -- two appends here record two
    # staging entries even though chunk_size then splits the combined 3 rows
    # into two *different* final files at finalize().
    writer.append(pl.DataFrame({"CompanyName": ["Alpha", "Beta"]}))
    writer.append(pl.DataFrame({"CompanyName": ["Gamma"]}))
    written_paths = writer.finalize()

    assert [path.name for path in written_paths] == ["ie-001.parquet", "ie-002.parquet"]
    assert perf.kept_rows == 3
    assert perf.write_elapsed_s > 0.0
    assert len(perf.chunk_throughputs) == 2
    assert any("chunk 1:" in message for message in messages)
    assert not (tmp_path / "chunks").exists()


def test_streaming_parquet_writer_stages_chunks_visibly_before_finalize(
    tmp_path: Path,
):
    writer = sharding._StreamingParquetWriter(
        output_dir=tmp_path,
        prefix="ie",
        chunk_size=None,
    )

    writer.append(pl.DataFrame({"CompanyName": ["Alpha"]}))

    staged = sorted((tmp_path / "chunks").glob("*.parquet"))
    assert [path.name for path in staged] == ["ie-001.parquet"]
    assert not (tmp_path / "ie-001.parquet").exists()

    written_paths = writer.finalize()

    assert [path.name for path in written_paths] == ["ie-001.parquet"]
    assert not (tmp_path / "chunks").exists()


def test_shard_system_source_cleans_stale_chunks_scratch_dir_before_run(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    acquire_dir = layer_fixture_dir("fr", layer="acquire") / "2026-06-01"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    output_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"

    source_path = acquire_dir / "stockunitelegale.parquet"
    pl.DataFrame({"CompanyName": ["Alpha"], "value": [1]}).write_parquet(source_path)

    stale_chunks_dir = output_dir / "chunks"
    stale_chunks_dir.mkdir(parents=True, exist_ok=True)
    (stale_chunks_dir / "leftover.parquet").write_bytes(b"not a real parquet file")

    written_paths = shard_system_source(
        "fr", roots=workspace_roots, run_date="2026-06-15", chunk_size=100
    )

    assert not stale_chunks_dir.exists()
    assert len(written_paths) == 1


def test_shard_country_source_keeps_gb_company_number_from_spaced_csv_header(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("gb", layer="acquire") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "BasicCompanyDataAsOneFile-2026-06-01.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "BasicCompanyDataAsOneFile-2026-06-01.csv",
            "CompanyName, CompanyNumber,CompanyCategory,CompanyStatus\nAlpha Ltd, 12345678,Private Limited Company,Active\nBeta Ltd, 87654321,Private Limited Company,Active\n",
        )

    written_paths = shard_system_source(
        "gb", roots=workspace_roots, run_date="2026-06-15", chunk_size=10
    )

    assert len(written_paths) == 1
    out = pl.read_parquet(written_paths[0])
    assert "CompanyNumber" in out.columns
    assert out["CompanyNumber"].to_list() == [" 12345678", " 87654321"]
    assert all(value.startswith("gb://") for value in out["system_uri"].to_list())


def test_shard_country_source_writes_sidecar_with_expected_prefix(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    output_dir = layer_fixture_dir("ie", layer="source") / "2026-06-15"

    source_path = acquire_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "company_name,company_num,status,custom_payload\nDelta,4,Active,foo\nEcho,5,Active,bar\n",
        )

    written_paths = shard_system_source(
        "ie",
        roots=workspace_roots,
        run_date="2026-06-15",
        chunk_size=10,
        sidecar=True,
    )

    assert [path.name for path in written_paths] == ["ie-001.parquet"]

    main_frame = pl.read_parquet(written_paths[0])
    sidecar_paths = sorted(output_dir.glob("ie-sidecar-*.parquet"))
    assert [path.name for path in sidecar_paths] == ["ie-sidecar-001.parquet"]

    sidecar_frame = pl.read_parquet(sidecar_paths[0])
    assert "company_name" in main_frame.columns
    assert "system_uri" in main_frame.columns
    assert "system_uri" in sidecar_frame.columns
    assert main_frame.columns[0] == "system_uri"
    assert sidecar_frame.columns[0] == "system_uri"
    assert all(
        value.startswith("ie://") for value in main_frame["system_uri"].to_list()
    )
    assert all("/fp/" not in value for value in main_frame["system_uri"].to_list())
    assert "company_num" in main_frame.columns
    assert "company_num" not in sidecar_frame.columns
    assert "custom_payload" in sidecar_frame.columns
    assert "custom_payload" not in main_frame.columns

    # Rejoin should be deterministic via the first-class system URI.
    rejoined = main_frame.join(sidecar_frame, on="system_uri", how="inner")
    assert rejoined.height == 2
    assert sorted(rejoined["custom_payload"].to_list()) == ["bar", "foo"]


def test_split_main_sidecar_retains_canonical_input_columns():
    frame = pl.DataFrame(
        {
            "system_uri": ["wikidata://Q1"],
            "label_en": ["Acme"],
            "official_name": [["Acme Corporation"]],
            "aliases_en": [["Acme Co"]],
            "custom_payload": ["only-sidecar"],
        }
    )

    main_frame, sidecar_frame = sharding._split_main_sidecar_frame(
        frame=frame,
        system_code="wikidata",
        system_field_candidates={"name": ["label_en"]},
        canonical_input_columns=("official_name", "aliases_en"),
        canonical_source_column_aliases=None,
    )

    assert "label_en" in main_frame.columns
    assert "official_name" in main_frame.columns
    assert "aliases_en" in main_frame.columns
    assert "custom_payload" not in main_frame.columns
    assert "custom_payload" in sidecar_frame.columns


def test_split_main_sidecar_retains_identity_name_column():
    frame = pl.DataFrame(
        {
            "system_uri": ["offeneregister://A"],
            "name": ["Example GmbH"],
            "jurisdiction_code": ["de"],
            "custom_payload": ["only-sidecar"],
        }
    )

    main_frame, sidecar_frame = sharding._split_main_sidecar_frame(
        frame=frame,
        system_code="offeneregister",
        system_field_candidates={
            "company_type": ["register_art"],
            "registered_address_in_full": ["registered_address"],
        },
        canonical_input_columns=None,
        canonical_source_column_aliases=None,
    )

    assert "name" in main_frame.columns
    assert "name" not in sidecar_frame.columns
    assert "custom_payload" not in main_frame.columns
    assert "custom_payload" in sidecar_frame.columns


def test_shard_country_source_uses_fingerprint_uri_when_identifier_missing(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    acquire_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    output_dir = layer_fixture_dir("ie", layer="source") / "2026-06-15"

    source_path = acquire_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,status,custom_payload\nDelta,Active,foo\nEcho,Active,bar\n",
        )

    written_paths = shard_system_source(
        "ie",
        roots=workspace_roots,
        run_date="2026-06-15",
        chunk_size=10,
        sidecar=True,
    )

    main_frame = pl.read_parquet(written_paths[0])
    sidecar_frame = pl.read_parquet(output_dir / "ie-sidecar-001.parquet")

    assert all(
        value.startswith("ie://fp/") for value in main_frame["system_uri"].to_list()
    )

    rejoined = main_frame.join(sidecar_frame, on="system_uri", how="inner")
    assert rejoined.height == 2


def test_shard_country_source_uses_identifier_uri_when_identifier_duplicates(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,company_num,status,custom_payload\nDelta,4,Active,foo\nEcho,4,Active,bar\n",
        )

    written_paths = shard_system_source(
        "ie",
        roots=workspace_roots,
        run_date="2026-06-15",
        chunk_size=10,
        sidecar=True,
    )

    main_frame = pl.read_parquet(written_paths[0])
    assert main_frame["system_uri"].to_list() == ["ie://4", "ie://4"]


def test_shard_country_source_uses_fingerprint_uri_when_identifier_metadata_disabled(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,company_num,status,custom_payload\nDelta,4,Active,foo\nEcho,5,Active,bar\n",
        )

    original_plan = sharding.get_system_plan("ie")
    monkeypatch.setattr(
        sharding,
        "get_system_plan",
        lambda _: replace(original_plan, system_uri_identifier_candidates=None),
    )

    written_paths = shard_system_source(
        "ie",
        roots=workspace_roots,
        run_date="2026-06-15",
        chunk_size=10,
        sidecar=True,
    )

    main_frame = pl.read_parquet(written_paths[0])
    assert all(
        value.startswith("ie://fp/") for value in main_frame["system_uri"].to_list()
    )


def test_shard_country_source_populates_system_uri_without_sidecar(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,company_num,status\nDelta,4,Active\nEcho,5,Active\n",
        )

    written_paths = shard_system_source(
        "ie",
        roots=workspace_roots,
        run_date="2026-06-15",
        chunk_size=10,
        sidecar=False,
    )

    out = pl.read_parquet(written_paths[0])
    assert "system_uri" in out.columns
    assert out.columns[0] == "system_uri"
    assert all(value.startswith("ie://") for value in out["system_uri"].to_list())


def test_system_uri_uses_source_uri_identifier_without_reformatting():
    frame = pl.DataFrame(
        {
            "dbpedia_uri": ["http://dbpedia.org/resource/Acme_Corp"],
            "CompanyName": ["Acme Corp"],
        }
    )

    out = sharding._with_system_uri(
        frame=frame,
        system_code="dbpedia",
        identifier_candidates=("dbpedia_uri",),
    )

    assert out["system_uri"].to_list() == ["http://dbpedia.org/resource/Acme_Corp"]


def test_shard_country_source_removes_stale_sidecar_shards(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    output_dir = layer_fixture_dir("ie", layer="source") / "2026-06-15"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Simulate stale primary and sidecar shards from a prior run.
    pl.DataFrame({"legacy": ["old"]}).write_parquet(output_dir / "ie-001.parquet")
    pl.DataFrame({"legacy": ["old"]}).write_parquet(
        output_dir / "ie-sidecar-001.parquet"
    )

    source_path = acquire_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,company_num,status,custom_payload\nDelta,4,Active,foo\n",
        )

    written_paths = shard_system_source(
        "ie",
        roots=workspace_roots,
        run_date="2026-06-15",
        chunk_size=10,
        sidecar=True,
    )

    assert [path.name for path in written_paths] == ["ie-001.parquet"]

    main_frame = pl.read_parquet(output_dir / "ie-001.parquet")
    sidecar_frame = pl.read_parquet(output_dir / "ie-sidecar-001.parquet")

    # If stale files were not removed, the legacy schema/value would remain.
    assert "legacy" not in main_frame.columns
    assert "legacy" not in sidecar_frame.columns


def test_shard_system_source_leaves_legacy_source_inputs_in_place(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    legacy_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    legacy_input = legacy_dir / "companies.csv.zip"
    with zipfile.ZipFile(legacy_input, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName,value\nDelta,4\nEcho,5\n",
        )

    written_paths = shard_system_source(
        "ie", roots=workspace_roots, run_date="2026-06-15", chunk_size=10
    )

    assert len(written_paths) == 1
    assert all(
        path.parent == layer_fixture_dir("ie", layer="source") / "2026-06-15"
        for path in written_paths
    )
    assert legacy_input.exists()


def test_shard_system_source_migrates_wikidata_projection_to_prepare(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    legacy_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = legacy_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]},"labels":{"en":{"value":"Acme"}}}\n'
        )

    projection = legacy_dir / "wikidata-companies.jsonl"
    projection.write_text('{"id":"Q1","label_en":"Acme"}\n', encoding="utf-8")

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"projection_engine": "python"},
    )

    assert (
        layer_fixture_dir("wikidata", layer="source")
        / "2026-06-29"
        / "wikidata-001.parquet"
    ).exists()
    prepare_dir = layer_fixture_dir("wikidata", layer="prepare") / "2026-06-29"
    prepared_file = prepare_dir / "wikidata-companies.jsonl"
    prepared_chunks = prepare_dir / "wikidata-companies.chunks"
    assert prepared_file.exists() or any(prepared_chunks.rglob("*.jsonl"))
    assert raw_dump.exists()
    assert projection.exists()


def test_shard_system_source_auto_prepares_wikidata_projection_when_missing(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    written_paths = shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"projection_engine": "python"},
    )

    assert len(written_paths) == 1
    prepared = (
        layer_fixture_dir("wikidata", layer="prepare")
        / "2026-06-29"
        / "wikidata-companies.jsonl"
    )
    prepared_chunks = (
        layer_fixture_dir("wikidata", layer="prepare")
        / "2026-06-29"
        / "wikidata-companies.chunks"
    )
    assert prepared.exists() or any(prepared_chunks.rglob("*.jsonl"))
    assert (
        written_paths[0].parent
        == layer_fixture_dir("wikidata", layer="source") / "2026-06-29"
    )
    assert pl.read_parquet(written_paths[0]).height == 1


def test_shard_system_source_honors_max_companies_in_prepare_projection(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )
        handle.write(
            '{"id":"Q2","type":"item","labels":{"en":{"value":"Beta Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    written_paths = shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"max_companies": 1, "projection_engine": "python"},
    )

    assert len(written_paths) == 1
    assert (
        written_paths[0].parent
        == layer_fixture_dir("wikidata", layer="source") / "2026-06-29"
    )
    assert pl.read_parquet(written_paths[0]).height == 1


def test_shard_system_source_resumes_existing_wikidata_prepare_when_limit_override_is_set(
    tmp_path: Path,
    layer_fixture_dir,
    monkeypatch,
    workspace_roots: WorkspaceRoots,
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    prepare_dir = layer_fixture_dir("wikidata", layer="prepare") / "2026-06-29"
    chunk_dir = prepare_dir / "wikidata-companies.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    (chunk_dir / "wikidata-companies-part-000001.jsonl").write_text(
        '{"id":"Q-old"}\n',
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="file",
        stream_chunk_bytes=None,
        stream_resume=False,
        cleanup_intermediate=True,
        projection_defaults=None,
    ):
        captured["source_path"] = source_path
        captured["destination_path"] = destination_path
        captured["max_companies"] = max_companies
        captured["max_lines"] = max_lines
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text('{"id":"Q-new"}\n', encoding="utf-8")
        return 1

    monkeypatch.setattr(
        "acquisition.sharding.extract_wikidata_company_projection_two_pass",
        fake_extract,
    )

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"max_companies": 2},
    )

    assert captured["source_path"] == raw_dump
    assert captured["destination_path"] == (prepare_dir / "wikidata-companies.jsonl")
    assert captured["max_companies"] == 2
    assert captured["max_lines"] is None
    assert chunk_dir.exists()


def test_shard_system_source_passes_max_lines_to_wikidata_prepare(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    captured: dict[str, object] = {}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="file",
        stream_chunk_bytes=None,
        stream_resume=False,
        cleanup_intermediate=True,
        projection_defaults=None,
    ):
        captured["source_path"] = source_path
        captured["destination_path"] = destination_path
        captured["max_companies"] = max_companies
        captured["max_lines"] = max_lines
        captured["phase1_wiring_mode"] = phase1_wiring_mode
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text('{"id":"Q1"}\n', encoding="utf-8")
        return 1

    monkeypatch.setattr(
        "acquisition.sharding.extract_wikidata_company_projection_two_pass",
        fake_extract,
    )

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"max_lines": 1},
    )

    assert captured["max_lines"] == 1
    assert captured["max_companies"] is None
    assert captured["phase1_wiring_mode"] == "stream"
    assert captured["source_path"] == raw_dump


def test_shard_system_source_passes_max_lines_as_source_limiter(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    captured: dict[str, object] = {}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="file",
        stream_chunk_bytes=None,
        stream_resume=False,
        cleanup_intermediate=True,
        projection_defaults=None,
    ):
        captured["source_path"] = source_path
        captured["destination_path"] = destination_path
        captured["max_companies"] = max_companies
        captured["max_lines"] = max_lines
        captured["phase1_wiring_mode"] = phase1_wiring_mode
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text('{"id":"Q1"}\n', encoding="utf-8")
        return 1

    monkeypatch.setattr(
        "acquisition.sharding.extract_wikidata_company_projection_two_pass",
        fake_extract,
    )

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"max_lines": 1},
    )

    assert captured["max_lines"] == 1
    assert captured["max_companies"] is None
    assert captured["phase1_wiring_mode"] == "stream"
    assert captured["source_path"] == raw_dump


def test_shard_system_source_force_reextracts_existing_wikidata_prepare(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    """`shard.force` must reach the wikidata prepare stage, not just the
    outer shard-stage skip decision, so a stale projected artifact can be
    forced to re-extract via `--additional-args shard.force=true` (or the
    CLI's own `--force`, wired through by `process_companies.py`)."""
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    prepare_dir = layer_fixture_dir("wikidata", layer="prepare") / "2026-06-29"
    prepare_dir.mkdir(parents=True, exist_ok=True)
    (prepare_dir / "wikidata-companies.jsonl").write_text(
        '{"id":"Q-stale"}\n', encoding="utf-8"
    )

    captured: dict[str, object] = {"called": False}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="file",
        stream_chunk_bytes=None,
        stream_resume=False,
        cleanup_intermediate=True,
        projection_defaults=None,
    ):
        captured["called"] = True
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text('{"id":"Q-fresh"}\n', encoding="utf-8")
        return 1

    monkeypatch.setattr(
        "acquisition.sharding.extract_wikidata_company_projection_two_pass",
        fake_extract,
    )

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"force": True},
    )

    assert captured["called"] is True
    assert (prepare_dir / "wikidata-companies.jsonl").read_text(
        encoding="utf-8"
    ) == '{"id":"Q-fresh"}\n'


def test_shard_system_source_ignores_prepare_mode_for_wikidata(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    captured: dict[str, object] = {}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="file",
        stream_chunk_bytes=None,
        stream_resume=False,
        cleanup_intermediate=True,
        projection_defaults=None,
    ):
        captured["source_path"] = source_path
        captured["max_companies"] = max_companies
        captured["max_lines"] = max_lines
        captured["phase1_wiring_mode"] = phase1_wiring_mode
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "acquisition.sharding.extract_wikidata_company_projection_two_pass",
        fake_extract,
    )

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"prepare_mode": "noop"},
    )

    assert captured["source_path"] == raw_dump
    assert captured["max_companies"] is None
    assert captured["max_lines"] is None
    assert captured["phase1_wiring_mode"] == "stream"


def test_shard_system_source_passes_prepare_wiring_mode_to_wikidata_prepare(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    captured: dict[str, object] = {}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="file",
        stream_chunk_bytes=None,
        stream_resume=False,
        cleanup_intermediate=True,
        projection_defaults=None,
    ):
        captured["source_path"] = source_path
        captured["phase1_wiring_mode"] = phase1_wiring_mode
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "acquisition.sharding.extract_wikidata_company_projection_two_pass",
        fake_extract,
    )

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"prepare_wiring_mode": "stream"},
    )

    assert captured["phase1_wiring_mode"] == "stream"
    assert captured["source_path"] == raw_dump


def test_shard_system_source_passes_projection_engine_override_to_wikidata_prepare(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("wikidata", layer="acquire") / "2026-06-29"
    acquire_dir.mkdir(parents=True, exist_ok=True)

    raw_dump = acquire_dir / "wikidata-all.json.gz"
    with gzip.open(raw_dump, mode="wt", encoding="utf-8") as handle:
        handle.write(
            '{"id":"Q1","type":"item","labels":{"en":{"value":"Acme Corp"}},"claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )

    captured: dict[str, object] = {}

    def fake_extract(
        source_path: Path,
        destination_path: Path,
        *,
        progress=None,
        max_companies=None,
        max_lines=None,
        phase1_wiring_mode="file",
        stream_chunk_bytes=None,
        stream_resume=False,
        cleanup_intermediate=True,
        projection_defaults=None,
    ):
        captured["projection_defaults"] = projection_defaults
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text("", encoding="utf-8")
        return 0

    monkeypatch.setattr(
        "acquisition.sharding.extract_wikidata_company_projection_two_pass",
        fake_extract,
    )

    shard_system_source(
        "wikidata",
        roots=workspace_roots,
        run_date="2026-06-29",
        chunk_size=10,
        allow_research=True,
        stage_options={"projection_engine": "wikisieve"},
    )

    assert isinstance(captured.get("projection_defaults"), dict)
    projection_defaults = cast(dict[str, object], captured["projection_defaults"])
    assert projection_defaults["engine"] == "wikisieve"


def test_shard_country_source_splits_semicolon_zip_csv(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            "CompanyName;value\nDelta;4\nEcho;5\nFoxtrot;6\n",
        )

    written_paths = shard_system_source(
        "ie", roots=workspace_roots, run_date="2026-06-15", chunk_size=2
    )

    assert all(
        path.parent == layer_fixture_dir("ie", layer="source") / "2026-06-15"
        for path in written_paths
    )
    assert [path.name for path in written_paths] == ["ie-001.parquet", "ie-002.parquet"]
    assert [pl.read_parquet(path).height for path in written_paths] == [2, 1]


def test_shard_country_source_splits_semicolon_zip_csv_with_quoted_values(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "companies.csv",
            'CompanyName;value\n"Delta; Holdings";4\nEcho;5\nFoxtrot;6\n',
        )

    written_paths = shard_system_source(
        "ie", roots=workspace_roots, run_date="2026-06-15", chunk_size=10
    )

    frame = pl.read_parquet(written_paths[0])
    assert frame.height == 3
    assert frame["CompanyName"].to_list()[0] == "Delta; Holdings"


def test_shard_country_source_handles_zip_member_with_parent_path_safely(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("ie", layer="acquire") / "2026-06-15"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "companies.csv.zip"
    with zipfile.ZipFile(source_path, "w") as archive:
        archive.writestr(
            "../companies.csv",
            "CompanyName;value\nDelta;4\nEcho;5\n",
        )

    written_paths = shard_system_source(
        "ie", roots=workspace_roots, run_date="2026-06-15", chunk_size=10
    )

    assert len(written_paths) == 1
    assert not (tmp_path / "companies.csv").exists()


def test_shard_country_source_splits_finland_jsonl(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    acquire_dir = layer_fixture_dir("fi", layer="acquire") / "2026-06-15"
    acquire_dir.mkdir(parents=True, exist_ok=True)
    output_dir = layer_fixture_dir("fi", layer="source") / "2026-06-15"

    source_path = acquire_dir / "fi-prh-companies-2026-06-15.jsonl"
    source_path.write_text(
        '{"businessId":"1234567-8","name":"Alpha Oy"}\n'
        '{"businessId":"7654321-0","name":"Beta Oy"}\n'
        '{"businessId":"1111111-1","name":"Gamma Oy"}\n',
        encoding="utf-8",
    )

    written_paths = shard_system_source(
        "fi", roots=workspace_roots, run_date="2026-06-15", chunk_size=2
    )

    assert all(path.parent == output_dir for path in written_paths)
    assert [path.name for path in written_paths] == ["fi-001.parquet", "fi-002.parquet"]
    assert [pl.read_parquet(path).height for path in written_paths] == [2, 1]


def test_shard_offeneregister_flattens_and_drops_nested_fields(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("offeneregister", layer="acquire") / "2026-06-26"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "de_companies_ocdata.jsonl.bz2"
    with bz2.open(source_path, mode="wt", encoding="utf-8") as handle:
        handle.write(
            "{"
            '"all_attributes":{"_registerArt":"HRB","_registerNummer":"12345","additional_data":{"AD":true,"CD":false,"DK":false,"HD":false,"SI":false,"UT":false,"V\\u00d6":false},"federal_state":"Hamburg","former_registrar":null,"native_company_number":"HRB 12345","registered_office":"Hamburg","registrar":"Hamburg"},'
            '"company_number":"HRB12345",'
            '"current_status":"active",'
            '"jurisdiction_code":"de",'
            '"name":"Beispiel GmbH",'
            '"officers":[{"name":"Jane Doe"}],'
            '"previous_names":[{"company_name":"Legacy GmbH"},{"company_name":"Legacy Zwei GmbH"}],'
            '"registered_address":"Musterstrasse 1",'
            '"retrieved_at":"2026-06-26",'
            '"subsequent_registrations":[{"confidence":"MEDIUM","subsequent_entity":{"entity_properties":{"company_number":"P2305_HRB217590","jurisdiction_code":"de"}}}]'
            "}\n"
        )

    written_paths = shard_system_source(
        "offeneregister",
        roots=workspace_roots,
        run_date="2026-06-26",
        chunk_size=10,
        allow_research=True,
    )

    assert [path.name for path in written_paths] == ["offeneregister-001.parquet"]
    out = pl.read_parquet(written_paths[0])
    row = out.row(0, named=True)

    assert "all_attributes" not in out.columns
    assert "officers" not in out.columns
    assert "previous_names" not in out.columns
    assert "subsequent_registrations" not in out.columns
    assert row["previous_names_list"] == ["Legacy GmbH", "Legacy Zwei GmbH"]
    assert row["subsequent_registration_identifiers"] == ["de:P2305_HRB217590"]

    assert row["company_type"] == "HRB"
    assert row["register_art"] == "HRB"
    assert row["register_number"] == "12345"
    assert row["register_flag_ad"] is True
    assert row["register_flag_vo"] is False


def test_shard_offeneregister_accepts_struct_nested_fields(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("offeneregister", layer="acquire") / "2026-06-26"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "de_companies_ocdata.jsonl.bz2"
    with bz2.open(source_path, mode="wt", encoding="utf-8") as handle:
        handle.write(
            "{"
            '"all_attributes":{"_registerArt":"HRA","_registerNummer":"987","additional_data":{"AD":false,"CD":false,"DK":false,"HD":false,"SI":false,"UT":false,"V\\u00d6":true}},'
            '"company_number":"HRA987",'
            '"current_status":"active",'
            '"jurisdiction_code":"de",'
            '"name":"Einzel Beispiel e.K.",'
            '"previous_names":{"company_name":"Altname e.K."},'
            '"subsequent_registrations":{"subsequent_entity":{"entity_properties":{"company_number":"HRB123","jurisdiction_code":"de"}}}'
            "}\n"
        )

    written_paths = shard_system_source(
        "offeneregister",
        roots=workspace_roots,
        run_date="2026-06-26",
        chunk_size=10,
        allow_research=True,
    )

    out = pl.read_parquet(written_paths[0])
    row = out.row(0, named=True)
    assert row["previous_names_list"] == ["Altname e.K."]
    assert row["subsequent_registration_identifiers"] == ["de:HRB123"]


def test_shard_offeneregister_keeps_register_number_in_main_when_sidecar_enabled(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    source_dir = layer_fixture_dir("offeneregister", layer="acquire") / "2026-06-26"
    source_dir.mkdir(parents=True, exist_ok=True)

    source_path = source_dir / "de_companies_ocdata.jsonl.bz2"
    with bz2.open(source_path, mode="wt", encoding="utf-8") as handle:
        handle.write(
            "{"
            '"all_attributes":{"_registerArt":"HRB","_registerNummer":"12345","additional_data":{"AD":true,"CD":false,"DK":false,"HD":false,"SI":false,"UT":false,"V\\u00d6":false}},'
            '"company_number":"B8535_HRB12345",'
            '"current_status":"active",'
            '"jurisdiction_code":"de",'
            '"name":"Beispiel GmbH",'
            '"registered_address":"Musterstrasse 1, 50667 Koeln"'
            "}\n"
        )

    written_paths = shard_system_source(
        "offeneregister",
        roots=workspace_roots,
        run_date="2026-06-26",
        chunk_size=10,
        allow_research=True,
        sidecar=True,
    )

    out = pl.read_parquet(written_paths[0])
    sidecar = pl.read_parquet(
        layer_fixture_dir("offeneregister", layer="source")
        / "2026-06-26"
        / "offeneregister-sidecar-001.parquet"
    )

    assert "register_art" in out.columns
    assert "register_number" in out.columns
    assert out.row(0, named=True)["register_number"] == "12345"
    assert "register_number" not in sidecar.columns


def test_shard_system_source_rejects_stopped_dbpedia_even_with_allow_research(
    tmp_path: Path,
    layer_fixture_dir,
    workspace_roots: WorkspaceRoots,
):
    """DBpedia's catalog status is `stopped` (acquisition deliberately
    discontinued, see catalog/systems/dbpedia.json), not `research_required`
    -- `allow_research` only ever unlocks `research_required`, never
    `stopped`, so this must still reject regardless. Superseded a prior
    version of this test asserting the opposite (that dbpedia sharded
    successfully under allow_research, with real RDF-triple extraction
    assertions), back when dbpedia's status was still `research_required`."""
    source_dir = layer_fixture_dir("dbpedia", layer="acquire") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="stopped"):
        shard_system_source(
            "dbpedia",
            roots=workspace_roots,
            run_date="2026-06-26",
            chunk_size=10,
            allow_research=True,
        )
