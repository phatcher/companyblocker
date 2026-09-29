from __future__ import annotations

import csv
import gzip
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import polars as pl
import pytest

from acquisition import sharding, sharding_io
from acquisition.constants_status import STATUS_RESEARCH_REQUIRED
from acquisition.models_plan import SystemPlan
from workspace.roots import WorkspaceRoots


def test_resolve_source_artifact_raises_when_run_date_explicit_and_file_missing(
    tmp_path: Path,
    layer_fixture_dir,
):
    resource = SimpleNamespace(
        resolve_snapshot_date=lambda run_date: run_date,
        resolve_file_name=lambda run_date: f"file-{run_date}.csv",
    )

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        sharding._resolve_source_artifact(
            source_root=layer_fixture_dir("xx", layer="source"),
            resource=resource,
            effective_run_date="2026-06-20",
            run_date_provided=True,
        )


def test_resolve_source_artifact_raises_when_source_root_missing_and_run_date_implicit(
    tmp_path: Path,
    layer_fixture_dir,
):
    resource = SimpleNamespace(
        resolve_snapshot_date=lambda run_date: run_date,
        resolve_file_name=lambda run_date: f"file-{run_date}.csv",
    )

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        sharding._resolve_source_artifact(
            source_root=layer_fixture_dir("xx", layer="source"),
            resource=resource,
            effective_run_date="2026-06-20",
            run_date_provided=False,
        )


def test_resolve_source_artifact_raises_when_no_snapshot_candidates_match(
    tmp_path: Path,
    layer_fixture_dir,
):
    source_root = layer_fixture_dir("xx", layer="source")
    (source_root / "2026-06-18").mkdir(parents=True, exist_ok=True)
    resource = SimpleNamespace(
        resolve_snapshot_date=lambda run_date: run_date,
        resolve_file_name=lambda run_date: f"file-{run_date}.csv",
    )

    with pytest.raises(FileNotFoundError, match="Source file not found"):
        sharding._resolve_source_artifact(
            source_root=source_root,
            resource=resource,
            effective_run_date="2026-06-20",
            run_date_provided=False,
        )


def test_detect_csv_separator_from_text_covers_blank_and_sniffer_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    assert sharding_io.detect_csv_separator_from_text("   ") == ","

    class FailingSniffer:
        def sniff(self, sample: str, delimiters: str):
            raise csv.Error("cannot detect")

    monkeypatch.setattr(csv, "Sniffer", lambda: FailingSniffer())

    assert sharding_io.detect_csv_separator_from_text("col1|col2\nleft|right\n") == "|"
    assert sharding_io.detect_csv_separator_from_text("header\nvalue\n") == ","


def test_read_csv_bytes_with_detected_separator_reads_semicolon_payload():
    frame = sharding_io.read_csv_bytes_with_detected_separator(
        b"CompanyName;value\nAlpha;1\nBeta;2\n"
    )

    assert frame.columns == ["CompanyName", "value"]
    assert frame.height == 2


def test_read_source_frame_supports_csv_json_and_jsonl(tmp_path: Path):
    csv_path = tmp_path / "companies.csv"
    csv_path.write_text("CompanyName;value\nAlpha;1\n", encoding="utf-8")
    json_path = tmp_path / "companies.json"
    json_path.write_text('[{"CompanyName":"Alpha","value":1}]', encoding="utf-8")
    jsonl_path = tmp_path / "companies.jsonl"
    jsonl_path.write_text('{"CompanyName":"Alpha","value":1}\n', encoding="utf-8")

    assert sharding_io.read_source_frame(csv_path, "csv").height == 1
    assert sharding_io.read_source_frame(json_path, "json").height == 1
    assert sharding_io.read_source_frame(jsonl_path, "jsonl").height == 1


def test_read_source_frame_zip_rejects_empty_archive(tmp_path: Path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w"):
        pass

    with pytest.raises(ValueError, match="No files found inside archive"):
        sharding_io.read_source_frame(zip_path, "zip")


def test_read_source_frame_zip_reads_parquet_member(tmp_path: Path):
    parquet_path = tmp_path / "inner.parquet"
    pl.DataFrame({"value": [1, 2]}).write_parquet(parquet_path)
    zip_path = tmp_path / "parquet.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(parquet_path, arcname="folder/inner.parquet")

    frame = sharding_io.read_source_frame(zip_path, "zip")

    assert frame.height == 2


def test_read_source_frame_rejects_unsupported_format(tmp_path: Path):
    source_path = tmp_path / "source.txt"
    source_path.write_text("x", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported source format 'txt'"):
        sharding_io.read_source_frame(source_path, "txt")


def test_shard_system_source_rejects_non_supported_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workspace_roots: WorkspaceRoots
):
    monkeypatch.setattr(
        sharding,
        "get_system_plan",
        lambda system: SimpleNamespace(
            code=system, status="blocked", notes="No access", resources=()
        ),
    )

    with pytest.raises(RuntimeError, match="xx: blocked - No access"):
        sharding.shard_system_source("xx", roots=workspace_roots)


def test_shard_system_source_removes_stale_output_shards(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("fr", layer="source") / "2026-06-01"
    source_dir.mkdir(parents=True, exist_ok=True)
    stale_path = source_dir / "fr-999.parquet"
    stale_path.write_text("stale", encoding="utf-8")

    source_path = source_dir / "stockunitelegale.parquet"
    pl.DataFrame({"CompanyName": ["Alpha"], "value": [1]}).write_parquet(source_path)

    written_paths = sharding.shard_system_source(
        "fr", roots=workspace_roots, run_date="2026-06-15", chunk_size=10
    )

    assert [path.name for path in written_paths] == ["fr-001.parquet"]
    assert not stale_path.exists()


def test_resolve_supported_country_codes_rejects_non_tuple_policy():
    plan = SimpleNamespace(code="xx", supported_countries=["gb"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="supported_countries must be a list or null"):
        sharding._resolve_supported_country_codes(plan)


def test_resolve_supported_country_codes_rejects_local_for_non_country_registry_plan():
    plan = SimpleNamespace(code="dbpedia", supported_countries=("local",))
    with pytest.raises(ValueError, match="requires a country registry plan"):
        sharding._resolve_supported_country_codes(plan)


def test_resolve_supported_country_codes_supports_local_and_supported_selectors():
    local_plan = SimpleNamespace(code="gb", supported_countries=("local",))
    assert sharding._resolve_supported_country_codes(local_plan) == {"GB"}

    supported_plan = SimpleNamespace(
        code="dbpedia", supported_countries=("supported", "ie")
    )
    resolved = sharding._resolve_supported_country_codes(supported_plan)
    assert "GB" in resolved
    assert "IE" in resolved


def test_filter_frame_by_supported_countries_handles_missing_and_multi_columns():
    frame_without_country = pl.DataFrame({"name": ["A", "B"]})
    out = sharding._filter_frame_by_supported_countries(frame_without_country, {"GB"})
    assert out.height == 2

    frame_multi = pl.DataFrame(
        {
            "country": ["fr", "de", None],
            "CountryCode": [None, "GB", "ES"],
            "name": ["A", "B", "C"],
        }
    )
    filtered = sharding._filter_frame_by_supported_countries(frame_multi, {"FR", "GB"})
    assert filtered.height == 2
    assert filtered["name"].to_list() == ["A", "B"]


def test_detect_csv_separator_from_text_skips_leading_blank_lines():
    sample = "\n\ncol1;col2\nA;B\n"
    assert sharding_io.detect_csv_separator_from_text(sample) == ";"


def test_detect_csv_separator_from_text_returns_default_when_only_whitespace_lines():
    assert sharding_io.detect_csv_separator_from_text("\n  \n\t\n") == ","


def test_read_source_batches_jsonl_bz2_uses_eta_enabled_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_path = tmp_path / "records.jsonl.bz2"
    source_path.write_bytes(b"placeholder")

    payload = b'{"CompanyName":"Alpha","value":1}\n{"CompanyName":"Beta","value":2}\n'
    captured: dict[str, object] = {}

    class _FakeHandle:
        def __init__(self, data: bytes) -> None:
            self._buffer = io.BytesIO(data)

        def __enter__(self):
            return self._buffer

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._buffer.close()
            return False

    def fake_open_eta_enabled_source(
        path: Path,
        *,
        reader_mode: str | None = None,
        indexed_bzip2_parallelization: object | None = None,
    ):
        captured["path"] = path
        captured["reader_mode"] = reader_mode
        captured["indexed_bzip2_parallelization"] = indexed_bzip2_parallelization
        return _FakeHandle(payload)

    monkeypatch.setattr(
        sharding_io, "_open_eta_enabled_source", fake_open_eta_enabled_source
    )

    batches = list(sharding_io.read_source_batches(source_path, "jsonl", batch_size=1))

    assert captured["path"] == source_path
    assert [batch.height for batch in batches] == [1, 1]


def test_read_source_batches_jsonl_bz2_honours_chunked_reader_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_path = tmp_path / "records.jsonl.bz2"
    source_path.write_bytes(b"placeholder")

    payload = b'{"CompanyName":"Alpha","value":1}\n{"CompanyName":"Beta","value":2}\n'
    captured: dict[str, object] = {}

    class _FakeHandle:
        def __init__(self, data: bytes) -> None:
            self._buffer = io.BytesIO(data)

        def __enter__(self):
            return self._buffer

        def __exit__(self, exc_type, exc_val, exc_tb):
            self._buffer.close()
            return False

    def fake_open_eta_enabled_source(
        path: Path,
        *,
        reader_mode: str | None = None,
        indexed_bzip2_parallelization: object | None = None,
    ):
        captured["path"] = path
        captured["reader_mode"] = reader_mode
        captured["indexed_bzip2_parallelization"] = indexed_bzip2_parallelization
        return _FakeHandle(payload)

    monkeypatch.setattr(
        sharding_io, "_open_eta_enabled_source", fake_open_eta_enabled_source
    )

    batches = list(
        sharding_io.read_source_batches(
            source_path,
            "jsonl",
            batch_size=1,
            resolved_read_options={
                "reader_mode": "chunked_indexed",
                "chunk_size_bytes": 1024,
                "indexed_bzip2_parallelization": 6,
            },
        )
    )

    assert captured["path"] == source_path
    assert captured["reader_mode"] == "chunked_indexed"
    assert captured["indexed_bzip2_parallelization"] == 6
    assert [batch.height for batch in batches] == [1, 1]


def test_read_source_batches_jsonl_gz_uses_compressed_source(tmp_path: Path):
    source_path = tmp_path / "records.jsonl.gz"

    with gzip.open(source_path, "wt", encoding="utf-8") as handle:
        handle.write('{"CompanyName":"Alpha","value":1}\n')
        handle.write('{"CompanyName":"Beta","value":2}\n')

    batches = list(sharding_io.read_source_batches(source_path, "jsonl", batch_size=1))

    assert [batch.height for batch in batches] == [1, 1]
    assert batches[0]["CompanyName"].to_list() == ["Alpha"]


def test_read_source_batches_parquet_uses_slice_pushdown_chunking(tmp_path: Path):
    source_path = tmp_path / "records.parquet"
    pl.DataFrame({"CompanyName": ["Alpha", "Beta", "Gamma"]}).write_parquet(source_path)

    batches = list(
        sharding_io.read_source_batches(source_path, "parquet", batch_size=2)
    )

    assert [batch.height for batch in batches] == [2, 1]
    assert pl.concat(batches)["CompanyName"].to_list() == ["Alpha", "Beta", "Gamma"]


def test_read_source_batches_parquet_handles_exact_batch_size_multiple(
    tmp_path: Path,
):
    # Row count an exact multiple of batch_size is the real edge case for a
    # slice-loop reader: confirms the loop stops via the empty-chunk check
    # rather than yielding a trailing empty batch or missing the last one.
    source_path = tmp_path / "records.parquet"
    pl.DataFrame({"CompanyName": ["Alpha", "Beta", "Gamma", "Delta"]}).write_parquet(
        source_path
    )

    batches = list(
        sharding_io.read_source_batches(source_path, "parquet", batch_size=2)
    )

    assert [batch.height for batch in batches] == [2, 2]
    assert pl.concat(batches)["CompanyName"].to_list() == [
        "Alpha",
        "Beta",
        "Gamma",
        "Delta",
    ]


def test_harmonize_shard_schemas_noop_on_empty_input():
    assert sharding._harmonize_shard_schemas([]) == 0


# The research-stage policy gate the shard entry point runs is now one shared
# helper (acquisition.constants_status.require_runnable_plan); its own
# behaviour is covered in test_constants_status.py rather than re-tested per
# calling module.


def test_shard_system_source_research_policy_denies_stage(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    # A stub standing in for the real `SystemPlan`: only the attributes
    # `shard_system_source` consults are set.
    denied_plan = cast(
        "SystemPlan",
        SimpleNamespace(
            code="xx",
            status=STATUS_RESEARCH_REQUIRED,
            notes="research pending",
            resources=(),
            supported_countries=None,
            research=SimpleNamespace(
                allow_research_runtime=False, allowed_stages=("canonical",)
            ),
        ),
    )
    original = sharding.get_system_plan
    sharding.get_system_plan = lambda system_code: denied_plan

    with pytest.raises(
        RuntimeError, match="research execution policy denies stage 'shard'"
    ):
        sharding.shard_system_source(
            "xx", roots=workspace_roots, run_date="2026-06-26", allow_research=True
        )

    sharding.get_system_plan = original


def test_resolve_shard_plan_context_supported_status_passes_gate_without_allow_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`supported` must clear the runnable-status gate on its own, exactly
    like `live`, with no `allow_research` flag involved."""
    supported_plan = SimpleNamespace(
        code="xx",
        status="supported",
        notes="",
        supported_countries=None,
        research=None,
    )
    monkeypatch.setattr(sharding, "get_system_plan", lambda _system: supported_plan)

    plan, *_rest = sharding._resolve_shard_plan_context(
        system="xx",
        run_date=None,
        sidecar=False,
        allow_research=False,
        stage_options=None,
        emit=lambda _message: None,
    )

    assert plan is supported_plan


def test_resolve_shard_plan_context_blocked_status_rejected_even_with_allow_research(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`blocked` is never folded into the runnable-status set the way
    `research_required` is -- so `allow_research=True` must not let a
    blocked plan through, even when its research policy would otherwise
    permit the `shard` stage."""
    blocked_plan = SimpleNamespace(
        code="xx",
        status="blocked",
        notes="No access",
        supported_countries=None,
        research=SimpleNamespace(
            allow_research_runtime=True, allowed_stages=("shard",)
        ),
    )
    monkeypatch.setattr(sharding, "get_system_plan", lambda _system: blocked_plan)

    with pytest.raises(RuntimeError, match="xx: blocked - No access"):
        sharding._resolve_shard_plan_context(
            system="xx",
            run_date=None,
            sidecar=False,
            allow_research=True,
            stage_options=None,
            emit=lambda _message: None,
        )


def test_shard_system_source_emits_filter_message_when_rows_are_removed(
    tmp_path: Path, layer_fixture_dir, workspace_roots: WorkspaceRoots
):
    source_dir = layer_fixture_dir("xx", layer="source") / "2026-06-17"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_path = source_dir / "records-2026-06-17.csv"
    source_path.write_text(
        "country,CompanyName\nGB,Alpha\nDE,Beta\n",
        encoding="utf-8",
    )

    resource = SimpleNamespace(
        name="records",
        file_format="csv",
        resolve_snapshot_date=lambda run_date: "2026-06-17",
        resolve_file_name=lambda run_date: "records-2026-06-17.csv",
    )
    plan = cast(
        "SystemPlan",
        SimpleNamespace(
            code="xx",
            status=sharding.STATUS_LIVE,
            notes="",
            resources=(resource,),
            supported_countries=("GB",),
            system_uri_identifier_candidates=None,
            system_field_candidates=None,
            canonical_source_column_aliases=None,
            research=None,
        ),
    )
    original = sharding.get_system_plan
    sharding.get_system_plan = lambda system_code: plan

    messages: list[str] = []
    try:
        sharding.shard_system_source(
            "xx",
            roots=workspace_roots,
            run_date="2026-06-17",
            chunk_size=10,
            progress=messages.append,
        )
    finally:
        sharding.get_system_plan = original

    assert any(
        "filtered 1 row(s) by supported country policy" in message
        for message in messages
    )


def test_streaming_writer_handles_schema_width_drift(tmp_path: Path):
    writer = sharding._StreamingParquetWriter(
        output_dir=tmp_path,
        prefix="xx-sidecar",
        chunk_size=10,
    )

    writer.append(
        pl.DataFrame(
            {
                "system_uri": ["xx://1"],
                "custom_payload": ["foo"],
                "optional_flag": [True],
            }
        )
    )
    writer.append(
        pl.DataFrame(
            {
                "system_uri": ["xx://2"],
                "custom_payload": ["bar"],
            }
        )
    )

    paths = writer.finalize()
    assert [path.name for path in paths] == ["xx-sidecar-001.parquet"]

    out = pl.read_parquet(paths[0]).sort("system_uri")
    assert out.height == 2
    assert "optional_flag" in out.columns
    rows = out.select(["system_uri", "optional_flag"]).rows(named=True)
    assert rows == [
        {"system_uri": "xx://1", "optional_flag": True},
        {"system_uri": "xx://2", "optional_flag": None},
    ]
