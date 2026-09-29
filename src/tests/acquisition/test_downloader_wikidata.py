from __future__ import annotations

# isort: skip_file
# fmt: off
import gzip
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from acquisition import downloader_wikidata
from acquisition import wikidata_pipeline_helpers
from acquisition.downloader_wikidata import (
    _extract_claim_text_values_with_english,
    _is_wikidata_company_candidate_raw_line, _iter_wikidata_entities,
    extract_wikidata_company_projection,
    extract_wikidata_company_projection_two_pass)

# fmt: on


def _write_gz(path: Path, contents: str) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(contents.encode("utf-8"))


def _read_gz(path: Path) -> str:
    """Reads a gzip artifact back through a real decoder.

    Every raw-candidate assertion goes through this rather than through the
    bytes on disk: a stream concatenated wrongly, or finished without its
    trailer, still leaves a plausible-looking file and only fails on decode.
    """
    with gzip.open(path, "rb") as handle:
        return handle.read().decode("utf-8")


class _ReadHandle:
    def __init__(self, lines: list[bytes]):
        self._payload = b"".join(lines)
        self._offset = 0
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int) -> bytes:
        self.read_calls += 1
        if self._offset >= len(self._payload):
            return b""
        end = min(self._offset + size, len(self._payload))
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk


def test_close_wikidata_projection_outputs_preserves_temp_chunk_for_resume(
    tmp_path: Path,
):
    temp_path = tmp_path / "projection-part.tmp"
    temp_path.write_bytes(b"partial")
    projection_chunk_handle = temp_path.open("wb")

    wikidata_pipeline_helpers._close_wikidata_projection_outputs(
        projection_chunk_handle=projection_chunk_handle,
        projection_chunk_temp_path=temp_path,
    )

    assert temp_path.exists()


def test_extract_wikidata_company_projection_filters_and_projects_fields(
    tmp_path: Path,
):
    source = tmp_path / "wikidata.jsonl"
    destination = tmp_path / "wikidata-companies.jsonl"

    company_entity = {
        "id": "Q100",
        "type": "item",
        "modified": "2026-01-01T00:00:00Z",
        "labels": {"en": {"language": "en", "value": "Acme Corp"}},
        "descriptions": {"en": {"language": "en", "value": "A company"}},
        "aliases": {"en": [{"language": "en", "value": "Acme"}]},
        "sitelinks": {"enwiki": {"site": "enwiki", "title": "Acme"}},
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
            "P17": [{"mainsnak": {"datavalue": {"value": {"id": "Q145"}}}}],
            "P1001": [{"mainsnak": {"datavalue": {"value": {"id": "Q30"}}}}],
            "P571": [
                {
                    "mainsnak": {
                        "datavalue": {"value": {"time": "+2000-01-01T00:00:00Z"}}
                    }
                }
            ],
            "P576": [
                {
                    "mainsnak": {
                        "datavalue": {"value": {"time": "+2020-01-01T00:00:00Z"}}
                    }
                }
            ],
            "P159": [{"mainsnak": {"datavalue": {"value": {"id": "Q84"}}}}],
            "P452": [{"mainsnak": {"datavalue": {"value": {"id": "Q11661"}}}}],
            "P1454": [{"mainsnak": {"datavalue": {"value": {"id": "Q12308941"}}}}],
            "P856": [{"mainsnak": {"datavalue": {"value": "https://example.com"}}}],
            "P1278": [{"mainsnak": {"datavalue": {"value": "123456789"}}}],
            "P946": [{"mainsnak": {"datavalue": {"value": "US0000000001"}}}],
            "P2622": [{"mainsnak": {"datavalue": {"value": "01234567"}}}],
            "P1448": [
                {
                    "mainsnak": {
                        "datavalue": {
                            "value": {"text": "Acme Corporation", "language": "en"}
                        }
                    }
                },
                {
                    "mainsnak": {
                        "datavalue": {
                            "value": {"text": "Societe Acme", "language": "fr"}
                        }
                    }
                },
            ],
            "P1813": [
                {
                    "mainsnak": {
                        "datavalue": {"value": {"text": "ACME", "language": "en"}}
                    }
                },
            ],
        },
    }

    non_company_entity = {
        "id": "Q200",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
        },
    }

    with source.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(company_entity) + "\n")
        handle.write(json.dumps(non_company_entity) + "\n")

    rows_written = extract_wikidata_company_projection(source, destination)
    assert rows_written == 1

    lines = destination.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])

    assert payload["id"] == "Q100"
    assert payload["entity_type"] == "item"
    assert payload["modified"] == "2026-01-01T00:00:00Z"
    assert payload["label_en"] == "Acme Corp"
    assert payload["description_en"] == "A company"
    assert payload["aliases_en"] == ["Acme"]
    assert payload["official_name"] == ["Acme Corporation", "Societe Acme"]
    assert payload["official_name_en"] == ["Acme Corporation"]
    assert payload["official_name_variants"] == [
        {"value": "Acme Corporation", "language": "en"},
        {"value": "Societe Acme", "language": "fr"},
    ]
    assert payload["short_name"] == ["ACME"]
    assert payload["short_name_en"] == ["ACME"]
    assert payload["short_name_variants"] == [
        {"value": "ACME", "language": "en"},
    ]
    assert payload["instance_of"] == ["Q783794"]
    assert payload["country"] == ["Q145"]
    assert payload["jurisdiction"] == ["Q30"]
    assert payload["inception"] == "+2000-01-01T00:00:00Z"
    assert payload["dissolved"] == "+2020-01-01T00:00:00Z"
    assert payload["headquarters_location"] == ["Q84"]
    assert payload["industry"] == ["Q11661"]
    assert payload["legal_form"] == ["Q12308941"]
    assert payload["website"] == ["https://example.com"]
    assert payload["lei"] == ["123456789"]
    assert payload["company_number_gb"] == ["01234567"]
    assert payload["company_number_fr"] == []
    assert payload["company_number_de"] == []
    assert payload["isin"] == ["US0000000001"]
    assert payload["sitelinks_count"] == 1


def test_extract_wikidata_company_projection_emits_progress_messages(tmp_path: Path):
    source = tmp_path / "wikidata.jsonl"
    destination = tmp_path / "wikidata-companies.jsonl"
    progress_messages: list[str] = []

    entity = {
        "id": "Q100",
        "type": "item",
        "labels": {"en": {"language": "en", "value": "Acme Corp"}},
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }

    with source.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(entity) + "\n")

    extract_wikidata_company_projection(
        source,
        destination,
        progress=progress_messages.append,
        progress_every_entities=1,
        progress_every_seconds=0,
    )

    assert (
        progress_messages[0]
        == "[wikidata] prepare projection started: source=wikidata.jsonl"
    )
    assert progress_messages[1] == "[wikidata] prepare projection scanning dump"
    assert any("lines_processed=" in message for message in progress_messages)
    assert any("eta=" in message for message in progress_messages)


def test_extract_wikidata_company_projection_can_stop_after_max_lines(tmp_path: Path):
    source = tmp_path / "wikidata.jsonl"
    destination = tmp_path / "wikidata-companies.jsonl"
    progress_messages: list[str] = []

    non_company_entity = {
        "id": "Q200",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
        },
    }
    company_entity = {
        "id": "Q100",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }

    with source.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(non_company_entity) + "\n")
        handle.write(json.dumps(company_entity) + "\n")

    rows_written = extract_wikidata_company_projection(
        source,
        destination,
        progress=progress_messages.append,
        progress_every_entities=1,
        progress_every_seconds=0,
        max_lines=1,
    )

    assert rows_written == 0
    assert any("max_lines=1" in message for message in progress_messages)
    assert destination.read_text(encoding="utf-8") == ""


def test_extract_wikidata_company_projection_max_lines_is_inclusive(tmp_path: Path):
    source = tmp_path / "wikidata.jsonl"
    destination = tmp_path / "wikidata-companies.jsonl"

    company_entity = {
        "id": "Q100",
        "type": "item",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }
    second_company_entity = {
        "id": "Q101",
        "type": "item",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }

    with source.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(company_entity) + "\n")
        handle.write(json.dumps(second_company_entity) + "\n")

    rows_written = extract_wikidata_company_projection(
        source,
        destination,
        max_lines=1,
    )

    assert rows_written == 1
    lines = [
        line
        for line in destination.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["id"] == "Q100"


def test_extract_claim_text_values_with_english_returns_both_lists_in_one_pass():
    entity = {
        "claims": {
            "P1448": [
                {
                    "mainsnak": {
                        "datavalue": {
                            "value": {"text": "Acme Corporation", "language": "en"}
                        }
                    }
                },
                {
                    "mainsnak": {
                        "datavalue": {
                            "value": {"text": "Societe Acme", "language": "fr"}
                        }
                    }
                },
                {"mainsnak": {"datavalue": {"value": "fallback-plain-string"}}},
            ]
        }
    }

    all_values, english_values, tagged_values = _extract_claim_text_values_with_english(
        entity, "P1448"
    )

    assert all_values == ["Acme Corporation", "Societe Acme", "fallback-plain-string"]
    assert english_values == ["Acme Corporation"]
    assert tagged_values == [
        {"value": "Acme Corporation", "language": "en"},
        {"value": "Societe Acme", "language": "fr"},
        {"value": "fallback-plain-string", "language": None},
    ]


def test_iter_wikidata_entities_streams_bz2_input_without_bulk_reads(monkeypatch):
    handle = _ReadHandle(
        [
            b"[\n",
            b'{"id":"Q1","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n',
            b'{"id":"Q2","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}}\n',
            b"]\n",
        ]
    )

    def fake_indexed_open(path, parallelization):
        assert parallelization >= 2
        return handle

    monkeypatch.setattr(downloader_wikidata._indexed_bzip2, "open", fake_indexed_open)

    entities = list(_iter_wikidata_entities(Path("wikidata.json.bz2")))

    assert handle.read_calls > 0
    assert [line_number for line_number, _ in entities] == [2, 3]
    assert [entity["id"] for _, entity in entities] == ["Q1", "Q2"]


def test_iter_wikidata_entities_streams_gz_input_without_bulk_reads(tmp_path: Path):
    source = tmp_path / "wikidata.json.gz"

    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write("[\n")
        handle.write(
            '{"id":"Q1","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}\n'
        )
        handle.write(
            '{"id":"Q2","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}}\n'
        )
        handle.write("]\n")

    entities = list(_iter_wikidata_entities(source))

    assert [line_number for line_number, _ in entities] == [2, 3]
    assert [entity["id"] for _, entity in entities] == ["Q1", "Q2"]


def test_is_wikidata_company_candidate_raw_line_filters_obvious_non_companies():
    company_line = b'{"id":"Q1","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q783794"}}}}]}}'
    non_company_line = b'{"id":"Q2","claims":{"P31":[{"mainsnak":{"datavalue":{"value":{"id":"Q5"}}}}]}}'
    unrelated_line = b'{"id":"Q3","claims":{"P17":[{"mainsnak":{"datavalue":{"value":{"id":"Q145"}}}}]}}'

    assert _is_wikidata_company_candidate_raw_line(company_line) is True
    assert _is_wikidata_company_candidate_raw_line(non_company_line) is False
    assert _is_wikidata_company_candidate_raw_line(unrelated_line) is False


def test_extract_wikidata_company_projection_two_pass_migrates_flat_chunks_to_partitioned_layout(
    tmp_path: Path,
):
    source = tmp_path / "wikidata.jsonl"
    destination = tmp_path / "wikidata-companies-stream.jsonl"
    source.write_text("", encoding="utf-8")

    chunk_dir = destination.parent / "wikidata-companies.chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_prefix = "wikidata-companies-part-"
    (chunk_dir / f"{chunk_prefix}000001.jsonl").write_text(
        '{"id":"Q1"}\n', encoding="utf-8"
    )
    (chunk_dir / f"{chunk_prefix}000002.jsonl").write_text(
        '{"id":"Q2"}\n', encoding="utf-8"
    )

    rows_stream = extract_wikidata_company_projection_two_pass(
        source,
        destination,
        phase1_wiring_mode="stream",
        stream_resume=True,
    )

    assert rows_stream == 1
    partition_dir = chunk_dir / "000000-000999"
    assert (partition_dir / f"{chunk_prefix}000001.jsonl").exists()
    assert not (chunk_dir / f"{chunk_prefix}000001.jsonl").exists()
    assert not (chunk_dir / f"{chunk_prefix}000002.jsonl").exists()


def test_extract_wikidata_company_projection_two_pass_stream_starts_chunk_once_per_active_chunk(
    tmp_path: Path,
    mocker,
):
    source = tmp_path / "wikidata.jsonl"
    destination = tmp_path / "wikidata-companies-stream.jsonl"

    company_entity_1 = {
        "id": "Q100",
        "type": "item",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }
    company_entity_2 = {
        "id": "Q101",
        "type": "item",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }
    with source.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(company_entity_1) + "\n")
        handle.write(json.dumps(company_entity_2) + "\n")

    start_spy = mocker.spy(
        wikidata_pipeline_helpers, "_start_wikidata_projection_chunk"
    )

    rows = extract_wikidata_company_projection_two_pass(
        source,
        destination,
        phase1_wiring_mode="stream",
    )

    assert rows == 2
    assert start_spy.call_count == 1


def test_extract_wikidata_company_projection_two_pass_stream_emits_telemetry_contract(
    tmp_path: Path,
):
    source = tmp_path / "wikidata.jsonl"
    destination = tmp_path / "wikidata-companies-stream.jsonl"
    progress_messages: list[str] = []

    company_entity = {
        "id": "Q100",
        "type": "item",
        "claims": {
            "P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q783794"}}}}],
        },
    }
    with source.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(company_entity) + "\n")

    rows = extract_wikidata_company_projection_two_pass(
        source,
        destination,
        phase1_wiring_mode="stream",
        progress=progress_messages.append,
        progress_every_entities=1,
        progress_every_seconds=0,
    )

    assert rows == 1
    assert progress_messages
    assert all(message.startswith("[wikidata] ") for message in progress_messages)
    assert any("streaming progress:" in message for message in progress_messages)
    assert any("streaming complete:" in message for message in progress_messages)

    eta_regex = re.compile(
        r"eta=(unknown|\d+m|\d+h\d{2}m|\d+m-\d+m|\d+h\d{2}m-\d+h\d{2}m|\d+h\d{2}m-\d+m)"
    )
    eta_messages = [message for message in progress_messages if "eta=" in message]
    assert eta_messages
    assert all(eta_regex.search(message) for message in eta_messages)


def test_build_wikisieve_command_uses_spec_flag_and_max_records_not_max_companies():
    command = downloader_wikidata._build_wikisieve_command(
        binary_path=Path("wikisieve.exe"),
        source_path=Path("wikidata-all.json.gz"),
        spec_path=Path("resolved-spec.json"),
        summary_path=Path("summary.json"),
        output_mode="jsonl",
        resume_enabled=True,
        resume_state_path=Path("state.json"),
        chunk_dir_path=Path("chunks"),
        chunk_prefix="wikidata-companies-part-",
        destination_path=Path("destination.jsonl"),
        max_lines=100,
        max_companies=7,
    )

    assert command[0] == "wikisieve.exe"
    assert "--spec" in command
    assert command[command.index("--spec") + 1] == "resolved-spec.json"
    assert "--p279" not in command
    assert "--max-rows" in command
    assert command[command.index("--max-rows") + 1] == "100"
    assert "--max-records" in command
    assert command[command.index("--max-records") + 1] == "7"
    assert "--max-companies" not in command
    # resume_enabled=True routes chunked output through --chunk-dir, not
    # destination_path directly.
    assert command[command.index("--output") + 1] == "chunks"
    assert "--resume" in command
    assert "--state-path" in command
    assert "--replay-last-chunk" not in command


def test_build_wikisieve_command_writes_flat_file_when_resume_disabled():
    command = downloader_wikidata._build_wikisieve_command(
        binary_path=Path("wikisieve.exe"),
        source_path=Path("wikidata-all.json.gz"),
        spec_path=Path("resolved-spec.json"),
        summary_path=Path("summary.json"),
        output_mode="jsonl",
        resume_enabled=False,
        resume_state_path=Path("state.json"),
        chunk_dir_path=Path("chunks"),
        chunk_prefix="wikidata-companies-part-",
        destination_path=Path("destination.jsonl"),
        max_lines=None,
        max_companies=None,
    )

    # wikisieve only chunks when --resume is passed, so a non-resuming run
    # writes the --output path it is given (the caller's staging path).
    assert command[command.index("--output") + 1] == "destination.jsonl"
    assert "--resume" not in command
    assert "--state-path" not in command
    assert "--chunk-dir" not in command
    assert "--chunk-prefix" not in command
    assert "--replay-last-chunk" not in command


def test_write_resolved_wikisieve_spec_substitutes_absolute_p279_path(tmp_path: Path):
    spec_path = tmp_path / "company.json"
    spec_path.write_text(
        json.dumps(
            {
                "markers": [
                    {
                        "property": "P1454",
                        "match": {"type": "qid_closure_file", "path": "p279.json"},
                    }
                ],
                "match_logic": "any",
                "projected_fields": [],
            }
        ),
        encoding="utf-8",
    )
    p279_path = tmp_path / "acquire" / "p279.json"
    p279_path.parent.mkdir(parents=True, exist_ok=True)
    p279_path.write_text("[]", encoding="utf-8")
    destination_path = tmp_path / "resolved-spec.json"

    downloader_wikidata._write_resolved_wikisieve_spec(
        spec_path=spec_path, p279_path=p279_path, destination_path=destination_path
    )

    resolved = json.loads(destination_path.read_text(encoding="utf-8"))
    assert resolved["markers"][0]["match"]["path"] == str(p279_path.resolve())


def test_hash_file_sha256_matches_hashlib_over_the_same_bytes(tmp_path: Path):
    path = tmp_path / "wikisieve.exe"
    path.write_bytes(b"a fake binary's bytes, not a real PE image")

    assert (
        downloader_wikidata._hash_file_sha256(path)
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_write_wikidata_run_manifest_carries_the_binarys_hash_version_and_commit(
    tmp_path: Path,
):
    binary_path = tmp_path / "wikisieve.exe"
    binary_path.write_bytes(b"binary contents")
    manifest_path = tmp_path / "wikidata-run-manifest.json"

    downloader_wikidata._write_wikidata_run_manifest(
        manifest_path=manifest_path,
        binary_path=binary_path,
        engine="wikisieve",
        summary={
            "records_emitted": 2,
            "crate_version": "0.1.0",
            "git_commit": "abc1234",
            "git_dirty": False,
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["engine"] == "wikisieve"
    assert manifest["binary_path"] == str(binary_path)
    assert (
        manifest["binary_sha256"]
        == hashlib.sha256(binary_path.read_bytes()).hexdigest()
    )
    assert manifest["reported_version"] == "0.1.0"
    assert manifest["reported_commit"] == "abc1234"
    assert manifest["reported_dirty"] is False


def test_write_wikidata_run_manifest_reports_none_when_the_summary_lacks_build_provenance(
    tmp_path: Path,
):
    # A binary built before provenance was embedded, or a different engine's summary, still gets a manifest --
    # the absence of self-reported provenance is recorded, not treated as an error.
    binary_path = tmp_path / "wikisieve.exe"
    binary_path.write_bytes(b"binary contents")
    manifest_path = tmp_path / "wikidata-run-manifest.json"

    downloader_wikidata._write_wikidata_run_manifest(
        manifest_path=manifest_path,
        binary_path=binary_path,
        engine="wikisieve",
        summary={"records_emitted": 2},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["reported_version"] is None
    assert manifest["reported_commit"] is None
    assert manifest["reported_dirty"] is None


def _write_wikisieve_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A placeholder dump, a p279.json beside it, a fake binary and a minimal
    spec: everything the wikisieve path checks for before it runs anything."""
    source = tmp_path / "wikidata-all.json.gz"
    binary_path = tmp_path / "wikisieve.exe"
    spec_path = tmp_path / "company.json"
    source.write_bytes(b"placeholder")
    binary_path.write_bytes(b"binary")
    spec_path.write_text(
        json.dumps(
            {
                "markers": [
                    {
                        "property": "P1454",
                        "match": {"type": "qid_closure_file", "path": "p279.json"},
                    }
                ],
                "match_logic": "any",
                "projected_fields": [],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "p279.json").write_text("[]", encoding="utf-8")
    return source, binary_path, spec_path


def _wikisieve_defaults(binary_path: Path, spec_path: Path) -> dict[str, object]:
    return {
        "engine": "wikisieve",
        "binary_path": str(binary_path),
        "spec_path": str(spec_path),
        "output_mode": "jsonl",
    }


def _flag_value(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_extract_wikidata_company_projection_two_pass_wikisieve_merges_chunks_with_merge_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source, binary_path, spec_path = _write_wikisieve_fixture(tmp_path)
    destination = tmp_path / "wikidata-companies.jsonl"
    chunk_dir = tmp_path / "wikidata-companies.chunks"
    cache_path = tmp_path / "wikidata-companies-raw.jsonl.gz"
    progress_messages: list[str] = []
    commands: list[list[str]] = []
    captured: dict[str, Any] = {}

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs):
            commands.append(command)
            if command[1] != "merge-chunks":
                # The spec on the command line is a resolved temp copy with an
                # absolute p279 path; read it while its temp directory is alive.
                captured["resolved_spec"] = json.loads(
                    Path(_flag_value(command, "--spec")).read_text(encoding="utf-8")
                )
                run_chunk_dir = Path(_flag_value(command, "--output"))
                run_chunk_dir.mkdir(parents=True, exist_ok=True)
                (run_chunk_dir / "wikidata-companies-part-000001.jsonl").write_text(
                    '{"id":"Q100"}\n{"id":"Q101"}\n', encoding="utf-8"
                )
                Path(_flag_value(command, "--summary-json")).write_text(
                    '{"records_emitted": 2}', encoding="utf-8"
                )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

    monkeypatch.setattr(downloader_wikidata.subprocess, "Popen", FakeProcess)

    rows = extract_wikidata_company_projection_two_pass(
        source,
        destination,
        progress=progress_messages.append,
        stream_resume=True,
        projection_defaults=_wikisieve_defaults(binary_path, spec_path),
    )

    assert rows == 2
    extract_command, merge_command = commands
    assert extract_command[0] == str(binary_path)
    assert _flag_value(extract_command, "--output") == str(chunk_dir)
    assert "--resume" in extract_command
    assert "--p279" not in extract_command
    assert _flag_value(extract_command, "--raw-candidate-output") == str(cache_path)
    assert captured["resolved_spec"]["markers"][0]["match"]["path"] == str(
        (tmp_path / "p279.json").resolve()
    )
    # The merge is wikisieve's own, checked against both the run's summary and
    # its resume state, and it assembles the raw-candidate cache too.
    assert merge_command[:2] == [str(binary_path), "merge-chunks"]
    assert _flag_value(merge_command, "--chunk-dir") == str(chunk_dir)
    assert _flag_value(merge_command, "--output") == str(destination)
    assert _flag_value(merge_command, "--summary-json") == _flag_value(
        extract_command, "--summary-json"
    )
    assert _flag_value(merge_command, "--state-path") == _flag_value(
        extract_command, "--state-path"
    )
    assert _flag_value(merge_command, "--raw-candidate-output") == str(cache_path)
    assert any(
        message.startswith("[wikidata] wikisieve started:")
        for message in progress_messages
    )


def test_extract_wikidata_company_projection_two_pass_wikisieve_keeps_chunks_for_a_bounded_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source, binary_path, spec_path = _write_wikisieve_fixture(tmp_path)
    commands: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs):
            commands.append(command)
            run_chunk_dir = Path(_flag_value(command, "--output"))
            run_chunk_dir.mkdir(parents=True, exist_ok=True)
            (run_chunk_dir / "wikidata-companies-part-000001.jsonl").write_text(
                '{"id":"Q100"}\n', encoding="utf-8"
            )
            Path(_flag_value(command, "--summary-json")).write_text(
                '{"records_emitted": 1}', encoding="utf-8"
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

    monkeypatch.setattr(downloader_wikidata.subprocess, "Popen", FakeProcess)

    rows = extract_wikidata_company_projection_two_pass(
        source,
        tmp_path / "wikidata-companies.jsonl",
        max_lines=10,
        stream_resume=True,
        projection_defaults=_wikisieve_defaults(binary_path, spec_path),
    )

    assert rows == 1
    assert len(commands) == 1
    assert (
        tmp_path / "wikidata-companies.chunks" / "wikidata-companies-part-000001.jsonl"
    ).exists()


def test_extract_wikidata_company_projection_two_pass_wikisieve_stages_a_flat_run_then_renames_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source, binary_path, spec_path = _write_wikisieve_fixture(tmp_path)
    destination = tmp_path / "wikidata-companies.jsonl"
    cache_path = tmp_path / "wikidata-companies-raw.jsonl.gz"
    captured: dict[str, list[str]] = {}

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs):
            captured["command"] = command
            Path(_flag_value(command, "--output")).write_text(
                '{"id":"Q100"}\n', encoding="utf-8"
            )
            _write_gz(
                Path(_flag_value(command, "--raw-candidate-output")),
                '{"id":"Q100","claims":{}}\n',
            )
            Path(_flag_value(command, "--summary-json")).write_text(
                json.dumps(
                    {
                        "records_emitted": 1,
                        "crate_version": "0.1.0",
                        "git_commit": "deadbeef",
                        "git_dirty": False,
                    }
                ),
                encoding="utf-8",
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

    monkeypatch.setattr(downloader_wikidata.subprocess, "Popen", FakeProcess)

    rows = extract_wikidata_company_projection_two_pass(
        source,
        destination,
        stream_resume=False,
        projection_defaults=_wikisieve_defaults(binary_path, spec_path),
    )

    assert rows == 1
    command = captured["command"]
    # wikisieve writes staging siblings, never the live files.
    assert _flag_value(command, "--output") == str(destination) + ".tmp"
    assert _flag_value(command, "--raw-candidate-output") == str(cache_path) + ".tmp"
    assert "--resume" not in command
    assert destination.read_text(encoding="utf-8") == '{"id":"Q100"}\n'
    assert _read_gz(cache_path) == '{"id":"Q100","claims":{}}\n'
    assert sorted(path.name for path in tmp_path.glob("*.tmp")) == []
    # The run manifest ties the output back to the binary that made it.
    manifest = json.loads(
        (tmp_path / "wikidata-run-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["engine"] == "wikisieve"
    assert (
        manifest["binary_sha256"]
        == hashlib.sha256(binary_path.read_bytes()).hexdigest()
    )
    assert manifest["reported_version"] == "0.1.0"
    assert manifest["reported_commit"] == "deadbeef"
    assert manifest["reported_dirty"] is False


def _write_live_extract(tmp_path: Path) -> tuple[Path, Path]:
    destination = tmp_path / "wikidata-companies.jsonl"
    cache_path = tmp_path / "wikidata-companies-raw.jsonl.gz"
    destination.write_text('{"id":"Q1","live":true}\n', encoding="utf-8")
    _write_gz(cache_path, '{"id":"Q1","claims":{"live":true}}\n')
    return destination, cache_path


def _assert_live_extract_untouched(destination: Path, cache_path: Path) -> None:
    assert destination.read_text(encoding="utf-8") == '{"id":"Q1","live":true}\n'
    assert _read_gz(cache_path) == '{"id":"Q1","claims":{"live":true}}\n'


def test_extract_wikidata_company_projection_two_pass_wikisieve_failed_flat_run_leaves_the_live_extract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source, binary_path, spec_path = _write_wikisieve_fixture(tmp_path)
    destination, cache_path = _write_live_extract(tmp_path)

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs):
            # Dies part way: some rows written, no summary.
            Path(_flag_value(command, "--output")).write_text(
                '{"id":"Q100"}\n', encoding="utf-8"
            )
            _write_gz(Path(_flag_value(command, "--raw-candidate-output")), "partial\n")
            self.stderr = io.StringIO("killed\n")
            self.returncode = 1

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

    monkeypatch.setattr(downloader_wikidata.subprocess, "Popen", FakeProcess)

    with pytest.raises(RuntimeError, match="wikisieve projection failed"):
        extract_wikidata_company_projection_two_pass(
            source,
            destination,
            stream_resume=False,
            projection_defaults=_wikisieve_defaults(binary_path, spec_path),
        )

    _assert_live_extract_untouched(destination, cache_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_extract_wikidata_company_projection_two_pass_wikisieve_refuses_a_short_flat_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source, binary_path, spec_path = _write_wikisieve_fixture(tmp_path)
    destination, cache_path = _write_live_extract(tmp_path)

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs):
            Path(_flag_value(command, "--output")).write_text(
                '{"id":"Q100"}\n', encoding="utf-8"
            )
            _write_gz(Path(_flag_value(command, "--raw-candidate-output")), "raw\n")
            Path(_flag_value(command, "--summary-json")).write_text(
                '{"records_emitted": 2}', encoding="utf-8"
            )
            self.stderr = io.StringIO("")
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

    monkeypatch.setattr(downloader_wikidata.subprocess, "Popen", FakeProcess)

    with pytest.raises(RuntimeError, match="flat output row count mismatch"):
        extract_wikidata_company_projection_two_pass(
            source,
            destination,
            stream_resume=False,
            projection_defaults=_wikisieve_defaults(binary_path, spec_path),
        )

    _assert_live_extract_untouched(destination, cache_path)
    assert list(tmp_path.glob("*.tmp")) == []


def test_extract_wikidata_company_projection_two_pass_wikisieve_failed_resumable_run_leaves_the_live_extract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source, binary_path, spec_path = _write_wikisieve_fixture(tmp_path)
    destination, cache_path = _write_live_extract(tmp_path)
    commands: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs):
            commands.append(command)
            run_chunk_dir = Path(_flag_value(command, "--output"))
            run_chunk_dir.mkdir(parents=True, exist_ok=True)
            (run_chunk_dir / "wikidata-companies-part-000001.jsonl").write_text(
                '{"id":"Q100"}\n', encoding="utf-8"
            )
            self.stderr = io.StringIO("killed\n")
            self.returncode = 1

        def wait(self, timeout: int | None = None) -> int:
            return self.returncode

    monkeypatch.setattr(downloader_wikidata.subprocess, "Popen", FakeProcess)

    with pytest.raises(RuntimeError, match="wikisieve projection failed"):
        extract_wikidata_company_projection_two_pass(
            source,
            destination,
            stream_resume=True,
            projection_defaults=_wikisieve_defaults(binary_path, spec_path),
        )

    _assert_live_extract_untouched(destination, cache_path)
    # No merge was attempted, and the finished chunk stays for the next --resume.
    assert len(commands) == 1
    assert (
        tmp_path / "wikidata-companies.chunks" / "wikidata-companies-part-000001.jsonl"
    ).exists()


def test_extract_wikidata_company_projection_two_pass_wikisieve_raises_when_binary_missing(
    tmp_path: Path,
):
    source = tmp_path / "wikidata-all.json.gz"
    destination = tmp_path / "wikidata-companies.jsonl"
    source.write_bytes(b"placeholder")
    (tmp_path / "p279.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="wikisieve binary not found"):
        extract_wikidata_company_projection_two_pass(
            source,
            destination,
            projection_defaults={
                "engine": "wikisieve",
                "binary_path": str(tmp_path / "missing-wikisieve.exe"),
                "spec_path": str(tmp_path / "company.json"),
            },
        )


def test_resolve_wikidata_raw_candidate_cache_path_defaults_to_a_destination_sibling(
    tmp_path: Path,
):
    destination = tmp_path / "prepare" / "wikidata-companies.jsonl"

    assert downloader_wikidata._resolve_wikidata_raw_candidate_cache_path(
        None, destination_path=destination
    ) == (tmp_path / "prepare" / "wikidata-companies-raw.jsonl.gz")
    assert (
        downloader_wikidata._resolve_wikidata_raw_candidate_cache_path(
            {"raw_candidate_cache": False}, destination_path=destination
        )
        is None
    )
    assert downloader_wikidata._resolve_wikidata_raw_candidate_cache_path(
        {"raw_candidate_cache": str(tmp_path / "elsewhere.jsonl.gz")},
        destination_path=destination,
    ) == (tmp_path / "elsewhere.jsonl.gz")

    with pytest.raises(ValueError, match="raw_candidate_cache"):
        downloader_wikidata._resolve_wikidata_raw_candidate_cache_path(
            {"raw_candidate_cache": 3}, destination_path=destination
        )


def test_build_wikisieve_command_omits_raw_candidate_output_when_capture_is_off():
    command = downloader_wikidata._build_wikisieve_command(
        binary_path=Path("wikisieve.exe"),
        source_path=Path("dump.json.gz"),
        spec_path=Path("spec.json"),
        summary_path=Path("summary.json"),
        output_mode="jsonl",
        resume_enabled=False,
        resume_state_path=Path("state.json"),
        chunk_dir_path=Path("chunks"),
        chunk_prefix="part-",
        destination_path=Path("out.jsonl"),
        max_lines=None,
        max_companies=None,
        raw_candidate_cache_path=None,
    )

    assert "--raw-candidate-output" not in command
