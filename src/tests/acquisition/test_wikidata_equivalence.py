from __future__ import annotations

# isort: skip_file

import json
from pathlib import Path

import pytest

# Keep this import layout stable across isort/ruff-format in pre-commit.
# fmt: off
from acquisition import wikidata_equivalence
from acquisition.wikidata_equivalence import (compare_projected_record_sets,
                                              load_jsonl_records_by_id,
                                              run_wikisieve,
                                              write_resolved_spec)

# fmt: on


def test_load_jsonl_records_by_id_reads_records(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "Q1", "label_en": "Acme"}),
                json.dumps({"id": "Q2", "label_en": "Beta"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_jsonl_records_by_id(path)

    assert set(records) == {"Q1", "Q2"}
    assert records["Q1"]["label_en"] == "Acme"


def test_load_jsonl_records_by_id_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "Q1", "label_en": "Acme"}),
                json.dumps({"id": "Q1", "label_en": "Acme Duplicate"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate record id 'Q1'"):
        load_jsonl_records_by_id(path)


def test_compare_projected_record_sets_detects_missing_extra_and_content_drift() -> (
    None
):
    expected = {
        "Q1": {"id": "Q1", "label_en": "Acme", "country": ["Q145"]},
        "Q2": {"id": "Q2", "label_en": "Beta", "country": ["Q30"]},
    }
    actual = {
        "Q1": {"id": "Q1", "label_en": "Acme", "country": ["Q840"]},
        "Q3": {"id": "Q3", "label_en": "Gamma", "country": ["Q183"]},
    }

    result = compare_projected_record_sets(expected, actual)

    assert result.is_equivalent is False
    assert result.expected_count == 2
    assert result.actual_count == 2
    assert result.missing_ids == ("Q2",)
    assert result.extra_ids == ("Q3",)
    assert len(result.mismatches) == 1
    assert result.mismatches[0].record_id == "Q1"
    assert result.mismatches[0].differing_fields == ("country",)


def test_compare_projected_record_sets_accepts_exact_content_match() -> None:
    expected = {
        "Q1": {"id": "Q1", "label_en": "Acme", "country": ["Q145"]},
    }
    actual = {
        "Q1": {"id": "Q1", "label_en": "Acme", "country": ["Q145"]},
    }

    result = compare_projected_record_sets(expected, actual)

    assert result.is_equivalent is True
    assert result.missing_ids == ()
    assert result.extra_ids == ()
    assert result.mismatches == ()


def test_write_resolved_spec_points_every_closure_marker_at_the_absolute_p279(
    tmp_path: Path,
) -> None:
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "markers": [
                    {"match": {"type": "qid_closure_file", "path": "p279.json"}},
                    {"match": {"type": "qid_set", "qids": ["Q4830453"]}},
                ]
            }
        ),
        encoding="utf-8",
    )
    p279_path = tmp_path / "closure" / "p279.json"
    destination = tmp_path / "resolved.json"

    write_resolved_spec(
        spec_path=spec_path, p279_path=p279_path, destination_path=destination
    )

    resolved = json.loads(destination.read_text(encoding="utf-8"))
    assert resolved["markers"][0]["match"]["path"] == str(p279_path.resolve())
    assert resolved["markers"][1] == {
        "match": {"type": "qid_set", "qids": ["Q4830453"]}
    }
    # The tracked spec is read, never rewritten.
    assert (
        json.loads(spec_path.read_text(encoding="utf-8"))["markers"][0]["match"]["path"]
        == "p279.json"
    )


def test_run_wikisieve_builds_the_command_and_reads_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}

    class FakeCompleted:
        returncode = 0
        stdout = "scanned 10 lines\n"

    def fake_run(command: list[str], **kwargs):
        captured["command"] = command
        Path(command[command.index("--summary-json") + 1]).write_text(
            json.dumps({"records_emitted": 3, "lines_scanned": 10}), encoding="utf-8"
        )
        return FakeCompleted()

    monkeypatch.setattr(wikidata_equivalence.subprocess, "run", fake_run)
    source = tmp_path / "sample.jsonl.gz"
    source.write_bytes(b"placeholder")
    manifest = tmp_path / "wikisieve-manifest.json"

    emitted, stdout = run_wikisieve(
        wikisieve_binary=tmp_path / "wikisieve.exe",
        source_path=source,
        spec_path=tmp_path / "spec.json",
        destination_path=tmp_path / "out.jsonl",
        summary_path=tmp_path / "summary.json",
        max_rows=500,
        manifest_path=manifest,
    )

    assert emitted == 3
    assert stdout == "scanned 10 lines\n"
    command = captured["command"]
    assert command[0] == str(tmp_path / "wikisieve.exe")
    assert command[command.index("--max-rows") + 1] == "500"
    assert command[command.index("--output-mode") + 1] == "jsonl"
    written = json.loads(manifest.read_text(encoding="utf-8"))
    assert written["engine"] == "wikisieve"
    assert written["summary"]["records_emitted"] == 3
    assert written["dump_input_bytes"] == len(b"placeholder")


def test_run_wikisieve_raises_on_a_failed_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailedCompleted:
        returncode = 2
        stdout = "spec error"

    monkeypatch.setattr(
        wikidata_equivalence.subprocess, "run", lambda *a, **k: FailedCompleted()
    )
    source = tmp_path / "sample.jsonl.gz"
    source.write_bytes(b"")

    with pytest.raises(RuntimeError, match="exit code 2"):
        run_wikisieve(
            wikisieve_binary=tmp_path / "wikisieve.exe",
            source_path=source,
            spec_path=tmp_path / "spec.json",
            destination_path=tmp_path / "out.jsonl",
            summary_path=tmp_path / "summary.json",
            max_rows=None,
        )
