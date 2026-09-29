import re
from pathlib import Path

import polars as pl

from acquisition.chunking import write_chunked_parquet


def test_write_chunked_parquet_uses_zero_padded_ascending_names(tmp_path: Path):
    frame = pl.DataFrame({"value": list(range(250_001))})

    paths = write_chunked_parquet(frame, tmp_path, "fr", chunk_size=100_000)

    assert [path.name for path in paths] == [
        "fr-001.parquet",
        "fr-002.parquet",
        "fr-003.parquet",
    ]
    assert [pl.read_parquet(path).height for path in paths] == [
        100_000,
        100_000,
        50_001,
    ]


def test_write_chunked_parquet_rejects_non_positive_chunk_size(tmp_path: Path):
    frame = pl.DataFrame({"value": [1]})

    try:
        write_chunked_parquet(frame, tmp_path, "fr", chunk_size=0)
    except ValueError as exc:
        assert "chunk_size must be greater than zero" in str(exc)
    else:
        raise AssertionError("Expected non-positive chunk_size to be rejected")


def test_write_chunked_parquet_returns_no_files_for_empty_frame(tmp_path: Path):
    frame = pl.DataFrame({"value": []}, schema={"value": pl.Int64})

    paths = write_chunked_parquet(frame, tmp_path / "chunks", "fr")

    assert paths == []
    assert (tmp_path / "chunks").exists()


def test_write_chunked_parquet_emits_shared_telemetry_summary(tmp_path: Path):
    frame = pl.DataFrame({"value": list(range(5))})
    messages: list[str] = []

    write_chunked_parquet(frame, tmp_path, "fr", chunk_size=2, progress=messages.append)

    assert any("chunk 1:" in message for message in messages)
    assert any("shard throughput summary" in message for message in messages)
    assert any("shard end-to-end throughput" in message for message in messages)
    assert any("shard phase summary" in message for message in messages)


def test_write_chunked_parquet_telemetry_contract_shapes(tmp_path: Path):
    frame = pl.DataFrame({"value": list(range(5))})
    messages: list[str] = []

    write_chunked_parquet(frame, tmp_path, "fr", chunk_size=2, progress=messages.append)

    chunk_messages = [message for message in messages if " chunk " in message]
    summary_messages = [
        message for message in messages if "shard throughput summary" in message
    ]
    end_to_end_messages = [
        message for message in messages if "shard end-to-end throughput" in message
    ]
    phase_summary_messages = [
        message for message in messages if "shard phase summary" in message
    ]

    assert chunk_messages
    assert summary_messages
    assert end_to_end_messages
    assert phase_summary_messages

    chunk_regex = re.compile(
        r"^\[fr\] chunk [0-9]+: [0-9,]+ rows in [0-9]+\.[0-9]{2}s \([0-9,]+ rows/s\)$"
    )
    summary_regex = re.compile(
        r"^\[fr\] shard throughput summary - min: [0-9,]+ rows/s, median: [0-9,]+ rows/s, max: [0-9,]+ rows/s$"
    )
    end_to_end_regex = re.compile(
        r"^\[fr\] shard end-to-end throughput - [0-9,]+ rows in [0-9]+\.[0-9]{2}s \([0-9,]+ rows/s\)$"
    )
    phase_summary_regex = re.compile(r"^\[fr\] shard phase summary - .+")

    assert all(chunk_regex.fullmatch(message) for message in chunk_messages)
    assert all(summary_regex.fullmatch(message) for message in summary_messages)
    assert all(end_to_end_regex.fullmatch(message) for message in end_to_end_messages)
    assert all(
        phase_summary_regex.fullmatch(message) for message in phase_summary_messages
    )
