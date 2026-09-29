from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from acquisition.canonical_dedupe import (
    CAPTURE_ROLE_COLUMN,
    CAPTURE_ROLE_DROPPED,
    CAPTURE_ROLE_KEPT,
    DEDUPE_REPORT_FILE_NAME,
    DROPPED_DUPLICATE_IDENTITY,
    DROPPED_MISSING_IDENTITY,
    DROPPED_REASON_COLUMN,
    write_dedupe_report,
    write_deduped_output,
)
from acquisition.output_chunking import NULL_PARTITION_SENTINEL
from workspace.data_file_naming import (
    COMPANION_DUPLICATES,
    is_primary_data_file,
)

_DROPPED_PREFIX = f"acme-{COMPANION_DUPLICATES}"


def _write_chunks(source_dir: Path, frames: list[pl.DataFrame]) -> list[Path]:
    source_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames, start=1):
        path = source_dir / f"chunk-{index:03d}.parquet"
        frame.write_parquet(path)
        paths.append(path)
    return paths


def _run(
    tmp_path: Path,
    frames: list[pl.DataFrame],
    *,
    chunk_size: int | None = 10,
    partition_by: str | None = "jurisdiction_code",
):
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_deduped_output(
        chunk_sources=_write_chunks(tmp_path / "in", frames),
        output_dir=output_dir,
        chunk_size=chunk_size,
        frame_transformer=pl.read_parquet,
        partition_by=partition_by,
        flat_prefix="acme",
        dropped_rows_prefix=_DROPPED_PREFIX,
    )


def test_keeps_first_row_per_identity_across_chunk_boundaries(tmp_path: Path):
    written, findings = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://2"],
                    "jurisdiction_code": ["GB", "GB"],
                    "name": ["first", "two"],
                }
            ),
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://3"],
                    "jurisdiction_code": ["GB", "GB"],
                    "name": ["second", "three"],
                }
            ),
        ],
    )

    kept = pl.read_parquet(written).sort("system_uri")
    assert kept["system_uri"].to_list() == ["acme://1", "acme://2", "acme://3"]
    # First-seen wins, so the later chunk's row for acme://1 is the one dropped.
    assert kept.filter(pl.col("system_uri") == "acme://1")["name"].item() == "first"
    assert findings.duplicate_group_count == 1
    assert findings.duplicate_rows_dropped == 1
    assert findings.written_rows == 3
    assert findings.input_rows == 4


def test_dropped_rows_are_captured_verbatim_with_a_reason(tmp_path: Path):
    _, findings = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://1", None, ""],
                    "jurisdiction_code": ["GB", "GB", "GB", "GB"],
                    "name": ["keep", "dup", "no-uri", "empty-uri"],
                }
            )
        ],
    )

    capture = pl.read_parquet(findings.dropped_rows_paths)
    dropped = capture.filter(pl.col(CAPTURE_ROLE_COLUMN) == CAPTURE_ROLE_DROPPED).sort(
        "name"
    )
    assert dropped["name"].to_list() == ["dup", "empty-uri", "no-uri"]
    assert dropped.filter(pl.col("name") == "dup")[DROPPED_REASON_COLUMN].item() == (
        DROPPED_DUPLICATE_IDENTITY
    )
    assert set(
        dropped.filter(pl.col("name") != "dup")[DROPPED_REASON_COLUMN].to_list()
    ) == {DROPPED_MISSING_IDENTITY}
    assert findings.dropped_missing_identity_rows == 2


def test_capture_holds_both_sides_of_each_collision(tmp_path: Path):
    """The capture is read on its own, so it has to carry the surviving row
    too: comparing a dropped row against the finished canonical view is
    misleading, because later Canonical steps rewrite columns on surviving
    rows only."""
    _, findings = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://1", "acme://2", None],
                    "jurisdiction_code": ["GB", "GB", "IE", "GB"],
                    "name": ["kept one", "dropped one", "untouched", "no-uri"],
                }
            )
        ],
    )

    capture = pl.read_parquet(findings.dropped_rows_paths)
    roles = dict(zip(capture["name"].to_list(), capture[CAPTURE_ROLE_COLUMN].to_list()))
    assert roles == {
        "kept one": CAPTURE_ROLE_KEPT,
        "dropped one": CAPTURE_ROLE_DROPPED,
        "no-uri": CAPTURE_ROLE_DROPPED,
    }
    # A counterpart is not a dropped row, so it carries no reason.
    assert (
        capture.filter(pl.col(CAPTURE_ROLE_COLUMN) == CAPTURE_ROLE_KEPT)[
            DROPPED_REASON_COLUMN
        ].null_count()
        == 1
    )
    # Only collisions get a counterpart: acme://2 never collided, and the
    # identity-less row has no surviving row to pair with.
    assert "untouched" not in roles
    assert findings.duplicate_group_count == 1


def test_dropped_rows_file_is_never_read_back_as_primary_data(tmp_path: Path):
    """The capture file is entity-shaped, so only its registered companion
    family keeps a primary-file glob from undoing the dedupe."""
    _, findings = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://1"],
                    "jurisdiction_code": ["GB", "GB"],
                    "name": ["keep", "dup"],
                }
            )
        ],
    )

    assert findings.dropped_rows_paths
    for path in findings.dropped_rows_paths:
        assert not is_primary_data_file(file_name=path.name, system_code="acme")


def test_partition_directories_are_lowercased_and_null_safe(tmp_path: Path):
    written, _ = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://2", "acme://3"],
                    "jurisdiction_code": ["GB", "AE-DU", None],
                    "name": ["a", "b", "c"],
                }
            )
        ],
    )

    assert sorted(path.parent.name for path in written) == [
        f"jurisdiction_code={NULL_PARTITION_SENTINEL}",
        "jurisdiction_code=ae-du",
        "jurisdiction_code=gb",
    ]
    # The directory name is normalized; the column value in the row is not.
    assert set(pl.read_parquet(written)["jurisdiction_code"].to_list()) == {
        "AE-DU",
        "GB",
        None,
    }


def test_partition_buffers_flush_at_chunk_size_across_inputs(tmp_path: Path):
    written, _ = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": [f"acme://{index}" for index in range(1, 4)],
                    "jurisdiction_code": ["GB"] * 3,
                    "name": ["a", "b", "c"],
                }
            ),
            pl.DataFrame(
                {
                    "system_uri": [f"acme://{index}" for index in range(4, 7)],
                    "jurisdiction_code": ["GB"] * 3,
                    "name": ["d", "e", "f"],
                }
            ),
        ],
        chunk_size=2,
    )

    # Six rows at two per file: buffering across the two inputs gives three
    # full files, not one file per (input, partition) pair.
    assert [path.name for path in written] == [
        "part-00001.parquet",
        "part-00002.parquet",
        "part-00003.parquet",
    ]
    assert [pl.read_parquet(path).height for path in written] == [2, 2, 2]


def test_flat_layout_dedupes_the_same_way(tmp_path: Path):
    written, findings = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://1", "acme://2"],
                    "jurisdiction_code": ["GB", "GB", "GB"],
                    "name": ["keep", "dup", "other"],
                }
            )
        ],
        partition_by=None,
    )

    assert [path.name for path in written] == ["acme-001.parquet"]
    assert pl.read_parquet(written)["system_uri"].to_list() == ["acme://1", "acme://2"]
    assert findings.duplicate_rows_dropped == 1


def test_report_is_written_even_when_nothing_was_dropped(tmp_path: Path):
    _, findings = _run(
        tmp_path,
        [
            pl.DataFrame(
                {
                    "system_uri": ["acme://1", "acme://2"],
                    "jurisdiction_code": ["GB", "GB"],
                    "name": ["a", "b"],
                }
            )
        ],
    )

    report_path = write_dedupe_report(
        output_dir=tmp_path / "out",
        system="acme",
        snapshot_date="2026-06-01",
        findings=findings,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report_path.name == DEDUPE_REPORT_FILE_NAME
    assert payload["duplicate_system_uri_groups"] == 0
    assert payload["duplicate_rows_dropped"] == 0
    assert payload["dropped_missing_system_uri_rows"] == 0
    assert payload["dropped_rows_files"] == []
    assert payload["written_rows"] == 2
    assert not findings.has_findings


def test_missing_identity_column_is_an_error_not_a_silent_drop(tmp_path: Path):
    with pytest.raises(ValueError, match="no 'system_uri' column"):
        _run(
            tmp_path,
            [pl.DataFrame({"jurisdiction_code": ["GB"], "name": ["a"]})],
        )
