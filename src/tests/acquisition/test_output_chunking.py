from __future__ import annotations

from pathlib import Path

import polars as pl

from acquisition.output_chunking import (
    collate_chunk_name,
    write_chunked_output_files,
    write_chunked_output_files_from_frame,
    write_partitioned_output_files,
    write_partitioned_output_files_from_frame,
)


def _write_source(path: Path, frame: pl.DataFrame) -> Path:
    frame.write_parquet(path)
    return path


def test_collate_chunk_name_passthrough_and_reindex() -> None:
    assert collate_chunk_name("sample.parquet", "gb") == "sample.parquet"
    assert collate_chunk_name("gb-7.parquet", "ie") == "ie-007.parquet"


def test_write_chunked_output_files_single_file_when_chunk_size_none(tmp_path: Path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = [
        _write_source(source_dir / "a.parquet", pl.DataFrame({"value": [1, 2]})),
        _write_source(source_dir / "b.parquet", pl.DataFrame({"value": [3]})),
    ]
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = write_chunked_output_files(
        chunk_sources=sources,
        output_dir=output_dir,
        prefix="xx",
        chunk_size=None,
        frame_transformer=pl.read_parquet,
    )

    assert [path.name for path in written] == ["xx-001.parquet"]
    assert pl.read_parquet(written[0])["value"].to_list() == [1, 2, 3]


def test_write_chunked_output_files_splits_on_exact_multiple(tmp_path: Path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = [
        _write_source(source_dir / "a.parquet", pl.DataFrame({"value": list(range(4))}))
    ]
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = write_chunked_output_files(
        chunk_sources=sources,
        output_dir=output_dir,
        prefix="xx",
        chunk_size=2,
        frame_transformer=pl.read_parquet,
    )

    assert [path.name for path in written] == ["xx-001.parquet", "xx-002.parquet"]
    assert [pl.read_parquet(path).height for path in written] == [2, 2]


def test_write_chunked_output_files_carries_remainder_into_final_chunk(
    tmp_path: Path,
):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = [
        _write_source(source_dir / "a.parquet", pl.DataFrame({"value": list(range(5))}))
    ]
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = write_chunked_output_files(
        chunk_sources=sources,
        output_dir=output_dir,
        prefix="xx",
        chunk_size=2,
        frame_transformer=pl.read_parquet,
    )

    assert [path.name for path in written] == [
        "xx-001.parquet",
        "xx-002.parquet",
        "xx-003.parquet",
    ]
    assert [pl.read_parquet(path).height for path in written] == [2, 2, 1]


def test_write_partitioned_output_files_groups_by_column(tmp_path: Path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = [
        _write_source(
            source_dir / "a.parquet",
            pl.DataFrame(
                {
                    "jurisdiction_code": ["ie", "gb", "ie"],
                    "value": [1, 2, 3],
                }
            ),
        ),
        _write_source(
            source_dir / "b.parquet",
            pl.DataFrame({"jurisdiction_code": ["gb"], "value": [4]}),
        ),
    ]
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = write_partitioned_output_files(
        chunk_sources=sources,
        output_dir=output_dir,
        partition_by="jurisdiction_code",
        chunk_size=None,
        frame_transformer=pl.read_parquet,
    )

    written_by_name = {
        path.relative_to(output_dir).as_posix(): path for path in written
    }
    assert set(written_by_name) == {
        "jurisdiction_code=ie/part-00001.parquet",
        "jurisdiction_code=gb/part-00001.parquet",
    }
    ie_values = sorted(
        pl.read_parquet(written_by_name["jurisdiction_code=ie/part-00001.parquet"])[
            "value"
        ].to_list()
    )
    gb_values = sorted(
        pl.read_parquet(written_by_name["jurisdiction_code=gb/part-00001.parquet"])[
            "value"
        ].to_list()
    )
    assert ie_values == [1, 3]
    assert gb_values == [2, 4]


def test_write_partitioned_output_files_chunks_within_a_partition(tmp_path: Path):
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources = [
        _write_source(
            source_dir / "a.parquet",
            pl.DataFrame(
                {
                    "jurisdiction_code": ["ie", "ie", "ie"],
                    "value": [1, 2, 3],
                }
            ),
        ),
    ]
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = write_partitioned_output_files(
        chunk_sources=sources,
        output_dir=output_dir,
        partition_by="jurisdiction_code",
        chunk_size=2,
        frame_transformer=pl.read_parquet,
    )

    names = sorted(path.relative_to(output_dir).as_posix() for path in written)
    assert names == [
        "jurisdiction_code=ie/part-00001.parquet",
        "jurisdiction_code=ie/part-00002.parquet",
    ]


def test_write_partitioned_output_files_returns_empty_for_no_sources(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    written = write_partitioned_output_files(
        chunk_sources=[],
        output_dir=output_dir,
        partition_by="jurisdiction_code",
        chunk_size=None,
        frame_transformer=pl.read_parquet,
    )

    assert written == []


def test_write_chunked_output_files_from_frame_single_file(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    frame = pl.DataFrame({"value": [1, 2, 3]})

    written = write_chunked_output_files_from_frame(
        frame=frame, output_dir=output_dir, prefix="xx", chunk_size=None
    )

    assert [path.name for path in written] == ["xx-001.parquet"]
    assert pl.read_parquet(written[0])["value"].to_list() == [1, 2, 3]


def test_write_chunked_output_files_from_frame_splits_with_remainder(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    frame = pl.DataFrame({"value": list(range(5))})

    written = write_chunked_output_files_from_frame(
        frame=frame, output_dir=output_dir, prefix="xx", chunk_size=2
    )

    assert [path.name for path in written] == [
        "xx-001.parquet",
        "xx-002.parquet",
        "xx-003.parquet",
    ]
    assert [pl.read_parquet(path).height for path in written] == [2, 2, 1]


def test_write_chunked_output_files_from_frame_empty_frame_returns_no_files(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    frame = pl.DataFrame({"value": []}, schema={"value": pl.Int64})

    written = write_chunked_output_files_from_frame(
        frame=frame, output_dir=output_dir, prefix="xx", chunk_size=None
    )

    assert written == []


def test_write_partitioned_output_files_from_frame_groups_by_column(tmp_path: Path):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    frame = pl.DataFrame(
        {
            "jurisdiction_code": ["ie", "gb", "ie", "gb"],
            "value": [1, 2, 3, 4],
        }
    )

    written = write_partitioned_output_files_from_frame(
        frame=frame,
        output_dir=output_dir,
        partition_by="jurisdiction_code",
        chunk_size=None,
    )

    written_by_name = {
        path.relative_to(output_dir).as_posix(): path for path in written
    }
    assert set(written_by_name) == {
        "jurisdiction_code=ie/part-00001.parquet",
        "jurisdiction_code=gb/part-00001.parquet",
    }
    ie_values = sorted(
        pl.read_parquet(written_by_name["jurisdiction_code=ie/part-00001.parquet"])[
            "value"
        ].to_list()
    )
    gb_values = sorted(
        pl.read_parquet(written_by_name["jurisdiction_code=gb/part-00001.parquet"])[
            "value"
        ].to_list()
    )
    assert ie_values == [1, 3]
    assert gb_values == [2, 4]


def test_write_partitioned_output_files_from_frame_chunks_within_a_partition(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    frame = pl.DataFrame(
        {
            "jurisdiction_code": ["ie", "ie", "ie"],
            "value": [1, 2, 3],
        }
    )

    written = write_partitioned_output_files_from_frame(
        frame=frame,
        output_dir=output_dir,
        partition_by="jurisdiction_code",
        chunk_size=2,
    )

    names = sorted(path.relative_to(output_dir).as_posix() for path in written)
    assert names == [
        "jurisdiction_code=ie/part-00001.parquet",
        "jurisdiction_code=ie/part-00002.parquet",
    ]


def test_write_partitioned_output_files_from_frame_empty_returns_no_files(
    tmp_path: Path,
):
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    frame = pl.DataFrame({"jurisdiction_code": [], "value": []})

    written = write_partitioned_output_files_from_frame(
        frame=frame,
        output_dir=output_dir,
        partition_by="jurisdiction_code",
        chunk_size=None,
    )

    assert written == []
