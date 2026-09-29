from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import polars as pl

# Hive/Spark's standard null-partition sentinel: DuckDB's
# read_parquet(..., hive_partitioning=1) recognizes this literal directory
# name as NULL (verified empirically), not the string "__HIVE_DEFAULT_PARTITION__"
# itself. A plain empty jurisdiction_code=/ directory name is ambiguous --
# both DuckDB's hive-partition inference and a human reading the directory
# listing read it as an empty-string value, when the underlying parquet
# column is genuinely null (confirmed on real Wikidata cleansed output,
# jurisdiction_code=/ rows carry a null column value, but
# read_parquet(hive_partitioning=1) silently substitutes "" for it because
# the directory name and the file's own column collide). Using the Hive
# null sentinel for the partition-naming column avoids that substitution.
# Lives here, with the partitioned writers, so the Canonical and Cleanse
# stages name their partition directories from one declaration rather than a
# copy each.
NULL_PARTITION_SENTINEL = "__HIVE_DEFAULT_PARTITION__"


def collate_chunk_name(base_name: str, prefix: str) -> str:
    """Rewrite a chunked output filename (e.g. "foo-003.parquet", matching
    the naming scheme written by `_write_frame_chunks` above) onto a new
    `prefix`, preserving its numeric index and extension.
    """
    match = re.match(
        r"^(?P<source_prefix>.+?)-(?P<index>\d+)(?P<suffix>\.parquet|\.csv)$", base_name
    )
    if not match:
        return base_name

    index = int(match.group("index"))
    suffix = match.group("suffix")
    return f"{prefix}-{index:03d}{suffix}"


def _write_frame_chunks(
    *,
    frame: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    chunk_size: int | None,
    width: int = 3,
    start_index: int = 1,
) -> list[Path]:
    """Slice one already-materialized frame into chunk_size-bounded (or
    single, if chunk_size is None) parquet files named
    f"{prefix}-{index:0{width}d}.parquet" under output_dir. Shared primitive
    behind both the chunk_sources-based and frame-based entry points below.
    """
    if frame.height == 0:
        return []

    step = chunk_size if chunk_size is not None else frame.height
    written_paths: list[Path] = []
    index = start_index
    for start in range(0, frame.height, step):
        output_path = output_dir / f"{prefix}-{index:0{width}d}.parquet"
        frame.slice(start, step).write_parquet(output_path, compression="snappy")
        written_paths.append(output_path)
        index += 1
    return written_paths


def write_chunked_output_files_from_frame(
    *,
    frame: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    chunk_size: int | None,
) -> list[Path]:
    """Write an already-combined, already-materialized frame straight to
    chunk_size-bounded (or single, if chunk_size is None) output files --
    no intermediate chunk_sources round-trip. Use this when the caller has
    already had to fully materialize the combined frame for its own reasons
    (e.g. a whole-dataset dedupe) rather than reading it from disk again.
    """
    return _write_frame_chunks(
        frame=frame, output_dir=output_dir, prefix=prefix, chunk_size=chunk_size
    )


def write_partitioned_output_files_from_frame(
    *,
    frame: pl.DataFrame,
    output_dir: Path,
    partition_by: str,
    chunk_size: int | None,
    label: str | None = None,
    include_key: bool = True,
) -> list[Path]:
    """Group an already-combined, already-materialized frame by
    `partition_by` and write Hive-style `{label}={value}/part-{NNNNN}.parquet`
    files (label defaults to partition_by), chunked at `chunk_size` rows per
    group (or one file per group when chunk_size is None) -- same naming
    scheme as cleanser_orchestrate._write_country_partition_files. No
    intermediate chunk_sources round-trip; see
    write_chunked_output_files_from_frame.

    `label` and `include_key=False` exist for the case where the grouping
    column is a derived/normalized value (e.g. a lowercased alias used only
    for directory naming) distinct from the column name that should appear
    in the written rows and the partition-directory label.
    """
    if frame.height == 0:
        return []

    partition_label = label if label is not None else partition_by
    written_paths: list[Path] = []
    for group_value, group_frame in frame.partition_by(
        partition_by, as_dict=True, include_key=include_key
    ).items():
        value = group_value[0] if isinstance(group_value, tuple) else group_value
        partition_dir = output_dir / f"{partition_label}={value}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        written_paths.extend(
            _write_frame_chunks(
                frame=group_frame,
                output_dir=partition_dir,
                prefix="part",
                chunk_size=chunk_size,
                width=5,
            )
        )

    return written_paths


class FlatChunkWriter:
    """Accumulates incrementally-supplied frames and flushes them as
    chunk_size-bounded `{prefix}-{NNN}.parquet` files under output_dir.

    chunk_size=None means "don't split -- write a single output file",
    matching SourceResource.resolve_output_chunk_size / sharding.py's
    _StreamingParquetWriter: rows are accumulated across every added frame
    and only ever flushed once, at close().

    Stateful rather than a single function call so a caller that has to walk
    its inputs one at a time for its own reasons (a streaming dedupe pass,
    say) can hand rows over as it goes instead of materializing the whole
    dataset first; write_chunked_output_files below is the plain
    read-transform-write loop over it.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        prefix: str,
        chunk_size: int | None,
        width: int = 3,
    ) -> None:
        self._output_dir = output_dir
        self._prefix = prefix
        self._chunk_size = chunk_size
        self._width = width
        self._buffer_frames: list[pl.DataFrame] = []
        self._buffered_rows = 0
        self._next_index = 1
        self._written_paths: list[Path] = []

    def add(self, frame: pl.DataFrame) -> None:
        if frame.height == 0:
            return
        self._buffer_frames.append(frame)
        self._buffered_rows += frame.height
        self._flush()

    def close(self) -> list[Path]:
        self._flush(final=True)
        return self._written_paths

    def _flush(self, *, final: bool = False) -> None:
        if not self._buffer_frames:
            return
        if not final and (
            self._chunk_size is None or self._buffered_rows < self._chunk_size
        ):
            return

        combined = pl.concat(self._buffer_frames, how="diagonal_relaxed")
        if final or self._chunk_size is None:
            to_write = combined
            remainder = None
        else:
            ready_rows = (combined.height // self._chunk_size) * self._chunk_size
            if ready_rows == 0:
                return
            to_write = combined.slice(0, ready_rows)
            remainder = combined.slice(ready_rows, combined.height - ready_rows)

        new_paths = _write_frame_chunks(
            frame=to_write,
            output_dir=self._output_dir,
            prefix=self._prefix,
            chunk_size=self._chunk_size,
            width=self._width,
            start_index=self._next_index,
        )
        self._written_paths.extend(new_paths)
        self._next_index += len(new_paths)

        if remainder is None or remainder.height == 0:
            self._buffer_frames = []
            self._buffered_rows = 0
        else:
            self._buffer_frames = [remainder]
            self._buffered_rows = remainder.height


class PartitionedChunkWriter:
    """Groups incrementally-supplied frames by `partition_by` and flushes
    each group as chunk_size-bounded `{label}={value}/part-{NNNNN}.parquet`
    files -- the same naming, `label` and `include_key` semantics as
    write_partitioned_output_files_from_frame, but with one FlatChunkWriter
    per partition value so rows can arrive a frame at a time.

    A partition's buffer is flushed as soon as it reaches chunk_size rows, so
    peak memory is bounded by (live partitions x chunk_size) rather than by
    the whole dataset. Buffering across several added frames rather than
    writing one file per (frame, partition) pair is deliberate: the
    one-file-per-input-chunk shape produces far more, far smaller files and
    benchmarked 1.3-2.6x slower than a full-materialize pass.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        partition_by: str,
        chunk_size: int | None,
        label: str | None = None,
        include_key: bool = True,
    ) -> None:
        self._output_dir = output_dir
        self._partition_by = partition_by
        self._chunk_size = chunk_size
        self._label = label if label is not None else partition_by
        self._include_key = include_key
        self._writers: dict[str, FlatChunkWriter] = {}

    def add(self, frame: pl.DataFrame) -> None:
        if frame.height == 0:
            return
        for group_value, group_frame in frame.partition_by(
            self._partition_by, as_dict=True, include_key=self._include_key
        ).items():
            value = group_value[0] if isinstance(group_value, tuple) else group_value
            self._writer_for(str(value)).add(group_frame)

    def close(self) -> list[Path]:
        written_paths: list[Path] = []
        for value in sorted(self._writers):
            written_paths.extend(self._writers[value].close())
        return written_paths

    def _writer_for(self, value: str) -> FlatChunkWriter:
        writer = self._writers.get(value)
        if writer is None:
            partition_dir = self._output_dir / f"{self._label}={value}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            writer = FlatChunkWriter(
                output_dir=partition_dir,
                prefix="part",
                chunk_size=self._chunk_size,
                width=5,
            )
            self._writers[value] = writer
        return writer


def write_chunked_output_files(
    *,
    chunk_sources: list[Path],
    output_dir: Path,
    prefix: str,
    chunk_size: int | None,
    frame_transformer: Callable[[Path], pl.DataFrame],
) -> list[Path]:
    """Read-transform-write loop over FlatChunkWriter; see its docstring for
    the chunk_size=None semantics. Shared by both the canonical and shard
    stages, either directly on their real inputs or (for their staged-write
    durability pattern) on already-materialized scratch chunk files.
    """
    writer = FlatChunkWriter(
        output_dir=output_dir, prefix=prefix, chunk_size=chunk_size
    )
    for chunk_source in chunk_sources:
        writer.add(frame_transformer(chunk_source))
    return writer.close()


def write_partitioned_output_files(
    *,
    chunk_sources: list[Path],
    output_dir: Path,
    partition_by: str,
    chunk_size: int | None,
    frame_transformer: Callable[[Path], pl.DataFrame],
) -> list[Path]:
    """Reads and concatenates every chunk source (same read-everything-then-
    group approach cleanser_orchestrate.materialize_cleansed_merge already
    uses), then delegates to write_partitioned_output_files_from_frame.
    """
    frames = [frame_transformer(chunk_source) for chunk_source in chunk_sources]
    frames = [frame for frame in frames if frame.height > 0]
    if not frames:
        return []

    combined = pl.concat(frames, how="diagonal_relaxed")
    return write_partitioned_output_files_from_frame(
        frame=combined,
        output_dir=output_dir,
        partition_by=partition_by,
        chunk_size=chunk_size,
    )
