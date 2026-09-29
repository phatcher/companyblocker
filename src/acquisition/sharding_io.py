from __future__ import annotations

import csv
import gzip
import io
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import polars as pl

from .io_eta_helpers import (
    _iter_raw_lines_from_handle,
    _open_eta_enabled_source,
    _resolve_source_read_settings,
)


def _infer_format_from_member_name(member_name: str) -> str | None:
    lowered = member_name.strip().lower()
    if lowered.endswith((".jsonl.bz2", ".ndjson.bz2")):
        return "jsonl"
    if lowered.endswith((".jsonl", ".ndjson")):
        return "jsonl"
    if lowered.endswith(".parquet"):
        return "parquet"
    if lowered.endswith(".csv"):
        return "csv"
    if lowered.endswith(".json"):
        return "json"
    if lowered.endswith(".zip"):
        return "zip"
    return None


def detect_csv_separator_from_text(sample: str) -> str:
    candidates = (",", ";", "\t", "|")
    if not sample.strip():
        return ","

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="".join(candidates))
        if dialect.delimiter in candidates:
            return dialect.delimiter
    except csv.Error:
        pass

    for raw_line in sample.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        separator_counts = {sep: line.count(sep) for sep in candidates}
        best = max(separator_counts, key=separator_counts.__getitem__)
        if separator_counts[best] == 0:
            return ","
        return best

    return ","


def detect_csv_separator(source_path: Path) -> str:
    sample = source_path.read_text(encoding="utf-8-sig", errors="replace")[:8192]
    return detect_csv_separator_from_text(sample)


def read_csv_bytes_with_detected_separator(payload: bytes) -> pl.DataFrame:
    text_sample = payload.decode("utf-8-sig", errors="replace")[:8192]
    separator = detect_csv_separator_from_text(text_sample)
    return pl.read_csv(io.BytesIO(payload), separator=separator)


def read_csv_with_detected_separator(source_path: Path) -> pl.DataFrame:
    separator = detect_csv_separator(source_path)
    frame = pl.read_csv(source_path, separator=separator)
    rename_map = {
        column_name: column_name.strip()
        for column_name in frame.columns
        if column_name.strip() != column_name
    }
    return frame.rename(rename_map) if rename_map else frame


def _iter_clean_jsonl_lines_from_binary_handle(
    binary_handle, *, chunk_size_bytes: int
) -> Iterator[str]:
    for _line_number, raw_line in _iter_raw_lines_from_handle(
        binary_handle, read_chunk_size_bytes=chunk_size_bytes
    ):
        line = raw_line.strip()
        if not line:
            continue
        yield line.decode("utf-8", errors="replace")


@contextmanager
def _wrap_binary_as_text(
    binary_handle,
    *,
    encoding: str,
    errors: str,
):
    text_handle = io.TextIOWrapper(binary_handle, encoding=encoding, errors=errors)
    try:
        yield text_handle
    finally:
        try:
            text_handle.detach()
        except (ValueError, OSError):
            pass


@contextmanager
def _open_text_source(
    source_path: Path,
    *,
    encoding: str = "utf-8",
    errors: str = "replace",
    reader_mode: str | None = None,
    indexed_bzip2_parallelization: object | None = None,
):
    suffix = source_path.suffix.lower()
    if suffix == ".gz":
        with (
            gzip.open(source_path, "rb") as binary_handle,
            _wrap_binary_as_text(
                binary_handle, encoding=encoding, errors=errors
            ) as text_handle,
        ):
            yield text_handle
        return

    if suffix == ".bz2":
        with (
            _open_eta_enabled_source(
                source_path,
                reader_mode=reader_mode,
                indexed_bzip2_parallelization=indexed_bzip2_parallelization,
            ) as binary_handle,
            _wrap_binary_as_text(
                binary_handle, encoding=encoding, errors=errors
            ) as text_handle,
        ):
            yield text_handle
        return

    with source_path.open(mode="rt", encoding=encoding, errors=errors) as text_handle:
        yield text_handle


def read_ndjson_with_optional_bz2(source_path: Path) -> pl.DataFrame:
    if source_path.suffix.lower() not in {".bz2", ".gz"}:
        return pl.read_ndjson(source_path, infer_schema_length=None)

    with _open_text_source(source_path) as handle:
        return pl.read_ndjson(handle, infer_schema_length=None)


def _yield_ndjson_batches_from_lines(
    raw_lines: Iterator[str], *, batch_size: int
) -> Iterator[pl.DataFrame]:
    rows_buffer: list[str] = []
    for raw_line in raw_lines:
        line = raw_line.strip()
        if not line:
            continue
        rows_buffer.append(line)
        if len(rows_buffer) >= batch_size:
            payload = ("\n".join(rows_buffer) + "\n").encode("utf-8")
            yield pl.read_ndjson(io.BytesIO(payload), infer_schema_length=None)
            rows_buffer = []

    if rows_buffer:
        payload = ("\n".join(rows_buffer) + "\n").encode("utf-8")
        yield pl.read_ndjson(io.BytesIO(payload), infer_schema_length=None)


def _iter_jsonl_batches_from_directory(
    source_path: Path, *, batch_size: int
) -> Iterator[pl.DataFrame]:
    chunk_paths = sorted(
        path for path in source_path.rglob("*.jsonl") if path.is_file()
    )

    def _iter_chunk_lines() -> Iterator[str]:
        for chunk_path in chunk_paths:
            with chunk_path.open(
                mode="rt", encoding="utf-8", errors="replace"
            ) as handle:
                yield from handle

    yield from _yield_ndjson_batches_from_lines(
        _iter_chunk_lines(), batch_size=batch_size
    )


def _iter_jsonl_batches_from_file(
    source_path: Path,
    *,
    batch_size: int,
    reader_mode: str | None,
    indexed_bzip2_parallelization: object | None,
    chunk_size_bytes: int,
) -> Iterator[pl.DataFrame]:
    if reader_mode in {"chunked", "chunked_indexed"}:
        with _open_eta_enabled_source(
            source_path,
            reader_mode=reader_mode,
            indexed_bzip2_parallelization=indexed_bzip2_parallelization,
        ) as binary_handle:
            yield from _yield_ndjson_batches_from_lines(
                _iter_clean_jsonl_lines_from_binary_handle(
                    binary_handle,
                    chunk_size_bytes=chunk_size_bytes,
                ),
                batch_size=batch_size,
            )
        return

    with _open_text_source(
        source_path,
        reader_mode=reader_mode,
        indexed_bzip2_parallelization=indexed_bzip2_parallelization,
    ) as handle:
        yield from _yield_ndjson_batches_from_lines(handle, batch_size=batch_size)


def _extract_supported_archive_member(
    archive: zipfile.ZipFile,
    *,
    source_path: Path,
    temp_dir: str,
) -> tuple[Path, str]:
    members = [member for member in archive.infolist() if not member.is_dir()]
    if not members:
        raise ValueError(f"No files found inside archive: {source_path}")

    selected: tuple[zipfile.ZipInfo, str] | None = None
    for member in members:
        inferred = _infer_format_from_member_name(member.filename)
        if inferred is not None:
            selected = (member, inferred)
            break

    if selected is None:
        member_names = ", ".join(member.filename for member in members)
        raise ValueError(
            f"No supported files found inside archive: {source_path} ({member_names})"
        )

    selected_member, selected_format = selected
    target_name = Path(selected_member.filename).name
    if not target_name:
        raise ValueError(f"Archive member path is invalid: {selected_member.filename}")

    extracted_path = Path(temp_dir) / target_name
    with (
        archive.open(selected_member) as source_stream,
        extracted_path.open("wb") as output_stream,
    ):
        output_stream.write(source_stream.read())

    return extracted_path, selected_format


def read_source_frame(source_path: Path, file_format: str) -> pl.DataFrame:
    format_name = file_format.lower()

    if format_name == "parquet":
        return pl.read_parquet(source_path)

    if format_name == "csv":
        return read_csv_with_detected_separator(source_path)

    if format_name == "json":
        return pl.read_json(source_path)

    if format_name == "jsonl":
        return read_ndjson_with_optional_bz2(source_path)

    if format_name == "zip":
        with (
            zipfile.ZipFile(source_path) as archive,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            extracted_path, selected_format = _extract_supported_archive_member(
                archive,
                source_path=source_path,
                temp_dir=temp_dir,
            )
            return read_source_frame(extracted_path, selected_format)

    raise ValueError(f"Unsupported source format '{file_format}' for {source_path}")


def read_source_batches(
    source_path: Path,
    file_format: str,
    *,
    batch_size: int = 100_000,
    resolved_read_options: dict[str, object] | None = None,
) -> Iterator[pl.DataFrame]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    format_name = file_format.lower()
    reader_mode, configured_parallelization, chunk_size_bytes = (
        _resolve_source_read_settings(
            resolved_read_options,
        )
    )

    if format_name == "parquet":
        # Pure-Polars slice pushdown against parquet row-group metadata -- no
        # pyarrow needed. Cost is proportional to batch_size, not file size
        # (benchmarked: ~10ms per 100K-row slice regardless of offset into a
        # 2M-row file), so this reads no more eagerly than a real streaming
        # reader would.
        lazy_frame = pl.scan_parquet(source_path)
        offset = 0
        while True:
            chunk = lazy_frame.slice(offset, batch_size).collect()
            if chunk.height == 0:
                break
            yield chunk
            offset += batch_size
        return

    if format_name == "jsonl":
        if source_path.is_dir():
            yield from _iter_jsonl_batches_from_directory(
                source_path, batch_size=batch_size
            )
            return

        yield from _iter_jsonl_batches_from_file(
            source_path,
            batch_size=batch_size,
            reader_mode=reader_mode,
            indexed_bzip2_parallelization=configured_parallelization,
            chunk_size_bytes=chunk_size_bytes,
        )
        return

    if format_name == "zip":
        with (
            zipfile.ZipFile(source_path) as archive,
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            extracted_path, selected_format = _extract_supported_archive_member(
                archive,
                source_path=source_path,
                temp_dir=temp_dir,
            )
            yield from read_source_batches(
                extracted_path, selected_format, batch_size=batch_size
            )
            return

    yield read_source_frame(source_path, file_format)
