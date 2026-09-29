from __future__ import annotations

import bz2
import gzip
import os
from collections.abc import Mapping
from pathlib import Path

import indexed_bzip2 as _indexed_bzip2
import orjson

_READ_CHUNK_SIZE_BYTES = 16 * 1024 * 1024

_BYTE_NEWLINE = b"\n"
_BYTE_LEFT_BRACKET = b"["
_BYTE_RIGHT_BRACKET = b"]"
_BYTE_COMMA = b","
_BYTE_P31_PROPERTY = b'"P31"'

_ETA_WARMUP_PROGRESS_FLOOR = 0.005
_ETA_WARMUP_TARGET_PROGRESS = 0.20
_ETA_BAND_MIN_WIDTH_SECONDS = 120.0
_ETA_BAND_SINGLE_ESTIMATE_PAD_RATIO = 0.10


class _TrackedSourceHandle:
    """Wraps a readable handle and optionally tracks compressed-byte cursor."""

    def __init__(self, handle, *, compressed_tell_handle=None) -> None:
        self._handle = handle
        self._compressed_tell_handle = compressed_tell_handle

    def __enter__(self):
        enter = getattr(self._handle, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        suppress = False
        exit_fn = getattr(self._handle, "__exit__", None)
        if callable(exit_fn):
            suppress = bool(exit_fn(exc_type, exc_val, exc_tb))
        self._close_compressed_handle()
        return suppress

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def compressed_tell(self) -> int | None:
        if self._compressed_tell_handle is None:
            return None
        tell = getattr(self._compressed_tell_handle, "tell", None)
        if not callable(tell):
            return None
        try:
            value = tell()
        except (OSError, ValueError):
            return None
        return int(value) if isinstance(value, int | float) else None

    def close(self) -> None:
        close_fn = getattr(self._handle, "close", None)
        if callable(close_fn):
            close_fn()
        self._close_compressed_handle()

    def _close_compressed_handle(self) -> None:
        if (
            self._compressed_tell_handle is None
            or self._compressed_tell_handle is self._handle
        ):
            return
        close_fn = getattr(self._compressed_tell_handle, "close", None)
        if callable(close_fn):
            close_fn()


def _compressed_bytes_read_from_handle(source_handle) -> int | None:
    tell_compressed = getattr(source_handle, "tell_compressed", None)
    if callable(tell_compressed):
        try:
            value = tell_compressed()
        except (OSError, ValueError):
            value = None
        if isinstance(value, int | float) and value >= 0:
            compressed_value = int(value)
            # indexed_bzip2 reports compressed position in bits.
            handle_obj = getattr(source_handle, "_handle", source_handle)
            handle_module = getattr(type(handle_obj), "__module__", "")
            if "indexed_bzip2" in handle_module:
                return compressed_value // 8
            return compressed_value

    compressed_tell = getattr(source_handle, "compressed_tell", None)
    if callable(compressed_tell):
        value = compressed_tell()
        if isinstance(value, int) and value >= 0:
            return value

    for candidate in (
        getattr(source_handle, "_fp", None),
        getattr(source_handle, "fileobj", None),
        getattr(getattr(source_handle, "_buffer", None), "raw", None),
    ):
        if candidate is None:
            continue
        tell = getattr(candidate, "tell", None)
        if not callable(tell):
            continue
        try:
            value = tell()
        except (OSError, ValueError):
            continue
        if isinstance(value, int | float) and value >= 0:
            return int(value)

    return None


def _json_loads(payload_text: bytes | str) -> dict[str, object] | None:
    try:
        payload = orjson.loads(payload_text)
    except orjson.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _json_dumps_line(payload: dict[str, object]) -> bytes:
    return orjson.dumps(payload) + _BYTE_NEWLINE


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "eta=unknown"

    total_seconds = max(0, round(seconds))
    if total_seconds < 60:
        return f"eta={total_seconds}s"

    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds_remainder = divmod(remainder, 60)
    if hours > 0:
        return f"eta={hours}h{minutes:02d}m"
    return f"eta={minutes}m{seconds_remainder:02d}s"


def _format_eta_coarse(seconds: float | None) -> str:
    if seconds is None:
        return "eta=unknown"

    total_minutes = max(1, round(max(0.0, seconds) / 60.0))
    hours, minutes = divmod(total_minutes, 60)
    if hours > 0:
        return f"eta={hours}h{minutes:02d}m"
    return f"eta={minutes}m"


def _format_eta_band_and_stability(
    *,
    global_eta_seconds: float | None,
    estimated_total_lines: int | None,
    lines_processed: int,
    recent_line_rate: float | None,
) -> tuple[str, str]:
    if global_eta_seconds is None:
        return "eta=unknown", "unknown"

    lower = max(0.0, global_eta_seconds)
    upper = lower

    if (
        estimated_total_lines is not None
        and estimated_total_lines > lines_processed
        and recent_line_rate is not None
        and recent_line_rate > 0
    ):
        remaining_lines = max(0, estimated_total_lines - lines_processed)
        recent_eta_seconds = remaining_lines / recent_line_rate
        lower = min(lower, recent_eta_seconds)
        upper = max(upper, recent_eta_seconds)
    else:
        pad = max(
            _ETA_BAND_MIN_WIDTH_SECONDS / 2.0,
            lower * _ETA_BAND_SINGLE_ESTIMATE_PAD_RATIO,
        )
        lower = max(0.0, lower - pad)
        upper = lower + (2.0 * pad)

    if (upper - lower) < _ETA_BAND_MIN_WIDTH_SECONDS:
        upper = lower + _ETA_BAND_MIN_WIDTH_SECONDS

    spread_ratio = (upper - lower) / max(upper, 1.0)
    if spread_ratio < 0.10:
        stability = "stable"
    elif spread_ratio < 0.25:
        stability = "moderate"
    else:
        stability = "volatile"

    lower_text = _format_eta_coarse(lower).split("=", 1)[1]
    upper_text = _format_eta_coarse(upper).split("=", 1)[1]
    return f"eta={lower_text}-{upper_text}", stability


def _estimate_stream_eta(
    *,
    source_path: Path,
    source_handle,
    lines_processed: int,
    elapsed_seconds: float,
) -> tuple[int, int, float | None] | None:
    if lines_processed <= 0 or elapsed_seconds <= 0:
        return None

    compressed_bytes_read = _compressed_bytes_read_from_handle(source_handle)
    if compressed_bytes_read is None:
        return None
    total_compressed_bytes = source_path.stat().st_size
    if compressed_bytes_read <= 0 or total_compressed_bytes <= 0:
        return None

    compressed_progress = min(compressed_bytes_read / total_compressed_bytes, 1.0)
    if compressed_progress <= 0:
        return None

    estimated_total_lines = max(
        lines_processed, round(lines_processed / compressed_progress)
    )
    remaining_lines = max(0, estimated_total_lines - lines_processed)
    line_rate = lines_processed / elapsed_seconds
    if line_rate <= 0:
        return None

    eta_seconds = remaining_lines / line_rate

    # Early compressed progress tends to under-estimate end-to-end runtime on
    # large dumps; apply a conservative warm-up multiplier until progress has
    # reached a representative fraction of the file.
    if compressed_progress < _ETA_WARMUP_TARGET_PROGRESS:
        effective_progress = max(compressed_progress, _ETA_WARMUP_PROGRESS_FLOOR)
        eta_seconds *= _ETA_WARMUP_TARGET_PROGRESS / effective_progress

    return estimated_total_lines, compressed_bytes_read, eta_seconds


def _iter_raw_lines_from_handle(
    handle, *, read_chunk_size_bytes: int = _READ_CHUNK_SIZE_BYTES
):
    if read_chunk_size_bytes <= 0:
        raise ValueError("read_chunk_size_bytes must be greater than zero")

    line_number = 0
    carry = b""

    while True:
        chunk = handle.read(read_chunk_size_bytes)
        if not chunk:
            break

        if carry:
            chunk = carry + chunk

        start = 0
        while True:
            newline_index = chunk.find(_BYTE_NEWLINE, start)
            if newline_index < 0:
                break
            line_number += 1
            yield line_number, chunk[start:newline_index]
            start = newline_index + 1

        carry = chunk[start:]

    if carry:
        line_number += 1
        yield line_number, carry


def _get_optimal_bzip2_threads() -> int:
    """Calculate optimal thread count for indexed_bzip2.

    - Tries to use cgroup allocation (Linux/HPC/Docker)
    - Falls back to cpu_count
    - Caps at 24 to prevent block scanner bottleneck
    - Minimum 2 threads if multi-core available
    """
    try:
        cores = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        cores = os.cpu_count() or 1

    return max(2, min(cores, 24))


def _resolve_indexed_bzip2_parallelization(value: object | None) -> int:
    if value is None or value == "auto":
        return _get_optimal_bzip2_threads()
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(
        "indexed_bzip2_parallelization must be 'auto' or a positive integer"
    )


def _resolve_source_read_settings(
    # `Mapping`, not `dict`: callers hand this a `ResolvedReadOptions`
    # TypedDict, which is not a `dict[str, object]` to a type checker. Every
    # field below is `isinstance`-guarded anyway, so any mapping will do.
    resolved_read_options: Mapping[str, object] | None,
    *,
    default_chunk_size_bytes: int = _READ_CHUNK_SIZE_BYTES,
) -> tuple[str | None, object | None, int]:
    options = resolved_read_options or {}
    configured_reader_mode = options.get("reader_mode")
    reader_mode = (
        configured_reader_mode if isinstance(configured_reader_mode, str) else None
    )
    indexed_parallelization = options.get("indexed_bzip2_parallelization")
    configured_chunk_size = options.get("chunk_size_bytes")
    chunk_size_bytes = (
        configured_chunk_size
        if isinstance(configured_chunk_size, int) and configured_chunk_size > 0
        else default_chunk_size_bytes
    )
    return reader_mode, indexed_parallelization, chunk_size_bytes


def _open_eta_enabled_source(
    path: Path,
    *,
    reader_mode: str | None = None,
    indexed_bzip2_parallelization: object | None = None,
):
    """Open a source file for reading with compression cursor tracking support."""
    # Ownership of each raw handle transfers immediately to the returned
    # _TrackedSourceHandle (itself a context manager), so a `with` block here
    # would close the file before the caller ever reads from it.
    if path.suffix == ".gz":
        handle: object = gzip.open(path, "rb")  # noqa: SIM115
        return _TrackedSourceHandle(
            handle, compressed_tell_handle=getattr(handle, "fileobj", None)
        )
    if path.suffix == ".bz2":
        normalized_reader_mode = (reader_mode or "chunked_indexed").strip().lower()
        if normalized_reader_mode == "line_iter":
            handle = bz2.open(path, "rb")  # noqa: SIM115
            return _TrackedSourceHandle(handle)

        parallelization = _resolve_indexed_bzip2_parallelization(
            indexed_bzip2_parallelization
        )
        handle = _indexed_bzip2.open(str(path), parallelization=parallelization)
        return _TrackedSourceHandle(handle)
    handle = open(path, "rb")  # noqa: SIM115
    return _TrackedSourceHandle(handle, compressed_tell_handle=handle)
