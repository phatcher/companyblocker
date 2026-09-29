from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import polars as pl

from .sharding_perf import ShardPerformance


def write_chunked_parquet(
    frame: pl.DataFrame,
    output_dir: Path,
    prefix: str,
    *,
    chunk_size: int = 100_000,
    progress: Callable[[str], None] | None = None,
    telemetry: ShardPerformance | None = None,
    emit_summary: bool = True,
) -> list[Path]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = frame.height
    if total_rows == 0:
        return []

    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    width = max(3, len(str(total_chunks)))
    paths: list[Path] = []
    owned_telemetry = telemetry is None
    active_telemetry = telemetry or ShardPerformance(prefix=prefix, progress=progress)

    for chunk_index, start in enumerate(range(0, total_rows, chunk_size), start=1):
        t_chunk_start = time.perf_counter()
        chunk = frame.slice(start, chunk_size)
        path = output_dir / f"{prefix}-{chunk_index:0{width}d}.parquet"
        chunk.write_parquet(path, compression="snappy")
        paths.append(path)

        elapsed_s = max(time.perf_counter() - t_chunk_start, 1e-9)
        active_telemetry.increment_kept_rows(chunk.height)
        active_telemetry.add_write_elapsed(elapsed_s)
        active_telemetry.record_chunk_write(chunk.height, elapsed_s, chunk_index)

    if owned_telemetry and emit_summary:
        active_telemetry.emit_summary()

    return paths
