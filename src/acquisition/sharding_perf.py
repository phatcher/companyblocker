from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl


@dataclass(slots=True)
class ShardPerformance:
    prefix: str
    progress: Callable[[str], None] | None = None
    started_at: float = field(default_factory=time.perf_counter)
    input_rows: int = 0
    kept_rows: int = 0
    filtered_rows: int = 0
    phase_elapsed_s: dict[str, float] = field(default_factory=dict)
    chunk_throughputs: list[float] = field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return self.kept_rows

    @property
    def filtered_out_rows(self) -> int:
        return self.filtered_rows

    @property
    def extract_elapsed_s(self) -> float:
        return self.phase_elapsed_s.get("extract", 0.0)

    @property
    def write_elapsed_s(self) -> float:
        return self.phase_elapsed_s.get("write", 0.0)

    def add_phase_elapsed(self, phase: str, elapsed_s: float) -> None:
        self.phase_elapsed_s[phase] = self.phase_elapsed_s.get(phase, 0.0) + elapsed_s

    def add_extract_elapsed(self, elapsed_s: float) -> None:
        self.add_phase_elapsed("extract", elapsed_s)

    def add_write_elapsed(self, elapsed_s: float) -> None:
        self.add_phase_elapsed("write", elapsed_s)

    def increment_input_rows(self, count: int = 1) -> None:
        self.input_rows += count

    def increment_kept_rows(self, count: int = 1) -> None:
        self.kept_rows += count

    def increment_total_rows(self, count: int = 1) -> None:
        self.increment_kept_rows(count)

    def increment_filtered_rows(self, count: int = 1) -> None:
        self.filtered_rows += count

    def record_chunk_write(
        self, row_count: int, elapsed_s: float, chunk_index: int
    ) -> None:
        rows_per_sec = row_count / max(elapsed_s, 1e-9)
        self.chunk_throughputs.append(rows_per_sec)

        if self.progress is None:
            return

        self.progress(
            f"[{self.prefix}] chunk {chunk_index}: "
            f"{row_count:,} rows in {elapsed_s:.2f}s ({rows_per_sec:,.0f} rows/s)"
        )

    def record_chunk(
        self, row_count: int, chunk_started_at: float | None, chunk_index: int
    ) -> None:
        elapsed_s = max(
            time.perf_counter() - (chunk_started_at or time.perf_counter()), 1e-9
        )
        self.record_chunk_write(row_count, elapsed_s, chunk_index)

    def emit_summary(self) -> None:
        if self.progress is None:
            return

        if self.chunk_throughputs:
            self.progress(
                f"[{self.prefix}] shard throughput summary - "
                f"min: {min(self.chunk_throughputs):,.0f} rows/s, "
                f"median: {statistics.median(self.chunk_throughputs):,.0f} rows/s, "
                f"max: {max(self.chunk_throughputs):,.0f} rows/s"
            )

        if self.kept_rows == 0:
            return

        total_elapsed_s = max(time.perf_counter() - self.started_at, 1e-9)
        total_rows_per_sec = self.kept_rows / total_elapsed_s
        known_phase_elapsed_s = sum(self.phase_elapsed_s.values())
        other_elapsed_s = max(total_elapsed_s - known_phase_elapsed_s, 0.0)

        self.progress(
            f"[{self.prefix}] shard end-to-end throughput - "
            f"{self.kept_rows:,} rows in {total_elapsed_s:.2f}s ({total_rows_per_sec:,.0f} rows/s)"
        )
        phase_parts = [
            f"{phase}: {elapsed_s:.2f}s"
            for phase, elapsed_s in sorted(self.phase_elapsed_s.items())
        ]
        phase_parts.append(f"other: {other_elapsed_s:.2f}s")
        self.progress(
            f"[{self.prefix}] shard phase summary - " + ", ".join(phase_parts)
        )
        if self.filtered_rows:
            self.progress(
                f"[{self.prefix}] shard country filter excluded {self.filtered_rows:,} row(s)"
            )


def write_sidecar(
    *,
    sidecar_enabled: bool,
    output_dir: Path,
    prefix: str,
    split_main_sidecar: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]
    | None,
    chunk_index: int,
    sidecar_paths: list[Path],
    chunk_frame: pl.DataFrame,
    chunk_path: Path,
) -> None:
    if not sidecar_enabled:
        chunk_frame.write_parquet(chunk_path, compression="snappy")
        return

    if split_main_sidecar is None:
        raise ValueError("split_main_sidecar is required when sidecar_enabled is true.")

    main_frame, sidecar_frame = split_main_sidecar(chunk_frame)
    main_frame.write_parquet(chunk_path, compression="snappy")
    sidecar_path = output_dir / f"{prefix}-sidecar-{chunk_index:03d}.parquet"
    sidecar_frame.write_parquet(sidecar_path, compression="snappy")
    sidecar_paths.append(sidecar_path)


def write_shard_chunk(
    *,
    rows: list[dict[str, str | None]],
    output_dir: Path,
    prefix: str,
    chunk_index: int,
    sidecar_enabled: bool,
    split_main_sidecar: Callable[[pl.DataFrame], tuple[pl.DataFrame, pl.DataFrame]]
    | None,
    materialize_system_uri: Callable[[pl.DataFrame], pl.DataFrame] | None,
    written_paths: list[Path],
    sidecar_paths: list[Path],
    perf: ShardPerformance,
    chunk_started_at: float | None,
) -> None:
    if not rows:
        return

    write_started_at = time.perf_counter()
    chunk_frame = pl.DataFrame(rows, infer_schema_length=None)
    if materialize_system_uri is not None:
        chunk_frame = materialize_system_uri(chunk_frame)

    chunk_path = output_dir / f"{prefix}-{chunk_index:03d}.parquet"
    write_sidecar(
        sidecar_enabled=sidecar_enabled,
        output_dir=output_dir,
        prefix=prefix,
        split_main_sidecar=split_main_sidecar,
        chunk_index=chunk_index,
        sidecar_paths=sidecar_paths,
        chunk_frame=chunk_frame,
        chunk_path=chunk_path,
    )

    perf.add_write_elapsed(time.perf_counter() - write_started_at)
    written_paths.append(chunk_path)
    perf.record_chunk(len(rows), chunk_started_at, chunk_index)
