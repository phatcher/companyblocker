"""Runtime pieces the Wikidata projection paths share: which reader opens the dump (`build_wikidata_reader_strategy`) and how progress is reported (`WikidataProgressEmitter`, `CallbackProgressEmitter`)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from acquisition.io_eta_helpers import (
    _READ_CHUNK_SIZE_BYTES,
    _open_eta_enabled_source,
    _resolve_source_read_settings,
)

WikidataLineIterator = Callable[..., Iterator[tuple[int, bytes]]]


@dataclass(frozen=True)
class WikidataReaderStrategy:
    source_path: Path
    reader_mode: str | None
    indexed_bzip2_parallelization: object | None
    read_chunk_size_bytes: int
    iter_content_lines_fn: WikidataLineIterator

    def open_source(self):
        return _open_eta_enabled_source(
            self.source_path,
            reader_mode=self.reader_mode,
            indexed_bzip2_parallelization=self.indexed_bzip2_parallelization,
        )

    def iter_content_lines(self, source_handle) -> Iterator[tuple[int, bytes]]:
        yield from self.iter_content_lines_fn(
            source_handle,
            read_chunk_size_bytes=self.read_chunk_size_bytes,
        )


@dataclass(frozen=True)
class CallbackProgressEmitter:
    callback: Callable[[str], None] | None

    def emit(self, message: str) -> None:
        if self.callback is not None:
            self.callback(message)


@dataclass(frozen=True)
class WikidataProgressEmitter:
    callback: Callable[[str], None] | None

    def emit(self, message: str) -> None:
        if self.callback is not None:
            self.callback(message)

    def emit_started(self, *, phase: str, source_name: str) -> None:
        self.emit(f"[wikidata] {phase} started: source={source_name}")

    def emit_scanning(self, *, phase: str) -> None:
        self.emit(f"[wikidata] {phase} scanning dump")

    def emit_early_stop_lines(
        self, *, phase: str, lines_processed: int, max_lines: int
    ) -> None:
        self.emit(
            f"[wikidata] {phase} early stop: lines_processed={lines_processed:,} "
            f"(max_lines={max_lines:,})"
        )

    def emit_early_stop_rows(
        self, *, phase: str, rows_written: int, max_companies: int
    ) -> None:
        self.emit(
            f"[wikidata] {phase} early stop: rows_written={rows_written:,} "
            f"(max_companies={max_companies:,})"
        )

    def emit_progress(
        self,
        *,
        phase: str,
        counts_text: str,
        trailing_text: str,
    ) -> None:
        self.emit(f"[wikidata] {phase} progress: {counts_text}, {trailing_text}")

    def emit_complete(
        self,
        *,
        phase: str,
        counts_text: str,
        trailing_text: str,
    ) -> None:
        self.emit(f"[wikidata] {phase} complete: {counts_text}, {trailing_text}")

    def emit_resume(self, *, action: str, detail_text: str) -> None:
        self.emit(f"[wikidata] streaming resume: {action} {detail_text}")

    def emit_phase_counter_progress(
        self,
        *,
        phase: str,
        lines_processed: int,
        secondary_count_label: str,
        secondary_count: int,
        compressed_progress_text: str,
        eta_text: str,
        line_rate: float,
        secondary_rate_label: str,
        secondary_rate: float,
    ) -> None:
        self._emit_rate_progress(
            phase=phase,
            counts_text=(
                f"lines_processed={lines_processed:,}, "
                f"{secondary_count_label}={secondary_count:,}"
            ),
            compressed_progress_text=compressed_progress_text,
            eta_text=eta_text,
            line_rate=line_rate,
            secondary_rate_label=secondary_rate_label,
            secondary_rate=secondary_rate,
        )

    def emit_prepare_complete(
        self,
        *,
        lines_processed: int,
        rows_written: int,
        eta_text: str,
        elapsed: float,
        line_rate: float,
        write_rate: float,
    ) -> None:
        self.emit_complete(
            phase="prepare projection",
            counts_text=f"lines_processed={lines_processed:,}, written={rows_written:,}",
            trailing_text=(
                f"{eta_text}, elapsed={elapsed:.1f}s, "
                f"line_rate={line_rate:,.0f}/s, write_rate={write_rate:,.0f}/s"
            ),
        )

    def _emit_rate_progress(
        self,
        *,
        phase: str,
        counts_text: str,
        compressed_progress_text: str,
        eta_text: str,
        line_rate: float,
        secondary_rate_label: str,
        secondary_rate: float,
    ) -> None:
        self.emit_progress(
            phase=phase,
            counts_text=counts_text,
            trailing_text=(
                f"{compressed_progress_text}, {eta_text}, "
                f"line_rate={line_rate:,.0f}/s, {secondary_rate_label}={secondary_rate:,.0f}/s"
            ),
        )

    def emit_filter_complete(
        self,
        *,
        lines_processed: int,
        lines_written: int,
        lines_skipped_no_candidate: int,
        compressed_progress_text: str,
        eta_text: str,
        elapsed: float,
        line_rate: float,
        company_rate: float,
    ) -> None:
        self.emit_complete(
            phase="Company filter",
            counts_text=(
                f"lines_processed={lines_processed:,}, company_found={lines_written:,}, "
                f"no_candidate_skipped={lines_skipped_no_candidate:,}"
            ),
            trailing_text=(
                f"{compressed_progress_text}, {eta_text}, elapsed={elapsed:.1f}s, "
                f"line_rate={line_rate:,.0f}/s, company_rate={company_rate:,.0f}/s"
            ),
        )

    def emit_stream_layout_migration(self, *, migrated_count: int) -> None:
        self.emit(
            f"[wikidata] streaming layout migration: moved {migrated_count:,} chunk(s) into partitioned directories"
        )

    def emit_stream_resume_replaying_chunk(
        self, *, replay_idx: int, replay_rows: int
    ) -> None:
        self.emit_resume(
            action="replaying chunk",
            detail_text=f"{replay_idx:06d} ({replay_rows:,} rows)",
        )

    def emit_stream_resume_restored_rows(
        self, *, restored_rows: int, completed_chunks: int
    ) -> None:
        self.emit_resume(
            action="restored",
            detail_text=f"{restored_rows:,} rows from {completed_chunks} completed chunk(s)",
        )

    def emit_stream_progress(
        self,
        *,
        lines_processed: int,
        company_count: int,
        line_rate: float,
        company_rate: float,
        company_share: float,
        compressed_progress_text: str,
        estimated_totals_text: str,
        eta_text: str,
        eta_stability: str,
    ) -> None:
        self.emit_progress(
            phase="streaming",
            counts_text=f"lines_processed={lines_processed:,}, company_found={company_count:,}",
            trailing_text=(
                f"line_rate={line_rate:,.0f}/s, company_rate={company_rate:,.0f}/s, "
                f"company_share={company_share:.2%}, {compressed_progress_text}, "
                f"{estimated_totals_text}, {eta_text}, eta_stability={eta_stability}"
            ),
        )

    def emit_stream_resume_seek(self, *, resume_uncompressed_offset: int) -> None:
        self.emit_resume(
            action="seeking source to uncompressed offset",
            detail_text=f"{resume_uncompressed_offset:,}",
        )

    def emit_stream_complete(
        self,
        *,
        lines_processed: int,
        company_count: int,
        lines_skipped_no_candidate: int,
        elapsed: float,
        line_rate: float,
        company_rate: float,
    ) -> None:
        self.emit_complete(
            phase="streaming",
            counts_text=(
                f"lines_processed={lines_processed:,}, company_found={company_count:,}, "
                f"no_candidate_skipped={lines_skipped_no_candidate:,}"
            ),
            trailing_text=(
                f"elapsed={elapsed:.1f}s, line_rate={line_rate:,.0f}/s, "
                f"company_rate={company_rate:,.0f}/s"
            ),
        )

    def emit_stream_filter_projection_header(self) -> None:
        self.emit("[wikidata] ============ STREAMED FILTER + PROJECTION ============")

    def emit_stream_single_scan_notice(self) -> None:
        self.emit("[wikidata] phase 1 and phase 2 executed in one source scan")

    def emit_optimization_complete(
        self, *, company_count: int, projected_count: int
    ) -> None:
        self.emit("[wikidata] ============ Optimization Complete ============")
        self.emit(f"[wikidata] Pass 1 company count: {company_count:,}")
        self.emit(f"[wikidata] Final projected companies: {projected_count:,}")


def build_wikidata_reader_strategy(
    *,
    source_path: Path,
    # Widened from `ResolvedReadOptions`: this only forwards the mapping to
    # `_resolve_source_read_settings`, which reads each field defensively, and
    # the pipeline hops in between carry it as a plain dict.
    resolved_read_options: Mapping[str, object] | None,
    iter_content_lines_fn: WikidataLineIterator,
    default_chunk_size_bytes: int = _READ_CHUNK_SIZE_BYTES,
) -> WikidataReaderStrategy:
    reader_mode, indexed_parallelization, read_chunk_size_bytes = (
        _resolve_source_read_settings(
            resolved_read_options,
            default_chunk_size_bytes=default_chunk_size_bytes,
        )
    )
    return WikidataReaderStrategy(
        source_path=source_path,
        reader_mode=reader_mode,
        indexed_bzip2_parallelization=indexed_parallelization,
        read_chunk_size_bytes=read_chunk_size_bytes,
        iter_content_lines_fn=iter_content_lines_fn,
    )
