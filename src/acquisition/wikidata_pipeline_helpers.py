from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from acquisition.io_eta_helpers import (
    _BYTE_COMMA,
    _BYTE_LEFT_BRACKET,
    _BYTE_P31_PROPERTY,
    _BYTE_RIGHT_BRACKET,
    _READ_CHUNK_SIZE_BYTES,
    _estimate_stream_eta,
    _format_eta,
    _format_eta_band_and_stability,
    _iter_raw_lines_from_handle,
    _open_eta_enabled_source,
)
from acquisition.wikidata_projection_helpers import (
    _COMPANY_LIKE_INSTANCE_OF_QID_PATTERNS,
    _project_wikidata_company_record_line,
)
from acquisition.wikidata_runtime import (
    WikidataProgressEmitter,
    build_wikidata_reader_strategy,
)

_WIKIDATA_CHUNK_PARTITION_SIZE = 1000
_WIKIDATA_CHUNK_DIR_NAME = "wikidata-companies.chunks"
_WIKIDATA_CHUNK_PREFIX = "wikidata-companies-part-"
_WIKIDATA_RECENT_RATE_WINDOW_SECONDS = 600.0
_WIKIDATA_PREPARE_ROLLOVER_BYTES = 2 * 1024 * 1024 * 1024


class _BinaryWriteHandle(Protocol):
    def write(self, data: bytes) -> int: ...
    def close(self) -> None: ...


def _resolve_wikidata_eta_progress(
    *,
    source_path: Path,
    source_handle: object,
    source_size: int,
    lines_processed: int,
    elapsed_seconds: float,
    company_count: int | None = None,
    company_share: float | None = None,
) -> tuple[str, str, str | None, int | None, float | None]:
    eta_details = _estimate_stream_eta(
        source_path=source_path,
        source_handle=source_handle,
        lines_processed=lines_processed,
        elapsed_seconds=elapsed_seconds,
    )
    if eta_details is None:
        estimated_totals_text = (
            "estimated_total_companies=unknown" if company_count is not None else None
        )
        return (
            "compressed_bytes=unknown",
            "eta=unknown",
            estimated_totals_text,
            None,
            None,
        )

    estimated_total_lines, compressed_bytes_read, eta_seconds = eta_details
    compressed_progress_text = (
        f"compressed_bytes={compressed_bytes_read:,}/{source_size:,}, "
        f"estimated_total_lines={estimated_total_lines:,}"
    )
    eta_text = _format_eta(eta_seconds)

    if company_count is None:
        return (
            compressed_progress_text,
            eta_text,
            None,
            estimated_total_lines,
            eta_seconds,
        )

    share = 0.0 if company_share is None else company_share
    estimated_total_companies = max(company_count, round(share * estimated_total_lines))
    return (
        compressed_progress_text,
        eta_text,
        f"estimated_total_companies={estimated_total_companies:,}",
        estimated_total_lines,
        eta_seconds,
    )


def _should_emit_wikidata_progress(
    *,
    lines_processed: int,
    progress_every_lines: int,
    progress_every_seconds: float,
    since_last_progress: float,
) -> bool:
    return (
        progress_every_lines > 0 and lines_processed % progress_every_lines == 0
    ) or (progress_every_seconds > 0 and since_last_progress >= progress_every_seconds)


def _resolve_wikidata_progress_rates(
    *,
    lines_processed: int,
    company_count: int,
    elapsed_seconds: float,
) -> tuple[float, float]:
    if elapsed_seconds <= 0:
        return 0.0, 0.0
    return lines_processed / elapsed_seconds, company_count / elapsed_seconds


def _iter_wikidata_content_lines_from_handle(
    handle, *, read_chunk_size_bytes: int = _READ_CHUNK_SIZE_BYTES
):
    for line_number, raw_line in _iter_raw_lines_from_handle(
        handle,
        read_chunk_size_bytes=read_chunk_size_bytes,
    ):
        line = raw_line.strip()
        if not line or line == _BYTE_LEFT_BRACKET or line == _BYTE_RIGHT_BRACKET:
            continue
        if line.endswith(_BYTE_COMMA):
            line = line[:-1]
        if not line:
            continue
        yield line_number, line


def iter_clean_wikidata_json_lines(source_path: Path):
    with _open_eta_enabled_source(source_path) as handle:
        for _, line in _iter_wikidata_content_lines_from_handle(handle):
            yield line.decode("utf-8", errors="replace")


def _is_wikidata_company_candidate_raw_line(raw_line: bytes) -> bool:
    """Fast but incomplete: only matches the 6 hardcoded root instance-of QIDs.

    Misses any entity that is a company only via a P1454 legal-form claim
    (e.g. "GmbH", "S.A.") resolving into the p279 subclass closure rather than
    a direct P31 root match -- see
    `_make_wikidata_company_candidate_raw_line_with_legal_form` for the filter
    that covers that case too.
    """
    if _BYTE_P31_PROPERTY not in raw_line:
        return False
    return any(map(raw_line.__contains__, _COMPANY_LIKE_INSTANCE_OF_QID_PATTERNS))


_BYTE_P1454_PROPERTY = b'"P1454"'
_QID_TOKEN_RE = re.compile(rb'"(Q\d+)"')


def _raw_line_qid_tokens(raw_line: bytes) -> Iterator[str]:
    for match in _QID_TOKEN_RE.finditer(raw_line):
        yield match.group(1).decode("ascii")


def _make_wikidata_company_candidate_raw_line_with_legal_form(
    legal_form_qids: frozenset[str],
) -> Callable[[bytes], bool]:
    """Superset of `_is_wikidata_company_candidate_raw_line` that also admits
    entities whose only company signal is a P1454 legal-form claim resolving into
    the p279 subclass-of-company-legal-form closure (~49K QIDs as of 2026-07-16).

    A regex alternation over ~49K literal QIDs is exactly the "large regex"
    scaling problem `company_cleanse` hit with company-type suffix matching
    (see its rejected trie-prefilter note) -- Python's `re` backtracks through
    alternatives rather than building a shared-prefix automaton, so a single
    compiled pattern with tens of thousands of alternatives scales badly. A
    hand-built trie is not needed here either: unlike overlapping multi-token
    name suffixes, QIDs are single fixed-format tokens with no shared-prefix
    structure to exploit, so a single small extraction pattern (one compiled
    regex, not one per QID) plus an O(1) hash-set membership check gets the same
    "avoid the large pattern set" result more simply -- mirroring the Rust
    implementation's `HashSet<u32>` lookup.

    This is still a permissive prefilter (it does not confirm the matched QID is
    actually the P1454 claim's value rather than some other property's value on
    the same line), so pass 2 must make the authoritative decision -- see
    `_project_wikidata_company_record_line`'s `legal_form_qids` argument.
    """

    def matcher(raw_line: bytes) -> bool:
        if _BYTE_P31_PROPERTY in raw_line and any(
            map(raw_line.__contains__, _COMPANY_LIKE_INSTANCE_OF_QID_PATTERNS)
        ):
            return True
        if _BYTE_P1454_PROPERTY not in raw_line:
            return False
        return any(token in legal_form_qids for token in _raw_line_qid_tokens(raw_line))

    return matcher


def _parse_wikidata_chunk_index(*, chunk_prefix: str, path: Path) -> int | None:
    match = re.fullmatch(rf"{re.escape(chunk_prefix)}(\d+)\.jsonl", path.name)
    if not match:
        return None
    return int(match.group(1))


def _iter_wikidata_chunk_files(
    *, chunk_dir: Path, chunk_prefix: str
) -> list[tuple[int, Path]]:
    indexed_paths: list[tuple[int, Path]] = []
    for path in chunk_dir.rglob(f"{chunk_prefix}*.jsonl"):
        if not path.is_file():
            continue
        parsed = _parse_wikidata_chunk_index(chunk_prefix=chunk_prefix, path=path)
        if parsed is None:
            continue
        indexed_paths.append((parsed, path))
    indexed_paths.sort(key=lambda item: item[0])
    return indexed_paths


def _resolve_wikidata_chunk_partition_dir(*, chunk_dir: Path, chunk_index: int) -> Path:
    partition_start = (
        chunk_index // _WIKIDATA_CHUNK_PARTITION_SIZE
    ) * _WIKIDATA_CHUNK_PARTITION_SIZE
    partition_end = partition_start + (_WIKIDATA_CHUNK_PARTITION_SIZE - 1)
    return chunk_dir / f"{partition_start:06d}-{partition_end:06d}"


def _migrate_wikidata_flat_chunks_to_partitioned_layout(
    *,
    chunk_dir: Path,
    chunk_prefix: str,
    emitter: WikidataProgressEmitter,
) -> None:
    migrated_count = 0
    for path in sorted(chunk_dir.glob(f"{chunk_prefix}*.jsonl")):
        if not path.is_file():
            continue
        parsed = _parse_wikidata_chunk_index(chunk_prefix=chunk_prefix, path=path)
        if parsed is None:
            continue
        partition_dir = _resolve_wikidata_chunk_partition_dir(
            chunk_dir=chunk_dir, chunk_index=parsed
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        target_path = partition_dir / path.name
        if target_path.exists():
            if (
                target_path.stat().st_size == path.stat().st_size
                and target_path.read_bytes() == path.read_bytes()
            ):
                path.unlink()
                continue
            raise FileExistsError(
                "cannot migrate chunk due to non-identical existing target "
                f"{target_path.as_posix()}"
            )
        path.replace(target_path)
        migrated_count += 1

    if migrated_count > 0:
        emitter.emit_stream_layout_migration(migrated_count=migrated_count)


def _count_non_empty_jsonl_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _recover_wikidata_temp_chunks(
    *,
    chunk_dir: Path,
    chunk_prefix: str,
) -> None:
    for temp_path in sorted(chunk_dir.rglob(f"{chunk_prefix}*.tmp")):
        if not temp_path.is_file():
            continue
        parsed = _parse_wikidata_chunk_index(chunk_prefix=chunk_prefix, path=temp_path)
        if parsed is None:
            continue
        if temp_path.stat().st_size <= 0:
            temp_path.unlink(missing_ok=True)
            continue

        partition_dir = _resolve_wikidata_chunk_partition_dir(
            chunk_dir=chunk_dir,
            chunk_index=parsed,
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        recovered_path = partition_dir / f"{chunk_prefix}{parsed:06d}.jsonl"

        if recovered_path.exists() and recovered_path.stat().st_size > 0:
            temp_path.unlink(missing_ok=True)
            continue

        temp_path.replace(recovered_path)


def _prepare_wikidata_stream_chunks(
    *,
    chunk_dir: Path,
    chunk_prefix: str,
    stream_resume: bool,
    chunk_byte_limit: int,
    emitter: WikidataProgressEmitter,
) -> tuple[int, int, int, int]:
    resume_uncompressed_offset = 0
    projection_chunk_index = 1
    company_count = 0
    projected_count = 0

    chunk_dir.mkdir(parents=True, exist_ok=True)
    _recover_wikidata_temp_chunks(
        chunk_dir=chunk_dir,
        chunk_prefix=chunk_prefix,
    )
    _migrate_wikidata_flat_chunks_to_partitioned_layout(
        chunk_dir=chunk_dir,
        chunk_prefix=chunk_prefix,
        emitter=emitter,
    )
    existing_chunks = _iter_wikidata_chunk_files(
        chunk_dir=chunk_dir, chunk_prefix=chunk_prefix
    )

    if stream_resume and existing_chunks:
        existing_chunks.sort(key=lambda item: item[0])
        # Replay the last completed chunk to safely recover from interruptions.
        replay_idx, replay_path = existing_chunks[-1]
        replay_rows = _count_non_empty_jsonl_rows(replay_path)
        replay_path.unlink()
        emitter.emit_stream_resume_replaying_chunk(
            replay_idx=replay_idx, replay_rows=replay_rows
        )
        existing_chunks = existing_chunks[:-1]

        if existing_chunks:
            resume_uncompressed_offset = existing_chunks[-1][0] * chunk_byte_limit
            projection_chunk_index = existing_chunks[-1][0] + 1
            restored_rows = 0
            for _, existing_path in existing_chunks:
                restored_rows += _count_non_empty_jsonl_rows(existing_path)
            company_count = restored_rows
            projected_count = restored_rows
            emitter.emit_stream_resume_restored_rows(
                restored_rows=restored_rows,
                completed_chunks=len(existing_chunks),
            )
    elif existing_chunks:
        for _, path in existing_chunks:
            path.unlink()

    return (
        resume_uncompressed_offset,
        projection_chunk_index,
        company_count,
        projected_count,
    )


def _emit_wikidata_stream_progress(
    *,
    source_path: Path,
    source_handle: object,
    source_size: int,
    lines_processed: int,
    company_count: int,
    started: float,
    emitter: WikidataProgressEmitter,
    recent_line_rate: float | None,
) -> None:
    elapsed = max(time.perf_counter() - started, 1e-9)
    rate, company_rate = _resolve_wikidata_progress_rates(
        lines_processed=lines_processed,
        company_count=company_count,
        elapsed_seconds=elapsed,
    )
    company_share = company_count / lines_processed if lines_processed > 0 else 0.0
    (
        compressed_progress_text,
        eta_text,
        estimated_totals_text,
        estimated_total_lines,
        eta_seconds,
    ) = _resolve_wikidata_eta_progress(
        source_path=source_path,
        source_handle=source_handle,
        source_size=source_size,
        lines_processed=lines_processed,
        elapsed_seconds=elapsed,
        company_count=company_count,
        company_share=company_share,
    )
    if estimated_totals_text is None:
        estimated_totals_text = "estimated_total_companies=unknown"

    eta_band_text, eta_stability = _format_eta_band_and_stability(
        global_eta_seconds=eta_seconds,
        estimated_total_lines=estimated_total_lines,
        lines_processed=lines_processed,
        recent_line_rate=recent_line_rate,
    )

    emitter.emit_stream_progress(
        lines_processed=lines_processed,
        company_count=company_count,
        line_rate=rate,
        company_rate=company_rate,
        company_share=company_share,
        compressed_progress_text=compressed_progress_text,
        estimated_totals_text=estimated_totals_text,
        eta_text=eta_band_text if eta_seconds is not None else eta_text,
        eta_stability=eta_stability,
    )


def _close_wikidata_projection_outputs(
    *,
    projection_chunk_handle,
    projection_chunk_temp_path: Path | None,
) -> None:
    if projection_chunk_handle is not None:
        projection_chunk_handle.close()
        # Keep partial temp chunks so stream-resume can recover and replay the
        # final in-flight chunk after interruption.


def _start_wikidata_projection_chunk(
    *,
    projection_chunk_handle: _BinaryWriteHandle | None,
    projection_chunk_temp_path: Path | None,
    chunk_dir: Path,
    chunk_prefix: str,
    projection_chunk_index: int,
    current_uncompressed_offset: int,
) -> tuple[_BinaryWriteHandle, Path | None, int, int]:
    if projection_chunk_handle is not None:
        return (
            projection_chunk_handle,
            projection_chunk_temp_path,
            0,
            current_uncompressed_offset,
        )

    partition_dir = _resolve_wikidata_chunk_partition_dir(
        chunk_dir=chunk_dir,
        chunk_index=projection_chunk_index,
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    projection_chunk_temp_path = (
        partition_dir / f"{chunk_prefix}{projection_chunk_index:06d}.tmp"
    )
    projection_chunk_handle = projection_chunk_temp_path.open("wb")
    return (
        projection_chunk_handle,
        projection_chunk_temp_path,
        0,
        current_uncompressed_offset,
    )


def _finalize_wikidata_projection_chunk(
    *,
    projection_chunk_handle: _BinaryWriteHandle | None,
    projection_chunk_temp_path: Path | None,
    projection_chunk_row_count: int,
    projection_chunk_index: int,
    chunk_dir: Path,
    chunk_prefix: str,
) -> tuple[_BinaryWriteHandle | None, Path | None, int, int]:
    if projection_chunk_handle is None or projection_chunk_temp_path is None:
        return (
            projection_chunk_handle,
            projection_chunk_temp_path,
            projection_chunk_row_count,
            projection_chunk_index,
        )

    projection_chunk_handle.close()
    if projection_chunk_row_count > 0:
        if projection_chunk_temp_path.parent == chunk_dir:
            partition_dir = _resolve_wikidata_chunk_partition_dir(
                chunk_dir=chunk_dir,
                chunk_index=projection_chunk_index,
            )
        else:
            partition_dir = projection_chunk_temp_path.parent
        finalized_path = (
            partition_dir / f"{chunk_prefix}{projection_chunk_index:06d}.jsonl"
        )
        projection_chunk_temp_path.replace(finalized_path)
        projection_chunk_index += 1
    else:
        projection_chunk_temp_path.unlink(missing_ok=True)

    return None, None, 0, projection_chunk_index


def _maybe_emit_wikidata_max_lines_stop(
    *,
    stop_after_this_line: bool,
    lines_processed: int,
    max_lines: int | None,
    emitter: WikidataProgressEmitter,
) -> bool:
    # `stop_after_this_line` is only ever set when `max_lines` is not None (see
    # `_process_line`), but the two arrive as independent arguments, so the
    # `None` case is re-checked here rather than assumed away.
    if not stop_after_this_line or max_lines is None:
        return False
    emitter.emit_early_stop_lines(
        phase="streaming", lines_processed=lines_processed, max_lines=max_lines
    )
    return True


@dataclass
class _WikidataStreamProjectionState:
    lines_processed: int = 0
    lines_skipped_no_candidate: int = 0
    company_count: int = 0
    projected_count: int = 0
    current_uncompressed_offset: int = 0
    chunk_uncompressed_start_offset: int = 0
    projection_chunk_row_count: int = 0
    projection_chunk_index: int = 1
    projection_chunk_handle: _BinaryWriteHandle | None = None
    projection_chunk_temp_path: Path | None = None
    progress_samples: list[tuple[float, int]] | None = None
    last_progress_emit: float = 0.0


class WikidataProjectionEngine:
    """Single-scan filter+project engine: matches and projects each candidate
    line in one pass over the source, writing output as resumable chunks.
    """

    def __init__(
        self,
        *,
        source_path: Path,
        destination_path: Path,
        reader_strategy,
        emitter: WikidataProgressEmitter,
        matcher: Callable[[bytes], bool],
        progress_every_entities: int,
        progress_every_seconds: float,
        max_companies: int | None,
        max_lines: int | None,
        stream_resume: bool,
        resolved_read_options: dict[str, object] | None = None,
        legal_form_qids: frozenset[str] | None = None,
    ) -> None:
        self.legal_form_qids = legal_form_qids
        self.source_path = source_path
        self.destination_path = destination_path
        self.reader_strategy = reader_strategy
        self.emitter = emitter
        self.matcher = matcher
        self.progress_every_entities = progress_every_entities
        self.progress_every_seconds = progress_every_seconds
        self.max_companies = max_companies
        self.max_lines = max_lines
        self.stream_resume = stream_resume
        self.resolved_read_options = resolved_read_options

        self.source_size = source_path.stat().st_size
        self.started = time.perf_counter()
        self.chunk_byte_limit = _WIKIDATA_PREPARE_ROLLOVER_BYTES
        self.chunk_dir = destination_path.parent / _WIKIDATA_CHUNK_DIR_NAME
        self.chunk_prefix = _WIKIDATA_CHUNK_PREFIX
        self.resume_uncompressed_offset = 0
        self.state = _WikidataStreamProjectionState(last_progress_emit=self.started)
        self.state.progress_samples = []

    def run(self) -> tuple[int, int]:
        self._prepare()

        try:
            source_handle = self.reader_strategy.open_source()
            with source_handle:
                self._maybe_resume_source(source_handle)
                for line_number, line in self.reader_strategy.iter_content_lines(
                    source_handle
                ):
                    if self._process_line(
                        line=line, line_number=line_number, source_handle=source_handle
                    ):
                        break
                self._finalize_chunk_if_needed()
        finally:
            _close_wikidata_projection_outputs(
                projection_chunk_handle=self.state.projection_chunk_handle,
                projection_chunk_temp_path=self.state.projection_chunk_temp_path,
            )

        self._emit_complete()
        return self.state.company_count, self.state.projected_count

    def _prepare(self) -> None:
        (
            self.resume_uncompressed_offset,
            self.state.projection_chunk_index,
            self.state.company_count,
            self.state.projected_count,
        ) = _prepare_wikidata_stream_chunks(
            chunk_dir=self.chunk_dir,
            chunk_prefix=self.chunk_prefix,
            stream_resume=self.stream_resume,
            chunk_byte_limit=self.chunk_byte_limit,
            emitter=self.emitter,
        )

    def _maybe_resume_source(self, source_handle) -> None:
        if self.resume_uncompressed_offset <= 0:
            return
        source_handle.seek(self.resume_uncompressed_offset)
        self.state.current_uncompressed_offset = self.resume_uncompressed_offset
        self.emitter.emit_stream_resume_seek(
            resume_uncompressed_offset=self.resume_uncompressed_offset
        )

    def _finalize_chunk_if_needed(self) -> None:
        (
            self.state.projection_chunk_handle,
            self.state.projection_chunk_temp_path,
            self.state.projection_chunk_row_count,
            self.state.projection_chunk_index,
        ) = _finalize_wikidata_projection_chunk(
            projection_chunk_handle=self.state.projection_chunk_handle,
            projection_chunk_temp_path=self.state.projection_chunk_temp_path,
            projection_chunk_row_count=self.state.projection_chunk_row_count,
            projection_chunk_index=self.state.projection_chunk_index,
            chunk_dir=self.chunk_dir,
            chunk_prefix=self.chunk_prefix,
        )

    def _emit_complete(self) -> None:
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        line_rate = self.state.lines_processed / elapsed if elapsed > 0 else 0
        company_rate = self.state.company_count / elapsed if elapsed > 0 else 0
        self.emitter.emit_stream_complete(
            lines_processed=self.state.lines_processed,
            company_count=self.state.company_count,
            lines_skipped_no_candidate=self.state.lines_skipped_no_candidate,
            elapsed=elapsed,
            line_rate=line_rate,
            company_rate=company_rate,
        )

    def _process_line(self, *, line: bytes, line_number: int, source_handle) -> bool:
        self.state.current_uncompressed_offset = source_handle.tell()
        self.state.lines_processed = line_number
        stop_after_this_line = (
            self.max_lines is not None and self.state.lines_processed >= self.max_lines
        )

        if not self.matcher(line):
            self.state.lines_skipped_no_candidate += 1
            return _maybe_emit_wikidata_max_lines_stop(
                stop_after_this_line=stop_after_this_line,
                lines_processed=self.state.lines_processed,
                max_lines=self.max_lines,
                emitter=self.emitter,
            )

        self.state.company_count += 1

        projected_line = _project_wikidata_company_record_line(
            line, legal_form_qids=self.legal_form_qids
        )
        if projected_line is not None and self._write_projected_line(
            projected_line=projected_line
        ):
            return True

        if _maybe_emit_wikidata_max_lines_stop(
            stop_after_this_line=stop_after_this_line,
            lines_processed=self.state.lines_processed,
            max_lines=self.max_lines,
            emitter=self.emitter,
        ):
            return True

        since_last_progress = time.perf_counter() - self.state.last_progress_emit
        if _should_emit_wikidata_progress(
            lines_processed=self.state.lines_processed,
            progress_every_lines=self.progress_every_entities,
            progress_every_seconds=self.progress_every_seconds,
            since_last_progress=since_last_progress,
        ):
            recent_line_rate: float | None = None
            now = time.perf_counter()
            if self.state.progress_samples is not None:
                self.state.progress_samples.append((now, self.state.lines_processed))
                cutoff = now - _WIKIDATA_RECENT_RATE_WINDOW_SECONDS
                while (
                    len(self.state.progress_samples) > 1
                    and self.state.progress_samples[0][0] < cutoff
                ):
                    self.state.progress_samples.pop(0)
                if len(self.state.progress_samples) >= 2:
                    first_time, first_lines = self.state.progress_samples[0]
                    delta_seconds = now - first_time
                    delta_lines = self.state.lines_processed - first_lines
                    if delta_seconds > 0 and delta_lines > 0:
                        recent_line_rate = delta_lines / delta_seconds

            _emit_wikidata_stream_progress(
                source_path=self.source_path,
                source_handle=source_handle,
                source_size=self.source_size,
                lines_processed=self.state.lines_processed,
                company_count=self.state.company_count,
                started=self.started,
                emitter=self.emitter,
                recent_line_rate=recent_line_rate,
            )
            self.state.last_progress_emit = time.perf_counter()

        return False

    def _write_projected_line(self, *, projected_line: bytes) -> bool:
        if self.state.projection_chunk_handle is None:
            (
                self.state.projection_chunk_handle,
                self.state.projection_chunk_temp_path,
                self.state.projection_chunk_row_count,
                self.state.chunk_uncompressed_start_offset,
            ) = _start_wikidata_projection_chunk(
                projection_chunk_handle=self.state.projection_chunk_handle,
                projection_chunk_temp_path=self.state.projection_chunk_temp_path,
                chunk_dir=self.chunk_dir,
                chunk_prefix=self.chunk_prefix,
                projection_chunk_index=self.state.projection_chunk_index,
                current_uncompressed_offset=self.state.current_uncompressed_offset,
            )
        if self.state.projection_chunk_handle is not None:
            self.state.projection_chunk_handle.write(projected_line)
        self.state.projection_chunk_row_count += 1

        self.state.projected_count += 1
        if (
            self.max_companies is not None
            and self.state.projected_count >= self.max_companies
        ):
            self.emitter.emit_early_stop_rows(
                phase="streaming",
                rows_written=self.state.projected_count,
                max_companies=self.max_companies,
            )
            return True

        if (
            self.state.projection_chunk_row_count > 0
            and (
                self.state.current_uncompressed_offset
                - self.state.chunk_uncompressed_start_offset
            )
            >= self.chunk_byte_limit
        ):
            (
                self.state.projection_chunk_handle,
                self.state.projection_chunk_temp_path,
                self.state.projection_chunk_row_count,
                self.state.projection_chunk_index,
            ) = _finalize_wikidata_projection_chunk(
                projection_chunk_handle=self.state.projection_chunk_handle,
                projection_chunk_temp_path=self.state.projection_chunk_temp_path,
                projection_chunk_row_count=self.state.projection_chunk_row_count,
                projection_chunk_index=self.state.projection_chunk_index,
                chunk_dir=self.chunk_dir,
                chunk_prefix=self.chunk_prefix,
            )

        return False


def _run_wikidata_projection_stream_mode(
    *,
    source_path: Path,
    destination_path: Path,
    emit: Callable[[str], None],
    matcher: Callable[[bytes], bool],
    progress_every_entities: int,
    progress_every_seconds: float,
    max_companies: int | None,
    max_lines: int | None,
    stream_resume: bool,
    resolved_read_options: dict[str, object] | None,
    legal_form_qids: frozenset[str] | None = None,
) -> tuple[int, int]:
    reader_strategy = build_wikidata_reader_strategy(
        source_path=source_path,
        resolved_read_options=resolved_read_options,
        iter_content_lines_fn=_iter_wikidata_content_lines_from_handle,
        default_chunk_size_bytes=_READ_CHUNK_SIZE_BYTES,
    )
    emitter = WikidataProgressEmitter(emit)
    engine = WikidataProjectionEngine(
        source_path=source_path,
        destination_path=destination_path,
        reader_strategy=reader_strategy,
        emitter=emitter,
        matcher=matcher,
        progress_every_entities=progress_every_entities,
        progress_every_seconds=progress_every_seconds,
        max_companies=max_companies,
        max_lines=max_lines,
        stream_resume=stream_resume,
        resolved_read_options=resolved_read_options,
        legal_form_qids=legal_form_qids,
    )
    return engine.run()


def extract_wikidata_company_projection_two_pass(
    source_path: Path,
    destination_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every_entities: int = 50_000,
    progress_every_seconds: float = 30.0,
    max_companies: int | None = None,
    max_lines: int | None = None,
    stream_resume: bool = False,
    resolved_read_options: dict[str, object] | None = None,
    legal_form_qids: frozenset[str] | None = None,
) -> int:
    """Single-scan Wikidata company extraction: filters and projects each
    candidate line in one pass over the source.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    matcher: Callable[[bytes], bool] = _is_wikidata_company_candidate_raw_line
    if legal_form_qids is not None:
        matcher = _make_wikidata_company_candidate_raw_line_with_legal_form(
            legal_form_qids
        )

    emitter = WikidataProgressEmitter(progress)
    emitter.emit_stream_filter_projection_header()
    emitter.emit_stream_single_scan_notice()

    company_count, projected_count = _run_wikidata_projection_stream_mode(
        source_path=source_path,
        destination_path=destination_path,
        emit=emitter.emit,
        matcher=matcher,
        progress_every_entities=progress_every_entities,
        progress_every_seconds=progress_every_seconds,
        max_companies=max_companies,
        max_lines=max_lines,
        stream_resume=stream_resume,
        resolved_read_options=resolved_read_options,
        legal_form_qids=legal_form_qids,
    )

    emitter.emit_optimization_complete(
        company_count=company_count,
        projected_count=projected_count,
    )

    return projected_count
