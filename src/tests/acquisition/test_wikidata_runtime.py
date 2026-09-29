from __future__ import annotations

import re
from pathlib import Path

from acquisition.wikidata_runtime import (
    CallbackProgressEmitter,
    WikidataProgressEmitter,
    build_wikidata_reader_strategy,
)


def test_build_wikidata_reader_strategy_applies_resolved_read_options() -> None:
    seen_kwargs: dict[str, object] = {}

    def fake_iter_content_lines(handle, *, read_chunk_size_bytes: int):
        seen_kwargs["chunk_size"] = read_chunk_size_bytes
        yield 1, b'{"id":"Q1"}'

    strategy = build_wikidata_reader_strategy(
        source_path=Path("wikidata-all.json.bz2"),
        resolved_read_options={
            "reader_mode": "chunked_indexed",
            "chunk_size_bytes": 8 * 1024 * 1024,
            "indexed_bzip2_parallelization": 4,
        },
        iter_content_lines_fn=fake_iter_content_lines,
    )

    assert strategy.reader_mode == "chunked_indexed"
    assert strategy.indexed_bzip2_parallelization == 4
    assert strategy.read_chunk_size_bytes == 8 * 1024 * 1024

    rows = list(strategy.iter_content_lines(object()))
    assert rows == [(1, b'{"id":"Q1"}')]
    assert seen_kwargs["chunk_size"] == 8 * 1024 * 1024


def test_callback_progress_emitter_emits_when_callback_present() -> None:
    messages: list[str] = []
    emitter = CallbackProgressEmitter(messages.append)

    emitter.emit("hello")

    assert messages == ["hello"]


def test_callback_progress_emitter_is_noop_without_callback() -> None:
    emitter = CallbackProgressEmitter(None)

    # No exception expected.
    emitter.emit("ignored")


def test_wikidata_progress_emitter_formats_prepare_progress() -> None:
    messages: list[str] = []
    emitter = WikidataProgressEmitter(messages.append)

    emitter.emit_phase_counter_progress(
        phase="prepare projection",
        lines_processed=10,
        secondary_count_label="written",
        secondary_count=4,
        compressed_progress_text="compressed_bytes=100/200, estimated_total_lines=20",
        eta_text="eta=1s",
        line_rate=10.0,
        secondary_rate_label="write_rate",
        secondary_rate=4.0,
    )

    assert messages == [
        (
            "[wikidata] prepare projection progress: lines_processed=10, written=4, "
            "compressed_bytes=100/200, estimated_total_lines=20, eta=1s, line_rate=10/s, write_rate=4/s"
        )
    ]


def test_wikidata_progress_emitter_formats_generic_phase_messages() -> None:
    messages: list[str] = []
    emitter = WikidataProgressEmitter(messages.append)

    emitter.emit_started(phase="prepare projection", source_name="wikidata.jsonl")
    emitter.emit_scanning(phase="prepare projection")
    emitter.emit_early_stop_lines(
        phase="prepare projection", lines_processed=10, max_lines=10
    )
    emitter.emit_early_stop_rows(
        phase="prepare projection", rows_written=4, max_companies=4
    )

    assert messages == [
        "[wikidata] prepare projection started: source=wikidata.jsonl",
        "[wikidata] prepare projection scanning dump",
        "[wikidata] prepare projection early stop: lines_processed=10 (max_lines=10)",
        "[wikidata] prepare projection early stop: rows_written=4 (max_companies=4)",
    ]


def test_wikidata_progress_emitter_formats_filter_progress() -> None:
    messages: list[str] = []
    emitter = WikidataProgressEmitter(messages.append)

    emitter.emit_phase_counter_progress(
        phase="Company filter",
        lines_processed=10,
        secondary_count_label="company_found",
        secondary_count=3,
        compressed_progress_text="compressed_bytes=100/200, estimated_total_lines=20",
        eta_text="eta=1s",
        line_rate=10.0,
        secondary_rate_label="company_rate",
        secondary_rate=3.0,
    )

    assert messages == [
        (
            "[wikidata] Company filter progress: lines_processed=10, company_found=3, "
            "compressed_bytes=100/200, estimated_total_lines=20, eta=1s, line_rate=10/s, company_rate=3/s"
        )
    ]


def test_wikidata_progress_emitter_formats_stream_progress_and_summary() -> None:
    messages: list[str] = []
    emitter = WikidataProgressEmitter(messages.append)

    emitter.emit_stream_progress(
        lines_processed=10,
        company_count=3,
        line_rate=10.0,
        company_rate=3.0,
        company_share=0.3,
        compressed_progress_text="compressed_bytes=100/200, estimated_total_lines=20",
        estimated_totals_text="estimated_total_companies=6",
        eta_text="eta=1m-3m",
        eta_stability="stable",
    )
    emitter.emit_stream_complete(
        lines_processed=10,
        company_count=3,
        lines_skipped_no_candidate=7,
        elapsed=1.0,
        line_rate=10.0,
        company_rate=3.0,
    )

    assert messages == [
        "[wikidata] streaming progress: lines_processed=10, company_found=3, line_rate=10/s, company_rate=3/s, company_share=30.00%, compressed_bytes=100/200, estimated_total_lines=20, estimated_total_companies=6, eta=1m-3m, eta_stability=stable",
        "[wikidata] streaming complete: lines_processed=10, company_found=3, no_candidate_skipped=7, elapsed=1.0s, line_rate=10/s, company_rate=3/s",
    ]


def test_wikidata_progress_emitter_formats_phase_and_cleanup_messages() -> None:
    messages: list[str] = []
    emitter = WikidataProgressEmitter(messages.append)

    emitter.emit_stream_filter_projection_header()
    emitter.emit_stream_single_scan_notice()
    emitter.emit_optimization_complete(company_count=5, projected_count=4)

    assert messages == [
        "[wikidata] ============ STREAMED FILTER + PROJECTION ============",
        "[wikidata] phase 1 and phase 2 executed in one source scan",
        "[wikidata] ============ Optimization Complete ============",
        "[wikidata] Pass 1 company count: 5",
        "[wikidata] Final projected companies: 4",
    ]


def test_wikidata_progress_emitter_contract_shapes() -> None:
    messages: list[str] = []
    emitter = WikidataProgressEmitter(messages.append)

    emitter.emit_started(phase="streaming", source_name="wikidata.jsonl")
    emitter.emit_scanning(phase="streaming")
    emitter.emit_progress(
        phase="streaming",
        counts_text="lines_processed=10, company_found=3",
        trailing_text="eta=1s, line_rate=10/s",
    )
    emitter.emit_complete(
        phase="streaming",
        counts_text="lines_processed=10, company_found=3",
        trailing_text="elapsed=1.0s, line_rate=10/s",
    )
    emitter.emit_early_stop_lines(phase="streaming", lines_processed=10, max_lines=10)
    emitter.emit_early_stop_rows(phase="streaming", rows_written=3, max_companies=3)
    emitter.emit_resume(
        action="seeking source to uncompressed offset", detail_text="1,024"
    )

    assert len(messages) == 7
    assert re.fullmatch(
        r"\[wikidata\] [a-zA-Z0-9 ()+_-]+ started: source=.+", messages[0]
    )
    assert re.fullmatch(r"\[wikidata\] [a-zA-Z0-9 ()+_-]+ scanning dump", messages[1])
    assert re.fullmatch(
        r"\[wikidata\] [a-zA-Z0-9 ()+_-]+ progress: .+, .+", messages[2]
    )
    assert re.fullmatch(
        r"\[wikidata\] [a-zA-Z0-9 ()+_-]+ complete: .+, .+", messages[3]
    )
    assert re.fullmatch(
        r"\[wikidata\] [a-zA-Z0-9 ()+_-]+ early stop: lines_processed=[0-9,]+ \(max_lines=[0-9,]+\)",
        messages[4],
    )
    assert re.fullmatch(
        r"\[wikidata\] [a-zA-Z0-9 ()+_-]+ early stop: rows_written=[0-9,]+ \(max_companies=[0-9,]+\)",
        messages[5],
    )
    assert re.fullmatch(r"\[wikidata\] streaming resume: .+", messages[6])
