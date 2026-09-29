from __future__ import annotations

import time

import pytest

from workspace.telemetry import (
    PEAK_SAMPLE_SECONDS,
    TELEMETRY_SCHEMA,
    PhaseRecord,
    Telemetry,
    sample_rss_bytes,
)


def test_phase_records_elapsed_memory_cpu_and_rows() -> None:
    telemetry = Telemetry()

    with telemetry.phase("scoring", label="ie", rows_in=10) as handle:
        total = sum(range(200_000))
        handle.rows_out = 3

    (record,) = telemetry.records
    assert total > 0
    assert record.routine == "scoring"
    assert record.label == "ie"
    assert record.started_at.endswith("+00:00")
    assert record.elapsed_seconds > 0.0
    assert record.rss_bytes is not None and record.rss_bytes > 0
    assert record.cpu_percent is not None and record.cpu_percent >= 0.0
    assert (record.rows_in, record.rows_out) == (10, 3)


def test_phase_peak_catches_memory_allocated_and_freed_inside_it() -> None:
    """A block that allocates and frees a large buffer ends near where it
    began, but its sampled peak holds the buffer; the enclosing phase keeps a
    peak of its own at least as high."""
    telemetry = Telemetry()
    size = 200 * 1024 * 1024

    with telemetry.phase("outer"), telemetry.phase("inner"):
        buffer = b"\x01" * size
        time.sleep(4 * PEAK_SAMPLE_SECONDS)
        del buffer

    inner, outer = telemetry.records
    assert inner.peak_rss_bytes is not None and inner.rss_bytes is not None
    assert inner.peak_rss_bytes - inner.rss_bytes > size // 2
    assert outer.peak_rss_bytes is not None
    assert outer.peak_rss_bytes >= inner.peak_rss_bytes


def test_phase_records_even_when_the_block_raises() -> None:
    telemetry = Telemetry()

    with pytest.raises(RuntimeError), telemetry.phase("load", label="gb"):
        raise RuntimeError("boom")

    (record,) = telemetry.records
    assert record.routine == "load"
    assert record.rows_out is None


def test_listener_sees_each_record_as_its_phase_ends() -> None:
    seen: list[PhaseRecord] = []
    telemetry = Telemetry(listener=seen.append)

    with telemetry.phase("outer"):
        with telemetry.phase("inner"):
            pass
        assert [record.routine for record in seen] == ["inner"]

    assert [record.routine for record in seen] == ["inner", "outer"]
    assert [record.routine for record in telemetry.records] == ["inner", "outer"]


def test_records_serialise_to_dicts_and_a_schema_frame() -> None:
    telemetry = Telemetry()
    with telemetry.phase("index", label="fr", rows_in=5):
        pass

    dicts = telemetry.to_dicts()
    assert set(dicts[0]) == set(TELEMETRY_SCHEMA)
    frame = telemetry.frame()
    assert frame.schema == TELEMETRY_SCHEMA
    assert frame.height == 1
    assert frame.row(0, named=True)["label"] == "fr"


def test_record_total_adds_a_summed_step_with_no_memory_or_cpu_figure() -> None:
    seen: list[PhaseRecord] = []
    telemetry = Telemetry(listener=seen.append)

    record = telemetry.record_total(
        "scoring_backend", elapsed_seconds=1.5, label="ie", rows_in=10
    )

    assert telemetry.records == (record,)
    assert seen == [record]
    assert record.elapsed_seconds == 1.5
    assert record.label == "ie" and record.rows_in == 10
    assert record.rss_bytes is None and record.cpu_percent is None
    assert record.peak_rss_bytes is None
    assert telemetry.frame().schema == TELEMETRY_SCHEMA


def test_empty_recorder_gives_an_empty_frame_with_the_schema() -> None:
    frame = Telemetry().frame()
    assert frame.height == 0
    assert frame.schema == TELEMETRY_SCHEMA


def test_sample_rss_bytes_reads_this_process() -> None:
    rss = sample_rss_bytes()
    assert rss is not None and rss > 0
