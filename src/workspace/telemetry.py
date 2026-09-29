"""Run telemetry: wall-clock, memory and CPU per phase, recorded by any routine.

One `Telemetry` per run. A routine wraps each phase it wants measured in
`telemetry.phase(...)`, names it, labels it with the unit it worked on such
as a country, and states the rows it took in and, through the handle the
block receives, the rows it produced. The record is taken when the block
ends, whether it returned or raised: the phase's elapsed wall-clock, the
process's resident set size at that moment, and the share of one CPU the
process used across the phase. A listener given to the recorder sees each
record as it lands, which is how a script prints a line per completed phase
instead of staying silent through a long one.

Memory is recorded twice. `rss_bytes` is the resident set size at the phase's
end, so a phase that allocated and freed inside itself reports only what was
still held. `peak_rss_bytes` is the largest resident set size seen while the
phase was open, sampled on a background thread every
`PEAK_SAMPLE_SECONDS`: a sampled peak, which can miss a spike shorter than
the interval, and which a nested phase and the phase containing it each keep
separately. Either is `None` when sampling fails, never zero, so a missing
figure cannot be read as a small one. CPU percent is user plus system time
over wall-clock, so a phase parallel across cores reports above one hundred.

`TELEMETRY_SCHEMA` is the record's tabular shape, one row per phase, for a
run to persist beside its other outputs and for a log across runs to share.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import polars as pl
import psutil

PolarsDType = type[pl.DataType] | pl.DataType

TELEMETRY_SCHEMA: dict[str, PolarsDType] = {
    "routine": pl.Utf8,
    "label": pl.Utf8,
    "started_at": pl.Utf8,
    "elapsed_seconds": pl.Float64,
    "rss_bytes": pl.Int64,
    "peak_rss_bytes": pl.Int64,
    "cpu_percent": pl.Float64,
    "rows_in": pl.Int64,
    "rows_out": pl.Int64,
}
"""One row per recorded phase."""

PEAK_SAMPLE_SECONDS = 0.2
"""How often an open phase's resident set size is sampled for its peak."""


@dataclass(frozen=True)
class PhaseRecord:
    """One measured phase.

    Attributes:
        routine: What ran, the phase's name within its routine.
        label: The unit it ran over, such as a country or a system, or
            `None` when the phase has no unit.
        started_at: When the phase began, UTC, ISO 8601.
        elapsed_seconds: Wall-clock from entering the block to leaving it.
        rss_bytes: The process's resident set size when the block ended,
            `None` when it could not be sampled.
        peak_rss_bytes: The largest resident set size sampled while the
            block ran, its end included; `None` when none could be sampled.
        cpu_percent: User plus system CPU time over the phase as a share of
            its wall-clock, in percent; `None` when it could not be sampled.
        rows_in: Rows the phase took in, when the caller stated them.
        rows_out: Rows the phase produced, when the caller stated them.
    """

    routine: str
    label: str | None
    started_at: str
    elapsed_seconds: float
    rss_bytes: int | None
    peak_rss_bytes: int | None
    cpu_percent: float | None
    rows_in: int | None
    rows_out: int | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class PhaseHandle:
    """What a `phase()` block receives, to state what the phase produced."""

    rows_out: int | None = None


def sample_rss_bytes(process: psutil.Process | None = None) -> int | None:
    """The process's resident set size now, or `None` when unreadable."""
    try:
        return int((process or psutil.Process()).memory_info().rss)
    except (psutil.Error, OSError):
        return None


def _cpu_seconds(process: psutil.Process | None) -> float | None:
    if process is None:
        return None
    try:
        times = process.cpu_times()
    except (psutil.Error, OSError):
        return None
    return float(times.user + times.system)


class _PeakSampler:
    """Samples the process's resident set size on a thread while any phase is
    open, raising the running maximum of every open phase.

    Each open phase holds a one-element list, its peak so far. The thread
    starts with the first open phase and stops when the last one closes, so
    a run between phases pays nothing.
    """

    def __init__(self, process: psutil.Process | None) -> None:
        self._process = process
        self._lock = threading.Lock()
        self._open: list[list[int | None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _note(self, rss: int | None) -> None:
        if rss is None:
            return
        with self._lock:
            for peak in self._open:
                if peak[0] is None or rss > peak[0]:
                    peak[0] = rss

    def _run(self, stop: threading.Event) -> None:
        while not stop.wait(PEAK_SAMPLE_SECONDS):
            self._note(sample_rss_bytes(self._process))

    def open(self) -> list[int | None]:
        peak: list[int | None] = [None]
        if self._process is None:
            return peak
        with self._lock:
            self._open.append(peak)
            if self._thread is None:
                self._stop = threading.Event()
                self._thread = threading.Thread(
                    target=self._run, args=(self._stop,), daemon=True
                )
                self._thread.start()
        self._note(sample_rss_bytes(self._process))
        return peak

    def close(self, peak: list[int | None]) -> int | None:
        if self._process is None:
            return None
        self._note(sample_rss_bytes(self._process))
        thread: threading.Thread | None = None
        with self._lock:
            self._open = [other for other in self._open if other is not peak]
            if not self._open and self._thread is not None:
                self._stop.set()
                thread, self._thread = self._thread, None
        if thread is not None:
            thread.join()
        return peak[0]


class Telemetry:
    """Records the phases of one run.

    `listener`, when given, is called with each `PhaseRecord` as its phase
    ends. `records` is every phase so far in the order they ended, so a
    nested phase precedes the phase that contains it.
    """

    def __init__(self, *, listener: Callable[[PhaseRecord], None] | None = None):
        self._records: list[PhaseRecord] = []
        self._listener = listener
        try:
            self._process: psutil.Process | None = psutil.Process()
        except (psutil.Error, OSError):
            self._process = None
        self._peaks = _PeakSampler(self._process)

    @property
    def records(self) -> tuple[PhaseRecord, ...]:
        return tuple(self._records)

    @contextmanager
    def phase(
        self, routine: str, *, label: str | None = None, rows_in: int | None = None
    ) -> Iterator[PhaseHandle]:
        """Measure the block as one phase named `routine` over `label`."""
        handle = PhaseHandle()
        started_at = datetime.now(UTC).isoformat()
        peak = self._peaks.open()
        cpu_before = _cpu_seconds(self._process)
        clock_before = time.perf_counter()
        try:
            yield handle
        finally:
            elapsed = time.perf_counter() - clock_before
            cpu_after = _cpu_seconds(self._process)
            peak_rss_bytes = self._peaks.close(peak)
            cpu_percent = (
                (cpu_after - cpu_before) / elapsed * 100.0
                if cpu_before is not None and cpu_after is not None and elapsed > 0.0
                else None
            )
            record = PhaseRecord(
                routine=routine,
                label=label,
                started_at=started_at,
                elapsed_seconds=elapsed,
                rss_bytes=sample_rss_bytes(self._process),
                peak_rss_bytes=peak_rss_bytes,
                cpu_percent=cpu_percent,
                rows_in=None if rows_in is None else int(rows_in),
                rows_out=None if handle.rows_out is None else int(handle.rows_out),
            )
            self._records.append(record)
            if self._listener is not None:
                self._listener(record)

    def record_total(
        self,
        routine: str,
        *,
        elapsed_seconds: float,
        label: str | None = None,
        rows_in: int | None = None,
    ) -> PhaseRecord:
        """Record a step whose wall-clock was summed by the caller, such as one
        repeated per chunk inside a measured phase.

        Only the elapsed time means anything for a sum of short intervals, so
        `rss_bytes`, `peak_rss_bytes` and `cpu_percent` are `None`, and
        `started_at` is when the total was recorded.
        """
        record = PhaseRecord(
            routine=routine,
            label=label,
            started_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=float(elapsed_seconds),
            rss_bytes=None,
            peak_rss_bytes=None,
            cpu_percent=None,
            rows_in=None if rows_in is None else int(rows_in),
            rows_out=None,
        )
        self._records.append(record)
        if self._listener is not None:
            self._listener(record)
        return record

    def to_dicts(self) -> list[dict[str, object]]:
        """Every record as a JSON-ready mapping, in the order the phases ended."""
        return [record.to_dict() for record in self._records]

    def frame(self) -> pl.DataFrame:
        """Every record as one `TELEMETRY_SCHEMA` row."""
        return pl.DataFrame(self.to_dicts(), schema=TELEMETRY_SCHEMA)
