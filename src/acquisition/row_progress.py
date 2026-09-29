"""Row-count progress emission for long-running stages.

Progress used to be a side effect of output file size: Canonical wrote a
file every 100,000 rows, so Cleanse and Tokenize printed one line per input
file and that happened to be every 100,000 rows. Raising the partitioned
view to 1,000,000 rows a file fixed 2-7MB fragments and silently made every
downstream stage ten times quieter -- `gb` cleanse went
from 57 lines about seven seconds apart to six about seventy-two seconds
apart, and Canonical, which never had per-chunk output at all, gained work
without gaining any way to see it.

Cadence belongs to the reporter, not to the writer. `RowProgress` emits
every `interval` rows however the caller batches its work, so changing file
size again cannot change how often a run reports.
"""

from __future__ import annotations

import time
from collections.abc import Callable

PROGRESS_ROW_INTERVAL = 100_000
"""Rows between progress lines. Matches the cadence stages had when Canonical
wrote a file every 100,000 rows, frequent enough to show a stall without
burying the stage's own output."""


class RowProgress:
    """Emits a progress line every `interval` rows.

    A no-op when given no `progress` callable, so a caller can pass one
    through unconditionally rather than branching on whether anyone is
    listening.
    """

    def __init__(
        self,
        *,
        label: str,
        progress: Callable[[str], None] | None = None,
        total_rows: int | None = None,
        interval: int = PROGRESS_ROW_INTERVAL,
    ) -> None:
        self._label = label
        self._progress = progress
        self._total_rows = total_rows
        self._interval = max(1, interval)
        self._rows = 0
        self._emitted_at = 0
        self._started = time.perf_counter()

    @property
    def rows(self) -> int:
        return self._rows

    def advance(self, rows: int) -> None:
        """Record `rows` more rows processed, emitting if a boundary passed.

        Emits once per crossed boundary at most, so a caller handing over a
        million rows in one call gets one line rather than ten.
        """
        if rows <= 0:
            return
        self._rows += rows
        if self._progress is None:
            return
        if self._rows - self._emitted_at < self._interval:
            return
        # Record where the line was actually emitted, not the boundary it
        # crossed: rounding down here makes `finish` believe the tail is
        # still unreported and print the same total a second time.
        self._emitted_at = self._rows
        self._progress(self._format(self._rows))

    def finish(self) -> None:
        """Emit a final line for whatever has not been reported yet."""
        if self._progress is None or self._rows == 0:
            return
        if self._rows != self._emitted_at:
            self._progress(self._format(self._rows, final=True))

    def _format(self, rows: int, *, final: bool = False) -> str:
        elapsed = max(time.perf_counter() - self._started, 1e-9)
        parts = [f"{rows:,} rows"]
        if self._total_rows:
            parts.append(f"{rows / self._total_rows:.0%}")
        parts.append(f"{rows / elapsed:,.0f} rows/s")
        suffix = " (complete)" if final else ""
        return f"[{self._label}] {', '.join(parts)}{suffix}"
