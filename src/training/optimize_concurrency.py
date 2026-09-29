"""Memory-aware concurrency for optimize waves.

`AdaptiveConcurrencyController` holds the per-candidate memory estimate and `MemoryBudgetLedger` reserves memory for each wave before it launches; together they size waves so concurrent candidates do not overcommit real memory.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl
import psutil


def sample_process_rss_bytes() -> int | None:
    """Best-effort RSS for the current process. None (not 0) on any psutil
    failure, so callers can tell "no sample" from "used nothing."
    """
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 -- a diagnostic sample must never crash a candidate
        return None


def available_memory_bytes() -> int:
    """Live system-wide available memory. Thin wrapper so tests can inject a
    fake instead of depending on the real machine's state.
    """
    return int(psutil.virtual_memory().available)


def compute_target_concurrency(
    *,
    available_memory_bytes: int,
    per_candidate_memory_bytes: int,
    headroom_fraction: float,
    hard_cap: int,
    min_concurrency: int = 1,
) -> int:
    """Pure function: how many candidates can run concurrently right now.

    No psutil, no state -- directly unit-testable with plain ints. A
    non-positive per_candidate_memory_bytes (no estimate available yet)
    defensively falls back to hard_cap rather than dividing by zero or
    returning an artificially tiny number.
    """
    usable_bytes = max(0.0, available_memory_bytes * (1.0 - headroom_fraction))
    if per_candidate_memory_bytes <= 0:
        target = hard_cap
    else:
        target = int(usable_bytes // per_candidate_memory_bytes)
    return max(min_concurrency, min(hard_cap, target))


def historical_worker_rss_estimate(history_df: pl.DataFrame) -> int | None:
    """Max observed worker_rss_bytes from already-loaded run-log history, or
    None if the column is absent (pre-Phase-1 history) or entirely null.
    """
    if "worker_rss_bytes" not in history_df.columns:
        return None
    non_null = history_df["worker_rss_bytes"].drop_nulls()
    if non_null.len() == 0:
        return None
    return int(non_null.max())  # type: ignore[arg-type]


@dataclass
class AdaptiveConcurrencyController:
    """Tracks the per-candidate memory estimate for one optimize run.

    Purely an estimate holder -- the windowing/submission mechanics live in
    the wave scheduler, not here. `per_candidate_memory_estimate_bytes`
    prefers a real observed sample (this run, or seeded from history) over
    the cold-start floor, and never goes backwards: a run that observes
    something bigger than history predicted adapts upward immediately.
    """

    hard_cap: int
    memory_floor_bytes: int
    bootstrap_rss_bytes: int | None = None
    enabled: bool = True
    _peak_observed_rss_bytes: int = field(default=0, init=False, repr=False)

    def per_candidate_memory_estimate_bytes(self) -> int:
        return max(
            self.memory_floor_bytes,
            self.bootstrap_rss_bytes or 0,
            self._peak_observed_rss_bytes,
        )

    def record_worker_rss_sample(self, rss_bytes: int | None) -> None:
        if rss_bytes is not None and rss_bytes > self._peak_observed_rss_bytes:
            self._peak_observed_rss_bytes = rss_bytes


class MemoryBudgetLedger:
    """Thread-safe admission control for concurrently-running waves.

    Live memory alone isn't a safe admission signal once more than one
    wave-launch decision can happen close together: a just-spawned worker
    hasn't ramped up to its real footprint yet, so two near-simultaneous
    checks can both see headroom and both launch, overcommitting real
    memory. `committed_bytes` is debited immediately on reservation --
    before any of that wave's workers consume anything -- and every check
    subtracts the *full* current commitment from the live reading, on the
    pessimistic assumption that none of it has ramped up (shown up as real
    usage) yet. That's deliberately conservative once a wave's workers
    *have* ramped up (their real usage is then counted twice: once via the
    live reading actually dropping, once via the still-held commitment) --
    correct trade for this mechanism, which exists to prevent OOM crashes,
    not to squeeze out maximum utilization. Credited back on release.
    """

    def __init__(
        self,
        *,
        headroom_fraction: float,
        available_memory_bytes_func: Callable[[], int] = available_memory_bytes,
    ) -> None:
        self.headroom_fraction = headroom_fraction
        self._available_memory_bytes_func = available_memory_bytes_func
        self._lock = threading.Lock()
        self._committed_bytes = 0

    def _usable_bytes_locked(self) -> int:
        live_available = self._available_memory_bytes_func()
        return int(live_available * (1.0 - self.headroom_fraction))

    def remaining_budget_bytes(self) -> int:
        """Snapshot of budget not currently committed. A wave sizes itself
        off this, then confirms with try_reserve -- a stale peek just means
        that reserve can fail/retry smaller, never overcommit, since
        try_reserve re-checks for real under the lock.
        """
        with self._lock:
            usable = self._usable_bytes_locked()
            return max(0, usable - self._committed_bytes)

    def try_reserve(self, bytes_needed: int) -> bool:
        with self._lock:
            usable = self._usable_bytes_locked()
            if self._committed_bytes + bytes_needed > usable:
                return False
            self._committed_bytes += bytes_needed
            print(
                f"[optimize] wave_budget action=reserve requested_gb={bytes_needed / 2**30:.2f} "
                f"committed_gb={self._committed_bytes / 2**30:.2f} budget_gb={usable / 2**30:.2f}"
            )
            return True

    def release(self, bytes_amount: int) -> None:
        with self._lock:
            self._committed_bytes = max(0, self._committed_bytes - bytes_amount)
            print(
                f"[optimize] wave_budget action=release requested_gb={bytes_amount / 2**30:.2f} "
                f"committed_gb={self._committed_bytes / 2**30:.2f}"
            )
