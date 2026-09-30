"""Retry/error-policy taxonomy for candidate job execution.

A candidate job (one ``seed``/``vocab_requested``/``min_frequency``
combination, trained and evaluated by ``run_optimize_candidate`` in
``optimize_execution.py``) is deterministic on its own inputs. That means
retrying one unchanged only makes sense where the failure cause is external
to the job's own inputs, never the job itself:

- ``memory_pressure`` (a ``MemoryError``, despite ``AdaptiveConcurrencyController``/
  ``MemoryBudgetLedger`` already trying to prevent it proactively) -- retried
  once, in isolation, at reduced concurrency.
- ``transient_io`` (any other ``OSError`` -- a momentary filesystem race
  against another concurrent job) -- retried a small fixed number of times.
- ``terminal`` (a hang/timeout, or any other exception) -- never retried;
  retrying a deterministic job's own training-logic exception would only
  paper over a real bug, and a hang reproduces identically on the same
  inputs.

This module owns the taxonomy and the failure sidecar log; ``optimize_execution.py``
owns wiring it into both dispatch paths (the fixed pool and the adaptive
wave scheduler).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent import futures
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self

CandidateFailureCategory = Literal["memory_pressure", "transient_io", "terminal"]

CANDIDATE_FAILURE_CATEGORIES: tuple[CandidateFailureCategory, ...] = (
    "memory_pressure",
    "transient_io",
    "terminal",
)

# Total attempts allowed per category, including the first (non-retry)
# attempt -- e.g. memory_pressure's "retried once" is 2 total attempts.
MAX_ATTEMPTS_BY_CATEGORY: dict[CandidateFailureCategory, int] = {
    "memory_pressure": 2,
    "transient_io": 3,
    "terminal": 1,
}

# Worker count a memory_pressure retry runs at. Deliberately the minimum (1),
# not some fraction of the original pool: the point of a reduced-concurrency
# retry is to remove every *other* concurrent candidate as a contributing
# cause, not merely to shrink their number.
MEMORY_PRESSURE_RETRY_WORKERS = 1

# CONSERVATIVE PLACEHOLDER -- not yet anchored to a real measurement. The
# open task is measuring real candidate runtimes to set
# this from evidence; there is no existing real-scale measurement to anchor
# one against yet (unlike RUNTIME_BUDGET_SECONDS in comparison_protocol.py,
# which rests on a measured cell runtime). Set high enough that it should not false-trigger
# on any real candidate (one WordPiece/SentencePiece training run plus its
# two evaluation passes) while still bounding a genuine hang.
#
# Tighten with real data: run, e.g.,
#   uv run python scripts/train_tokenizer.py --mode optimize \
#     --systems ie --pilot-seed-count 1 --vocab-sizes 30000 --min-frequencies 3
# (a single pilot candidate; `ie` is the smallest real corpus -- see
# DEFAULT_VOCAB_GRID's module comment) and read `elapsed_seconds` from the
# resulting run-log row. Multiply the observed figure by a safety factor
# before lowering this constant.
DEFAULT_CANDIDATE_TIMEOUT_SECONDS = 3600.0


def classify_candidate_exception(exc: BaseException) -> CandidateFailureCategory:
    """Classify a candidate job's raised exception into the retry taxonomy.

    Checked in this order because `MemoryError` and `TimeoutError` are not
    `OSError` subclasses in the standard library, so the `OSError` branch
    only ever catches what's left: a genuine transient I/O condition, not a
    hang or a memory condition.
    """
    if isinstance(exc, MemoryError):
        return "memory_pressure"
    if isinstance(exc, TimeoutError):
        # concurrent.futures.TimeoutError is the builtin TimeoutError as of
        # the Python version this repo targets. A future hang is handled by
        # its own caller (a bare `except futures.TimeoutError` around
        # `future.result(timeout=...)`) before this classifier ever runs;
        # this branch only covers a *worker* raising TimeoutError itself.
        return "terminal"
    if isinstance(exc, OSError):
        return "transient_io"
    return "terminal"


@dataclass(frozen=True)
class CandidateFailureRecord:
    """One candidate job's terminal outcome, for the failures sidecar log.

    Recorded only once a job is no longer being retried -- either its
    category never retries (`terminal`), or every attempt its category
    allows was exhausted.
    """

    seed: int
    vocab_requested: int
    min_frequency: int
    category: CandidateFailureCategory
    attempt_count: int
    error_message: str


def candidate_failure_row(record: CandidateFailureRecord) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": record.seed,
        "vocab_requested": record.vocab_requested,
        "min_frequency": record.min_frequency,
        "category": record.category,
        "attempt_count": record.attempt_count,
        "error_message": record.error_message,
    }


def append_candidate_failure_log(
    failure_log_path: Path, failures: list[CandidateFailureRecord]
) -> None:
    """Append terminal failures to the sidecar log, mirroring the JSONL
    sidecar-failure-log pattern `compare_blocking_strategies.py`'s strategy
    comparison sweep already writes for its own per-cell failures
    (`strategy_comparison_failures.jsonl`). A no-op for an empty list, so a
    healthy sweep never creates the file.
    """
    if not failures:
        return
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    with failure_log_path.open("a", encoding="utf-8") as fh:
        for record in failures:
            fh.write(json.dumps(candidate_failure_row(record), sort_keys=True) + "\n")


def default_process_pool_factory(max_workers: int) -> futures.Executor:
    return futures.ProcessPoolExecutor(max_workers=max_workers)


class RecreatableExecutorPool:
    """A process-pool handle that can be discarded and replaced.

    `concurrent.futures` has no supported way to kill a hung worker
    mid-flight: the only lever available on a timeout is to stop using the
    executor that hosts it and hand out a fresh one for further
    submissions, accepting that the hung worker keeps running in the
    background until the interpreter exits. Used only by the fixed-pool
    dispatch path (`--disable-adaptive-concurrency`); the adaptive wave
    scheduler already gets equivalent behaviour for free from its own
    short-lived per-wave pools.
    """

    def __init__(
        self,
        *,
        max_workers: int,
        executor_factory: Callable[[int], futures.Executor] = (
            default_process_pool_factory
        ),
    ) -> None:
        self.max_workers = max_workers
        self.executor_factory = executor_factory
        self._executor = executor_factory(max_workers)

    @property
    def current(self) -> futures.Executor:
        return self._executor

    def recreate(self) -> None:
        """Discard the current pool without waiting on any hung worker, and
        replace it with a fresh one before further jobs are submitted.
        """
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = self.executor_factory(self.max_workers)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown(wait=True)
