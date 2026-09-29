"""State shared across one optimize run: `OptimizeExecutionTracker`, `OptimizeRunContext`, and the ETA and worker-count helpers."""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from company_tokenize import (
    OptimizeCandidatePolicy,
    OptimizeCandidateResult,
    TrainerOptions,
)

from .optimize_retry import DEFAULT_CANDIDATE_TIMEOUT_SECONDS


@dataclass
class OptimizeExecutionTracker:
    session_run_log_df: pl.DataFrame
    candidate_results: list[OptimizeCandidateResult] = field(default_factory=list)
    pilot_results: list[OptimizeCandidateResult] = field(default_factory=list)
    completed_jobs: int = 0
    submitted_jobs: int = 0
    skipped_existing_candidates_total: int = 0
    observed_job_elapsed_seconds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizeRunContext:
    args: argparse.Namespace
    trainer_options: TrainerOptions
    candidate_policy: OptimizeCandidatePolicy
    session_id: str
    target_scope: str
    systems: list[str]
    shard_run_log_path: Path
    models_dir: Path
    reuse_existing_candidates: bool
    excluded_candidate_keys: set[tuple[int, int, int]]
    total_candidate_jobs: int
    max_workers: int
    # Must match optimize_expansion_strategy.HEURISTIC_STRATEGY_NAME -- not
    # imported here to keep this module dependency-light; kept in sync by
    # test_expansion_strategy_name_constant_matches_runtime_default.
    expansion_strategy_name: str = "heuristic"
    # Per-job future.result(timeout=...) budget and the sidecar
    # failure log a terminal candidate is recorded to. Defaulted (rather than
    # required) so existing call sites/tests that build a context without
    # them keep working unchanged; optimize_retry.DEFAULT_CANDIDATE_TIMEOUT_SECONDS
    # is a placeholder, not yet anchored to a real measurement -- see that
    # constant's own docstring for the command to tighten it.
    candidate_timeout_seconds: float = DEFAULT_CANDIDATE_TIMEOUT_SECONDS
    candidate_failure_log_path: Path | None = None


def resolve_optimize_max_workers(requested: int | None, job_count: int) -> int:
    if job_count <= 0:
        return 1
    if requested is not None:
        if requested <= 0:
            raise ValueError("--max-workers must be greater than zero when provided.")
        return min(requested, job_count)

    cpu_count = os.cpu_count() or 1
    default_workers = max(1, cpu_count - 1)
    return min(default_workers, job_count)


def format_eta_hms(total_seconds: float | None) -> str:
    if total_seconds is None or total_seconds < 0:
        return "unknown"
    rounded = round(total_seconds)
    hours, rem = divmod(rounded, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def percentile_75(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_elapsed = sorted(values)
    p75_index = min(
        len(sorted_elapsed) - 1, max(0, int(math.ceil(len(sorted_elapsed) * 0.75) - 1))
    )
    return sorted_elapsed[p75_index]
