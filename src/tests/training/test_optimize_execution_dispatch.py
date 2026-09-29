from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import polars as pl
from company_tokenize import TrainerOptions

from training import optimize_execution, optimize_retry
from training.optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
)
from training.runtime import OptimizeExecutionTracker, OptimizeRunContext

_ComboProgress = dict[tuple[int, int], optimize_execution.ComboProgressState]


def _policy() -> SimpleNamespace:
    return SimpleNamespace(
        fertility_target=1.25,
        unk_rate_threshold=0.005,
        fertility_tolerance=0.10,
        fertility_min=1.05,
        fertility_max=1.60,
        token_count_median_max=4.0,
        token_count_p95_max=6.0,
        eligibility_pass_rate=0.8,
        diagnostic_top_n=20,
        delete_rejected_models=True,
    )


def _result(
    seed: int, vocab: int, min_frequency: int, distance: float, rejected: bool
) -> SimpleNamespace:
    return SimpleNamespace(
        trainer="wordpiece",
        seed=seed,
        vocab_requested=vocab,
        min_frequency=min_frequency,
        train_rows=10,
        validation_rows=2,
        resolved_vocab_size=40000,
        train_metrics={
            "fertility": 1.2,
            "unk_rate": 0.001,
            "fertility_distance": 0.0,
            "single_char_token_pct": 0.0,
            "token_count_mean": 1.0,
            "token_count_median": 1.0,
            "token_count_p95": 1.0,
        },
        validation_metrics={
            "fertility": 1.2,
            "unk_rate": 0.001,
            "fertility_distance": distance,
            "single_char_token_pct": 0.0,
            "token_count_mean": 1.0,
            "token_count_median": 1.0,
            "token_count_p95": 1.0,
        },
        fertility_generalization_delta=0.0,
        unk_rate_generalization_delta=0.0,
        model_path="m",
        rejected=rejected,
        rejection_reason="r" if rejected else "",
        selection_score=1.0,
        elapsed_seconds=0.1,
    )


def _context(
    tmp_path: Path,
    *,
    reuse_existing_candidates: bool = False,
    excluded_candidate_keys: set[tuple[int, int, int]] | None = None,
    total_candidate_jobs: int = 10,
    max_workers: int = 2,
    elbow_min_points: int = 1,
    elbow_min_improvement: float = 0.001,
) -> OptimizeRunContext:
    return OptimizeRunContext(
        args=argparse.Namespace(
            tokenizer="wordpiece",
            elbow_min_points=elbow_min_points,
            elbow_min_improvement=elbow_min_improvement,
            validation_fraction=0.2,
        ),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        session_id="sess-1",
        target_scope="country",
        systems=["fr"],
        shard_run_log_path=tmp_path / "shard.parquet",
        models_dir=tmp_path / "models",
        reuse_existing_candidates=reuse_existing_candidates,
        excluded_candidate_keys=excluded_candidate_keys or set(),
        total_candidate_jobs=total_candidate_jobs,
        max_workers=max_workers,
    )


def _tracker() -> OptimizeExecutionTracker:
    return OptimizeExecutionTracker(session_run_log_df=pl.DataFrame())


def _disabled_concurrency_controller() -> AdaptiveConcurrencyController:
    # enabled=False reduces run_candidate_batch to exactly the old
    # submit_optimize_candidate_jobs + drain_candidate_futures pair, so every
    # pre-Phase-2 test keeps its original behavior unchanged.
    return AdaptiveConcurrencyController(
        hard_cap=2, memory_floor_bytes=1, enabled=False
    )


def _inert_memory_budget_ledger() -> MemoryBudgetLedger:
    # Never actually consulted when the controller above is disabled, but
    # every phase-function call site requires one.
    return MemoryBudgetLedger(headroom_fraction=0.20)


def _split(
    train_rows: int = 10, validation_rows: int = 2
) -> optimize_execution.OptimizeSeedSplit:
    return {
        "train_split_path": Path("train.parquet"),
        "validation_split_path": Path("validation.parquet"),
        "train_rows": train_rows,
        "validation_rows": validation_rows,
    }


def _fake_run_candidate(distances: dict[tuple[int, int, int], float] | None = None):
    distances = distances or {}

    def _run(
        *,
        trainer,
        train_split_path,
        validation_split_path,
        seed,
        vocab_requested,
        min_frequency,
        train_rows,
        validation_rows,
        policy,
        run_model_path,
        trainer_options,
    ):
        key = (seed, vocab_requested, min_frequency)
        distance = distances.get(key, 0.05)
        return _result(
            seed=seed,
            vocab=vocab_requested,
            min_frequency=min_frequency,
            distance=distance,
            rejected=distance > 0.5,
        )

    return _run


def _write_stub_corpus_and_count(corpus_path: object, **kwargs: object) -> int:
    """Write a one-row stub corpus and report its row count.

    A named function rather than a lambda: `write_parquet()` returns `None`, so
    sequencing it inside a tuple and indexing the result reads as using a value
    that does not exist.
    """
    pl.DataFrame({"name": ["a"]}).write_parquet(corpus_path)  # type: ignore[arg-type]
    return 10


def test_collect_candidate_jobs_for_pilot_vocab_skips_excluded_only(
    tmp_path: Path,
):
    context = _context(
        tmp_path,
        reuse_existing_candidates=True,
        excluded_candidate_keys={(1, 100, 2)},
    )
    seed_splits = {1: _split(), 2: _split()}

    jobs, skipped = optimize_execution.collect_candidate_jobs_for_pilot_vocab(
        context=context,
        vocab_requested=100,
        pilot_seeds=[1, 2],
        min_frequencies=[1, 2],
        seed_splits=seed_splits,
    )

    # Elbow-cutoff state (if any) never excludes a combo from the pilot -- the
    # full vocab grid is always evaluated for every seed/min_frequency combo.
    # Only an already-existing fresh candidate output is skipped.
    job_keys = {(job["seed"], job["min_frequency"]) for job in jobs}
    assert (1, 2) not in job_keys
    assert (1, 1) in job_keys
    assert (2, 1) in job_keys
    assert (2, 2) in job_keys
    assert skipped == 1


def test_collect_candidate_jobs_for_expansion_seed_builds_jobs_for_active_combos(
    tmp_path: Path,
):
    context = _context(tmp_path)

    jobs, skipped = optimize_execution.collect_candidate_jobs_for_expansion_seed(
        context=context,
        seed=5,
        active_combos=[(1, 100), (2, 200)],
        split=_split(),
    )

    assert skipped == 0
    assert {(job["min_frequency"], job["vocab_requested"]) for job in jobs} == {
        (1, 100),
        (2, 200),
    }


def test_submit_and_drain_candidate_futures_updates_tracker_and_run_log(
    tmp_path: Path,
):
    context = _context(tmp_path)
    tracker = _tracker()
    jobs = [
        optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
    ]

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_map = optimize_execution.submit_optimize_candidate_jobs(
            executor=executor,
            candidate_jobs=jobs,
            context=context,
            run_candidate=_fake_run_candidate(),
        )
        results = optimize_execution.drain_candidate_futures(
            future_map=future_map,
            context=context,
            tracker=tracker,
            run_candidate=_fake_run_candidate(),
            include_in_pilot=True,
        )

    assert len(results) == 1
    assert tracker.completed_jobs == 1
    assert tracker.pilot_results == results
    assert tracker.candidate_results == results
    assert context.shard_run_log_path.exists()


class _FlakyRunCandidate:
    """A run_candidate that raises exception_factory() the first fail_times
    calls, then delegates to a real _fake_run_candidate() -- for asserting a
    retried job's eventual output matches an unretried success's.
    """

    def __init__(self, *, fail_times: int, exception_factory):
        self.fail_times = fail_times
        self.exception_factory = exception_factory
        self.call_count = 0
        self._succeed = _fake_run_candidate()

    def __call__(self, **kwargs):
        self.call_count += 1
        if self.call_count <= self.fail_times:
            raise self.exception_factory()
        return self._succeed(**kwargs)


def _flaky_run_candidate(*, fail_times: int, exception_factory) -> _FlakyRunCandidate:
    return _FlakyRunCandidate(
        fail_times=fail_times, exception_factory=exception_factory
    )


class TestDrainCandidateFuturesRetryPolicy:
    def _thread_factory(self, max_workers: int) -> futures.Executor:
        return futures.ThreadPoolExecutor(max_workers=max_workers)

    def test_terminal_failure_is_recorded_and_sweep_continues(self, tmp_path: Path):
        context = _context(tmp_path)
        tracker = _tracker()
        failing_job = optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
        healthy_job = optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=2,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )

        def _run(**kwargs):
            if kwargs["seed"] == 1:
                raise ValueError("bad training-logic input")
            return _fake_run_candidate()(**kwargs)

        failure_log_path = tmp_path / "candidate_failures.jsonl"
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_map = optimize_execution.submit_optimize_candidate_jobs(
                executor=executor,
                candidate_jobs=[failing_job, healthy_job],
                context=context,
                run_candidate=_run,
            )
            results = optimize_execution.drain_candidate_futures(
                future_map=future_map,
                context=context,
                tracker=tracker,
                run_candidate=_run,
                include_in_pilot=True,
                failure_log_path=failure_log_path,
                retry_executor_factory=self._thread_factory,
            )

        # The sweep continues: the healthy job's result is still recorded
        # despite the other job's terminal failure -- never crashes the batch.
        assert len(results) == 1
        assert results[0].seed == 2
        assert tracker.completed_jobs == 1

        lines = failure_log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        failure_row = json.loads(lines[0])
        assert failure_row["seed"] == 1
        assert failure_row["category"] == "terminal"
        assert failure_row["attempt_count"] == 1
        assert "bad training-logic input" in failure_row["error_message"]

    def test_transient_io_failure_retries_and_matches_unretried_success(
        self, tmp_path: Path
    ):
        context = _context(tmp_path)
        tracker = _tracker()
        job = optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
        flaky = _flaky_run_candidate(
            fail_times=1, exception_factory=lambda: OSError("disk busy")
        )
        baseline = _fake_run_candidate()(
            trainer="wordpiece",
            train_split_path=job["train_split_path"],
            validation_split_path=job["validation_split_path"],
            seed=job["seed"],
            vocab_requested=job["vocab_requested"],
            min_frequency=job["min_frequency"],
            train_rows=job["train_rows"],
            validation_rows=job["validation_rows"],
            policy=_policy(),
            run_model_path=job["run_model_path"],
            trainer_options=TrainerOptions(),
        )

        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_map = optimize_execution.submit_optimize_candidate_jobs(
                executor=executor,
                candidate_jobs=[job],
                context=context,
                run_candidate=flaky,
            )
            results = optimize_execution.drain_candidate_futures(
                future_map=future_map,
                context=context,
                tracker=tracker,
                run_candidate=flaky,
                include_in_pilot=True,
                retry_executor_factory=self._thread_factory,
            )

        assert flaky.call_count == 2  # one failed attempt, one retry
        assert len(results) == 1
        retried_result = results[0]
        # Retry logic must not perturb the seed or any other input: a
        # retried job's output is identical to an unretried success on the
        # same inputs.
        assert retried_result.seed == baseline.seed
        assert retried_result.vocab_requested == baseline.vocab_requested
        assert retried_result.min_frequency == baseline.min_frequency
        assert retried_result.resolved_vocab_size == baseline.resolved_vocab_size
        assert retried_result.validation_metrics == baseline.validation_metrics
        assert retried_result.rejected == baseline.rejected
        assert retried_result.selection_score == baseline.selection_score

    def test_memory_pressure_failure_retries_once_then_records_terminal(
        self, tmp_path: Path
    ):
        context = _context(tmp_path)
        tracker = _tracker()
        job = optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
        always_oom = _flaky_run_candidate(
            fail_times=99, exception_factory=lambda: MemoryError("oom")
        )
        retry_worker_counts: list[int] = []

        def _spy_thread_factory(max_workers: int) -> futures.Executor:
            retry_worker_counts.append(max_workers)
            return futures.ThreadPoolExecutor(max_workers=max_workers)

        failure_log_path = tmp_path / "candidate_failures.jsonl"
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_map = optimize_execution.submit_optimize_candidate_jobs(
                executor=executor,
                candidate_jobs=[job],
                context=context,
                run_candidate=always_oom,
            )
            results = optimize_execution.drain_candidate_futures(
                future_map=future_map,
                context=context,
                tracker=tracker,
                run_candidate=always_oom,
                include_in_pilot=True,
                failure_log_path=failure_log_path,
                retry_executor_factory=_spy_thread_factory,
            )

        # memory_pressure allows exactly one retry (2 total attempts), unlike
        # transient_io's small fixed count.
        assert always_oom.call_count == 2
        assert results == []
        assert retry_worker_counts == [optimize_execution.MEMORY_PRESSURE_RETRY_WORKERS]

        failure_row = json.loads(
            failure_log_path.read_text(encoding="utf-8").splitlines()[0]
        )
        assert failure_row["category"] == "memory_pressure"
        assert failure_row["attempt_count"] == 2

    def test_primary_timeout_is_terminal_and_invokes_on_hang(self, tmp_path: Path):
        context = _context(tmp_path)
        tracker = _tracker()
        job = optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )

        def _slow_run(**kwargs):
            time.sleep(0.2)
            return _fake_run_candidate()(**kwargs)

        hang_calls = {"count": 0}

        def _on_hang() -> None:
            hang_calls["count"] += 1

        failure_log_path = tmp_path / "candidate_failures.jsonl"
        with futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_map = optimize_execution.submit_optimize_candidate_jobs(
                executor=executor,
                candidate_jobs=[job],
                context=context,
                run_candidate=_slow_run,
            )
            results = optimize_execution.drain_candidate_futures(
                future_map=future_map,
                context=context,
                tracker=tracker,
                run_candidate=_slow_run,
                include_in_pilot=True,
                timeout_seconds=0.01,
                failure_log_path=failure_log_path,
                on_hang=_on_hang,
            )

        assert results == []
        assert hang_calls["count"] == 1
        failure_row = json.loads(
            failure_log_path.read_text(encoding="utf-8").splitlines()[0]
        )
        assert failure_row["category"] == "terminal"
        assert "runtime budget" in failure_row["error_message"]


class TestRunCandidateBatchFixedPoolHangRecovery:
    def test_timeout_recreates_pool_for_the_next_batch(self, tmp_path: Path):
        context = _context(tmp_path)
        tracker = _tracker()
        hanging_job = optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
        healthy_job = optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=2,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )

        def _run(**kwargs):
            if kwargs["seed"] == 1:
                time.sleep(0.2)
            return _fake_run_candidate()(**kwargs)

        pool = optimize_retry.RecreatableExecutorPool(
            max_workers=2,
            executor_factory=lambda max_workers: futures.ThreadPoolExecutor(
                max_workers=max_workers
            ),
        )
        try:
            first_executor = pool.current
            batch_one = optimize_execution.run_candidate_batch(
                executor=pool,
                candidate_jobs=[hanging_job],
                context=OptimizeRunContext(
                    **{**context.__dict__, "candidate_timeout_seconds": 0.01}
                ),
                tracker=tracker,
                run_candidate=_run,
                concurrency_controller=_disabled_concurrency_controller(),
                memory_budget_ledger=_inert_memory_budget_ledger(),
                wave_chunk_size=4,
                max_parallel_waves=2,
                include_in_pilot=True,
            )
            assert batch_one == []
            # The hang tore down and replaced the pool before any further
            # batch's jobs are submitted -- resolved fresh on every call, so
            # the *next* run_candidate_batch call picks up the new one.
            assert pool.current is not first_executor

            batch_two = optimize_execution.run_candidate_batch(
                executor=pool,
                candidate_jobs=[healthy_job],
                context=context,
                tracker=tracker,
                run_candidate=_run,
                concurrency_controller=_disabled_concurrency_controller(),
                memory_budget_ledger=_inert_memory_budget_ledger(),
                wave_chunk_size=4,
                max_parallel_waves=2,
                include_in_pilot=True,
            )
            assert len(batch_two) == 1
            assert batch_two[0].seed == 2
        finally:
            pool.shutdown()


def test_drain_candidate_futures_records_worker_rss_and_available_memory(
    tmp_path: Path, capsys
):
    context = _context(tmp_path)
    tracker = _tracker()
    jobs = [
        optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
    ]

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_map = optimize_execution.submit_optimize_candidate_jobs(
            executor=executor,
            candidate_jobs=jobs,
            context=context,
            run_candidate=_fake_run_candidate(),
        )
        optimize_execution.drain_candidate_futures(
            future_map=future_map,
            context=context,
            tracker=tracker,
            run_candidate=_fake_run_candidate(),
            include_in_pilot=True,
        )

    assert tracker.session_run_log_df.height == 1
    row = tracker.session_run_log_df.row(0, named=True)
    assert "worker_rss_bytes" in row
    assert "system_available_bytes" in row
    assert row["system_available_bytes"] > 0
    assert isinstance(row["worker_pid"], int)

    captured = capsys.readouterr()
    assert "[optimize] worker_rss seed=1 vocab=100 min_freq=1" in captured.out
    assert "pid=" in captured.out


def test_record_completed_candidate_result_writes_worker_pid_into_run_row(
    tmp_path: Path,
):
    context = _context(tmp_path)
    tracker = _tracker()
    result = _result(seed=1, vocab=100, min_frequency=1, distance=0.05, rejected=False)

    optimize_execution.record_completed_candidate_result(
        result=result,
        worker_rss_bytes=123_456,
        worker_pid=98765,
        context=context,
        tracker=tracker,
        include_in_pilot=True,
    )

    row = tracker.session_run_log_df.row(0, named=True)
    assert row["worker_rss_bytes"] == 123_456
    assert row["worker_pid"] == 98765


def test_record_completed_candidate_result_reports_missing_rss_and_excludes_expansion_from_pilot(
    tmp_path: Path, capsys
):
    # Two degenerate-progress edges at once: worker_rss_bytes can be None (the
    # RSS sample failed/wasn't available), which must render as "n/a" rather
    # than crashing on a None/int division; and an expansion-phase result
    # (include_in_pilot=False) must still update candidate_results/
    # completed_jobs bookkeeping while leaving pilot_results untouched.
    context = _context(tmp_path)
    tracker = _tracker()
    result = _result(seed=3, vocab=200, min_frequency=1, distance=0.05, rejected=False)

    optimize_execution.record_completed_candidate_result(
        result=result,
        worker_rss_bytes=None,
        worker_pid=4242,
        context=context,
        tracker=tracker,
        include_in_pilot=False,
    )

    assert tracker.pilot_results == []
    assert len(tracker.candidate_results) == 1
    assert tracker.completed_jobs == 1
    row = tracker.session_run_log_df.row(0, named=True)
    assert row["worker_rss_bytes"] is None
    assert "rss_gb=n/a" in capsys.readouterr().out


def test_run_candidate_batch_dispatches_to_fixed_pair_when_disabled(
    tmp_path: Path, mocker
):
    context = _context(tmp_path)
    tracker = _tracker()
    jobs = [
        optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
    ]
    waves_spy = mocker.patch.object(optimize_execution, "run_candidate_jobs_in_waves")

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = optimize_execution.run_candidate_batch(
            executor=executor,
            candidate_jobs=jobs,
            context=context,
            tracker=tracker,
            run_candidate=_fake_run_candidate(),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
            include_in_pilot=True,
        )

    waves_spy.assert_not_called()
    assert len(results) == 1
    assert tracker.completed_jobs == 1


def test_run_candidate_batch_dispatches_to_waves_when_enabled(tmp_path: Path, mocker):
    context = _context(tmp_path)
    tracker = _tracker()
    jobs = [
        optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
    ]
    controller = AdaptiveConcurrencyController(hard_cap=2, memory_floor_bytes=1)
    ledger = _inert_memory_budget_ledger()
    waves_spy = mocker.patch.object(
        optimize_execution, "run_candidate_jobs_in_waves", return_value=["sentinel"]
    )

    results = optimize_execution.run_candidate_batch(
        executor=None,
        candidate_jobs=jobs,
        context=context,
        tracker=tracker,
        run_candidate=_fake_run_candidate(),
        concurrency_controller=controller,
        memory_budget_ledger=ledger,
        wave_chunk_size=4,
        max_parallel_waves=2,
        include_in_pilot=True,
    )

    waves_spy.assert_called_once_with(
        candidate_jobs=jobs,
        context=context,
        tracker=tracker,
        run_candidate=mocker.ANY,
        concurrency_controller=controller,
        memory_budget_ledger=ledger,
        wave_chunk_size=4,
        max_parallel_waves=2,
        include_in_pilot=True,
        expansion_round=None,
        phase_override=None,
    )
    assert results == ["sentinel"]


def test_run_candidate_jobs_in_waves_uses_more_than_one_wave_for_a_small_chunk_size(
    tmp_path: Path,
):
    context = _context(tmp_path)
    tracker = _tracker()
    jobs = [
        optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=seed,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
        for seed in range(1, 7)  # 6 jobs
    ]
    controller = AdaptiveConcurrencyController(hard_cap=4, memory_floor_bytes=1)
    ledger = MemoryBudgetLedger(
        headroom_fraction=0.0, available_memory_bytes_func=lambda: 10**12
    )

    results = optimize_execution.run_candidate_jobs_in_waves(
        candidate_jobs=jobs,
        context=context,
        tracker=tracker,
        run_candidate=_fake_run_candidate(),
        concurrency_controller=controller,
        memory_budget_ledger=ledger,
        wave_chunk_size=2,  # forces 3 waves for 6 jobs
        max_parallel_waves=4,
        include_in_pilot=True,
        executor_factory=lambda max_workers: futures.ThreadPoolExecutor(
            max_workers=max_workers
        ),
    )

    assert len(results) == 6
    assert {result.seed for result in results} == {1, 2, 3, 4, 5, 6}
    assert tracker.completed_jobs == 6
    # Every reservation must have been released by the time all waves finish.
    assert ledger._committed_bytes == 0


def test_run_candidate_jobs_in_waves_runs_strictly_sequentially_with_one_parallel_wave(
    tmp_path: Path,
):
    context = _context(tmp_path)
    tracker = _tracker()
    jobs = [
        optimize_execution.build_optimize_candidate_job(
            trainer="wordpiece",
            models_dir=context.models_dir,
            seed=seed,
            vocab_requested=100,
            min_frequency=1,
            split=_split(),
        )
        for seed in range(1, 5)
    ]
    controller = AdaptiveConcurrencyController(hard_cap=4, memory_floor_bytes=1)
    ledger = MemoryBudgetLedger(
        headroom_fraction=0.0, available_memory_bytes_func=lambda: 10**12
    )
    in_flight = 0
    max_observed_in_flight = 0
    lock = threading.Lock()

    def _tracking_run_candidate(**kwargs):
        nonlocal in_flight, max_observed_in_flight
        with lock:
            in_flight += 1
            max_observed_in_flight = max(max_observed_in_flight, in_flight)
        try:
            return _fake_run_candidate()(**kwargs)
        finally:
            with lock:
                in_flight -= 1

    results = optimize_execution.run_candidate_jobs_in_waves(
        candidate_jobs=jobs,
        context=context,
        tracker=tracker,
        run_candidate=_tracking_run_candidate,
        concurrency_controller=controller,
        memory_budget_ledger=ledger,
        wave_chunk_size=1,
        max_parallel_waves=1,
        include_in_pilot=True,
        executor_factory=lambda max_workers: futures.ThreadPoolExecutor(
            max_workers=max_workers
        ),
    )

    assert len(results) == 4
    assert max_observed_in_flight == 1


def test_log_optimize_progress_prints_unknown_eta_without_history(capsys):
    optimize_execution.log_optimize_progress(
        phase="pilot",
        completed_jobs=0,
        submitted_jobs=0,
        pending_batch_jobs=2,
        upper_bound_jobs=10,
        max_workers=2,
        observed_job_elapsed_seconds=[],
    )

    assert "eta_submitted=unknown" in capsys.readouterr().out


def test_log_optimize_progress_computes_eta_from_history(capsys):
    optimize_execution.log_optimize_progress(
        phase="pilot",
        completed_jobs=1,
        submitted_jobs=2,
        pending_batch_jobs=1,
        upper_bound_jobs=10,
        max_workers=2,
        observed_job_elapsed_seconds=[1.0, 2.0, 3.0],
    )

    out = capsys.readouterr().out
    assert "eta_submitted=" in out
    assert "unknown" not in out


def test_log_optimize_progress_reports_zero_eta_when_run_already_complete(capsys):
    # completed_jobs has caught up with (or overrun) both the submitted+pending
    # count and the upper bound -- e.g. the final progress line of a finished
    # sweep, or a reused/interrupted run resuming past where it left off.
    # remaining_* must clamp to 0 rather than go negative, and the resulting
    # ETA should read as "already done" (00:00:00), not "unknown".
    optimize_execution.log_optimize_progress(
        phase="expansion",
        completed_jobs=12,
        submitted_jobs=10,
        pending_batch_jobs=0,
        upper_bound_jobs=10,
        max_workers=2,
        observed_job_elapsed_seconds=[1.0, 2.0],
    )

    out = capsys.readouterr().out
    assert "remaining_submitted=0" in out
    assert "remaining_upper_bound=0" in out
    assert "eta_submitted=00:00:00" in out
    assert "eta_upper_bound=00:00:00" in out


def test_log_optimize_progress_stays_unknown_when_max_workers_is_zero(capsys):
    # Guards the max_workers > 0 gate: with job-elapsed history present but no
    # workers configured, ETA must stay "unknown" instead of dividing by zero
    # parallelism.
    optimize_execution.log_optimize_progress(
        phase="pilot",
        completed_jobs=1,
        submitted_jobs=2,
        pending_batch_jobs=1,
        upper_bound_jobs=10,
        max_workers=0,
        observed_job_elapsed_seconds=[1.0, 2.0, 3.0],
    )

    out = capsys.readouterr().out
    assert "eta_submitted=unknown" in out
    assert "eta_upper_bound=unknown" in out
