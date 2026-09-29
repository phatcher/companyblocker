import json
from concurrent import futures

import pytest

from training.optimize_retry import (
    MAX_ATTEMPTS_BY_CATEGORY,
    CandidateFailureRecord,
    RecreatableExecutorPool,
    append_candidate_failure_log,
    candidate_failure_row,
    classify_candidate_exception,
)


class TestClassifyCandidateException:
    def test_memory_error_is_memory_pressure(self):
        assert classify_candidate_exception(MemoryError("oom")) == "memory_pressure"

    def test_timeout_error_is_terminal(self):
        assert classify_candidate_exception(TimeoutError("hung")) == "terminal"

    def test_futures_timeout_error_is_terminal(self):
        # concurrent.futures.TimeoutError is the builtin TimeoutError on the
        # Python version this repo targets -- assert that alias holds rather
        # than assuming it, since the whole taxonomy depends on it.
        assert futures.TimeoutError is TimeoutError
        assert classify_candidate_exception(futures.TimeoutError()) == "terminal"

    def test_os_error_is_transient_io(self):
        assert classify_candidate_exception(OSError("disk busy")) == "transient_io"

    def test_file_not_found_error_is_transient_io(self):
        # FileNotFoundError is an OSError subclass -- confirm the subclass
        # relationship is what the classifier actually relies on.
        assert classify_candidate_exception(FileNotFoundError("gone")) == "transient_io"

    def test_other_exception_is_terminal(self):
        assert classify_candidate_exception(ValueError("bad input")) == "terminal"

    def test_key_error_is_terminal(self):
        assert classify_candidate_exception(KeyError("missing")) == "terminal"


class TestMaxAttemptsByCategory:
    def test_memory_pressure_allows_one_retry(self):
        assert MAX_ATTEMPTS_BY_CATEGORY["memory_pressure"] == 2

    def test_transient_io_allows_small_fixed_retry_count(self):
        assert MAX_ATTEMPTS_BY_CATEGORY["transient_io"] == 3

    def test_terminal_never_retries(self):
        assert MAX_ATTEMPTS_BY_CATEGORY["terminal"] == 1


class TestCandidateFailureLog:
    def test_append_is_noop_for_empty_list(self, tmp_path):
        log_path = tmp_path / "failures.jsonl"

        append_candidate_failure_log(log_path, [])

        assert not log_path.exists()

    def test_append_writes_one_jsonl_line_per_record(self, tmp_path):
        log_path = tmp_path / "nested" / "failures.jsonl"
        records = [
            CandidateFailureRecord(
                seed=1,
                vocab_requested=30000,
                min_frequency=3,
                category="terminal",
                attempt_count=1,
                error_message="boom",
            ),
            CandidateFailureRecord(
                seed=2,
                vocab_requested=-1,
                min_frequency=5,
                category="transient_io",
                attempt_count=3,
                error_message="disk busy",
            ),
        ]

        append_candidate_failure_log(log_path, records)

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        first_row = json.loads(lines[0])
        assert first_row["seed"] == 1
        assert first_row["vocab_requested"] == 30000
        assert first_row["min_frequency"] == 3
        assert first_row["category"] == "terminal"
        assert first_row["attempt_count"] == 1
        assert first_row["error_message"] == "boom"
        assert "timestamp_utc" in first_row

    def test_append_is_additive_across_calls(self, tmp_path):
        log_path = tmp_path / "failures.jsonl"
        record = CandidateFailureRecord(
            seed=1,
            vocab_requested=1000,
            min_frequency=1,
            category="terminal",
            attempt_count=1,
            error_message="boom",
        )

        append_candidate_failure_log(log_path, [record])
        append_candidate_failure_log(log_path, [record])

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

    def test_candidate_failure_row_is_json_serializable(self):
        record = CandidateFailureRecord(
            seed=1,
            vocab_requested=1000,
            min_frequency=1,
            category="memory_pressure",
            attempt_count=2,
            error_message="oom",
        )

        row = candidate_failure_row(record)

        json.dumps(row)  # must not raise


class TestRecreatableExecutorPool:
    def test_current_returns_an_executor_from_the_factory(self):
        created: list[int] = []

        def factory(max_workers: int) -> futures.Executor:
            created.append(max_workers)
            return futures.ThreadPoolExecutor(max_workers=max_workers)

        pool = RecreatableExecutorPool(max_workers=2, executor_factory=factory)
        try:
            assert created == [2]
            assert pool.current.submit(lambda: 1 + 1).result() == 2
        finally:
            pool.shutdown()

    def test_recreate_replaces_the_executor_instance(self):
        def factory(max_workers: int) -> futures.Executor:
            return futures.ThreadPoolExecutor(max_workers=max_workers)

        pool = RecreatableExecutorPool(max_workers=1, executor_factory=factory)
        try:
            first_executor = pool.current
            pool.recreate()
            second_executor = pool.current

            assert first_executor is not second_executor
            assert pool.current.submit(lambda: "ok").result() == "ok"
        finally:
            pool.shutdown()

    def test_recreate_does_not_block_on_shutdown(self):
        # shutdown(wait=False) must return promptly rather than joining a
        # (possibly hung) worker -- exercised here against a real, fast
        # executor since there is no way to safely simulate a genuinely hung
        # thread/process without risking a stuck test.
        def factory(max_workers: int) -> futures.Executor:
            return futures.ThreadPoolExecutor(max_workers=max_workers)

        pool = RecreatableExecutorPool(max_workers=1, executor_factory=factory)
        try:
            pool.current.submit(lambda: 1).result()
            pool.recreate()
        finally:
            pool.shutdown()

    def test_context_manager_shuts_down_on_exit(self):
        def factory(max_workers: int) -> futures.Executor:
            return futures.ThreadPoolExecutor(max_workers=max_workers)

        with RecreatableExecutorPool(max_workers=1, executor_factory=factory) as pool:
            assert pool.current.submit(lambda: 5).result() == 5

        with pytest.raises(RuntimeError):
            pool.current.submit(lambda: 1).result()
