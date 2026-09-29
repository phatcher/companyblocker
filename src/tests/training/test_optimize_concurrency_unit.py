import threading

import polars as pl

from training import optimize_concurrency
from training.optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
    compute_target_concurrency,
    historical_worker_rss_estimate,
)

GB = 2**30


def test_sample_process_rss_bytes_returns_positive_int_under_normal_conditions():
    sample = optimize_concurrency.sample_process_rss_bytes()

    assert isinstance(sample, int)
    assert sample > 0


def test_sample_process_rss_bytes_returns_none_on_psutil_failure(mocker):
    mocker.patch.object(
        optimize_concurrency.psutil,
        "Process",
        side_effect=RuntimeError("boom"),
    )

    assert optimize_concurrency.sample_process_rss_bytes() is None


def test_available_memory_bytes_returns_positive_int():
    assert optimize_concurrency.available_memory_bytes() > 0


def test_compute_target_concurrency_scales_with_available_memory():
    target = compute_target_concurrency(
        available_memory_bytes=32 * GB,
        per_candidate_memory_bytes=4 * GB,
        headroom_fraction=0.20,
        hard_cap=8,
    )
    # usable = 32*0.8 = 25.6GB; 25.6 // 4 = 6
    assert target == 6


def test_compute_target_concurrency_clamps_to_hard_cap():
    target = compute_target_concurrency(
        available_memory_bytes=1000 * GB,
        per_candidate_memory_bytes=1 * GB,
        headroom_fraction=0.0,
        hard_cap=8,
    )
    assert target == 8


def test_compute_target_concurrency_clamps_to_min_concurrency_when_memory_scarce():
    target = compute_target_concurrency(
        available_memory_bytes=1,
        per_candidate_memory_bytes=4 * GB,
        headroom_fraction=0.20,
        hard_cap=8,
    )
    assert target == 1


def test_compute_target_concurrency_treats_zero_per_candidate_estimate_as_hard_cap():
    target = compute_target_concurrency(
        available_memory_bytes=4 * GB,
        per_candidate_memory_bytes=0,
        headroom_fraction=0.20,
        hard_cap=8,
    )
    assert target == 8


def test_controller_estimate_prefers_floor_when_nothing_else_known():
    controller = AdaptiveConcurrencyController(hard_cap=8, memory_floor_bytes=2 * GB)

    assert controller.per_candidate_memory_estimate_bytes() == 2 * GB


def test_controller_estimate_prefers_bootstrap_over_floor():
    controller = AdaptiveConcurrencyController(
        hard_cap=8, memory_floor_bytes=2 * GB, bootstrap_rss_bytes=5 * GB
    )

    assert controller.per_candidate_memory_estimate_bytes() == 5 * GB


def test_controller_estimate_prefers_observed_sample_over_bootstrap_and_floor():
    controller = AdaptiveConcurrencyController(
        hard_cap=8, memory_floor_bytes=2 * GB, bootstrap_rss_bytes=5 * GB
    )

    controller.record_worker_rss_sample(9 * GB)

    assert controller.per_candidate_memory_estimate_bytes() == 9 * GB


def test_controller_record_worker_rss_sample_never_lowers_the_estimate():
    controller = AdaptiveConcurrencyController(
        hard_cap=8, memory_floor_bytes=2 * GB, bootstrap_rss_bytes=5 * GB
    )

    controller.record_worker_rss_sample(1 * GB)

    assert controller.per_candidate_memory_estimate_bytes() == 5 * GB


def test_controller_record_worker_rss_sample_ignores_none():
    controller = AdaptiveConcurrencyController(hard_cap=8, memory_floor_bytes=2 * GB)

    controller.record_worker_rss_sample(None)

    assert controller.per_candidate_memory_estimate_bytes() == 2 * GB


def test_historical_worker_rss_estimate_returns_max_of_non_null_values():
    df = pl.DataFrame({"worker_rss_bytes": [1 * GB, 5 * GB, None, 3 * GB]})

    assert historical_worker_rss_estimate(df) == 5 * GB


def test_historical_worker_rss_estimate_returns_none_when_column_absent():
    df = pl.DataFrame({"seed": [1, 2]})

    assert historical_worker_rss_estimate(df) is None


def test_historical_worker_rss_estimate_returns_none_when_all_null():
    df = pl.DataFrame({"worker_rss_bytes": pl.Series([None, None], dtype=pl.Int64)})

    assert historical_worker_rss_estimate(df) is None


def test_memory_budget_ledger_try_reserve_succeeds_under_budget():
    ledger = MemoryBudgetLedger(
        headroom_fraction=0.0, available_memory_bytes_func=lambda: 10 * GB
    )

    assert ledger.try_reserve(4 * GB) is True


def test_memory_budget_ledger_try_reserve_fails_over_budget():
    ledger = MemoryBudgetLedger(
        headroom_fraction=0.0, available_memory_bytes_func=lambda: 10 * GB
    )

    assert ledger.try_reserve(11 * GB) is False


def test_memory_budget_ledger_release_frees_capacity_for_a_later_reservation():
    ledger = MemoryBudgetLedger(
        headroom_fraction=0.0, available_memory_bytes_func=lambda: 10 * GB
    )

    assert ledger.try_reserve(8 * GB) is True
    assert ledger.try_reserve(4 * GB) is False  # only 2GB left

    ledger.release(8 * GB)

    assert ledger.try_reserve(4 * GB) is True


def test_memory_budget_ledger_stays_pessimistic_about_its_own_commitments():
    # Live available reads a fixed 10GB throughout (simulating a just-reserved
    # wave's workers not having ramped up yet, so the live reading hasn't
    # dropped to reflect them). The ledger must NOT add committed_bytes back
    # into the budget -- doing so would let committed grow the budget to
    # match every new commitment, removing the limit entirely (this is
    # exactly the bug this test was written to catch).
    ledger = MemoryBudgetLedger(
        headroom_fraction=0.0, available_memory_bytes_func=lambda: 10 * GB
    )
    assert ledger.try_reserve(8 * GB) is True
    # committed=8GB against a fixed 10GB budget; 4GB more would exceed it,
    # even though the live reading never moved.
    assert ledger.try_reserve(4 * GB) is False
    # But exactly 2GB more still fits.
    assert ledger.try_reserve(2 * GB) is True
    assert ledger.try_reserve(1) is False


def test_memory_budget_ledger_concurrent_try_reserve_never_exceeds_budget():
    # The actual race this mechanism exists to close: many threads
    # reserving concurrently against a fixed budget must never let total
    # committed exceed it, regardless of scheduling order.
    ledger = MemoryBudgetLedger(
        headroom_fraction=0.0, available_memory_bytes_func=lambda: 100 * GB
    )
    successes: list[bool] = []
    successes_lock = threading.Lock()

    def _reserve() -> None:
        result = ledger.try_reserve(1 * GB)
        with successes_lock:
            successes.append(result)

    threads = [threading.Thread(target=_reserve) for _ in range(500)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    granted = sum(1 for result in successes if result)
    assert granted == 100
    assert ledger._committed_bytes == 100 * GB
