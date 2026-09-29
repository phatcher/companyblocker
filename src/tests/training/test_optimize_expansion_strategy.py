from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from company_tokenize import TrainerOptions

from training import optimize_execution, optimize_expansion_strategy
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


def test_select_vocab_frontier_prefers_low_distance_and_keeps_top_k():
    pilot_results = [
        _result(seed=1, vocab=10000, min_frequency=1, distance=0.08, rejected=True),
        _result(seed=1, vocab=20000, min_frequency=1, distance=0.03, rejected=False),
        _result(seed=1, vocab=30000, min_frequency=1, distance=0.01, rejected=False),
        _result(seed=1, vocab=40000, min_frequency=1, distance=0.02, rejected=False),
    ]

    frontier = optimize_execution.select_vocab_frontier(
        pilot_results=pilot_results,
        vocab_schedule=[10000, 20000, 30000, 40000],
        min_frequencies=[1],
        top_k=2,
        distance_tolerance=0.005,
        vocab_sort_key=optimize_execution.vocab_sort_key,
    )

    assert (1, 30000) in frontier
    assert (1, 40000) in frontier
    assert (1, 10000) not in frontier


def test_select_vocab_frontier_keeps_out_of_range_points_out_of_the_top_k_fill():
    # A short pilot line: three points near the target and the auto point far
    # from it. top_k=4 would take all four; max_distance keeps auto out. A line
    # with nothing in range (min_frequency=2) still gets its top_k.
    pilot_results = [
        _result(seed=1, vocab=20000, min_frequency=1, distance=0.016, rejected=False),
        _result(seed=1, vocab=25000, min_frequency=1, distance=0.008, rejected=False),
        _result(seed=1, vocab=30000, min_frequency=1, distance=0.025, rejected=False),
        _result(seed=1, vocab=-1, min_frequency=1, distance=0.177, rejected=False),
        _result(seed=1, vocab=20000, min_frequency=2, distance=0.30, rejected=False),
        _result(seed=1, vocab=-1, min_frequency=2, distance=0.40, rejected=False),
    ]

    frontier = optimize_execution.select_vocab_frontier(
        pilot_results=pilot_results,
        vocab_schedule=[20000, 25000, 30000, -1],
        min_frequencies=[1, 2],
        top_k=4,
        distance_tolerance=0.02,
        vocab_sort_key=optimize_execution.vocab_sort_key,
        max_distance=0.10,
    )

    assert frontier == {
        (1, 20000),
        (1, 25000),
        (1, 30000),
        (2, 20000),
        (2, -1),
    }


def test_combo_prune_reason_detects_unrecoverable_pass_rate_and_consistent_fertility_failures():
    policy = _policy()

    reason = optimize_execution.combo_prune_reason(
        runs=3,
        passes=0,
        fertilities=[1.7, 1.8, 1.75],
        total_seeds=5,
        policy=policy,
    )
    assert reason in {
        "ineligible_pass_rate_ceiling",
        "fertility_outside_range_consistent",
    }

    reason = optimize_execution.combo_prune_reason(
        runs=2,
        passes=2,
        fertilities=[1.2, 1.25],
        total_seeds=5,
        policy=policy,
    )
    assert reason is None


def test_resolve_expansion_strategy_returns_heuristic_by_default():
    strategy = optimize_execution.resolve_expansion_strategy("heuristic")

    assert strategy.name == "heuristic"

    pilot_results = [
        _result(seed=1, vocab=10000, min_frequency=1, distance=0.05, rejected=False),
    ]
    frontier = strategy.select_frontier(
        pilot_results=pilot_results,
        vocab_schedule=[10000],
        min_frequencies=[1],
        top_k=1,
        distance_tolerance=0.0,
    )
    assert frontier == {(1, 10000)}

    assert (
        strategy.prune_reason(
            runs=3,
            passes=0,
            fertilities=[1.7, 1.8, 1.75],
            total_seeds=5,
            policy=_policy(),
        )
        is not None
    )


def test_resolve_expansion_strategy_raises_on_unknown_name():
    with pytest.raises(ValueError, match="Unknown --expansion-strategy"):
        optimize_execution.resolve_expansion_strategy("does-not-exist")


def test_default_record_result_is_a_noop():
    result = _result(
        seed=1, vocab=10000, min_frequency=1, distance=0.01, rejected=False
    )

    assert (
        optimize_expansion_strategy.default_record_result(
            combo=(1, 10000), result=result, expansion_round=1
        )
        is None
    )


def test_select_refining_bracket_uses_the_winners_two_frontier_neighbours():
    frontier = {(1, 10000), (1, 20000), (1, 25000), (1, 30000), (1, 40000)}

    bracket = optimize_expansion_strategy.select_refining_bracket(
        frontier=frontier,
        winning_min_frequency=1,
        winning_vocab_requested=25000,
    )

    assert bracket == (20000, 25000, 30000)


def test_select_refining_bracket_returns_none_for_auto_vocab_winner():
    frontier = {(1, 20000), (1, 25000), (1, 30000)}

    bracket = optimize_expansion_strategy.select_refining_bracket(
        frontier=frontier,
        winning_min_frequency=1,
        winning_vocab_requested=-1,
    )

    assert bracket is None


def test_select_refining_bracket_returns_none_when_winner_is_a_frontier_edge():
    frontier = {(1, 20000), (1, 25000), (1, 30000)}

    bracket = optimize_expansion_strategy.select_refining_bracket(
        frontier=frontier,
        winning_min_frequency=1,
        winning_vocab_requested=30000,
    )

    assert bracket is None


def test_select_refining_bracket_returns_none_with_fewer_than_three_points():
    frontier = {(1, 20000), (1, 25000)}

    bracket = optimize_expansion_strategy.select_refining_bracket(
        frontier=frontier,
        winning_min_frequency=1,
        winning_vocab_requested=25000,
    )

    assert bracket is None


def test_build_crossing_vocab_points_is_the_crossing_and_a_step_either_side():
    points = optimize_expansion_strategy.build_crossing_vocab_points(
        crossing=22704.6, bracket_low=20000, bracket_high=30000, step=100
    )

    assert points == [22605, 22705, 22805]


def test_build_crossing_vocab_points_stays_strictly_inside_the_bracket():
    points = optimize_expansion_strategy.build_crossing_vocab_points(
        crossing=20050, bracket_low=20000, bracket_high=30000, step=100
    )

    assert points == [20050, 20150]


def test_build_crossing_vocab_points_rejects_non_positive_step():
    with pytest.raises(ValueError, match="step must be positive"):
        optimize_expansion_strategy.build_crossing_vocab_points(
            crossing=25000, bracket_low=20000, bracket_high=30000, step=0
        )


def test_build_crossing_vocab_points_rejects_a_non_increasing_bracket():
    with pytest.raises(ValueError, match="bracket_high must be greater"):
        optimize_expansion_strategy.build_crossing_vocab_points(
            crossing=30000, bracket_low=30000, bracket_high=30000, step=1000
        )


def test_expansion_strategy_name_constant_matches_runtime_default():
    context = _context(Path("."))

    assert (
        context.expansion_strategy_name
        == optimize_expansion_strategy.HEURISTIC_STRATEGY_NAME
    )
