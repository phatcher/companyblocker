from __future__ import annotations

import argparse
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import polars as pl
from company_tokenize import TrainerOptions

from training import optimize_execution
from training.optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
)
from training.runtime import OptimizeExecutionTracker, OptimizeRunContext


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


def _history_result(
    *,
    seed: int,
    vocab: int,
    min_frequency: int,
    train_distance: float,
    trained_vocab_size: int | None = None,
    fertility: float = 1.2,
):
    return SimpleNamespace(
        trained_vocab_size=trained_vocab_size,
        trainer="wordpiece",
        seed=seed,
        vocab_requested=vocab,
        min_frequency=min_frequency,
        train_rows=10,
        validation_rows=2,
        resolved_vocab_size=vocab,
        train_metrics={
            "fertility": fertility,
            "unk_rate": 0.001,
            "fertility_distance": train_distance,
            "single_char_token_pct": 0.0,
            "token_count_mean": 1.0,
            "token_count_median": 1.0,
            "token_count_p95": 1.0,
        },
        validation_metrics={
            "fertility": 1.2,
            "unk_rate": 0.001,
            "fertility_distance": train_distance,
            "single_char_token_pct": 0.0,
            "token_count_mean": 1.0,
            "token_count_median": 1.0,
            "token_count_p95": 1.0,
        },
        fertility_generalization_delta=0.0,
        unk_rate_generalization_delta=0.0,
        model_path="m",
        rejected=False,
        rejection_reason="",
        selection_score=1.0,
        elapsed_seconds=0.1,
    )


def _context(tmp_path: Path) -> tuple[OptimizeRunContext, Path, Path]:
    run_log_dir = tmp_path / "run_logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = tmp_path / "run_log.parquet"
    context = OptimizeRunContext(
        args=argparse.Namespace(
            tokenizer="wordpiece",
            elbow_min_points=1,
            elbow_min_improvement=0.001,
            validation_fraction=0.2,
        ),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        session_id="sess-1",
        target_scope="country",
        systems=["fr"],
        shard_run_log_path=run_log_dir / "sess-1.parquet",
        models_dir=tmp_path / "models",
        reuse_existing_candidates=False,
        excluded_candidate_keys=set(),
        total_candidate_jobs=10,
        max_workers=2,
    )
    return context, run_log_dir, run_log_path


def _tracker() -> OptimizeExecutionTracker:
    return OptimizeExecutionTracker(session_run_log_df=pl.DataFrame())


def _disabled_concurrency_controller() -> AdaptiveConcurrencyController:
    return AdaptiveConcurrencyController(
        hard_cap=2, memory_floor_bytes=1, enabled=False
    )


def _inert_memory_budget_ledger() -> MemoryBudgetLedger:
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


def _seed_bracket_history(
    *, context: OptimizeRunContext, tracker: OptimizeExecutionTracker
) -> None:
    # Fertility 1.28 -> 1.255 -> 1.23 across the bracket against a 1.25
    # target, so the distance is a V with its point at 25928, interpolated in
    # log vocab; the seeds at 25000 give a noise estimate.
    bracket_fertilities = {20000: 1.28, 25000: 1.255, 30000: 1.23}
    seed_jitter = {25000: {1: 1.2502, 2: 1.2598}}
    for vocab, bracket_fertility in bracket_fertilities.items():
        for seed in (1, 2):
            fertility = seed_jitter.get(vocab, {}).get(seed, bracket_fertility)
            result = _history_result(
                seed=seed,
                vocab=vocab,
                min_frequency=1,
                train_distance=abs(fertility - 1.25),
                fertility=fertility,
            )
            optimize_execution.record_completed_candidate_result(
                result=result,
                worker_rss_bytes=None,
                worker_pid=1,
                context=context,
                tracker=tracker,
                include_in_pilot=False,
            )


def _fake_refined_run_candidate(distances: dict[int, float]):
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
        return _history_result(
            seed=seed,
            vocab=vocab_requested,
            min_frequency=min_frequency,
            train_distance=distances[vocab_requested],
        )

    return _run


def test_run_optimize_refining_phase_runs_interior_points_and_reports_fit_match(
    tmp_path: Path, capsys
):
    context, run_log_dir, run_log_path = _context(tmp_path)
    tracker = _tracker()
    _seed_bracket_history(context=context, tracker=tracker)

    # The predicted crossing 25928 and a step=3000 either side: 22928 and
    # 25928 are set on the V's predictions; 28928 is set far off it.
    refined_distances = {22928: 0.01469, 25928: 0.0, 28928: 0.1}

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        report = optimize_execution.run_optimize_refining_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            seeds=[1, 2],
            frontier={(1, 20000), (1, 25000), (1, 30000)},
            seed_splits={1: _split(), 2: _split()},
            run_log_dir=run_log_dir,
            run_log_path=run_log_path,
            eligibility_pass_rate=0.8,
            vocab_size_distance_tolerance=0.01,
            run_candidate=_fake_refined_run_candidate(refined_distances),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
            refine_step_override=3000,
        )

    assert report is not None
    assert report["bracket_low"] == 20000
    assert report["bracket_high"] == 30000
    assert report["winning_vocab"] == 25000
    assert report["min_frequency"] == 1
    assert report["refine_step"] == 3000
    assert round(report["crossing"]) == 25928
    assert sorted(report["refined_points"]) == [22928, 25928, 28928]

    matches_by_vocab = {
        vocab: matched for vocab, _residual, matched in report["fit_matches"]
    }
    assert matches_by_vocab[22928] is True
    assert matches_by_vocab[25928] is True
    assert matches_by_vocab[28928] is False

    # Every refined point ran at both seeds -- full seeds, no pilot pruning.
    refining_rows = [
        job
        for job in tracker.candidate_results
        if job.vocab_requested in (22928, 25928, 28928)
    ]
    assert len(refining_rows) == 6

    out = capsys.readouterr().out
    assert "refining bracket=[20000,30000]" in out
    assert "refining fit_check matched=2/3" in out


def test_run_optimize_refining_phase_centres_off_a_plateau_winner(
    tmp_path: Path, capsys
):
    # min_frequency=12 at vocab=25000 is nearest the target but trained only
    # 21400 tokens: it is on its plateau, where vocab changes nothing. Refining
    # centres on vocab 25000, the best that trained its full vocabulary, at
    # min_frequency=1, the lowest that did, and refines that line.
    context, run_log_dir, run_log_path = _context(tmp_path)
    tracker = _tracker()
    cells = {
        (12, 20000): (0.016, 20000),
        (12, 25000): (0.0084, 21400),
        (1, 20000): (0.016, 20000),
        (1, 25000): (0.009, 25000),
        (1, 30000): (0.025, 30000),
        # The same tokenizer as min_frequency=1 at 25000, scoring a hair
        # better, on a line that plateaus just above it.
        (8, 25000): (0.008999, 25000),
        (8, 30000): (0.0207, 26000),
    }
    for (min_frequency, vocab), (distance, trained_size) in cells.items():
        for seed, jitter in ((1, -0.0002), (2, 0.0002)):
            optimize_execution.record_completed_candidate_result(
                result=_history_result(
                    seed=seed,
                    vocab=vocab,
                    min_frequency=min_frequency,
                    train_distance=distance + jitter,
                    trained_vocab_size=trained_size,
                ),
                worker_rss_bytes=None,
                worker_pid=1,
                context=context,
                tracker=tracker,
                include_in_pilot=False,
            )

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        report = optimize_execution.run_optimize_refining_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            seeds=[1, 2],
            frontier={
                (1, 20000),
                (1, 25000),
                (1, 30000),
                (8, 25000),
                (8, 30000),
                (12, 20000),
                (12, 25000),
            },
            seed_splits={1: _split(), 2: _split()},
            run_log_dir=run_log_dir,
            run_log_path=run_log_path,
            eligibility_pass_rate=0.8,
            vocab_size_distance_tolerance=0.01,
            run_candidate=_fake_refined_run_candidate(
                {22000: 0.002, 28000: 0.022}
            ),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
            refine_step_override=3000,
        )

    assert report is not None
    assert report["min_frequency"] == 1
    assert report["winning_vocab"] == 25000
    # Fertility here never crosses the target, so refining centres on the
    # winning vocab, already measured, and trains a step either side.
    assert sorted(report["refined_points"]) == [22000, 28000]
    refined = [r for r in tracker.candidate_results if r.vocab_requested == 22000]
    assert {r.min_frequency for r in refined} == {1}
    assert "is on its min_frequency plateau; centring on min_freq=1" in (
        capsys.readouterr().out
    )


def test_run_optimize_refining_phase_skips_when_no_interior_bracket(tmp_path: Path):
    context, run_log_dir, run_log_path = _context(tmp_path)
    tracker = _tracker()
    # Only two bracket points on record -- the winner is one of them, so
    # select_refining_bracket has no interior point to fit through.
    for vocab in (20000, 25000):
        for seed in (1, 2):
            optimize_execution.record_completed_candidate_result(
                result=_history_result(
                    seed=seed, vocab=vocab, min_frequency=1, train_distance=0.01
                ),
                worker_rss_bytes=None,
                worker_pid=1,
                context=context,
                tracker=tracker,
                include_in_pilot=False,
            )

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        report = optimize_execution.run_optimize_refining_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            seeds=[1, 2],
            frontier={(1, 20000), (1, 25000)},
            seed_splits={1: _split(), 2: _split()},
            run_log_dir=run_log_dir,
            run_log_path=run_log_path,
            eligibility_pass_rate=0.8,
            vocab_size_distance_tolerance=0.01,
            run_candidate=_fake_refined_run_candidate({}),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert report is None


def test_run_optimize_refining_phase_skips_when_a_bracket_point_has_no_history(
    tmp_path: Path,
):
    context, run_log_dir, run_log_path = _context(tmp_path)
    tracker = _tracker()
    # Only the low/winner cells have run-log rows; the frontier still names
    # 30000 as the bracket's high point (a mismatch a caller could pass, or
    # a run whose expansion never actually reached that frontier member).
    for vocab in (20000, 25000):
        for seed in (1, 2):
            optimize_execution.record_completed_candidate_result(
                result=_history_result(
                    seed=seed,
                    vocab=vocab,
                    min_frequency=1,
                    train_distance=0.03 if vocab == 20000 else 0.005,
                ),
                worker_rss_bytes=None,
                worker_pid=1,
                context=context,
                tracker=tracker,
                include_in_pilot=False,
            )

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        report = optimize_execution.run_optimize_refining_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            seeds=[1, 2],
            frontier={(1, 20000), (1, 25000), (1, 30000)},
            seed_splits={1: _split(), 2: _split()},
            run_log_dir=run_log_dir,
            run_log_path=run_log_path,
            eligibility_pass_rate=0.8,
            vocab_size_distance_tolerance=0.01,
            run_candidate=_fake_refined_run_candidate({}),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert report is None


def test_run_optimize_refining_phase_skips_when_winner_has_a_single_seed(
    tmp_path: Path,
):
    context, run_log_dir, run_log_path = _context(tmp_path)
    tracker = _tracker()
    # Bracket edges get two seeds each; the winning cell (25000) gets only
    # one, so summarize_cell_train_fertility_distance reports no stdev.
    bracket_seed_counts = {20000: (1, 2), 25000: (1,), 30000: (1, 2)}
    bracket_distances = {20000: 0.03, 25000: 0.005, 30000: 0.03}
    for vocab, seeds in bracket_seed_counts.items():
        for seed in seeds:
            optimize_execution.record_completed_candidate_result(
                result=_history_result(
                    seed=seed,
                    vocab=vocab,
                    min_frequency=1,
                    train_distance=bracket_distances[vocab],
                ),
                worker_rss_bytes=None,
                worker_pid=1,
                context=context,
                tracker=tracker,
                include_in_pilot=False,
            )

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        report = optimize_execution.run_optimize_refining_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            seeds=[1, 2],
            frontier={(1, 20000), (1, 25000), (1, 30000)},
            seed_splits={1: _split(), 2: _split()},
            run_log_dir=run_log_dir,
            run_log_path=run_log_path,
            eligibility_pass_rate=0.8,
            vocab_size_distance_tolerance=0.01,
            run_candidate=_fake_refined_run_candidate({}),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert report is None


def test_run_optimize_refining_phase_skips_when_no_eligible_winner(tmp_path: Path):
    context, run_log_dir, run_log_path = _context(tmp_path)
    tracker = _tracker()
    # A rejected-only cell gives scoped_history real columns (so
    # select_best_pair's own grouping runs) with nothing eligible to select
    # -- unlike a genuinely empty run log, which pilot always rules out in
    # production (it evaluates the full grid before refining ever runs).
    rejected_result = _history_result(
        seed=1, vocab=20000, min_frequency=1, train_distance=0.9
    )
    # Fails the unk_rate gate, so reclassify_rejections (select_best_pair's
    # policy-driven recheck) marks this rejected regardless of the flag
    # _history_result set.
    rejected_result.validation_metrics["unk_rate"] = 0.5
    optimize_execution.record_completed_candidate_result(
        result=rejected_result,
        worker_rss_bytes=None,
        worker_pid=1,
        context=context,
        tracker=tracker,
        include_in_pilot=False,
    )

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        report = optimize_execution.run_optimize_refining_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            seeds=[1, 2],
            frontier=set(),
            seed_splits={1: _split(), 2: _split()},
            run_log_dir=run_log_dir,
            run_log_path=run_log_path,
            eligibility_pass_rate=0.8,
            vocab_size_distance_tolerance=0.01,
            run_candidate=_fake_refined_run_candidate({}),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert report is None


def test_collect_candidate_jobs_for_refining_vocab_uses_every_seed():
    context = OptimizeRunContext(
        args=argparse.Namespace(tokenizer="wordpiece"),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        session_id="sess-1",
        target_scope="country",
        systems=["fr"],
        shard_run_log_path=Path("shard.parquet"),
        models_dir=Path("models"),
        reuse_existing_candidates=False,
        excluded_candidate_keys=set(),
        total_candidate_jobs=10,
        max_workers=2,
    )

    jobs, skipped = optimize_execution.collect_candidate_jobs_for_refining_vocab(
        context=context,
        vocab_requested=23000,
        min_frequency=1,
        seeds=[1, 2, 3],
        seed_splits={1: _split(), 2: _split(), 3: _split()},
    )

    assert skipped == 0
    assert [job["seed"] for job in jobs] == [1, 2, 3]
    assert {job["vocab_requested"] for job in jobs} == {23000}
    assert {job["min_frequency"] for job in jobs} == {1}
