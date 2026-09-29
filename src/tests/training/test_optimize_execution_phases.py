from __future__ import annotations

import argparse
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import polars as pl
from company_tokenize import OptimizeCandidateResult, TrainerOptions

from training import optimize_execution
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
    seed: int,
    vocab: int,
    min_frequency: int,
    distance: float,
    rejected: bool,
    train_fertility: float | None = None,
) -> SimpleNamespace:
    # Train-split fertility defaults to a line still above the 1.25 target and
    # `distance` away from it, so the pilot's enforced stop stays out of tests
    # that are not about it.
    if train_fertility is None:
        train_fertility = 1.25 + distance
    return SimpleNamespace(
        trainer="wordpiece",
        seed=seed,
        vocab_requested=vocab,
        min_frequency=min_frequency,
        train_rows=10,
        validation_rows=2,
        resolved_vocab_size=40000,
        train_metrics={
            "fertility": train_fertility,
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


def _fake_run_candidate(
    distances: dict[tuple[int, int, int], float] | None = None,
    train_fertilities: dict[tuple[int, int, int], float] | None = None,
):
    distances = distances or {}
    train_fertilities = train_fertilities or {}

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
            train_fertility=train_fertilities.get(key),
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


def test_run_optimize_expansion_phase_calls_record_result_once_per_completed_result(
    tmp_path: Path,
):
    context = _context(tmp_path)
    tracker = _tracker()
    combo_progress: _ComboProgress = {
        (1, 100): {"runs": 0, "passes": 0, "fertilities": []}
    }
    recorded: list[tuple[tuple[int, int], int]] = []

    def spy_record_result(*, combo, result, expansion_round):
        recorded.append((combo, expansion_round))

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_expansion_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            expansion_seeds=[1, 2],
            frontier={(1, 100)},
            seed_splits={1: _split(), 2: _split()},
            pruned_combos=set(),
            combo_progress=combo_progress,
            total_seed_budget=2,
            combo_prune_reason=lambda **kwargs: None,
            run_candidate=_fake_run_candidate(),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
            record_result=spy_record_result,
        )

    assert recorded == [((1, 100), 1), ((1, 100), 2)]


def test_handle_pilot_vocab_no_candidates_reports_reuse(capsys):
    optimize_execution._handle_pilot_vocab_no_candidates(
        skipped_existing_candidates=0,
        vocab_label="100",
    )
    assert capsys.readouterr().out == ""

    optimize_execution._handle_pilot_vocab_no_candidates(
        skipped_existing_candidates=3,
        vocab_label="100",
    )
    assert "reuse-only vocab=100" in capsys.readouterr().out


def test_log_pilot_vocab_non_monotonic_results_prints_mixed_outcomes(capsys):
    results = [
        _result(seed=1, vocab=100, min_frequency=1, distance=0.1, rejected=False),
        _result(seed=1, vocab=100, min_frequency=2, distance=0.2, rejected=True),
    ]

    optimize_execution._log_pilot_vocab_non_monotonic_results(
        pilot_seeds=[1],
        current_vocab_results=results,
        vocab_requested=100,
        vocab_label="100",
    )

    assert "non_monotonic seed=1" in capsys.readouterr().out


def test_log_non_monotonic_pilot_results_silent_when_seed_uniformly_passes(capsys):
    results = [
        _result(seed=1, vocab=100, min_frequency=1, distance=0.1, rejected=False),
        _result(seed=1, vocab=100, min_frequency=2, distance=0.2, rejected=False),
    ]

    optimize_execution._log_non_monotonic_pilot_results(results, 1, "100")

    assert capsys.readouterr().out == ""


def test_log_non_monotonic_pilot_results_silent_when_seed_uniformly_rejects(capsys):
    results = [
        _result(seed=1, vocab=100, min_frequency=1, distance=0.9, rejected=True),
        _result(seed=1, vocab=100, min_frequency=2, distance=0.95, rejected=True),
    ]

    optimize_execution._log_non_monotonic_pilot_results(results, 1, "100")

    assert capsys.readouterr().out == ""


def test_run_optimize_pilot_phase_threads_concurrency_params_when_enabled(
    tmp_path: Path, mocker
):
    context = _context(tmp_path)
    tracker = _tracker()
    controller = AdaptiveConcurrencyController(hard_cap=2, memory_floor_bytes=1)
    ledger = _inert_memory_budget_ledger()
    batch_spy = mocker.patch.object(
        optimize_execution, "run_candidate_batch", return_value=[]
    )

    optimize_execution.run_optimize_pilot_phase(
        executor=None,
        context=context,
        tracker=tracker,
        vocab_schedule=[100],
        pilot_seeds=[1],
        min_frequencies=[1],
        seed_splits={1: _split()},
        effective_elbow_extra_steps=0,
        run_candidate=_fake_run_candidate(),
        concurrency_controller=controller,
        memory_budget_ledger=ledger,
        wave_chunk_size=3,
        max_parallel_waves=5,
    )

    assert batch_spy.call_count == 1
    call_kwargs = batch_spy.call_args.kwargs
    assert call_kwargs["concurrency_controller"] is controller
    assert call_kwargs["memory_budget_ledger"] is ledger
    assert call_kwargs["wave_chunk_size"] == 3
    assert call_kwargs["max_parallel_waves"] == 5


def test_run_optimize_expansion_phase_threads_concurrency_params_when_enabled(
    tmp_path: Path, mocker
):
    context = _context(tmp_path)
    tracker = _tracker()
    controller = AdaptiveConcurrencyController(hard_cap=2, memory_floor_bytes=1)
    ledger = _inert_memory_budget_ledger()
    combo_progress: _ComboProgress = {
        (1, 100): {"runs": 0, "passes": 0, "fertilities": []}
    }
    batch_spy = mocker.patch.object(
        optimize_execution, "run_candidate_batch", return_value=[]
    )

    optimize_execution.run_optimize_expansion_phase(
        executor=None,
        context=context,
        tracker=tracker,
        expansion_seeds=[1],
        frontier={(1, 100)},
        seed_splits={1: _split()},
        pruned_combos=set(),
        combo_progress=combo_progress,
        total_seed_budget=1,
        combo_prune_reason=lambda **kwargs: None,
        run_candidate=_fake_run_candidate(),
        concurrency_controller=controller,
        memory_budget_ledger=ledger,
        wave_chunk_size=3,
        max_parallel_waves=5,
    )

    assert batch_spy.call_count == 1
    call_kwargs = batch_spy.call_args.kwargs
    assert call_kwargs["concurrency_controller"] is controller
    assert call_kwargs["memory_budget_ledger"] is ledger
    assert call_kwargs["wave_chunk_size"] == 3
    assert call_kwargs["max_parallel_waves"] == 5


def test_run_optimize_pilot_phase_skips_reused_vocab_then_runs_next(tmp_path: Path):
    context = _context(
        tmp_path,
        reuse_existing_candidates=True,
        excluded_candidate_keys={(1, 100, 1)},
    )
    tracker = _tracker()

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_pilot_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            vocab_schedule=[100, 200],
            pilot_seeds=[1],
            min_frequencies=[1],
            seed_splits={1: _split()},
            effective_elbow_extra_steps=5,
            run_candidate=_fake_run_candidate(),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert tracker.skipped_existing_candidates_total == 1
    assert tracker.completed_jobs == 1
    assert tracker.submitted_jobs == 1
    assert tracker.pilot_results[0].vocab_requested == 200


def test_run_optimize_pilot_phase_evaluates_full_grid_despite_elbow_signal(
    tmp_path: Path,
    capsys,
):
    # Even with elbow-signal thresholds set to fire immediately (min_points=1,
    # min_improvement=1.0 is never met by these distances), every vocab point
    # in the schedule is still evaluated: the signal is diagnostic-only and no
    # longer excludes a combo from later, larger vocab points (2026-08-25).
    context = _context(tmp_path, elbow_min_points=1, elbow_min_improvement=1.0)
    tracker = _tracker()
    distances = {(1, 100, 1): 0.5, (1, 200, 1): 0.4, (1, 300, 1): 0.1}

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_pilot_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            vocab_schedule=[100, 200, 300],
            pilot_seeds=[1],
            min_frequencies=[1],
            seed_splits={1: _split()},
            effective_elbow_extra_steps=0,
            run_candidate=_fake_run_candidate(distances),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert tracker.submitted_jobs == 3
    assert {result.vocab_requested for result in tracker.pilot_results} == {
        100,
        200,
        300,
    }
    assert "elbow_signal (diagnostic, not enforced)" in capsys.readouterr().out


def test_run_optimize_pilot_phase_stops_a_line_that_has_passed_the_target(
    tmp_path: Path,
    capsys,
):
    # min_frequency=1 passes below the 1.25 target at vocab=200, runs its one
    # extra point at 300, and skips 400. min_frequency=2 stays above target and
    # keeps moving, so it runs the whole schedule. The auto point always runs.
    context = _context(tmp_path)
    tracker = _tracker()
    train_fertilities = {
        (1, 100, 1): 1.30,
        (1, 200, 1): 1.24,
        (1, 300, 1): 1.22,
        (1, -1, 1): 1.10,
        (1, 100, 2): 1.40,
        (1, 200, 2): 1.35,
        (1, 300, 2): 1.30,
        (1, 400, 2): 1.27,
        (1, -1, 2): 1.26,
    }

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_pilot_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            vocab_schedule=[100, 200, 300, 400, -1],
            pilot_seeds=[1],
            min_frequencies=[1, 2],
            seed_splits={1: _split()},
            effective_elbow_extra_steps=0,
            run_candidate=_fake_run_candidate(train_fertilities=train_fertilities),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    evaluated = {
        (result.min_frequency, result.vocab_requested)
        for result in tracker.pilot_results
    }
    assert evaluated == set(
        (min_frequency, vocab)
        for (_seed, vocab, min_frequency) in train_fertilities
    )
    assert "reason=passed_target" in capsys.readouterr().out


def _sized_run_candidate(plateau_by_min_frequency: dict[int, int], trained: list):
    """A trainer whose vocabulary stops growing at its min_frequency's plateau,
    returning real results so they can be carried."""

    def _run(*, seed, vocab_requested, min_frequency, **_kwargs):
        trained_size = min(vocab_requested, plateau_by_min_frequency[min_frequency])
        trained.append((vocab_requested, min_frequency))
        metrics = {
            "fertility": 2.0 - trained_size / 1000,
            "unk_rate": 0.001,
            "fertility_distance": 0.75 - trained_size / 1000,
            "single_char_token_pct": 0.0,
            "token_count_mean": 1.0,
            "token_count_median": 1.0,
            "token_count_p95": 1.0,
        }
        return OptimizeCandidateResult(
            trainer="wordpiece",
            seed=seed,
            vocab_requested=vocab_requested,
            min_frequency=min_frequency,
            train_rows=10,
            validation_rows=2,
            resolved_vocab_size=vocab_requested,
            train_metrics=dict(metrics),
            validation_metrics=dict(metrics),
            fertility_generalization_delta=0.0,
            unk_rate_generalization_delta=0.0,
            model_path="m",
            rejected=False,
            rejection_reason="",
            selection_score=1.0,
            elapsed_seconds=0.1,
            trained_vocab_size=trained_size,
        )

    return _run


def test_run_optimize_pilot_phase_trains_only_cells_no_other_cell_fixes(
    tmp_path: Path,
):
    # min_frequency 3 plateaus at 150 tokens and 2 at 250; 1 never does.
    #   vocab=100: the probe (mf=3) reaches 100, so mf=1 and mf=2 are carried.
    #   vocab=200: the probe falls short at 150, so mf=1 and mf=2 are trained.
    #   vocab=300: mf=3 is carried from its plateau; the new probe (mf=2) falls
    #              short at 250, so mf=1 is trained.
    context = _context(tmp_path)
    tracker = _tracker()
    trained: list[tuple[int, int]] = []

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_pilot_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            vocab_schedule=[100, 200, 300],
            pilot_seeds=[1],
            min_frequencies=[1, 2, 3],
            seed_splits={1: _split()},
            effective_elbow_extra_steps=0,
            run_candidate=_sized_run_candidate({1: 10**9, 2: 250, 3: 150}, trained),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert sorted(trained) == [(100, 3), (200, 1), (200, 2), (200, 3), (300, 1), (300, 2)]
    carried = {
        (result.vocab_requested, result.min_frequency): result
        for result in tracker.pilot_results
        if result.carried_from is not None
    }
    assert {cell: result.carried_from for cell, result in carried.items()} == {
        (100, 1): "v100_mf3",
        (100, 2): "v100_mf3",
        (300, 3): "v200_mf3",
    }
    assert len(tracker.pilot_results) == 9
    assert carried[(300, 3)].trained_vocab_size == 150
    assert carried[(300, 3)].resolved_vocab_size == 300
    assert tracker.submitted_jobs == tracker.completed_jobs == 6
    logged = tracker.session_run_log_df
    assert logged.height == 9
    assert logged.filter(pl.col("carried_from").is_not_null()).height == 3


def test_run_optimize_expansion_phase_carries_within_the_pilots_classes(
    tmp_path: Path,
):
    # The pilot above leaves two classes: every min_frequency at vocab=100, and
    # min_frequency=3 from vocab=200 up. On the expansion seed each trains one
    # representative, which shows the same outcome on its own split, and the
    # rest of the frontier is carried. min_frequency=3 plateaus at 250 on the
    # second expansion seed's split, so vocab=200 reaches its full size there
    # and vocab=300 is trained rather than carried.
    context = _context(tmp_path)
    tracker = _tracker()
    trained: list[tuple[int, int]] = []
    frontier = {(1, 100), (2, 100), (3, 100), (3, 200), (3, 300)}
    combo_progress: _ComboProgress = {
        combo: {"runs": 0, "passes": 0, "fertilities": []} for combo in frontier
    }
    plateaus = {1: 10**9, 2: 250, 3: 150}
    run_sized = _sized_run_candidate(plateaus, trained)
    run_moved = _sized_run_candidate({**plateaus, 3: 250}, trained)

    def run_candidate(*, seed, **kwargs):
        return (run_moved if seed == 3 else run_sized)(seed=seed, **kwargs)

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        common = {
            "executor": executor,
            "context": context,
            "tracker": tracker,
            "seed_splits": {1: _split(), 2: _split(), 3: _split()},
            "run_candidate": run_candidate,
            "concurrency_controller": _disabled_concurrency_controller(),
            "memory_budget_ledger": _inert_memory_budget_ledger(),
            "wave_chunk_size": 4,
            "max_parallel_waves": 2,
        }
        optimize_execution.run_optimize_pilot_phase(
            vocab_schedule=[100, 200, 300],
            pilot_seeds=[1],
            min_frequencies=[1, 2, 3],
            effective_elbow_extra_steps=0,
            **common,
        )
        trained.clear()
        optimize_execution.run_optimize_expansion_phase(
            expansion_seeds=[2, 3],
            frontier=frontier,
            pruned_combos=set(),
            combo_progress=combo_progress,
            total_seed_budget=3,
            combo_prune_reason=lambda **kwargs: None,
            **common,
        )

    assert sorted(trained) == [(100, 3), (100, 3), (200, 3), (200, 3), (300, 3)]
    assert all(state["runs"] == 2 for state in combo_progress.values())
    seed_three = {
        (result.vocab_requested, result.min_frequency): result.carried_from
        for result in tracker.candidate_results
        if result.seed == 3
    }
    assert seed_three == {
        (100, 1): "v100_mf3",
        (100, 2): "v100_mf3",
        (100, 3): None,
        (200, 3): None,
        (300, 3): None,
    }


def test_build_pilot_cell_classes_groups_cells_the_pilot_trained_anyway():
    # vocab=200: the probe (mf=3) fell short, so mf=1 and mf=2 were trained in
    # the second batch and both reached 200: one class, named for mf=2. The
    # auto point on mf=3 fell short too, so it joins mf=3's plateau class.
    run = _sized_run_candidate({1: 10**9, 2: 250, 3: 150}, [])
    pilot_results = [
        run(seed=1, vocab_requested=200, min_frequency=min_frequency)
        for min_frequency in (1, 2, 3)
    ]
    auto = run(seed=1, vocab_requested=900, min_frequency=3)
    pilot_results.append(
        optimize_execution.dataclasses.replace(auto, vocab_requested=-1)
    )

    classes = optimize_execution.build_pilot_cell_classes(pilot_results)

    assert classes == {
        (1, 200): "v200_mf2",
        (2, 200): "v200_mf2",
        (3, 200): "v200_mf3",
        (3, -1): "v200_mf3",
    }


def test_backfill_trained_vocab_sizes_reads_only_the_rows_that_lack_one(mocker):
    read = mocker.patch.object(
        optimize_execution,
        "read_trained_vocab_size",
        side_effect=lambda *, trainer, tokenizer_path: {
            "kept.json": 21400
        }.get(tokenizer_path.name),
    )
    run_log_df = pl.DataFrame(
        {
            "model_path": ["logged.json", "kept.json", "gone.json", ""],
            "vocab_size_trained": [25000, None, None, None],
        }
    )

    filled = optimize_execution.backfill_trained_vocab_sizes(
        run_log_df=run_log_df, trainer="wordpiece"
    )

    assert filled.get_column("vocab_size_trained").to_list() == [
        25000,
        21400,
        None,
        None,
    ]
    assert {call.kwargs["tokenizer_path"].name for call in read.call_args_list} == {
        "kept.json",
        "gone.json",
    }
    # A log written before the column existed is filled the same way.
    assert optimize_execution.backfill_trained_vocab_sizes(
        run_log_df=run_log_df.drop("vocab_size_trained"), trainer="wordpiece"
    ).get_column("vocab_size_trained").to_list() == [None, 21400, None, None]


def test_winning_min_frequency_plateau_size_reads_the_shortfall_rows():
    run_log_df = pl.DataFrame(
        {
            "min_frequency": [12, 12, 12, 1],
            "vocab_size_resolved": [20000, 25000, 40000, 40000],
            "vocab_size_trained": [20000, 21475, 21474, 40000],
        }
    )

    assert (
        optimize_execution.winning_min_frequency_plateau_size(
            run_log_df=run_log_df, min_frequency=12
        )
        == 21474
    )
    assert (
        optimize_execution.winning_min_frequency_plateau_size(
            run_log_df=run_log_df, min_frequency=1
        )
        is None
    )
    assert (
        optimize_execution.winning_min_frequency_plateau_size(
            run_log_df=run_log_df.drop("vocab_size_trained"), min_frequency=12
        )
        is None
    )


def test_update_pilot_combo_cutoffs_stops_a_line_on_its_plateau():
    pilot_combo_state = optimize_execution._initialize_pilot_combo_state([1], [12])
    common = {
        "pilot_combo_state": pilot_combo_state,
        "elbow_min_points": 99,
        "elbow_min_improvement": 0.0,
        "effective_elbow_extra_steps": 1,
        "fertility_target": 1.25,
    }

    # Still above target throughout: 1.30, then flat at 1.2578.
    for vocab, fertility, stopped in (
        (100, 1.30, False),
        (200, 1.2578, False),
        (300, 1.25781, False),
        (400, 1.2578, True),
    ):
        optimize_execution.update_pilot_combo_cutoffs(
            current_vocab_results=[
                _result(
                    seed=1,
                    vocab=vocab,
                    min_frequency=12,
                    distance=0.1,
                    rejected=False,
                    train_fertility=fertility,
                )
            ],
            vocab_label=str(vocab),
            **common,
        )
        assert pilot_combo_state[(1, 12)]["stopped"] is stopped


def test_update_pilot_combo_cutoffs_transitions_state_across_successive_vocab_points(
    capsys,
):
    # Simulates one combo's (seed, min_frequency) trajectory across a
    # multi-point vocab schedule, exercising the cutoff state machine's
    # transitions directly rather than only through run_optimize_pilot_phase:
    #   1) too few points yet -> no improvement check, no cutoff
    #   2) a real improvement -> post_elbow_steps resets to 0, still no cutoff
    #   3) a below-threshold improvement -> post_elbow_steps increments and
    #      crosses effective_elbow_extra_steps -> cutoff flips True, and the
    #      diagnostic line prints exactly once
    #   4) another below-threshold improvement with cutoff already True -> the
    #      diagnostic line does NOT print again (not re-announced every step)
    pilot_combo_state = optimize_execution._initialize_pilot_combo_state([1], [1])
    combo_key = (1, 1)
    common = {
        "elbow_min_points": 2,
        "elbow_min_improvement": 0.01,
        "effective_elbow_extra_steps": 1,
    }

    optimize_execution.update_pilot_combo_cutoffs(
        current_vocab_results=[
            _result(seed=1, vocab=100, min_frequency=1, distance=0.50, rejected=False)
        ],
        pilot_combo_state=pilot_combo_state,
        vocab_label="100",
        **common,
    )
    assert pilot_combo_state[combo_key]["points"] == 1
    assert pilot_combo_state[combo_key]["cutoff"] is False
    assert capsys.readouterr().out == ""

    optimize_execution.update_pilot_combo_cutoffs(
        current_vocab_results=[
            _result(seed=1, vocab=200, min_frequency=1, distance=0.45, rejected=False)
        ],
        pilot_combo_state=pilot_combo_state,
        vocab_label="200",
        **common,
    )
    assert pilot_combo_state[combo_key]["post_elbow_steps"] == 0
    assert pilot_combo_state[combo_key]["cutoff"] is False
    assert capsys.readouterr().out == ""

    optimize_execution.update_pilot_combo_cutoffs(
        current_vocab_results=[
            _result(seed=1, vocab=300, min_frequency=1, distance=0.449, rejected=False)
        ],
        pilot_combo_state=pilot_combo_state,
        vocab_label="300",
        **common,
    )
    assert pilot_combo_state[combo_key]["cutoff"] is True
    first_transition_out = capsys.readouterr().out
    assert "elbow_signal (diagnostic, not enforced)" in first_transition_out
    assert "vocab=300" in first_transition_out

    optimize_execution.update_pilot_combo_cutoffs(
        current_vocab_results=[
            _result(seed=1, vocab=400, min_frequency=1, distance=0.448, rejected=False)
        ],
        pilot_combo_state=pilot_combo_state,
        vocab_label="400",
        **common,
    )
    assert pilot_combo_state[combo_key]["cutoff"] is True
    assert capsys.readouterr().out == ""


def test_initialize_frontier_progress_ignores_results_outside_frontier():
    pilot_results = [
        _result(seed=1, vocab=100, min_frequency=1, distance=0.1, rejected=False),
        _result(seed=2, vocab=999, min_frequency=9, distance=0.1, rejected=False),
    ]

    combo_progress = optimize_execution.initialize_frontier_progress(
        frontier={(1, 100)},
        pilot_results=pilot_results,
    )

    assert set(combo_progress.keys()) == {(1, 100)}
    assert combo_progress[(1, 100)]["runs"] == 1
    assert combo_progress[(1, 100)]["passes"] == 1


def test_initialize_frontier_progress_seeds_from_historical_expansion_results():
    # Regression guard: a partially-reused expansion phase must not undercount
    # a frontier combo's runs/passes just because some of its expansion seeds
    # were skipped via reuse rather than freshly trained this invocation.
    pilot_results = [
        _result(seed=1, vocab=100, min_frequency=1, distance=0.1, rejected=False),
    ]
    historical_expansion_results = [
        _result(seed=2, vocab=100, min_frequency=1, distance=0.1, rejected=False),
        _result(seed=3, vocab=100, min_frequency=1, distance=0.1, rejected=True),
        _result(seed=2, vocab=999, min_frequency=9, distance=0.1, rejected=False),
    ]

    combo_progress = optimize_execution.initialize_frontier_progress(
        frontier={(1, 100)},
        pilot_results=pilot_results,
        historical_expansion_results=historical_expansion_results,
    )

    assert set(combo_progress.keys()) == {(1, 100)}
    assert combo_progress[(1, 100)]["runs"] == 3  # 1 pilot + 2 historical
    assert combo_progress[(1, 100)]["passes"] == 2  # seed=3 was rejected
    assert len(combo_progress[(1, 100)]["fertilities"]) == 3


def test_prune_combos_collects_combos_flagged_by_policy():
    combo_progress: _ComboProgress = {
        (1, 100): {"runs": 3, "passes": 3, "fertilities": [1.0, 1.1, 1.2]},
        (2, 200): {"runs": 1, "passes": 0, "fertilities": [0.5]},
    }

    pruned = optimize_execution.prune_combos(
        combo_progress=combo_progress,
        total_seed_budget=3,
        policy=_policy(),
        combo_prune_reason=lambda **kwargs: "bad" if kwargs["passes"] == 0 else None,
    )

    assert pruned == {(2, 200)}


def test_run_optimize_expansion_phase_breaks_when_all_combos_pruned(
    tmp_path: Path, capsys
):
    context = _context(tmp_path)
    tracker = _tracker()

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_expansion_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            expansion_seeds=[1, 2],
            frontier={(1, 100)},
            seed_splits={1: _split(), 2: _split()},
            pruned_combos={(1, 100)},
            combo_progress={},
            total_seed_budget=2,
            combo_prune_reason=lambda **kwargs: None,
            run_candidate=_fake_run_candidate(),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert (
        "expansion ended early; all frontier combos pruned" in capsys.readouterr().out
    )
    assert tracker.completed_jobs == 0


def test_run_optimize_expansion_phase_skips_seed_when_all_candidates_excluded(
    tmp_path: Path,
):
    context = _context(
        tmp_path,
        reuse_existing_candidates=True,
        excluded_candidate_keys={(1, 100, 1)},
    )
    tracker = _tracker()
    combo_progress: _ComboProgress = {
        (1, 100): {"runs": 0, "passes": 0, "fertilities": []}
    }

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_expansion_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            expansion_seeds=[1],
            frontier={(1, 100)},
            seed_splits={1: _split()},
            pruned_combos=set(),
            combo_progress=combo_progress,
            total_seed_budget=1,
            combo_prune_reason=lambda **kwargs: None,
            run_candidate=_fake_run_candidate(),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    assert tracker.skipped_existing_candidates_total == 1
    assert tracker.completed_jobs == 0


def test_run_optimize_expansion_phase_runs_seeds_and_prunes_combo(
    tmp_path: Path, capsys
):
    context = _context(tmp_path)
    tracker = _tracker()
    combo_progress: _ComboProgress = {
        (1, 100): {"runs": 0, "passes": 0, "fertilities": []}
    }

    def prune_after_first_run(*, runs, passes, fertilities, total_seeds, policy):
        return "unrecoverable" if runs >= 1 else None

    with futures.ThreadPoolExecutor(max_workers=2) as executor:
        optimize_execution.run_optimize_expansion_phase(
            executor=executor,
            context=context,
            tracker=tracker,
            expansion_seeds=[1, 2],
            frontier={(1, 100)},
            seed_splits={1: _split(), 2: _split()},
            pruned_combos=set(),
            combo_progress=combo_progress,
            total_seed_budget=2,
            combo_prune_reason=prune_after_first_run,
            run_candidate=_fake_run_candidate(),
            concurrency_controller=_disabled_concurrency_controller(),
            memory_budget_ledger=_inert_memory_budget_ledger(),
            wave_chunk_size=4,
            max_parallel_waves=2,
        )

    out = capsys.readouterr().out
    assert "prune min_freq=1 vocab=100 reason=unrecoverable" in out
    assert "expansion ended early; all frontier combos pruned" in out
    assert tracker.completed_jobs == 1
