from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import polars as pl
from company_tokenize import TrainerOptions

from training import optimize_execution, optimize_expansion_strategy, optimize_retry
from training.optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
)
from training.runtime import OptimizeExecutionTracker, OptimizeRunContext
from workspace.roots import WorkspaceRoots

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


def _run_target_args(**overrides) -> argparse.Namespace:
    base = {
        "tokenizer": "wordpiece",
        "profile": "default",
        "corpus_filename": "wordpiece_training_corpus.txt",
        "pilot_seed_count": 1,
        "force": False,
        "name_col": "name",
        "dry_run": False,
        "validation_fraction": 0.2,
        "max_workers": 1,
        "vocab_frontier_top_k": 1,
        "vocab_frontier_distance_tolerance": 0.0,
        "finalize": True,
        "eligibility_pass_rate": 0.8,
        "vocab_size_distance_tolerance": 0,
        "expansion_strategy": "heuristic",
        "worker_memory_floor_gb": 2.0,
        "memory_headroom_fraction": 0.20,
        "wave_chunk_size": 4,
        "max_parallel_waves": None,
        "disable_adaptive_concurrency": True,
        "candidate_timeout_seconds": optimize_execution.DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _StubExecutor:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def shutdown(self, *, wait=True, cancel_futures=False):
        return None


def _execution_setup(
    tmp_path: Path,
    *,
    reuse: bool = False,
    historical_results: list | None = None,
    bootstrap_rss_bytes: int | None = None,
) -> dict:
    return {
        "final_tokenizer_path": tmp_path / "final.json",
        "optimize_dir": tmp_path / "optimize",
        "models_dir": tmp_path / "optimize" / "models",
        "run_log_dir": tmp_path / "optimize" / "run_logs",
        "run_log_path": tmp_path / "optimize" / "run_log.parquet",
        "summary_path": tmp_path / "optimize" / "summary.json",
        "reuse_existing_candidates": reuse,
        "excluded_candidate_keys": set(),
        "historical_results": historical_results or [],
        "bootstrap_rss_bytes": bootstrap_rss_bytes,
        "shard_run_log_path": tmp_path / "optimize" / "run_logs" / "sess-1.parquet",
        "session_run_log_df": pl.DataFrame(),
    }


def test_run_optimize_target_dry_run_emits_plan_without_full_execution(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet",
        directory=tmp_path,
    )
    target = SimpleNamespace(scope="country", systems=["fr"])
    args = argparse.Namespace(
        tokenizer="wordpiece",
        profile="default",
        corpus_filename="wordpiece_training_corpus.txt",
        pilot_seed_count=1,
        force=True,
        optimize_wipe=False,
        name_col="name",
        dry_run=True,
        validation_fraction=0.2,
        max_workers=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        finalize=False,
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0,
    )

    resolve_paths = mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    emit_plan = mocker.patch.object(optimize_execution, "emit_optimize_dry_run_plan")
    mocker.patch.object(
        optimize_execution,
        "resolve_tokenizer_path_for_trainer",
        return_value=tmp_path / "final.json",
    )

    optimize_execution.run_optimize_target(
        args=args,
        roots=workspace_roots,
        target=target,
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1, 2],
        vocab_schedule=[10000, -1],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=_write_stub_corpus_and_count,
    )

    resolve_paths.assert_called_once()
    emit_plan.assert_called_once()


def test_run_optimize_target_returns_when_corpus_resolution_is_none(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet", directory=tmp_path
    )
    mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    mocker.patch.object(
        optimize_execution, "list_cleansed_parquet_files", return_value=[]
    )
    mocker.patch.object(
        optimize_execution, "resolve_optimize_corpus_for_target", return_value=None
    )
    prepare_execution = mocker.patch.object(
        optimize_execution, "prepare_optimize_target_execution"
    )

    optimize_execution.run_optimize_target(
        args=_run_target_args(),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["fr"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1, 2],
        vocab_schedule=[100],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=lambda **kwargs: None,
    )

    prepare_execution.assert_not_called()


def test_run_optimize_target_returns_when_execution_setup_is_none(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet", directory=tmp_path
    )
    mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    mocker.patch.object(
        optimize_execution, "list_cleansed_parquet_files", return_value=[]
    )
    mocker.patch.object(
        optimize_execution,
        "resolve_optimize_corpus_for_target",
        return_value=(10, "abc123hash"),
    )
    mocker.patch.object(
        optimize_execution, "prepare_optimize_target_execution", return_value=None
    )
    seed_splits_spy = mocker.patch.object(
        optimize_execution, "prepare_optimize_seed_splits"
    )

    optimize_execution.run_optimize_target(
        args=_run_target_args(),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["fr"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1],
        vocab_schedule=[100],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=lambda **kwargs: 10,
    )

    seed_splits_spy.assert_not_called()


def test_run_optimize_target_full_execution_skips_finalize_when_not_eligible(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet", directory=tmp_path
    )
    execution_setup = _execution_setup(tmp_path)

    mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    mocker.patch.object(
        optimize_execution, "list_cleansed_parquet_files", return_value=[]
    )
    mocker.patch.object(
        optimize_execution,
        "resolve_optimize_corpus_for_target",
        return_value=(10, "abc123hash"),
    )
    mocker.patch.object(
        optimize_execution,
        "prepare_optimize_target_execution",
        return_value=execution_setup,
    )
    mocker.patch.object(
        optimize_retry.futures, "ProcessPoolExecutor", return_value=_StubExecutor()
    )
    pilot_phase = mocker.patch.object(optimize_execution, "run_optimize_pilot_phase")
    expansion_phase = mocker.patch.object(
        optimize_execution, "run_optimize_expansion_phase"
    )
    mocker.patch.object(
        optimize_execution, "merge_run_logs", return_value=pl.DataFrame()
    )
    mocker.patch.object(
        optimize_execution, "scoped_history_for_trainer", return_value=pl.DataFrame()
    )
    mocker.patch.object(optimize_execution, "select_best_pair", return_value=None)
    finalize_impl = mocker.Mock()

    optimize_execution.run_optimize_target(
        args=_run_target_args(finalize=True),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["fr"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1],
        vocab_schedule=[100],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=lambda **kwargs: 10,
        prepare_optimize_seed_splits_func=lambda **kwargs: ({}, 0, 0),
        combo_prune_reason_func=lambda **kwargs: None,
        finalize_optimize_target_func=finalize_impl,
        resolve_optimize_max_workers_func=lambda *args: 1,
    )

    pilot_phase.assert_called_once()
    expansion_phase.assert_called_once()
    finalize_impl.assert_not_called()


def test_run_optimize_target_full_execution_shard_complete_when_finalize_disabled(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker, capsys
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet", directory=tmp_path
    )
    execution_setup = _execution_setup(tmp_path)
    best_pair = {"vocab_size_requested": 100, "min_frequency": 1}

    mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    mocker.patch.object(
        optimize_execution, "list_cleansed_parquet_files", return_value=[]
    )
    mocker.patch.object(
        optimize_execution,
        "resolve_optimize_corpus_for_target",
        return_value=(10, "abc123hash"),
    )
    mocker.patch.object(
        optimize_execution,
        "prepare_optimize_target_execution",
        return_value=execution_setup,
    )
    mocker.patch.object(
        optimize_retry.futures, "ProcessPoolExecutor", return_value=_StubExecutor()
    )
    mocker.patch.object(optimize_execution, "run_optimize_pilot_phase")
    mocker.patch.object(optimize_execution, "run_optimize_expansion_phase")
    mocker.patch.object(
        optimize_execution, "merge_run_logs", return_value=pl.DataFrame()
    )
    mocker.patch.object(
        optimize_execution, "scoped_history_for_trainer", return_value=pl.DataFrame()
    )
    mocker.patch.object(optimize_execution, "select_best_pair", return_value=best_pair)
    finalize_impl = mocker.Mock()

    optimize_execution.run_optimize_target(
        args=_run_target_args(finalize=False),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["fr"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1],
        vocab_schedule=[100, -1],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=lambda **kwargs: 10,
        prepare_optimize_seed_splits_func=lambda **kwargs: ({}, 0, 0),
        select_vocab_frontier_func=lambda **kwargs: {(1, 100)},
        combo_prune_reason_func=lambda **kwargs: None,
        finalize_optimize_target_func=finalize_impl,
        resolve_optimize_max_workers_func=lambda *args: 1,
    )

    finalize_impl.assert_not_called()
    assert "shard complete" in capsys.readouterr().out


def test_run_optimize_target_full_execution_finalizes_when_eligible(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet", directory=tmp_path
    )
    execution_setup = _execution_setup(tmp_path, reuse=True)
    best_pair = {"vocab_size_requested": 100, "min_frequency": 1}

    mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    mocker.patch.object(
        optimize_execution, "list_cleansed_parquet_files", return_value=[]
    )
    mocker.patch.object(
        optimize_execution,
        "resolve_optimize_corpus_for_target",
        return_value=(10, "abc123hash"),
    )
    mocker.patch.object(
        optimize_execution,
        "prepare_optimize_target_execution",
        return_value=execution_setup,
    )
    mocker.patch.object(
        optimize_retry.futures, "ProcessPoolExecutor", return_value=_StubExecutor()
    )
    mocker.patch.object(optimize_execution, "run_optimize_pilot_phase")
    mocker.patch.object(optimize_execution, "run_optimize_expansion_phase")
    mocker.patch.object(
        optimize_execution, "merge_run_logs", return_value=pl.DataFrame()
    )
    mocker.patch.object(
        optimize_execution, "scoped_history_for_trainer", return_value=pl.DataFrame()
    )
    mocker.patch.object(optimize_execution, "select_best_pair", return_value=best_pair)
    finalize_impl = mocker.Mock()

    optimize_execution.run_optimize_target(
        args=_run_target_args(finalize=True),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["fr"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1],
        vocab_schedule=[100, -1],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=lambda **kwargs: 10,
        prepare_optimize_seed_splits_func=lambda **kwargs: ({}, 0, 0),
        select_vocab_frontier_func=lambda **kwargs: {(1, 100)},
        combo_prune_reason_func=lambda **kwargs: None,
        finalize_optimize_target_func=finalize_impl,
        resolve_optimize_max_workers_func=lambda *args: 1,
    )

    finalize_impl.assert_called_once()
    _, finalize_kwargs = finalize_impl.call_args
    assert finalize_kwargs["best_pair"] == best_pair


def test_run_optimize_target_default_expansion_strategy_matches_explicit_overrides(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet", directory=tmp_path
    )
    execution_setup = _execution_setup(tmp_path, reuse=True)
    best_pair = {"vocab_size_requested": 100, "min_frequency": 1}

    mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    mocker.patch.object(
        optimize_execution, "list_cleansed_parquet_files", return_value=[]
    )
    mocker.patch.object(
        optimize_execution,
        "resolve_optimize_corpus_for_target",
        return_value=(10, "abc123hash"),
    )
    mocker.patch.object(
        optimize_execution,
        "prepare_optimize_target_execution",
        return_value=execution_setup,
    )
    mocker.patch.object(
        optimize_retry.futures, "ProcessPoolExecutor", return_value=_StubExecutor()
    )
    mocker.patch.object(optimize_execution, "run_optimize_pilot_phase")
    expansion_phase = mocker.patch.object(
        optimize_execution, "run_optimize_expansion_phase"
    )
    mocker.patch.object(
        optimize_execution, "merge_run_logs", return_value=pl.DataFrame()
    )
    mocker.patch.object(
        optimize_execution, "scoped_history_for_trainer", return_value=pl.DataFrame()
    )
    mocker.patch.object(optimize_execution, "select_best_pair", return_value=best_pair)
    finalize_impl = mocker.Mock()

    optimize_execution.run_optimize_target(
        args=_run_target_args(finalize=True),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["fr"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1],
        vocab_schedule=[100, -1],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=lambda **kwargs: 10,
        prepare_optimize_seed_splits_func=lambda **kwargs: ({}, 0, 0),
        # Only the frontier is overridden (needed since the pilot phase is
        # mocked out, so pilot_results stays empty and the real heuristic
        # frontier selection would otherwise find nothing to expand) --
        # combo_prune_reason_func and expansion_strategy are both left unset,
        # relying purely on args.expansion_strategy="heuristic" resolution,
        # to prove that path matches today's explicit-override behavior.
        select_vocab_frontier_func=lambda **kwargs: {(1, 100)},
        finalize_optimize_target_func=finalize_impl,
        resolve_optimize_max_workers_func=lambda *args: 1,
    )

    expansion_phase.assert_called_once()
    _, expansion_kwargs = expansion_phase.call_args
    assert (
        expansion_kwargs["combo_prune_reason"]
        is optimize_expansion_strategy.combo_prune_reason
    )
    assert (
        expansion_kwargs["record_result"]
        is optimize_expansion_strategy.default_record_result
    )
    assert (
        expansion_kwargs["context"].expansion_strategy_name
        == optimize_expansion_strategy.HEURISTIC_STRATEGY_NAME
    )


def test_run_optimize_target_full_reuse_still_produces_non_empty_frontier(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    # The actual bug this session hit twice on FR: a 100%-reused pilot phase
    # left tracker.pilot_results empty, so the real (unmocked) frontier
    # selection found nothing and expansion never ran. This asserts the fix:
    # historical_results (seeded from run-log history at reuse time) must
    # populate tracker.pilot_results before frontier selection runs, so a
    # fully-reused run still finds the same frontier a fresh run would.
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet", directory=tmp_path
    )
    historical_results = [
        _result(seed=1, vocab=100, min_frequency=1, distance=0.01, rejected=False),
    ]
    execution_setup = _execution_setup(
        tmp_path, reuse=True, historical_results=historical_results
    )
    best_pair = {"vocab_size_requested": 100, "min_frequency": 1}

    mocker.patch.object(
        optimize_execution, "resolve_tokenizer_paths", return_value=artifact_paths
    )
    mocker.patch.object(
        optimize_execution, "list_cleansed_parquet_files", return_value=[]
    )
    mocker.patch.object(
        optimize_execution,
        "resolve_optimize_corpus_for_target",
        return_value=(10, "abc123hash"),
    )
    mocker.patch.object(
        optimize_execution,
        "prepare_optimize_target_execution",
        return_value=execution_setup,
    )
    mocker.patch.object(
        optimize_retry.futures, "ProcessPoolExecutor", return_value=_StubExecutor()
    )
    # Simulates 100% reuse: the pilot phase itself contributes nothing to
    # tracker.pilot_results, exactly like a real fully-reused run.
    mocker.patch.object(optimize_execution, "run_optimize_pilot_phase")
    expansion_phase = mocker.patch.object(
        optimize_execution, "run_optimize_expansion_phase"
    )
    mocker.patch.object(
        optimize_execution, "merge_run_logs", return_value=pl.DataFrame()
    )
    mocker.patch.object(
        optimize_execution, "scoped_history_for_trainer", return_value=pl.DataFrame()
    )
    mocker.patch.object(optimize_execution, "select_best_pair", return_value=best_pair)

    optimize_execution.run_optimize_target(
        args=_run_target_args(finalize=False),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["fr"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_policy(),
        seeds=[1, 2],
        vocab_schedule=[100],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=lambda **kwargs: 10,
        prepare_optimize_seed_splits_func=lambda **kwargs: ({}, 0, 0),
        resolve_optimize_max_workers_func=lambda *args: 1,
        # No select_vocab_frontier_func/combo_prune_reason_func override --
        # this is the whole point: the real heuristic strategy must find a
        # non-empty frontier purely from the historical seeding.
    )

    expansion_phase.assert_called_once()
    _, expansion_kwargs = expansion_phase.call_args
    assert expansion_kwargs["frontier"] == {(1, 100)}
