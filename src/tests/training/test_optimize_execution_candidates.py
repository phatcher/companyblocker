from __future__ import annotations

import argparse
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


def test_build_optimize_run_row_includes_phase_round_and_strategy():
    result = _result(
        seed=7, vocab=10000, min_frequency=1, distance=0.01, rejected=False
    )
    args = argparse.Namespace(validation_fraction=0.2)

    default_row = optimize_execution.build_optimize_run_row(
        result=result,
        args=args,
        session_id="sess-1",
        target_scope="country",
        systems=["fr"],
    )
    assert default_row["phase"] is None
    assert default_row["expansion_round"] is None
    assert default_row["strategy"] is None

    tagged_row = optimize_execution.build_optimize_run_row(
        result=result,
        args=args,
        session_id="sess-1",
        target_scope="country",
        systems=["fr"],
        phase="expansion",
        expansion_round=2,
        strategy="heuristic",
    )
    assert tagged_row["phase"] == "expansion"
    assert tagged_row["expansion_round"] == 2
    assert tagged_row["strategy"] == "heuristic"


def test_reconstruct_optimize_candidate_result_from_history_row_round_trips_all_fields():
    # Regression guard for the reuse-reconstruction fix: every field
    # build_optimize_run_row writes must map back onto OptimizeCandidateResult
    # exactly, since reconstruct_optimize_candidate_result_from_history_row is
    # its inverse.
    original = _result(
        seed=7, vocab=25000, min_frequency=8, distance=0.0068, rejected=False
    )
    args = argparse.Namespace(validation_fraction=0.2)
    row = optimize_execution.build_optimize_run_row(
        result=original,
        args=args,
        session_id="sess-1",
        target_scope="country",
        systems=["ie"],
    )

    reconstructed = (
        optimize_execution.reconstruct_optimize_candidate_result_from_history_row(row)
    )

    assert reconstructed.trainer == original.trainer
    assert reconstructed.seed == original.seed
    assert reconstructed.vocab_requested == original.vocab_requested
    assert reconstructed.min_frequency == original.min_frequency
    assert reconstructed.train_rows == original.train_rows
    assert reconstructed.validation_rows == original.validation_rows
    assert reconstructed.resolved_vocab_size == original.resolved_vocab_size
    assert reconstructed.train_metrics == original.train_metrics
    assert reconstructed.validation_metrics == original.validation_metrics
    assert (
        reconstructed.fertility_generalization_delta
        == original.fertility_generalization_delta
    )
    assert (
        reconstructed.unk_rate_generalization_delta
        == original.unk_rate_generalization_delta
    )
    assert reconstructed.model_path == original.model_path
    assert reconstructed.rejected == original.rejected
    assert reconstructed.rejection_reason == original.rejection_reason
    assert reconstructed.selection_score == original.selection_score
    assert reconstructed.elapsed_seconds == original.elapsed_seconds


def test_reconstruct_optimize_candidate_result_from_history_row_tolerates_missing_columns():
    minimal_row = {
        "seed": 42,
        "vocab_size_requested": 25000,
        "vocab_size_resolved": 25000,
        "min_frequency": 8,
    }

    reconstructed = (
        optimize_execution.reconstruct_optimize_candidate_result_from_history_row(
            minimal_row
        )
    )

    assert reconstructed.trainer == "wordpiece"
    assert reconstructed.seed == 42
    assert reconstructed.rejected is False
    assert reconstructed.validation_metrics["fertility_distance"] == 0.0


def test_policy_bridges_and_row_formatter_delegate_and_emit_expected_fields(mocker):
    rejection = mocker.patch.object(
        optimize_execution, "classify_candidate_rejection", return_value=(False, "")
    )
    selection = mocker.patch.object(
        optimize_execution, "compute_candidate_selection_score", return_value=0.55
    )
    policy = _policy()
    metrics = {
        "unk_rate": 0.001,
        "fertility_distance": 0.01,
        "fertility": 1.2,
        "token_count_median": 1.0,
        "token_count_p95": 2.0,
        "single_char_token_pct": 0.0,
        "token_count_mean": 1.0,
    }

    rejected, reason = optimize_execution.classify_candidate_rejection_with_policy(
        metrics=metrics, policy=policy
    )
    score = optimize_execution.compute_candidate_selection_score_with_policy(
        metrics=metrics,
        fertility_distance_for_selection=metrics["fertility_distance"],
        policy=policy,
        fertility_generalization_delta=0.0,
        unk_rate_generalization_delta=0.0,
    )

    assert rejected is False
    assert reason == ""
    assert score == 0.55
    rejection.assert_called_once()
    selection.assert_called_once()

    result = _result(
        seed=7, vocab=10000, min_frequency=1, distance=0.01, rejected=False
    )
    args = argparse.Namespace(validation_fraction=0.2)
    row = optimize_execution.build_optimize_run_row(
        result=result,
        args=args,
        session_id="sess-1",
        target_scope="country",
        systems=["fr"],
    )
    status = optimize_execution.format_optimize_gate_status(
        result=result, policy=policy
    )

    assert row["session_id"] == "sess-1"
    assert row["scope"] == "country"
    assert row["systems_key"] == "fr"
    assert row["vocab_size_requested"] == 10000
    assert "gate_values(" in status
    assert "selection_score=" in status


def test_format_optimize_gate_status_reports_none_reason_when_candidate_passes():
    policy = _policy()
    result = _result(seed=1, vocab=100, min_frequency=1, distance=0.02, rejected=False)

    status = optimize_execution.format_optimize_gate_status(
        result=result, policy=policy
    )

    assert "rejection_reason=none" in status
    assert f"selection_score={result.selection_score:.6f}" in status
    assert f"unk={result.validation_metrics['unk_rate']:.6f}" in status
    assert f"dist={result.validation_metrics['fertility_distance']:.4f}" in status


def test_format_optimize_gate_status_reports_reason_when_candidate_rejected():
    policy = _policy()
    result = _result(seed=1, vocab=100, min_frequency=1, distance=0.9, rejected=True)

    status = optimize_execution.format_optimize_gate_status(
        result=result, policy=policy
    )

    assert "rejection_reason=r" in status
    assert "rejection_reason=none" not in status


def test_print_optimize_candidate_result_labels_passing_candidate(capsys):
    policy = _policy()
    result = _result(seed=1, vocab=100, min_frequency=1, distance=0.02, rejected=False)

    optimize_execution.print_optimize_candidate_result(result=result, policy=policy)

    out = capsys.readouterr().out
    assert "=> PASS" in out
    assert "seed=1" in out
    assert "vocab=100" in out


def test_print_optimize_candidate_result_labels_rejected_candidate(capsys):
    policy = _policy()
    result = _result(seed=2, vocab=-1, min_frequency=2, distance=0.9, rejected=True)

    optimize_execution.print_optimize_candidate_result(result=result, policy=policy)

    out = capsys.readouterr().out
    assert "=> REJECT" in out
    # vocab=-1 (auto) is rendered as the "auto" tag, not the literal integer.
    assert "vocab=auto" in out
    assert "rejection_reason=r" in out


def test_run_optimize_candidate_stubs_train_and_evaluate_for_wrapper_behaviour(
    tmp_path: Path, mocker
):
    train_split_path = tmp_path / "train.parquet"
    validation_split_path = tmp_path / "validation.parquet"
    train_split_path.write_text("train", encoding="utf-8")
    validation_split_path.write_text("validation", encoding="utf-8")
    run_model_path = tmp_path / "candidate.json"
    run_model_path.write_text("model", encoding="utf-8")
    trainer_options = SimpleNamespace()

    train_tokenizer = mocker.patch.object(
        optimize_execution, "train_tokenizer_with_trainer", return_value=42000
    )
    evaluate = mocker.patch.object(
        optimize_execution,
        "evaluate_tokenizer_metrics",
        side_effect=[
            (
                {
                    "fertility": 1.35,
                    "unk_rate": 0.004,
                    "fertility_distance": 0.02,
                    "single_char_token_pct": 0.01,
                    "token_count_mean": 1.1,
                    "token_count_median": 1.0,
                    "token_count_p95": 2.0,
                },
                [],
                [],
                [],
            ),
            (
                {
                    "fertility": 1.20,
                    "unk_rate": 0.002,
                    "fertility_distance": 0.005,
                    "single_char_token_pct": 0.01,
                    "token_count_mean": 1.0,
                    "token_count_median": 1.0,
                    "token_count_p95": 2.0,
                },
                [],
                [],
                [],
            ),
        ],
    )
    classify = mocker.patch.object(
        optimize_execution,
        "classify_candidate_rejection_with_policy",
        return_value=(True, "ineligible"),
    )
    selection_score = mocker.patch.object(
        optimize_execution,
        "compute_candidate_selection_score_with_policy",
        return_value=0.123,
    )
    # The stub trainer above leaves "model" on disk, not a tokenizer file.
    mocker.patch.object(
        optimize_execution, "read_trained_vocab_size", return_value=31000
    )

    result = optimize_execution.run_optimize_candidate(
        trainer="wordpiece",
        train_split_path=train_split_path,
        validation_split_path=validation_split_path,
        seed=7,
        vocab_requested=-1,
        min_frequency=2,
        train_rows=100,
        validation_rows=20,
        policy=_policy(),
        run_model_path=run_model_path,
        trainer_options=trainer_options,
    )

    train_tokenizer.assert_called_once_with(
        trainer="wordpiece",
        corpus_path=train_split_path,
        tokenizer_path=run_model_path,
        vocab_size=None,
        min_frequency=2,
        options=trainer_options,
    )
    assert evaluate.call_count == 2
    evaluate.assert_any_call(
        corpus_path=validation_split_path,
        tokenizer_path=run_model_path,
        trainer="wordpiece",
        fertility_target=_policy().fertility_target,
        diagnostic_top_n=_policy().diagnostic_top_n,
    )
    evaluate.assert_any_call(
        corpus_path=train_split_path,
        tokenizer_path=run_model_path,
        trainer="wordpiece",
        fertility_target=_policy().fertility_target,
        diagnostic_top_n=_policy().diagnostic_top_n,
    )
    classify.assert_called_once()
    selection_score.assert_called_once()
    # Regression guard for the in-sample/generalization fix (2026-08-26): the
    # selection score must be computed from TRAIN-split fertility_distance
    # (0.005 here), not validation's (0.02) -- see optimize_search_methodology.md
    # Part 5 and compute_candidate_selection_score's docstring for why.
    assert selection_score.call_args.kwargs["fertility_distance_for_selection"] == 0.005
    assert result.rejected is True
    assert result.rejection_reason == "ineligible"
    assert result.model_path == ""
    assert result.resolved_vocab_size == 42000
    assert result.trained_vocab_size == 31000
    assert run_model_path.exists() is False


def test_run_optimize_candidate_with_rss_sample_returns_result_and_rss_tuple():
    fake = _fake_run_candidate()

    result, rss_bytes, worker_pid = (
        optimize_execution.run_optimize_candidate_with_rss_sample(
            fake,
            trainer="wordpiece",
            train_split_path=Path("train.parquet"),
            validation_split_path=Path("validation.parquet"),
            seed=1,
            vocab_requested=100,
            min_frequency=1,
            train_rows=10,
            validation_rows=2,
            policy=_policy(),
            run_model_path=Path("model.json"),
            trainer_options=TrainerOptions(),
        )
    )

    assert result.seed == 1
    assert rss_bytes is None or isinstance(rss_bytes, int)
    assert isinstance(worker_pid, int)


def test_reconstruct_optimize_candidate_result_from_history_row_ignores_resource_diagnostic_columns():
    # Regression guard: the new worker_rss_bytes/system_available_bytes columns
    # aren't part of OptimizeCandidateResult, so a history row that has them
    # (written by a run after this change) must still reconstruct cleanly,
    # exactly like a row that predates them (the tolerates_missing_columns
    # case above already covers their absence).
    row = {
        "seed": 42,
        "vocab_size_requested": 25000,
        "vocab_size_resolved": 25000,
        "min_frequency": 8,
        "worker_rss_bytes": 4_294_967_296,
        "system_available_bytes": 8_589_934_592,
    }

    reconstructed = (
        optimize_execution.reconstruct_optimize_candidate_result_from_history_row(row)
    )

    assert reconstructed.seed == 42
