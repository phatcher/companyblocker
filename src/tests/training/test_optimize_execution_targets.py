from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from company_tokenize import TrainerOptions

from training import optimize_execution
from training.optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
)
from training.runtime import OptimizeExecutionTracker, OptimizeRunContext
from workspace.roots import WorkspaceRoots, default_workspace_roots

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


def _preset_args(**overrides) -> argparse.Namespace:
    base = {
        "tokenizer": "sentencepiece",
        "optimize_preset": None,
        "validation_fraction": 0.20,
        "unk_rate_threshold": 0.005,
        "fertility_target": 1.25,
        "fertility_tolerance": 0.10,
        "fertility_min": 1.05,
        "fertility_max": 1.60,
        "token_count_median_max": 4.0,
        "token_count_p95_max": 6.0,
        "vocab_frontier_distance_tolerance": 0.02,
        "vocab_sizes": optimize_execution.DEFAULT_VOCAB_GRID,
        "min_frequencies": optimize_execution.DEFAULT_MIN_FREQ_GRID,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_apply_optimize_preset_auto_for_sentencepiece_uses_calibrated_defaults():
    args = _preset_args()

    preset_name, overrides = optimize_execution.apply_optimize_preset(
        args=args,
        presets=optimize_execution.OPTIMIZE_PRESETS,
        base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
        default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
        default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
    )

    assert preset_name == "sentencepiece"
    assert overrides == {}
    assert args.fertility_target == 1.35
    assert args.fertility_tolerance == 0.14
    assert args.fertility_max == 1.80
    assert args.optimize_preset == "sentencepiece"
    assert args.optimize_preset_overrides == {}


def test_apply_optimize_preset_respects_explicit_overrides():
    args = _preset_args(fertility_target=1.45)

    preset_name, overrides = optimize_execution.apply_optimize_preset(
        args=args,
        presets=optimize_execution.OPTIMIZE_PRESETS,
        base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
        default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
        default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
    )

    assert preset_name == "sentencepiece"
    assert args.fertility_target == 1.45
    assert overrides == {"fertility_target": 1.45}
    assert args.optimize_preset_overrides == {"fertility_target": 1.45}


def test_apply_optimize_preset_uses_explicit_optimize_preset_flag():
    # --tokenizer wordpiece would auto-select the "wordpiece" preset; an
    # explicit --optimize-preset overrides that trainer-based default.
    args = _preset_args(tokenizer="wordpiece", optimize_preset="sentencepiece")

    preset_name, _overrides = optimize_execution.apply_optimize_preset(
        args=args,
        presets=optimize_execution.OPTIMIZE_PRESETS,
        base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
        default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
        default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
    )

    assert preset_name == "sentencepiece"
    assert args.fertility_target == 1.35


def test_apply_optimize_preset_rejects_unknown_preset_name():
    args = _preset_args(optimize_preset="does-not-exist")

    with pytest.raises(ValueError, match="Unknown optimize preset"):
        optimize_execution.apply_optimize_preset(
            args=args,
            presets=optimize_execution.OPTIMIZE_PRESETS,
            base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
            default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
            default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
        )


def test_apply_optimize_preset_fills_grid_defaults_and_tracks_grid_overrides():
    presets = {
        "quick": optimize_execution.OptimizePreset(
            thresholds={},
            vocab_sizes="1000,2000",
            min_frequencies="1,2",
        ),
    }
    args = _preset_args(
        tokenizer="wordpiece",
        optimize_preset="quick",
        min_frequencies="9,10",
    )

    preset_name, overrides = optimize_execution.apply_optimize_preset(
        args=args,
        presets=presets,
        base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
        default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
        default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
    )

    assert preset_name == "quick"
    assert args.vocab_sizes == "1000,2000"
    assert args.min_frequencies == "9,10"
    assert overrides == {"min_frequencies": "9,10"}


def test_resolve_targets_requires_cleansed_data_for_global(
    workspace_roots: WorkspaceRoots,
):
    args = argparse.Namespace(systems=["global"])

    with pytest.raises(ValueError, match="No cleansed parquet shards found"):
        optimize_execution.resolve_targets(
            args=args,
            roots=workspace_roots,
            resolve_system_selection_func=lambda systems, add_global_for_all=False: (
                argparse.Namespace(
                    requested=["global"],
                    concrete=[],
                    include_global_target=True,
                )
            ),
            supported_system_codes_func=lambda: ["fr", "ie"],
            has_cleansed_parquet_func=lambda root, code: False,
        )


def test_build_apply_optimize_preset_delegates_to_apply_function(mocker):
    presets = {"wordpiece": optimize_execution.OPTIMIZE_PRESETS["wordpiece"]}
    base_defaults = {"validation_fraction": 0.2}
    default_vocab_grid = "1,2"
    default_min_freq_grid = "1,2"
    args = argparse.Namespace(tokenizer="wordpiece")

    apply_preset = optimize_execution.build_apply_optimize_preset(
        presets=presets,
        base_defaults=base_defaults,
        default_vocab_grid=default_vocab_grid,
        default_min_freq_grid=default_min_freq_grid,
    )

    delegated = mocker.patch.object(
        optimize_execution,
        "apply_optimize_preset",
        return_value=("auto", {}),
    )

    result = apply_preset(args)

    assert result == ("auto", {})
    delegated.assert_called_once_with(
        args=args,
        presets=presets,
        base_defaults=base_defaults,
        default_vocab_grid=default_vocab_grid,
        default_min_freq_grid=default_min_freq_grid,
    )


def test_build_train_mode_global_runner_delegates_with_bound_dependencies(mocker):
    def resolve_paths(**kwargs):
        return None

    def resolve_tokenizer_path(**kwargs):
        return Path("wordpiece.json")

    def sample_global(**kwargs):
        return 1

    def train_tokenizer(**kwargs):
        return 1000

    args = argparse.Namespace(tokenizer="wordpiece")
    trainer_options = SimpleNamespace()

    runner = optimize_execution.build_train_mode_global_runner(
        all_rows_target=321,
        resolve_tokenizer_paths_func=resolve_paths,
        resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path,
        sample_training_corpus_for_global_func=sample_global,
        train_tokenizer_with_trainer_func=train_tokenizer,
    )

    delegated = mocker.patch.object(optimize_execution, "run_train_mode_global")
    roots = default_workspace_roots(Path("C:/repo"))
    runner(args, roots, ["fr", "ie"], trainer_options)

    delegated.assert_called_once_with(
        args=args,
        roots=roots,
        systems=["fr", "ie"],
        trainer_options=trainer_options,
        all_rows_target=321,
        resolve_tokenizer_paths_func=resolve_paths,
        resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path,
        sample_training_corpus_for_global_func=sample_global,
        train_tokenizer_with_trainer_func=train_tokenizer,
    )


def test_build_train_mode_system_runner_delegates_with_bound_dependencies(mocker):
    def resolve_paths(**kwargs):
        return None

    def sample_corpus(**kwargs):
        return 1

    def resolve_tokenizer_path(**kwargs):
        return Path("wordpiece.json")

    def train_tokenizer(**kwargs):
        return 1000

    args = argparse.Namespace(tokenizer="wordpiece")
    trainer_options = SimpleNamespace()

    runner = optimize_execution.build_train_mode_system_runner(
        resolve_tokenizer_paths_func=resolve_paths,
        sample_training_corpus_func=sample_corpus,
        resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path,
        train_tokenizer_with_trainer_func=train_tokenizer,
    )

    delegated = mocker.patch.object(
        optimize_execution, "run_train_mode_system", return_value=17
    )
    roots = default_workspace_roots(Path("C:/repo"))
    line_count = runner(args, roots, "fr", 1, 3, trainer_options)

    assert line_count == 17
    delegated.assert_called_once_with(
        args=args,
        roots=roots,
        system_code="fr",
        system_index=1,
        total_systems=3,
        trainer_options=trainer_options,
        resolve_tokenizer_paths_func=resolve_paths,
        sample_training_corpus_func=sample_corpus,
        resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path,
        train_tokenizer_with_trainer_func=train_tokenizer,
    )


def test_parse_int_list_and_seed_resolution_cover_success_and_errors():
    assert optimize_execution.parse_int_list("1, 2, 3") == [1, 2, 3]
    assert optimize_execution.parse_int_list("100,-1", allow_negative_one=True) == [
        100,
        -1,
    ]

    with pytest.raises(ValueError, match="Expected positive integer list item"):
        optimize_execution.parse_int_list("0")

    with pytest.raises(ValueError, match="Expected at least one value"):
        optimize_execution.parse_int_list("  ,  ")

    explicit_seeds = optimize_execution.resolve_seed_values(
        args=argparse.Namespace(seeds="7,9", base_seed=1, num_runs=2),
        parse_int_list_func=optimize_execution.parse_int_list,
    )
    generated_seeds = optimize_execution.resolve_seed_values(
        args=argparse.Namespace(seeds="", base_seed=5, num_runs=3),
        parse_int_list_func=optimize_execution.parse_int_list,
    )
    assert explicit_seeds == [7, 9]
    assert generated_seeds == [5, 6, 7]

    with pytest.raises(ValueError, match="--num-runs must be greater than zero"):
        optimize_execution.resolve_seed_values(
            args=argparse.Namespace(seeds="", base_seed=1, num_runs=0),
            parse_int_list_func=optimize_execution.parse_int_list,
        )


def test_resolve_optimize_search_space_sorts_vocab_and_normalizes_frequencies(mocker):
    normalize = mocker.Mock(return_value=[1])
    args = argparse.Namespace(
        seeds="",
        base_seed=3,
        num_runs=2,
        vocab_sizes="5000,-1,10000",
        min_frequencies="1,2",
        tokenizer="sentencepiece",
    )

    seeds, vocab_schedule, min_frequencies = (
        optimize_execution.resolve_optimize_search_space(
            args,
            normalize_min_frequencies=normalize,
        )
    )

    assert seeds == [3, 4]
    assert vocab_schedule == [5000, 10000, -1]
    assert min_frequencies == [1]
    normalize.assert_called_once_with(trainer="sentencepiece", min_frequencies=[1, 2])


def test_validate_optimize_args_rejects_invalid_thresholds():
    valid = argparse.Namespace(
        validation_fraction=0.2,
        unk_rate_threshold=0.01,
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0,
        elbow_min_points=1,
        elbow_min_improvement=0,
        elbow_extra_steps=0,
        fertility_tolerance=0.1,
        fertility_min=1.0,
        fertility_max=2.0,
        token_count_median_max=3.0,
        token_count_p95_max=4.0,
        diagnostic_top_n=10,
        pilot_seed_count=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        worker_memory_floor_gb=2.0,
        memory_headroom_fraction=0.20,
        wave_chunk_size=4,
        max_parallel_waves=None,
        candidate_timeout_seconds=3600.0,
    )
    optimize_execution.validate_optimize_args(valid)

    invalid_validation = argparse.Namespace(
        **{**valid.__dict__, "validation_fraction": 1.0}
    )
    invalid_p95 = argparse.Namespace(**{**valid.__dict__, "token_count_p95_max": 2.0})
    invalid_pilot = argparse.Namespace(**{**valid.__dict__, "pilot_seed_count": 0})
    invalid_timeout = argparse.Namespace(
        **{**valid.__dict__, "candidate_timeout_seconds": 0.0}
    )

    with pytest.raises(ValueError, match="--validation-fraction"):
        optimize_execution.validate_optimize_args(invalid_validation)
    with pytest.raises(ValueError, match="--token-count-p95-max"):
        optimize_execution.validate_optimize_args(invalid_p95)
    with pytest.raises(ValueError, match="--pilot-seed-count"):
        optimize_execution.validate_optimize_args(invalid_pilot)
    with pytest.raises(ValueError, match="--candidate-timeout-seconds"):
        optimize_execution.validate_optimize_args(invalid_timeout)


def test_resolve_targets_raises_when_nothing_resolved(
    workspace_roots: WorkspaceRoots,
):
    selection = SimpleNamespace(concrete=[], include_global_target=False)

    with pytest.raises(ValueError, match="No training targets resolved"):
        optimize_execution.resolve_targets(
            args=argparse.Namespace(systems=[]),
            roots=workspace_roots,
            resolve_system_selection_func=lambda *a, **k: selection,
            supported_system_codes_func=list,
        )
