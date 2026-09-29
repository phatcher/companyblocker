"""Run a tokenizer train or optimize target end to end, and the optimize sweep's phases.

Train mode trains one tokenizer per target and finalizes it. Optimize mode builds a candidate job per `seed`/`vocab_requested`/`min_frequency` cell, runs the pilot, expansion and refining phases over them, and finalizes the winner. `apply_optimize_preset` resolves the named `--optimize-preset` for both modes, so the run manifest records it either way.

A preset is a named budget and quality bar, `wordpiece` and `sentencepiece` among them, supplying defaults only: an explicit flag still wins, and the manifest records the preset as `optimize_preset` and every field a flag overrode as `optimize_preset_overrides`. The two blocks differ by trainer, not by scope. The default grid is banded rather than log-spaced because a larger vocabulary is not always better: past a point it goes sparse, which is why candidates are held to a fertility band. Run `ie` first: it is the smallest real corpus, and it shows whether the fertility pattern holds before `gb` or `global` pays for the same grid.

The phases:

- Pilot: one seed over the vocab grid, ascending. A line, one seed at one `min_frequency`, stops at larger vocabs once its train fertility is below the target or has stopped moving, after one further point; `--pilot-early-stop false` runs the full grid. Each vocab point trains its highest `min_frequency` first and a lower one only when that falls short of the requested vocabulary.
- Equivalent candidates: one that reaches its requested vocabulary fixes every lower `min_frequency` at that vocab; one that falls short is that `min_frequency`'s plateau and fixes every larger vocab on it. An untrained candidate takes the result of the one that fixes it, and the run log names it in `carried_from`, beside `vocab_size_trained` and the requested `vocab_size_resolved`. `--skip-equivalent-candidates false` trains them all.
- Expansion: the remaining seeds, for the frontier: per `min_frequency`, the `--vocab-frontier-top-k` vocab points nearest the fertility target, kept to those within `--fertility-tolerance` when any is, plus every point within `--vocab-frontier-distance-tolerance` of the nearest.
- Refining: inside the winning bracket at the winning `min_frequency`, fertility is interpolated linearly in log vocab, and the vocab where it crosses the target is trained with one step either side, or the winner's two neighbours when fertility stays on one side of the target, every point on every seed. A winner on its `min_frequency`'s plateau is replaced as the centre by the best candidate that trained its full vocabulary, and no refined point is placed at or past the plateau. The step is the winner's seed-to-seed fertility noise over the fertility slope, floored at `--refine-vocab-step-minimum`, unless `--refine-vocab-step` sets it; `optimize_search_methodology.md` Part 8 derives it. Each refined point's residual against the fitted curve is logged. The phase skips itself, saying why, with no eligible interim winner, no room inside the bracket, or no history for a bracket point; `--refining-phase false` skips it outright.

Resampling a corpus that an optimize run still depends on is refused, since the corpus is shared across trainers and the run's canary tracks its content hash; `--optimize-wipe` proceeds anyway.

Each candidate job is resolved with its own `future.result(timeout=...)` budget (`--candidate-timeout-seconds`, default `optimize_retry.DEFAULT_CANDIDATE_TIMEOUT_SECONDS`), on both the fixed pool and the adaptive wave scheduler. A timeout recreates the pool the job belongs to (`optimize_retry.RecreatableExecutorPool` on the fixed-pool path) rather than killing the hung worker. Every terminal failure is appended to `<session_id>_candidate_failures.jsonl` beside the run log, and the sweep continues with the remaining candidates.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import shutil
import statistics
import sys
import threading
import time
from collections.abc import Callable
from concurrent import futures
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

import polars as pl
import psutil
from company_tokenize import (
    OptimizeCandidatePolicy,
    OptimizeCandidateResult,
    TrainerOptions,
    TrainingTarget,
    build_optimize_canary_payload,
    build_optimize_summary_payload,
    build_run_manifest,
    candidate_key,
    compute_corpus_content_hash,
    is_candidate_excluded,
    normalize_optimize_min_frequencies_for_trainer,
    resolve_candidate_model_suffix_for_trainer,
    resolve_metadata_path_for_trainer,
    resolve_optimize_canary_pointer_path,
    resolve_optimize_splits_dir,
    resolve_optimize_sweep_paths,
    resolve_tokenizer_path_for_trainer,
    resolve_tokenizer_paths,
    scoped_history_for_trainer,
    to_portable_path_str,
    tokenizer_directory_files,
    train_tokenizer_with_trainer,
    trainer_params_payload,
)
from company_tokenize.name_preprocessing import TRAINING_PREPROCESS_PROFILE
from company_tokenize.optimize import (
    append_run_log,
    classify_candidate_rejection,
    compute_candidate_selection_score,
    compute_optimize_grid_hash,
    compute_standard_error_of_difference,
    derive_refine_step_vocab,
    evaluate_refined_points_against_fit,
    evaluate_tokenizer_metrics,
    fertility_slope_at,
    merge_run_logs,
    predict_fertility_crossing,
    select_best_pair,
    summarize_cell_train_fertility_distance,
    write_seed_split_artifacts,
)
from company_tokenize.paths import scope_directory_files
from company_tokenize.training import OPTIMIZE_DIRNAME, load_tokenizer_backend

from training.promotion import (
    NAIVE_PROFILE,
    find_stored_candidate,
    point_profile_at_candidate,
    publish_naive_candidate,
    publish_promoted_candidate,
)
from workspace.artifact_layout import tokenizer_artifact_root, tokenizer_scope_dir
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.layer_layout import resolve_primary_files
from workspace.roots import WorkspaceRoots

from .optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
    compute_target_concurrency,
    historical_worker_rss_estimate,
    sample_process_rss_bytes,
)
from .optimize_expansion_strategy import (  # noqa: F401 -- re-exported for train_tokenizer.py  # noqa: F401 -- re-exported for existing test call sites  # noqa: F401 -- re-exported for existing test call sites
    EXPANSION_STRATEGIES,
    HEURISTIC_STRATEGY_NAME,
    ExpansionStrategy,
    build_crossing_vocab_points,
    combo_prune_reason,
    default_record_result,
    resolve_expansion_strategy,
    select_refining_bracket,
    select_vocab_frontier,
    vocab_sort_key,
)
from .optimize_retry import (
    DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
    MAX_ATTEMPTS_BY_CATEGORY,
    MEMORY_PRESSURE_RETRY_WORKERS,
    CandidateFailureCategory,
    CandidateFailureRecord,
    RecreatableExecutorPool,
    append_candidate_failure_log,
    classify_candidate_exception,
    default_process_pool_factory,
)
from .runtime import (
    OptimizeExecutionTracker,
    OptimizeRunContext,
    format_eta_hms,
    percentile_75,
    resolve_optimize_max_workers,
)
from .workloads import TrainGlobalRunner, TrainSystemRunner

ALL_ROWS_TARGET = 10**12
# This grid is banded, not a uniform log-spaced sweep, because vocab size is
# not monotonically better: past some point a larger vocab gets too sparse
# (many entries barely attested in the corpus), which is why evaluation
# rejects candidates outside a fertility band (fertility_min/max below)
# instead of just maximizing vocab size. Arrived at empirically from prior
# runs, not derived a priori.
#
# When running this sweep, do `ie` first -- it's the smallest real corpus, so
# it cheaply tells you expected job/execution size and lets you confirm the
# elbow/fertility pattern before spending compute on `gb` (~5.7M rows) or
# `global` (~28M rows) at the same grid. Scale up only once the pattern is
# confirmed to hold; don't assume it transfers.
DEFAULT_VOCAB_GRID = "10000,20000,25000,30000,40000,50000,60000,100000,150000,-1"
DEFAULT_MIN_FREQ_GRID = "1,2,3,5,8,12"
# The pilot's enforced stop (update_pilot_combo_cutoffs). The epsilon is what
# counts as fertility no longer moving between two vocab points: above the
# 0.000015 of noise measured on `ie` and `gb`, below their 0.001 real steps.
PILOT_PLATEAU_FERTILITY_EPSILON = 1e-4
PILOT_STOP_EXTRA_POINTS = 1
# Fertility targets differ by *trainer backend* (WordPiece vs SentencePiece),
# not by scope (country vs global) -- both scopes are evaluated against
# whichever block matches the trainer in use.
OPTIMIZE_BASE_DEFAULTS: dict[str, float | int] = {
    "validation_fraction": 0.20,
    "unk_rate_threshold": 0.005,
    "fertility_target": 1.25,
    "fertility_tolerance": 0.10,
    "fertility_min": 1.05,
    "fertility_max": 1.60,
    "token_count_median_max": 4.0,
    "token_count_p95_max": 6.0,
    "vocab_frontier_distance_tolerance": 0.02,
    "worker_memory_floor_gb": 2.0,
    "memory_headroom_fraction": 0.20,
    "wave_chunk_size": 4,
    # See optimize_retry.DEFAULT_CANDIDATE_TIMEOUT_SECONDS's own
    # docstring for why this is a placeholder, not a measured value.
    "candidate_timeout_seconds": DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
}
OPTIMIZE_SENTENCEPIECE_DEFAULTS: dict[str, float | int] = {
    "validation_fraction": 0.20,
    "unk_rate_threshold": 0.007,
    "fertility_target": 1.35,
    "fertility_tolerance": 0.14,
    "fertility_min": 1.00,
    "fertility_max": 1.80,
    "token_count_median_max": 4.5,
    "token_count_p95_max": 7.0,
    "vocab_frontier_distance_tolerance": 0.03,
}


class OptimizePreset(TypedDict):
    """One named bundle of optimize threshold and grid-search defaults,
    selectable via `--optimize-preset` instead of passing the dozen
    threshold/grid flags it stands in for individually.

    `thresholds` need only list what differs from a bare run's own argparse
    defaults -- exactly how `OPTIMIZE_SENTENCEPIECE_DEFAULTS` already lists
    a subset of `OPTIMIZE_BASE_DEFAULTS`'s keys below, relying on every
    field it omits already sitting at that same global default. A preset
    names *values*, never which metric acts as a gate: that set is fixed by
    `classify_candidate_rejection`/`compute_candidate_selection_score`, so a
    later change to a metric's role never has to revise every preset.
    """

    thresholds: dict[str, float | int]
    vocab_sizes: str
    min_frequencies: str


# The two trainer-keyed dicts above were already an unnamed preset pair,
# differing only by *trainer backend* -- this registry is what makes that
# pairing selectable and nameable instead of only reachable via --tokenizer.
# Both presets share the same grid today: nothing here has ever varied grid
# by trainer, so this is the hook a future budget-flavored preset (a
# smaller grid for a quick exploratory sweep) uses without disturbing these
# two.
OPTIMIZE_PRESETS: dict[str, OptimizePreset] = {
    "wordpiece": {
        "thresholds": OPTIMIZE_BASE_DEFAULTS,
        "vocab_sizes": DEFAULT_VOCAB_GRID,
        "min_frequencies": DEFAULT_MIN_FREQ_GRID,
    },
    "sentencepiece": {
        "thresholds": OPTIMIZE_SENTENCEPIECE_DEFAULTS,
        "vocab_sizes": DEFAULT_VOCAB_GRID,
        "min_frequencies": DEFAULT_MIN_FREQ_GRID,
    },
}

# Preserves this mechanism's pre-preset behavior: with no `--optimize-preset`
# given, the preset is still chosen from `--tokenizer` alone, exactly as
# `apply_optimize_threshold_profile` did before this registry existed.
DEFAULT_OPTIMIZE_PRESET = "wordpiece"
DEFAULT_OPTIMIZE_PRESET_BY_TRAINER: dict[str, str] = {
    "sentencepiece": "sentencepiece",
}


class OptimizeSeedSplit(TypedDict):
    train_split_path: Path
    validation_split_path: Path
    train_rows: int
    validation_rows: int


class OptimizeCandidateJob(TypedDict):
    seed: int
    vocab_requested: int
    min_frequency: int
    train_split_path: Path
    validation_split_path: Path
    train_rows: int
    validation_rows: int
    run_model_path: Path


class PilotComboState(TypedDict):
    points: int
    last_distance: float | None
    post_elbow_steps: int
    cutoff: bool
    # The enforced pilot stop (update_pilot_combo_cutoffs): the line's last
    # train-split fertility, the points run since its stop trigger fired
    # (None until it fires), and whether larger vocab points are now skipped.
    last_fertility: float | None
    stop_extra_points: int | None
    stopped: bool


class ComboProgressState(TypedDict):
    runs: int
    passes: int
    fertilities: list[float]


class OptimizeExecutionSetup(TypedDict):
    final_tokenizer_path: Path
    optimize_dir: Path
    models_dir: Path
    run_log_dir: Path
    run_log_path: Path
    summary_path: Path
    reuse_existing_candidates: bool
    excluded_candidate_keys: set[tuple[int, int, int]]
    historical_results: list[OptimizeCandidateResult]
    bootstrap_rss_bytes: int | None
    shard_run_log_path: Path
    session_run_log_df: pl.DataFrame


def build_apply_optimize_preset(
    *,
    presets: dict[str, OptimizePreset] | None = None,
    base_defaults: dict[str, float | int] | None = None,
    default_vocab_grid: str | None = None,
    default_min_freq_grid: str | None = None,
) -> Callable[[argparse.Namespace], tuple[str, dict[str, Any]]]:
    resolved_presets = presets or OPTIMIZE_PRESETS
    resolved_base_defaults = base_defaults or OPTIMIZE_BASE_DEFAULTS
    resolved_default_vocab_grid = default_vocab_grid or DEFAULT_VOCAB_GRID
    resolved_default_min_freq_grid = default_min_freq_grid or DEFAULT_MIN_FREQ_GRID

    def _apply(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
        return apply_optimize_preset(
            args=args,
            presets=resolved_presets,
            base_defaults=resolved_base_defaults,
            default_vocab_grid=resolved_default_vocab_grid,
            default_min_freq_grid=resolved_default_min_freq_grid,
        )

    return _apply


def build_train_mode_global_runner(
    *,
    all_rows_target: int = ALL_ROWS_TARGET,
    resolve_tokenizer_paths_func: Callable[..., Any] | None = None,
    resolve_tokenizer_path_for_trainer_func: Callable[..., Path] | None = None,
    sample_training_corpus_for_global_func: Callable[..., int] | None = None,
    train_tokenizer_with_trainer_func: Callable[..., int | None] | None = None,
) -> TrainGlobalRunner:
    def _run(
        args: argparse.Namespace,
        roots: WorkspaceRoots,
        systems: list[str],
        trainer_options: TrainerOptions,
    ) -> None:
        return run_train_mode_global(
            args=args,
            roots=roots,
            systems=systems,
            trainer_options=trainer_options,
            all_rows_target=all_rows_target,
            resolve_tokenizer_paths_func=resolve_tokenizer_paths_func,
            resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path_for_trainer_func,
            sample_training_corpus_for_global_func=sample_training_corpus_for_global_func,
            train_tokenizer_with_trainer_func=train_tokenizer_with_trainer_func,
        )

    return _run


def build_train_mode_system_runner(
    *,
    resolve_tokenizer_paths_func: Callable[..., Any] | None = None,
    sample_training_corpus_func: Callable[..., int] | None = None,
    resolve_tokenizer_path_for_trainer_func: Callable[..., Path] | None = None,
    train_tokenizer_with_trainer_func: Callable[..., int | None] | None = None,
) -> TrainSystemRunner:
    def _run(
        args: argparse.Namespace,
        roots: WorkspaceRoots,
        system_code: str,
        system_index: int,
        total_systems: int,
        trainer_options: TrainerOptions,
    ) -> int:
        return run_train_mode_system(
            args=args,
            roots=roots,
            system_code=system_code,
            system_index=system_index,
            total_systems=total_systems,
            trainer_options=trainer_options,
            resolve_tokenizer_paths_func=resolve_tokenizer_paths_func,
            sample_training_corpus_func=sample_training_corpus_func,
            resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path_for_trainer_func,
            train_tokenizer_with_trainer_func=train_tokenizer_with_trainer_func,
        )

    return _run


def corpus_content_hash_for_manifest(corpus_path: Path) -> str:
    """Corpus content hash for the run manifest, tolerating a missing corpus.

    By the time either promotion path calls this, the corpus has already
    been read for training, so a missing file here only happens in a stub/test
    scenario -- mirrors `write_token_score_artifact`'s own missing-corpus
    handling rather than raising, since the manifest schema still needs a
    valid str.
    """
    if not corpus_path.exists():
        return "missing"
    return compute_corpus_content_hash(corpus_path)


def write_token_score_artifact(
    *,
    corpus_path: Path,
    tokenizer_path: Path,
    trainer: str,
    roots: WorkspaceRoots | None = None,
) -> dict[str, Any]:
    """Emit a minimal tokenizer-token score artifact with token and score columns."""
    # Defer tfidf dependency import to keep module import lightweight for orchestration tests.
    from company_tokenize.tfidf import compute_tokenizer_token_tfidf_stats

    if not corpus_path.exists():
        return {
            "invoked": False,
            "status": "skipped_missing_corpus",
            "corpus_path": to_portable_path_str(
                corpus_path, project_root=roots.checkout if roots is not None else None
            ),
            "tokenizer_path": to_portable_path_str(
                tokenizer_path,
                project_root=roots.checkout if roots is not None else None,
            ),
            "trainer": trainer,
        }

    if not tokenizer_path.exists():
        return {
            "invoked": False,
            "status": "skipped_missing_tokenizer",
            "corpus_path": to_portable_path_str(
                corpus_path, project_root=roots.checkout if roots is not None else None
            ),
            "tokenizer_path": to_portable_path_str(
                tokenizer_path,
                project_root=roots.checkout if roots is not None else None,
            ),
            "trainer": trainer,
        }

    try:
        token_stats = compute_tokenizer_token_tfidf_stats(
            corpus_path,
            tokenizer_path=tokenizer_path,
            trainer=trainer,
        )

        artifact_path = tokenizer_directory_files(
            tokenizer_path.parent, trainer=trainer
        ).token_scores
        token_score = token_stats.select(
            [
                pl.col("token"),
                pl.col("idf").cast(pl.Float64).alias("score"),
            ]
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        token_score_rows = [
            {"token": str(token), "score": float(score)}
            for token, score in token_score.iter_rows()
        ]
        # Keep JSON valid while rendering one record per line for easier diff/review.
        if token_score_rows:
            row_lines = [json.dumps(row, ensure_ascii=True) for row in token_score_rows]
            payload = "[\n" + ",\n".join(row_lines) + "\n]\n"
        else:
            payload = "[]\n"
        artifact_path.write_text(payload, encoding="utf-8")

        return {
            "invoked": True,
            "status": "ok",
            "trainer": trainer,
            "corpus_path": to_portable_path_str(
                corpus_path, project_root=roots.checkout if roots is not None else None
            ),
            "tokenizer_path": to_portable_path_str(
                tokenizer_path,
                project_root=roots.checkout if roots is not None else None,
            ),
            "artifact_path": to_portable_path_str(
                artifact_path,
                project_root=roots.checkout if roots is not None else None,
            ),
            "token_count": int(token_score.height),
            "columns": ["token", "score"],
        }
    except Exception as exc:  # noqa: BLE001 -- pragma: no cover - defensive logging path
        return {
            "invoked": True,
            "status": "error",
            "trainer": trainer,
            "corpus_path": to_portable_path_str(
                corpus_path, project_root=roots.checkout if roots is not None else None
            ),
            "tokenizer_path": to_portable_path_str(
                tokenizer_path,
                project_root=roots.checkout if roots is not None else None,
            ),
            "error": str(exc),
        }


def _naive_settings(
    *, args: argparse.Namespace, trainer_options: TrainerOptions
) -> dict[str, object]:
    """What plain training records of its own settings, and what it looks a
    stored candidate up by: the sizes as asked for, since the size a trainer
    settles on is not known until it has trained."""
    return {
        "mode": "train",
        "requested_vocab_size": args.vocab_size,
        "min_frequency": int(args.min_frequency),
        "trainer_params": trainer_params_payload(
            trainer=args.tokenizer, options=trainer_options
        ),
    }


def reuse_stored_naive(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    system_code: str | None,
    corpus_path: Path,
    trainer_options: TrainerOptions,
) -> bool:
    """Point `naive` at the candidate already stored for this corpus and these
    settings, and say whether there was one.

    Training is not repeatable, so a second run would store a second, slightly
    different model where one is expected. With one stored, nothing is trained.
    """
    if not corpus_path.is_file():
        return False
    stored = find_stored_candidate(
        roots=roots,
        system=system_code,
        trainer=args.tokenizer,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
        corpus_path=corpus_path,
        settings=_naive_settings(args=args, trainer_options=trainer_options),
    )
    if stored is None:
        return False
    point_profile_at_candidate(roots, stored, profile=NAIVE_PROFILE)
    print(
        f"[tokenizer] naive={stored} is already stored for this corpus and "
        "these settings; nothing trained"
    )
    return True


def finalize_train_target(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    artifact_paths: Any,
    trainer_options: TrainerOptions,
    scope: str,
    systems: list[str],
    profile: str | None,
    tokenizer_path: Path,
    line_count: int,
    resolved_vocab_size: int | None,
    system_code: str | None = None,
    resolve_metadata_path_for_trainer_func: Callable[..., Path] | None = None,
    trainer_params_payload_builder: Callable[..., dict[str, Any]] | None = None,
    evaluate_tokenizer_metrics_func: Callable[..., Any] | None = None,
    build_run_manifest_func: Callable[..., dict[str, Any]] | None = None,
    publish_naive_candidate_func: Callable[..., Any] | None = None,
) -> None:
    resolve_metadata_path = (
        resolve_metadata_path_for_trainer_func or resolve_metadata_path_for_trainer
    )
    build_trainer_params_payload = (
        trainer_params_payload_builder or trainer_params_payload
    )
    evaluate_metrics = evaluate_tokenizer_metrics_func or evaluate_tokenizer_metrics
    build_manifest = build_run_manifest_func or build_run_manifest
    publish_naive = publish_naive_candidate_func or publish_naive_candidate

    token_score_artifact = write_token_score_artifact(
        corpus_path=artifact_paths.corpus_path,
        tokenizer_path=tokenizer_path,
        trainer=args.tokenizer,
        roots=roots,
    )
    if system_code is None:
        print(
            "[tokenizer] token_score_artifact "
            + f"scope={scope} status={token_score_artifact['status']} "
            + f"invoked={token_score_artifact['invoked']}"
        )
    else:
        print(
            "[tokenizer] token_score_artifact "
            + f"scope={scope} system={system_code} status={token_score_artifact['status']} "
            + f"invoked={token_score_artifact['invoked']}"
        )

    metadata_path = scope_directory_files(artifact_paths.directory).metadata
    trainer_metadata_path = resolve_metadata_path(
        trainer=args.tokenizer,
        scope_directory=artifact_paths.directory,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
    )
    trainer_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_manifest(
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode="train",
        scope=scope,
        systems=systems,
        profile=profile,
        trainer=args.tokenizer,
        all_rows=True,
        vocab_size=resolved_vocab_size
        if resolved_vocab_size is not None
        else args.vocab_size,
        min_frequency=args.min_frequency,
        name_col=args.name_col,
        preprocess_profile=TRAINING_PREPROCESS_PROFILE,
        corpus_path=to_portable_path_str(
            artifact_paths.corpus_path, project_root=roots.checkout
        ),
        corpus_content_hash=corpus_content_hash_for_manifest(
            artifact_paths.corpus_path
        ),
        tokenizer_path=to_portable_path_str(
            tokenizer_path, project_root=roots.checkout
        ),
        line_count=line_count,
        trainer_params=build_trainer_params_payload(
            trainer=args.tokenizer, options=trainer_options
        ),
        token_score_artifact=token_score_artifact,
        optimize_preset=getattr(args, "optimize_preset", None),
        optimize_preset_overrides=getattr(args, "optimize_preset_overrides", None),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trainer_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Measured against the whole training corpus, as the candidate an optimize
    # run chose is, so a naive candidate and a promoted one compare like for like.
    metrics, _, _, _ = evaluate_metrics(
        corpus_path=artifact_paths.corpus_path,
        tokenizer_path=tokenizer_path,
        trainer=args.tokenizer,
        fertility_target=OPTIMIZE_BASE_DEFAULTS["fertility_target"],
        diagnostic_top_n=1,
    )
    naive, replaced = publish_naive(
        roots=roots,
        system=system_code,
        trainer=args.tokenizer,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
        model_path=tokenizer_path,
        metadata_path=trainer_metadata_path,
        token_scores_path=tokenizer_directory_files(
            tokenizer_path.parent, trainer=args.tokenizer
        ).token_scores,
        corpus_path=artifact_paths.corpus_path,
        vocab_size=resolved_vocab_size
        if resolved_vocab_size is not None
        else args.vocab_size,
        min_frequency=args.min_frequency,
        metrics=metrics,
        parameters={
            **_naive_settings(args=args, trainer_options=trainer_options),
            "scope": scope,
            "systems": list(systems),
        },
        invocation=sys.argv,
    )
    print(f"[tokenizer] naive={naive} replaced={replaced}")

    print(f"[tokenizer] scope={scope}")
    print(f"[tokenizer] systems={','.join(systems)}")
    print(f"[tokenizer] corpus={artifact_paths.corpus_path}")
    print(f"[tokenizer] tokenizer={tokenizer_path}")
    print(f"[tokenizer] metadata={metadata_path}")
    print(f"[tokenizer] trainer_metadata={trainer_metadata_path}")


def run_train_mode_global(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    systems: list[str],
    trainer_options: TrainerOptions,
    all_rows_target: int = 10**12,
    resolve_tokenizer_paths_func: Callable[..., Any] | None = None,
    resolve_tokenizer_path_for_trainer_func: Callable[..., Path] | None = None,
    sample_training_corpus_for_global_func: Callable[..., int] | None = None,
    train_tokenizer_with_trainer_func: Callable[..., int | None] | None = None,
) -> None:
    from acquisition.tokenizer_ops import sample_training_corpus_for_global

    resolve_paths = resolve_tokenizer_paths_func or resolve_tokenizer_paths
    resolve_tokenizer_path = (
        resolve_tokenizer_path_for_trainer_func or resolve_tokenizer_path_for_trainer
    )
    sample_global_corpus = (
        sample_training_corpus_for_global_func or sample_training_corpus_for_global
    )
    train_tokenizer = train_tokenizer_with_trainer_func or train_tokenizer_with_trainer

    artifact_paths = resolve_paths(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope="global",
        profile=args.profile,
        corpus_filename=args.corpus_filename,
    )
    corpus_path = artifact_paths.corpus_path
    tokenizer_path = resolve_tokenizer_path(
        trainer=args.tokenizer,
        scope_directory=artifact_paths.directory,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
    )

    cleansed_dirs = [
        system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME) for system in systems
    ]
    print(f"[info] Building global corpus from {len(cleansed_dirs)} cleansed dir(s)")
    line_count = sample_global_corpus(
        cleansed_dirs=cleansed_dirs,
        corpus_path=corpus_path,
        sample_fraction=1.0,
        balance_mode="proportional",
        system_target_rows=all_rows_target,
        system_min_rows=0,
        system_max_rows=None,
        file_min_rows=0,
        balance_seed=42,
        name_col=args.name_col,
        allow_optimize_wipe=args.optimize_wipe,
    )

    if reuse_stored_naive(
        args=args,
        roots=roots,
        system_code=None,
        corpus_path=artifact_paths.corpus_path,
        trainer_options=trainer_options,
    ):
        return

    requested_vocab_label = args.vocab_size if args.vocab_size is not None else "auto"
    print(
        f"[info] Training global {args.tokenizer} model (vocab_size={requested_vocab_label})"
    )
    resolved_vocab_size = train_tokenizer(
        trainer=args.tokenizer,
        corpus_path=corpus_path,
        tokenizer_path=tokenizer_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        options=trainer_options,
    )
    finalize_train_target(
        args=args,
        roots=roots,
        artifact_paths=artifact_paths,
        trainer_options=trainer_options,
        scope="global",
        systems=systems,
        profile=args.profile,
        tokenizer_path=tokenizer_path,
        line_count=line_count,
        resolved_vocab_size=resolved_vocab_size,
    )


def run_train_mode_system(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    system_code: str,
    system_index: int,
    total_systems: int,
    trainer_options: TrainerOptions,
    has_cleansed_parquet_func: Callable[..., bool] | None = None,
    resolve_tokenizer_paths_func: Callable[..., Any] | None = None,
    sample_training_corpus_func: Callable[..., int] | None = None,
    resolve_tokenizer_path_for_trainer_func: Callable[..., Path] | None = None,
    train_tokenizer_with_trainer_func: Callable[..., int | None] | None = None,
) -> int:
    from acquisition.tokenizer_ops import sample_training_corpus

    check_cleansed_parquet = has_cleansed_parquet_func or has_cleansed_parquet
    resolve_paths = resolve_tokenizer_paths_func or resolve_tokenizer_paths
    sample_corpus = sample_training_corpus_func or sample_training_corpus
    resolve_tokenizer_path = (
        resolve_tokenizer_path_for_trainer_func or resolve_tokenizer_path_for_trainer
    )
    train_tokenizer = train_tokenizer_with_trainer_func or train_tokenizer_with_trainer

    print(
        f"[info] [{system_index}/{total_systems}] Training system tokenizer for '{system_code}'"
    )
    artifact_paths = resolve_paths(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope="country",
        system=system_code,
        profile=args.profile,
        corpus_filename=args.corpus_filename,
    )
    cleansed_dir = system_layer_dir(roots, system_code, layer=CLEANSED_LAYER_NAME)

    if not check_cleansed_parquet(roots, system_code):
        print(
            f"[warn] Skipping '{system_code}' system training: "
            f"no cleansed parquet shards found in {cleansed_dir}"
        )
        return 0

    line_count = sample_corpus(
        cleansed_dir=cleansed_dir,
        corpus_path=artifact_paths.corpus_path,
        name_col=args.name_col,
        allow_optimize_wipe=args.optimize_wipe,
    )

    if reuse_stored_naive(
        args=args,
        roots=roots,
        system_code=system_code,
        corpus_path=artifact_paths.corpus_path,
        trainer_options=trainer_options,
    ):
        return line_count

    requested_vocab_label = args.vocab_size if args.vocab_size is not None else "auto"
    print(
        f"[info] Training system {args.tokenizer} model for '{system_code}' (vocab_size={requested_vocab_label})"
    )
    tokenizer_path = resolve_tokenizer_path(
        trainer=args.tokenizer,
        scope_directory=artifact_paths.directory,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
    )
    resolved_vocab_size = train_tokenizer(
        trainer=args.tokenizer,
        corpus_path=artifact_paths.corpus_path,
        tokenizer_path=tokenizer_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        options=trainer_options,
    )
    finalize_train_target(
        args=args,
        roots=roots,
        artifact_paths=artifact_paths,
        trainer_options=trainer_options,
        scope="country",
        systems=[system_code],
        profile=None,
        tokenizer_path=tokenizer_path,
        line_count=line_count,
        resolved_vocab_size=resolved_vocab_size,
        system_code=system_code,
    )

    return line_count


def resolve_targets(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    resolve_system_selection_func: Callable[..., Any],
    supported_system_codes_func: Callable[[], list[str]],
    has_cleansed_parquet_func: Callable[..., bool] | None = None,
) -> list[TrainingTarget]:
    check_cleansed_parquet = has_cleansed_parquet_func or has_cleansed_parquet
    selection = resolve_system_selection_func(args.systems, add_global_for_all=True)
    targets: list[TrainingTarget] = []

    requested_countries = selection.concrete
    if requested_countries:
        targets.append(TrainingTarget(scope="country", systems=requested_countries))

    if selection.include_global_target:
        supported = supported_system_codes_func()
        active = [code for code in supported if check_cleansed_parquet(roots, code)]
        if not active:
            raise ValueError(
                "No cleansed parquet shards found for live systems under data/{system}/cleansed. "
                "Run cleanse first before global tokenizer training."
            )
        targets.append(TrainingTarget(scope="global", systems=active))

    if not targets:
        raise ValueError("No training targets resolved from --systems.")

    return targets


def classify_candidate_rejection_with_policy(
    *,
    metrics: dict[str, float],
    policy: OptimizeCandidatePolicy,
) -> tuple[bool, str]:
    return classify_candidate_rejection(
        metrics=metrics,
        unk_rate_threshold=policy.unk_rate_threshold,
        fertility_tolerance=policy.fertility_tolerance,
        fertility_min=policy.fertility_min,
        fertility_max=policy.fertility_max,
        token_count_median_max=policy.token_count_median_max,
        token_count_p95_max=policy.token_count_p95_max,
    )


def compute_candidate_selection_score_with_policy(
    *,
    metrics: dict[str, float],
    fertility_distance_for_selection: float,
    policy: OptimizeCandidatePolicy,
    fertility_generalization_delta: float,
    unk_rate_generalization_delta: float,
) -> float:
    return compute_candidate_selection_score(
        metrics=metrics,
        fertility_distance_for_selection=fertility_distance_for_selection,
        fertility_tolerance=policy.fertility_tolerance,
        unk_rate_threshold=policy.unk_rate_threshold,
        token_count_median_max=policy.token_count_median_max,
        token_count_p95_max=policy.token_count_p95_max,
        fertility_generalization_delta=fertility_generalization_delta,
        unk_rate_generalization_delta=unk_rate_generalization_delta,
    )


def read_trained_vocab_size(*, trainer: str, tokenizer_path: Path) -> int | None:
    """The size of the vocabulary a trained tokenizer file actually holds, or
    None when there is no file to read. A trainer reports the size it was
    asked for; a `min_frequency` that binds leaves the trained vocabulary
    short of it, and that shortfall is what marks a min_frequency plateau."""
    if not tokenizer_path.exists():
        return None
    backend = load_tokenizer_backend(tokenizer_path=tokenizer_path, trainer=trainer)
    return len(backend.vocab)


def backfill_trained_vocab_sizes(
    *, run_log_df: pl.DataFrame, trainer: str
) -> pl.DataFrame:
    """`run_log_df` with `vocab_size_trained` filled, for every row that lacks
    it, from the row's model file where that file is still on disk. A row
    logged without its trained size then takes part in everything that reads
    one (candidate_reached_full_vocab) exactly as a row logged with it."""
    if run_log_df.height == 0 or "model_path" not in run_log_df.columns:
        return run_log_df
    trained = (
        pl.col("vocab_size_trained").cast(pl.Int64)
        if "vocab_size_trained" in run_log_df.columns
        else pl.lit(None, dtype=pl.Int64)
    )
    run_log_df = run_log_df.with_columns(trained.alias("vocab_size_trained"))
    model_paths = (
        run_log_df.filter(
            pl.col("vocab_size_trained").is_null() & pl.col("model_path").is_not_null()
        )
        .get_column("model_path")
        .unique()
        .to_list()
    )
    sizes = {
        model_path: read_trained_vocab_size(
            trainer=trainer, tokenizer_path=Path(model_path)
        )
        for model_path in model_paths
        if model_path
    }
    sizes = {path: size for path, size in sizes.items() if size is not None}
    if not sizes:
        return run_log_df
    return run_log_df.with_columns(
        pl.when(pl.col("vocab_size_trained").is_null())
        .then(
            pl.col("model_path").replace_strict(
                sizes, default=None, return_dtype=pl.Int64
            )
        )
        .otherwise(pl.col("vocab_size_trained"))
        .alias("vocab_size_trained")
    )


def run_optimize_candidate(
    *,
    trainer: str,
    train_split_path: Path,
    validation_split_path: Path,
    seed: int,
    vocab_requested: int,
    min_frequency: int,
    train_rows: int,
    validation_rows: int,
    policy: OptimizeCandidatePolicy,
    run_model_path: Path,
    trainer_options: TrainerOptions,
) -> OptimizeCandidateResult:
    started = time.perf_counter()
    resolved_vocab_size = train_tokenizer_with_trainer(
        trainer=trainer,
        corpus_path=train_split_path,
        tokenizer_path=run_model_path,
        vocab_size=None if vocab_requested == -1 else vocab_requested,
        min_frequency=min_frequency,
        options=trainer_options,
    )
    trained_vocab_size = read_trained_vocab_size(
        trainer=trainer, tokenizer_path=run_model_path
    )

    validation_metrics, _, _, _ = evaluate_tokenizer_metrics(
        corpus_path=validation_split_path,
        tokenizer_path=run_model_path,
        trainer=trainer,
        fertility_target=policy.fertility_target,
        diagnostic_top_n=policy.diagnostic_top_n,
    )
    train_metrics, _, _, _ = evaluate_tokenizer_metrics(
        corpus_path=train_split_path,
        tokenizer_path=run_model_path,
        trainer=trainer,
        fertility_target=policy.fertility_target,
        diagnostic_top_n=policy.diagnostic_top_n,
    )

    rejected, rejection_reason = classify_candidate_rejection_with_policy(
        metrics=validation_metrics,
        policy=policy,
    )
    fertility_generalization_delta = (
        validation_metrics["fertility"] - train_metrics["fertility"]
    )
    unk_rate_generalization_delta = (
        validation_metrics["unk_rate"] - train_metrics["unk_rate"]
    )
    # Selection is driven by TRAIN-split fertility_distance, not validation's --
    # `--finalize` retrains on the full corpus and the promoted tokenizer is
    # evaluated in-sample against that same corpus, which train-split tracks
    # almost exactly (both are "fertility on data this vocab was built from").
    # Validation-split fertility crosses the target at a systematically
    # different vocab size (measured ~2,000 vocab off on `ie`), because a
    # WordPiece vocab always fits its own training data somewhat better than
    # held-out data -- see optimize_search_methodology.md Part 5.
    selection_score = compute_candidate_selection_score_with_policy(
        metrics=validation_metrics,
        fertility_distance_for_selection=train_metrics["fertility_distance"],
        policy=policy,
        fertility_generalization_delta=fertility_generalization_delta,
        unk_rate_generalization_delta=unk_rate_generalization_delta,
    )
    if rejected and policy.delete_rejected_models and run_model_path.exists():
        run_model_path.unlink()

    elapsed_seconds = time.perf_counter() - started
    return OptimizeCandidateResult(
        trainer=trainer,
        seed=seed,
        vocab_requested=vocab_requested,
        min_frequency=min_frequency,
        train_rows=train_rows,
        validation_rows=validation_rows,
        resolved_vocab_size=resolved_vocab_size,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        fertility_generalization_delta=fertility_generalization_delta,
        unk_rate_generalization_delta=unk_rate_generalization_delta,
        model_path="" if rejected else str(run_model_path),
        rejected=rejected,
        rejection_reason=rejection_reason,
        selection_score=selection_score,
        elapsed_seconds=elapsed_seconds,
        trained_vocab_size=trained_vocab_size,
    )


def run_optimize_candidate_with_rss_sample(
    run_candidate: Callable[..., OptimizeCandidateResult],
    /,
    **kwargs: Any,
) -> tuple[OptimizeCandidateResult, int | None, int]:
    """Submitted to the pool in place of run_candidate directly, so every
    candidate's worker memory gets sampled without changing run_candidate's
    own contract (still Callable[..., OptimizeCandidateResult] everywhere
    else -- existing fakes/tests untouched).

    Returns the worker's pid alongside the RSS sample: ProcessPoolExecutor
    workers are long-lived and handle many jobs in sequence, not one process
    per job, so a single job's end-of-call RSS conflates this job's cost with
    whatever the same worker accumulated across every prior job it ran. The
    pid is what lets that be told apart after the fact -- group samples by
    pid and look for RSS climbing across a worker's successive jobs (leaked/
    fragmented memory) versus staying flat (genuinely per-job cost).
    """
    result = run_candidate(**kwargs)
    return result, sample_process_rss_bytes(), os.getpid()


def build_optimize_run_row(
    *,
    result: OptimizeCandidateResult,
    args: argparse.Namespace,
    session_id: str,
    target_scope: str,
    systems: list[str],
    phase: str | None = None,
    expansion_round: int | None = None,
    strategy: str | None = None,
) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "session_id": session_id,
        "mode": "optimize",
        "phase": phase,
        "expansion_round": expansion_round,
        "strategy": strategy,
        "trainer": result.trainer,
        "scope": target_scope,
        "systems_key": ",".join(systems),
        "seed": result.seed,
        "train_rows": result.train_rows,
        "validation_rows": result.validation_rows,
        "validation_fraction": args.validation_fraction,
        "vocab_size_requested": result.vocab_requested,
        "vocab_size_resolved": result.resolved_vocab_size,
        "vocab_size_trained": getattr(result, "trained_vocab_size", None),
        "carried_from": getattr(result, "carried_from", None),
        "min_frequency": result.min_frequency,
        "unk_rate": result.validation_metrics["unk_rate"],
        "single_char_token_pct": result.validation_metrics["single_char_token_pct"],
        "fertility": result.validation_metrics["fertility"],
        "fertility_distance": result.validation_metrics["fertility_distance"],
        "token_count_mean": result.validation_metrics["token_count_mean"],
        "token_count_median": result.validation_metrics["token_count_median"],
        "token_count_p95": result.validation_metrics["token_count_p95"],
        "train_unk_rate": result.train_metrics["unk_rate"],
        "train_single_char_token_pct": result.train_metrics["single_char_token_pct"],
        "train_fertility": result.train_metrics["fertility"],
        "train_fertility_distance": result.train_metrics["fertility_distance"],
        "train_token_count_mean": result.train_metrics["token_count_mean"],
        "train_token_count_median": result.train_metrics["token_count_median"],
        "train_token_count_p95": result.train_metrics["token_count_p95"],
        "fertility_generalization_delta": result.fertility_generalization_delta,
        "unk_rate_generalization_delta": result.unk_rate_generalization_delta,
        "selection_score": result.selection_score,
        "rejected": result.rejected,
        "rejection_reason": result.rejection_reason,
        "model_path": result.model_path,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _metric_group_from_history_row(
    row: dict[str, Any], *, prefix: str
) -> dict[str, float]:
    def _get(key: str) -> float:
        value = row.get(f"{prefix}{key}")
        return float(value) if value is not None else 0.0

    return {
        "unk_rate": _get("unk_rate"),
        "single_char_token_pct": _get("single_char_token_pct"),
        "fertility": _get("fertility"),
        "fertility_distance": _get("fertility_distance"),
        "token_count_mean": _get("token_count_mean"),
        "token_count_median": _get("token_count_median"),
        "token_count_p95": _get("token_count_p95"),
    }


def reconstruct_optimize_candidate_result_from_history_row(
    row: dict[str, Any],
) -> OptimizeCandidateResult:
    """Rebuild the OptimizeCandidateResult a prior run produced, from one row
    of merged run-log history.

    The exact inverse of build_optimize_run_row -- every field there has a
    corresponding column here (validation_metrics/train_metrics are each the
    same 7 keys, unprefixed vs. train_-prefixed). Lets a candidate skipped via
    reuse (already on disk, never resubmitted this invocation) still
    participate in frontier selection and combo-progress accounting exactly
    as if it had just been freshly trained, instead of vanishing from
    tracker.pilot_results/combo_progress the way it does today. Non-essential
    fields degrade to a default rather than raising, so an older/partial
    schema row doesn't crash reconstruction -- matching the tolerance
    recompute_selection_scores already shows elsewhere for schema drift.
    """
    return OptimizeCandidateResult(
        trainer=str(row.get("trainer") or "wordpiece"),
        seed=int(row["seed"]),
        vocab_requested=int(row["vocab_size_requested"]),
        min_frequency=int(row["min_frequency"]),
        train_rows=int(row.get("train_rows") or 0),
        validation_rows=int(row.get("validation_rows") or 0),
        resolved_vocab_size=int(row["vocab_size_resolved"]),
        train_metrics=_metric_group_from_history_row(row, prefix="train_"),
        validation_metrics=_metric_group_from_history_row(row, prefix=""),
        fertility_generalization_delta=float(
            row.get("fertility_generalization_delta") or 0.0
        ),
        unk_rate_generalization_delta=float(
            row.get("unk_rate_generalization_delta") or 0.0
        ),
        model_path=str(row.get("model_path") or ""),
        rejected=bool(row.get("rejected") or False),
        rejection_reason=str(row.get("rejection_reason") or ""),
        selection_score=float(row.get("selection_score") or 0.0),
        elapsed_seconds=float(row.get("elapsed_seconds") or 0.0),
        trained_vocab_size=(
            None
            if row.get("vocab_size_trained") is None
            else int(row["vocab_size_trained"])
        ),
        carried_from=row.get("carried_from") or None,
    )


def format_optimize_gate_status(
    *,
    result: OptimizeCandidateResult,
    policy: OptimizeCandidatePolicy,
) -> str:
    metrics = result.validation_metrics
    gates = (
        f"unk<={policy.unk_rate_threshold:.6f}, "
        f"fert=[{policy.fertility_min:.4f},{policy.fertility_max:.4f}]"
    )
    score_scales = (
        f"dist/{policy.fertility_tolerance:.4f}, unk/{policy.unk_rate_threshold:.6f}"
    )
    values = (
        f"unk={metrics['unk_rate']:.6f}, "
        f"dist={metrics['fertility_distance']:.4f}, "
        f"fert={metrics['fertility']:.4f}, "
        f"median={metrics['token_count_median']:.4f}, "
        f"p95={metrics['token_count_p95']:.4f}"
    )
    reason = result.rejection_reason if result.rejection_reason else "none"
    return (
        f"gate_values({values}) gate_limits({gates}) scoring_scales({score_scales}) "
        f"rejection_reason={reason} selection_score={result.selection_score:.6f}"
    )


def print_optimize_candidate_result(
    *,
    result: OptimizeCandidateResult,
    policy: OptimizeCandidatePolicy,
) -> None:
    vocab_tag = "auto" if result.vocab_requested == -1 else str(result.vocab_requested)
    status = "REJECT" if result.rejected else "PASS"
    status_detail = format_optimize_gate_status(
        result=result,
        policy=policy,
    )
    print(
        f"[optimize] seed={result.seed} vocab={vocab_tag} min_freq={result.min_frequency} "
        f"unk_rate={result.validation_metrics['unk_rate']:.6f} fertility={result.validation_metrics['fertility']:.4f} "
        f"distance={result.validation_metrics['fertility_distance']:.4f} "
        f"delta_fertility={result.fertility_generalization_delta:+.4f} "
        f"token_median={result.validation_metrics['token_count_median']:.4f} "
        f"token_p95={result.validation_metrics['token_count_p95']:.4f} "
        f"score={result.selection_score:.6f} "
        f"=> {status} {status_detail}"
    )


def build_optimize_candidate_job(
    *,
    trainer: str,
    models_dir: Path,
    seed: int,
    vocab_requested: int,
    min_frequency: int,
    split: OptimizeSeedSplit,
) -> OptimizeCandidateJob:
    vocab_tag = "auto" if vocab_requested == -1 else str(vocab_requested)
    suffix = resolve_candidate_model_suffix_for_trainer(trainer=trainer)
    run_model_path = (
        models_dir / f"{trainer}_seed_{seed}_v{vocab_tag}_mf{min_frequency}.{suffix}"
    )
    return {
        "seed": seed,
        "vocab_requested": vocab_requested,
        "min_frequency": min_frequency,
        "train_split_path": split["train_split_path"],
        "validation_split_path": split["validation_split_path"],
        "train_rows": split["train_rows"],
        "validation_rows": split["validation_rows"],
        "run_model_path": run_model_path,
    }


def submit_optimize_candidate_jobs(
    *,
    executor: futures.Executor,
    candidate_jobs: list[OptimizeCandidateJob],
    context: OptimizeRunContext,
    run_candidate: Callable[..., OptimizeCandidateResult],
) -> dict[
    futures.Future[tuple[OptimizeCandidateResult, int | None, int]],
    OptimizeCandidateJob,
]:
    future_map: dict[
        futures.Future[tuple[OptimizeCandidateResult, int | None, int]],
        OptimizeCandidateJob,
    ] = {}
    for job in candidate_jobs:
        future = executor.submit(
            run_optimize_candidate_with_rss_sample,
            run_candidate,
            trainer=context.args.tokenizer,
            train_split_path=job["train_split_path"],
            validation_split_path=job["validation_split_path"],
            seed=int(job["seed"]),
            vocab_requested=int(job["vocab_requested"]),
            min_frequency=int(job["min_frequency"]),
            train_rows=int(job["train_rows"]),
            validation_rows=int(job["validation_rows"]),
            policy=context.candidate_policy,
            run_model_path=job["run_model_path"],
            trainer_options=context.trainer_options,
        )
        future_map[future] = job
    return future_map


def collect_candidate_jobs_for_pilot_vocab(
    *,
    context: OptimizeRunContext,
    vocab_requested: int,
    pilot_seeds: list[int],
    min_frequencies: list[int],
    seed_splits: dict[int, OptimizeSeedSplit],
    pilot_combo_state: dict[tuple[int, int], PilotComboState] | None = None,
) -> tuple[list[OptimizeCandidateJob], int]:
    # A (seed, min_frequency) combo is skipped only once it is "stopped"
    # (update_pilot_combo_cutoffs), never on its elbow-cutoff state. The auto
    # point (-1) is never skipped: its size is the trainer's estimate, not a
    # position in the ascending schedule.
    candidate_specs: list[tuple[int, int, int, OptimizeSeedSplit]] = [
        (seed, vocab_requested, min_frequency, seed_splits[seed])
        for seed in pilot_seeds
        for min_frequency in min_frequencies
        if vocab_requested == -1
        or pilot_combo_state is None
        or not pilot_combo_state[(seed, min_frequency)]["stopped"]
    ]

    return _collect_candidate_jobs(context=context, candidate_specs=candidate_specs)


def collect_candidate_jobs_for_expansion_seed(
    *,
    context: OptimizeRunContext,
    seed: int,
    active_combos: list[tuple[int, int]],
    split: OptimizeSeedSplit,
) -> tuple[list[OptimizeCandidateJob], int]:
    candidate_specs = [
        (seed, vocab_requested, min_frequency, split)
        for min_frequency, vocab_requested in active_combos
    ]
    return _collect_candidate_jobs(context=context, candidate_specs=candidate_specs)


def collect_candidate_jobs_for_refining_vocab(
    *,
    context: OptimizeRunContext,
    vocab_requested: int,
    min_frequency: int,
    seeds: list[int],
    seed_splits: dict[int, OptimizeSeedSplit],
) -> tuple[list[OptimizeCandidateJob], int]:
    """Every seed at one refined vocab point, holding min_frequency fixed at
    the winning value. Unlike the pilot/expansion phases, nothing here
    prunes or defers which seeds run: refined points get full seeds
    unconditionally (optimize_search_methodology.md Part 8 -- at 1,000-vocab
    spacing neighbouring points differ by order sigma, so 1-seed pilot
    pruning would select close to at random).
    """
    candidate_specs = [
        (seed, vocab_requested, min_frequency, seed_splits[seed]) for seed in seeds
    ]
    return _collect_candidate_jobs(context=context, candidate_specs=candidate_specs)


def _collect_candidate_jobs(
    *,
    context: OptimizeRunContext,
    candidate_specs: list[tuple[int, int, int, OptimizeSeedSplit]],
) -> tuple[list[OptimizeCandidateJob], int]:
    candidate_jobs: list[OptimizeCandidateJob] = []
    skipped_existing_candidates = 0
    for seed, vocab_requested, min_frequency, split in candidate_specs:
        candidate_job = build_optimize_candidate_job(
            trainer=context.args.tokenizer,
            models_dir=context.models_dir,
            seed=seed,
            vocab_requested=vocab_requested,
            min_frequency=min_frequency,
            split=split,
        )
        if context.reuse_existing_candidates and is_candidate_excluded(
            seed=int(seed),
            vocab_requested=int(vocab_requested),
            min_frequency=int(min_frequency),
            excluded_candidate_keys=context.excluded_candidate_keys,
        ):
            skipped_existing_candidates += 1
            continue
        candidate_jobs.append(candidate_job)

    return candidate_jobs, skipped_existing_candidates


def log_optimize_progress(
    *,
    phase: str,
    completed_jobs: int,
    submitted_jobs: int,
    pending_batch_jobs: int,
    upper_bound_jobs: int,
    max_workers: int,
    observed_job_elapsed_seconds: list[float],
) -> None:
    remaining_submitted = max(0, (submitted_jobs + pending_batch_jobs) - completed_jobs)
    remaining_upper = max(0, upper_bound_jobs - completed_jobs)
    avg_job = (
        (sum(observed_job_elapsed_seconds) / len(observed_job_elapsed_seconds))
        if observed_job_elapsed_seconds
        else None
    )

    p75_job = percentile_75(observed_job_elapsed_seconds)

    conservative_job = None
    if avg_job is not None and p75_job is not None:
        conservative_job = max(avg_job, p75_job)
    elif avg_job is not None:
        conservative_job = avg_job

    eta_submitted = None
    eta_upper = None
    if conservative_job is not None and max_workers > 0:
        submitted_parallelism = (
            max(1, min(max_workers, remaining_submitted))
            if remaining_submitted > 0
            else 1
        )
        upper_parallelism = (
            max(1, min(max_workers, remaining_upper)) if remaining_upper > 0 else 1
        )
        eta_submitted = (conservative_job * remaining_submitted) / submitted_parallelism
        eta_upper = (conservative_job * remaining_upper) / upper_parallelism

    print(
        f"[optimize] progress phase={phase} completed={completed_jobs} submitted={submitted_jobs} "
        f"pending_batch={pending_batch_jobs} remaining_submitted={remaining_submitted} "
        f"remaining_upper_bound={remaining_upper} "
        f"eta_submitted={format_eta_hms(eta_submitted)} eta_upper_bound={format_eta_hms(eta_upper)}"
    )


def record_completed_candidate_result(
    *,
    result: OptimizeCandidateResult,
    worker_rss_bytes: int | None,
    worker_pid: int,
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    include_in_pilot: bool,
    expansion_round: int | None = None,
    phase_override: str | None = None,
) -> None:
    system_available_bytes = int(psutil.virtual_memory().available)
    tracker.candidate_results.append(result)
    if include_in_pilot:
        tracker.pilot_results.append(result)
    tracker.completed_jobs += 1
    tracker.observed_job_elapsed_seconds.append(result.elapsed_seconds)

    # phase_override names a phase run_row logs that isn't distinguished by
    # include_in_pilot alone (today, only "refining": it's neither pilot
    # data nor an expansion-frontier round, but reuses this same batch
    # pipeline). None keeps every existing pilot/expansion call site's
    # behavior unchanged.
    phase = phase_override or ("pilot" if include_in_pilot else "expansion")
    run_row = build_optimize_run_row(
        result=result,
        args=context.args,
        session_id=context.session_id,
        target_scope=context.target_scope,
        systems=context.systems,
        phase=phase,
        expansion_round=expansion_round,
        strategy=context.expansion_strategy_name,
    )
    run_row["worker_rss_bytes"] = worker_rss_bytes
    run_row["worker_pid"] = worker_pid
    run_row["system_available_bytes"] = system_available_bytes
    tracker.session_run_log_df = append_run_log(
        run_log_path=context.shard_run_log_path,
        run_rows=[run_row],
        existing_df=tracker.session_run_log_df,
    )
    print_optimize_candidate_result(result=result, policy=context.candidate_policy)
    vocab_tag = "auto" if result.vocab_requested == -1 else str(result.vocab_requested)
    rss_gb = (
        f"{worker_rss_bytes / 2**30:.2f}" if worker_rss_bytes is not None else "n/a"
    )
    print(
        f"[optimize] worker_rss seed={result.seed} vocab={vocab_tag} "
        f"min_freq={result.min_frequency} rss_gb={rss_gb} "
        f"available_gb={system_available_bytes / 2**30:.2f} pid={worker_pid}"
    )


def candidate_reached_full_vocab(result: OptimizeCandidateResult) -> bool | None:
    """Whether the trainer produced the whole vocabulary it was asked for,
    or None when the trained size was not recorded.

    True means `min_frequency` did not bind: the same request at any lower
    `min_frequency` trains the same vocabulary. False means it did, and the
    result is that `min_frequency`'s plateau: any larger vocab request at
    this `min_frequency` trains the same vocabulary. "The same" is up to the
    trainer's tie-breaking among equal-count merges, a handful of tokens.
    """
    trained_vocab_size = getattr(result, "trained_vocab_size", None)
    if trained_vocab_size is None:
        return None
    return int(trained_vocab_size) >= int(result.resolved_vocab_size)


def candidate_cell_tag(*, vocab_requested: int, min_frequency: int) -> str:
    vocab_tag = "auto" if vocab_requested == -1 else str(vocab_requested)
    return f"v{vocab_tag}_mf{min_frequency}"


def carry_candidate_result(
    *,
    source: OptimizeCandidateResult,
    vocab_requested: int,
    min_frequency: int,
) -> OptimizeCandidateResult:
    """The result of a cell that was not trained because `source` already
    fixes it (candidate_reached_full_vocab). Every metric, the trained size
    and the model path are the source's; `carried_from` names the cell that
    was actually trained and no time is charged."""
    return dataclasses.replace(
        source,
        vocab_requested=vocab_requested,
        min_frequency=min_frequency,
        resolved_vocab_size=(
            source.resolved_vocab_size
            if vocab_requested in (-1, source.vocab_requested)
            else vocab_requested
        ),
        elapsed_seconds=0.0,
        carried_from=source.carried_from
        or candidate_cell_tag(
            vocab_requested=source.vocab_requested,
            min_frequency=source.min_frequency,
        ),
    )


def record_carried_candidate_result(
    *,
    result: OptimizeCandidateResult,
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    include_in_pilot: bool,
    expansion_round: int | None = None,
) -> None:
    """record_completed_candidate_result for a carried result: the same
    run-log row and the same place in the tracker's results, so frontier
    selection, pruning and a resumed run read it as any other cell, but no
    job was run, so the job counters and the elapsed-time sample the ETA is
    built from are left alone."""
    tracker.candidate_results.append(result)
    if include_in_pilot:
        tracker.pilot_results.append(result)
    run_row = build_optimize_run_row(
        result=result,
        args=context.args,
        session_id=context.session_id,
        target_scope=context.target_scope,
        systems=context.systems,
        phase="pilot" if include_in_pilot else "expansion",
        expansion_round=expansion_round,
        strategy=context.expansion_strategy_name,
    )
    run_row["worker_rss_bytes"] = None
    run_row["worker_pid"] = None
    run_row["system_available_bytes"] = None
    tracker.session_run_log_df = append_run_log(
        run_log_path=context.shard_run_log_path,
        run_rows=[run_row],
        existing_df=tracker.session_run_log_df,
    )
    vocab_tag = "auto" if result.vocab_requested == -1 else str(result.vocab_requested)
    print(
        f"[optimize] carried seed={result.seed} vocab={vocab_tag} "
        f"min_freq={result.min_frequency} from={result.carried_from} "
        f"train_fertility={result.train_metrics['fertility']:.6f}"
    )


def _retry_candidate_job(
    *,
    job: OptimizeCandidateJob,
    context: OptimizeRunContext,
    run_candidate: Callable[..., OptimizeCandidateResult],
    category: CandidateFailureCategory,
    timeout_seconds: float,
    executor_factory: Callable[[int], futures.Executor],
) -> tuple[tuple[OptimizeCandidateResult, int | None, int] | None, str | None]:
    """Retry one job on its own short-lived, isolated single-use pool.

    memory_pressure retries at MEMORY_PRESSURE_RETRY_WORKERS (the minimum --
    isolating the retry from every *other* concurrent candidate, not merely
    shrinking their number); transient_io retries at the same size (1),
    since concurrency was never the suspected cause there. Returns
    (result_tuple, None) on success, or (None, error_message) if this
    attempt also failed -- never both.
    """
    worker_count = MEMORY_PRESSURE_RETRY_WORKERS if category == "memory_pressure" else 1
    executor = executor_factory(worker_count)
    hung = False
    try:
        future_map = submit_optimize_candidate_jobs(
            executor=executor,
            candidate_jobs=[job],
            context=context,
            run_candidate=run_candidate,
        )
        (future,) = future_map
        try:
            return future.result(timeout=timeout_seconds), None
        except futures.TimeoutError:
            hung = True
            return None, f"retry exceeded {timeout_seconds:.0f}s runtime budget"
        except Exception as exc:  # noqa: BLE001 -- reported to the caller, not raised
            return None, str(exc) or exc.__class__.__name__
    finally:
        # Never wait=True here: a retry that itself hung must not block this
        # teardown, for the same reason the primary pool never waits on one.
        executor.shutdown(wait=not hung, cancel_futures=True)


def _retry_or_fail_candidate_job(
    *,
    job: OptimizeCandidateJob,
    exc: Exception,
    context: OptimizeRunContext,
    run_candidate: Callable[..., OptimizeCandidateResult],
    timeout_seconds: float,
    retry_executor_factory: Callable[[int], futures.Executor],
) -> tuple[
    tuple[OptimizeCandidateResult, int | None, int] | None,
    CandidateFailureRecord | None,
]:
    """Classify a primary attempt's exception and retry per the taxonomy.

    Returns (outcome, None) if a retry eventually succeeded, or (None,
    failure_record) once every attempt the category allows is exhausted.
    """
    category = classify_candidate_exception(exc)
    max_attempts = MAX_ATTEMPTS_BY_CATEGORY[category]
    attempt = 1
    last_message = str(exc) or exc.__class__.__name__
    while attempt < max_attempts:
        attempt += 1
        outcome, retry_error = _retry_candidate_job(
            job=job,
            context=context,
            run_candidate=run_candidate,
            category=category,
            timeout_seconds=timeout_seconds,
            executor_factory=retry_executor_factory,
        )
        if outcome is not None:
            return outcome, None
        last_message = retry_error or last_message

    failure = CandidateFailureRecord(
        seed=int(job["seed"]),
        vocab_requested=int(job["vocab_requested"]),
        min_frequency=int(job["min_frequency"]),
        category=category,
        attempt_count=attempt,
        error_message=last_message,
    )
    return None, failure


def _resolve_candidate_future(
    *,
    future: futures.Future[tuple[OptimizeCandidateResult, int | None, int]],
    job: OptimizeCandidateJob,
    context: OptimizeRunContext,
    run_candidate: Callable[..., OptimizeCandidateResult],
    timeout_seconds: float,
    retry_executor_factory: Callable[[int], futures.Executor],
    on_hang: Callable[[], None] | None,
) -> tuple[
    tuple[OptimizeCandidateResult, int | None, int] | None,
    CandidateFailureRecord | None,
]:
    """Resolve one already-submitted future per the candidate retry/error-policy
    taxonomy (optimize_retry.classify_candidate_exception).

    A primary TimeoutError is always terminal (a hang reproduces on a
    deterministic input) and fires on_hang so the caller can tear down and
    recreate the pool this future's executor belongs to before further jobs
    are submitted to it -- concurrent.futures has no supported way to kill
    the hung worker itself. Any other exception is classified and retried
    per _retry_or_fail_candidate_job. Exactly one of the two return values
    is non-None.
    """
    try:
        return future.result(timeout=timeout_seconds), None
    except futures.TimeoutError:
        if on_hang is not None:
            on_hang()
        return None, CandidateFailureRecord(
            seed=int(job["seed"]),
            vocab_requested=int(job["vocab_requested"]),
            min_frequency=int(job["min_frequency"]),
            category="terminal",
            attempt_count=1,
            error_message=f"exceeded {timeout_seconds:.0f}s runtime budget (hung worker)",
        )
    except Exception as exc:  # noqa: BLE001 -- classified and recorded, not re-raised
        return _retry_or_fail_candidate_job(
            job=job,
            exc=exc,
            context=context,
            run_candidate=run_candidate,
            timeout_seconds=timeout_seconds,
            retry_executor_factory=retry_executor_factory,
        )


def drain_candidate_futures(
    *,
    future_map: dict[
        futures.Future[tuple[OptimizeCandidateResult, int | None, int]],
        OptimizeCandidateJob,
    ],
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    run_candidate: Callable[..., OptimizeCandidateResult],
    include_in_pilot: bool,
    expansion_round: int | None = None,
    phase_override: str | None = None,
    timeout_seconds: float | None = None,
    retry_executor_factory: Callable[[int], futures.Executor] = (
        default_process_pool_factory
    ),
    failure_log_path: Path | None = None,
    on_hang: Callable[[], None] | None = None,
    on_worker_rss_sample: Callable[[int | None], None] | None = None,
    record_lock: threading.Lock | None = None,
) -> list[OptimizeCandidateResult]:
    """Drain future_map per the candidate retry/error-policy taxonomy
    (see optimize_retry.py's module docstring for the taxonomy itself).

    Iterates future_map in submission order (not futures.as_completed's
    completion order): each job needs its own bounded future.result(timeout=...)
    call, and as_completed's timeout parameter bounds the whole remaining
    iteration rather than any one job, so it can't express a per-job budget.
    Since a batch is fully drained before its caller moves on regardless of
    order, this doesn't change total wall time -- only which future is
    waited on first.

    run_candidate/retry_executor_factory are needed only for a memory_pressure/
    transient_io retry's fresh single-job submission, not for a normal
    completion. record_lock, given by the (multi-threaded) wave scheduler,
    guards both a successful completion's tracker/run-log mutation and a
    failure's sidecar-log append; the (single-threaded) fixed-pool path
    passes none.
    """
    resolved_timeout = (
        context.candidate_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )
    resolved_failure_log_path = (
        failure_log_path
        if failure_log_path is not None
        else context.candidate_failure_log_path
    )
    lock_cm: Any = record_lock if record_lock is not None else contextlib.nullcontext()

    batch_results: list[OptimizeCandidateResult] = []
    for future, job in future_map.items():
        outcome, failure = _resolve_candidate_future(
            future=future,
            job=job,
            context=context,
            run_candidate=run_candidate,
            timeout_seconds=resolved_timeout,
            retry_executor_factory=retry_executor_factory,
            on_hang=on_hang,
        )
        if failure is not None:
            if resolved_failure_log_path is not None:
                with lock_cm:
                    append_candidate_failure_log(resolved_failure_log_path, [failure])
            print(
                f"[optimize] candidate_failure seed={failure.seed} "
                f"vocab={failure.vocab_requested} min_freq={failure.min_frequency} "
                f"category={failure.category} attempts={failure.attempt_count} "
                f"error={failure.error_message}"
            )
            continue

        assert outcome is not None  # nosec B101 - exactly one of outcome/failure is set
        result, worker_rss_bytes, worker_pid = outcome
        if on_worker_rss_sample is not None:
            on_worker_rss_sample(worker_rss_bytes)
        batch_results.append(result)
        with lock_cm:
            record_completed_candidate_result(
                result=result,
                worker_rss_bytes=worker_rss_bytes,
                worker_pid=worker_pid,
                context=context,
                tracker=tracker,
                include_in_pilot=include_in_pilot,
                expansion_round=expansion_round,
                phase_override=phase_override,
            )

    return batch_results


def run_candidate_jobs_in_waves(
    *,
    candidate_jobs: list[OptimizeCandidateJob],
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    run_candidate: Callable[..., OptimizeCandidateResult],
    concurrency_controller: AdaptiveConcurrencyController,
    memory_budget_ledger: MemoryBudgetLedger,
    wave_chunk_size: int,
    max_parallel_waves: int,
    include_in_pilot: bool,
    expansion_round: int | None = None,
    phase_override: str | None = None,
    executor_factory: Callable[[int], futures.Executor] = (
        lambda max_workers: futures.ProcessPoolExecutor(max_workers=max_workers)
    ),
) -> list[OptimizeCandidateResult]:
    """Process candidate_jobs through short-lived waves of workers instead of
    one persistent pool. Each wave reserves a memory budget, spins up its own
    executor (a real ProcessPoolExecutor in production; executor_factory is
    only overridden by tests, to substitute ThreadPoolExecutor and stay
    subprocess-free), runs a small bounded chunk of jobs, and tears the
    pool down -- reclaiming every byte its workers held, including whatever a
    native allocator (Arrow/Polars, the tokenizers crate) never returned to
    the OS mid-run (see optimize_search_methodology.md / the plan's root-cause
    section: confirmed not fixable with gc.collect(), only process teardown
    actually reclaims it). Multiple waves run concurrently, gated by
    memory_budget_ledger so launch decisions can't overcommit from a
    just-spawned worker's not-yet-ramped-up footprint.
    """
    pending_jobs = list(candidate_jobs)
    pending_lock = threading.Lock()
    results_lock = threading.Lock()
    batch_results: list[OptimizeCandidateResult] = []

    def _pop_chunk() -> list[OptimizeCandidateJob]:
        with pending_lock:
            chunk = pending_jobs[:wave_chunk_size]
            del pending_jobs[:wave_chunk_size]
            return chunk

    def _run_wave(chunk: list[OptimizeCandidateJob]) -> None:
        per_candidate_estimate = (
            concurrency_controller.per_candidate_memory_estimate_bytes()
        )
        wave_worker_count = max(1, min(len(chunk), concurrency_controller.hard_cap))
        bytes_needed = wave_worker_count * per_candidate_estimate
        while not memory_budget_ledger.try_reserve(bytes_needed):
            if wave_worker_count > 1:
                wave_worker_count -= 1
                bytes_needed = wave_worker_count * per_candidate_estimate
            else:
                time.sleep(1.0)
                remaining = memory_budget_ledger.remaining_budget_bytes()
                wave_worker_count = max(
                    1,
                    compute_target_concurrency(
                        available_memory_bytes=remaining,
                        per_candidate_memory_bytes=per_candidate_estimate,
                        headroom_fraction=0.0,
                        hard_cap=min(len(chunk), concurrency_controller.hard_cap),
                    ),
                )
                bytes_needed = wave_worker_count * per_candidate_estimate

        executor = executor_factory(wave_worker_count)
        hung = False

        def _mark_hung() -> None:
            nonlocal hung
            hung = True

        try:
            future_map = submit_optimize_candidate_jobs(
                executor=executor,
                candidate_jobs=chunk,
                context=context,
                run_candidate=run_candidate,
            )
            # Own pool per wave already gives this the "tear down and
            # recreate on hang" behaviour the fixed-pool path needs
            # RecreatableExecutorPool for: a hang here just means this
            # wave's shutdown below can't wait=True, not that anything
            # needs recreating -- the next wave gets a brand new pool
            # regardless.
            chunk_results = drain_candidate_futures(
                future_map=future_map,
                context=context,
                tracker=tracker,
                run_candidate=run_candidate,
                include_in_pilot=include_in_pilot,
                expansion_round=expansion_round,
                phase_override=phase_override,
                timeout_seconds=context.candidate_timeout_seconds,
                failure_log_path=context.candidate_failure_log_path,
                on_hang=_mark_hung,
                on_worker_rss_sample=concurrency_controller.record_worker_rss_sample,
                record_lock=results_lock,
            )
            with results_lock:
                batch_results.extend(chunk_results)
        finally:
            # wait=True reclaims every byte a well-behaved wave's workers
            # held (see this function's docstring); wait=False only on a
            # genuine hang, since concurrent.futures has no supported way
            # to kill that worker and waiting would block here forever.
            executor.shutdown(wait=not hung, cancel_futures=True)
            memory_budget_ledger.release(bytes_needed)

    def _wave_thread_loop() -> None:
        while True:
            chunk = _pop_chunk()
            if not chunk:
                return
            _run_wave(chunk)

    thread_count = max(1, min(max_parallel_waves, len(candidate_jobs)))
    threads = [
        threading.Thread(target=_wave_thread_loop, name=f"optimize-wave-{i}")
        for i in range(thread_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    return batch_results


def _resolve_executor_handle(
    executor: futures.Executor | RecreatableExecutorPool,
) -> tuple[futures.Executor, Callable[[], None] | None]:
    """A RecreatableExecutorPool resolves to its live executor plus its
    recreate() as the on-hang callback; a plain executor (every existing
    test call site, and any future one that doesn't need hang recovery)
    passes through unchanged with no recreate capability.
    """
    if isinstance(executor, RecreatableExecutorPool):
        return executor.current, executor.recreate
    return executor, None


def run_candidate_batch(
    *,
    executor: futures.Executor | RecreatableExecutorPool | None,
    candidate_jobs: list[OptimizeCandidateJob],
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    run_candidate: Callable[..., OptimizeCandidateResult],
    concurrency_controller: AdaptiveConcurrencyController,
    memory_budget_ledger: MemoryBudgetLedger,
    wave_chunk_size: int,
    max_parallel_waves: int,
    include_in_pilot: bool,
    expansion_round: int | None = None,
    phase_override: str | None = None,
) -> list[OptimizeCandidateResult]:
    """Dispatcher used by both phase functions in place of calling
    submit_optimize_candidate_jobs + drain_candidate_futures directly.
    Disabled (--disable-adaptive-concurrency) -> exactly today's fixed-pool
    pair, using the shared executor -- now retry/timeout aware, and, when
    given a RecreatableExecutorPool, resolved fresh here on every call so a
    prior batch's timeout-triggered recreate() is picked up automatically.
    Enabled -> wave scheduling, which owns its own short-lived executors and
    never touches the shared one (executor may be None in that case).
    """
    if not concurrency_controller.enabled:
        assert executor is not None, (  # nosec B101 - narrows Optional for mypy
            "shared executor required when adaptive concurrency is disabled"
        )
        raw_executor, on_hang = _resolve_executor_handle(executor)
        future_map = submit_optimize_candidate_jobs(
            executor=raw_executor,
            candidate_jobs=candidate_jobs,
            context=context,
            run_candidate=run_candidate,
        )
        return drain_candidate_futures(
            future_map=future_map,
            context=context,
            tracker=tracker,
            run_candidate=run_candidate,
            include_in_pilot=include_in_pilot,
            expansion_round=expansion_round,
            phase_override=phase_override,
            timeout_seconds=context.candidate_timeout_seconds,
            failure_log_path=context.candidate_failure_log_path,
            on_hang=on_hang,
        )
    return run_candidate_jobs_in_waves(
        candidate_jobs=candidate_jobs,
        context=context,
        tracker=tracker,
        run_candidate=run_candidate,
        concurrency_controller=concurrency_controller,
        memory_budget_ledger=memory_budget_ledger,
        wave_chunk_size=wave_chunk_size,
        max_parallel_waves=max_parallel_waves,
        include_in_pilot=include_in_pilot,
        expansion_round=expansion_round,
        phase_override=phase_override,
    )


def _update_pilot_stop_state(
    *,
    state: PilotComboState,
    result: OptimizeCandidateResult,
    fertility_target: float | None,
    vocab_label: str,
) -> None:
    fertility = float(result.train_metrics["fertility"])
    last_fertility = state["last_fertility"]
    state["last_fertility"] = fertility
    if fertility_target is None or state["stopped"]:
        return

    extra_points = state["stop_extra_points"]
    if extra_points is not None:
        state["stop_extra_points"] = extra_points + 1
        state["stopped"] = extra_points + 1 >= PILOT_STOP_EXTRA_POINTS
        return

    if fertility < fertility_target:
        reason = "passed_target"
    elif (
        last_fertility is not None
        and abs(fertility - last_fertility) < PILOT_PLATEAU_FERTILITY_EPSILON
    ):
        reason = "plateau"
    else:
        return
    state["stop_extra_points"] = 0
    state["stopped"] = PILOT_STOP_EXTRA_POINTS <= 0
    print(
        f"[optimize] pilot_stop seed={result.seed} "
        f"min_freq={result.min_frequency} vocab={vocab_label} reason={reason} "
        f"train_fertility={fertility:.6f} extra_points={PILOT_STOP_EXTRA_POINTS}"
    )


def update_pilot_combo_cutoffs(
    *,
    current_vocab_results: list[OptimizeCandidateResult],
    pilot_combo_state: dict[tuple[int, int], PilotComboState],
    elbow_min_points: int,
    elbow_min_improvement: float,
    effective_elbow_extra_steps: int,
    vocab_label: str,
    fertility_target: float | None = None,
) -> None:
    """Track each (seed, min_frequency) trajectory: where it would elbow, and
    whether it has stopped.

    The elbow "cutoff" is diagnostic only: it is logged, and nothing reads it
    to decide which vocab points get evaluated, since a line still above its
    target can show slowing improvement.

    "stopped" is enforced: collect_candidate_jobs_for_pilot_vocab skips a
    stopped line. Train-split fertility falls as vocab grows, so it fires once
    a line is below `fertility_target`, or once its fertility has stopped
    moving, which is its min_frequency plateau. PILOT_STOP_EXTRA_POINTS more
    points run after the trigger. `fertility_target=None` disables the stop.
    """
    for result in current_vocab_results:
        combo_key = (result.seed, result.min_frequency)
        state = pilot_combo_state[combo_key]
        _update_pilot_stop_state(
            state=state,
            result=result,
            fertility_target=fertility_target,
            vocab_label=vocab_label,
        )
        points = int(state["points"]) + 1
        prior_distance_obj = state["last_distance"]
        prior_distance = (
            None if prior_distance_obj is None else float(prior_distance_obj)
        )
        current_distance = float(result.validation_metrics["fertility_distance"])
        post_elbow_steps = int(state["post_elbow_steps"])
        cutoff = bool(state["cutoff"])

        if points >= elbow_min_points and prior_distance is not None:
            improvement = prior_distance - current_distance
            if improvement < elbow_min_improvement:
                post_elbow_steps += 1
            else:
                post_elbow_steps = 0

            if not cutoff and post_elbow_steps >= effective_elbow_extra_steps:
                cutoff = True
                print(
                    f"[optimize] elbow_signal (diagnostic, not enforced) "
                    f"seed={result.seed} min_freq={result.min_frequency} "
                    f"vocab={vocab_label} improvement={improvement:.6f} "
                    f"post_elbow_steps={post_elbow_steps}"
                )

        state["points"] = points
        state["last_distance"] = current_distance
        state["post_elbow_steps"] = post_elbow_steps
        state["cutoff"] = cutoff


def _log_non_monotonic_pilot_results(
    seed_results: list[OptimizeCandidateResult], seed: int, vocab_label: str
) -> None:
    has_pass = any(not row.rejected for row in seed_results)
    has_reject = any(row.rejected for row in seed_results)
    if has_pass and has_reject:
        pass_min_freqs = sorted(
            row.min_frequency for row in seed_results if not row.rejected
        )
        reject_min_freqs = sorted(
            row.min_frequency for row in seed_results if row.rejected
        )
        print(
            f"[optimize] non_monotonic seed={seed} vocab={vocab_label} "
            f"pass_min_freqs={pass_min_freqs} reject_min_freqs={reject_min_freqs}"
        )


def _initialize_pilot_combo_state(
    pilot_seeds: list[int], min_frequencies: list[int]
) -> dict[tuple[int, int], PilotComboState]:
    return {
        (seed, min_frequency): {
            "points": 0,
            "last_distance": None,
            "post_elbow_steps": 0,
            "cutoff": False,
            "last_fertility": None,
            "stop_extra_points": None,
            "stopped": False,
        }
        for seed in pilot_seeds
        for min_frequency in min_frequencies
    }


def _handle_pilot_vocab_no_candidates(
    *,
    skipped_existing_candidates: int,
    vocab_label: str,
) -> None:
    if skipped_existing_candidates:
        print(
            f"[optimize] pilot reuse-only vocab={vocab_label}; "
            "all candidate jobs skipped because fresh model outputs already exist"
        )


def _log_pilot_vocab_non_monotonic_results(
    *,
    pilot_seeds: list[int],
    current_vocab_results: list[OptimizeCandidateResult],
    vocab_requested: int,
    vocab_label: str,
) -> None:
    for seed in pilot_seeds:
        seed_results = [
            row
            for row in current_vocab_results
            if row.seed == seed and row.vocab_requested == vocab_requested
        ]
        if not seed_results:
            continue
        _log_non_monotonic_pilot_results(seed_results, seed, vocab_label)


def _register_plateau_sources(
    *,
    results: list[OptimizeCandidateResult],
    vocab_requested: int,
    plateau_sources: dict[tuple[int, int], OptimizeCandidateResult],
) -> None:
    # The auto point is left out: its size is the trainer's estimate, not a
    # position in the ascending schedule a later point can be compared with.
    if vocab_requested == -1:
        return
    for result in results:
        key = (result.seed, result.min_frequency)
        if key not in plateau_sources and (
            candidate_reached_full_vocab(result) is False
        ):
            plateau_sources[key] = result


def run_pilot_vocab_with_carry(
    *,
    candidate_jobs: list[OptimizeCandidateJob],
    vocab_requested: int,
    min_frequencies: list[int],
    reused_vocab_results: list[OptimizeCandidateResult],
    plateau_sources: dict[tuple[int, int], OptimizeCandidateResult],
    run_batch: Callable[[list[OptimizeCandidateJob]], list[OptimizeCandidateResult]],
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
) -> list[OptimizeCandidateResult]:
    """One pilot vocab point, training only the cells no other cell fixes
    (candidate_reached_full_vocab). In order:

    1. A line already on its `min_frequency` plateau is carried from the
       result that found the plateau.
    2. Per seed, the highest `min_frequency` not yet on a plateau is trained
       as the probe (or read from history on a resumed run).
    3. A probe that reached its full vocabulary is carried to every lower
       `min_frequency`. One that fell short has found its own plateau, and
       the lower ones are trained, in one batch, since any of them may have
       fallen short too.

    At most two batches run per vocab point. A result with no trained size
    recorded decides nothing, and every cell it could have fixed is trained.
    """
    results: list[OptimizeCandidateResult] = []

    def _carry(source: OptimizeCandidateResult, job: OptimizeCandidateJob) -> None:
        carried = carry_candidate_result(
            source=source,
            vocab_requested=vocab_requested,
            min_frequency=int(job["min_frequency"]),
        )
        record_carried_candidate_result(
            result=carried, context=context, tracker=tracker, include_in_pilot=True
        )
        results.append(carried)

    _register_plateau_sources(
        results=reused_vocab_results,
        vocab_requested=vocab_requested,
        plateau_sources=plateau_sources,
    )

    jobs_by_seed: dict[int, list[OptimizeCandidateJob]] = {}
    for job in candidate_jobs:
        source = plateau_sources.get((int(job["seed"]), int(job["min_frequency"])))
        if (
            vocab_requested != -1
            and source is not None
            and source.trained_vocab_size is not None
            and vocab_requested >= source.trained_vocab_size
        ):
            _carry(source, job)
        else:
            jobs_by_seed.setdefault(int(job["seed"]), []).append(job)

    reused = {
        (result.seed, result.min_frequency): result for result in reused_vocab_results
    }
    probe_jobs: list[OptimizeCandidateJob] = []
    probe_min_frequency: dict[int, int] = {}
    probe_results: dict[int, OptimizeCandidateResult] = {}
    rest_jobs: list[OptimizeCandidateJob] = []
    for seed, jobs in jobs_by_seed.items():
        unbound = [
            min_frequency
            for min_frequency in min_frequencies
            if vocab_requested == -1 or (seed, min_frequency) not in plateau_sources
        ]
        if not unbound:
            rest_jobs.extend(jobs)
            continue
        probe = max(unbound)
        probe_job = next(
            (job for job in jobs if int(job["min_frequency"]) == probe), None
        )
        if (seed, probe) in reused:
            probe_results[seed] = reused[(seed, probe)]
        elif probe_job is not None:
            probe_jobs.append(probe_job)
        else:
            rest_jobs.extend(jobs)
            continue
        probe_min_frequency[seed] = probe

    if probe_jobs:
        for result in run_batch(probe_jobs):
            results.append(result)
            probe_results[result.seed] = result

    for seed, probe in probe_min_frequency.items():
        probe_result = probe_results.get(seed)
        reached_full = (
            None if probe_result is None else candidate_reached_full_vocab(probe_result)
        )
        for job in jobs_by_seed[seed]:
            min_frequency = int(job["min_frequency"])
            if min_frequency == probe and job in probe_jobs:
                continue
            if reached_full and probe_result is not None and min_frequency < probe:
                _carry(probe_result, job)
            else:
                rest_jobs.append(job)

    if rest_jobs:
        results.extend(run_batch(rest_jobs))

    _register_plateau_sources(
        results=[
            result
            for result in results
            if getattr(result, "carried_from", None) is None
        ],
        vocab_requested=vocab_requested,
        plateau_sources=plateau_sources,
    )
    return results


def run_optimize_pilot_phase(
    *,
    executor: futures.Executor | RecreatableExecutorPool | None,
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    vocab_schedule: list[int],
    pilot_seeds: list[int],
    min_frequencies: list[int],
    seed_splits: dict[int, OptimizeSeedSplit],
    effective_elbow_extra_steps: int,
    run_candidate: Callable[..., OptimizeCandidateResult],
    concurrency_controller: AdaptiveConcurrencyController,
    memory_budget_ledger: MemoryBudgetLedger,
    wave_chunk_size: int,
    max_parallel_waves: int,
) -> None:
    pilot_combo_state = _initialize_pilot_combo_state(pilot_seeds, min_frequencies)
    # (seed, min_frequency) -> the first result of that line whose trained
    # vocabulary fell short of its request: the line's plateau, which every
    # larger vocab point on it repeats (run_pilot_vocab_with_carry).
    plateau_sources: dict[tuple[int, int], OptimizeCandidateResult] = {}
    # --pilot-early-stop false runs the full grid.
    fertility_target = (
        float(context.candidate_policy.fertility_target)
        if getattr(context.args, "pilot_early_stop", True)
        else None
    )

    for vocab_requested in vocab_schedule:
        candidate_jobs, skipped_existing_candidates = (
            collect_candidate_jobs_for_pilot_vocab(
                context=context,
                vocab_requested=vocab_requested,
                pilot_seeds=pilot_seeds,
                min_frequencies=min_frequencies,
                seed_splits=seed_splits,
                pilot_combo_state=pilot_combo_state,
            )
        )
        tracker.skipped_existing_candidates_total += skipped_existing_candidates

        vocab_label = "auto" if vocab_requested == -1 else str(vocab_requested)
        if skipped_existing_candidates:
            print(
                f"[optimize] reuse skip_existing_candidates={skipped_existing_candidates} "
                f"phase=pilot vocab={vocab_label}"
            )
        stopped_lines = sum(
            1 for state in pilot_combo_state.values() if state["stopped"]
        )
        if stopped_lines and vocab_requested != -1:
            print(
                f"[optimize] pilot_stop skipped_lines={stopped_lines} vocab={vocab_label}"
            )

        # A point reused from history never reaches the batch below, so its
        # line would otherwise see a gap and never stop on a resumed run.
        reused_vocab_results = [
            result
            for result in tracker.pilot_results
            if result.vocab_requested == vocab_requested
            and (result.seed, result.min_frequency) in pilot_combo_state
        ]

        if not candidate_jobs:
            _handle_pilot_vocab_no_candidates(
                skipped_existing_candidates=skipped_existing_candidates,
                vocab_label=vocab_label,
            )
            update_pilot_combo_cutoffs(
                current_vocab_results=reused_vocab_results,
                pilot_combo_state=pilot_combo_state,
                elbow_min_points=context.args.elbow_min_points,
                elbow_min_improvement=context.args.elbow_min_improvement,
                effective_elbow_extra_steps=effective_elbow_extra_steps,
                vocab_label=vocab_label,
                fertility_target=fertility_target,
            )
            continue

        log_optimize_progress(
            phase=f"pilot:queue:vocab={vocab_label}",
            completed_jobs=tracker.completed_jobs,
            submitted_jobs=tracker.submitted_jobs,
            pending_batch_jobs=len(candidate_jobs),
            upper_bound_jobs=context.total_candidate_jobs,
            max_workers=context.max_workers,
            observed_job_elapsed_seconds=tracker.observed_job_elapsed_seconds,
        )

        def _run_pilot_batch(
            jobs: list[OptimizeCandidateJob],
            vocab_label: str = vocab_label,
        ) -> list[OptimizeCandidateResult]:
            tracker.submitted_jobs += len(jobs)
            print(
                f"[optimize] pilot evaluating vocab={vocab_label} across jobs={len(jobs)}"
            )
            return run_candidate_batch(
                executor=executor,
                candidate_jobs=jobs,
                context=context,
                tracker=tracker,
                run_candidate=run_candidate,
                concurrency_controller=concurrency_controller,
                memory_budget_ledger=memory_budget_ledger,
                wave_chunk_size=wave_chunk_size,
                max_parallel_waves=max_parallel_waves,
                include_in_pilot=True,
            )

        if not getattr(context.args, "skip_equivalent_candidates", True):
            current_vocab_results = _run_pilot_batch(candidate_jobs)
        else:
            current_vocab_results = run_pilot_vocab_with_carry(
                candidate_jobs=candidate_jobs,
                vocab_requested=vocab_requested,
                min_frequencies=min_frequencies,
                reused_vocab_results=reused_vocab_results,
                plateau_sources=plateau_sources,
                run_batch=_run_pilot_batch,
                context=context,
                tracker=tracker,
            )

        _log_pilot_vocab_non_monotonic_results(
            pilot_seeds=pilot_seeds,
            current_vocab_results=current_vocab_results,
            vocab_requested=vocab_requested,
            vocab_label=vocab_label,
        )

        log_optimize_progress(
            phase=f"pilot:vocab={vocab_label}",
            completed_jobs=tracker.completed_jobs,
            submitted_jobs=tracker.submitted_jobs,
            pending_batch_jobs=0,
            upper_bound_jobs=context.total_candidate_jobs,
            max_workers=context.max_workers,
            observed_job_elapsed_seconds=tracker.observed_job_elapsed_seconds,
        )

        update_pilot_combo_cutoffs(
            current_vocab_results=[*reused_vocab_results, *current_vocab_results],
            pilot_combo_state=pilot_combo_state,
            elbow_min_points=context.args.elbow_min_points,
            elbow_min_improvement=context.args.elbow_min_improvement,
            effective_elbow_extra_steps=effective_elbow_extra_steps,
            vocab_label=vocab_label,
            fertility_target=fertility_target,
        )


def initialize_frontier_progress(
    *,
    frontier: set[tuple[int, int]],
    pilot_results: list[OptimizeCandidateResult],
    historical_expansion_results: list[OptimizeCandidateResult] | None = None,
) -> dict[tuple[int, int], ComboProgressState]:
    combo_progress: dict[tuple[int, int], ComboProgressState] = {
        combo: {"runs": 0, "passes": 0, "fertilities": []} for combo in frontier
    }
    # historical_expansion_results covers combos that already had some (or
    # all) of their expansion seeds completed in a prior, reused run --
    # without folding those in, a partially-reused expansion phase would
    # undercount a combo's runs/passes and prune_combos would decide on
    # incomplete information.
    for result in [*pilot_results, *(historical_expansion_results or [])]:
        combo = (result.min_frequency, result.vocab_requested)
        if combo not in combo_progress:
            continue
        _update_combo_progress_state(combo_progress[combo], result)
    return combo_progress


def _update_combo_progress_state(
    state: ComboProgressState,
    result: OptimizeCandidateResult,
) -> list[float]:
    state["runs"] = int(state["runs"]) + 1
    if not result.rejected:
        state["passes"] = int(state["passes"]) + 1
    fertilities = list(state["fertilities"])
    fertilities.append(float(result.validation_metrics["fertility"]))
    state["fertilities"] = fertilities
    return fertilities


def prune_combos(
    *,
    combo_progress: dict[tuple[int, int], ComboProgressState],
    total_seed_budget: int,
    policy: OptimizeCandidatePolicy,
    combo_prune_reason: Callable[..., str | None],  # noqa: F811
) -> set[tuple[int, int]]:
    pruned_combos: set[tuple[int, int]] = set()
    for combo, state in combo_progress.items():
        reason = combo_prune_reason(
            runs=int(state["runs"]),
            passes=int(state["passes"]),
            fertilities=list(state["fertilities"]),
            total_seeds=total_seed_budget,
            policy=policy,
        )
        if reason is not None:
            pruned_combos.add(combo)
    return pruned_combos


def build_pilot_cell_classes(
    pilot_results: list[OptimizeCandidateResult],
) -> dict[tuple[int, int], str]:
    """(min_frequency, vocab_requested) -> the tag of the cell that stands for
    every cell the pilot found trains the same vocabulary
    (candidate_reached_full_vocab). At one vocab, every cell that reached its
    full vocabulary shares the class of the highest such `min_frequency`. On
    one `min_frequency`, every cell that fell short shares the class of the
    smallest such vocab, the auto point included. That covers the cells the
    pilot trained anyway, in its second batch, as well as the ones it carried.
    Pilot seeds that disagree about a cell leave it in a class of its own."""

    def _cell(result: OptimizeCandidateResult) -> tuple[int, int]:
        return (result.min_frequency, result.vocab_requested)

    by_seed: dict[int, list[OptimizeCandidateResult]] = {}
    for result in pilot_results:
        by_seed.setdefault(result.seed, []).append(result)

    tags: dict[tuple[int, int], set[str]] = {}
    for seed_results in by_seed.values():
        seed_tags = {
            _cell(result): getattr(result, "carried_from", None)
            or candidate_cell_tag(
                vocab_requested=result.vocab_requested,
                min_frequency=result.min_frequency,
            )
            for result in seed_results
        }
        reached_full = [
            result
            for result in seed_results
            if candidate_reached_full_vocab(result) is True
        ]
        for vocab in {result.vocab_requested for result in reached_full}:
            at_vocab = [r for r in reached_full if r.vocab_requested == vocab]
            top = max(at_vocab, key=lambda r: r.min_frequency)
            for result in at_vocab:
                seed_tags[_cell(result)] = seed_tags[_cell(top)]
        fell_short = [
            result
            for result in seed_results
            if candidate_reached_full_vocab(result) is False
        ]
        for min_frequency in {result.min_frequency for result in fell_short}:
            on_line = [r for r in fell_short if r.min_frequency == min_frequency]
            numeric = [r for r in on_line if r.vocab_requested != -1]
            if not numeric:
                continue
            low = min(numeric, key=lambda r: r.vocab_requested)
            for result in on_line:
                seed_tags[_cell(result)] = seed_tags[_cell(low)]
        for cell, tag in seed_tags.items():
            tags.setdefault(cell, set()).add(tag)
    return {
        cell: next(iter(cell_tags))
        if len(cell_tags) == 1
        else candidate_cell_tag(vocab_requested=cell[1], min_frequency=cell[0])
        for cell, cell_tags in tags.items()
    }


def run_expansion_seed_with_carry(
    *,
    candidate_jobs: list[OptimizeCandidateJob],
    cell_classes: dict[tuple[int, int], str],
    run_batch: Callable[[list[OptimizeCandidateJob]], list[OptimizeCandidateResult]],
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    expansion_round: int,
) -> list[OptimizeCandidateResult]:
    """One expansion seed, training one representative of each class the
    pilot found (build_pilot_cell_classes) and carrying it to the class's
    other frontier cells. The pilot's finding is checked again on this seed's
    own split before anything is carried: cells sharing a vocab are carried
    only from a highest-`min_frequency` representative that reached its full
    vocabulary, and cells sharing a `min_frequency` only from a
    smallest-vocab representative that fell short of it. Where the check
    fails the rest of the class is trained, in a second batch.
    """
    classes: dict[str, list[OptimizeCandidateJob]] = {}
    for job in candidate_jobs:
        cell = (int(job["min_frequency"]), int(job["vocab_requested"]))
        own_tag = candidate_cell_tag(vocab_requested=cell[1], min_frequency=cell[0])
        classes.setdefault(cell_classes.get(cell, own_tag), []).append(job)

    first_batch: list[OptimizeCandidateJob] = []
    # representative job, the rest of its class, the full-vocab outcome the
    # representative must show for the class to hold on this seed.
    pending: list[tuple[OptimizeCandidateJob, list[OptimizeCandidateJob], bool]] = []
    for jobs in classes.values():
        vocabs = {int(job["vocab_requested"]) for job in jobs}
        min_frequencies = {int(job["min_frequency"]) for job in jobs}
        if len(jobs) > 1 and len(vocabs) == 1:
            representative = max(jobs, key=lambda job: int(job["min_frequency"]))
            expect_full = True
        elif len(jobs) > 1 and len(min_frequencies) == 1 and vocabs != {-1}:
            # The auto point is carried to, never from: its size is known
            # only once it has trained.
            representative = min(
                (job for job in jobs if int(job["vocab_requested"]) != -1),
                key=lambda job: int(job["vocab_requested"]),
            )
            expect_full = False
        else:
            first_batch.extend(jobs)
            continue
        first_batch.append(representative)
        pending.append(
            (
                representative,
                [job for job in jobs if job is not representative],
                expect_full,
            )
        )

    results = list(run_batch(first_batch)) if first_batch else []
    by_cell = {
        (result.min_frequency, result.vocab_requested): result for result in results
    }
    second_batch: list[OptimizeCandidateJob] = []
    for representative, rest, expect_full in pending:
        source = by_cell.get(
            (
                int(representative["min_frequency"]),
                int(representative["vocab_requested"]),
            )
        )
        if source is None or candidate_reached_full_vocab(source) is not expect_full:
            second_batch.extend(rest)
            continue
        for job in rest:
            carried = carry_candidate_result(
                source=source,
                vocab_requested=int(job["vocab_requested"]),
                min_frequency=int(job["min_frequency"]),
            )
            record_carried_candidate_result(
                result=carried,
                context=context,
                tracker=tracker,
                include_in_pilot=False,
                expansion_round=expansion_round,
            )
            results.append(carried)

    if second_batch:
        results.extend(run_batch(second_batch))
    return results


def run_optimize_expansion_phase(
    *,
    executor: futures.Executor | RecreatableExecutorPool | None,
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    expansion_seeds: list[int],
    frontier: set[tuple[int, int]],
    seed_splits: dict[int, OptimizeSeedSplit],
    pruned_combos: set[tuple[int, int]],
    combo_progress: dict[tuple[int, int], ComboProgressState],
    total_seed_budget: int,
    combo_prune_reason: Callable[..., str | None],  # noqa: F811
    run_candidate: Callable[..., OptimizeCandidateResult],
    concurrency_controller: AdaptiveConcurrencyController,
    memory_budget_ledger: MemoryBudgetLedger,
    wave_chunk_size: int,
    max_parallel_waves: int,
    record_result: Callable[..., None] = default_record_result,
) -> None:
    cell_classes = build_pilot_cell_classes(tracker.pilot_results)
    for round_index, seed in enumerate(expansion_seeds, start=1):
        active_combos = [combo for combo in frontier if combo not in pruned_combos]
        if not active_combos:
            print("[optimize] expansion ended early; all frontier combos pruned")
            break

        split = seed_splits[seed]
        candidate_jobs, skipped_existing_candidates = (
            collect_candidate_jobs_for_expansion_seed(
                context=context,
                seed=seed,
                active_combos=active_combos,
                split=split,
            )
        )
        tracker.skipped_existing_candidates_total += skipped_existing_candidates

        print(
            f"[optimize] expansion seed={seed} evaluating active_combos={len(candidate_jobs)}"
        )
        if skipped_existing_candidates:
            print(
                f"[optimize] reuse skip_existing_candidates={skipped_existing_candidates} "
                f"phase=expansion seed={seed}"
            )
        if not candidate_jobs:
            log_optimize_progress(
                phase=f"expansion:seed={seed}",
                completed_jobs=tracker.completed_jobs,
                submitted_jobs=tracker.submitted_jobs,
                pending_batch_jobs=0,
                upper_bound_jobs=context.total_candidate_jobs,
                max_workers=context.max_workers,
                observed_job_elapsed_seconds=tracker.observed_job_elapsed_seconds,
            )
            continue

        log_optimize_progress(
            phase=f"expansion:queue:seed={seed}",
            completed_jobs=tracker.completed_jobs,
            submitted_jobs=tracker.submitted_jobs,
            pending_batch_jobs=len(candidate_jobs),
            upper_bound_jobs=context.total_candidate_jobs,
            max_workers=context.max_workers,
            observed_job_elapsed_seconds=tracker.observed_job_elapsed_seconds,
        )

        def _run_expansion_batch(
            jobs: list[OptimizeCandidateJob],
            round_index: int = round_index,
        ) -> list[OptimizeCandidateResult]:
            tracker.submitted_jobs += len(jobs)
            return run_candidate_batch(
                executor=executor,
                candidate_jobs=jobs,
                context=context,
                tracker=tracker,
                run_candidate=run_candidate,
                concurrency_controller=concurrency_controller,
                memory_budget_ledger=memory_budget_ledger,
                wave_chunk_size=wave_chunk_size,
                max_parallel_waves=max_parallel_waves,
                include_in_pilot=False,
                expansion_round=round_index,
            )

        if not getattr(context.args, "skip_equivalent_candidates", True):
            batch_results = _run_expansion_batch(candidate_jobs)
        else:
            batch_results = run_expansion_seed_with_carry(
                candidate_jobs=candidate_jobs,
                cell_classes=cell_classes,
                run_batch=_run_expansion_batch,
                context=context,
                tracker=tracker,
                expansion_round=round_index,
            )

        for result in batch_results:
            combo = (result.min_frequency, result.vocab_requested)
            if combo in combo_progress:
                state = combo_progress[combo]
                fertilities = _update_combo_progress_state(state, result)
                record_result(combo=combo, result=result, expansion_round=round_index)

                reason = combo_prune_reason(
                    runs=int(state["runs"]),
                    passes=int(state["passes"]),
                    fertilities=fertilities,
                    total_seeds=total_seed_budget,
                    policy=context.candidate_policy,
                )
                if reason is not None and combo not in pruned_combos:
                    pruned_combos.add(combo)
                    vocab_label = "auto" if combo[1] == -1 else str(combo[1])
                    print(
                        f"[optimize] prune min_freq={combo[0]} vocab={vocab_label} reason={reason}"
                    )

        log_optimize_progress(
            phase=f"expansion:seed={seed}",
            completed_jobs=tracker.completed_jobs,
            submitted_jobs=tracker.submitted_jobs,
            pending_batch_jobs=0,
            upper_bound_jobs=context.total_candidate_jobs,
            max_workers=context.max_workers,
            observed_job_elapsed_seconds=tracker.observed_job_elapsed_seconds,
        )


def winning_min_frequency_plateau_size(
    *, run_log_df: pl.DataFrame, min_frequency: int
) -> int | None:
    """The trained vocabulary size `min_frequency` plateaus at, read from the
    run log's rows whose trained size fell short of the request, or None when
    no row shows one (or the log predates the trained size being recorded)."""
    required = {"min_frequency", "vocab_size_trained", "vocab_size_resolved"}
    if not required.issubset(run_log_df.columns):
        return None
    fell_short = run_log_df.filter(
        (pl.col("min_frequency") == min_frequency)
        & pl.col("vocab_size_trained").is_not_null()
        & (pl.col("vocab_size_trained") < pl.col("vocab_size_resolved"))
    )
    if fell_short.height == 0:
        return None
    return int(fell_short.get_column("vocab_size_trained").min())  # type: ignore[arg-type]


_TRAINED_SIZE_COLUMNS = {"vocab_size_trained", "vocab_size_resolved"}


def reached_full_vocab_rows(run_log_df: pl.DataFrame) -> pl.DataFrame | None:
    """The run-log rows whose trained vocabulary reached the size requested,
    or None when the log does not record the trained size."""
    if not _TRAINED_SIZE_COLUMNS.issubset(run_log_df.columns):
        return None
    return run_log_df.filter(
        pl.col("vocab_size_trained").is_not_null()
        & (pl.col("vocab_size_trained") >= pl.col("vocab_size_resolved"))
    )


def cell_fell_short_of_its_vocab(
    *, run_log_df: pl.DataFrame, min_frequency: int, vocab_requested: int
) -> bool:
    """Whether any of a cell's rows trained a vocabulary short of its request,
    which puts the cell on its `min_frequency`'s plateau."""
    if not _TRAINED_SIZE_COLUMNS.issubset(run_log_df.columns):
        return False
    return (
        run_log_df.filter(
            (pl.col("min_frequency") == min_frequency)
            & (pl.col("vocab_size_requested") == vocab_requested)
            & pl.col("vocab_size_trained").is_not_null()
            & (pl.col("vocab_size_trained") < pl.col("vocab_size_resolved"))
        ).height
        > 0
    )


class RefiningPhaseReport(TypedDict):
    bracket_low: int
    bracket_high: int
    winning_vocab: int
    min_frequency: int
    crossing: float
    slope: float
    se_diff: float
    refine_step: int
    refined_points: list[int]
    fit_matches: list[tuple[int, float, bool]]


def run_optimize_refining_phase(
    *,
    executor: futures.Executor | RecreatableExecutorPool | None,
    context: OptimizeRunContext,
    tracker: OptimizeExecutionTracker,
    seeds: list[int],
    frontier: set[tuple[int, int]],
    seed_splits: dict[int, OptimizeSeedSplit],
    run_log_dir: Path,
    run_log_path: Path,
    eligibility_pass_rate: float,
    vocab_size_distance_tolerance: float,
    run_candidate: Callable[..., OptimizeCandidateResult],
    concurrency_controller: AdaptiveConcurrencyController,
    memory_budget_ledger: MemoryBudgetLedger,
    wave_chunk_size: int,
    max_parallel_waves: int,
    refine_step_override: int | None = None,
    refine_step_minimum: int = 100,
) -> RefiningPhaseReport | None:
    """Third sweep phase: refine inside the winning bracket at a
    noise-derived vocab step, holding min_frequency at its winning value
    (optimize_search_methodology.md Part 8).

    Reuses the same merge_run_logs -> scoped_history_for_trainer ->
    select_best_pair path run_optimize_target's own final selection takes,
    against the shard log pilot/expansion have already written, to find an
    interim winner to refine around -- rather than re-deriving selection
    from tracker state, which would have to duplicate that path's
    reuse/reclassification handling to agree with it.

    Returns None (skipping refinement, with a printed reason) when there's
    no eligible interim winner yet, no interior bracket to search inside
    (select_vocab_frontier's docstring on select_refining_bracket), or the
    winning cell has fewer than two seeds to estimate noise from.
    """
    run_log_df = merge_run_logs(
        run_log_dir=run_log_dir, merged_run_log_path=run_log_path
    )
    scoped_history = scoped_history_for_trainer(
        run_log_df=run_log_df,
        scope=context.target_scope,
        systems_key=",".join(context.systems),
        trainer=context.args.tokenizer,
    )
    scoped_history = backfill_trained_vocab_sizes(
        run_log_df=scoped_history, trainer=context.args.tokenizer
    )
    # dict[str, Any], matching finalize_optimize_target's best_pair param
    # below: select_best_pair returns dict[str, object], and indexing an
    # object-valued dict to build the noise-derived step needs the same
    # widening that call site already relies on.
    interim_best_pair: dict[str, Any] | None = select_best_pair(
        run_log_df=scoped_history,
        eligibility_pass_rate=eligibility_pass_rate,
        vocab_size_distance_tolerance=vocab_size_distance_tolerance,
        policy=context.candidate_policy,
    )
    if interim_best_pair is None:
        print("[optimize] refining skipped; no eligible interim winner yet.")
        return None

    winning_min_frequency = int(interim_best_pair["min_frequency"])
    winning_vocab = int(interim_best_pair["vocab_size_requested"])

    # On a min_frequency plateau vocab changes nothing, so there is nothing to
    # refine around a winner that sits on one. The fertility target is
    # crossed where vocab still moves fertility, so refining centres on the
    # best cell among those that trained their full vocabulary instead.
    full_vocab_history = reached_full_vocab_rows(scoped_history)
    if full_vocab_history is not None and cell_fell_short_of_its_vocab(
        run_log_df=scoped_history,
        min_frequency=winning_min_frequency,
        vocab_requested=winning_vocab,
    ):
        full_vocab_best_pair: dict[str, Any] | None = select_best_pair(
            run_log_df=full_vocab_history,
            eligibility_pass_rate=eligibility_pass_rate,
            vocab_size_distance_tolerance=vocab_size_distance_tolerance,
            policy=context.candidate_policy,
        )
        if full_vocab_best_pair is None:
            print(
                f"[optimize] refining skipped; interim winner min_freq="
                f"{winning_min_frequency} vocab={winning_vocab} is on its "
                "min_frequency plateau and no eligible cell trained its full vocabulary."
            )
            return None
        centre_vocab = int(full_vocab_best_pair["vocab_size_requested"])
        # Every cell at one vocab that trained its full vocabulary is the same
        # tokenizer, so which of them scores best is noise. The lowest
        # min_frequency is the line that binds last, the one vocab still moves
        # on both sides of the centre.
        centre_min_frequency = min(
            (
                int(min_frequency)
                for min_frequency in full_vocab_history.filter(
                    pl.col("vocab_size_requested") == centre_vocab
                )
                .get_column("min_frequency")
                .unique()
                .to_list()
                if not cell_fell_short_of_its_vocab(
                    run_log_df=scoped_history,
                    min_frequency=int(min_frequency),
                    vocab_requested=centre_vocab,
                )
            ),
            default=int(full_vocab_best_pair["min_frequency"]),
        )
        print(
            f"[optimize] refining interim winner min_freq={winning_min_frequency} "
            f"vocab={winning_vocab} is on its min_frequency plateau; centring on "
            f"min_freq={centre_min_frequency} vocab={centre_vocab}, the lowest "
            "min_frequency at the best vocab that trained its full vocabulary"
        )
        winning_min_frequency = centre_min_frequency
        winning_vocab = centre_vocab

    bracket = select_refining_bracket(
        frontier=frontier,
        winning_min_frequency=winning_min_frequency,
        winning_vocab_requested=winning_vocab,
    )
    if bracket is None:
        print(
            f"[optimize] refining skipped; no interior bracket around "
            f"min_freq={winning_min_frequency} vocab={winning_vocab} to search inside."
        )
        return None
    bracket_low, _bracket_mid, bracket_high = bracket

    cell_summaries = {
        vocab: summarize_cell_train_fertility_distance(
            run_log_df=scoped_history,
            min_frequency=winning_min_frequency,
            vocab_requested=vocab,
        )
        for vocab in bracket
    }
    if any(summary is None for summary in cell_summaries.values()):
        print("[optimize] refining skipped; a bracket point has no run-log history.")
        return None

    fit_vocab_points = list(bracket)
    fit_fertilities = [cell_summaries[vocab].fertility for vocab in bracket]  # type: ignore[union-attr]
    if any(fertility is None for fertility in fit_fertilities):
        print("[optimize] refining skipped; a bracket point records no fertility.")
        return None
    fertility_target = float(context.candidate_policy.fertility_target)
    crossing = predict_fertility_crossing(
        vocab_points=fit_vocab_points,
        fertilities=fit_fertilities,  # type: ignore[arg-type]
        fertility_target=fertility_target,
    )
    if crossing is None:
        print(
            "[optimize] refining: fertility does not cross the target inside "
            f"the bracket; centring on the winning vocab {winning_vocab}."
        )
        crossing = float(winning_vocab)
    slope = fertility_slope_at(
        vocab_points=fit_vocab_points,
        fertilities=fit_fertilities,  # type: ignore[arg-type]
        vocab=crossing,
    )

    winner_summary = cell_summaries[winning_vocab]
    assert winner_summary is not None  # nosec B101 - narrowed by the any() check above
    if winner_summary.fertility_stdev is None:
        print(
            f"[optimize] refining skipped; winning cell min_freq="
            f"{winning_min_frequency} vocab={winning_vocab} has fewer than two "
            "seeds, no noise estimate to derive a step from."
        )
        return None

    se_diff = compute_standard_error_of_difference(
        sigma=winner_summary.fertility_stdev, seed_count=winner_summary.seed_count
    )
    refine_step = (
        refine_step_override
        if refine_step_override is not None
        else derive_refine_step_vocab(
            se_diff=se_diff, slope=slope, minimum_step=refine_step_minimum
        )
    )
    print(
        f"[optimize] refining bracket=[{bracket_low},{bracket_high}] "
        f"min_freq={winning_min_frequency} winning_vocab={winning_vocab} "
        f"crossing={round(crossing)} slope={slope:.6e} se_diff={se_diff:.6f} "
        f"step={refine_step}"
    )

    refine_vocab_points = build_crossing_vocab_points(
        crossing=crossing,
        bracket_low=bracket_low,
        bracket_high=bracket_high,
        step=refine_step,
    )
    interior_points = [
        vocab_requested
        for vocab_requested in refine_vocab_points
        if vocab_requested not in bracket
    ]
    # A refined point at or past the winning min_frequency's plateau trains
    # the plateau's vocabulary again (candidate_reached_full_vocab), so full
    # seeds there measure nothing the plateau cell has not already measured.
    plateau_vocab_size = winning_min_frequency_plateau_size(
        run_log_df=scoped_history, min_frequency=winning_min_frequency
    )
    if plateau_vocab_size is not None:
        past_plateau = [
            vocab_requested
            for vocab_requested in interior_points
            if vocab_requested >= plateau_vocab_size
        ]
        if past_plateau:
            print(
                f"[optimize] refining skips {len(past_plateau)} point(s) at or past "
                f"min_freq={winning_min_frequency}'s plateau of "
                f"{plateau_vocab_size} trained tokens"
            )
            interior_points = [
                vocab_requested
                for vocab_requested in interior_points
                if vocab_requested < plateau_vocab_size
            ]

    for vocab_requested in interior_points:
        candidate_jobs, skipped_existing_candidates = (
            collect_candidate_jobs_for_refining_vocab(
                context=context,
                vocab_requested=vocab_requested,
                min_frequency=winning_min_frequency,
                seeds=seeds,
                seed_splits=seed_splits,
            )
        )
        tracker.skipped_existing_candidates_total += skipped_existing_candidates
        if not candidate_jobs:
            continue

        tracker.submitted_jobs += len(candidate_jobs)
        print(
            f"[optimize] refining evaluating vocab={vocab_requested} "
            f"across jobs={len(candidate_jobs)}"
        )
        run_candidate_batch(
            executor=executor,
            candidate_jobs=candidate_jobs,
            context=context,
            tracker=tracker,
            run_candidate=run_candidate,
            concurrency_controller=concurrency_controller,
            memory_budget_ledger=memory_budget_ledger,
            wave_chunk_size=wave_chunk_size,
            max_parallel_waves=max_parallel_waves,
            include_in_pilot=False,
            phase_override="refining",
        )

    refined_history = merge_run_logs(
        run_log_dir=run_log_dir, merged_run_log_path=run_log_path
    )
    refined_scoped = scoped_history_for_trainer(
        run_log_df=refined_history,
        scope=context.target_scope,
        systems_key=",".join(context.systems),
        trainer=context.args.tokenizer,
    )
    refined_points_measured: list[tuple[int, float]] = []
    for vocab_requested in interior_points:
        summary = summarize_cell_train_fertility_distance(
            run_log_df=refined_scoped,
            min_frequency=winning_min_frequency,
            vocab_requested=vocab_requested,
        )
        if summary is not None:
            refined_points_measured.append((vocab_requested, summary.median))

    fit_matches = evaluate_refined_points_against_fit(
        fit_vocab_points=fit_vocab_points,
        fit_fertilities=fit_fertilities,  # type: ignore[arg-type]
        fertility_target=fertility_target,
        refined_points=refined_points_measured,
        noise_threshold=2 * se_diff,
    )
    matched_count = sum(1 for _, _, matched in fit_matches if matched)
    print(
        f"[optimize] refining fit_check matched={matched_count}/{len(fit_matches)} "
        f"points within {2 * se_diff:.6f} of the fitted curve"
    )
    for vocab_requested, residual, matched in fit_matches:
        print(
            f"[optimize] refining point vocab={vocab_requested} "
            f"residual={residual:.6f} matched_fit={'yes' if matched else 'no'}"
        )

    return {
        "bracket_low": bracket_low,
        "bracket_high": bracket_high,
        "winning_vocab": winning_vocab,
        "min_frequency": winning_min_frequency,
        "crossing": crossing,
        "slope": slope,
        "se_diff": se_diff,
        "refine_step": refine_step,
        "refined_points": interior_points,
        "fit_matches": fit_matches,
    }


def compute_optimize_candidate_stats(
    *,
    candidate_results: list[OptimizeCandidateResult],
) -> dict[str, Any]:
    candidate_elapsed_seconds = [result.elapsed_seconds for result in candidate_results]
    candidate_elapsed_sorted = sorted(candidate_elapsed_seconds)
    candidate_unk_rates = [
        float(result.validation_metrics["unk_rate"]) for result in candidate_results
    ]
    candidate_single_char_token_pcts = [
        float(result.validation_metrics["single_char_token_pct"])
        for result in candidate_results
    ]
    candidate_fertilities = [
        float(result.validation_metrics["fertility"]) for result in candidate_results
    ]
    candidate_distances = [
        float(result.validation_metrics["fertility_distance"])
        for result in candidate_results
    ]
    candidate_fertility_generalization_deltas = [
        float(result.fertility_generalization_delta) for result in candidate_results
    ]
    candidate_unk_generalization_deltas = [
        float(result.unk_rate_generalization_delta) for result in candidate_results
    ]
    accepted_candidates = [
        result for result in candidate_results if not result.rejected
    ]

    return {
        "total_candidates": len(candidate_results),
        "accepted_candidates": len(accepted_candidates),
        "rejected_candidates": len(candidate_results) - len(accepted_candidates),
        "elapsed_seconds": {
            "min": min(candidate_elapsed_seconds)
            if candidate_elapsed_seconds
            else None,
            "median": statistics.median(candidate_elapsed_seconds)
            if candidate_elapsed_seconds
            else None,
            "p95": candidate_elapsed_sorted[
                min(
                    len(candidate_elapsed_sorted) - 1,
                    max(0, int(len(candidate_elapsed_sorted) * 0.95) - 1),
                )
            ]
            if candidate_elapsed_sorted
            else None,
            "max": max(candidate_elapsed_seconds)
            if candidate_elapsed_seconds
            else None,
            "mean": (sum(candidate_elapsed_seconds) / len(candidate_elapsed_seconds))
            if candidate_elapsed_seconds
            else None,
        },
        "unk_rate": {
            "min": min(candidate_unk_rates) if candidate_unk_rates else None,
            "median": statistics.median(candidate_unk_rates)
            if candidate_unk_rates
            else None,
            "max": max(candidate_unk_rates) if candidate_unk_rates else None,
        },
        "single_char_token_pct": {
            "min": min(candidate_single_char_token_pcts)
            if candidate_single_char_token_pcts
            else None,
            "median": statistics.median(candidate_single_char_token_pcts)
            if candidate_single_char_token_pcts
            else None,
            "max": max(candidate_single_char_token_pcts)
            if candidate_single_char_token_pcts
            else None,
        },
        "fertility": {
            "min": min(candidate_fertilities) if candidate_fertilities else None,
            "median": statistics.median(candidate_fertilities)
            if candidate_fertilities
            else None,
            "max": max(candidate_fertilities) if candidate_fertilities else None,
        },
        "fertility_distance": {
            "min": min(candidate_distances) if candidate_distances else None,
            "median": statistics.median(candidate_distances)
            if candidate_distances
            else None,
            "max": max(candidate_distances) if candidate_distances else None,
        },
        "fertility_generalization_delta": {
            "min": min(candidate_fertility_generalization_deltas)
            if candidate_fertility_generalization_deltas
            else None,
            "median": statistics.median(candidate_fertility_generalization_deltas)
            if candidate_fertility_generalization_deltas
            else None,
            "max": max(candidate_fertility_generalization_deltas)
            if candidate_fertility_generalization_deltas
            else None,
        },
        "unk_rate_generalization_delta": {
            "min": min(candidate_unk_generalization_deltas)
            if candidate_unk_generalization_deltas
            else None,
            "median": statistics.median(candidate_unk_generalization_deltas)
            if candidate_unk_generalization_deltas
            else None,
            "max": max(candidate_unk_generalization_deltas)
            if candidate_unk_generalization_deltas
            else None,
        },
    }


def list_cleansed_parquet_files(
    *, roots: WorkspaceRoots, systems: list[str]
) -> list[Path]:
    files: list[Path] = []
    for system in systems:
        files.extend(
            resolve_primary_files(
                system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME),
                system_code=system,
            )
        )
    return files


def latest_mtime_ns(paths: list[Path]) -> int:
    if not paths:
        return 0
    return max(path.stat().st_mtime_ns for path in paths if path.exists())


def emit_optimize_dry_run_plan(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    target_scope: str,
    systems: list[str],
    final_tokenizer_path: Path,
    models_dir: Path,
    vocab_schedule: list[int],
    min_frequencies: list[int],
    seeds: list[int],
    pilot_seeds: list[int],
    expansion_seeds: list[int],
    total_candidate_jobs: int,
) -> None:
    suffix = resolve_candidate_model_suffix_for_trainer(trainer=args.tokenizer)
    if target_scope == "country":
        planned_input_files = len(
            resolve_primary_files(
                system_layer_dir(roots, systems[0], layer=CLEANSED_LAYER_NAME),
                system_code=systems[0],
            )
        )
    else:
        planned_input_files = sum(
            len(
                resolve_primary_files(
                    system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME),
                    system_code=system,
                )
            )
            for system in systems
        )

    planned_paths: list[Path] = []
    for vocab_requested in vocab_schedule:
        vocab_tag = "auto" if vocab_requested == -1 else str(vocab_requested)
        for seed in seeds:
            for min_frequency in min_frequencies:
                planned_paths.append(
                    models_dir
                    / f"{args.tokenizer}_seed_{seed}_v{vocab_tag}_mf{min_frequency}.{suffix}"
                )

    unique_paths = len({str(path) for path in planned_paths})
    duplicate_count = len(planned_paths) - unique_paths
    first_path = str(planned_paths[0]) if planned_paths else "none"
    last_path = str(planned_paths[-1]) if planned_paths else "none"

    print(f"[optimize] dry_run scope={target_scope} systems={','.join(systems)}")
    print(f"[optimize] dry_run planned_input_parquet_files={planned_input_files}")
    print(
        f"[optimize] dry_run plan seeds={len(seeds)} pilot_seeds={len(pilot_seeds)} "
        f"expansion_seeds={len(expansion_seeds)} vocab_points={len(vocab_schedule)} "
        f"min_freq_points={len(min_frequencies)}"
    )
    print(
        f"[optimize] dry_run upper_bound_jobs_no_pruning={total_candidate_jobs} "
        f"pilot_upper_bound_jobs={len(vocab_schedule) * len(min_frequencies) * len(pilot_seeds)} "
        f"expansion_upper_bound_jobs={len(vocab_schedule) * len(min_frequencies) * len(expansion_seeds)}"
    )
    print(
        f"[optimize] dry_run model_output_paths_total={len(planned_paths)} "
        f"unique={unique_paths} duplicates={duplicate_count}"
    )
    print(f"[optimize] dry_run first_model_path={first_path}")
    print(f"[optimize] dry_run last_model_path={last_path}")


def path_mtime_ns(path: Path) -> int:
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


def is_final_result_fresh(
    *,
    corpus_path: Path,
    final_tokenizer_path: Path,
    summary_path: Path,
    run_log_dir: Path | None = None,
) -> bool:
    """Whether a finalized result still stands for what it was selected from:
    the final tokenizer and the summary are both newer than the corpus, and
    the summary is no older than any run-log shard under `run_log_dir`, so no
    trial has been logged since the winner was chosen."""
    corpus_mtime_ns = path_mtime_ns(corpus_path)
    if corpus_mtime_ns <= 0:
        return False
    newest_shard_mtime_ns = max(
        (
            path_mtime_ns(shard_path)
            for shard_path in (run_log_dir.glob("*.parquet") if run_log_dir else ())
        ),
        default=0,
    )
    return (
        final_tokenizer_path.exists()
        and summary_path.exists()
        and path_mtime_ns(final_tokenizer_path) >= corpus_mtime_ns
        and path_mtime_ns(summary_path) >= corpus_mtime_ns
        and path_mtime_ns(summary_path) >= newest_shard_mtime_ns
    )


def summary_winner_is_reselected(
    *,
    summary_path: Path,
    run_log_dir: Path,
    run_log_path: Path,
    target_scope: str,
    systems: list[str],
    trainer: str,
    eligibility_pass_rate: float,
    vocab_size_distance_tolerance: float,
    policy: OptimizeCandidatePolicy,
) -> bool:
    """Whether selecting over the run log now picks the winner the summary records.

    `is_final_result_fresh` reads file times only, so a change to the gates
    or the selection score leaves an old winner looking fresh. Selection is
    re-run here from the stored trials, which trains nothing, and a different
    `(vocab_size_requested, min_frequency)` means the result no longer stands.
    """
    summary = load_json_object(summary_path)
    winner = summary.get("winner_candidate") if summary else None
    if not isinstance(winner, dict):
        return False
    history = scoped_history_for_trainer(
        run_log_df=merge_run_logs(
            run_log_dir=run_log_dir, merged_run_log_path=run_log_path
        ),
        scope=target_scope,
        systems_key=",".join(systems),
        trainer=trainer,
    )
    if history.height == 0:
        return False
    best_pair = select_best_pair(
        run_log_df=history,
        eligibility_pass_rate=eligibility_pass_rate,
        vocab_size_distance_tolerance=vocab_size_distance_tolerance,
        policy=policy,
    )
    if best_pair is None:
        return False
    return _grid_value(best_pair["vocab_size_requested"]) == _grid_value(
        winner["vocab_size_requested"]
    ) and _grid_value(best_pair["min_frequency"]) == _grid_value(
        winner["min_frequency"]
    )


def _grid_value(value: object) -> int:
    """A grid coordinate read back from a run log or summary as an int.

    Run logs hold them as ints, a JSON summary may hold them as floats or
    digit strings, and anything else is a malformed record rather than a
    candidate to compare.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"grid coordinate {value!r} is not a number")
    return int(value)


def load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def purge_sweep_candidates(
    *,
    sweep_paths: Any,
    dry_run: bool = False,
) -> tuple[int, int]:
    """Wipe everything under one optimize-sweep leaf (`sweep_paths.models_dir`,
    `run_log_dir`, and the merged run-log/summary). A leaf is already unique
    to exactly one (corpus, trainer, model-variant, grid) configuration --
    see `company_tokenize.resolve_optimize_sweep_paths` -- so, unlike the old
    flat layout, there is no risk of touching another config's data and no
    need to scope by trainer/systems-key. Returns
    (removed_model_files, removed_run_log_rows). `dry_run=True` counts
    without deleting.
    """
    removed_models = 0
    if sweep_paths.models_dir.exists():
        for file_path in sweep_paths.models_dir.glob("*"):
            if file_path.is_file():
                if not dry_run:
                    file_path.unlink(missing_ok=True)
                removed_models += 1

    removed_rows = 0
    if sweep_paths.run_log_dir.exists():
        for shard_path in sorted(sweep_paths.run_log_dir.glob("*.parquet")):
            try:
                shard_df = pl.read_parquet(shard_path)
            except (OSError, pl.exceptions.PolarsError):
                continue
            removed_rows += shard_df.height
            if not dry_run:
                shard_path.unlink(missing_ok=True)

    if not dry_run:
        sweep_paths.run_log_path.unlink(missing_ok=True)
        sweep_paths.summary_path.unlink(missing_ok=True)

    return removed_models, removed_rows


def has_cleansed_parquet(roots: WorkspaceRoots, system_code: str) -> bool:
    cleansed_dir = system_layer_dir(roots, system_code, layer=CLEANSED_LAYER_NAME)
    return bool(resolve_primary_files(cleansed_dir, system_code=system_code))


def resolve_optimize_corpus_for_target(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    target_scope: str,
    systems: list[str],
    corpus_path: Path,
    input_parquet_files: list[Path],
    rebuild_optimize_corpus: Callable[..., int | None],
) -> tuple[int, str] | None:
    latest_input_mtime = latest_mtime_ns(input_parquet_files)
    corpus_exists = corpus_path.exists()
    corpus_mtime = path_mtime_ns(corpus_path)
    corpus_is_fresh = corpus_exists and corpus_mtime >= latest_input_mtime

    if args.force:
        rebuilt_rows = rebuild_optimize_corpus(
            roots=roots,
            target_scope=target_scope,
            systems=systems,
            corpus_path=corpus_path,
            name_col=args.name_col,
            allow_optimize_wipe=args.optimize_wipe,
        )
        if rebuilt_rows is None:
            return None
        return rebuilt_rows, compute_corpus_content_hash(corpus_path)

    if corpus_is_fresh:
        corpus_rows = (
            pl.scan_parquet(corpus_path)
            .select(pl.len().alias("n"))
            .collect()
            .item(0, "n")
        )
        print(
            f"[optimize] reuse_corpus=yes path={corpus_path} "
            f"rows={corpus_rows:,} input_files={len(input_parquet_files)}"
        )
        return int(corpus_rows), compute_corpus_content_hash(corpus_path)

    rebuilt_rows = rebuild_optimize_corpus(
        roots=roots,
        target_scope=target_scope,
        systems=systems,
        corpus_path=corpus_path,
        name_col=args.name_col,
        allow_optimize_wipe=args.optimize_wipe,
    )
    if rebuilt_rows is None:
        return None
    print(
        f"[optimize] reuse_corpus=no rebuilt path={corpus_path} "
        f"rows={rebuilt_rows:,} input_files={len(input_parquet_files)}"
    )
    return rebuilt_rows, compute_corpus_content_hash(corpus_path)


def rebuild_optimize_corpus(
    *,
    roots: WorkspaceRoots,
    target_scope: str,
    systems: list[str],
    corpus_path: Path,
    name_col: str,
    allow_optimize_wipe: bool = False,
    has_cleansed_parquet_func: Callable[..., bool] | None = None,
    sample_country_corpus: Callable[..., int] | None = None,
    sample_global_corpus: Callable[..., int] | None = None,
    all_rows_target: int = 10**12,
) -> int | None:
    from acquisition.tokenizer_ops import (
        sample_training_corpus,
        sample_training_corpus_for_global,
    )

    check_cleansed_parquet = has_cleansed_parquet_func or has_cleansed_parquet
    sample_country = sample_country_corpus or sample_training_corpus
    sample_global = sample_global_corpus or sample_training_corpus_for_global

    if target_scope == "country":
        system_code = systems[0]
        cleansed_dir = system_layer_dir(roots, system_code, layer=CLEANSED_LAYER_NAME)
        if not check_cleansed_parquet(roots=roots, system_code=system_code):
            print(
                f"[warn] Skipping '{system_code}' optimize: no cleansed parquet shards found in {cleansed_dir}"
            )
            return None
        return sample_country(
            cleansed_dir=cleansed_dir,
            corpus_path=corpus_path,
            name_col=name_col,
            allow_optimize_wipe=allow_optimize_wipe,
        )

    cleansed_dirs = [
        system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME) for system in systems
    ]
    return sample_global(
        cleansed_dirs=cleansed_dirs,
        corpus_path=corpus_path,
        sample_fraction=1.0,
        balance_mode="proportional",
        system_target_rows=all_rows_target,
        system_min_rows=0,
        allow_optimize_wipe=allow_optimize_wipe,
        system_max_rows=None,
        file_min_rows=0,
        balance_seed=42,
        name_col=name_col,
    )


def prepare_optimize_target_execution(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    target_scope: str,
    systems: list[str],
    artifact_paths: Any,
    trainer_options: TrainerOptions,
    candidate_policy: OptimizeCandidatePolicy,
    seeds: list[int],
    vocab_schedule: list[int],
    min_frequencies: list[int],
    effective_elbow_extra_steps: int,
    corpus_content_hash: str,
    session_id: str,
) -> OptimizeExecutionSetup | None:
    final_tokenizer_path = resolve_tokenizer_path_for_trainer(
        trainer=args.tokenizer,
        scope_directory=artifact_paths.directory,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
    )
    optimize_dir = final_tokenizer_path.parent / OPTIMIZE_DIRNAME
    optimize_dir.mkdir(parents=True, exist_ok=True)

    active_canary_payload = build_optimize_canary_payload(
        target_scope=target_scope,
        systems=systems,
        trainer=args.tokenizer,
        trainer_options=trainer_options,
        profile=args.profile,
        vocab_schedule=vocab_schedule,
        min_frequencies=min_frequencies,
        seeds=seeds,
        elbow_extra_steps=effective_elbow_extra_steps,
        validation_fraction=args.validation_fraction,
        policy=candidate_policy,
        vocab_size_distance_tolerance=args.vocab_size_distance_tolerance,
        pilot_seed_count=args.pilot_seed_count,
        vocab_frontier_top_k=args.vocab_frontier_top_k,
        vocab_frontier_distance_tolerance=args.vocab_frontier_distance_tolerance,
        corpus_path=artifact_paths.corpus_path,
        corpus_content_hash=corpus_content_hash,
        project_root=roots.checkout,
    )
    grid_hash = compute_optimize_grid_hash(active_canary_payload)
    sweep_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=corpus_content_hash,
        grid_hash=grid_hash,
    )
    sweep_paths.sweep_dir.mkdir(parents=True, exist_ok=True)
    sweep_paths.models_dir.mkdir(parents=True, exist_ok=True)
    sweep_paths.run_log_dir.mkdir(parents=True, exist_ok=True)
    models_dir = sweep_paths.models_dir
    run_log_dir = sweep_paths.run_log_dir
    run_log_path = sweep_paths.run_log_path
    summary_path = sweep_paths.summary_path

    if args.force:
        removed_models, removed_rows = purge_sweep_candidates(
            sweep_paths=sweep_paths,
        )
        if removed_models or removed_rows:
            print(
                f"[optimize] force purge trainer={args.tokenizer} models={removed_models} run_log_rows={removed_rows}"
            )

    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    pointer_path.write_text(
        json.dumps(
            {
                "corpus_hash": corpus_content_hash,
                "grid_hash": grid_hash,
                **active_canary_payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sweep_paths.canary_sidecar_path.write_text(
        json.dumps(active_canary_payload, indent=2), encoding="utf-8"
    )
    # A leaf directory is unique to exactly one (corpus, trainer,
    # model-variant, grid) configuration by construction (see
    # `resolve_optimize_sweep_paths`), so simply landing on it already means
    # reuse is valid -- there is no longer a stored-vs-active canary
    # comparison to make.
    reuse_existing_candidates = not args.force
    print(
        f"[optimize] reuse_existing_candidates={'yes' if reuse_existing_candidates else 'no'}"
    )

    final_result_is_fresh = is_final_result_fresh(
        corpus_path=artifact_paths.corpus_path,
        final_tokenizer_path=final_tokenizer_path,
        summary_path=summary_path,
        run_log_dir=run_log_dir,
    )
    if (
        reuse_existing_candidates
        and args.finalize
        and final_result_is_fresh
        and getattr(args, "reoptimize", False)
    ):
        print(
            f"[optimize] reoptimize target={target_scope}:{','.join(systems)} "
            "reason=requested"
        )
        final_result_is_fresh = False
    if (
        reuse_existing_candidates
        and args.finalize
        and final_result_is_fresh
        and not summary_winner_is_reselected(
            summary_path=summary_path,
            run_log_dir=run_log_dir,
            run_log_path=run_log_path,
            target_scope=target_scope,
            systems=systems,
            trainer=args.tokenizer,
            eligibility_pass_rate=args.eligibility_pass_rate,
            vocab_size_distance_tolerance=args.vocab_size_distance_tolerance,
            policy=candidate_policy,
        )
    ):
        print(
            f"[optimize] stale target={target_scope}:{','.join(systems)} "
            "reason=selection_now_picks_a_different_winner"
        )
        final_result_is_fresh = False
    if reuse_existing_candidates and args.finalize and final_result_is_fresh:
        print(
            f"[optimize] skip target={target_scope}:{','.join(systems)} "
            "reason=final_result_fresh_vs_corpus"
        )
        return None

    historical_candidate_keys: set[tuple[int, int, int]] = set()
    historical_results: list[OptimizeCandidateResult] = []
    bootstrap_rss_bytes: int | None = None
    if reuse_existing_candidates:
        existing_history_df = merge_run_logs(
            run_log_dir=run_log_dir,
            merged_run_log_path=run_log_path,
        )
        # Scope before using: run_log_dir is shared across trainers for the
        # same scope/system (e.g. wordpiece and sentencepiece both write into
        # optimize/run_logs/), so an unfiltered history could treat a
        # different trainer's candidate as satisfying this one's exclusion
        # key/reconstruction whenever they happen to coincide on
        # (seed, vocab, min_frequency).
        scoped_history_df = scoped_history_for_trainer(
            run_log_df=existing_history_df,
            scope=target_scope,
            systems_key=",".join(systems),
            trainer=args.tokenizer,
        )
        scoped_history_df = backfill_trained_vocab_sizes(
            run_log_df=scoped_history_df, trainer=args.tokenizer
        )
        required_columns = {"seed", "vocab_size_requested", "min_frequency"}
        if scoped_history_df.height > 0 and required_columns.issubset(
            set(scoped_history_df.columns)
        ):
            for row in scoped_history_df.iter_rows(named=True):
                historical_candidate_keys.add(
                    candidate_key(
                        seed=int(row["seed"]),
                        vocab_requested=int(row["vocab_size_requested"]),
                        min_frequency=int(row["min_frequency"]),
                    )
                )
                historical_results.append(
                    reconstruct_optimize_candidate_result_from_history_row(row)
                )
        bootstrap_rss_bytes = historical_worker_rss_estimate(scoped_history_df)
        print(
            f"[optimize] reuse historical_candidates={len(historical_candidate_keys)}"
        )

    excluded_candidate_keys: set[tuple[int, int, int]] = (
        set(historical_candidate_keys) if reuse_existing_candidates else set()
    )
    if excluded_candidate_keys:
        print(
            f"[optimize] exclusion_list size={len(excluded_candidate_keys)} source=history"
        )

    shard_run_log_path = run_log_dir / f"{session_id}.parquet"
    session_run_log_df = (
        pl.read_parquet(shard_run_log_path)
        if shard_run_log_path.exists()
        else pl.DataFrame()
    )

    return {
        "final_tokenizer_path": final_tokenizer_path,
        "optimize_dir": optimize_dir,
        "models_dir": models_dir,
        "run_log_dir": run_log_dir,
        "run_log_path": run_log_path,
        "summary_path": summary_path,
        "reuse_existing_candidates": reuse_existing_candidates,
        "excluded_candidate_keys": excluded_candidate_keys,
        "historical_results": historical_results,
        "bootstrap_rss_bytes": bootstrap_rss_bytes,
        "shard_run_log_path": shard_run_log_path,
        "session_run_log_df": session_run_log_df,
    }


def seed_split_artifact_paths(
    *, optimize_dir: Path, seed: int
) -> tuple[Path, Path, Path]:
    train_split_path = optimize_dir / f"train_seed_{seed}.parquet"
    validation_split_path = optimize_dir / f"validation_seed_{seed}.parquet"
    metadata_path = optimize_dir / f"split_seed_{seed}.json"
    return train_split_path, validation_split_path, metadata_path


def resolve_or_create_seed_split_artifacts(
    *,
    corpus_path: Path,
    optimize_dir: Path,
    seed: int,
    validation_fraction: float,
    corpus_content_hash: str,
    roots: WorkspaceRoots | None = None,
) -> tuple[Path, Path, int, int, bool]:
    train_split_path, validation_split_path, metadata_path = seed_split_artifact_paths(
        optimize_dir=optimize_dir,
        seed=seed,
    )
    metadata = load_json_object(metadata_path)
    if (
        metadata is not None
        and train_split_path.exists()
        and validation_split_path.exists()
        and int(metadata.get("seed", -1)) == seed
        and float(metadata.get("validation_fraction", -1.0))
        == float(validation_fraction)
        and str(metadata.get("corpus_content_hash", "")) == corpus_content_hash
    ):
        train_rows = int(metadata.get("train_rows", 0))
        validation_rows = int(metadata.get("validation_rows", 0))
        if train_rows > 0 and validation_rows > 0:
            return (
                train_split_path,
                validation_split_path,
                train_rows,
                validation_rows,
                True,
            )

    train_split_path, validation_split_path, train_rows, validation_rows = (
        write_seed_split_artifacts(
            corpus_path=corpus_path,
            optimize_dir=optimize_dir,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    )
    metadata_payload = {
        "seed": seed,
        "validation_fraction": float(validation_fraction),
        "corpus_path": to_portable_path_str(
            corpus_path, project_root=roots.checkout if roots is not None else None
        ),
        "corpus_content_hash": corpus_content_hash,
        "train_rows": int(train_rows),
        "validation_rows": int(validation_rows),
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return train_split_path, validation_split_path, train_rows, validation_rows, False


def prepare_optimize_seed_splits(
    *,
    seeds: list[int],
    corpus_path: Path,
    optimize_dir: Path,
    validation_fraction: float,
    corpus_content_hash: str,
    roots: WorkspaceRoots | None = None,
) -> tuple[dict[int, OptimizeSeedSplit], int, int]:
    seed_splits: dict[int, OptimizeSeedSplit] = {}
    reused_seed_splits = 0
    created_seed_splits = 0

    for seed in seeds:
        (
            train_split_path,
            validation_split_path,
            train_rows,
            validation_rows,
            reused_split,
        ) = resolve_or_create_seed_split_artifacts(
            corpus_path=corpus_path,
            optimize_dir=optimize_dir,
            seed=seed,
            validation_fraction=validation_fraction,
            corpus_content_hash=corpus_content_hash,
            roots=roots,
        )
        if reused_split:
            reused_seed_splits += 1
        else:
            created_seed_splits += 1

        seed_splits[seed] = {
            "train_split_path": train_split_path,
            "validation_split_path": validation_split_path,
            "train_rows": train_rows,
            "validation_rows": validation_rows,
        }

    return seed_splits, reused_seed_splits, created_seed_splits


def resolve_optimize_preset_name(
    *, args: argparse.Namespace, presets: dict[str, OptimizePreset]
) -> str:
    """Resolve which named preset governs this run.

    An explicit `--optimize-preset` always wins. With none given, falls back
    to the trainer-keyed default that reproduces this mechanism's pre-preset
    behavior (a WordPiece vs. SentencePiece run picked different threshold
    defaults purely from `--tokenizer`, with no selection flag involved at
    all) -- an existing invocation that never mentions `--optimize-preset`
    keeps choosing the same values it always did.
    """
    explicit = getattr(args, "optimize_preset", None)
    if explicit:
        return explicit
    return DEFAULT_OPTIMIZE_PRESET_BY_TRAINER.get(
        args.tokenizer, DEFAULT_OPTIMIZE_PRESET
    )


def apply_optimize_preset(
    *,
    args: argparse.Namespace,
    presets: dict[str, OptimizePreset],
    base_defaults: dict[str, float | int],
    default_vocab_grid: str,
    default_min_freq_grid: str,
) -> tuple[str, dict[str, Any]]:
    """Apply a named preset's threshold and grid defaults to `args`.

    A preset only fills a field still sitting at its own global argparse
    default -- any explicitly-passed flag always wins, whether or not it
    happens to match the preset's own value, so no existing invocation's
    behavior changes just because presets now exist. Returns
    `(resolved_preset_name, overrides)`, where `overrides` names every field
    whose final value differs from what the resolved preset would have set
    (an explicit flag that won), for the run manifest to record alongside
    the preset name.

    Raises `ValueError` naming the available presets if `resolved_preset_name`
    is not one of `presets` -- the CLI's own `--optimize-preset` choices
    already reject an unknown name at parse time, so this only fires for a
    caller that bypassed argparse (a test, or another script's own parser).
    """
    preset_name = resolve_optimize_preset_name(args=args, presets=presets)
    if preset_name not in presets:
        available = ", ".join(sorted(presets))
        raise ValueError(
            f"Unknown optimize preset {preset_name!r}. Available presets: {available}."
        )
    preset = presets[preset_name]

    overrides: dict[str, Any] = {}
    for key, preset_value in preset["thresholds"].items():
        base_default = base_defaults[key]
        current = getattr(args, key)
        if current == base_default:
            setattr(args, key, preset_value)
        elif current != preset_value:
            overrides[key] = current

    grid_fields = (
        ("vocab_sizes", preset["vocab_sizes"], default_vocab_grid),
        ("min_frequencies", preset["min_frequencies"], default_min_freq_grid),
    )
    for attr_name, preset_grid_value, global_default in grid_fields:
        current = getattr(args, attr_name)
        if current == global_default:
            setattr(args, attr_name, preset_grid_value)
        elif current != preset_grid_value:
            overrides[attr_name] = current

    args.optimize_preset = preset_name
    args.optimize_preset_overrides = overrides
    return preset_name, overrides


def parse_int_list(raw: str, *, allow_negative_one: bool = False) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0 and not (allow_negative_one and value == -1):
            raise ValueError(f"Expected positive integer list item, got {value}.")
        values.append(value)
    if not values:
        raise ValueError("Expected at least one value.")
    return values


def resolve_seed_values(
    *, args: argparse.Namespace, parse_int_list_func: Callable[..., list[int]]
) -> list[int]:
    if args.seeds.strip():
        return parse_int_list_func(args.seeds)
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be greater than zero.")
    return [args.base_seed + i for i in range(args.num_runs)]


def _raise_if_invalid(condition: bool, message: str) -> None:
    if condition:
        raise ValueError(message)


def validate_optimize_args(args: argparse.Namespace) -> None:
    _raise_if_invalid(
        not (0 < args.validation_fraction < 1),
        "--validation-fraction must be between 0 and 1.",
    )
    _raise_if_invalid(
        args.unk_rate_threshold <= 0,
        "--unk-rate-threshold must be greater than zero.",
    )
    _raise_if_invalid(
        args.eligibility_pass_rate <= 0 or args.eligibility_pass_rate > 1,
        "--eligibility-pass-rate must be in the interval (0, 1].",
    )
    _raise_if_invalid(
        args.vocab_size_distance_tolerance < 0,
        "--vocab-size-distance-tolerance must be greater than or equal to zero.",
    )
    _raise_if_invalid(
        args.elbow_min_points < 1,
        "--elbow-min-points must be greater than or equal to 1.",
    )
    _raise_if_invalid(
        args.elbow_min_improvement < 0,
        "--elbow-min-improvement must be greater than or equal to zero.",
    )
    _raise_if_invalid(
        args.elbow_extra_steps < 0,
        "--elbow-extra-steps must be greater than or equal to zero.",
    )
    _raise_if_invalid(
        args.fertility_tolerance <= 0,
        "--fertility-tolerance must be greater than zero.",
    )
    _raise_if_invalid(
        args.fertility_min >= args.fertility_max,
        "--fertility-min must be less than --fertility-max.",
    )
    _raise_if_invalid(
        args.token_count_median_max <= 0,
        "--token-count-median-max must be greater than zero.",
    )
    _raise_if_invalid(
        args.token_count_p95_max <= 0,
        "--token-count-p95-max must be greater than zero.",
    )
    _raise_if_invalid(
        args.token_count_p95_max < args.token_count_median_max,
        "--token-count-p95-max must be greater than or equal to --token-count-median-max.",
    )
    _raise_if_invalid(
        args.diagnostic_top_n <= 0,
        "--diagnostic-top-n must be greater than zero.",
    )
    _raise_if_invalid(
        args.pilot_seed_count <= 0,
        "--pilot-seed-count must be greater than zero.",
    )
    _raise_if_invalid(
        args.vocab_frontier_top_k <= 0,
        "--vocab-frontier-top-k must be greater than zero.",
    )
    _raise_if_invalid(
        args.vocab_frontier_distance_tolerance < 0,
        "--vocab-frontier-distance-tolerance must be greater than or equal to zero.",
    )
    _raise_if_invalid(
        args.worker_memory_floor_gb <= 0,
        "--worker-memory-floor-gb must be greater than zero.",
    )
    _raise_if_invalid(
        not (0.0 <= args.memory_headroom_fraction < 1.0),
        "--memory-headroom-fraction must be in the interval [0, 1).",
    )
    _raise_if_invalid(
        args.wave_chunk_size <= 0,
        "--wave-chunk-size must be greater than zero.",
    )
    _raise_if_invalid(
        args.max_parallel_waves is not None and args.max_parallel_waves <= 0,
        "--max-parallel-waves must be greater than zero when provided.",
    )
    _raise_if_invalid(
        args.candidate_timeout_seconds <= 0,
        "--candidate-timeout-seconds must be greater than zero.",
    )


def resolve_optimize_search_space(
    args: argparse.Namespace,
    resolve_seed_values_func: Callable[..., list[int]] | None = None,
    parse_int_list_func: Callable[..., list[int]] | None = None,
    vocab_sort_key_func: Callable[[int], tuple[int, int]] | None = None,
    normalize_min_frequencies: Callable[..., list[int]] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    resolved_seed_values = resolve_seed_values_func or resolve_seed_values
    parsed_int_list = parse_int_list_func or parse_int_list
    resolved_vocab_sort_key = vocab_sort_key_func or vocab_sort_key
    normalize_min_frequencies_func = (
        normalize_min_frequencies or normalize_optimize_min_frequencies_for_trainer
    )

    seeds = resolved_seed_values(args=args, parse_int_list_func=parsed_int_list)
    vocab_sizes = parsed_int_list(args.vocab_sizes, allow_negative_one=True)
    vocab_schedule = sorted(vocab_sizes, key=resolved_vocab_sort_key)
    min_frequencies = normalize_min_frequencies_func(
        trainer=args.tokenizer,
        min_frequencies=parsed_int_list(args.min_frequencies),
    )
    return seeds, vocab_schedule, min_frequencies


def build_optimize_candidate_policy(
    args: argparse.Namespace,
) -> OptimizeCandidatePolicy:
    return OptimizeCandidatePolicy(
        fertility_target=args.fertility_target,
        unk_rate_threshold=args.unk_rate_threshold,
        fertility_tolerance=args.fertility_tolerance,
        fertility_min=args.fertility_min,
        fertility_max=args.fertility_max,
        token_count_median_max=args.token_count_median_max,
        token_count_p95_max=args.token_count_p95_max,
        eligibility_pass_rate=args.eligibility_pass_rate,
        diagnostic_top_n=args.diagnostic_top_n,
        delete_rejected_models=args.delete_rejected_models,
    )


def finalize_optimize_target(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    target_scope: str,
    systems: list[str],
    artifact_paths: Any,
    trainer_options: TrainerOptions,
    tracker: OptimizeExecutionTracker,
    best_pair: dict[str, Any],
    final_tokenizer_path: Path,
    run_log_path: Path,
    summary_path: Path,
    session_id: str,
    wall_started: float,
    effective_elbow_extra_steps: int,
    corpus_rows: int,
    write_token_score_artifact_func: Callable[..., dict[str, Any]] | None = None,
    build_summary_payload: Callable[..., dict[str, Any]] | None = None,
    resolve_metadata_path_for_trainer_func: Callable[..., Path] | None = None,
    trainer_params_payload_builder: Callable[..., dict[str, Any]] | None = None,
    copy_file: Callable[[Path, Path], Any] | None = None,
    build_run_manifest_func: Callable[..., dict[str, Any]] | None = None,
    publish_promoted_candidate_func: Callable[..., Any] | None = None,
) -> None:
    write_token_score = write_token_score_artifact_func or write_token_score_artifact
    publish_promoted = publish_promoted_candidate_func or publish_promoted_candidate
    build_summary = build_summary_payload or build_optimize_summary_payload
    resolve_metadata_path = (
        resolve_metadata_path_for_trainer_func or resolve_metadata_path_for_trainer
    )
    build_trainer_params_payload = (
        trainer_params_payload_builder or trainer_params_payload
    )
    copy_file_func = copy_file or shutil.copy2
    build_manifest = build_run_manifest_func or build_run_manifest

    best_vocab_requested = int(best_pair["vocab_size_requested"])
    best_min_frequency = int(best_pair["min_frequency"])
    final_vocab_arg = None if best_vocab_requested == -1 else best_vocab_requested

    print(
        "[optimize] Selected best pair from history "
        f"vocab={'auto' if best_vocab_requested == -1 else best_vocab_requested} "
        f"effective_vocab={int(float(best_pair['selection_vocab_size'])) if best_pair.get('selection_vocab_size') is not None else 'unknown'} "
        f"min_frequency={best_min_frequency} "
        f"median_distance={float(best_pair['median_fertility_distance']):.4f} "
        f"pass_rate={float(best_pair['pass_rate']):.3f}"
    )

    retrain_started = time.perf_counter()
    final_resolved_vocab = train_tokenizer_with_trainer(
        trainer=args.tokenizer,
        corpus_path=artifact_paths.corpus_path,
        tokenizer_path=final_tokenizer_path,
        vocab_size=final_vocab_arg,
        min_frequency=best_min_frequency,
        options=trainer_options,
    )
    token_score_artifact = write_token_score(
        corpus_path=artifact_paths.corpus_path,
        tokenizer_path=final_tokenizer_path,
        trainer=args.tokenizer,
        roots=roots,
    )
    print(
        "[optimize] token_score_artifact "
        + f"status={token_score_artifact['status']} invoked={token_score_artifact['invoked']}"
    )
    final_metrics, per_system_rows, high_fertility_rows, unk_rows = (
        evaluate_tokenizer_metrics(
            corpus_path=artifact_paths.corpus_path,
            tokenizer_path=final_tokenizer_path,
            trainer=args.tokenizer,
            fertility_target=args.fertility_target,
            diagnostic_top_n=args.diagnostic_top_n,
        )
    )

    easy_result_path = None
    if target_scope == "country":
        easy_result_path = resolve_tokenizer_path_for_trainer(
            trainer=args.tokenizer,
            scope_directory=tokenizer_scope_dir(roots, system=systems[0]),
            tokenizer_encoding=trainer_options.tokenizer_encoding,
        )
        easy_result_path.parent.mkdir(parents=True, exist_ok=True)
        if easy_result_path != final_tokenizer_path:
            copy_file_func(final_tokenizer_path, easy_result_path)

    candidate_stats = compute_optimize_candidate_stats(
        candidate_results=tracker.candidate_results,
    )
    summary = build_summary(
        trainer=args.tokenizer,
        target_scope=target_scope,
        systems=systems,
        session_id=session_id,
        wall_seconds=time.perf_counter() - wall_started,
        effective_elbow_extra_steps=effective_elbow_extra_steps,
        elbow_min_points=args.elbow_min_points,
        elbow_min_improvement=args.elbow_min_improvement,
        best_pair=best_pair,
        final_resolved_vocab=final_resolved_vocab,
        best_vocab_requested=best_vocab_requested,
        best_min_frequency=best_min_frequency,
        final_tokenizer_path=final_tokenizer_path,
        easy_result_path=easy_result_path,
        final_metrics=final_metrics,
        retrain_elapsed_seconds=time.perf_counter() - retrain_started,
        per_system_rows=per_system_rows,
        high_fertility_rows=high_fertility_rows,
        unk_rows=unk_rows,
        run_log_path=run_log_path,
        summary_path=summary_path,
        trainer_options=trainer_options,
        candidate_stats=candidate_stats,
        eligibility_pass_rate=args.eligibility_pass_rate,
        vocab_size_distance_tolerance=args.vocab_size_distance_tolerance,
        unk_rate_threshold=args.unk_rate_threshold,
        delete_rejected_models=args.delete_rejected_models,
        fertility_target=args.fertility_target,
        fertility_tolerance=args.fertility_tolerance,
        fertility_min=args.fertility_min,
        fertility_max=args.fertility_max,
        token_count_median_max=args.token_count_median_max,
        token_count_p95_max=args.token_count_p95_max,
        completed_jobs=tracker.completed_jobs,
        submitted_jobs=tracker.submitted_jobs,
        expansion_strategy=args.expansion_strategy,
        project_root=roots.checkout,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metadata_path = scope_directory_files(artifact_paths.directory).metadata
    trainer_metadata_path = resolve_metadata_path(
        trainer=args.tokenizer,
        scope_directory=artifact_paths.directory,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
    )
    trainer_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_manifest(
        created_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        mode="optimize",
        trainer=args.tokenizer,
        scope=target_scope,
        systems=systems,
        profile=args.profile if target_scope == "global" else None,
        all_rows=True,
        vocab_size=final_resolved_vocab,
        min_frequency=best_min_frequency,
        name_col=args.name_col,
        preprocess_profile=TRAINING_PREPROCESS_PROFILE,
        corpus_path=to_portable_path_str(
            artifact_paths.corpus_path, project_root=roots.checkout
        ),
        corpus_content_hash=corpus_content_hash_for_manifest(
            artifact_paths.corpus_path
        ),
        tokenizer_path=to_portable_path_str(
            final_tokenizer_path, project_root=roots.checkout
        ),
        line_count=corpus_rows,
        run_log_path=to_portable_path_str(run_log_path, project_root=roots.checkout),
        summary_path=to_portable_path_str(summary_path, project_root=roots.checkout),
        trainer_params=build_trainer_params_payload(
            trainer=args.tokenizer, options=trainer_options
        ),
        token_score_artifact=token_score_artifact,
        optimize_preset=getattr(args, "optimize_preset", None),
        optimize_preset_overrides=getattr(args, "optimize_preset_overrides", None),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trainer_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    promoted, replaced = publish_promoted(
        roots=roots,
        system=systems[0] if target_scope == "country" else None,
        trainer=args.tokenizer,
        tokenizer_encoding=trainer_options.tokenizer_encoding,
        model_path=final_tokenizer_path,
        metadata_path=trainer_metadata_path,
        token_scores_path=tokenizer_directory_files(
            final_tokenizer_path.parent, trainer=args.tokenizer
        ).token_scores,
        corpus_path=artifact_paths.corpus_path,
        vocab_size=final_resolved_vocab,
        min_frequency=best_min_frequency,
        metrics=final_metrics,
        optimize_summary=summary,
        parameters={
            "mode": "optimize",
            "scope": target_scope,
            "systems": list(systems),
            "session_id": session_id,
            "trainer_params": metadata.get("trainer_params"),
        },
        invocation=sys.argv,
    )
    print(f"[optimize] promoted={promoted} replaced={replaced}")

    print("[optimize] final tokenizer retrained and summary written")
    print(f"[optimize] tokenizer={final_tokenizer_path}")
    if easy_result_path is not None:
        print(f"[optimize] easy_result={easy_result_path}")
    print(f"[optimize] metadata={metadata_path}")
    print(f"[optimize] trainer_metadata={trainer_metadata_path}")
    print(f"[optimize] run_log={run_log_path}")
    print(f"[optimize] summary={summary_path}")


def run_optimize_target(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    target: Any,
    trainer_options: TrainerOptions,
    candidate_policy: OptimizeCandidatePolicy,
    seeds: list[int],
    vocab_schedule: list[int],
    min_frequencies: list[int],
    session_id: str,
    effective_elbow_extra_steps: int,
    rebuild_optimize_corpus_func: Callable[..., int | None] | None = None,
    prepare_optimize_seed_splits_func: Callable[
        ..., tuple[dict[int, OptimizeSeedSplit], int, int]
    ]
    | None = None,
    select_vocab_frontier_func: Callable[..., set[tuple[int, int]]] | None = None,
    combo_prune_reason_func: Callable[..., str | None] | None = None,
    finalize_optimize_target_func: Callable[..., None] | None = None,
    resolve_optimize_max_workers_func: Callable[..., int] | None = None,
    expansion_strategy: ExpansionStrategy | None = None,
    resolve_expansion_strategy_func: Callable[[str], ExpansionStrategy] | None = None,
) -> None:
    rebuild_corpus = rebuild_optimize_corpus_func or rebuild_optimize_corpus
    prepare_seed_splits = (
        prepare_optimize_seed_splits_func or prepare_optimize_seed_splits
    )
    resolve_strategy = resolve_expansion_strategy_func or resolve_expansion_strategy
    strategy = expansion_strategy or resolve_strategy(
        getattr(args, "expansion_strategy", HEURISTIC_STRATEGY_NAME)
    )
    # select_vocab_frontier_func/combo_prune_reason_func stay as higher-precedence
    # overrides so existing callers/tests that pass them keep working unchanged.
    select_frontier = select_vocab_frontier_func or strategy.select_frontier
    combo_reason = combo_prune_reason_func or strategy.prune_reason
    record_result = strategy.record_result
    finalize_optimize_target_impl = (
        finalize_optimize_target_func or finalize_optimize_target
    )
    resolve_max_workers = (
        resolve_optimize_max_workers_func or resolve_optimize_max_workers
    )

    systems = target.systems
    artifact_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope=target.scope,
        system=systems[0] if target.scope == "country" else None,
        profile=args.profile,
        corpus_filename=args.corpus_filename,
    )

    pilot_seed_count = min(args.pilot_seed_count, len(seeds))
    pilot_seeds = seeds[:pilot_seed_count]
    expansion_seeds = seeds[pilot_seed_count:]
    total_candidate_jobs = len(vocab_schedule) * len(min_frequencies) * len(seeds)

    input_parquet_files = list_cleansed_parquet_files(roots=roots, systems=systems)
    corpus_resolution = resolve_optimize_corpus_for_target(
        args=args,
        roots=roots,
        target_scope=target.scope,
        systems=systems,
        corpus_path=artifact_paths.corpus_path,
        input_parquet_files=input_parquet_files,
        rebuild_optimize_corpus=rebuild_corpus,
    )
    if corpus_resolution is None:
        return
    corpus_rows, corpus_content_hash = corpus_resolution

    if args.dry_run:
        final_tokenizer_path = resolve_tokenizer_path_for_trainer(
            trainer=args.tokenizer,
            scope_directory=artifact_paths.directory,
            tokenizer_encoding=trainer_options.tokenizer_encoding,
        )
        dry_run_canary_payload = build_optimize_canary_payload(
            target_scope=target.scope,
            systems=systems,
            trainer=args.tokenizer,
            trainer_options=trainer_options,
            profile=args.profile,
            vocab_schedule=vocab_schedule,
            min_frequencies=min_frequencies,
            seeds=seeds,
            elbow_extra_steps=effective_elbow_extra_steps,
            validation_fraction=args.validation_fraction,
            policy=candidate_policy,
            vocab_size_distance_tolerance=args.vocab_size_distance_tolerance,
            pilot_seed_count=args.pilot_seed_count,
            vocab_frontier_top_k=args.vocab_frontier_top_k,
            vocab_frontier_distance_tolerance=args.vocab_frontier_distance_tolerance,
            corpus_path=artifact_paths.corpus_path,
            corpus_content_hash=corpus_content_hash,
            project_root=roots.checkout,
        )
        dry_run_sweep_paths = resolve_optimize_sweep_paths(
            optimize_dir=final_tokenizer_path.parent / OPTIMIZE_DIRNAME,
            corpus_content_hash=corpus_content_hash,
            grid_hash=compute_optimize_grid_hash(dry_run_canary_payload),
        )
        emit_optimize_dry_run_plan(
            args=args,
            roots=roots,
            target_scope=target.scope,
            systems=systems,
            final_tokenizer_path=final_tokenizer_path,
            models_dir=dry_run_sweep_paths.models_dir,
            vocab_schedule=vocab_schedule,
            min_frequencies=min_frequencies,
            seeds=seeds,
            pilot_seeds=pilot_seeds,
            expansion_seeds=expansion_seeds,
            total_candidate_jobs=total_candidate_jobs,
        )
        return

    execution_setup = prepare_optimize_target_execution(
        args=args,
        roots=roots,
        target_scope=target.scope,
        systems=systems,
        artifact_paths=artifact_paths,
        trainer_options=trainer_options,
        candidate_policy=candidate_policy,
        seeds=seeds,
        vocab_schedule=vocab_schedule,
        min_frequencies=min_frequencies,
        effective_elbow_extra_steps=effective_elbow_extra_steps,
        corpus_content_hash=corpus_content_hash,
        session_id=session_id,
    )
    if execution_setup is None:
        return

    final_tokenizer_path = execution_setup["final_tokenizer_path"]
    models_dir = execution_setup["models_dir"]
    run_log_dir = execution_setup["run_log_dir"]
    run_log_path = execution_setup["run_log_path"]
    summary_path = execution_setup["summary_path"]
    reuse_existing_candidates = bool(execution_setup["reuse_existing_candidates"])
    excluded_candidate_keys = execution_setup["excluded_candidate_keys"]
    historical_results = execution_setup["historical_results"]
    bootstrap_rss_bytes = execution_setup["bootstrap_rss_bytes"]
    shard_run_log_path = execution_setup["shard_run_log_path"]
    session_run_log_df = execution_setup["session_run_log_df"]

    # Candidates skipped via reuse never pass through drain_candidate_futures,
    # so without this, a fully-reused pilot phase leaves tracker.pilot_results
    # empty and select_vocab_frontier finds nothing to expand -- even though
    # the real data was sitting in history a moment ago. pilot_seeds and
    # expansion_seeds are disjoint by construction, so this split can't
    # double-count a candidate between the two halves.
    pilot_seed_set = set(pilot_seeds)
    historical_pilot_results = [
        result for result in historical_results if result.seed in pilot_seed_set
    ]
    historical_expansion_results = [
        result for result in historical_results if result.seed not in pilot_seed_set
    ]

    # Splits live one level above the sweep leaf, keyed only by corpus
    # identity -- they're valid for any trainer/model-variant/grid sharing
    # this corpus, not just this sweep's own configuration.
    splits_dir = resolve_optimize_splits_dir(
        scope_directory=artifact_paths.directory,
        corpus_content_hash=corpus_content_hash,
    )
    splits_dir.mkdir(parents=True, exist_ok=True)
    seed_splits, reused_seed_splits, created_seed_splits = prepare_seed_splits(
        seeds=seeds,
        corpus_path=artifact_paths.corpus_path,
        optimize_dir=splits_dir,
        validation_fraction=args.validation_fraction,
        corpus_content_hash=corpus_content_hash,
        roots=roots,
    )
    print(
        f"[optimize] seed_splits reused={reused_seed_splits} created={created_seed_splits} "
        f"total={len(seeds)}"
    )

    max_workers = resolve_max_workers(args.max_workers, total_candidate_jobs)
    schedule_text = ",".join(
        "auto" if value == -1 else str(value) for value in vocab_schedule
    )
    print(
        f"[optimize] vocab_schedule_asc={schedule_text} "
        f"pilot_seeds={len(pilot_seeds)} expansion_seeds={len(expansion_seeds)} "
        f"potential_jobs={total_candidate_jobs} workers={max_workers}"
    )

    tracker = OptimizeExecutionTracker(session_run_log_df=session_run_log_df)
    tracker.pilot_results.extend(historical_pilot_results)
    # Per-session shard, mirroring shard_run_log_path -- a concurrent run
    # writing its own failures never interleaves lines with this one's.
    candidate_failure_log_path = run_log_dir / f"{session_id}_candidate_failures.jsonl"
    run_context = OptimizeRunContext(
        args=args,
        trainer_options=trainer_options,
        candidate_policy=candidate_policy,
        session_id=session_id,
        target_scope=target.scope,
        systems=systems,
        shard_run_log_path=shard_run_log_path,
        models_dir=models_dir,
        reuse_existing_candidates=reuse_existing_candidates,
        excluded_candidate_keys=excluded_candidate_keys,
        total_candidate_jobs=total_candidate_jobs,
        max_workers=max_workers,
        expansion_strategy_name=strategy.name,
        candidate_timeout_seconds=args.candidate_timeout_seconds,
        candidate_failure_log_path=candidate_failure_log_path,
    )

    adaptive_enabled = not args.disable_adaptive_concurrency
    concurrency_controller = AdaptiveConcurrencyController(
        hard_cap=max_workers,
        memory_floor_bytes=int(args.worker_memory_floor_gb * 2**30),
        bootstrap_rss_bytes=bootstrap_rss_bytes,
        enabled=adaptive_enabled,
    )
    memory_budget_ledger = MemoryBudgetLedger(
        headroom_fraction=args.memory_headroom_fraction
    )
    max_parallel_waves = args.max_parallel_waves or max_workers
    if adaptive_enabled:
        bootstrap_gb = (
            f"{bootstrap_rss_bytes / 2**30:.2f}"
            if bootstrap_rss_bytes is not None
            else "n/a"
        )
        print(
            f"[optimize] adaptive_concurrency enabled=yes hard_cap={max_workers} "
            f"wave_chunk_size={args.wave_chunk_size} max_parallel_waves={max_parallel_waves} "
            f"bootstrap_rss_gb={bootstrap_gb}"
        )

    wall_started = time.perf_counter()
    executor_cm = (
        contextlib.nullcontext(None)
        if adaptive_enabled
        # A RecreatableExecutorPool, not a bare ProcessPoolExecutor: the
        # fixed-pool dispatch path needs to tear down and recreate this pool
        # on a candidate timeout, which a bare executor can't do.
        else RecreatableExecutorPool(max_workers=max_workers)
    )
    with executor_cm as executor:
        run_optimize_pilot_phase(
            executor=executor,
            context=run_context,
            tracker=tracker,
            vocab_schedule=vocab_schedule,
            pilot_seeds=pilot_seeds,
            min_frequencies=min_frequencies,
            seed_splits=seed_splits,
            effective_elbow_extra_steps=effective_elbow_extra_steps,
            run_candidate=run_optimize_candidate,
            concurrency_controller=concurrency_controller,
            memory_budget_ledger=memory_budget_ledger,
            wave_chunk_size=args.wave_chunk_size,
            max_parallel_waves=max_parallel_waves,
        )

        frontier = select_frontier(
            pilot_results=tracker.pilot_results,
            vocab_schedule=vocab_schedule,
            min_frequencies=min_frequencies,
            top_k=args.vocab_frontier_top_k,
            distance_tolerance=args.vocab_frontier_distance_tolerance,
            max_distance=candidate_policy.fertility_tolerance,
        )
        if not frontier:
            print("[optimize] No pilot frontier candidates found; skipping expansion.")
        else:
            frontier_tokens = sorted(
                frontier, key=lambda item: (item[0], vocab_sort_key(item[1]))
            )
            frontier_text = ", ".join(
                f"mf={min_freq}:v={'auto' if vocab == -1 else vocab}"
                for min_freq, vocab in frontier_tokens
            )
            print(f"[optimize] frontier combos -> {frontier_text}")

        total_seed_budget = len(seeds)
        combo_progress = initialize_frontier_progress(
            frontier=frontier,
            pilot_results=tracker.pilot_results,
            historical_expansion_results=historical_expansion_results,
        )
        pruned_combos = prune_combos(
            combo_progress=combo_progress,
            total_seed_budget=total_seed_budget,
            policy=candidate_policy,
            combo_prune_reason=combo_reason,
        )

        run_optimize_expansion_phase(
            executor=executor,
            context=run_context,
            tracker=tracker,
            expansion_seeds=expansion_seeds,
            frontier=frontier,
            seed_splits=seed_splits,
            pruned_combos=pruned_combos,
            combo_progress=combo_progress,
            total_seed_budget=total_seed_budget,
            combo_prune_reason=combo_reason,
            run_candidate=run_optimize_candidate,
            concurrency_controller=concurrency_controller,
            memory_budget_ledger=memory_budget_ledger,
            wave_chunk_size=args.wave_chunk_size,
            max_parallel_waves=max_parallel_waves,
            record_result=record_result,
        )

        # Refines inside pilot+expansion's winning bracket at a noise-derived
        # vocab step. getattr defaults keep a Namespace built without these
        # flags working; --refining-phase false skips it outright.
        if getattr(args, "refining_phase", True):
            run_optimize_refining_phase(
                executor=executor,
                context=run_context,
                tracker=tracker,
                seeds=seeds,
                frontier=frontier,
                seed_splits=seed_splits,
                run_log_dir=run_log_dir,
                run_log_path=run_log_path,
                eligibility_pass_rate=args.eligibility_pass_rate,
                vocab_size_distance_tolerance=args.vocab_size_distance_tolerance,
                run_candidate=run_optimize_candidate,
                concurrency_controller=concurrency_controller,
                memory_budget_ledger=memory_budget_ledger,
                wave_chunk_size=args.wave_chunk_size,
                max_parallel_waves=max_parallel_waves,
                refine_step_override=getattr(args, "refine_vocab_step", None),
                refine_step_minimum=getattr(args, "refine_vocab_step_minimum", 100),
            )

    if reuse_existing_candidates:
        print(
            f"[optimize] reuse skipped_existing_candidates_total={tracker.skipped_existing_candidates_total}"
        )

    run_log_df = merge_run_logs(
        run_log_dir=run_log_dir,
        merged_run_log_path=run_log_path,
    )

    scoped_history = scoped_history_for_trainer(
        run_log_df=run_log_df,
        scope=target.scope,
        systems_key=",".join(systems),
        trainer=args.tokenizer,
    )
    best_pair = select_best_pair(
        run_log_df=scoped_history,
        eligibility_pass_rate=args.eligibility_pass_rate,
        vocab_size_distance_tolerance=args.vocab_size_distance_tolerance,
        policy=candidate_policy,
    )
    would_promote = best_pair is not None
    print(
        f"[optimize] would_promote={'yes' if would_promote else 'no'} "
        f"finalize_enabled={'yes' if args.finalize else 'no'}"
    )
    # Tested against `best_pair` rather than `would_promote` so the `None` case
    # is narrowed away before `finalize_optimize_target_impl` below.
    if best_pair is None:
        print(
            "[optimize] No eligible parameter pair passed rejection and eligibility thresholds."
        )
        if not args.finalize:
            print(f"[optimize] shard complete -> {shard_run_log_path}")
        return

    if not args.finalize:
        print(f"[optimize] shard complete -> {shard_run_log_path}")
        return

    finalize_optimize_target_impl(
        args=args,
        roots=roots,
        target_scope=target.scope,
        systems=systems,
        artifact_paths=artifact_paths,
        trainer_options=trainer_options,
        tracker=tracker,
        best_pair=best_pair,
        final_tokenizer_path=final_tokenizer_path,
        run_log_path=run_log_path,
        summary_path=summary_path,
        session_id=session_id,
        wall_started=wall_started,
        effective_elbow_extra_steps=effective_elbow_extra_steps,
        corpus_rows=corpus_rows,
    )
