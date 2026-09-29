from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import polars as pl
from company_tokenize import TrainerOptions

from training import optimize_execution
from training.optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
)
from training.runtime import OptimizeExecutionTracker, OptimizeRunContext
from workspace.artifact_layout import tokenizer_scope_dir
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


def test_resolve_or_create_seed_split_artifacts_reuses_when_metadata_matches(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
):
    corpus_path = tmp_path / "training_corpus.parquet"
    # 40 rows (not 5) so a 20% validation split reliably lands on both sides
    # regardless of seed/hash outcome -- a handful of rows risks all of them
    # landing on the same side by chance with a real hash function.
    pl.DataFrame(
        {
            "system_uri": [f"ie://{i}" for i in range(40)],
            "name": [f"name{i}" for i in range(40)],
        }
    ).write_parquet(corpus_path)
    corpus_content_hash = optimize_execution.compute_corpus_content_hash(corpus_path)

    optimize_dir = tmp_path / "optimize"
    optimize_dir.mkdir(parents=True, exist_ok=True)

    first = optimize_execution.resolve_or_create_seed_split_artifacts(
        corpus_path=corpus_path,
        optimize_dir=optimize_dir,
        seed=42,
        validation_fraction=0.2,
        corpus_content_hash=corpus_content_hash,
    )
    second = optimize_execution.resolve_or_create_seed_split_artifacts(
        corpus_path=corpus_path,
        optimize_dir=optimize_dir,
        seed=42,
        validation_fraction=0.2,
        corpus_content_hash=corpus_content_hash,
    )

    assert first[4] is False
    assert second[4] is True


def test_resolve_or_create_seed_split_artifacts_writes_portable_corpus_path(
    workspace_roots: WorkspaceRoots,
):
    corpus_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "training_corpus.parquet"
    )
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"ie://{i}" for i in range(40)],
            "name": [f"name{i}" for i in range(40)],
        }
    ).write_parquet(corpus_path)
    corpus_content_hash = optimize_execution.compute_corpus_content_hash(corpus_path)

    optimize_dir = corpus_path.parent / "optimize"
    optimize_dir.mkdir(parents=True, exist_ok=True)

    optimize_execution.resolve_or_create_seed_split_artifacts(
        corpus_path=corpus_path,
        optimize_dir=optimize_dir,
        seed=42,
        validation_fraction=0.2,
        corpus_content_hash=corpus_content_hash,
        roots=workspace_roots,
    )

    metadata_path = optimize_dir / "split_seed_42.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert (
        metadata["corpus_path"]
        == "artifacts/tokenizers/work/ie/training_corpus.parquet"
    )


def test_prepare_optimize_seed_splits_aggregates_reused_and_created(mocker):
    resolver = mocker.patch.object(
        optimize_execution,
        "resolve_or_create_seed_split_artifacts",
        side_effect=[
            (Path("train1.parquet"), Path("val1.parquet"), 10, 3, True),
            (Path("train2.parquet"), Path("val2.parquet"), 9, 4, False),
        ],
    )

    seed_splits, reused, created = optimize_execution.prepare_optimize_seed_splits(
        seeds=[1, 2],
        corpus_path=Path("corpus.parquet"),
        optimize_dir=Path("optimize"),
        validation_fraction=0.2,
        corpus_content_hash="deadbeef",
    )

    assert resolver.call_count == 2
    assert reused == 1
    assert created == 1
    assert seed_splits[1]["train_rows"] == 10
    assert seed_splits[2]["validation_rows"] == 4


def test_prepare_optimize_target_execution_scopes_history_before_reuse(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    # Regression guard: run_log_dir is shared across trainers for the same
    # scope/system, so history must be scoped by trainer (and scope/systems)
    # before it's used for exclusion keys or reconstruction -- otherwise a
    # different trainer's row could wrongly satisfy this one's exclusion key.
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet",
        directory=tmp_path,
    )
    args = argparse.Namespace(
        tokenizer="wordpiece",
        profile="default",
        force=False,
        validation_fraction=0.2,
        vocab_size_distance_tolerance=0.01,
        pilot_seed_count=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        finalize=False,
    )
    trainer_options = SimpleNamespace(
        tokenizer_encoding="bpe", sp_character_coverage=1.0, sp_byte_fallback=False
    )
    common_kwargs: dict[str, Any] = {
        "args": args,
        "roots": workspace_roots,
        "target_scope": "country",
        "systems": ["ie"],
        "artifact_paths": artifact_paths,
        "trainer_options": trainer_options,
        "candidate_policy": _policy(),
        "seeds": [42],
        "vocab_schedule": [25000],
        "min_frequencies": [8],
        "effective_elbow_extra_steps": 0,
        "corpus_content_hash": "abc123",
        "session_id": "sess-1",
    }
    # A canary payload identical to what the real function will independently
    # compute makes reuse_existing_candidates=True without needing to know
    # the on-disk canary file path/format ahead of time.
    matching_canary = optimize_execution.build_optimize_canary_payload(
        target_scope="country",
        systems=["ie"],
        trainer="wordpiece",
        trainer_options=trainer_options,
        profile="default",
        vocab_schedule=[25000],
        min_frequencies=[8],
        seeds=[42],
        elbow_extra_steps=0,
        validation_fraction=0.2,
        policy=_policy(),
        vocab_size_distance_tolerance=0.01,
        pilot_seed_count=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        corpus_path=artifact_paths.corpus_path,
        corpus_content_hash="abc123",
        project_root=tmp_path,
    )
    mocker.patch.object(
        optimize_execution, "load_json_object", return_value=matching_canary
    )
    mixed_history = pl.DataFrame(
        {
            "trainer": ["wordpiece", "sentencepiece"],
            "scope": ["country", "country"],
            "systems_key": ["ie", "ie"],
            "seed": [42, 42],
            "vocab_size_requested": [25000, 25000],
            "vocab_size_resolved": [25000, 25000],
            "min_frequency": [8, 1],
            "rejected": [False, False],
            "fertility_distance": [0.0068, 0.02],
            "selection_score": [2.1, 3.0],
        }
    )
    mocker.patch.object(
        optimize_execution, "merge_run_logs", return_value=mixed_history
    )

    execution_setup = optimize_execution.prepare_optimize_target_execution(
        **common_kwargs
    )

    assert execution_setup is not None
    assert execution_setup["reuse_existing_candidates"] is True
    assert execution_setup["excluded_candidate_keys"] == {(42, 25000, 8)}
    assert len(execution_setup["historical_results"]) == 1
    assert execution_setup["historical_results"][0].trainer == "wordpiece"
    assert execution_setup["historical_results"][0].min_frequency == 8


def test_prepare_optimize_target_execution_different_model_types_use_disjoint_dirs(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
):
    # A bpe sweep and a unigram sweep on the same corpus must never collide
    # or need to purge each other -- they should land in structurally
    # different directories (see resolve_optimize_sweep_paths) rather than
    # needing a canary comparison to decide whether to reuse or refuse.
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet",
        directory=tmp_path,
    )
    args = argparse.Namespace(
        tokenizer="sentencepiece",
        profile="default",
        force=False,
        optimize_wipe=False,
        validation_fraction=0.2,
        vocab_size_distance_tolerance=0.01,
        pilot_seed_count=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        finalize=False,
    )
    common_kwargs: dict[str, Any] = {
        "args": args,
        "roots": workspace_roots,
        "target_scope": "country",
        "systems": ["ie"],
        "artifact_paths": artifact_paths,
        "candidate_policy": _policy(),
        "seeds": [42],
        "vocab_schedule": [25000],
        "min_frequencies": [1],
        "effective_elbow_extra_steps": 0,
        "corpus_content_hash": "abc123",
        "session_id": "sess-1",
    }

    bpe_setup = optimize_execution.prepare_optimize_target_execution(
        trainer_options=TrainerOptions(tokenizer_encoding="bpe"), **common_kwargs
    )
    unigram_setup = optimize_execution.prepare_optimize_target_execution(
        trainer_options=TrainerOptions(tokenizer_encoding="unigram"), **common_kwargs
    )

    assert bpe_setup is not None
    assert unigram_setup is not None
    assert bpe_setup["models_dir"] != unigram_setup["models_dir"]
    assert bpe_setup["run_log_dir"] != unigram_setup["run_log_dir"]
    assert bpe_setup["final_tokenizer_path"] != unigram_setup["final_tokenizer_path"]

    # Write a candidate into each and force-purge the bpe one -- the
    # unigram sweep's candidate must survive untouched.
    bpe_setup["models_dir"].mkdir(parents=True, exist_ok=True)
    unigram_setup["models_dir"].mkdir(parents=True, exist_ok=True)
    (bpe_setup["models_dir"] / "candidate.model").write_text("x", encoding="utf-8")
    (unigram_setup["models_dir"] / "candidate.model").write_text("x", encoding="utf-8")

    force_args = argparse.Namespace(**{**vars(args), "force": True})
    optimize_execution.prepare_optimize_target_execution(
        trainer_options=TrainerOptions(tokenizer_encoding="bpe"),
        **{**common_kwargs, "args": force_args},
    )

    assert not (bpe_setup["models_dir"] / "candidate.model").exists()
    assert (unigram_setup["models_dir"] / "candidate.model").exists()


def test_prepare_optimize_target_execution_threads_bootstrap_rss_bytes_from_history(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet",
        directory=tmp_path,
    )
    args = argparse.Namespace(
        tokenizer="wordpiece",
        profile="default",
        force=False,
        validation_fraction=0.2,
        vocab_size_distance_tolerance=0.01,
        pilot_seed_count=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        finalize=False,
    )
    trainer_options = SimpleNamespace(
        tokenizer_encoding="bpe", sp_character_coverage=1.0, sp_byte_fallback=False
    )
    common_kwargs: dict[str, Any] = {
        "args": args,
        "roots": workspace_roots,
        "target_scope": "country",
        "systems": ["ie"],
        "artifact_paths": artifact_paths,
        "trainer_options": trainer_options,
        "candidate_policy": _policy(),
        "seeds": [42],
        "vocab_schedule": [25000],
        "min_frequencies": [8],
        "effective_elbow_extra_steps": 0,
        "corpus_content_hash": "abc123",
        "session_id": "sess-1",
    }
    matching_canary = optimize_execution.build_optimize_canary_payload(
        target_scope="country",
        systems=["ie"],
        trainer="wordpiece",
        trainer_options=trainer_options,
        profile="default",
        vocab_schedule=[25000],
        min_frequencies=[8],
        seeds=[42],
        elbow_extra_steps=0,
        validation_fraction=0.2,
        policy=_policy(),
        vocab_size_distance_tolerance=0.01,
        pilot_seed_count=1,
        vocab_frontier_top_k=1,
        vocab_frontier_distance_tolerance=0.0,
        corpus_path=artifact_paths.corpus_path,
        corpus_content_hash="abc123",
        project_root=tmp_path,
    )
    mocker.patch.object(
        optimize_execution, "load_json_object", return_value=matching_canary
    )
    # Two wordpiece rows with different worker_rss_bytes -- bootstrap should
    # pick the max, not the first/last.
    history_with_rss = pl.DataFrame(
        {
            "trainer": ["wordpiece", "wordpiece"],
            "scope": ["country", "country"],
            "systems_key": ["ie", "ie"],
            "seed": [42, 43],
            "vocab_size_requested": [25000, 25000],
            "vocab_size_resolved": [25000, 25000],
            "min_frequency": [8, 8],
            "rejected": [False, False],
            "fertility_distance": [0.0068, 0.0072],
            "selection_score": [2.1, 2.2],
            "worker_rss_bytes": [2_000_000_000, 3_500_000_000],
        }
    )
    mocker.patch.object(
        optimize_execution, "merge_run_logs", return_value=history_with_rss
    )

    execution_setup = optimize_execution.prepare_optimize_target_execution(
        **common_kwargs
    )

    assert execution_setup is not None
    assert execution_setup["bootstrap_rss_bytes"] == 3_500_000_000


def test_finalize_optimize_target_writes_summary_and_metadata_with_mocks(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"name": ["alpha", "beta"]}).write_parquet(corpus_path)
    artifact_paths = SimpleNamespace(
        corpus_path=corpus_path,
        directory=tmp_path,
    )
    final_tokenizer_path = tmp_path / "final" / "wordpiece.json"
    final_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    final_tokenizer_path.write_text("model", encoding="utf-8")
    run_log_path = tmp_path / "run_log.parquet"
    summary_path = tmp_path / "summary.json"
    trainer_options = TrainerOptions()
    # A deliberate stub: `finalize_optimize_target` reads only these
    # attributes off the tracker.
    tracker = cast(
        "OptimizeExecutionTracker",
        SimpleNamespace(
            candidate_results=[
                _result(
                    seed=1, vocab=10000, min_frequency=1, distance=0.01, rejected=False
                )
            ],
            completed_jobs=1,
            submitted_jobs=1,
        ),
    )
    args = argparse.Namespace(
        tokenizer="wordpiece",
        diagnostic_top_n=10,
        fertility_target=1.25,
        profile="default",
        name_col="name",
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0,
        unk_rate_threshold=0.005,
        delete_rejected_models=True,
        fertility_tolerance=0.1,
        fertility_min=1.0,
        fertility_max=2.0,
        token_count_median_max=3.0,
        token_count_p95_max=4.0,
        elbow_min_points=3,
        elbow_min_improvement=0.002,
        expansion_strategy="heuristic",
        # Set by apply_optimize_preset in the real CLI pipeline
        # before finalize_optimize_target runs; set directly here since this
        # test drives finalize_optimize_target standalone.
        optimize_preset="wordpiece",
        optimize_preset_overrides={},
    )
    best_pair = {
        "vocab_size_requested": 10000,
        "min_frequency": 1,
        "selection_vocab_size": 10000,
        "median_fertility_distance": 0.02,
        "median_selection_score": 0.15,
        "pass_rate": 1.0,
    }

    mocker.patch.object(
        optimize_execution, "train_tokenizer_with_trainer", return_value=12345
    )
    mocker.patch.object(
        optimize_execution,
        "evaluate_tokenizer_metrics",
        return_value=(
            {
                "fertility": 1.2,
                "unk_rate": 0.001,
                "fertility_distance": 0.01,
                "single_char_token_pct": 0.0,
                "token_count_mean": 1.0,
                "token_count_median": 1.0,
                "token_count_p95": 2.0,
            },
            {"fr": 2},
            [],
            [],
        ),
    )

    optimize_execution.finalize_optimize_target(
        args=args,
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        artifact_paths=artifact_paths,
        trainer_options=trainer_options,
        tracker=tracker,
        best_pair=best_pair,
        final_tokenizer_path=final_tokenizer_path,
        run_log_path=run_log_path,
        summary_path=summary_path,
        session_id="sess-1",
        wall_started=0.0,
        effective_elbow_extra_steps=0,
        corpus_rows=2,
        write_token_score_artifact_func=lambda **kwargs: {
            "status": "ok",
            "invoked": True,
        },
        build_summary_payload=lambda **kwargs: {
            "ok": True,
            "session_id": kwargs["session_id"],
        },
        resolve_metadata_path_for_trainer_func=lambda **kwargs: (
            artifact_paths.directory / "wordpiece" / "metadata.json"
        ),
        trainer_params_payload_builder=lambda **kwargs: {"trainer": kwargs["trainer"]},
        copy_file=lambda src, dst: dst.write_text("copied", encoding="utf-8"),
    )

    assert summary_path.exists()
    assert optimize_execution.load_json_object(summary_path) == {
        "ok": True,
        "session_id": "sess-1",
    }
    metadata = optimize_execution.load_json_object(
        artifact_paths.directory / "metadata.json"
    )
    assert metadata is not None
    assert metadata["mode"] == "optimize"

    # The promoted candidate carries what it measured and the summary of the
    # optimize run that chose it, so a reader holding its reference needs no
    # other lookup.
    from company_tokenize import tokenizer_directory_files

    from workspace.tokenizer_store import promoted_tokenizer_directory

    held = tokenizer_directory_files(
        promoted_tokenizer_directory(
            workspace_roots, system="fr", tokenizer_id="wordpiece"
        ),
        trainer="wordpiece",
    )
    held_metrics = optimize_execution.load_json_object(held.metrics)
    assert held_metrics is not None
    assert held_metrics["fertility_distance"] == 0.01
    assert optimize_execution.load_json_object(held.optimize_summary) == {
        "ok": True,
        "session_id": "sess-1",
    }

    from company_tokenize.manifest import (
        RUN_MANIFEST_MANDATORY_FIELDS,
        RUN_MANIFEST_OPTIONAL_FIELDS,
    )

    for field in RUN_MANIFEST_MANDATORY_FIELDS:
        assert field in metadata, f"missing mandatory manifest field: {field}"
    # optimize mode is the only mode that points back at a sweep run log and
    # summary, and args carries a resolved preset (as the real CLI pipeline
    # always sets before finalize runs) -- every optional field should be
    # present here.
    for field in RUN_MANIFEST_OPTIONAL_FIELDS:
        assert field in metadata, f"optimize manifest missing optional field: {field}"
    assert metadata["optimize_preset"] == "wordpiece"
    assert metadata["optimize_preset_overrides"] == {}

    # `corpus_content_hash` proves what corpus the promoted tokenizer was
    # actually trained on; it must match a hash computed independently from
    # the same corpus file, not just be present.
    assert metadata[
        "corpus_content_hash"
    ] == optimize_execution.compute_corpus_content_hash(corpus_path)

    # The country-scope wordpiece copy target resolves through
    # `workspace.artifact_layout.tokenizer_scope_dir` rather than composing
    # `project_root / "artifacts" / "tokenizers" / system` inline; it must
    # still land at the same place as before the migration.
    easy_result_path = (
        tokenizer_scope_dir(workspace_roots, system="fr") / "wordpiece" / "model.json"
    )
    assert easy_result_path.exists()
    assert easy_result_path.read_text(encoding="utf-8") == "copied"


def test_finalize_optimize_target_honors_injected_build_run_manifest_func(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"name": ["alpha", "beta"]}).write_parquet(corpus_path)
    artifact_paths = SimpleNamespace(
        corpus_path=corpus_path,
        directory=tmp_path,
    )
    final_tokenizer_path = tmp_path / "final" / "wordpiece.json"
    final_tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    final_tokenizer_path.write_text("model", encoding="utf-8")
    run_log_path = tmp_path / "run_log.parquet"
    summary_path = tmp_path / "summary.json"
    trainer_options = TrainerOptions()
    # A deliberate stub: `finalize_optimize_target` reads only these
    # attributes off the tracker.
    tracker = cast(
        "OptimizeExecutionTracker",
        SimpleNamespace(
            candidate_results=[
                _result(
                    seed=1, vocab=10000, min_frequency=1, distance=0.01, rejected=False
                )
            ],
            completed_jobs=1,
            submitted_jobs=1,
        ),
    )
    args = argparse.Namespace(
        tokenizer="wordpiece",
        diagnostic_top_n=10,
        fertility_target=1.25,
        profile="default",
        name_col="name",
        eligibility_pass_rate=0.8,
        vocab_size_distance_tolerance=0,
        unk_rate_threshold=0.005,
        delete_rejected_models=True,
        fertility_tolerance=0.1,
        fertility_min=1.0,
        fertility_max=2.0,
        token_count_median_max=3.0,
        token_count_p95_max=4.0,
        elbow_min_points=3,
        elbow_min_improvement=0.002,
        expansion_strategy="heuristic",
    )
    best_pair = {
        "vocab_size_requested": 10000,
        "min_frequency": 1,
        "selection_vocab_size": 10000,
        "median_fertility_distance": 0.02,
        "median_selection_score": 0.15,
        "pass_rate": 1.0,
    }

    mocker.patch.object(
        optimize_execution, "train_tokenizer_with_trainer", return_value=12345
    )
    mocker.patch.object(
        optimize_execution,
        "evaluate_tokenizer_metrics",
        return_value=(
            {
                "fertility": 1.2,
                "unk_rate": 0.001,
                "fertility_distance": 0.01,
                "single_char_token_pct": 0.0,
                "token_count_mean": 1.0,
                "token_count_median": 1.0,
                "token_count_p95": 2.0,
            },
            {"fr": 2},
            [],
            [],
        ),
    )

    manifest_calls: list[dict] = []

    def fake_build_run_manifest(**kwargs):
        manifest_calls.append(kwargs)
        return {"fake_manifest": True, "mode": kwargs["mode"]}

    optimize_execution.finalize_optimize_target(
        args=args,
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        artifact_paths=artifact_paths,
        trainer_options=trainer_options,
        tracker=tracker,
        best_pair=best_pair,
        final_tokenizer_path=final_tokenizer_path,
        run_log_path=run_log_path,
        summary_path=summary_path,
        session_id="sess-1",
        wall_started=0.0,
        effective_elbow_extra_steps=0,
        corpus_rows=2,
        write_token_score_artifact_func=lambda **kwargs: {
            "status": "ok",
            "invoked": True,
        },
        build_summary_payload=lambda **kwargs: {
            "ok": True,
            "session_id": kwargs["session_id"],
        },
        resolve_metadata_path_for_trainer_func=lambda **kwargs: (
            artifact_paths.directory / "wordpiece" / "metadata.json"
        ),
        trainer_params_payload_builder=lambda **kwargs: {"trainer": kwargs["trainer"]},
        copy_file=lambda src, dst: dst.write_text("copied", encoding="utf-8"),
        build_run_manifest_func=fake_build_run_manifest,
    )

    assert len(manifest_calls) == 1
    assert manifest_calls[0]["mode"] == "optimize"
    metadata = optimize_execution.load_json_object(
        artifact_paths.directory / "metadata.json"
    )
    assert metadata == {"fake_manifest": True, "mode": "optimize"}


def test_plain_training_reuses_the_naive_already_stored_for_its_corpus_and_settings(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """One corpus and one set of settings give one `naive`: a second run finds
    the stored candidate, points `naive` at it and trains nothing, while a
    different size or an empty store leaves training to go ahead."""
    from company_tokenize import tokenizer_directory_files

    from training.promotion import publish_naive_candidate
    from workspace.pointer import promoted_reference
    from workspace.tokenizer_store import tokenizer_selection

    work = tmp_path / "work"
    work.mkdir()
    files = tokenizer_directory_files(work, trainer="wordpiece")
    files.model.write_text('{"model": 1}', encoding="utf-8")
    files.metadata.write_text("{}", encoding="utf-8")
    corpus = work / "training_corpus.parquet"
    pl.DataFrame({"name": ["alpha ltd", "beta gmbh"]}).write_parquet(corpus)
    trainer_options = optimize_execution.TrainerOptions()
    args = argparse.Namespace(tokenizer="wordpiece", vocab_size=40000, min_frequency=1)

    def _reuse(run_args: argparse.Namespace) -> bool:
        return optimize_execution.reuse_stored_naive(
            args=run_args,
            roots=workspace_roots,
            system_code="ie",
            corpus_path=corpus,
            trainer_options=trainer_options,
        )

    assert _reuse(args) is False
    stored, _ = publish_naive_candidate(
        roots=workspace_roots,
        system="ie",
        trainer="wordpiece",
        tokenizer_encoding=None,
        model_path=files.model,
        metadata_path=files.metadata,
        token_scores_path=None,
        corpus_path=corpus,
        vocab_size=39500,
        min_frequency=1,
        metrics={"fertility_distance": 0.01},
        parameters=optimize_execution._naive_settings(
            args=args, trainer_options=trainer_options
        ),
        invocation=["train_tokenizer.py", "--systems", "ie"],
    )

    assert _reuse(args) is True
    selection = tokenizer_selection(system="ie", tokenizer_id="wordpiece")
    assert promoted_reference(workspace_roots, selection, profile="naive") == stored
    assert _reuse(argparse.Namespace(**{**vars(args), "vocab_size": 30000})) is False


def test_finalize_train_target_stores_its_measured_metrics_and_no_optimize_summary(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
):
    # Plain --mode train chooses among no candidates, so it measures its
    # tokenizer itself before storing it, and hands over no optimize summary.
    artifact_paths = SimpleNamespace(
        corpus_path=tmp_path / "corpus.parquet",
        directory=tmp_path,
    )
    tokenizer_path = tmp_path / "wordpiece.json"
    tokenizer_path.write_text("model", encoding="utf-8")
    trainer_options = SimpleNamespace(tokenizer_encoding="bpe")
    args = argparse.Namespace(
        tokenizer="wordpiece", name_col="name", vocab_size=40000, min_frequency=1
    )

    measure_calls: list[dict] = []
    publish_calls: list[dict] = []
    measured = {"fertility_distance": 0.01}

    def fake_evaluate_tokenizer_metrics(**kwargs):
        measure_calls.append(kwargs)
        return measured, [], [], []

    naive = SimpleNamespace(uri="tokenizer://fr/wordpiece/k")

    def fake_publish_naive_candidate(**kwargs):
        publish_calls.append(kwargs)
        return naive, None

    optimize_execution.finalize_train_target(
        args=args,
        roots=workspace_roots,
        artifact_paths=artifact_paths,
        trainer_options=trainer_options,
        scope="country",
        systems=["fr"],
        profile=None,
        tokenizer_path=tokenizer_path,
        line_count=2,
        resolved_vocab_size=40000,
        system_code="fr",
        resolve_metadata_path_for_trainer_func=lambda **kwargs: (
            artifact_paths.directory / "wordpiece" / "metadata.json"
        ),
        trainer_params_payload_builder=lambda **kwargs: {"trainer": kwargs["trainer"]},
        evaluate_tokenizer_metrics_func=fake_evaluate_tokenizer_metrics,
        publish_naive_candidate_func=fake_publish_naive_candidate,
    )

    assert len(publish_calls) == 1
    assert publish_calls[0]["system"] == "fr"
    assert publish_calls[0]["model_path"] == tokenizer_path
    assert publish_calls[0]["vocab_size"] == 40000
    assert publish_calls[0]["parameters"]["mode"] == "train"
    assert len(measure_calls) == 1
    assert measure_calls[0]["tokenizer_path"] == tokenizer_path
    assert measure_calls[0]["corpus_path"] == artifact_paths.corpus_path
    assert publish_calls[0]["metrics"] == measured
    assert publish_calls[0].get("optimize_summary") is None

    from company_tokenize.manifest import (
        RUN_MANIFEST_MANDATORY_FIELDS,
        RUN_MANIFEST_OPTIONAL_FIELDS,
    )

    metadata = optimize_execution.load_json_object(
        artifact_paths.directory / "metadata.json"
    )
    assert metadata is not None
    assert metadata["mode"] == "train"
    for field in RUN_MANIFEST_MANDATORY_FIELDS:
        assert field in metadata, f"missing mandatory manifest field: {field}"
    # plain --mode train has no sweep run log/summary to point at -- both
    # optimize-only optional fields should be absent, not written as null.
    for field in RUN_MANIFEST_OPTIONAL_FIELDS:
        assert field not in metadata, f"train manifest has optimize-only field: {field}"
    # `artifact_paths.corpus_path` is never created by this stub fixture, so
    # the missing-corpus sentinel is what should land in the manifest.
    assert metadata["corpus_content_hash"] == "missing"
