from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import polars as pl
from company_tokenize import OptimizeSweepPaths, TrainerOptions

from training import optimize_execution
from training.optimize_concurrency import (
    AdaptiveConcurrencyController,
    MemoryBudgetLedger,
)
from training.runtime import OptimizeExecutionTracker, OptimizeRunContext
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
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


def test_is_final_result_fresh_true_when_outputs_newer_than_corpus(tmp_path: Path):
    corpus_path = tmp_path / "training_corpus.parquet"
    tokenizer_path = tmp_path / "wordpiece.json"
    summary_path = tmp_path / "optimize_summary.json"

    corpus_path.write_text("c", encoding="utf-8")
    tokenizer_path.write_text("t", encoding="utf-8")
    summary_path.write_text("s", encoding="utf-8")

    corpus_ns = 1_000_000_000
    tokenizer_ns = 2_000_000_000
    summary_ns = 3_000_000_000
    corpus_path.touch()
    tokenizer_path.touch()
    summary_path.touch()

    import os

    os.utime(corpus_path, ns=(corpus_ns, corpus_ns))
    os.utime(tokenizer_path, ns=(tokenizer_ns, tokenizer_ns))
    os.utime(summary_path, ns=(summary_ns, summary_ns))

    assert (
        optimize_execution.is_final_result_fresh(
            corpus_path=corpus_path,
            final_tokenizer_path=tokenizer_path,
            summary_path=summary_path,
        )
        is True
    )


def test_is_final_result_fresh_false_when_a_trial_was_logged_after_the_summary(
    tmp_path: Path,
):
    import os

    corpus_path = tmp_path / "training_corpus.parquet"
    tokenizer_path = tmp_path / "wordpiece.json"
    summary_path = tmp_path / "optimize_summary.json"
    run_log_dir = tmp_path / "run_logs"
    run_log_dir.mkdir()
    shard_path = run_log_dir / "sess-2.parquet"
    for path, mtime_ns in (
        (corpus_path, 1_000_000_000),
        (tokenizer_path, 2_000_000_000),
        (summary_path, 3_000_000_000),
        (shard_path, 2_500_000_000),
    ):
        path.write_text("x", encoding="utf-8")
        os.utime(path, ns=(mtime_ns, mtime_ns))
    paths = {
        "corpus_path": corpus_path,
        "final_tokenizer_path": tokenizer_path,
        "summary_path": summary_path,
        "run_log_dir": run_log_dir,
    }

    assert optimize_execution.is_final_result_fresh(**paths) is True

    os.utime(shard_path, ns=(4_000_000_000, 4_000_000_000))

    assert optimize_execution.is_final_result_fresh(**paths) is False


def test_summary_winner_is_reselected_only_while_selection_still_picks_it(
    tmp_path: Path,
):
    # vocab 200000 was the recorded winner; under the current gates vocab
    # 40000 is eligible and closer to the target, so selection has moved.
    run_log_dir = tmp_path / "run_logs"
    run_log_dir.mkdir()
    pl.DataFrame(
        {
            "scope": ["country"] * 2,
            "systems_key": ["de"] * 2,
            "trainer": ["wordpiece"] * 2,
            "vocab_size_requested": [40000, 200000],
            "vocab_size_resolved": [40000, 200000],
            "min_frequency": [1, 1],
            "rejected": [True, False],
            "rejection_reason": ["token_count_median", ""],
            "selection_score": [9.0, 1.0],
            "fertility": [1.25, 1.14],
            "fertility_distance": [0.003, 0.11],
            "train_fertility_distance": [0.003, 0.11],
            "unk_rate": [0.0, 0.0],
            "token_count_median": [5.0, 4.0],
            "token_count_p95": [10.0, 9.0],
            "fertility_generalization_delta": [0.0, 0.0],
            "unk_rate_generalization_delta": [0.0, 0.0],
        }
    ).write_parquet(run_log_dir / "sess-1.parquet")
    summary_path = tmp_path / "optimize_summary.json"

    def reselected(winner_vocab: int) -> bool:
        summary_path.write_text(
            '{"winner_candidate": {"vocab_size_requested": '
            f'{winner_vocab}, "min_frequency": 1}}}}',
            encoding="utf-8",
        )
        return optimize_execution.summary_winner_is_reselected(
            summary_path=summary_path,
            run_log_dir=run_log_dir,
            run_log_path=tmp_path / "run_log.parquet",
            target_scope="country",
            systems=["de"],
            trainer="wordpiece",
            eligibility_pass_rate=0.8,
            vocab_size_distance_tolerance=0.01,
            policy=_policy(),  # type: ignore[arg-type]
        )

    assert reselected(200000) is False
    assert reselected(40000) is True


def test_emit_optimize_dry_run_plan_reports_paths_and_counts(
    tmp_path: Path, workspace_roots: WorkspaceRoots, capsys
):
    args = argparse.Namespace(tokenizer="wordpiece")
    final_tokenizer_path = tmp_path / "out" / "wordpiece.json"

    optimize_execution.emit_optimize_dry_run_plan(
        args=args,
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        final_tokenizer_path=final_tokenizer_path,
        models_dir=tmp_path
        / "optimize"
        / "abcd1234"
        / "wordpiece"
        / "default"
        / "efgh5678"
        / "models",
        vocab_schedule=[10000, -1],
        min_frequencies=[1, 2],
        seeds=[1, 2],
        pilot_seeds=[1],
        expansion_seeds=[2],
        total_candidate_jobs=8,
    )

    out = capsys.readouterr().out
    assert "[optimize] dry_run scope=country systems=fr" in out
    assert "[optimize] dry_run upper_bound_jobs_no_pruning=8" in out
    assert "[optimize] dry_run model_output_paths_total=8" in out


def test_emit_optimize_dry_run_plan_reports_family_split_input_file_count(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
):
    """`planned_input_parquet_files` must count real cleansed output under
    `primary/`, not only the pre-split top-level layout other tests in this
    file exercise.
    """
    partition_dir = (
        layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME)
        / "primary"
        / "jurisdiction_code=fr"
    )
    partition_dir.mkdir(parents=True)
    (partition_dir / "part-00001.parquet").write_text("x", encoding="utf-8")

    optimize_execution.emit_optimize_dry_run_plan(
        args=argparse.Namespace(tokenizer="wordpiece"),
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        final_tokenizer_path=tmp_path / "out" / "wordpiece.json",
        models_dir=tmp_path
        / "optimize"
        / "abcd1234"
        / "wordpiece"
        / "default"
        / "efgh5678"
        / "models",
        vocab_schedule=[10000],
        min_frequencies=[1],
        seeds=[1],
        pilot_seeds=[1],
        expansion_seeds=[],
        total_candidate_jobs=1,
    )

    out = capsys.readouterr().out
    assert "[optimize] dry_run planned_input_parquet_files=1" in out


def test_load_json_object_covers_edge_cases(tmp_path: Path):
    missing = optimize_execution.load_json_object(tmp_path / "missing.json")
    assert missing is None

    valid_path = tmp_path / "valid.json"
    valid_path.write_text('{"a": 1}', encoding="utf-8")
    assert optimize_execution.load_json_object(valid_path) == {"a": 1}

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text('{"a":', encoding="utf-8")
    assert optimize_execution.load_json_object(invalid_path) is None

    non_object_path = tmp_path / "list.json"
    non_object_path.write_text("[1,2]", encoding="utf-8")
    assert optimize_execution.load_json_object(non_object_path) is None


def test_resolve_optimize_corpus_for_target_reuse_and_force_paths(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"name": ["alpha", "beta"]}).write_parquet(corpus_path)

    input_file = layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME) / "fr-1.parquet"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.write_text("x", encoding="utf-8")

    # Make corpus newer than input so reuse branch is selected.
    input_ns = 1_000_000_000
    corpus_ns = 2_000_000_000
    import os

    os.utime(input_file, ns=(input_ns, input_ns))
    os.utime(corpus_path, ns=(corpus_ns, corpus_ns))

    args_reuse = argparse.Namespace(force=False, name_col="name", optimize_wipe=False)
    result_reuse = optimize_execution.resolve_optimize_corpus_for_target(
        args=args_reuse,
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        corpus_path=corpus_path,
        input_parquet_files=[input_file],
        rebuild_optimize_corpus=lambda **kwargs: 999,
    )
    assert result_reuse is not None
    assert result_reuse[0] == 2

    args_force = argparse.Namespace(force=True, name_col="name", optimize_wipe=False)
    result_force = optimize_execution.resolve_optimize_corpus_for_target(
        args=args_force,
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        corpus_path=corpus_path,
        input_parquet_files=[input_file],
        rebuild_optimize_corpus=lambda **kwargs: 12,
    )
    assert result_force is not None
    assert result_force[0] == 12


def test_write_token_score_artifact_skips_when_tokenizer_missing(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    corpus_path.write_text("x", encoding="utf-8")

    result = optimize_execution.write_token_score_artifact(
        corpus_path=corpus_path,
        tokenizer_path=tmp_path / "missing_tokenizer.json",
        trainer="wordpiece",
    )

    assert result == {
        "invoked": False,
        "status": "skipped_missing_tokenizer",
        "corpus_path": str(corpus_path),
        "tokenizer_path": str(tmp_path / "missing_tokenizer.json"),
        "trainer": "wordpiece",
    }


def test_has_cleansed_parquet_true_and_false(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    assert optimize_execution.has_cleansed_parquet(workspace_roots, "fr") is False

    cleansed_dir = layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME)
    cleansed_dir.mkdir(parents=True)
    assert optimize_execution.has_cleansed_parquet(workspace_roots, "fr") is False

    partition_dir = cleansed_dir / "jurisdiction_code=fr"
    partition_dir.mkdir(parents=True)
    (partition_dir / "part-00001.parquet").write_text("x", encoding="utf-8")
    assert optimize_execution.has_cleansed_parquet(workspace_roots, "fr") is True


def test_has_cleansed_parquet_true_for_family_split_primary_directory(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    """A real cleansed/ view lives under `primary/` -- must be found there,
    not only in the pre-split top-level layout the test above exercises.
    """
    partition_dir = (
        layer_fixture_dir("de", layer=CLEANSED_LAYER_NAME)
        / "primary"
        / "jurisdiction_code=de"
    )
    partition_dir.mkdir(parents=True)
    (partition_dir / "part-00001.parquet").write_text("x", encoding="utf-8")
    assert optimize_execution.has_cleansed_parquet(workspace_roots, "de") is True


def _sweep_paths_at(root: Path) -> OptimizeSweepPaths:
    return OptimizeSweepPaths(
        sweep_dir=root,
        models_dir=root / "models",
        run_log_dir=root / "run_logs",
        run_log_path=root / "optimize_runs.parquet",
        summary_path=root / "optimize_summary.json",
        canary_sidecar_path=root / "canary.json",
    )


def test_purge_sweep_candidates_missing_dirs_returns_zero(tmp_path: Path):
    removed_models, removed_rows = optimize_execution.purge_sweep_candidates(
        sweep_paths=_sweep_paths_at(tmp_path / "missing_sweep"),
    )
    assert (removed_models, removed_rows) == (0, 0)


def test_purge_sweep_candidates_wipes_models_and_run_log_shards(tmp_path: Path):
    sweep_paths = _sweep_paths_at(tmp_path / "sweep")
    sweep_paths.models_dir.mkdir(parents=True)
    sweep_paths.run_log_dir.mkdir(parents=True)
    model_file = sweep_paths.models_dir / "sentencepiece_seed_42_v10000_mf1.model"
    model_file.write_text("x", encoding="utf-8")
    shard_path = sweep_paths.run_log_dir / "opt-shard.parquet"
    pl.DataFrame({"seed": [42, 43]}).write_parquet(shard_path)
    sweep_paths.run_log_path.write_text("{}", encoding="utf-8")
    sweep_paths.summary_path.write_text("{}", encoding="utf-8")

    removed_models, removed_rows = optimize_execution.purge_sweep_candidates(
        sweep_paths=sweep_paths,
    )

    assert (removed_models, removed_rows) == (1, 2)
    assert not model_file.exists()
    assert not shard_path.exists()
    assert not sweep_paths.run_log_path.exists()
    assert not sweep_paths.summary_path.exists()


def test_purge_sweep_candidates_does_not_touch_a_sibling_sweep_dir(tmp_path: Path):
    # A leaf directory is unique to one (corpus, trainer, model-variant,
    # grid) configuration by construction (see resolve_optimize_sweep_paths),
    # so purging one sweep must never reach into a sibling leaf -- unlike the
    # old flat layout, this isn't a scoped filter, it's just a different dir.
    bpe_paths = _sweep_paths_at(
        tmp_path / "corpushash" / "sentencepiece" / "bpe" / "grid1"
    )
    unigram_paths = _sweep_paths_at(
        tmp_path / "corpushash" / "sentencepiece" / "unigram" / "grid1"
    )
    for sweep_paths in (bpe_paths, unigram_paths):
        sweep_paths.models_dir.mkdir(parents=True)
        (sweep_paths.models_dir / "candidate.model").write_text("x", encoding="utf-8")

    optimize_execution.purge_sweep_candidates(sweep_paths=bpe_paths)

    assert not (bpe_paths.models_dir / "candidate.model").exists()
    assert (unigram_paths.models_dir / "candidate.model").exists()


def test_is_final_result_fresh_false_when_corpus_missing(tmp_path: Path):
    assert (
        optimize_execution.is_final_result_fresh(
            corpus_path=tmp_path / "missing_corpus.parquet",
            final_tokenizer_path=tmp_path / "final.json",
            summary_path=tmp_path / "summary.json",
        )
        is False
    )


def test_list_cleansed_parquet_files_and_latest_mtime(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    fr_dir = layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME) / "jurisdiction_code=fr"
    de_dir = layer_fixture_dir("de", layer=CLEANSED_LAYER_NAME) / "jurisdiction_code=de"
    fr_dir.mkdir(parents=True)
    de_dir.mkdir(parents=True)
    fr_file = fr_dir / "part-00001.parquet"
    de_file = de_dir / "part-00001.parquet"
    fr_file.write_text("x", encoding="utf-8")
    de_file.write_text("y", encoding="utf-8")

    files = optimize_execution.list_cleansed_parquet_files(
        roots=workspace_roots, systems=["fr", "de"]
    )
    assert sorted(files) == sorted([fr_file, de_file])
    assert optimize_execution.latest_mtime_ns([]) == 0
    assert optimize_execution.latest_mtime_ns(files) > 0


def test_list_cleansed_parquet_files_finds_family_split_primary_directories(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    """A real cleansed/ view lives under `primary/`, not directly at the
    layer's own top level -- the old hand-built PARTITION_PREFIX glob only
    found the pre-split shape the other test above exercises.
    """
    fr_dir = (
        layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME)
        / "primary"
        / "jurisdiction_code=fr"
    )
    fr_dir.mkdir(parents=True)
    fr_file = fr_dir / "part-00001.parquet"
    fr_file.write_text("x", encoding="utf-8")

    files = optimize_execution.list_cleansed_parquet_files(
        roots=workspace_roots, systems=["fr"]
    )
    assert files == [fr_file]


def test_emit_optimize_dry_run_plan_reports_global_scope_input_files(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir, capsys
):
    args = argparse.Namespace(tokenizer="wordpiece")
    for system in ["fr", "de"]:
        partition_dir = (
            layer_fixture_dir(system, layer=CLEANSED_LAYER_NAME)
            / f"jurisdiction_code={system}"
        )
        partition_dir.mkdir(parents=True)
        (partition_dir / "part-00001.parquet").write_text("x", encoding="utf-8")

    optimize_execution.emit_optimize_dry_run_plan(
        args=args,
        roots=workspace_roots,
        target_scope="global",
        systems=["fr", "de"],
        final_tokenizer_path=tmp_path / "out" / "wordpiece.json",
        models_dir=tmp_path
        / "optimize"
        / "abcd1234"
        / "wordpiece"
        / "default"
        / "efgh5678"
        / "models",
        vocab_schedule=[-1],
        min_frequencies=[1],
        seeds=[1],
        pilot_seeds=[1],
        expansion_seeds=[],
        total_candidate_jobs=1,
    )

    out = capsys.readouterr().out
    assert "[optimize] dry_run planned_input_parquet_files=2" in out


def test_resolve_optimize_corpus_for_target_returns_none_when_rebuild_fails(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    corpus_path = tmp_path / "corpus.parquet"

    result_force = optimize_execution.resolve_optimize_corpus_for_target(
        args=argparse.Namespace(force=True, name_col="name", optimize_wipe=False),
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        corpus_path=corpus_path,
        input_parquet_files=[],
        rebuild_optimize_corpus=lambda **kwargs: None,
    )
    assert result_force is None

    result_stale = optimize_execution.resolve_optimize_corpus_for_target(
        args=argparse.Namespace(force=False, name_col="name", optimize_wipe=False),
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        corpus_path=corpus_path,
        input_parquet_files=[],
        rebuild_optimize_corpus=lambda **kwargs: None,
    )
    assert result_stale is None


def test_resolve_optimize_corpus_for_target_rebuilds_when_corpus_stale(
    tmp_path: Path, workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"name": ["alpha"]}).write_parquet(corpus_path)

    input_file = layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME) / "fr-1.parquet"
    input_file.parent.mkdir(parents=True, exist_ok=True)
    input_file.write_text("x", encoding="utf-8")

    import os

    os.utime(corpus_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(input_file, ns=(2_000_000_000, 2_000_000_000))

    result = optimize_execution.resolve_optimize_corpus_for_target(
        args=argparse.Namespace(force=False, name_col="name", optimize_wipe=False),
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        corpus_path=corpus_path,
        input_parquet_files=[input_file],
        rebuild_optimize_corpus=lambda **kwargs: 42,
    )

    assert result is not None
    assert result[0] == 42


def test_rebuild_optimize_corpus_country_scope_skips_when_no_cleansed_parquet(
    tmp_path: Path, workspace_roots: WorkspaceRoots, capsys
):
    result = optimize_execution.rebuild_optimize_corpus(
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        corpus_path=tmp_path / "corpus.parquet",
        name_col="name",
        has_cleansed_parquet_func=lambda **kwargs: False,
    )

    assert result is None
    assert "Skipping 'fr' optimize" in capsys.readouterr().out


def test_rebuild_optimize_corpus_country_scope_delegates_to_sampler(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}

    def fake_sample_country(**kwargs):
        captured.update(kwargs)
        return 7

    result = optimize_execution.rebuild_optimize_corpus(
        roots=workspace_roots,
        target_scope="country",
        systems=["fr"],
        corpus_path=tmp_path / "corpus.parquet",
        name_col="name",
        has_cleansed_parquet_func=lambda **kwargs: True,
        sample_country_corpus=fake_sample_country,
    )

    assert result == 7
    assert captured["cleansed_dir"] == system_layer_dir(
        workspace_roots, "fr", layer=CLEANSED_LAYER_NAME
    )
    assert captured["name_col"] == "name"


def test_rebuild_optimize_corpus_global_scope_delegates_to_sampler(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}

    def fake_sample_global(**kwargs):
        captured.update(kwargs)
        return 21

    result = optimize_execution.rebuild_optimize_corpus(
        roots=workspace_roots,
        target_scope="global",
        systems=["fr", "de"],
        corpus_path=tmp_path / "corpus.parquet",
        name_col="name",
        sample_global_corpus=fake_sample_global,
    )

    assert result == 21
    assert captured["cleansed_dirs"] == [
        system_layer_dir(workspace_roots, "fr", layer=CLEANSED_LAYER_NAME),
        system_layer_dir(workspace_roots, "de", layer=CLEANSED_LAYER_NAME),
    ]


def test_corpus_content_hash_for_manifest_falls_back_to_sentinel_for_missing_corpus(
    tmp_path: Path,
):
    # By the time a promotion path calls this, training already read the
    # corpus, so a missing file only happens in a stub/test scenario -- the
    # manifest schema still needs a valid str rather than a crash.
    missing_corpus_path = tmp_path / "missing_corpus.parquet"

    assert (
        optimize_execution.corpus_content_hash_for_manifest(missing_corpus_path)
        == "missing"
    )


def test_corpus_content_hash_for_manifest_hashes_real_corpus_bytes(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"name": ["alpha", "beta"]}).write_parquet(corpus_path)

    assert optimize_execution.corpus_content_hash_for_manifest(
        corpus_path
    ) == optimize_execution.compute_corpus_content_hash(corpus_path)
