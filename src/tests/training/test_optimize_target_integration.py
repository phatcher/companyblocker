"""One end-to-end path for the optimize stage.

Drives `run_optimize_target`'s real pilot -> expansion pipeline against a
tiny synthetic corpus, training and evaluating real WordPiece tokenizers via
`company_tokenize`, with nothing patched except the one OS-level boundary:
the process pool constructor, swapped for `ThreadPoolExecutor` so the test
stays in-process and fast rather than paying real subprocess-spawn cost.
Every module `run_optimize_target` calls already has its own contract
asserted directly, with collaborators mocked, in
`test_optimize_execution_*.py` and `test_optimize_expansion_strategy.py`;
this is the one test proving those real modules are actually wired
together, per `docs/TESTSTYLE.md`'s integration tier.

Every broad-looking test candidate previously surveyed in this area cost
about 0.02s for real, because every substantive collaborator in each was
stubbed; none of them exercised real training or evaluation. This file
exists as a separate, genuinely unmocked path for that reason, rather than
retagging one of those candidates.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from company_tokenize import OptimizeCandidatePolicy, TrainerOptions

from training import optimize_execution, optimize_retry
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def _write_corpus(path: Path, *, row_count: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = ["acme systems ltd", "beta holdings ltd", "gamma trading co"]
    pl.DataFrame(
        {
            "system_uri": [f"ie:{i}" for i in range(row_count)],
            "name": [names[i % len(names)] for i in range(row_count)],
        }
    ).write_parquet(path)


def _run_target_args(**overrides: object) -> argparse.Namespace:
    base = {
        "tokenizer": "wordpiece",
        "profile": "default",
        "corpus_filename": "training_corpus.parquet",
        "pilot_seed_count": 1,
        "force": False,
        "optimize_wipe": False,
        "name_col": "name",
        "dry_run": False,
        "validation_fraction": 0.2,
        "elbow_min_points": 1,
        "elbow_min_improvement": 0.001,
        "max_workers": 1,
        "vocab_frontier_top_k": 1,
        "vocab_frontier_distance_tolerance": 0.0,
        "finalize": False,
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


def _lenient_policy() -> OptimizeCandidatePolicy:
    # Wide enough that no safety gate can reject a real tiny-corpus
    # candidate: this test proves the phases are wired together end to end
    # with real training/evaluation, not that any one gate fires correctly
    # (that is test_optimize_execution_candidates.py's job).
    return OptimizeCandidatePolicy(
        fertility_target=1.0,
        unk_rate_threshold=1.0,
        fertility_tolerance=1.0,
        fertility_min=0.0,
        fertility_max=100.0,
        token_count_median_max=100.0,
        token_count_p95_max=100.0,
        eligibility_pass_rate=0.8,
        diagnostic_top_n=5,
        delete_rejected_models=False,
    )


def test_run_optimize_target_end_to_end_trains_and_selects_a_real_candidate(
    tmp_path: Path, workspace_roots: WorkspaceRoots, mocker
) -> None:
    def _rebuild(*, corpus_path: Path, **_kwargs: object) -> int:
        _write_corpus(corpus_path)
        return 30

    # The only boundary swapped out is the OS process pool itself: a real
    # ProcessPoolExecutor costs real spawn overhead per test and Windows
    # spawn semantics are fragile under pytest. ThreadPoolExecutor honors
    # the identical concurrent.futures.Executor interface run_candidate_batch
    # depends on, so every candidate below still runs run_optimize_candidate,
    # train_tokenizer_with_trainer and evaluate_tokenizer_metrics for real.
    mocker.patch.object(
        optimize_retry.futures, "ProcessPoolExecutor", ThreadPoolExecutor
    )

    optimize_execution.run_optimize_target(
        args=_run_target_args(),
        roots=workspace_roots,
        target=SimpleNamespace(scope="country", systems=["ie"]),
        trainer_options=TrainerOptions(),
        candidate_policy=_lenient_policy(),
        seeds=[1, 2],
        vocab_schedule=[50],
        min_frequencies=[1],
        session_id="sess-1",
        effective_elbow_extra_steps=0,
        rebuild_optimize_corpus_func=_rebuild,
    )

    shard_run_logs = list(tmp_path.rglob("sess-1.parquet"))
    assert len(shard_run_logs) == 1
    run_log_df = pl.read_parquet(shard_run_logs[0])
    # One pilot candidate (seed=1) plus one expansion candidate (seed=2),
    # both against the same (vocab=50, min_frequency=1) combo -- the
    # heuristic frontier's only option -- and both accepted under the
    # lenient policy above.
    assert run_log_df.height == 2
    assert set(run_log_df["seed"].to_list()) == {1, 2}
    assert not run_log_df["rejected"].any()

    trained_models = list(tmp_path.rglob("wordpiece_seed_*.json"))
    # Real WordPiece training wrote a model file per accepted candidate
    # (delete_rejected_models=False, and nothing was rejected).
    assert len(trained_models) == 2
