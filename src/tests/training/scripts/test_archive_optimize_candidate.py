from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl
import pytest
from company_tokenize import (
    compute_corpus_content_hash,
    read_archive_index,
    resolve_archive_dir,
    resolve_optimize_canary_pointer_path,
    resolve_optimize_dir,
    resolve_optimize_sweep_paths,
    resolve_tokenizer_paths,
)
from company_tokenize.training import train_wordpiece

from scripts import archive_optimize_candidate as archive_optimize_candidate_script
from workspace.artifact_layout import tokenizer_artifact_root, tokenizer_scope_dir
from workspace.roots import WorkspaceRoots

GRID_HASH = "testgrid"


def _write_corpus_and_candidate(roots: WorkspaceRoots) -> Path:
    corpus_path = tokenizer_scope_dir(roots, system="ie") / "training_corpus.parquet"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"ie:{i}" for i in range(43)],
            "name": ["acme systems ltd", "beta holdings ltd", "gamma trading co"] * 14
            + ["delta group"],
        }
    ).write_parquet(corpus_path)

    optimize_dir = resolve_optimize_dir(
        scope_directory=tokenizer_scope_dir(roots, system="ie"),
        trainer="wordpiece",
        tokenizer_encoding=None,
    )
    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps({"grid_hash": GRID_HASH}), encoding="utf-8")
    sweep_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=compute_corpus_content_hash(corpus_path),
        grid_hash=GRID_HASH,
    )
    sweep_paths.models_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = sweep_paths.models_dir / "wordpiece_seed_1_v64_mf1.json"
    train_wordpiece(
        corpus_path=corpus_path,
        tokenizer_path=candidate_path,
        vocab_size=64,
        min_frequency=1,
    )
    return corpus_path


def _resolve_sweep_dir(roots: WorkspaceRoots) -> Path:
    tokenizer_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope="country",
        system="ie",
        profile="default",
    )
    optimize_dir = resolve_optimize_dir(
        scope_directory=tokenizer_paths.directory, trainer="wordpiece"
    )
    sweep_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=compute_corpus_content_hash(tokenizer_paths.corpus_path),
        grid_hash=GRID_HASH,
    )
    return sweep_paths.sweep_dir


def _run_main(tmp_path: Path, monkeypatch, extra_args: list[str]) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "archive_optimize_candidate.py",
            "--root",
            str(tmp_path),
            "--scope",
            "country",
            "--system",
            "ie",
            "--tokenizer",
            "wordpiece",
            "--seed",
            "1",
            "--vocab-size",
            "64",
            "--min-frequency",
            "1",
            "--label",
            "candidate_v64",
            *extra_args,
        ],
    )
    return archive_optimize_candidate_script.main()


@pytest.mark.integration
def test_main_archives_the_candidate_and_attaches_no_report(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    """A report is built for a promoted candidate only, so archiving one an
    optimize run trained writes its entry and nothing else."""
    _write_corpus_and_candidate(workspace_roots)

    exit_code = _run_main(tmp_path, monkeypatch, [])

    assert exit_code == 0
    archive_dir = resolve_archive_dir(
        tokenizer_root=tokenizer_artifact_root(workspace_roots),
        scope="country",
        system="ie",
    )
    entries = {entry["label"]: entry for entry in read_archive_index(archive_dir)}
    assert "report_path" not in entries["candidate_v64"]
