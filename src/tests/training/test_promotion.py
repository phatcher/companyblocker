"""Direct tests for `training.promotion`.

What is pinned is what a reader of the pointer relies on after a promotion: the
candidate sits whole at the location its reference names, carries a record of
what made it, and `config/tokenizers.json` names it and what it replaced.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from company_tokenize import tokenizer_directory_files

from training.promotion import (
    find_stored_candidate,
    point_profile_at_candidate,
    publish_naive_candidate,
    publish_promoted_candidate,
    store_candidate,
)
from workspace.kind_layout import Kind
from workspace.pointer import pointer_path, promoted_reference
from workspace.records import read_record
from workspace.reference import locate
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import tokenizer_selection

METRICS = {"fertility_distance": 0.01}


def _trained(work: Path, model_text: str) -> dict[str, Any]:
    work.mkdir(parents=True, exist_ok=True)
    files = tokenizer_directory_files(work, trainer="wordpiece")
    files.model.write_text(model_text, encoding="utf-8")
    files.metadata.write_text("{}", encoding="utf-8")
    files.token_scores.write_text("[]\n", encoding="utf-8")
    corpus = work / "training_corpus.parquet"
    pl.DataFrame({"name": ["alpha ltd", "beta gmbh"]}).write_parquet(corpus)
    return {
        "model_path": files.model,
        "metadata_path": files.metadata,
        "token_scores_path": files.token_scores,
        "corpus_path": corpus,
        "metrics": METRICS,
    }


def _publish(roots: WorkspaceRoots, trained: dict[str, Any], *, vocab_size: int):
    return publish_promoted_candidate(
        roots=roots,
        system="IE",
        trainer="wordpiece",
        tokenizer_encoding=None,
        vocab_size=vocab_size,
        min_frequency=2,
        parameters={"mode": "optimize"},
        invocation=["train_tokenizer.py", "--systems", "ie"],
        **trained,
    )


def test_a_promotion_writes_a_recorded_candidate_and_points_at_it(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    trained = _trained(tmp_path / "work" / "ie", '{"model": 1}')

    candidate, replaced = _publish(workspace_roots, trained, vocab_size=24000)

    assert replaced is None
    assert candidate.uri.startswith("tokenizer://ie/wordpiece/")
    assert "_v24000_mf2_" in candidate.fields["key"]
    selection = tokenizer_selection(system="ie", tokenizer_id="wordpiece")
    assert promoted_reference(workspace_roots, selection) == candidate

    held = tokenizer_directory_files(
        locate(workspace_roots, candidate), trainer="wordpiece"
    )
    assert held.model.read_text(encoding="utf-8") == '{"model": 1}'
    assert held.metadata.is_file() and held.token_scores.is_file()
    assert json.loads(held.metrics.read_text(encoding="utf-8")) == METRICS
    assert not held.optimize_summary.exists()

    record = read_record(workspace_roots, candidate)
    assert record is not None
    assert record.key == candidate.fields["key"]
    assert record.parameters["mode"] == "optimize"
    assert record.parameters["vocab_size"] == 24000
    assert record.invocation == ("train_tokenizer.py", "--systems", "ie")


def test_a_candidate_an_optimize_run_chose_holds_that_runs_summary(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    trained = _trained(tmp_path / "work" / "ie", '{"model": 1}')
    summary = {"winner_candidate": {"vocab_size_resolved": 24000}}

    candidate, _ = _publish(
        workspace_roots, {**trained, "optimize_summary": summary}, vocab_size=24000
    )

    held = tokenizer_directory_files(
        locate(workspace_roots, candidate), trainer="wordpiece"
    )
    assert json.loads(held.optimize_summary.read_text(encoding="utf-8")) == summary


def _store(roots: WorkspaceRoots, trained: dict[str, Any], *, vocab_size: int):
    return store_candidate(
        roots=roots,
        system="IE",
        trainer="wordpiece",
        tokenizer_encoding=None,
        vocab_size=vocab_size,
        min_frequency=2,
        parameters={"mode": "train"},
        invocation=["train_tokenizer.py", "--systems", "ie"],
        **trained,
    )


def test_storing_a_candidate_records_it_and_moves_no_profile(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    trained = _trained(tmp_path / "work" / "ie", '{"model": 1}')

    candidate = _store(workspace_roots, trained, vocab_size=40000)

    assert locate(workspace_roots, candidate).is_dir()
    record = read_record(workspace_roots, candidate)
    assert record is not None and record.parameters["vocab_size"] == 40000
    assert not pointer_path(workspace_roots, Kind.TOKENIZER).exists()


def test_a_stored_candidate_is_pointed_at_under_any_profile_without_retraining(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    work = tmp_path / "work" / "ie"
    naive = _store(workspace_roots, _trained(work, '{"model": 1}'), vocab_size=40000)
    tuned, _ = _publish(
        workspace_roots, _trained(work, '{"model": 2}'), vocab_size=24000
    )
    selection = tokenizer_selection(system="ie", tokenizer_id="wordpiece")

    assert point_profile_at_candidate(workspace_roots, naive, profile="naive") is None
    assert promoted_reference(workspace_roots, selection, profile="naive") == naive
    assert promoted_reference(workspace_roots, selection) == tuned

    assert point_profile_at_candidate(workspace_roots, naive) == tuned
    assert promoted_reference(workspace_roots, selection) == naive


def test_plain_training_publishes_as_naive_and_leaves_promoted_alone(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    work = tmp_path / "work" / "ie"
    tuned, _ = _publish(
        workspace_roots, _trained(work, '{"model": 2}'), vocab_size=24000
    )

    naive, replaced = publish_naive_candidate(
        roots=workspace_roots,
        system="IE",
        trainer="wordpiece",
        tokenizer_encoding=None,
        vocab_size=40000,
        min_frequency=1,
        parameters={"mode": "train"},
        invocation=["train_tokenizer.py", "--systems", "ie"],
        **_trained(work, '{"model": 1}'),
    )

    assert replaced is None
    selection = tokenizer_selection(system="ie", tokenizer_id="wordpiece")
    assert promoted_reference(workspace_roots, selection, profile="naive") == naive
    assert promoted_reference(workspace_roots, selection) == tuned
    record = read_record(workspace_roots, naive)
    assert record is not None and record.parameters["vocab_size"] == 40000


def test_a_stored_candidate_is_found_by_its_corpus_and_settings(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """Training is not repeatable, so plain training asks for the candidate it
    already stored before it trains a second, slightly different one."""
    trained = _trained(tmp_path / "work" / "ie", '{"model": 1}')
    stored = _store(workspace_roots, trained, vocab_size=40000)

    def _find(corpus: Path, **settings: object):
        return find_stored_candidate(
            roots=workspace_roots,
            system="ie",
            trainer="wordpiece",
            tokenizer_encoding=None,
            corpus_path=corpus,
            settings=settings,
        )

    assert _find(trained["corpus_path"], mode="train", min_frequency=2) == stored
    assert _find(trained["corpus_path"], mode="train", min_frequency=8) is None
    other_corpus = tmp_path / "other.parquet"
    pl.DataFrame({"name": ["gamma plc"]}).write_parquet(other_corpus)
    assert _find(other_corpus, mode="train", min_frequency=2) is None


def test_a_second_promotion_records_the_candidate_it_replaced(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    work = tmp_path / "work" / "ie"
    first, _ = _publish(
        workspace_roots, _trained(work, '{"model": 1}'), vocab_size=24000
    )

    second, replaced = _publish(
        workspace_roots, _trained(work, '{"model": 2}'), vocab_size=26000
    )

    assert replaced == first and second != first
    assert locate(workspace_roots, first).is_dir()


def test_promoting_the_same_model_again_names_the_same_candidate(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    trained = _trained(tmp_path / "work" / "ie", '{"model": 1}')

    first, _ = _publish(workspace_roots, trained, vocab_size=24000)
    again, _ = _publish(workspace_roots, trained, vocab_size=24000)

    assert again == first
