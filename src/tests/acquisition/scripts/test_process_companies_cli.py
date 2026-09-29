from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import process_companies
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def test_resolve_effective_and_tokens_returns_none_when_missing() -> None:
    plan = SimpleNamespace(and_tokens=None)

    assert process_companies._resolve_effective_and_tokens(system_plan=plan) is None


def test_resolve_effective_and_tokens_returns_none_when_empty_list() -> None:
    plan = SimpleNamespace(and_tokens=[])

    assert process_companies._resolve_effective_and_tokens(system_plan=plan) is None


def test_resolve_effective_and_tokens_normalizes_explicit_values() -> None:
    plan = SimpleNamespace(and_tokens=[" and ", "", "ET", "and"])

    assert process_companies._resolve_effective_and_tokens(system_plan=plan) == (
        "AND",
        "ET",
        "AND",
    )


def test_parse_tokenizer_specs_parses_and_normalizes_paths(tmp_path: Path):
    model_path = tmp_path / "models" / "sp.model"
    specs = process_companies._parse_tokenizer_specs(
        [
            f"label=sp_bpe,tokenizer=sentencepiece,path={model_path},col=tokens_sp_bpe",
        ],
        project_root=tmp_path,
    )

    assert len(specs) == 1
    assert specs[0].label == "sp_bpe"
    assert specs[0].trainer == "sentencepiece"
    assert specs[0].token_col == "tokens_sp_bpe"
    assert specs[0].tokenizer_path == model_path


def test_parse_tokenizer_specs_parses_noise_words_path(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    model_path = tmp_path / "models" / "tok.json"
    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    )
    specs = process_companies._parse_tokenizer_specs(
        [
            f"label=tok,tokenizer=wordpiece,path={model_path},col=tokens,noise_words_path={noise_words_path}",
        ],
        project_root=tmp_path,
    )

    assert len(specs) == 1
    assert specs[0].noise_words_path == noise_words_path


def test_process_companies_rejects_invalid_sidecar_value(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_companies.py",
            "--systems",
            "ie",
            "--processes",
            "shard",
            "--root",
            str(tmp_path),
            "--sidecar=maybe",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        process_companies.main()

    assert exc_info.value.code == 2


def test_process_companies_rejects_train_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="handled by train_tokenizer.py"):
        process_companies.run_pipeline(
            systems=["gb"],
            processes=["tokenize"],
            root=tmp_path,
            train=True,
        )


def test_process_companies_rejects_unknown_system_code_before_any_stage(
    tmp_path: Path, mocker
):
    """`--systems glief` (a typo of `gleif`) must fail as a bad argument, not
    reach a stage and fail late as a missing directory."""
    run_non_train_strict_flow = mocker.patch.object(
        process_companies, "_run_non_train_strict_flow", return_value=0
    )
    execute_acquire = mocker.patch.object(
        process_companies, "_execute_acquire", return_value=False
    )

    with pytest.raises(ValueError, match="Unknown system 'glief'. Known values:"):
        process_companies.run_pipeline(
            systems=["glief"],
            processes=["cleanse", "match"],
            root=tmp_path,
        )

    run_non_train_strict_flow.assert_not_called()
    execute_acquire.assert_not_called()


def test_process_companies_logs_resolved_systems_for_all(
    tmp_path: Path, mocker, capsys
):
    def fake_resolve_system_selection(
        systems, add_global_for_all: bool = False, global_expands_to_all: bool = False
    ):
        return SimpleNamespace(
            requested=["all"],
            concrete=["fr", "gb"],
            include_global_target=False,
        )

    resolve_system_selection = mocker.patch.object(
        process_companies,
        "resolve_system_selection",
        side_effect=fake_resolve_system_selection,
    )
    run_non_train_strict_flow = mocker.patch.object(
        process_companies, "_run_non_train_strict_flow", return_value=0
    )

    exit_code = process_companies.run_pipeline(
        systems=["all"],
        processes=["shard"],
        root=tmp_path,
        train=False,
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Resolved systems for --systems all: fr, gb" in out
    resolve_system_selection.assert_called_once()
    run_non_train_strict_flow.assert_called_once()
