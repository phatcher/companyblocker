"""Direct tests for `workspace.artifact_layout`.

Each assertion pins a root this module exposes to the real shape found on
disk across `src/analysis`, `src/training`, `src/validation`, `scripts/` and
`packages/company_tokenize` -- see `src/workspace/artifact_layout.py`'s
module docstring for the inventory. A caller composing its own `"artifacts"`
path is exactly the drift this module exists to close, so these are pinned
independently of any one caller's test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from workspace.artifact_layout import (
    GLOBAL_TOKENIZER_SCOPE,
    analysis_artifact_root,
    analysis_report_run_dir,
    analysis_report_runs_root,
    artifact_root,
    artifact_store_root,
    perf_artifact_root,
    pretrained_vector_artifact_root,
    tokenizer_artifact_root,
    tokenizer_scope_dir,
    trained_model_artifact_root,
    validation_artifact_root,
)
from workspace.roots import WorkspaceRoots


def test_artifact_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert artifact_root(workspace_roots) == tmp_path / "artifacts"


def test_analysis_artifact_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert (
        analysis_artifact_root(workspace_roots) == tmp_path / "artifacts" / "analysis"
    )


def test_analysis_report_runs_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert (
        analysis_report_runs_root(workspace_roots, "token_zipf")
        == tmp_path / "artifacts" / "analysis" / "token_zipf" / "runs"
    )


def test_analysis_report_run_dir(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert (
        analysis_report_run_dir(workspace_roots, "noise_layers", "2026-09-05")
        == tmp_path / "artifacts" / "analysis" / "noise_layers" / "runs" / "2026-09-05"
    )


def test_analysis_report_run_dir_is_under_its_own_runs_root(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert analysis_report_run_dir(
        workspace_roots, "token_rarity", "2026-01-01"
    ).parent == (analysis_report_runs_root(workspace_roots, "token_rarity"))


def test_tokenizer_artifact_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert (
        tokenizer_artifact_root(workspace_roots)
        == tmp_path / "artifacts" / "tokenizers" / "work"
    )


def test_artifact_store_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert artifact_store_root(workspace_roots) == tmp_path / "artifacts" / "store"


def test_tokenizer_scope_dir_for_a_system_code(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert (
        tokenizer_scope_dir(workspace_roots, system="GB")
        == tmp_path / "artifacts" / "tokenizers" / "work" / "gb"
    )


def test_tokenizer_scope_dir_normalizes_whitespace_and_case(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert (
        tokenizer_scope_dir(workspace_roots, system="  Ie ")
        == tmp_path / "artifacts" / "tokenizers" / "work" / "ie"
    )


def test_tokenizer_scope_dir_defaults_to_global(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert (
        tokenizer_scope_dir(workspace_roots)
        == tmp_path / "artifacts" / "tokenizers" / "work" / GLOBAL_TOKENIZER_SCOPE
    )


def test_tokenizer_scope_dir_rejects_blank_system(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    with pytest.raises(ValueError, match="system code is required"):
        tokenizer_scope_dir(workspace_roots, system="   ")


def test_trained_model_artifact_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert (
        trained_model_artifact_root(workspace_roots)
        == tmp_path / "artifacts" / "models"
    )


def test_pretrained_vector_artifact_root(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert (
        pretrained_vector_artifact_root(workspace_roots)
        == tmp_path / "artifacts" / "pretrained_vectors"
    )


def test_perf_artifact_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert perf_artifact_root(workspace_roots) == tmp_path / "artifacts" / "perf"


def test_validation_artifact_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert (
        validation_artifact_root(workspace_roots)
        == tmp_path / "artifacts" / "validation"
    )
