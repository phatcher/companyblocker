from __future__ import annotations

import sys

import polars as pl
import pytest

from scripts import compute_sif_word_probabilities
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_cleansed_fixture(roots: WorkspaceRoots, system: str) -> None:
    directory = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    directory.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["ie://1", "ie://2"],
            "name_cleansed": ["acme ltd", "beta ltd"],
        }
    ).write_parquet(directory / f"{system}-001.parquet")


def test_main_resolves_one_system_and_reports_its_artifact_path(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    _write_cleansed_fixture(workspace_roots, "ie")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_sif_word_probabilities.py",
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--system",
            "ie",
        ],
    )

    exit_code = compute_sif_word_probabilities.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Word probabilities:" in out
    assert str(workspace_roots.artifacts) in out


def test_main_resolves_the_global_list_when_flagged(
    workspace_roots: WorkspaceRoots, monkeypatch, capsys
):
    _write_cleansed_fixture(workspace_roots, "ie")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compute_sif_word_probabilities.py",
            "--data-dir",
            str(workspace_roots.data),
            "--output-dir",
            str(workspace_roots.artifacts),
            "--global",
        ],
    )

    exit_code = compute_sif_word_probabilities.main()

    assert exit_code == 0
    assert "Word probabilities:" in capsys.readouterr().out


def test_build_parser_requires_exactly_one_of_system_or_global():
    parser = compute_sif_word_probabilities.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(["--system", "ie", "--global"])
