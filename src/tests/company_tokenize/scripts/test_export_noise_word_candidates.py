from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from scripts import export_noise_word_candidates
from workspace.artifact_layout import tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def _write_stats(path: Path, rows: list[tuple[str, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "token": [row[0] for row in rows],
            "document_frequency": [1 for _ in rows],
            "document_frequency_pct": [row[1] for row in rows],
            "idf": [row[2] for row in rows],
        }
    ).write_parquet(path)


def test_run_export_writes_bounded_scored_candidates(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    stats_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "token_tfidf_stats.parquet"
    )
    rows = [(f"tok{i}", 1.0 - (i * 0.01), 1.0 + i) for i in range(20)]
    _write_stats(stats_path, rows)

    exit_code = export_noise_word_candidates.run_export(
        system="ie",
        profile="auto",
        root=tmp_path,
        stats_path=None,
        out=None,
        max_candidates=5,
        min_token_length=2,
    )

    assert exit_code == 0
    out_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_word_candidates.json"
    )
    assert out_path.exists()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["system"] == "ie"
    assert payload["tfidf_use_case"] == "corpus"
    assert payload["tfidf_namespace"] == "corpus_tfidf"
    assert len(payload["candidates"]) == 5
    assert payload["selection"]["max_candidates"] == 5
    assert payload["selection"]["source_token_count"] == 20
    # Lowest-idf/highest-df tokens sort first.
    assert payload["candidates"][0]["token"] == "tok0"
    assert payload["source_stats_path"] == stats_path.relative_to(tmp_path).as_posix()


def test_run_export_raises_when_stats_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Token TF-IDF stats parquet not found"):
        export_noise_word_candidates.run_export(
            system="ie",
            profile="auto",
            root=tmp_path,
            stats_path=None,
            out=None,
            max_candidates=200,
            min_token_length=2,
        )


def test_run_export_respects_explicit_stats_and_out_paths(tmp_path: Path):
    stats_path = tmp_path / "custom_stats.parquet"
    out_path = tmp_path / "custom_out.json"
    _write_stats(stats_path, [("acme", 0.5, 2.0)])

    exit_code = export_noise_word_candidates.run_export(
        system="gb",
        profile="auto",
        root=tmp_path,
        stats_path=str(stats_path),
        out=str(out_path),
        max_candidates=200,
        min_token_length=2,
    )

    assert exit_code == 0
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["source_stats_path"] == "custom_stats.parquet"


def test_run_export_records_a_stats_path_outside_the_root_as_given(tmp_path: Path):
    stats_path = tmp_path / "elsewhere" / "stats.parquet"
    root = tmp_path / "checkout"
    root.mkdir()
    _write_stats(stats_path, [("acme", 0.5, 2.0)])

    export_noise_word_candidates.run_export(
        system="gb",
        profile="auto",
        root=root,
        stats_path=str(stats_path),
        out=str(tmp_path / "out.json"),
        max_candidates=200,
        min_token_length=2,
    )

    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert payload["source_stats_path"] == str(stats_path)


def test_default_max_candidates_matches_package_constant():
    from company_tokenize import DEFAULT_MAX_NOISE_WORD_CANDIDATES

    parser = export_noise_word_candidates.build_parser()
    args = parser.parse_args(["--system", "ie"])
    assert args.max_candidates == DEFAULT_MAX_NOISE_WORD_CANDIDATES
