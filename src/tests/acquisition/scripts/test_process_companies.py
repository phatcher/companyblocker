from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from scripts import process_companies
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    layer_system,
    system_layer_dir,
)
from workspace.roots import WorkspaceRoots

pytestmark = pytest.mark.integration


def test_process_companies_runs_all_stages_per_system(
    tmp_path: Path, monkeypatch, mocker
):
    """One CLI invocation through `main()` covering two systems and all
    four stages, in the pipeline's real per-system, per-stage call order:
    shard, canonical, cleanse, tokenize for each system in turn. Proves the
    orchestration `process_companies.py` itself owns -- stage sequencing,
    per-stage collaborator wiring, argument threading -- on one real path;
    each collaborator's own branches are covered directly in
    `test_process_companies_cli.py`, `test_process_companies_shard_stage.py`,
    `test_process_companies_stage_flow.py`, `test_process_companies_cleanse.py`,
    `test_process_companies_tokenize.py` and `test_process_companies_dry_run.py`.
    """
    calls: list[tuple[str, str]] = []

    def fake_shard_system_source(system_code: str, **kwargs):
        calls.append(("shard", system_code))
        return [tmp_path / f"{system_code}-source.parquet"]

    def fake_canonicalize_system_shards(system_code: str, **kwargs):
        calls.append(("canonical", system_code))
        return [tmp_path / f"{system_code}-canonical.parquet"]

    def fake_configure(*, country: str, roots: WorkspaceRoots):
        return SimpleNamespace(
            cleansed_dir=system_layer_dir(roots, country, layer=CLEANSED_LAYER_NAME),
        )

    def fake_resolve_input_dir(
        *, roots: WorkspaceRoots, run_date: str | None, system: str
    ):
        return system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME) / (
            run_date or "latest"
        )

    def fake_get_system_plan(system_code: str):
        return SimpleNamespace(
            code=system_code, and_tokens=("AND",), company_type_column="CompanyCategory"
        )

    def fake_get_system_company_type_mapping(system_code: str):
        return {}

    def fake_cleanse_canonical_view(**kwargs):
        calls.append(("cleanse", layer_system(kwargs["output_dir"])))
        output_dir = kwargs["output_dir"]
        system = layer_system(output_dir)
        partition_dir = output_dir / "jurisdiction_code=gb"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "system_uri": [f"{system}:1"],
                "name": [f"{system} ltd"],
                "name_cleansed_basic": [f"{system} ltd"],
                "name_cleansed": [f"{system} ltd"],
                "jurisdiction_code": ["gb"],
            }
        ).write_parquet(partition_dir / "part-00001.parquet")
        return 10

    def fake_resolve_tokenizer_paths(
        *,
        roots: WorkspaceRoots,
        scope: str,
        system: str | None = None,
        trainer: str = "wordpiece",
    ) -> Path | None:
        tokenizer_root = roots.artifacts / "fixture-tokenizers"
        profile = "default"
        tokenizer_path = tokenizer_root / scope / (system or profile) / "wordpiece.json"
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        return tokenizer_path if (tokenizer_path).exists() else None

    def fake_tokenize_name(
        *,
        cleansed_dir: Path,
        tokenized_dir: Path,
        tokenizer_path: Path | None,
        name_col: str,
        token_col: str,
        **kwargs,
    ):
        calls.append(("tokenize", layer_system(cleansed_dir)))
        assert token_col == "name_tokens"
        system = layer_system(cleansed_dir)
        (tokenized_dir / f"{system}-001.parquet").write_text(
            "tokenized", encoding="utf-8"
        )
        return 10

    shard_system_source = mocker.patch.object(
        process_companies, "shard_system_source", side_effect=fake_shard_system_source
    )
    canonicalize_system_shards = mocker.patch.object(
        process_companies,
        "canonicalize_system_shards",
        side_effect=fake_canonicalize_system_shards,
    )
    mocker.patch.object(process_companies, "configure", side_effect=fake_configure)
    mocker.patch.object(
        process_companies, "resolve_input_dir", side_effect=fake_resolve_input_dir
    )
    mocker.patch.object(
        process_companies, "get_system_plan", side_effect=fake_get_system_plan
    )
    mocker.patch.object(
        process_companies,
        "get_system_company_type_mapping",
        side_effect=fake_get_system_company_type_mapping,
    )
    cleanse_canonical_view = mocker.patch.object(
        process_companies,
        "cleanse_canonical_view",
        side_effect=fake_cleanse_canonical_view,
    )
    mocker.patch.object(
        process_companies,
        "resolve_promoted_tokenizer_path",
        side_effect=fake_resolve_tokenizer_paths,
    )
    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_companies.py",
            "--systems",
            "fr",
            "gb",
            "--root",
            str(tmp_path),
            "--date",
            "2026-06-15",
        ],
    )

    exit_code = process_companies.main()

    assert exit_code == 0
    assert calls == [
        ("shard", "fr"),
        ("canonical", "fr"),
        ("cleanse", "fr"),
        ("tokenize", "fr"),
        ("shard", "gb"),
        ("canonical", "gb"),
        ("cleanse", "gb"),
        ("tokenize", "gb"),
    ]
    assert shard_system_source.call_count == 2
    assert canonicalize_system_shards.call_count == 2
    assert cleanse_canonical_view.call_count == 2
    assert tokenize_name.call_count == 2
