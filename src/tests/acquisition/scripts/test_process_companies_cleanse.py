from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl

from scripts import process_companies
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    system_layer_dir,
)
from workspace.roots import WorkspaceRoots


def _write_cleansed_parquet(roots: WorkspaceRoots, system: str) -> None:
    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [f"{system}:1"],
            "name": ["example ltd"],
            "name_cleansed_basic": ["example ltd"],
            "name_cleansed": ["example ltd"],
            "jurisdiction_code": [system],
        }
    ).write_parquet(cleansed_dir / f"{system}-001.parquet")


def test_process_companies_cleanse_uses_canonical_name_column_by_default(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    observed: dict[str, object] = {}

    canonical_dir = layer_fixture_dir("gb", layer="canonical") / "latest"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name": ["EXAMPLE LIMITED"]}).write_parquet(
        canonical_dir / "gb-001.parquet"
    )

    def fake_configure(*, country: str, roots: WorkspaceRoots):
        return SimpleNamespace(
            cleansed_dir=system_layer_dir(roots, country, layer=CLEANSED_LAYER_NAME),
        )

    def fake_resolve_input_dir(
        *, roots: WorkspaceRoots, run_date: str | None, system: str
    ):
        return system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME) / "latest"

    def fake_get_system_plan(system_code: str):
        return SimpleNamespace(
            code=system_code,
            and_tokens=("AND",),
            company_type_column="CompanyCategory",
            canonical_input_columns=("name",),
        )

    def fake_get_system_company_type_mapping(system_code: str):
        return {}

    def fake_cleanse_canonical_view(**kwargs):
        observed.update(kwargs)
        output_dir = kwargs["output_dir"]
        partition_dir = output_dir / "jurisdiction_code=gb"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "system_uri": ["gb:1"],
                "name": ["example limited"],
                "name_cleansed_basic": ["example limited"],
                "name_cleansed": ["example limited"],
                "jurisdiction_code": ["gb"],
            }
        ).write_parquet(partition_dir / "part-00001.parquet")
        return 1

    monkeypatch.setattr(process_companies, "configure", fake_configure)
    monkeypatch.setattr(process_companies, "resolve_input_dir", fake_resolve_input_dir)
    monkeypatch.setattr(process_companies, "get_system_plan", fake_get_system_plan)
    monkeypatch.setattr(
        process_companies,
        "get_system_company_type_mapping",
        fake_get_system_company_type_mapping,
    )
    monkeypatch.setattr(
        process_companies, "cleanse_canonical_view", fake_cleanse_canonical_view
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse"],
        root=tmp_path,
    )

    assert exit_code == 0
    assert observed["company_col"] == "name"


def test_process_companies_cleanse_allows_company_col_override(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    observed: dict[str, object] = {}

    canonical_dir = layer_fixture_dir("gb", layer="canonical") / "latest"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"custom_name": ["EXAMPLE LIMITED"]}).write_parquet(
        canonical_dir / "gb-001.parquet"
    )

    def fake_configure(*, country: str, roots: WorkspaceRoots):
        return SimpleNamespace(
            cleansed_dir=system_layer_dir(roots, country, layer=CLEANSED_LAYER_NAME),
        )

    def fake_resolve_input_dir(
        *, roots: WorkspaceRoots, run_date: str | None, system: str
    ):
        return system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME) / "latest"

    def fake_get_system_plan(system_code: str):
        return SimpleNamespace(
            code=system_code,
            and_tokens=("AND",),
            company_type_column="CompanyCategory",
            canonical_input_columns=("name",),
        )

    def fake_get_system_company_type_mapping(system_code: str):
        return {}

    def fake_cleanse_canonical_view(**kwargs):
        observed.update(kwargs)
        output_dir = kwargs["output_dir"]
        partition_dir = output_dir / "jurisdiction_code=gb"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "system_uri": ["gb:1"],
                "name": ["example limited"],
                "name_cleansed_basic": ["example limited"],
                "name_cleansed": ["example limited"],
                "jurisdiction_code": ["gb"],
            }
        ).write_parquet(partition_dir / "part-00001.parquet")
        return 1

    monkeypatch.setattr(process_companies, "configure", fake_configure)
    monkeypatch.setattr(process_companies, "resolve_input_dir", fake_resolve_input_dir)
    monkeypatch.setattr(process_companies, "get_system_plan", fake_get_system_plan)
    monkeypatch.setattr(
        process_companies,
        "get_system_company_type_mapping",
        fake_get_system_company_type_mapping,
    )
    monkeypatch.setattr(
        process_companies, "cleanse_canonical_view", fake_cleanse_canonical_view
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse"],
        root=tmp_path,
        company_col="custom_name",
    )

    assert exit_code == 0
    assert observed["company_col"] == "custom_name"


def test_process_companies_sample_mode_preserves_unrelated_files(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    _write_cleansed_parquet(workspace_roots, "gb")
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    tokenized_dir = layer_fixture_dir("gb", layer="tokenized")
    tokenized_dir.mkdir(parents=True, exist_ok=True)
    kept_cleansed = cleansed_dir / "gb-999.parquet"
    kept_tokenized = tokenized_dir / "gb-999.parquet"
    kept_cleansed.write_text("keep", encoding="utf-8")
    kept_tokenized.write_text("keep", encoding="utf-8")

    def fake_configure(*, country: str, roots: WorkspaceRoots):
        return SimpleNamespace(
            cleansed_dir=system_layer_dir(roots, country, layer=CLEANSED_LAYER_NAME),
        )

    def fake_resolve_input_dir(
        *, roots: WorkspaceRoots, run_date: str | None, system: str
    ):
        return system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME) / "latest"

    def fake_get_system_plan(system_code: str):
        return SimpleNamespace(
            code=system_code, and_tokens=("AND",), company_type_column="CompanyCategory"
        )

    def fake_get_system_company_type_mapping(system_code: str):
        return {}

    def fake_cleanse_canonical_view(**kwargs):
        output_dir = kwargs["output_dir"]
        partition_dir = output_dir / "jurisdiction_code=gb"
        partition_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "system_uri": ["gb:sample"],
                "name": ["EXAMPLE LIMITED"],
                "name_cleansed_basic": ["EXAMPLE LIMITED"],
                "name_cleansed": ["EXAMPLE LIMITED"],
                "jurisdiction_code": ["gb"],
            }
        ).write_parquet(output_dir / "sample.parquet")
        return 1

    def fake_tokenize_name(**kwargs):
        tokenized_dir = kwargs["tokenized_dir"]
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"name_tokens": ["EXAMPLE LIMITED"]}).write_parquet(
            tokenized_dir / "sample.parquet"
        )
        return 1

    monkeypatch.setattr(process_companies, "configure", fake_configure)
    monkeypatch.setattr(process_companies, "resolve_input_dir", fake_resolve_input_dir)
    monkeypatch.setattr(process_companies, "get_system_plan", fake_get_system_plan)
    monkeypatch.setattr(
        process_companies,
        "get_system_company_type_mapping",
        fake_get_system_company_type_mapping,
    )
    monkeypatch.setattr(
        process_companies, "cleanse_canonical_view", fake_cleanse_canonical_view
    )
    monkeypatch.setattr(process_companies, "tokenize_name", fake_tokenize_name)

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse", "tokenize"],
        root=tmp_path,
        input_file="sample.parquet",
    )

    assert exit_code == 0
    assert kept_cleansed.exists()
    assert kept_tokenized.exists()


def test_process_companies_general_mode_preserves_sample_parquet(
    tmp_path: Path, layer_fixture_dir, monkeypatch, workspace_roots: WorkspaceRoots
):
    # cleansed/cleansed_merge and tokenized/tokenized_merge collapsed into
    # one directory each -- there's no longer a second directory
    # for a general run's stale-cleanup to simply miss. sample.parquet
    # remains a distinct, protected artifact a general run never touches;
    # a stale general-numbered file *does* get cleared now, since general
    # runs and sample runs share the same directory.
    _write_cleansed_parquet(workspace_roots, "gb")
    cleansed_dir = layer_fixture_dir("gb", layer="cleansed")
    tokenized_dir = layer_fixture_dir("gb", layer="tokenized")
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    tokenized_dir.mkdir(parents=True, exist_ok=True)

    sample_cleansed = cleansed_dir / "sample.parquet"
    sample_tokenized = tokenized_dir / "sample.parquet"
    general_cleansed = cleansed_dir / "gb-001.parquet"
    general_tokenized = tokenized_dir / "gb-001.parquet"
    for path in [
        sample_cleansed,
        sample_tokenized,
        general_cleansed,
        general_tokenized,
    ]:
        path.write_text("keep", encoding="utf-8")

    def fake_configure(*, country: str, roots: WorkspaceRoots):
        return SimpleNamespace(
            cleansed_dir=system_layer_dir(roots, country, layer=CLEANSED_LAYER_NAME),
        )

    def fake_resolve_input_dir(
        *, roots: WorkspaceRoots, run_date: str | None, system: str
    ):
        return system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME) / "latest"

    def fake_get_system_plan(system_code: str):
        return SimpleNamespace(
            code=system_code, and_tokens=("AND",), company_type_column="CompanyCategory"
        )

    def fake_get_system_company_type_mapping(system_code: str):
        return {}

    def fake_cleanse_canonical_view(**kwargs):
        output_dir = kwargs["output_dir"]
        partition_dir = output_dir / "jurisdiction_code=gb"
        partition_dir.mkdir(parents=True, exist_ok=True)
        stale_general_output = output_dir / "gb-001.parquet"
        if stale_general_output.exists():
            stale_general_output.unlink()
        pl.DataFrame(
            {
                "system_uri": ["gb:2"],
                "name": ["EXAMPLE LIMITED"],
                "name_cleansed_basic": ["EXAMPLE LIMITED"],
                "name_cleansed": ["EXAMPLE LIMITED"],
                "jurisdiction_code": ["gb"],
            }
        ).write_parquet(partition_dir / "part-00001.parquet")
        return 1

    def fake_tokenize_name(**kwargs):
        tokenized_dir = kwargs["tokenized_dir"]
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"name_tokens": ["EXAMPLE LIMITED"]}).write_parquet(
            tokenized_dir / "gb-002.parquet"
        )
        return 1

    monkeypatch.setattr(process_companies, "configure", fake_configure)
    monkeypatch.setattr(process_companies, "resolve_input_dir", fake_resolve_input_dir)
    monkeypatch.setattr(process_companies, "get_system_plan", fake_get_system_plan)
    monkeypatch.setattr(
        process_companies,
        "get_system_company_type_mapping",
        fake_get_system_company_type_mapping,
    )
    monkeypatch.setattr(
        process_companies, "cleanse_canonical_view", fake_cleanse_canonical_view
    )
    monkeypatch.setattr(process_companies, "tokenize_name", fake_tokenize_name)

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse", "tokenize"],
        root=tmp_path,
    )

    assert exit_code == 0
    assert sample_cleansed.exists()
    assert sample_tokenized.exists()
    assert not general_cleansed.exists()
    assert not general_tokenized.exists()
    assert (tokenized_dir / "gb-002.parquet").exists()
