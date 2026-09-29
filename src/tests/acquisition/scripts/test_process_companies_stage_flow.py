from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from scripts import process_companies
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    system_layer_dir,
)
from workspace.roots import WorkspaceRoots


def test_process_companies_acquire_stage_continues_without_allow_research(
    tmp_path: Path, mocker
):
    calls: list[tuple[str, str]] = []

    def fake_run_acquisition(**kwargs):
        calls.append(("acquire", ",".join(kwargs["systems"])))
        return [SimpleNamespace(status="downloaded", message="ok")]

    def fake_shard_system_source(system_code: str, **kwargs):
        calls.append(("shard", system_code))
        return [tmp_path / f"{system_code}-source.parquet"]

    run_acquisition = mocker.patch.object(
        process_companies, "run_acquisition", side_effect=fake_run_acquisition
    )
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        side_effect=fake_shard_system_source,
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["acquire", "shard"],
        root=tmp_path,
    )

    assert exit_code == 0
    assert calls == [("acquire", "fr"), ("shard", "fr")]
    run_acquisition.assert_called_once()
    shard_system_source.assert_called_once()


def test_process_companies_acquire_stage_aborts_when_allow_research_set(
    tmp_path: Path, mocker
):
    calls: list[tuple[str, str]] = []

    def fake_run_acquisition(**kwargs):
        calls.append(("acquire", ",".join(kwargs["systems"])))
        return [SimpleNamespace(status="downloaded", message="ok")]

    def fake_shard_system_source(system_code: str, **kwargs):
        calls.append(("shard", system_code))
        return [tmp_path / f"{system_code}-source.parquet"]

    run_acquisition = mocker.patch.object(
        process_companies, "run_acquisition", side_effect=fake_run_acquisition
    )
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        side_effect=fake_shard_system_source,
    )
    get_system_plan = mocker.patch.object(
        process_companies,
        "get_system_plan",
        side_effect=lambda _system_code: SimpleNamespace(status="research_required"),
    )

    exit_code = process_companies.run_pipeline(
        systems=["dbpedia"],
        processes=["acquire", "shard", "canonical"],
        root=tmp_path,
        allow_research=True,
    )

    assert exit_code == 0
    assert calls == [("acquire", "dbpedia")]
    run_acquisition.assert_called_once()
    shard_system_source.assert_not_called()
    get_system_plan.assert_called_once_with("dbpedia")


def test_process_companies_acquire_stage_does_not_abort_for_live_targets(
    tmp_path: Path, mocker
):
    calls: list[tuple[str, str]] = []

    def fake_run_acquisition(**kwargs):
        calls.append(("acquire", ",".join(kwargs["systems"])))
        return [SimpleNamespace(status="downloaded", message="ok")]

    def fake_shard_system_source(system_code: str, **kwargs):
        calls.append(("shard", system_code))
        return [tmp_path / f"{system_code}-source.parquet"]

    run_acquisition = mocker.patch.object(
        process_companies, "run_acquisition", side_effect=fake_run_acquisition
    )
    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        side_effect=fake_shard_system_source,
    )
    get_system_plan = mocker.patch.object(
        process_companies,
        "get_system_plan",
        side_effect=lambda _system_code: SimpleNamespace(status="live"),
    )

    exit_code = process_companies.run_pipeline(
        systems=["fr"],
        processes=["acquire", "shard"],
        root=tmp_path,
        allow_research=True,
    )

    assert exit_code == 0
    assert calls == [("acquire", "fr"), ("shard", "fr")]
    run_acquisition.assert_called_once()
    shard_system_source.assert_called_once()
    get_system_plan.assert_called_once_with("fr")


def test_process_companies_runs_match_stage_when_selected(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    calls: list[str] = []

    def fake_execute_match(
        *,
        system_code: str,
        roots: WorkspaceRoots,
        target_system: str | None,
    ):
        calls.append(system_code)
        assert roots == workspace_roots
        assert target_system == "ie"
        return False, 5

    execute_match = mocker.patch.object(
        process_companies, "_execute_match", side_effect=fake_execute_match
    )

    exit_code = process_companies.run_pipeline(
        systems=["gleif"],
        processes=["match"],
        root=tmp_path,
        additional_args=[["match.target-system=ie"]],
    )

    assert exit_code == 0
    assert calls == ["gleif"]
    execute_match.assert_called_once()


def test_process_companies_refuses_a_match_with_no_target_before_any_stage(
    tmp_path: Path, mocker
):
    """A match is one source against one target: with the target unnamed, or
    several sources given, nothing runs, not even an earlier stage."""
    execute_match = mocker.patch.object(process_companies, "_execute_match")
    run_stage = mocker.patch.object(process_companies, "_run_shard_or_canonical_stage")

    with pytest.raises(ValueError, match="needs its target named"):
        process_companies.run_pipeline(
            systems=["gleif"], processes=["canonical", "match"], root=tmp_path
        )
    with pytest.raises(ValueError, match="exactly one source system"):
        process_companies.run_pipeline(
            systems=["gleif", "ie"],
            processes=["match"],
            root=tmp_path,
            additional_args=[["match.target-system=gb"]],
        )

    execute_match.assert_not_called()
    run_stage.assert_not_called()


def test_process_companies_passes_explicit_match_target_system(
    tmp_path: Path, mocker, workspace_roots: WorkspaceRoots
):
    execute_match = mocker.patch.object(
        process_companies, "_execute_match", return_value=(False, 1)
    )

    exit_code = process_companies.run_pipeline(
        systems=["ie"],
        processes=["match"],
        root=tmp_path,
        additional_args=[["match.target-system=gleif"]],
    )

    assert exit_code == 0
    execute_match.assert_called_once_with(
        system_code="ie",
        roots=workspace_roots,
        target_system="gleif",
    )


def test_process_companies_skips_tokenize_when_inputs_missing(
    tmp_path: Path, mocker, capsys
):
    calls: list[str] = []

    def fake_tokenize_name(**kwargs):
        calls.append("tokenize")
        return 1

    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["tokenize"],
        root=tmp_path,
    )

    assert exit_code == 1
    assert calls == []
    out = capsys.readouterr().out
    assert "[error] [gb] Stage tokenize failed" in out
    assert "[error] [gb] Stopping remaining stages for this system" in out
    tokenize_name.assert_not_called()


def test_process_companies_skips_shard_canonical_cleanse_on_missing_prereqs(
    tmp_path: Path, mocker, capsys
):
    def fake_shard_system_source(system_code: str, **kwargs):
        raise FileNotFoundError("source data missing")

    def fake_canonicalize_system_shards(system_code: str, **kwargs):
        raise FileNotFoundError("sharded input missing")

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
        raise ValueError("required company column missing")

    shard_system_source = mocker.patch.object(
        process_companies,
        "shard_system_source",
        side_effect=fake_shard_system_source,
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

    exit_code = process_companies.run_pipeline(
        systems=["dk"],
        processes=["shard", "canonical", "cleanse"],
        root=tmp_path,
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "[error] [dk] Stage shard failed" in out
    assert "[error] [dk] Stopping remaining stages for this system" in out
    shard_system_source.assert_called_once()
    canonicalize_system_shards.assert_not_called()
    cleanse_canonical_view.assert_not_called()


def test_process_companies_threads_input_file_to_downstream_stages(
    tmp_path: Path, layer_fixture_dir, mocker
):
    observed: dict[str, str | None] = {
        "cleanse": None,
        "tokenize": None,
    }

    canonical_dir = layer_fixture_dir("gb", layer="canonical") / "latest"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"CompanyName": ["EXAMPLE LIMITED"]}).write_parquet(
        canonical_dir / "sample.parquet"
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
            code=system_code, and_tokens=("AND",), company_type_column="CompanyCategory"
        )

    def fake_get_system_company_type_mapping(system_code: str):
        return {}

    def fake_cleanse_canonical_view(**kwargs):
        observed["cleanse"] = kwargs.get("input_file")
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
        observed["tokenize"] = kwargs.get("input_file")
        tokenized_dir = kwargs["tokenized_dir"]
        tokenized_dir.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"name_tokens": ["EXAMPLE LIMITED"]}).write_parquet(
            tokenized_dir / "sample.parquet"
        )
        return 1

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
    tokenize_name = mocker.patch.object(
        process_companies, "tokenize_name", side_effect=fake_tokenize_name
    )

    exit_code = process_companies.run_pipeline(
        systems=["gb"],
        processes=["cleanse", "tokenize"],
        root=tmp_path,
        input_file="sample.parquet",
    )

    assert exit_code == 0
    assert observed == {
        "cleanse": "sample.parquet",
        "tokenize": "sample.parquet",
    }
    cleanse_canonical_view.assert_called_once()
    tokenize_name.assert_called_once()


def test_process_companies_skips_canonical_on_runtime_error(
    tmp_path: Path, monkeypatch, capsys
):
    def fake_canonicalize_system_shards(system_code: str, **kwargs):
        raise RuntimeError("No system_field_candidates configured for system 'dk'.")

    monkeypatch.setattr(
        process_companies, "canonicalize_system_shards", fake_canonicalize_system_shards
    )

    exit_code = process_companies.run_pipeline(
        systems=["dk"],
        processes=["canonical"],
        root=tmp_path,
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "[error] [dk] Stage canonical failed" in out
    assert "No system_field_candidates configured for system 'dk'." in out
