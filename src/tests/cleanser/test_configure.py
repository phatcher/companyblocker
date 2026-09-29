from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from acquisition import cleanser_orchestrate
from acquisition.cleanser_orchestrate import configure, name_cleanse
from workspace.roots import WorkspaceRoots


def test_configure_azure_ml_starts_run_and_logs_params(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    start_calls: list[str] = []
    logged_params: list[dict[str, object]] = []

    monkeypatch.setenv("AZUREML_RUN_ID", "run-123")
    monkeypatch.setenv("BLOB_ROOT", str(tmp_path))
    monkeypatch.setattr(cleanser_orchestrate.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(
        cleanser_orchestrate.mlflow,
        "start_run",
        lambda: start_calls.append("started"),
    )
    monkeypatch.setattr(
        cleanser_orchestrate.mlflow,
        "log_params",
        lambda params: logged_params.append(params),
    )

    cfg = configure(country="ie")

    assert cfg.is_azure_ml is True
    assert cfg.input_dir == tmp_path / "data" / "ie"
    assert cfg.cleansed_dir.exists()
    assert start_calls == ["started"]
    assert logged_params == [
        {
            "country": "ie",
            "environment": "azure_ml",
        }
    ]


def test_configure_reuses_existing_run_and_skips_param_logging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workspace_roots: WorkspaceRoots,
):
    start_calls: list[str] = []
    logged_params: list[dict[str, object]] = []
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="active-1"))

    monkeypatch.delenv("AZUREML_RUN_ID", raising=False)
    monkeypatch.delenv("BLOB_ROOT", raising=False)
    monkeypatch.setattr(cleanser_orchestrate.mlflow, "active_run", lambda: active_run)
    monkeypatch.setattr(
        cleanser_orchestrate.mlflow,
        "start_run",
        lambda: start_calls.append("started"),
    )
    monkeypatch.setattr(
        cleanser_orchestrate.mlflow,
        "log_params",
        lambda params: logged_params.append(params),
    )

    cfg = configure(country="gb", roots=workspace_roots)

    assert cfg.is_azure_ml is False
    assert cfg.input_dir == tmp_path / "data" / "gb"
    assert cfg.cleansed_dir.exists()
    assert start_calls == []
    assert logged_params == []


def test_name_cleanse_uses_default_rules_and_raises_when_no_parquet_files(
    tmp_path: Path,
):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="No parquet files found"):
        name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=None,
            company_type_mapping=None,
        )
