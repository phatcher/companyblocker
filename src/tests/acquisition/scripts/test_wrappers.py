from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from scripts import acquire_companies
from workspace.roots import WorkspaceRoots


def test_acquire_wrapper_delegates_to_run_acquisition(
    tmp_path: Path, monkeypatch, workspace_roots: WorkspaceRoots
):
    captured: dict[str, object] = {}

    def fake_run_acquisition(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(status="downloaded", message="ok")]

    monkeypatch.setattr(acquire_companies, "run_acquisition", fake_run_acquisition)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_companies.py",
            "--systems",
            "gb",
            "gleif",
            "--root",
            str(tmp_path),
            "--date",
            "2026-6-15",
            "--dry-run",
        ],
    )

    assert acquire_companies.main() == 0
    assert captured["systems"] == ["gb", "gleif"]
    assert captured["roots"] == workspace_roots
    assert captured["run_date"] == "2026-06-15"
    assert captured["dry_run"] is True
    assert captured["skip_unsupported"] is True
    assert captured["allow_research"] is False
    assert callable(captured["progress"])


def test_acquire_wrapper_passes_allow_research(tmp_path: Path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_acquisition(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(status="downloaded", message="ok")]

    monkeypatch.setattr(acquire_companies, "run_acquisition", fake_run_acquisition)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_companies.py",
            "--systems",
            "dbpedia",
            "--root",
            str(tmp_path),
            "--allow-research",
        ],
    )

    assert acquire_companies.main() == 0
    assert captured["allow_research"] is True


def test_acquire_wrapper_returns_error_when_fail_on_unsupported(
    tmp_path: Path, monkeypatch
):
    def fake_run_acquisition(**kwargs):
        return [SimpleNamespace(status="blocked", message="blocked")]

    monkeypatch.setattr(acquire_companies, "run_acquisition", fake_run_acquisition)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "acquire_companies.py",
            "--systems",
            "gb",
            "--root",
            str(tmp_path),
            "--fail-on-unsupported",
        ],
    )

    assert acquire_companies.main() == 2


def test_acquire_wrapper_entrypoint_main_executes(monkeypatch):
    fake_pipeline_module = ModuleType("acquisition.pipeline")

    def fake_run_acquisition(**kwargs):
        return [SimpleNamespace(status="downloaded", message="ok")]

    fake_pipeline_module.run_acquisition = fake_run_acquisition
    monkeypatch.setitem(sys.modules, "acquisition.pipeline", fake_pipeline_module)
    monkeypatch.setattr(
        sys, "argv", ["acquire_companies.py", "--systems", "gb", "--root", "."]
    )
    monkeypatch.setattr(sys, "path", [""])

    with monkeypatch.context() as context:
        context.setattr(
            sys, "argv", ["acquire_companies.py", "--systems", "gb", "--root", "."]
        )
        try:
            runpy.run_module("scripts.acquire_companies", run_name="__main__")
        except SystemExit as exc:
            assert exc.code == 0
        else:
            raise AssertionError("Expected SystemExit from wrapper entrypoint")
