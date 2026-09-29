from __future__ import annotations

from pathlib import Path

from analysis.report_layout import metrics_dir, reports_dir, viz_dir


def test_each_function_names_its_own_subdirectory_beneath_the_run_root(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "runs" / "2026-09-16"

    assert metrics_dir(run_root) == run_root / "metrics"
    assert viz_dir(run_root) == run_root / "viz"
    assert reports_dir(run_root) == run_root / "reports"


def test_every_report_subdirectory_is_distinct(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    resolved = {metrics_dir(run_root), viz_dir(run_root), reports_dir(run_root)}

    assert len(resolved) == 3
