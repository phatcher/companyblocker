from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from blocking.contracts import BLOCKING_ARTIFACT_SCHEMAS
from blocking.run_layout import (
    BlockingComparisonLocation,
    blocking_pairing,
    resolve_combined_comparison_location,
)
from scripts import aggregate_strategy_comparisons
from workspace.kind_layout import Kind, data_directory
from workspace.roots import WorkspaceRoots

_SCHEMA = BLOCKING_ARTIFACT_SCHEMAS["strategy_comparison"]


def _roots_argv(roots: WorkspaceRoots) -> list[str]:
    return [
        "--data-dir",
        str(roots.data),
        "--output-dir",
        str(roots.artifacts),
        "--temp-dir",
        str(roots.temp),
    ]


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = dict.fromkeys(_SCHEMA, None)
    row.update(
        {
            "label": "tfidf",
            "representation": "tfidf",
            "similarity_backend": "sklearn",
            "is_exact_backend": True,
            "accelerator_settings": "exact_name_filter=true",
            "runtime_seconds": 100.0,
            "stage": "pruned",
            "source_system": "gleif",
            "target_system": "gb",
            "country": "gb",
            "target_rows": 5_698_275,
            "precision": 0.88,
            "recall": 0.96,
        }
    )
    row.update(overrides)
    return row


def _write_pair(
    blocking_dir: Path, *, target_system: str, drop_columns: tuple[str, ...] = ()
) -> Path:
    rows = [
        _row(stage="pruned", target_system=target_system, country=target_system),
        _row(stage="raw", target_system=target_system, country=target_system),
    ]
    frame = pl.DataFrame(rows, schema=_SCHEMA).drop(drop_columns)
    comparison_dir = blocking_dir / target_system / "data" / "gleif" / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    path = comparison_dir / "strategy_comparison.parquet"
    frame.write_parquet(path)
    return path


def test_dry_run_reads_every_pairing_under_the_output_root_and_writes_nothing(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    blocking_dir = data_directory(workspace_roots, Kind.BLOCKING)
    gb = _write_pair(blocking_dir, target_system="gb")
    ie = _write_pair(blocking_dir, target_system="ie")

    exit_code = aggregate_strategy_comparisons.main(
        [*_roots_argv(workspace_roots), "--dry-run"]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert repr([gb, ie]) in out
    combined = resolve_combined_comparison_location(workspace_roots)
    for name in (
        "strategy_comparison_combined.parquet",
        "strategy_comparison_combined.csv",
        "strategy_comparison_runtime_scaling.parquet",
        "strategy_comparison_runtime_scaling.csv",
    ):
        assert f"would clear {combined.directory / name}" in out
    assert not combined.directory.exists()


def test_load_notes_a_comparison_predating_the_accelerator_column(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_pair(
        data_directory(workspace_roots, Kind.BLOCKING),
        target_system="gb",
        drop_columns=("accelerator_settings",),
    )

    aggregate_strategy_comparisons._load(
        BlockingComparisonLocation(
            directory=path.parent,
            pairing=blocking_pairing(source_system="gleif", target_system="gb"),
        )
    )

    assert "before the comparison schema recorded accelerator settings" in (
        capsys.readouterr().out
    )


def test_load_warns_when_a_report_sits_under_a_pair_it_does_not_describe(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write_pair(
        data_directory(workspace_roots, Kind.BLOCKING), target_system="gb"
    )

    aggregate_strategy_comparisons._load(
        BlockingComparisonLocation(
            directory=path.parent,
            pairing=blocking_pairing(source_system="gleif", target_system="ie"),
        )
    )

    assert "sits under 'ie/data/gleif'" in capsys.readouterr().out


def test_aggregate_says_what_to_run_when_no_comparison_has_been_written(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    data_directory(workspace_roots, Kind.BLOCKING).mkdir(parents=True)

    exit_code = aggregate_strategy_comparisons.main(_roots_argv(workspace_roots))

    assert exit_code == 1
    stderr = capsys.readouterr().err
    assert "no pairing has a persisted comparison report" in stderr
    assert "compare_blocking_strategies.py" in stderr
