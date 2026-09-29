from __future__ import annotations

import polars as pl

from analysis.benchmark_loader import benchmark_loader_modes
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_system_data(
    roots: WorkspaceRoots, system: str, file_name: str, rows: list[dict[str, object]]
) -> None:
    target = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / file_name)


def test_benchmark_loader_modes_returns_summary_with_comparison_metrics(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "ie",
        "sample.parquet",
        [
            {
                "country_tokens": ["ALPHA"],
                "global_tokens": ["ALPHA"],
                "extra_col": "one",
            }
        ],
    )

    summary = benchmark_loader_modes(
        roots=workspace_roots,
        system="ie",
        input_file="sample.parquet",
        modes=["full_read", "duckdb_pruned"],
        runs=1,
    )

    assert summary.height == 2
    assert set(summary.get_column("mode").to_list()) == {"full_read", "duckdb_pruned"}
    assert "speedup_vs_full" in summary.columns
    assert "memory_reduction_pct_vs_full" in summary.columns


def test_benchmark_loader_modes_default_input_file_excludes_sample(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gb",
        "gb-001.parquet",
        [
            {
                "country_tokens": ["MAIN"],
                "global_tokens": ["MAIN"],
            }
        ],
    )
    _write_system_data(
        workspace_roots,
        "gb",
        "sample.parquet",
        [
            {
                "country_tokens": ["SAMPLE"],
                "global_tokens": ["SAMPLE"],
            }
        ],
    )

    summary = benchmark_loader_modes(
        roots=workspace_roots,
        system="gb",
        input_file=None,
        modes=["full_read"],
        runs=1,
    )

    assert summary.height == 1
    assert summary.item(0, "rows") == 1


def test_benchmark_loader_modes_respects_max_files(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "ie",
        "ie-001.parquet",
        [{"country_tokens": ["A"], "global_tokens": ["A"]}],
    )
    _write_system_data(
        workspace_roots,
        "ie",
        "ie-002.parquet",
        [{"country_tokens": ["B"], "global_tokens": ["B"]}],
    )
    _write_system_data(
        workspace_roots,
        "ie",
        "ie-003.parquet",
        [{"country_tokens": ["C"], "global_tokens": ["C"]}],
    )

    summary = benchmark_loader_modes(
        roots=workspace_roots,
        system="ie",
        input_file=None,
        max_files=2,
        modes=["full_read"],
        runs=1,
    )

    assert summary.height == 1
    assert summary.item(0, "rows") == 2


def test_benchmark_loader_modes_max_files_excludes_name_variant_companion_files(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gleif",
        "gleif-001.parquet",
        [{"country_tokens": ["A"], "global_tokens": ["A"]}],
    )
    # A names-corpus companion file must never be counted as a regular
    # tokenized shard, even when max_files is large enough to include it.
    _write_system_data(
        workspace_roots,
        "gleif",
        "gleif-names-001.parquet",
        [
            {"country_tokens": ["B"], "global_tokens": ["B"]},
            {"country_tokens": ["C"], "global_tokens": ["C"]},
        ],
    )

    summary = benchmark_loader_modes(
        roots=workspace_roots,
        system="gleif",
        input_file=None,
        max_files=5,
        modes=["full_read"],
        runs=1,
    )

    assert summary.item(0, "rows") == 1


def test_benchmark_loader_modes_max_files_minus_one_means_all_files(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "ie",
        "ie-001.parquet",
        [{"country_tokens": ["A"], "global_tokens": ["A"]}],
    )
    _write_system_data(
        workspace_roots,
        "ie",
        "ie-002.parquet",
        [{"country_tokens": ["B"], "global_tokens": ["B"]}],
    )
    _write_system_data(
        workspace_roots,
        "ie",
        "ie-003.parquet",
        [{"country_tokens": ["C"], "global_tokens": ["C"]}],
    )

    summary = benchmark_loader_modes(
        roots=workspace_roots,
        system="ie",
        input_file=None,
        max_files=-1,
        modes=["full_read"],
        runs=1,
    )

    assert summary.height == 1
    assert summary.item(0, "rows") == 3


def test_benchmark_loader_modes_default_does_not_filter_empty_cleansed_names(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gb",
        "gb-001.parquet",
        [
            {
                "country_tokens": ["KEEP_EMPTY"],
                "global_tokens": ["KEEP_EMPTY"],
            },
            {
                "country_tokens": ["KEEP_NON_EMPTY"],
                "global_tokens": ["KEEP_NON_EMPTY"],
            },
        ],
    )

    summary = benchmark_loader_modes(
        roots=workspace_roots,
        system="gb",
        input_file=None,
        max_files=1,
        modes=["full_read", "duckdb_pruned"],
        runs=1,
    )

    assert summary.height == 2
    assert set(summary.get_column("rows").to_list()) == {2}
