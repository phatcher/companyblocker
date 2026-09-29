from __future__ import annotations

import polars as pl

from analysis.data_loader import load_analysis_frame, resolve_system_input_glob
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _write_system_data(
    roots: WorkspaceRoots, system: str, file_name: str, rows: list[dict[str, object]]
) -> None:
    target = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    target.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(target / file_name)


def test_resolve_system_input_glob_supports_optional_input_file(
    workspace_roots: WorkspaceRoots,
) -> None:
    glob_all = resolve_system_input_glob(workspace_roots, "gb")
    glob_single = resolve_system_input_glob(workspace_roots, "gb", "sample.parquet")

    tokenized = system_layer_dir(
        workspace_roots, "gb", layer=TOKENIZED_LAYER_NAME
    ).as_posix()
    assert glob_all == f"{tokenized}/gb-*.parquet"
    assert glob_single == f"{tokenized}/sample.parquet"


def test_load_analysis_frame_duckdb_pruned_projects_and_filters(
    workspace_roots: WorkspaceRoots,
) -> None:
    _write_system_data(
        workspace_roots,
        "gb",
        "sample.parquet",
        [
            {
                "country_tokens": ["ACME", "LTD"],
                "global_tokens": ["ACME", "LTD"],
                "extra_col": "x",
            },
            {
                "country_tokens": ["BLANK"],
                "global_tokens": ["BLANK"],
                "extra_col": "y",
            },
        ],
    )

    input_glob = resolve_system_input_glob(workspace_roots, "gb", "sample.parquet")
    frame = load_analysis_frame(
        input_glob=input_glob,
        mode="duckdb_pruned",
        required_columns=["country_tokens", "global_tokens"],
    )

    assert frame.height == 2
    assert frame.columns == ["country_tokens", "global_tokens"]
