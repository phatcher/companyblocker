"""Compare `data_loader`'s load modes on the same input, reporting the speed and memory difference."""

from __future__ import annotations

from collections.abc import Iterable
from time import perf_counter

import polars as pl

from analysis.data_loader import (
    DEFAULT_REQUIRED_COLUMNS,
    LOAD_MODES,
    load_analysis_frame,
    resolve_system_input_glob,
)
from workspace.data_file_naming import is_primary_data_file
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _estimate_df_size_mb(df: pl.DataFrame) -> float:
    try:
        return float(df.estimated_size("mb"))
    except Exception:  # noqa: BLE001 -- polars version fallback, any failure means retry without the unit kwarg
        return float(df.estimated_size()) / (1024.0 * 1024.0)


def _resolve_input_paths(
    *,
    roots: WorkspaceRoots,
    system: str,
    input_file: str | None,
    max_files: int | None,
) -> str | list[str]:
    if input_file:
        return resolve_system_input_glob(roots, system, input_file)

    if max_files is None:
        return resolve_system_input_glob(roots, system, None)

    if max_files == -1:
        # -1 means all benchmark input files for the selected system only.
        return resolve_system_input_glob(roots, system, None)

    if max_files < 1:
        raise ValueError(
            "max_files must be -1 (all {system}-*.parquet files) or >= 1 when provided"
        )

    system_dir = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    files = sorted(
        path
        for path in system_dir.glob(f"{system}-*.parquet")
        if is_primary_data_file(file_name=path.name, system_code=system)
    )
    selected = files[:max_files]
    return [str(path).replace("\\", "/") for path in selected]


def benchmark_loader_modes(
    *,
    roots: WorkspaceRoots,
    system: str,
    input_file: str | None = None,
    max_files: int | None = None,
    modes: Iterable[str] = LOAD_MODES,
    runs: int = 1,
    required_columns: Iterable[str] = DEFAULT_REQUIRED_COLUMNS,
    filter_non_empty_column: str | None = None,
) -> pl.DataFrame:
    if runs < 1:
        raise ValueError("runs must be >= 1")

    input_paths = _resolve_input_paths(
        roots=roots,
        system=system,
        input_file=input_file,
        max_files=max_files,
    )

    rows: list[dict[str, object]] = []
    for mode in modes:
        for run_index in range(1, runs + 1):
            t0 = perf_counter()
            frame = load_analysis_frame(
                input_glob=input_paths,
                mode=mode,
                required_columns=required_columns,
                filter_non_empty_column=filter_non_empty_column,
            )
            elapsed = perf_counter() - t0

            rows.append(
                {
                    "system": system,
                    "mode": mode,
                    "run": run_index,
                    "seconds": elapsed,
                    "rows": frame.height,
                    "columns": len(frame.columns),
                    "estimated_mb": _estimate_df_size_mb(frame),
                }
            )

    result = pl.DataFrame(rows).with_columns(
        pl.col("seconds").round(4),
        pl.col("estimated_mb").round(2),
    )
    if result.height == 0:
        return result

    summary = (
        result.group_by(["system", "mode"])
        .agg(
            pl.col("seconds").mean().alias("seconds_mean"),
            pl.col("seconds").min().alias("seconds_min"),
            pl.col("rows").max().alias("rows"),
            pl.col("columns").max().alias("columns"),
            pl.col("estimated_mb").mean().alias("estimated_mb_mean"),
        )
        .sort(["system", "mode"])
        .with_columns(
            pl.col("seconds_mean").round(4),
            pl.col("seconds_min").round(4),
            pl.col("estimated_mb_mean").round(2),
        )
    )

    full_seconds = summary.filter(pl.col("mode") == "full_read").select(
        [
            "system",
            pl.col("seconds_mean").alias("full_seconds_mean"),
            pl.col("estimated_mb_mean").alias("full_mb_mean"),
        ]
    )

    summary = (
        summary.join(full_seconds, on="system", how="left")
        .with_columns(
            pl.when(pl.col("full_seconds_mean") > 0)
            .then(pl.col("full_seconds_mean") / pl.col("seconds_mean"))
            .otherwise(None)
            .alias("speedup_vs_full"),
            pl.when(pl.col("full_mb_mean") > 0)
            .then(
                (pl.col("full_mb_mean") - pl.col("estimated_mb_mean"))
                / pl.col("full_mb_mean")
                * 100.0
            )
            .otherwise(None)
            .alias("memory_reduction_pct_vs_full"),
        )
        .drop(["full_seconds_mean", "full_mb_mean"])
        .with_columns(
            pl.col("speedup_vs_full").round(3),
            pl.col("memory_reduction_pct_vs_full").round(2),
        )
    )

    return summary
