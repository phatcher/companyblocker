"""Load a system's parquet output for analysis, reading only the columns and rows asked for.

`load_analysis_frame` reads in one of two modes: `full_read`, or `duckdb_pruned`, which runs the column-pruned, filtered SQL `build_projected_parquet_sql` builds.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

import duckdb
import polars as pl

from acquisition.tokenizer_ops import TOKENIZED_OUTPUT_COLUMNS
from workspace.data_layout import TOKENIZED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots

LOAD_MODES = ("full_read", "duckdb_pruned")
TOKENIZED_REQUIRED_COLUMNS = TOKENIZED_OUTPUT_COLUMNS
DEFAULT_REQUIRED_COLUMNS = TOKENIZED_REQUIRED_COLUMNS


def resolve_system_input_glob(
    roots: WorkspaceRoots, system: str, input_file: str | None = None
) -> str:
    pattern = input_file if input_file else f"{system}-*.parquet"
    tokenized_dir = system_layer_dir(roots, system, layer=TOKENIZED_LAYER_NAME)
    return (tokenized_dir / pattern).as_posix()


def _validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in LOAD_MODES:
        supported = ", ".join(LOAD_MODES)
        raise ValueError(
            f"Unsupported load mode '{mode}'. Supported modes: {supported}"
        )
    return normalized


def _quote_ident(name: str) -> str:
    escaped = str(name).replace('"', '""')
    return f'"{escaped}"'


def _quote_literal(value: str) -> str:
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def build_projected_parquet_sql(
    *,
    input_paths: str | Iterable[str],
    required_columns: Iterable[str],
    filter_non_empty_column: str | None = None,
) -> str:
    columns = [str(col).strip() for col in required_columns if str(col).strip()]
    if not columns:
        raise ValueError("required_columns must include at least one column")

    if isinstance(input_paths, str):
        from_clause = f"read_parquet({_quote_literal(input_paths)}, union_by_name=true)"
    else:
        normalized_paths = [
            str(path).replace("\\", "/") for path in input_paths if str(path).strip()
        ]
        if not normalized_paths:
            raise ValueError("input_paths must include at least one path")
        literals = ", ".join(_quote_literal(path) for path in normalized_paths)
        from_clause = f"read_parquet([{literals}], union_by_name=true)"

    select_sql = ",\n    ".join(_quote_ident(col) for col in columns)
    where_sql = ""
    if filter_non_empty_column:
        col = _quote_ident(filter_non_empty_column)
        where_sql = f"\nWHERE {col} IS NOT NULL\n  AND LENGTH(TRIM({col})) > 0"

    # nosec B608 - every interpolated fragment above passes through _quote_ident /
    # _quote_literal, which escapes/quotes the identifier or literal it wraps.
    query_sql = f"SELECT\n    {select_sql}\nFROM {from_clause}{where_sql}"  # nosec B608
    return query_sql


def load_analysis_frame(
    *,
    input_glob: str | Iterable[str],
    mode: str,
    required_columns: Iterable[str] = DEFAULT_REQUIRED_COLUMNS,
    filter_non_empty_column: str | None = None,
) -> pl.DataFrame:
    selected_mode = _validate_mode(mode)

    if selected_mode == "full_read":
        if isinstance(input_glob, str):
            return pl.scan_parquet(input_glob).collect()
        input_paths = [str(path) for path in input_glob]
        return pl.scan_parquet(input_paths).collect()

    # duckdb_pruned mode: apply projection and row filtering before materialization.
    sql = build_projected_parquet_sql(
        input_paths=input_glob,
        required_columns=required_columns,
        filter_non_empty_column=filter_non_empty_column,
    )

    conn = duckdb.connect(database=":memory:")
    try:
        return cast(pl.DataFrame, pl.from_arrow(conn.sql(sql).to_arrow_table()))
    finally:
        conn.close()
