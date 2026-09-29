from __future__ import annotations

from typing import Literal

from .sql_query_common import resolve_select_and_where, validate_query_limit

DuckDbDistanceMetric = Literal["cosine", "l2", "dot"]


def create_table_sql_multi(
    *,
    table: str,
    vector_dimensions: dict[str, int],
    id_col: str = "entity_id",
    text_cols: tuple[str, ...] = ("name",),
    include_metadata: bool = True,
) -> str:
    if not vector_dimensions:
        raise ValueError("vector_dimensions must contain at least one vector column.")

    for column, dimensions in vector_dimensions.items():
        if dimensions <= 0:
            raise ValueError(
                f"Vector dimensions for column '{column}' must be greater than zero."
            )

    text_col_sql = [f"    {column} VARCHAR" for column in text_cols]
    vector_col_sql = [f"    {column} DOUBLE[]" for column in vector_dimensions]
    metadata_col_sql = ["    metadata JSON"] if include_metadata else []

    all_columns = [
        f"    {id_col} VARCHAR PRIMARY KEY",
        *text_col_sql,
        *vector_col_sql,
        *metadata_col_sql,
    ]
    return (
        "CREATE TABLE IF NOT EXISTS "
        + table
        + " (\n"
        + ",\n".join(all_columns)
        + "\n);"
    )


def create_upsert_sql_multi(
    *,
    table: str,
    id_col: str,
    text_cols: tuple[str, ...],
    vector_cols: tuple[str, ...],
    include_metadata: bool = True,
) -> str:
    if not vector_cols:
        raise ValueError("vector_cols must contain at least one vector column.")

    insert_cols = [id_col, *text_cols, *vector_cols]
    if include_metadata:
        insert_cols.append("metadata")

    placeholders = ", ".join("?" for _ in insert_cols)
    updates = [f"{column} = EXCLUDED.{column}" for column in [*text_cols, *vector_cols]]
    if include_metadata:
        updates.append("metadata = EXCLUDED.metadata")

    # Only identifiers (table/column names) from caller config are interpolated; values
    # are bound through the `?` placeholders above, which identifiers can't be.
    return (
        f"INSERT INTO {table} ({', '.join(insert_cols)})\n"  # nosec B608
        f"VALUES ({placeholders})\n"
        f"ON CONFLICT ({id_col}) DO UPDATE SET\n"
        f"  " + ",\n  ".join(updates) + ";"
    )


def create_exact_knn_query_sql(
    *,
    table: str,
    vector_col: str,
    limit: int = 10,
    distance: DuckDbDistanceMetric = "cosine",
    select_cols: tuple[str, ...] = ("entity_id", "name"),
    where_sql: str | None = None,
) -> str:
    validate_query_limit(limit=limit)

    distance_expr = _distance_expr(distance=distance, vector_col=vector_col)
    selected, where_clause = resolve_select_and_where(
        select_cols=select_cols, where_sql=where_sql
    )

    # Only identifiers (table/column names) from caller config are interpolated; the
    # nearest-neighbor vector itself is bound through the `?` placeholder above.
    return (
        f"SELECT {selected},\n"  # nosec B608
        f"       {distance_expr} AS distance\n"
        f"FROM {table}\n"
        f"{where_clause}"
        "ORDER BY distance ASC\n"
        f"LIMIT {limit};"
    )


def _distance_expr(*, distance: DuckDbDistanceMetric, vector_col: str) -> str:
    if distance == "cosine":
        return f"list_cosine_distance({vector_col}, ?::DOUBLE[])"
    if distance == "l2":
        return f"list_distance({vector_col}, ?::DOUBLE[])"
    if distance == "dot":
        # DuckDB does nearest first via ASC, so negate dot-product to rank highest dot first.
        return f"-list_dot_product({vector_col}, ?::DOUBLE[])"
    raise ValueError(f"Unsupported DuckDB distance metric: {distance}")
