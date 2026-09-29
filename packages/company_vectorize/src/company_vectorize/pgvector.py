from __future__ import annotations

from typing import Literal

from .sql_query_common import resolve_select_and_where, validate_query_limit

DistanceMetric = Literal["cosine", "l2", "ip"]


def create_extension_sql() -> str:
    return "CREATE EXTENSION IF NOT EXISTS vector;"


def create_table_sql(
    *,
    table: str,
    dimensions: int,
    id_col: str = "entity_id",
    text_col: str = "name",
    vector_col: str = "name_vec",
    include_metadata: bool = True,
) -> str:
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than zero.")

    metadata_sql = ",\n    metadata jsonb" if include_metadata else ""
    return (
        f"CREATE TABLE IF NOT EXISTS {table} (\n"
        f"    {id_col} text PRIMARY KEY,\n"
        f"    {text_col} text NOT NULL,\n"
        f"    {vector_col} vector({dimensions}) NOT NULL"
        f"{metadata_sql}\n"
        ");"
    )


def create_hnsw_index_sql(
    *,
    table: str,
    vector_col: str = "name_vec",
    distance: DistanceMetric = "cosine",
    index_name: str | None = None,
    m: int = 16,
    ef_construction: int = 64,
) -> str:
    if m <= 0:
        raise ValueError("m must be greater than zero.")
    if ef_construction <= 0:
        raise ValueError("ef_construction must be greater than zero.")

    ops = _distance_to_hnsw_ops(distance)
    resolved_index_name = index_name or f"idx_{table}_{vector_col}_hnsw"

    return (
        f"CREATE INDEX IF NOT EXISTS {resolved_index_name} ON {table}\n"
        f"USING hnsw ({vector_col} {ops})\n"
        f"WITH (m = {m}, ef_construction = {ef_construction});"
    )


def create_upsert_sql(
    *,
    table: str,
    id_col: str = "entity_id",
    text_col: str = "name",
    vector_col: str = "name_vec",
    include_metadata: bool = True,
) -> str:
    # Only identifiers (table/column names) from caller config are interpolated; row
    # values are bound through the `%s` placeholders, which identifiers can't be.
    if include_metadata:
        return (
            f"INSERT INTO {table} ({id_col}, {text_col}, {vector_col}, metadata)\n"  # nosec B608
            f"VALUES (%s, %s, %s::vector, %s::jsonb)\n"
            f"ON CONFLICT ({id_col}) DO UPDATE SET\n"
            f"  {text_col} = EXCLUDED.{text_col},\n"
            f"  {vector_col} = EXCLUDED.{vector_col},\n"
            "  metadata = EXCLUDED.metadata;"
        )

    return (
        f"INSERT INTO {table} ({id_col}, {text_col}, {vector_col})\n"  # nosec B608
        f"VALUES (%s, %s, %s::vector)\n"
        f"ON CONFLICT ({id_col}) DO UPDATE SET\n"
        f"  {text_col} = EXCLUDED.{text_col},\n"
        f"  {vector_col} = EXCLUDED.{vector_col};"
    )


def create_ann_query_sql(
    *,
    table: str,
    limit: int = 10,
    distance: DistanceMetric = "cosine",
    vector_col: str = "name_vec",
    select_cols: tuple[str, ...] = ("entity_id", "name"),
    where_sql: str | None = None,
) -> str:
    validate_query_limit(limit=limit)

    operator = _distance_to_operator(distance)
    selected, where_clause = resolve_select_and_where(
        select_cols=select_cols, where_sql=where_sql
    )

    # Only identifiers (table/column names) from caller config are interpolated; the
    # nearest-neighbor vector itself is bound through the `%s` placeholder above.
    return (
        f"SELECT {selected}\n"  # nosec B608
        f"FROM {table}\n"
        f"{where_clause}"
        f"ORDER BY {vector_col} {operator} %s::vector\n"
        f"LIMIT {limit};"
    )


def vector_literal(vector: list[float]) -> str:
    values = ",".join(f"{value:.10g}" for value in vector)
    return f"[{values}]"


def _distance_to_hnsw_ops(distance: DistanceMetric) -> str:
    if distance == "cosine":
        return "vector_cosine_ops"
    if distance == "l2":
        return "vector_l2_ops"
    if distance == "ip":
        return "vector_ip_ops"
    raise ValueError(f"Unsupported distance metric: {distance}")


def _distance_to_operator(distance: DistanceMetric) -> str:
    if distance == "cosine":
        return "<=>"
    if distance == "l2":
        return "<->"
    if distance == "ip":
        return "<#>"
    raise ValueError(f"Unsupported distance metric: {distance}")
