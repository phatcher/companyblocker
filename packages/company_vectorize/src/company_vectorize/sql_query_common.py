from __future__ import annotations


def validate_query_limit(*, limit: int) -> None:
    if limit <= 0:
        raise ValueError("limit must be greater than zero.")


def resolve_select_and_where(
    *,
    select_cols: tuple[str, ...],
    where_sql: str | None,
) -> tuple[str, str]:
    selected = ", ".join(select_cols)
    where_clause = f"WHERE {where_sql}\\n" if where_sql else ""
    return selected, where_clause
