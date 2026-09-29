"""Unit coverage for `sql_query_common`'s two SQL-fragment-building helpers.

Shared by the DuckDB and pgvector backends (`duckdb_sql.py`, `pgvector.py`)
so a query-limit or WHERE-clause change is asserted once, at its own module
path, rather than through whichever backend test happens to build a query.
"""

from __future__ import annotations

import pytest
from company_vectorize.sql_query_common import (
    resolve_select_and_where,
    validate_query_limit,
)


@pytest.mark.parametrize("limit", [0, -1, -100])
def test_validate_query_limit_rejects_non_positive_limits(limit):
    with pytest.raises(ValueError, match="greater than zero"):
        validate_query_limit(limit=limit)


@pytest.mark.parametrize("limit", [1, 100])
def test_validate_query_limit_accepts_positive_limits(limit):
    validate_query_limit(limit=limit)


def test_resolve_select_and_where_joins_select_columns():
    selected, where_clause = resolve_select_and_where(
        select_cols=("a", "b", "c"), where_sql=None
    )

    assert selected == "a, b, c"
    assert where_clause == ""


def test_resolve_select_and_where_wraps_a_where_clause():
    selected, where_clause = resolve_select_and_where(
        select_cols=("a",), where_sql="a = 1"
    )

    assert selected == "a"
    # The trailing sequence is a literal backslash followed by "n", not a
    # real newline -- existing behaviour asserted as-is, not a fix.
    assert where_clause == "WHERE a = 1\\n"


def test_resolve_select_and_where_empty_where_sql_is_falsy():
    _, where_clause = resolve_select_and_where(select_cols=("a",), where_sql="")

    assert where_clause == ""
