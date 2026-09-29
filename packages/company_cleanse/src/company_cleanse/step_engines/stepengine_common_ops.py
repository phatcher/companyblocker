from __future__ import annotations

import polars as pl


def with_company_type_column(
    lf: pl.LazyFrame,
    *,
    company_type_expr: pl.Expr,
) -> pl.LazyFrame:
    return lf.with_columns(
        [company_type_expr.fill_null("private").alias("company_type")]
    )
