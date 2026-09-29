from __future__ import annotations

import polars as pl

from company_cleanse.step_engines.stepengine_common_ops import with_company_type_column


def extract_canonical_company_type(
    lf: pl.LazyFrame,
    company_type_mapping: dict[str, str],
) -> pl.LazyFrame:
    del company_type_mapping
    return with_company_type_column(
        lf,
        company_type_expr=pl.col("_effective_company_type"),
    )


def generate_cleansed_company_name(
    lf: pl.LazyFrame,
    and_tokens: tuple[str, ...] | None,
    char_whitelist: str,
    normalization_operations: tuple[str, ...] = (),
) -> pl.LazyFrame:
    del and_tokens
    del char_whitelist
    del normalization_operations
    return lf.with_columns([pl.col("_normalized_name_body").alias("name_cleansed")])


def derive_acronym_field(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns([pl.lit(None, dtype=pl.Utf8).alias("acronym")])


def ensure_non_acronym_short_name(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf


def ensure_quoted_name_in_cleansed(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf
