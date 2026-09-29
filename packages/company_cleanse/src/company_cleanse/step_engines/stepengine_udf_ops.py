from __future__ import annotations

import polars as pl

from company_cleanse.extract import (
    _derive_acronym,
    _ensure_non_acronym_short_name_in_cleansed,
    _ensure_quoted_name_in_cleansed,
    _replace_company_type_with_canonical,
)
from company_cleanse.normalize import normalize_suffix_surface_with_operations
from company_cleanse.step_engines.stepengine_common_ops import with_company_type_column


def _ensure_token_in_cleansed(
    lf: pl.LazyFrame,
    *,
    token_column: str,
    ensure_fn,
) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.when(
                pl.col(token_column).is_not_null()
                & (pl.col(token_column).str.strip_chars() != "")
            )
            .then(
                pl.struct([token_column, "acronym", "name_cleansed"]).map_elements(
                    lambda row: ensure_fn(
                        row[token_column],
                        row["acronym"],
                        row["name_cleansed"],
                    ),
                    return_dtype=pl.Utf8,
                )
            )
            .otherwise(pl.col("name_cleansed"))
            .alias("name_cleansed")
        ]
    )


def extract_canonical_company_type(
    lf: pl.LazyFrame,
    company_type_mapping: dict[str, str],
) -> pl.LazyFrame:
    return with_company_type_column(
        lf,
        company_type_expr=pl.col("_effective_company_type").map_elements(
            lambda s: company_type_mapping.get(s, s) if s else None,
            return_dtype=pl.Utf8,
        ),
    )


def generate_cleansed_company_name(
    lf: pl.LazyFrame,
    and_tokens: tuple[str, ...] | None,
    char_whitelist: str,
    normalization_operations: tuple[str, ...] = (),
) -> pl.LazyFrame:
    no_canonical_replacement = pl.col("company_type").is_null() | (
        pl.col("company_type").str.to_lowercase() == "private"
    )

    normalized_name_body = pl.col("_normalized_name_body").map_elements(
        lambda s: normalize_suffix_surface_with_operations(
            s,
            and_tokens=and_tokens,
            operations=normalization_operations,
        ),
        return_dtype=pl.Utf8,
    )

    return lf.with_columns(
        [
            pl.when(no_canonical_replacement)
            .then(normalized_name_body)
            .otherwise(
                pl.struct(
                    ["_normalized_name_body", "_matched_company_type", "company_type"]
                )
                .map_elements(
                    lambda r: _replace_company_type_with_canonical(
                        r["_normalized_name_body"],
                        r["_matched_company_type"],
                        r["company_type"],
                    ),
                    return_dtype=pl.Utf8,
                )
                .map_elements(
                    lambda s: normalize_suffix_surface_with_operations(
                        s,
                        and_tokens=and_tokens,
                        operations=normalization_operations,
                    ),
                    return_dtype=pl.Utf8,
                )
            )
            .str.replace_all(char_whitelist, " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("name_cleansed")
        ]
    )


def derive_acronym_field(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Row-wise `acronym` derivation through `extract._derive_acronym`.

    Guard/intended constraint: see `extract._derive_acronym`, which this
    delegates to per row. Kept behaviourally identical to
    `stepengine_polars_ops.derive_acronym_field` and tested against it
    (`test_step_derive_acronym_field_native_matches_udf`).
    """
    return lf.with_columns(
        [
            pl.when(pl.col("_special_short_name").is_not_null())
            .then(pl.col("_special_short_name"))
            .otherwise(
                pl.struct(["short_name", "name_cleansed", "company_type"]).map_elements(
                    lambda r: _derive_acronym(
                        r["short_name"],
                        r["name_cleansed"],
                        r["company_type"],
                    ),
                    return_dtype=pl.Utf8,
                )
            )
            .alias("acronym")
        ]
    )


def ensure_non_acronym_short_name(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Row-wise twin of `stepengine_polars_ops.ensure_non_acronym_short_name`.

    Guard/intended constraint: see
    `extract._ensure_non_acronym_short_name_in_cleansed`, which this delegates
    to per row. Kept behaviourally identical to the polars-native
    implementation and tested against it
    (`test_step_ensure_non_acronym_short_name_native_matches_udf`).
    """
    return _ensure_token_in_cleansed(
        lf,
        token_column="short_name",  # nosec B106 - a dataframe column name, not a credential
        ensure_fn=_ensure_non_acronym_short_name_in_cleansed,
    )


def ensure_quoted_name_in_cleansed(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Row-wise twin of `stepengine_polars_ops.ensure_quoted_name_in_cleansed`.

    Guard/intended constraint: see `extract._ensure_quoted_name_in_cleansed`,
    which this delegates to per row. Kept behaviourally identical to the
    polars-native implementation and tested against it
    (`test_step_ensure_quoted_name_in_cleansed_native_matches_udf`).
    """
    return _ensure_token_in_cleansed(
        lf,
        token_column="quoted_name",  # nosec B106 - a dataframe column name, not a credential
        ensure_fn=_ensure_quoted_name_in_cleansed,
    )
