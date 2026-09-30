from __future__ import annotations

import polars as pl

from company_cleanse.normalize import (
    CASE_FOLDING_OPERATIONS,
)
from company_cleanse.rules import COMPANY_TYPE_TOKENS
from company_cleanse.step_engines.stepengine_common_ops import with_company_type_column

_COMPANY_TYPE_TOKENS_SORTED = tuple(
    sorted(token for token in COMPANY_TYPE_TOKENS if token)
)


def prefix_token_when_missing(
    lf: pl.LazyFrame,
    *,
    token_expr: pl.Expr,
    extra_guard: pl.Expr,
) -> pl.LazyFrame:
    cleansed = pl.col("name_cleansed")
    cleansed_value = pl.col("_name_cleansed_norm")
    should_prefix = (
        (cleansed_value != "")
        & (token_expr != "")
        & extra_guard
        & ~(cleansed_value == token_expr)
        & ~cleansed_value.str.starts_with(pl.concat_str([token_expr, pl.lit(" ")]))
    )

    return lf.with_columns(
        [
            pl.when(should_prefix)
            .then(
                pl.concat_str(
                    [
                        token_expr,
                        pl.lit(" "),
                        cleansed_value,
                    ]
                )
            )
            .otherwise(cleansed)
            .alias("name_cleansed")
        ]
    )


def extract_canonical_company_type(
    lf: pl.LazyFrame,
    company_type_mapping: dict[str, str],
) -> pl.LazyFrame:
    return with_company_type_column(
        lf,
        company_type_expr=pl.col("_effective_company_type").replace_strict(
            company_type_mapping,
            default=pl.col("_effective_company_type"),
        ),
    )


def generate_cleansed_company_name(
    lf: pl.LazyFrame,
    and_tokens: tuple[str, ...] | None,
    char_whitelist: str,
    normalization_operations: tuple[str, ...] = (),
) -> pl.LazyFrame:
    del and_tokens

    # `casefold` produces lowercase-equivalent output the same way `lowercase` does,
    # so a chain selecting either one still wants the case-folded variant used for
    # `name_cleansed`, not just for the `_for_match` comparisons below.
    lowercase_enabled = bool(
        CASE_FOLDING_OPERATIONS.intersection(normalization_operations)
    )

    normalized_name = pl.col("_normalized_name_body").fill_null("").str.strip_chars()
    matched_company_type = (
        pl.col("_matched_company_type").fill_null("").str.strip_chars()
    )
    canonical_company_type = pl.col("company_type").fill_null("").str.strip_chars()

    normalized_name_for_match = normalized_name.str.to_lowercase()
    matched_company_type_for_match = matched_company_type.str.to_lowercase()
    canonical_company_type_for_match = canonical_company_type.str.to_lowercase()

    if lowercase_enabled:
        # `.str.to_lowercase()` (polars has no native casefold) rather than a true
        # casefold is an adequate stand-in here: `canonical_company_type` is always
        # one of the ASCII legal-form tokens in COMPANY_TYPE_TOKENS, which never
        # contain the characters (German sharp s, Greek final sigma, ...) where
        # `lower()` and `casefold()` actually diverge.
        normalized_name = normalized_name_for_match
        matched_company_type = matched_company_type_for_match
        canonical_company_type = canonical_company_type_for_match

    no_canonical_replacement = (canonical_company_type_for_match == "") | (
        canonical_company_type_for_match == "private"
    )

    has_matched_type = matched_company_type_for_match != ""
    canonical_is_supported = canonical_company_type_for_match.is_in(
        _COMPANY_TYPE_TOKENS_SORTED
    )
    suffix_with_space = normalized_name_for_match.str.ends_with(
        pl.concat_str([pl.lit(" "), matched_company_type_for_match])
    )
    suffix_exact = normalized_name_for_match == matched_company_type_for_match
    can_replace_suffix = has_matched_type & (suffix_with_space | suffix_exact)

    prefix_without_suffix = (
        pl.when(suffix_exact)
        .then(pl.lit(""))
        .otherwise(
            normalized_name.str.slice(
                0,
                normalized_name.str.len_chars()
                - matched_company_type.str.len_chars()
                - 1,
            )
        )
    )

    canonicalized_name = (
        pl.when(can_replace_suffix)
        .then(
            pl.concat_str(
                [
                    prefix_without_suffix.str.strip_chars(),
                    pl.lit(" "),
                    canonical_company_type,
                ]
            ).str.strip_chars()
        )
        .otherwise(
            pl.concat_str(
                [
                    normalized_name.str.strip_chars(),
                    pl.lit(" "),
                    canonical_company_type,
                ]
            ).str.strip_chars()
        )
    )

    return lf.with_columns(
        [
            pl.when(no_canonical_replacement)
            .then(normalized_name)
            .otherwise(
                pl.when(canonical_is_supported)
                .then(canonicalized_name)
                .otherwise(normalized_name)
            )
            .str.replace_all(char_whitelist, " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("name_cleansed")
        ]
    )


def derive_acronym_field(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Polars-native `acronym` derivation: the vectorized twin of `extract._derive_acronym`.

    Guard/intended constraint: computes the cleansed name's initials with the
    legal form excluded and only accepts `_short_name_norm` as `acronym` when
    it matches exactly, so a legal-form or unrelated `short_name` never gets
    tagged as one. `stepengine_udf_ops.derive_acronym_field` computes the same
    thing row-wise through `extract._derive_acronym`; the two are kept
    behaviourally identical and tested against each other
    (`test_step_derive_acronym_field_native_matches_udf`).
    """
    companyname_value = pl.col("_name_cleansed_norm")
    company_type_value = pl.col("_company_type_norm")

    has_non_private_company_type = (company_type_value != "") & (
        company_type_value != "private"
    )
    has_company_type_suffix = companyname_value.str.ends_with(
        pl.concat_str([pl.lit(" "), company_type_value])
    )

    companyname_core = (
        pl.when(has_non_private_company_type & has_company_type_suffix)
        .then(
            companyname_value.str.slice(
                0,
                companyname_value.str.len_chars()
                - company_type_value.str.len_chars()
                - 1,
            )
        )
        .otherwise(companyname_value)
    )

    # Keep this inline expression. Materializing a temporary helper column
    # for initials was benchmarked and did not improve throughput.
    initials = (
        companyname_core.str.split(" ")
        .list.eval(pl.element().str.slice(0, 1))
        .list.join("")
    )

    short_name_value = pl.col("_short_name_norm")
    is_acronym_match = (
        (short_name_value != "") & (initials != "") & (initials == short_name_value)
    )

    return lf.with_columns(
        [
            pl.when(pl.col("_special_short_name").is_not_null())
            .then(pl.col("_special_short_name"))
            .otherwise(
                pl.when(is_acronym_match)
                .then(short_name_value)
                .otherwise(pl.lit(None, dtype=pl.Utf8))
            )
            .alias("acronym")
        ]
    )


def ensure_non_acronym_short_name(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Polars-native twin of `extract._ensure_non_acronym_short_name_in_cleansed`.

    Guard/intended constraint: restores a `short_name` that failed to qualify
    as `acronym` back into `name_cleansed`, but only when it is itself a bare
    legal-form token (`extra_guard` requires `acronym` to be null and the
    normalized short name to be in `COMPANY_TYPE_TOKENS`) -- any other
    rejected short name is left alone, since it is a distinct name rather
    than a legal form this guard exists to preserve. Kept behaviourally
    identical to `stepengine_udf_ops.ensure_non_acronym_short_name` and
    tested against it (`test_step_ensure_non_acronym_short_name_native_matches_udf`).
    """
    short_token = pl.col("_short_name_norm")
    extra_guard = pl.col("acronym").is_null() & short_token.is_in(
        _COMPANY_TYPE_TOKENS_SORTED
    )
    return prefix_token_when_missing(
        lf,
        token_expr=short_token,
        extra_guard=extra_guard,
    )


def ensure_quoted_name_in_cleansed(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Polars-native twin of `extract._ensure_quoted_name_in_cleansed`.

    Guard/intended constraint: restores a leading quoted brand into
    `name_cleansed`, skipping it (`extra_guard`) only when it already equals
    the derived `acronym`, so a name like `"IBM" International Business
    Machines Ltd` is not prefixed twice with the same token. Kept
    behaviourally identical to `stepengine_udf_ops.ensure_quoted_name_in_cleansed`
    and tested against it (`test_step_ensure_quoted_name_in_cleansed_native_matches_udf`).
    """
    quoted_token = pl.col("_quoted_name_norm")
    acronym_token = pl.col("acronym").fill_null("").str.to_lowercase()
    return prefix_token_when_missing(
        lf,
        token_expr=quoted_token,
        extra_guard=~(acronym_token == quoted_token),
    )
