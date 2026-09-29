"""The cleansing steps `cleanse_lazyframe` runs, native Polars where possible and UDF-backed otherwise."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    # `pl.when(...).then(...)` and a further `.when(...).then(...)` return
    # different classes, so a branch built up across `if` blocks needs both.
    from polars.expr.whenthen import ChainedThen, Then

from .config import get_effective_noise_words
from .extract import (
    _build_noise_word_sets,
    _compile_personal_owner_marker_regex,
    _derive_lead_token_decisions,
    _derive_short_name_from_cleansed,
    _extract_quoted_type_parenthesized_struct,
    _extract_special_and_lead_struct,
    _recombine_quoted_name_short_name,
    _resolve_short_name_stage_flags,
    _split_personal_owner_struct,
    _trim_quoted_name_company_type_suffix,
)
from .normalize import (
    CASE_FOLDING_OPERATIONS,
    DEFAULT_NORMALIZATION_PROFILE,
    NORMALIZATION_OPERATION_LOWERCASE,
    _normalize_company_type_value,
    normalize_suffix_surface_with_operations,
    parse_normalization_profile,
    resolve_char_whitelist_for_operations,
    with_bang_preserving_punctuation_operation,
    with_transliteration_operation,
)
from .rules import (
    _build_company_type_suffix_trie,
    _match_company_type_from_suffix_surface,
)
from .step_engines import (
    CleanseStepEngine,
    NoopCleanseStepEngine,
    PolarsCleanseStepEngine,
    UdfCleanseStepEngine,
)

CANONICAL_COMPANY_TYPE_STEP = "canonical_company_type"
GENERATE_CLEANSED_COMPANY_NAME_STEP = "generate_cleansed_company_name"
DERIVE_ACRONYM_FIELD_STEP = "derive_acronym_field"
ENSURE_NON_ACRONYM_SHORT_NAME_STEP = "ensure_non_acronym_short_name"
ENSURE_QUOTED_NAME_IN_CLEANSED_STEP = "ensure_quoted_name_in_cleansed"
STEP_ENGINE_UDF = "udf"
STEP_ENGINE_POLARS = "polars"
STEP_ENGINE_NOOP = "noop"
_SUPPORTED_STEP_ENGINES = {STEP_ENGINE_UDF, STEP_ENGINE_POLARS, STEP_ENGINE_NOOP}


_STEP_ENGINE_INSTANCES: dict[str, CleanseStepEngine] = {
    STEP_ENGINE_NOOP: NoopCleanseStepEngine(),
    STEP_ENGINE_UDF: UdfCleanseStepEngine(),
    STEP_ENGINE_POLARS: PolarsCleanseStepEngine(),
}


def _freeze_mapping(mapping: dict[str, str] | None) -> tuple[tuple[str, str], ...]:
    if not mapping:
        return ()
    return tuple(sorted((str(key), str(value)) for key, value in mapping.items()))


def _freeze_noise_words(
    noise_words: tuple[str | dict[str, str], ...] | None,
) -> tuple[tuple[str, str, str], ...]:
    frozen: list[tuple[str, str, str]] = []
    for entry in noise_words or ():
        if isinstance(entry, dict):
            token = str(entry.get("token", "")).strip()
            scope = str(entry.get("scope", "suffix")).strip().lower()
            frozen.append(("dict", token, scope))
        else:
            frozen.append(("str", str(entry).strip(), "suffix"))
    return tuple(frozen)


@lru_cache(maxsize=128)
def _cached_compiled_company_type_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


@lru_cache(maxsize=64)
def _cached_suffix_trie(
    frozen_company_type_mapping: tuple[tuple[str, str], ...],
) -> tuple[dict[str, object], int]:
    return _build_company_type_suffix_trie(dict(frozen_company_type_mapping))


@lru_cache(maxsize=64)
def _cached_normalized_source_company_type_mapping(
    frozen_source_company_type_mapping: tuple[tuple[str, str], ...],
) -> dict[str, str]:
    return {
        key.strip(): _normalize_company_type_value(value, transliterate=True)
        for key, value in frozen_source_company_type_mapping
        if key is not None and value is not None
    }


@lru_cache(maxsize=64)
def _cached_noise_word_sets(
    frozen_noise_words: tuple[tuple[str, str, str], ...],
) -> tuple[frozenset[str], frozenset[str]]:
    thawed_entries: list[str | dict[str, str]] = []
    for kind, token, scope in frozen_noise_words:
        if kind == "dict":
            thawed_entries.append({"token": token, "scope": scope})
        else:
            thawed_entries.append(token)
    return _build_noise_word_sets(tuple(thawed_entries))


@lru_cache(maxsize=64)
def _cached_personal_owner_marker_regex(
    frozen_personal_owner_markers: tuple[str, ...],
) -> re.Pattern[str] | None:
    return _compile_personal_owner_marker_regex(frozen_personal_owner_markers)


def _step_prepare_stripped_name(
    lf: pl.LazyFrame,
    company_col: str,
) -> pl.LazyFrame:
    stripped_name = pl.col(company_col).str.strip_chars()
    return lf.with_columns(
        [
            # Trim outer whitespace only; keep quote chars for leading-token extraction.
            stripped_name.alias("_stripped_name"),
            # Keep pre-owner stripped form so basic output remains stable after owner splitting.
            stripped_name.alias("_stripped_name_pre_owner"),
        ]
    )


def _step_extract_special_parts(
    lf: pl.LazyFrame,
    company_type_mapping: dict[str, str],
) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.col("_stripped_name")
            .map_elements(
                lambda s: _extract_quoted_type_parenthesized_struct(
                    s, company_type_mapping
                ),
                return_dtype=pl.Struct(
                    [
                        pl.Field("special_short_name", pl.Utf8),
                        pl.Field("special_company_type", pl.Utf8),
                        pl.Field("special_full_name", pl.Utf8),
                    ]
                ),
            )
            .alias("_special_parts")
        ]
    ).with_columns(
        [
            pl.col("_special_parts")
            .struct.field("special_short_name")
            .alias("_special_short_name"),
            pl.col("_special_parts")
            .struct.field("special_company_type")
            .alias("_special_company_type"),
            pl.col("_special_parts")
            .struct.field("special_full_name")
            .alias("_special_full_name"),
        ]
    )


def _step_extract_personal_owner(
    lf: pl.LazyFrame,
    personal_owner_markers: tuple[str, ...] | None,
) -> pl.LazyFrame:
    if not personal_owner_markers:
        return lf.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("personal_owner"),
            ]
        )

    frozen_owner_markers = tuple(
        marker.strip() for marker in personal_owner_markers if marker and marker.strip()
    )
    owner_marker_regex = _cached_personal_owner_marker_regex(frozen_owner_markers)

    if owner_marker_regex is None:
        return lf.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("personal_owner"),
            ]
        )

    return (
        lf.with_columns(
            [
                pl.col("_stripped_name")
                .map_elements(
                    lambda s: _split_personal_owner_struct(
                        s,
                        personal_owner_markers,
                        marker_regex=owner_marker_regex,
                    ),
                    return_dtype=pl.Struct(
                        [
                            pl.Field("stripped_name", pl.Utf8),
                            pl.Field("personal_owner", pl.Utf8),
                        ]
                    ),
                )
                .alias("_personal_owner_parts")
            ]
        )
        .with_columns(
            [
                pl.col("_personal_owner_parts")
                .struct.field("stripped_name")
                .alias("_stripped_name"),
                pl.col("_personal_owner_parts")
                .struct.field("personal_owner")
                .alias("personal_owner"),
            ]
        )
        .drop("_personal_owner_parts")
    )


def _step_build_name_cleansed_basic(
    lf: pl.LazyFrame,
    char_whitelist: str,
) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.col("_stripped_name_normalized")
            .str.replace_all(char_whitelist, " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("name_cleansed_basic")
        ]
    )


def _step_prepare_normalized_name_fields(
    lf: pl.LazyFrame,
    and_tokens: tuple[str, ...] | None,
    normalization_operations: tuple[str, ...],
) -> pl.LazyFrame:
    # Rejected optimization note (2026-07-01): a native-expression ASCII fast path was
    # tested here and reverted after regression in real-file script-parity hypothesis
    # runs (~35k rows/s down to ~30k rows/s on single-file benchmark).
    # Keep Python UDF normalization for this step until a clearly faster approach is proven.

    return (
        lf.with_columns(
            [
                pl.col("_stripped_name_pre_owner")
                .map_elements(
                    lambda s: normalize_suffix_surface_with_operations(
                        s,
                        and_tokens=and_tokens,
                        operations=normalization_operations,
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("_stripped_name_normalized"),
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col("_stripped_name_pre_owner") == pl.col("_name_body_reordered")
                )
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(pl.col("_name_body_reordered"))
                .alias("_name_body_reordered_for_normalize"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("_name_body_reordered_for_normalize").is_null())
                .then(pl.lit(None, dtype=pl.Utf8))
                .otherwise(
                    pl.col("_name_body_reordered_for_normalize").map_elements(
                        lambda s: normalize_suffix_surface_with_operations(
                            s,
                            and_tokens=and_tokens,
                            operations=normalization_operations,
                        ),
                        return_dtype=pl.Utf8,
                    )
                )
                .alias("_normalized_name_body_computed"),
            ]
        )
        .with_columns(
            [
                pl.coalesce(
                    [
                        pl.col("_normalized_name_body_computed"),
                        pl.col("_stripped_name_normalized"),
                    ]
                ).alias("_normalized_name_body"),
            ]
        )
        .drop("_name_body_reordered_for_normalize", "_normalized_name_body_computed")
    )


def _step_prepare_post_generate_normalized_columns(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.col("name_cleansed").fill_null("").alias("_name_cleansed_norm"),
            pl.col("company_type").fill_null("").alias("_company_type_norm"),
            pl.col("short_name").fill_null("").alias("_short_name_norm"),
            pl.col("quoted_name").fill_null("").alias("_quoted_name_norm"),
        ]
    )


def _step_trim_quoted_name_company_type_suffix(
    lf: pl.LazyFrame,
    company_type_mapping: dict[str, str],
) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.when(
                pl.col("quoted_name").is_not_null()
                & (pl.col("quoted_name").str.strip_chars() != "")
            )
            .then(
                pl.struct(["quoted_name", "company_type"]).map_elements(
                    lambda r: _trim_quoted_name_company_type_suffix(
                        r["quoted_name"],
                        r["company_type"],
                        company_type_mapping,
                    ),
                    return_dtype=pl.Utf8,
                )
            )
            .otherwise(pl.col("quoted_name"))
            .alias("quoted_name")
        ]
    )


def _step_derive_lead_token_fields(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        [
            # Derive lead-token decisions in one UDF step to keep logic centralized and testable.
            pl.struct(["_stripped_name", "_special_short_name"])
            .map_elements(
                lambda r: _derive_lead_token_decisions(
                    r["_stripped_name"],
                    r["_special_short_name"],
                ),
                return_dtype=pl.Struct(
                    [
                        pl.Field("lead_token", pl.Utf8),
                        pl.Field("lead_quoted_token", pl.Utf8),
                        pl.Field("lead_token_valid", pl.Boolean),
                        pl.Field("lead_token_is_acronym", pl.Boolean),
                        pl.Field("lead_quoted_token_valid", pl.Boolean),
                        pl.Field("short_name", pl.Utf8),
                        pl.Field("quoted_name", pl.Utf8),
                    ]
                ),
            )
            .alias("_lead_decisions")
        ]
    ).with_columns(
        [
            pl.col("_lead_decisions").struct.field("lead_token").alias("_lead_token"),
            pl.col("_lead_decisions")
            .struct.field("lead_quoted_token")
            .alias("_lead_quoted_token"),
            pl.col("_lead_decisions")
            .struct.field("lead_token_valid")
            .alias("_lead_token_valid"),
            pl.col("_lead_decisions")
            .struct.field("lead_token_is_acronym")
            .alias("_lead_token_is_acronym"),
            pl.col("_lead_decisions")
            .struct.field("lead_quoted_token_valid")
            .alias("_lead_quoted_token_valid"),
            pl.col("_lead_decisions").struct.field("short_name").alias("short_name"),
            pl.col("_lead_decisions").struct.field("quoted_name").alias("quoted_name"),
        ]
    )


def _step_extract_special_and_lead_fields(
    lf: pl.LazyFrame,
    company_type_mapping: dict[str, str],
) -> pl.LazyFrame:
    needs_delimited_extract = (
        pl.col("_stripped_name").fill_null("").str.starts_with("(")
        | pl.col("_stripped_name").fill_null("").str.starts_with('"')
        | pl.col("_stripped_name").fill_null("").str.starts_with("'")
    )

    return lf.with_columns(
        [
            pl.when(needs_delimited_extract)
            .then(pl.col("_stripped_name"))
            .otherwise(pl.lit(None, dtype=pl.Utf8))
            .map_elements(
                lambda s: _extract_special_and_lead_struct(s, company_type_mapping),
                return_dtype=pl.Struct(
                    [
                        pl.Field("special_short_name", pl.Utf8),
                        pl.Field("special_company_type", pl.Utf8),
                        pl.Field("special_full_name", pl.Utf8),
                        pl.Field("lead_token", pl.Utf8),
                        pl.Field("lead_quoted_token_valid", pl.Boolean),
                        pl.Field("short_name", pl.Utf8),
                        pl.Field("quoted_name", pl.Utf8),
                    ]
                ),
            )
            .alias("_special_and_lead")
        ]
    ).with_columns(
        [
            pl.col("_special_and_lead")
            .struct.field("special_short_name")
            .alias("_special_short_name"),
            pl.col("_special_and_lead")
            .struct.field("special_company_type")
            .alias("_special_company_type"),
            pl.col("_special_and_lead")
            .struct.field("special_full_name")
            .alias("_special_full_name"),
            pl.col("_special_and_lead").struct.field("lead_token").alias("_lead_token"),
            pl.col("_special_and_lead")
            .struct.field("lead_quoted_token_valid")
            .alias("_lead_quoted_token_valid"),
            pl.col("_special_and_lead").struct.field("short_name").alias("short_name"),
            pl.col("_special_and_lead")
            .struct.field("quoted_name")
            .alias("quoted_name"),
        ]
    )


def _step_build_name_body(lf: pl.LazyFrame) -> pl.LazyFrame:
    return (
        lf.with_columns(
            [
                pl.when(pl.col("_special_full_name").is_not_null())
                .then(pl.col("_special_full_name"))
                .otherwise(
                    pl.when(pl.col("_lead_token").str.to_lowercase() == "the")
                    .then(
                        pl.concat_str(
                            [
                                pl.lit("the "),
                                pl.col("_stripped_name")
                                .str.replace(r"^\(\s*[^)]+\s*\)\s*", "")
                                .str.replace(r"^(?:\"[^\"]+\"|'[^']+')\s*", "")
                                .str.strip_chars(),
                            ]
                        ).str.strip_chars()
                    )
                    .when(pl.col("_lead_quoted_token_valid").fill_null(False))
                    .then(
                        pl.col("_stripped_name")
                        .str.replace(r"^\(\s*[^)]+\s*\)\s*", "")
                        .str.replace(r"^(?:\"[^\"]+\"|'[^']+')\s*", "")
                        .str.strip_chars()
                    )
                    .when(
                        pl.col("_lead_token").is_not_null()
                        & (pl.col("_lead_token") != "")
                    )
                    .then(
                        pl.col("_stripped_name")
                        .str.replace(r"^\(\s*[^)]+\s*\)\s*", "")
                        .str.strip_chars()
                    )
                    .otherwise(pl.col("_stripped_name"))
                )
                .alias("_name_body")
            ]
        )
        .with_columns(
            [
                pl.col("_name_body")
                .str.extract(r"^\s*(.*?)\s*\(\s*(?i:the)\s*\)\s*$", 1)
                .str.strip_chars()
                .alias("_name_body_trailing_the_core")
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col("_name_body_trailing_the_core").is_not_null()
                    & (pl.col("_name_body_trailing_the_core") != "")
                )
                .then(
                    pl.concat_str(
                        [
                            pl.lit("the "),
                            pl.col("_name_body_trailing_the_core"),
                        ]
                    )
                )
                .otherwise(pl.col("_name_body"))
                .alias("_name_body_reordered")
            ]
        )
    )


def _resolve_matched_company_type_from_name(
    row: dict[str, str | None],
    *,
    matcher_mode: str,
    company_type_regex: re.Pattern[str] | None,
    suffix_trie: dict[str, object] | None,
    suffix_trie_max_tokens: int,
    and_tokens: tuple[str, ...] | None,
    normalization_operations: tuple[str, ...],
    transliteration_operations: tuple[str, ...],
) -> str | None:
    suffix_surface = row.get("_normalized_name_body_suffix_surface") or ""

    if matcher_mode == "trie":
        matched_company_type = _match_company_type_from_suffix_surface(
            suffix_surface,
            suffix_trie or {},
            suffix_trie_max_tokens,
        )
    else:
        match = (
            company_type_regex.search(suffix_surface) if company_type_regex else None
        )
        matched_company_type = match.group(1) if match else None

    if matched_company_type is not None:
        return matched_company_type

    return _resolve_matched_company_type_translit_fallback(
        row,
        matcher_mode=matcher_mode,
        company_type_regex=company_type_regex,
        suffix_trie=suffix_trie,
        suffix_trie_max_tokens=suffix_trie_max_tokens,
        and_tokens=and_tokens,
        normalization_operations=normalization_operations,
        transliteration_operations=transliteration_operations,
    )


def _resolve_company_type_match_from_suffix_surface(
    *,
    suffix_surface: str,
    matcher_mode: str,
    company_type_regex: re.Pattern[str] | None,
    suffix_trie: dict[str, object] | None,
    suffix_trie_max_tokens: int,
) -> str | None:
    if matcher_mode == "trie":
        return _match_company_type_from_suffix_surface(
            suffix_surface,
            suffix_trie or {},
            suffix_trie_max_tokens,
        )

    match = company_type_regex.search(suffix_surface) if company_type_regex else None
    return match.group(1) if match else None


def _resolve_transliterated_suffix_surface(
    row: dict[str, str | None],
    *,
    and_tokens: tuple[str, ...] | None,
    transliteration_operations: tuple[str, ...],
) -> str | None:
    return normalize_suffix_surface_with_operations(
        row.get("_name_body_reordered"),
        and_tokens=and_tokens,
        operations=transliteration_operations,
    )


def _resolve_matched_company_type_translit_fallback(
    row: dict[str, str | None],
    *,
    matcher_mode: str,
    company_type_regex: re.Pattern[str] | None,
    suffix_trie: dict[str, object] | None,
    suffix_trie_max_tokens: int,
    and_tokens: tuple[str, ...] | None,
    normalization_operations: tuple[str, ...],
    transliteration_operations: tuple[str, ...],
) -> str | None:
    suffix_surface = row.get("_normalized_name_body_suffix_surface") or ""

    # ASCII surfaces cannot gain new suffix matches via transliteration.
    if suffix_surface.isascii():
        return None

    suffix_surface_ascii = _resolve_transliterated_suffix_surface(
        row,
        and_tokens=and_tokens,
        transliteration_operations=transliteration_operations,
    )
    return _resolve_company_type_match_from_suffix_surface(
        suffix_surface=suffix_surface_ascii or "",
        matcher_mode=matcher_mode,
        company_type_regex=company_type_regex,
        suffix_trie=suffix_trie,
        suffix_trie_max_tokens=suffix_trie_max_tokens,
    )


def _step_resolve_and_unpack_company_type_decision(
    lf: pl.LazyFrame,
    matcher_mode: str,
    company_type_regex: re.Pattern[str] | None,
    suffix_trie: dict[str, object] | None,
    suffix_trie_max_tokens: int,
    and_tokens: tuple[str, ...] | None,
    company_type_mapping: dict[str, str] | None = None,
    normalization_operations: tuple[str, ...] = (NORMALIZATION_OPERATION_LOWERCASE,),
    transliteration_operations: tuple[str, ...] | None = None,
    use_source_company_type: bool = False,
    normalized_source_company_type_mapping: dict[str, str] | None = None,
) -> pl.LazyFrame:
    if transliteration_operations is None:
        transliteration_operations = with_transliteration_operation(
            normalization_operations
        )

    lf_resolved = lf.with_columns(
        [
            pl.col("_normalized_name_body")
            .fill_null("")
            .str.replace_all(r"!", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("_normalized_name_body_suffix_surface")
        ]
    )

    if matcher_mode == "regex" and company_type_regex is not None:
        lf_resolved = (
            lf_resolved.with_columns(
                [
                    pl.col("_normalized_name_body_suffix_surface")
                    .str.extract(company_type_regex.pattern, 1)
                    .alias("_matched_company_type_native_from_name")
                ]
            )
            .with_columns(
                [
                    pl.when(
                        pl.col("_matched_company_type_native_from_name").is_not_null()
                    )
                    .then(pl.col("_matched_company_type_native_from_name"))
                    .when(
                        pl.col("_normalized_name_body_suffix_surface").str.contains(
                            r"^[\x00-\x7F]*$"
                        )
                    )
                    .then(pl.lit(None, dtype=pl.Utf8))
                    .otherwise(
                        pl.struct(
                            [
                                "_normalized_name_body_suffix_surface",
                                "_name_body_reordered",
                            ]
                        ).map_elements(
                            lambda r: _resolve_matched_company_type_translit_fallback(
                                r,
                                matcher_mode=matcher_mode,
                                company_type_regex=company_type_regex,
                                suffix_trie=suffix_trie,
                                suffix_trie_max_tokens=suffix_trie_max_tokens,
                                and_tokens=and_tokens,
                                normalization_operations=normalization_operations,
                                transliteration_operations=transliteration_operations,
                            ),
                            return_dtype=pl.Utf8,
                        )
                    )
                    .alias("_matched_company_type_from_name")
                ]
            )
            .drop("_matched_company_type_native_from_name")
        )
    else:
        lf_resolved = lf_resolved.with_columns(
            [
                pl.struct(
                    ["_normalized_name_body_suffix_surface", "_name_body_reordered"]
                )
                .map_elements(
                    lambda r: _resolve_matched_company_type_from_name(
                        r,
                        matcher_mode=matcher_mode,
                        company_type_regex=company_type_regex,
                        suffix_trie=suffix_trie,
                        suffix_trie_max_tokens=suffix_trie_max_tokens,
                        and_tokens=and_tokens,
                        normalization_operations=normalization_operations,
                        transliteration_operations=transliteration_operations,
                    ),
                    return_dtype=pl.Utf8,
                )
                .alias("_matched_company_type_from_name")
            ]
        )

    lf_resolved = lf_resolved.with_columns(
        [
            pl.coalesce(
                [
                    pl.col("_special_company_type"),
                    pl.col("_matched_company_type_from_name"),
                ]
            ).alias("_matched_company_type")
        ]
    )

    if use_source_company_type and normalized_source_company_type_mapping:
        lf_resolved = lf_resolved.with_columns(
            [
                pl.col("_source_company_type_raw")
                .replace_strict(
                    normalized_source_company_type_mapping,
                    default=pl.lit(None, dtype=pl.Utf8),
                )
                .alias("_source_company_type_mapped")
            ]
        )
        # Only normalize empty-string mapped values when mapping configuration can emit them.
        if "" in normalized_source_company_type_mapping.values():
            lf_resolved = lf_resolved.with_columns(
                [
                    pl.when(pl.col("_source_company_type_mapped") == "")
                    .then(pl.lit(None, dtype=pl.Utf8))
                    .otherwise(pl.col("_source_company_type_mapped"))
                    .alias("_source_company_type_mapped")
                ]
            )
    else:
        lf_resolved = lf_resolved.with_columns(
            [pl.lit(None, dtype=pl.Utf8).alias("_source_company_type_mapped")]
        )

    if use_source_company_type:
        source_mapping_for_canonical = company_type_mapping or {}
        lf_resolved = lf_resolved.with_columns(
            [
                pl.col("_source_company_type_mapped")
                .replace_strict(
                    source_mapping_for_canonical,
                    default=pl.col("_source_company_type_mapped"),
                )
                .alias("_source_company_type_canonical")
            ]
        ).with_columns(
            [
                # Source mapping should drive non-extension classifications, while
                # explicit legal-form suffixes extracted from the name stay authoritative.
                pl.when(pl.col("_source_company_type_mapped").is_null())
                .then(pl.col("_matched_company_type"))
                .when(pl.col("_matched_company_type").is_null())
                .then(pl.lit(None, dtype=pl.Utf8))
                .when(
                    pl.col("_source_company_type_canonical")
                    != pl.col("_source_company_type_mapped")
                )
                .then(pl.col("_matched_company_type"))
                .otherwise(pl.lit(None, dtype=pl.Utf8))
                .alias("_matched_company_type")
            ]
        )
    else:
        lf_resolved = lf_resolved.with_columns(
            [pl.lit(None, dtype=pl.Utf8).alias("_source_company_type_canonical")]
        )

    return (
        lf_resolved.with_columns(
            [
                pl.coalesce(
                    [
                        pl.col("_matched_company_type"),
                        pl.col("_source_company_type_mapped"),
                    ]
                ).alias("_effective_company_type"),
                pl.when(pl.col("_matched_company_type").is_not_null())
                .then(pl.lit("name_suffix"))
                .when(pl.col("_source_company_type_mapped").is_not_null())
                .then(pl.lit("source_mapping"))
                .otherwise(pl.lit("default_private"))
                .alias("company_type_source"),
            ]
        )
        .with_columns(
            [pl.col("_effective_company_type").is_null().alias("company_type_missing")]
        )
        .drop(
            "_matched_company_type_from_name",
            "_normalized_name_body_suffix_surface",
            "_source_company_type_canonical",
        )
    )


def _resolve_step_engine_name(
    *,
    step_name: str,
    step_engines: dict[str, str] | None,
    default_engine: str,
) -> str:
    configured_engine = (step_engines or {}).get(step_name, default_engine)
    normalized = str(configured_engine).strip().lower()
    if normalized not in _SUPPORTED_STEP_ENGINES:
        supported = ", ".join(sorted(_SUPPORTED_STEP_ENGINES))
        raise ValueError(
            f"Unsupported step engine '{configured_engine}' for step '{step_name}'. "
            f"Supported engines: {supported}."
        )
    return normalized


def _resolve_step_engine(
    *,
    step_name: str,
    step_engines: dict[str, str] | None,
    default_engine: str,
) -> CleanseStepEngine:
    resolved_name = _resolve_step_engine_name(
        step_name=step_name,
        step_engines=step_engines,
        default_engine=default_engine,
    )
    return _STEP_ENGINE_INSTANCES[resolved_name]


def _apply_step_engine_method(
    *,
    lf: pl.LazyFrame,
    step_name: str,
    method_name: str,
    method_args: tuple[object, ...] = (),
    engine: str = STEP_ENGINE_POLARS,
) -> pl.LazyFrame:
    resolved_engine = _resolve_step_engine(
        step_name=step_name,
        step_engines={step_name: engine},
        default_engine=STEP_ENGINE_POLARS,
    )
    method = getattr(resolved_engine, method_name)
    return method(lf, *method_args)


def _step_extract_canonical_company_type(
    lf: pl.LazyFrame,
    company_type_mapping: dict[str, str],
    *,
    engine: str = STEP_ENGINE_POLARS,
) -> pl.LazyFrame:
    return _apply_step_engine_method(
        lf=lf,
        step_name=CANONICAL_COMPANY_TYPE_STEP,
        method_name="extract_canonical_company_type",
        method_args=(company_type_mapping,),
        engine=engine,
    )


def _step_generate_cleansed_company_name(
    lf: pl.LazyFrame,
    and_tokens: tuple[str, ...] | None,
    char_whitelist: str,
    normalization_operations: tuple[str, ...] = (),
    *,
    engine: str = STEP_ENGINE_POLARS,
) -> pl.LazyFrame:
    return _apply_step_engine_method(
        lf=lf,
        step_name=GENERATE_CLEANSED_COMPANY_NAME_STEP,
        method_name="generate_cleansed_company_name",
        method_args=(and_tokens, char_whitelist, normalization_operations),
        engine=engine,
    )


def _build_engine_only_step(
    *,
    step_name: str,
    method_name: str,
):
    def _step(
        lf: pl.LazyFrame,
        *,
        engine: str = STEP_ENGINE_POLARS,
    ) -> pl.LazyFrame:
        return _apply_step_engine_method(
            lf=lf,
            step_name=step_name,
            method_name=method_name,
            engine=engine,
        )

    return _step


def _step_fill_missing_short_name(
    lf: pl.LazyFrame,
    *,
    noise_words_profile: str,
    noise_words_set_kind: str,
    short_name_profile: str = DEFAULT_NORMALIZATION_PROFILE,
    jurisdiction_col: str | None = None,
) -> pl.LazyFrame:
    flags = _resolve_short_name_stage_flags(short_name_profile)
    include_company_type = flags.include_company_type
    include_noise_words = flags.include_noise_words
    geographic_tiers = flags.geographic_tiers

    # The native fast paths below decide a row's short_name without consulting the
    # UDF at all, so they are only valid while the UDF would agree. Turning the
    # geographic stage on gives the UDF something extra to do, the same reason
    # -company_type disables them.
    use_company_type_fast_paths = include_company_type and geographic_tiers is None

    effective_noise_words: tuple[str | dict[str, str], ...] = ()
    precomputed_noise_word_sets: tuple[frozenset[str], frozenset[str]] | None = None
    if include_noise_words:
        effective_noise_words = get_effective_noise_words(
            noise_words_profile=flags.noise_words_level or noise_words_profile,
            noise_words_set_kind=noise_words_set_kind,
        )
        precomputed_noise_word_sets = _cached_noise_word_sets(
            _freeze_noise_words(effective_noise_words)
        )

    fallback_branch: Then | ChainedThen = pl.when(
        pl.col("_name_cleansed_norm") == ""
    ).then(pl.col("short_name"))
    if use_company_type_fast_paths:
        # These two fast paths short-circuit the UDF below by trimming a trailing
        # company-type token via native Polars string ops -- skip them when
        # -company_type is set so rows fall through to the (slower but correct)
        # UDF, which itself will honor include_company_type=False.
        quoted_name_exact_company_stem = (
            (pl.col("_quoted_name_norm") != "")
            & (pl.col("_company_type_norm") != "")
            & (
                pl.col("name_cleansed").fill_null("")
                == pl.concat_str(
                    [
                        pl.col("_quoted_name_norm"),
                        pl.lit(" "),
                        pl.col("_company_type_norm"),
                    ]
                )
            )
        )
        fallback_branch = fallback_branch.when(quoted_name_exact_company_stem).then(
            pl.col("quoted_name")
        )
    fallback_branch = fallback_branch.when(
        ~pl.col("_name_cleansed_norm").str.contains(r"\s")
    ).then(
        # Keep single-token cleansed names as-is. This intentionally leaves
        # data-quality edge cases (for example symbol-heavy source names like
        # `£`/`@` or leading legal-form names such as `LTD ...`) to downstream
        # review instead of hardcoding source-specific overrides here.
        pl.col("_name_cleansed_norm")
    )
    if use_company_type_fast_paths:
        simple_two_token_suffix = (
            (pl.col("_company_type_norm") != "")
            & (pl.col("_company_type_norm") != "private")
            & ~pl.col("_company_type_norm").str.contains(r"\s")
            & pl.col("_name_cleansed_norm").str.contains(r"^[^\s]+\s+[^\s]+$")
            & pl.col("_name_cleansed_norm").str.ends_with(
                pl.concat_str([pl.lit(" "), pl.col("_company_type_norm")])
            )
        )
        fallback_branch = fallback_branch.when(simple_two_token_suffix).then(
            pl.col("_name_cleansed_norm").str.extract(r"^([^\s]+)\s+[^\s]+$", 1)
        )
    short_name_struct_fields = ["name_cleansed", "company_type"]
    if jurisdiction_col is not None:
        short_name_struct_fields.append(jurisdiction_col)

    def _derive_short_name_row(r: dict[str, Any]) -> str | None:
        return _derive_short_name_from_cleansed(
            r["name_cleansed"],
            r["company_type"],
            effective_noise_words,
            noise_word_sets=precomputed_noise_word_sets,
            include_company_type=include_company_type,
            include_noise_words=include_noise_words,
            geographic_tiers=geographic_tiers,
            jurisdiction_code=r[jurisdiction_col]
            if jurisdiction_col is not None
            else None,
        )

    fallback_expr = fallback_branch.otherwise(
        # Keep `company_type` here. Passing `_company_type_norm` into the UDF
        # was benchmarked and did not improve end-to-end throughput.
        pl.struct(short_name_struct_fields).map_elements(
            _derive_short_name_row,
            return_dtype=pl.Utf8,
        )
    )

    return lf.with_columns(
        [
            pl.when(pl.col("_short_name_norm") == "")
            .then(fallback_expr)
            .otherwise(pl.col("short_name"))
            .alias("short_name")
        ]
    )


def _step_recombine_quoted_name_short_name(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        [
            pl.when(
                pl.col("short_name").is_not_null()
                & (pl.col("short_name").str.strip_chars() != "")
            )
            .then(
                pl.struct(
                    ["quoted_name", "short_name", "company_type", "name_cleansed"]
                ).map_elements(
                    lambda r: _recombine_quoted_name_short_name(
                        r["quoted_name"],
                        r["short_name"],
                        r["company_type"],
                        r["name_cleansed"],
                    ),
                    return_dtype=pl.Utf8,
                )
            )
            .otherwise(pl.col("short_name"))
            .alias("short_name")
        ]
    )


def _run_post_company_type_steps(
    lf: pl.LazyFrame,
    *,
    company_type_mapping: dict[str, str],
    step_engines: dict[str, str] | None,
    and_tokens: tuple[str, ...] | None,
    char_whitelist: str,
    bang_preserving_operations: tuple[str, ...],
    noise_words_profile: str,
    noise_words_set_kind: str,
    short_name_profile: str = DEFAULT_NORMALIZATION_PROFILE,
    jurisdiction_col: str | None = None,
) -> pl.LazyFrame:
    canonical_company_type_engine = _resolve_step_engine_name(
        step_name=CANONICAL_COMPANY_TYPE_STEP,
        step_engines=step_engines,
        default_engine=STEP_ENGINE_POLARS,
    )
    lf = _step_extract_canonical_company_type(
        lf,
        company_type_mapping,
        engine=canonical_company_type_engine,
    )

    generate_cleansed_engine = _resolve_step_engine_name(
        step_name=GENERATE_CLEANSED_COMPANY_NAME_STEP,
        step_engines=step_engines,
        default_engine=STEP_ENGINE_POLARS,
    )
    lf = _step_generate_cleansed_company_name(
        lf,
        and_tokens,
        char_whitelist,
        normalization_operations=bang_preserving_operations,
        engine=generate_cleansed_engine,
    )
    lf = _step_trim_quoted_name_company_type_suffix(
        lf,
        company_type_mapping,
    )
    lf = _step_prepare_post_generate_normalized_columns(lf)

    derive_acronym_engine = _resolve_step_engine_name(
        step_name=DERIVE_ACRONYM_FIELD_STEP,
        step_engines=step_engines,
        default_engine=STEP_ENGINE_POLARS,
    )
    lf = _step_derive_acronym_field(
        lf,
        engine=derive_acronym_engine,
    )

    ensure_non_acronym_engine = _resolve_step_engine_name(
        step_name=ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
        step_engines=step_engines,
        default_engine=STEP_ENGINE_POLARS,
    )
    lf = _step_ensure_non_acronym_short_name(
        lf,
        engine=ensure_non_acronym_engine,
    )

    ensure_quoted_engine = _resolve_step_engine_name(
        step_name=ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
        step_engines=step_engines,
        default_engine=STEP_ENGINE_POLARS,
    )
    lf = _step_ensure_quoted_name_in_cleansed(
        lf,
        engine=ensure_quoted_engine,
    )

    lf = _step_fill_missing_short_name(
        lf,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        short_name_profile=short_name_profile,
        jurisdiction_col=jurisdiction_col,
    )
    return _step_recombine_quoted_name_short_name(lf)


def _finalize_cleanse_output(
    lf: pl.LazyFrame,
    *,
    input_columns: list[str],
) -> pl.LazyFrame:
    schema_names = lf.collect_schema().names()

    # Ensure helper/intermediate columns never leak into final output.
    helper_columns = [
        column_name for column_name in schema_names if column_name.startswith("_")
    ]
    if helper_columns:
        lf = lf.drop(*helper_columns)
        schema_names = [
            column_name
            for column_name in schema_names
            if column_name not in helper_columns
        ]

    return _finalize_output_column_order(
        lf,
        input_columns=input_columns,
        schema_names=schema_names,
    )


_step_derive_acronym_field = _build_engine_only_step(
    step_name=DERIVE_ACRONYM_FIELD_STEP,
    method_name="derive_acronym_field",
)


_step_ensure_non_acronym_short_name = _build_engine_only_step(
    step_name=ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
    method_name="ensure_non_acronym_short_name",
)


_step_ensure_quoted_name_in_cleansed = _build_engine_only_step(
    step_name=ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
    method_name="ensure_quoted_name_in_cleansed",
)


def _resolve_runtime_normalization(
    *,
    normalization_profile: str,
    char_whitelist: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]:
    normalization_operations = parse_normalization_profile(normalization_profile)
    transliteration_operations = with_transliteration_operation(
        normalization_operations
    )
    bang_preserving_operations = with_bang_preserving_punctuation_operation(
        transliteration_operations
    )

    resolved_char_whitelist = char_whitelist
    if not CASE_FOLDING_OPERATIONS.intersection(normalization_operations):
        resolved_char_whitelist = resolve_char_whitelist_for_operations(
            char_whitelist,
            operations=normalization_operations,
        )

    return (
        normalization_operations,
        transliteration_operations,
        bang_preserving_operations,
        resolved_char_whitelist,
    )


def _resolve_company_type_matcher_runtime(
    *,
    company_type_matcher: str,
    company_type_regex: str,
    company_type_mapping: dict[str, str],
) -> tuple[str, re.Pattern[str] | None, dict[str, object] | None, int]:
    matcher_mode = (company_type_matcher or "regex").strip().lower()
    if matcher_mode not in {"regex", "trie"}:
        raise ValueError("company_type_matcher must be 'regex' or 'trie'.")

    if matcher_mode == "trie":
        # Benchmark note (2026-07-01): terminal-token trie prefiltering before this
        # suffix matcher showed no net win on real files (offeneregister/fr/gb).
        suffix_trie, suffix_trie_max_tokens = _cached_suffix_trie(
            _freeze_mapping(company_type_mapping)
        )
        return matcher_mode, None, suffix_trie, suffix_trie_max_tokens

    compiled_company_type_regex = _cached_compiled_company_type_regex(
        company_type_regex
    )
    return matcher_mode, compiled_company_type_regex, None, 0


def _resolve_normalized_source_company_type_mapping(
    *,
    use_source_company_type: bool,
    source_company_type_mapping: dict[str, str] | None,
) -> dict[str, str] | None:
    if not use_source_company_type or source_company_type_mapping is None:
        return None

    frozen_source_mapping = tuple(
        sorted(
            (key.strip(), str(value))
            for key, value in source_company_type_mapping.items()
            if isinstance(key, str) and value is not None
        )
    )
    return _cached_normalized_source_company_type_mapping(frozen_source_mapping)


def _step_prepare_source_company_type_raw(
    lf: pl.LazyFrame,
    *,
    use_source_company_type: bool,
    source_company_type_col: str | None,
) -> pl.LazyFrame:
    if use_source_company_type:
        assert source_company_type_col is not None, (  # nosec B101 - narrows Optional for mypy
            "use_source_company_type=True requires a non-None source_company_type_col"
        )
        return lf.with_columns(
            [
                pl.col(source_company_type_col)
                .cast(pl.Utf8)
                .str.strip_chars()
                .alias("_source_company_type_raw")
            ]
        )

    return lf.with_columns(
        [
            pl.lit(None, dtype=pl.Utf8).alias("_source_company_type_raw"),
        ]
    )


def _finalize_output_column_order(
    lf: pl.LazyFrame,
    *,
    input_columns: list[str],
    schema_names: list[str] | None = None,
) -> pl.LazyFrame:
    if schema_names is None:
        schema_names = lf.collect_schema().names()
    schema_set = set(schema_names)

    # Preserve incoming canonical/source order, then append cleanse-derived fields.
    preserved_input_order = [
        column_name for column_name in input_columns if column_name in schema_set
    ]
    derived_order = [
        "short_name",
        "quoted_name",
        "acronym",
        "name_cleansed_basic",
        "name_cleansed",
        "personal_owner",
        "company_type_source",
        "company_type_missing",
    ]
    appended_derived = [
        column_name
        for column_name in derived_order
        if column_name in schema_set and column_name not in preserved_input_order
    ]

    selected = set(preserved_input_order).union(appended_derived)
    remaining = sorted(
        column_name for column_name in schema_names if column_name not in selected
    )

    ordered_columns = preserved_input_order + appended_derived + remaining
    return lf.select([pl.col(column_name) for column_name in ordered_columns])


def _process_cleanse_lazyframe(
    lf: pl.LazyFrame,
    company_col: str,
    company_type_regex: str,
    company_type_mapping: dict[str, str],
    company_type_matcher: str,
    char_whitelist: str,
    source_company_type_col: str | None,
    source_company_type_mapping: dict[str, str] | None,
    use_source_company_type: bool,
    normalization_profile: str = "default",
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
    short_name_profile: str = DEFAULT_NORMALIZATION_PROFILE,
    and_tokens: tuple[str, ...] | None = None,
    personal_owner_markers: tuple[str, ...] | None = None,
    step_engines: dict[str, str] | None = None,
    input_columns: list[str] | None = None,
    jurisdiction_col: str | None = None,
) -> pl.LazyFrame:
    if input_columns is None:
        input_columns = lf.collect_schema().names()

    # Only activate the per-jurisdiction corpus noise-word lookup when
    # the configured column name is actually present in this input -- an
    # input without it (most standalone/single-string usage) falls through
    # to _derive_short_name_from_cleansed's own jurisdiction_code=None
    # default, unchanged from before this column-detection was added.
    resolved_jurisdiction_col = (
        jurisdiction_col if jurisdiction_col in input_columns else None
    )

    (
        normalization_operations,
        transliteration_operations,
        bang_preserving_operations,
        char_whitelist,
    ) = _resolve_runtime_normalization(
        normalization_profile=normalization_profile,
        char_whitelist=char_whitelist,
    )

    lf_cleansed = _step_prepare_stripped_name(
        lf,
        company_col,
    )
    lf_cleansed = _step_extract_personal_owner(lf_cleansed, personal_owner_markers)
    lf_cleansed = _step_extract_special_and_lead_fields(
        lf_cleansed, company_type_mapping
    )
    lf_cleansed = _step_build_name_body(lf_cleansed)
    lf_cleansed = _step_prepare_normalized_name_fields(
        lf_cleansed,
        and_tokens,
        bang_preserving_operations,
    )
    lf_cleansed = _step_build_name_cleansed_basic(lf_cleansed, char_whitelist)

    (
        matcher_mode,
        compiled_company_type_regex,
        suffix_trie,
        suffix_trie_max_tokens,
    ) = _resolve_company_type_matcher_runtime(
        company_type_matcher=company_type_matcher,
        company_type_regex=company_type_regex,
        company_type_mapping=company_type_mapping,
    )

    normalized_source_company_type_mapping = (
        _resolve_normalized_source_company_type_mapping(
            use_source_company_type=use_source_company_type,
            source_company_type_mapping=source_company_type_mapping,
        )
    )

    lf_cleansed = _step_prepare_source_company_type_raw(
        lf_cleansed,
        use_source_company_type=use_source_company_type,
        source_company_type_col=source_company_type_col,
    )

    lf_cleansed = _step_resolve_and_unpack_company_type_decision(
        lf_cleansed,
        matcher_mode=matcher_mode,
        company_type_regex=compiled_company_type_regex,
        suffix_trie=suffix_trie,
        suffix_trie_max_tokens=suffix_trie_max_tokens,
        and_tokens=and_tokens,
        company_type_mapping=company_type_mapping,
        normalization_operations=normalization_operations,
        transliteration_operations=transliteration_operations,
        use_source_company_type=use_source_company_type,
        normalized_source_company_type_mapping=normalized_source_company_type_mapping,
    )
    lf_cleansed = _run_post_company_type_steps(
        lf_cleansed,
        company_type_mapping=company_type_mapping,
        step_engines=step_engines,
        and_tokens=and_tokens,
        char_whitelist=char_whitelist,
        bang_preserving_operations=bang_preserving_operations,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        short_name_profile=short_name_profile,
        jurisdiction_col=resolved_jurisdiction_col,
    )
    lf_cleansed = _finalize_cleanse_output(
        lf_cleansed,
        input_columns=input_columns,
    )

    return lf_cleansed


def _write_cleanse_output(lf_cleansed: pl.LazyFrame, output_path: str) -> int:
    df = lf_cleansed.collect()
    df.write_parquet(output_path, compression="snappy")
    return len(df)
