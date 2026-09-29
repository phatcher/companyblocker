from __future__ import annotations

from dataclasses import dataclass

import polars as pl

NAME_ROW_COLUMNS: tuple[str, ...] = (
    "system_uri",
    "id",
    "name",
    "source_type",
    "language_code",
    "derivation_note",
)
"""Column contract for a Wikidata name-variant row.

Mirrors GLEIF's Shard-stage names rows (`_extract_gleif_name_rows` in
sharding_gleif.py) exactly, with `id` (the QID) standing in for GLEIF's `LEI`
as the source-native identifier, so `canonicalize_system_name_rows` consumes
both unchanged.
"""


@dataclass(frozen=True)
class _WikidataNameFieldSpec:
    column: str
    source_type: str
    is_struct_list: bool
    """True for official_name_variants/short_name_variants (list[{value,
    language}], real per-value language). False for label_en (scalar Utf8)
    and aliases_en (list[str]), which are still English-only by design,
    and get a fixed `language_code`.
    """
    language_code: str | None
    derivation_note: str
    is_list: bool


_WIKIDATA_NAME_FIELD_SPECS: tuple[_WikidataNameFieldSpec, ...] = (
    _WikidataNameFieldSpec(
        column="label_en",
        source_type="LABEL_EN",
        is_struct_list=False,
        language_code="en",
        derivation_note="wikidata labels.en",
        is_list=False,
    ),
    _WikidataNameFieldSpec(
        column="aliases_en",
        source_type="ALIAS_EN",
        is_struct_list=False,
        language_code="en",
        derivation_note="wikidata aliases.en",
        is_list=True,
    ),
    _WikidataNameFieldSpec(
        column="official_name_variants",
        source_type="OFFICIAL_NAME",
        is_struct_list=True,
        language_code=None,
        derivation_note="wikidata P1448",
        is_list=True,
    ),
    _WikidataNameFieldSpec(
        column="short_name_variants",
        source_type="SHORT_NAME",
        is_struct_list=True,
        language_code=None,
        derivation_note="wikidata P1813",
        is_list=True,
    ),
)

WIKIDATA_NAME_SOURCE_TYPES: frozenset[str] = frozenset(
    spec.source_type for spec in _WIKIDATA_NAME_FIELD_SPECS
)


def _derive_scalar_part(
    frame: pl.DataFrame, spec: _WikidataNameFieldSpec
) -> pl.DataFrame:
    return frame.select(
        pl.col("system_uri"),
        pl.col("id"),
        pl.col(spec.column).str.strip_chars().alias("name"),
    )


def _derive_list_str_part(
    frame: pl.DataFrame, spec: _WikidataNameFieldSpec
) -> pl.DataFrame:
    return (
        frame.select(
            pl.col("system_uri"),
            pl.col("id"),
            pl.col(spec.column)
            .list.eval(pl.element().str.strip_chars())
            .list.unique(maintain_order=True)
            .alias("name"),
        )
        .explode("name")
        .with_columns(pl.lit(spec.language_code, dtype=pl.Utf8).alias("language_code"))
    )


def _derive_struct_list_part(
    frame: pl.DataFrame, spec: _WikidataNameFieldSpec
) -> pl.DataFrame:
    # `list.unique()` on the struct list dedupes identical (value, language)
    # pairs within one entity's own list before exploding, so a value
    # appearing twice under the same language (real Wikidata data does this)
    # doesn't produce duplicate rows. A value under two *different* languages
    # is a distinct struct and is kept as two rows.
    return (
        frame.select(
            pl.col("system_uri"),
            pl.col("id"),
            pl.col(spec.column).list.unique(maintain_order=True).alias("variant"),
        )
        .explode("variant")
        .filter(pl.col("variant").is_not_null())
        .with_columns(
            pl.col("variant").struct.field("value").str.strip_chars().alias("name"),
            pl.col("variant").struct.field("language").alias("language_code"),
        )
        .drop("variant")
    )


def derive_wikidata_name_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Explode one Wikidata Shard-stage frame into per-name-variant rows.

    Pure: no I/O, no plan lookup, no filesystem. Always returns exactly
    NAME_ROW_COLUMNS, in that order, with a `pl.Utf8` dtype on every column
    (including an all-null `language_code`), so an empty result concatenates
    and round-trips through parquet identically to a populated one.

    Tolerates a missing optional column (e.g. an older shard snapshot
    predating a given field) by skipping that spec rather than raising.
    Requires `system_uri` and `id` -- both are always present in real
    Wikidata shard output -- and raises if either is missing, since that
    indicates the wrong file was handed in, not an absent optional field.
    """
    missing_keys = {"system_uri", "id"} - set(frame.columns)
    if missing_keys:
        raise RuntimeError(
            f"wikidata name-row derivation requires columns {sorted(missing_keys)}, "
            f"which are missing from the input frame (columns: {frame.columns})"
        )

    empty_result = pl.DataFrame(schema={column: pl.Utf8 for column in NAME_ROW_COLUMNS})

    parts: list[pl.DataFrame] = []
    for spec in _WIKIDATA_NAME_FIELD_SPECS:
        if spec.column not in frame.columns:
            continue

        if spec.is_struct_list:
            part = _derive_struct_list_part(frame, spec)
        elif spec.is_list:
            part = _derive_list_str_part(frame, spec)
        else:
            part = _derive_scalar_part(frame, spec)
            part = part.with_columns(
                pl.lit(spec.language_code, dtype=pl.Utf8).alias("language_code")
            )

        part = part.filter(pl.col("name").is_not_null() & (pl.col("name") != ""))
        part = part.with_columns(
            pl.lit(spec.source_type, dtype=pl.Utf8).alias("source_type"),
            pl.lit(spec.derivation_note, dtype=pl.Utf8).alias("derivation_note"),
        )
        parts.append(part.select(NAME_ROW_COLUMNS))

    if not parts:
        return empty_result

    return pl.concat(parts, how="vertical")
