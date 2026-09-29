from __future__ import annotations

import polars as pl

from .wikidata_name_rows import NAME_ROW_COLUMNS

_OFFENEREGISTER_SYSTEM_URI_PREFIX = "offeneregister://"
_SOURCE_TYPE = "PREVIOUS_NAME"
_DERIVATION_NOTE = "OffeneRegister previous_names sidecar field"


def derive_offeneregister_name_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Explode an OffeneRegister sidecar shard frame's `previous_names_list`
    column into one row per previous name.

    Pure: no I/O, no plan lookup, no filesystem. Always returns exactly
    NAME_ROW_COLUMNS, in that order, with a `pl.Utf8` dtype on every column
    (including an all-null `language_code` -- OffeneRegister doesn't supply
    one), so an empty result concatenates and round-trips through parquet
    identically to a populated one.

    Mirrors `derive_gb_name_rows`: tolerates a missing
    `previous_names_list` column (e.g. an older shard snapshot predating
    this field) by returning the empty result rather than raising.
    Requires `system_uri` -- always present in real OffeneRegister sidecar
    shard output -- and raises if it is missing, since that indicates the
    wrong file was handed in, not an absent optional field.

    `id` is reconstructed from `system_uri` by stripping the
    `offeneregister://` scheme prefix, the same way GB's deriver
    reconstructs `id` from its own `gb://` prefix.
    """
    if "system_uri" not in frame.columns:
        raise RuntimeError(
            "offeneregister name-row derivation requires column 'system_uri', "
            f"which is missing from the input frame (columns: {frame.columns})"
        )

    empty_result = pl.DataFrame(schema={column: pl.Utf8 for column in NAME_ROW_COLUMNS})

    if "previous_names_list" not in frame.columns:
        return empty_result

    result = (
        frame.select(
            pl.col("system_uri"),
            pl.col("previous_names_list")
            .list.eval(pl.element().str.strip_chars())
            .list.unique(maintain_order=True)
            .alias("name"),
        )
        .explode("name")
        .filter(pl.col("name").is_not_null() & (pl.col("name") != ""))
        .with_columns(
            pl.col("system_uri")
            .str.strip_prefix(_OFFENEREGISTER_SYSTEM_URI_PREFIX)
            .alias("id"),
            pl.lit(_SOURCE_TYPE, dtype=pl.Utf8).alias("source_type"),
            pl.lit(None, dtype=pl.Utf8).alias("language_code"),
            pl.lit(_DERIVATION_NOTE, dtype=pl.Utf8).alias("derivation_note"),
        )
    )

    return result.select(NAME_ROW_COLUMNS)
