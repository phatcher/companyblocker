from __future__ import annotations

import re

import polars as pl

from .wikidata_name_rows import NAME_ROW_COLUMNS

_PREVIOUS_NAME_COLUMN_RE = re.compile(r"^PreviousName_(\d+)\.CompanyName$")
_GB_SYSTEM_URI_PREFIX = "gb://"
_SOURCE_TYPE = "PREVIOUS_NAME"
_DERIVATION_NOTE = "Companies House PreviousName_N sidecar field"


def _previous_name_columns_in_slot_order(columns: list[str]) -> list[str]:
    matches = ((_PREVIOUS_NAME_COLUMN_RE.match(name), name) for name in columns)
    slots = [
        (int(match.group(1)), name) for match, name in matches if match is not None
    ]
    slots.sort(key=lambda slot: slot[0])
    return [name for _, name in slots]


def derive_gb_name_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Explode a GB sidecar shard frame's `PreviousName_N.CompanyName` slots
    into one row per previous name.

    Pure: no I/O, no plan lookup, no filesystem. Always returns exactly
    NAME_ROW_COLUMNS, in that order, with a `pl.Utf8` dtype on every column
    (including an all-null `language_code` -- Companies House doesn't supply
    one), so an empty result concatenates and round-trips through parquet
    identically to a populated one.

    Tolerates any number of `PreviousName_N.CompanyName` slots present in
    the frame (Companies House's export currently has 10), skipping the
    derivation entirely if none are found rather than raising. Requires
    `system_uri` -- always present in real GB sidecar shard output -- and
    raises if it is missing, since that indicates the wrong file was handed
    in, not an absent optional field.

    `id` is reconstructed from `system_uri` by stripping the `gb://` scheme
    prefix, reproducing the source CompanyNumber for the normal case (GB's
    CompanyNumber is effectively always present, so the `gb://fp/<hash>`
    fallback URI shape does not arise in practice here).
    """
    if "system_uri" not in frame.columns:
        raise RuntimeError(
            "gb name-row derivation requires column 'system_uri', which is "
            f"missing from the input frame (columns: {frame.columns})"
        )

    empty_result = pl.DataFrame(schema={column: pl.Utf8 for column in NAME_ROW_COLUMNS})

    slot_columns = _previous_name_columns_in_slot_order(frame.columns)
    if not slot_columns:
        return empty_result

    parts: list[pl.DataFrame] = []
    for slot_column in slot_columns:
        part = (
            frame.select(
                pl.col("system_uri"),
                pl.col(slot_column).str.strip_chars().alias("name"),
            )
            .filter(pl.col("name").is_not_null() & (pl.col("name") != ""))
            .with_columns(
                pl.col("system_uri")
                .str.strip_prefix(_GB_SYSTEM_URI_PREFIX)
                .alias("id"),
                pl.lit(_SOURCE_TYPE, dtype=pl.Utf8).alias("source_type"),
                pl.lit(None, dtype=pl.Utf8).alias("language_code"),
                pl.lit(_DERIVATION_NOTE, dtype=pl.Utf8).alias("derivation_note"),
            )
        )
        parts.append(part.select(NAME_ROW_COLUMNS))

    return pl.concat(parts, how="vertical")
