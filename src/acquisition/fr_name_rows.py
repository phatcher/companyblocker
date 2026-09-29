from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .wikidata_name_rows import NAME_ROW_COLUMNS

_FR_SYSTEM_URI_PREFIX = "fr://"

# INSEE's SIRENE export uses the literal string "[ND]" ("non diffusible" --
# not disclosable) as a sentinel value on every one of these fields when the
# real value exists but is withheld from public redistribution (routinely
# the case for a physical-person unite legale). It is not a real name and
# must never surface as a name-variant row: confirmed on a real 2M-row
# sample (2026-08-31) that "[ND]" accounts for essentially all of the
# nominal ~9-18% raw population rate on every one of these fields --
# genuine post-filter population is far lower (2.65% sigleUniteLegale,
# 3.09% denominationUsuelle1UniteLegale, ~0.005% denominationUsuelle2/3,
# 9.09% nomUsageUniteLegale, 0.19% pseudonymeUniteLegale).
_ND_SENTINEL = "[ND]"


@dataclass(frozen=True)
class _FrNameFieldSpec:
    columns: tuple[str, ...]
    """One or more source columns sharing the same source_type/derivation
    (denominationUsuelle1/2/3UniteLegale are three slots of the same "usual
    name" field, exactly like GB's PreviousName_N slots)."""

    source_type: str
    derivation_note: str


_FR_NAME_FIELD_SPECS: tuple[_FrNameFieldSpec, ...] = (
    _FrNameFieldSpec(
        columns=("sigleUniteLegale",),
        source_type="ACRONYM",
        derivation_note="SIRENE sigleUniteLegale",
    ),
    _FrNameFieldSpec(
        columns=(
            "denominationUsuelle1UniteLegale",
            "denominationUsuelle2UniteLegale",
            "denominationUsuelle3UniteLegale",
        ),
        source_type="USUAL_NAME",
        derivation_note="SIRENE denominationUsuelleNUniteLegale",
    ),
    _FrNameFieldSpec(
        columns=("nomUsageUniteLegale",),
        source_type="USAGE_NAME",
        derivation_note="SIRENE nomUsageUniteLegale",
    ),
    _FrNameFieldSpec(
        columns=("pseudonymeUniteLegale",),
        source_type="PSEUDONYM",
        derivation_note="SIRENE pseudonymeUniteLegale",
    ),
)

FR_NAME_SOURCE_TYPES: frozenset[str] = frozenset(
    spec.source_type for spec in _FR_NAME_FIELD_SPECS
)


def derive_fr_name_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Explode a FR (SIRENE) sidecar shard frame's alternate-name fields
    into one row per real name variant.

    Pure: no I/O, no plan lookup, no filesystem. Always returns exactly
    NAME_ROW_COLUMNS, in that order, with a `pl.Utf8` dtype on every column
    (including an all-null `language_code` -- SIRENE doesn't supply one), so
    an empty result concatenates and round-trips through parquet identically
    to a populated one.

    Filters out both null/blank values and INSEE's "[ND]" (non-disclosable)
    sentinel -- see `_ND_SENTINEL`'s docstring -- so every emitted row is a
    genuine name, not a redaction placeholder.

    Tolerates any subset of the known source columns being absent (skips
    that spec's contribution rather than raising) so an older/narrower
    sidecar snapshot degrades gracefully. Requires `system_uri` -- always
    present in real FR sidecar shard output -- and raises if it is missing,
    since that indicates the wrong file was handed in, not an absent
    optional field.

    `id` is reconstructed from `system_uri` by stripping the `fr://` scheme
    prefix, reproducing the source siren for the normal case (FR's siren is
    effectively always present, so the `fr://fp/<hash>` fallback URI shape
    does not arise in practice here).
    """
    if "system_uri" not in frame.columns:
        raise RuntimeError(
            "fr name-row derivation requires column 'system_uri', which is "
            f"missing from the input frame (columns: {frame.columns})"
        )

    empty_result = pl.DataFrame(schema={column: pl.Utf8 for column in NAME_ROW_COLUMNS})

    parts: list[pl.DataFrame] = []
    for spec in _FR_NAME_FIELD_SPECS:
        present_columns = [column for column in spec.columns if column in frame.columns]
        for source_column in present_columns:
            part = (
                frame.select(
                    pl.col("system_uri"),
                    pl.col(source_column).str.strip_chars().alias("name"),
                )
                .filter(
                    pl.col("name").is_not_null()
                    & (pl.col("name") != "")
                    & (pl.col("name") != _ND_SENTINEL)
                )
                .with_columns(
                    pl.col("system_uri")
                    .str.strip_prefix(_FR_SYSTEM_URI_PREFIX)
                    .alias("id"),
                    pl.lit(spec.source_type, dtype=pl.Utf8).alias("source_type"),
                    pl.lit(None, dtype=pl.Utf8).alias("language_code"),
                    pl.lit(spec.derivation_note, dtype=pl.Utf8).alias(
                        "derivation_note"
                    ),
                )
            )
            parts.append(part.select(NAME_ROW_COLUMNS))

    if not parts:
        return empty_result

    return pl.concat(parts, how="vertical")
