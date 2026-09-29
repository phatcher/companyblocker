from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

import polars as pl

from .authority_scoped_ids import get_crosswalk
from .canonical_system_config import (
    GLEIF_COMPANY_NUMBER_AUTHORITY_SCOPED_JURISDICTIONS,
    GLEIF_COMPANY_NUMBER_PAD_WIDTH,
    GLEIF_VALID_REGISTRATION_AUTHORITIES,
    OFFENEREGISTER_PLACEHOLDER_REGISTER_NUMBER_RANGE,
    get_system_canonical_config,
)
from .canonical_utils import (
    clean_text,
    coalesce_date_iso,
    coalesce_name_list,
    coalesce_text,
    compose_gb_registered_address,
    compose_gleif_registered_address,
    derive_inactive,
    resolve_wikidata_country_iso2_expr,
    resolve_wikidata_name_bundle_expr,
)


def _compose_fr_name(columns: set[str]) -> pl.Expr:
    company_name_expr = coalesce_text(
        columns,
        ["denominationUniteLegale", "raisonSociale"],
    )
    given_name_expr = coalesce_text(
        columns,
        ["prenomUsuelUniteLegale", "prenom1UniteLegale", "prenom2UniteLegale"],
    )
    surname_expr = coalesce_text(
        columns,
        ["nomUsageUniteLegale", "nomUniteLegale", "nom"],
    )
    person_name_expr = clean_text(
        pl.concat_str([given_name_expr, surname_expr], separator=" ", ignore_nulls=True)
    )
    return pl.coalesce([company_name_expr, person_name_expr, surname_expr])


def _resolve_name_expr(
    *,
    columns: set[str],
    field_candidates: dict[str, list[str]],
    config_name_mode: str,
) -> tuple[pl.Expr, pl.Expr | None]:
    if config_name_mode == "fr":
        return _compose_fr_name(columns), None
    if config_name_mode == "wikidata":
        bundle_expr = resolve_wikidata_name_bundle_expr(columns)
        name_expr = pl.coalesce(
            [
                bundle_expr.struct.field("name"),
                coalesce_text(
                    columns,
                    _field_candidates_or_same_name(columns, field_candidates, "name"),
                ),
            ]
        )
        return name_expr, bundle_expr
    return coalesce_text(
        columns, _field_candidates_or_same_name(columns, field_candidates, "name")
    ), None


def _field_candidates_or_same_name(
    columns: set[str],
    field_candidates: dict[str, list[str]],
    field_name: str,
) -> list[str]:
    candidates = list(field_candidates.get(field_name, []))
    if field_name in columns and field_name not in candidates:
        candidates.append(field_name)
    return candidates


def _resolve_registered_address_expr(
    *,
    columns: set[str],
    field_candidates: dict[str, list[str]],
    mode: str,
) -> pl.Expr:
    if mode == "none":
        return pl.lit(None, dtype=pl.Utf8)
    if mode == "gb":
        return compose_gb_registered_address(columns)
    if mode == "gleif":
        return compose_gleif_registered_address(columns)
    return coalesce_text(
        columns,
        _field_candidates_or_same_name(
            columns, field_candidates, "registered_address_in_full"
        ),
    )


def _resolve_jurisdiction_expr(
    *,
    country: str,
    columns: set[str],
    mode: str,
    candidates: tuple[str, ...],
) -> pl.Expr:
    if mode == "wikidata_qid":
        return resolve_wikidata_country_iso2_expr(columns)
    if mode == "candidates":
        return coalesce_text(columns, list(candidates))
    return pl.lit(country)


def _resolve_registry_url_expr(
    *,
    columns: set[str],
    field_candidates: dict[str, list[str]],
    mode: str,
) -> pl.Expr:
    if mode == "gleif_lei":
        lei_expr = coalesce_text(
            columns,
            _field_candidates_or_same_name(columns, field_candidates, "company_number"),
        )
        return pl.coalesce(
            [
                coalesce_text(
                    columns,
                    _field_candidates_or_same_name(
                        columns, field_candidates, "registry_url"
                    ),
                ),
                pl.when(lei_expr.is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit("https://search.gleif.org/#/record/"), lei_expr]
                    )
                )
                .otherwise(pl.lit(None, dtype=pl.Utf8)),
            ]
        )

    if mode == "wikidata_entity":
        entity_id_expr = coalesce_text(columns, ["id"])
        wikidata_registry_candidates = [
            candidate
            for candidate in _field_candidates_or_same_name(
                columns, field_candidates, "registry_url"
            )
            if candidate != "id"
        ]
        return pl.coalesce(
            [
                coalesce_text(columns, wikidata_registry_candidates),
                pl.when(entity_id_expr.is_not_null())
                .then(
                    pl.concat_str(
                        [pl.lit("https://www.wikidata.org/wiki/"), entity_id_expr]
                    )
                )
                .otherwise(pl.lit(None, dtype=pl.Utf8)),
            ]
        )

    return coalesce_text(
        columns,
        _field_candidates_or_same_name(columns, field_candidates, "registry_url"),
    )


def apply_canonical_source_aliases(
    frame: pl.DataFrame,
    *,
    aliases: tuple[tuple[str, str], ...] | None,
) -> pl.DataFrame:
    if not aliases:
        return frame

    rename_map: dict[str, str] = {}
    for source_name, target_name in aliases:
        if source_name in frame.columns and target_name not in frame.columns:
            rename_map[source_name] = target_name

    if not rename_map:
        return frame

    return frame.rename(rename_map)


def _resolve_derived_concat_expr(
    *,
    columns: set[str],
    candidates: tuple[str, ...],
    separator: str,
) -> pl.Expr:
    part_exprs = [coalesce_text(columns, [candidate]) for candidate in candidates]
    joined = clean_text(
        pl.concat_str(part_exprs, separator=separator, ignore_nulls=True)
    )
    return (
        pl.when(joined.str.strip_chars() == "")
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(joined)
    )


def _resolve_derived_field_expr(
    *,
    columns: set[str],
    field_name: str,
    fallback_candidates: list[str],
    canonical_derived_fields: tuple[tuple[str, tuple[str, ...], str], ...] | None,
) -> pl.Expr:
    if canonical_derived_fields:
        for derived_field_name, derived_parts, separator in canonical_derived_fields:
            if derived_field_name == field_name:
                return _resolve_derived_concat_expr(
                    columns=columns,
                    candidates=derived_parts,
                    separator=separator,
                )
    return coalesce_text(columns, fallback_candidates)


_GLEIF_REGISTRATION_AUTHORITY_ID_COLUMN = (
    "Entity_RegistrationAuthority_RegistrationAuthorityID"
)

# Wikidata's real, per-jurisdiction company-number source columns
# (extracted at Shard stage from each jurisdiction's own Wikidata property --
# see `wikidata_projection_helpers.py`'s `_TEXT_PROPERTIES`). No single
# Wikidata property covers `company_number` the way P1278 covers `lei`, so
# each jurisdiction reads its own claim-derived column, picked by the row's
# own resolved `jurisdiction_code` -- mirrors the jurisdiction-scoped
# `company_number_key` normalization in `match_ops.py`, one stage further
# down the pipeline. IE has no confirmed source property (the population
# survey found none); deliberately left out rather than
# guessed, so `jurisdiction_code == "IE"` falls through to null below.
_WIKIDATA_COMPANY_NUMBER_JURISDICTION_COLUMNS: dict[str, str] = {
    "GB": "company_number_gb",  # P2622, Companies House company ID
    "FR": "company_number_fr",  # P1616, SIREN number
    "DE": "company_number_de",  # P12012, record number (Germany)
}


def _resolve_wikidata_jurisdiction_scoped_company_number_expr(
    *,
    columns: set[str],
    jurisdiction_expr: pl.Expr,
) -> pl.Expr:
    """Wikidata's canonical `company_number`, picked per resolved jurisdiction.

    Deliberately does not fall back to `lei` for jurisdictions with no known
    source column, which would silently copy a global LEI into
    `company_number` for every row that carried one -- an entity with no
    jurisdiction-specific claim resolves to null, not a guessed value.
    """
    result_expr = pl.lit(None, dtype=pl.Utf8)
    for (
        jurisdiction,
        column_name,
    ) in _WIKIDATA_COMPANY_NUMBER_JURISDICTION_COLUMNS.items():
        result_expr = (
            pl.when(jurisdiction_expr == jurisdiction)
            .then(coalesce_text(columns, [column_name]))
            .otherwise(result_expr)
        )
    return result_expr


def _resolve_company_number_expr(
    *,
    columns: set[str],
    field_candidates: dict[str, list[str]],
    mode: str,
    jurisdiction_expr: pl.Expr,
    canonical_derived_fields: tuple[tuple[str, tuple[str, ...], str], ...] | None,
) -> pl.Expr:
    if mode == "wikidata_jurisdiction_scoped":
        return _resolve_wikidata_jurisdiction_scoped_company_number_expr(
            columns=columns, jurisdiction_expr=jurisdiction_expr
        )

    default_expr = _resolve_derived_field_expr(
        columns=columns,
        field_name="company_number",
        fallback_candidates=_field_candidates_or_same_name(
            columns, field_candidates, "company_number"
        ),
        canonical_derived_fields=canonical_derived_fields,
    )

    if mode == "offeneregister_authority_scoped":
        return _resolve_offeneregister_authority_scoped_company_number_expr(
            columns=columns, derived_number_expr=default_expr
        )

    if (
        mode != "gleif_ra_filtered"
        or _GLEIF_REGISTRATION_AUTHORITY_ID_COLUMN not in columns
    ):
        return default_expr

    authority_expr = coalesce_text(columns, [_GLEIF_REGISTRATION_AUTHORITY_ID_COLUMN])

    allowed_expr = pl.lit(True)
    for jurisdiction, authority_codes in GLEIF_VALID_REGISTRATION_AUTHORITIES.items():
        allowed_expr = (
            pl.when(jurisdiction_expr == jurisdiction)
            .then(authority_expr.is_in(list(authority_codes)))
            .otherwise(allowed_expr)
        )

    filtered_expr = (
        pl.when(allowed_expr).then(default_expr).otherwise(pl.lit(None, dtype=pl.Utf8))
    )

    padded_expr = filtered_expr
    for jurisdiction, width in GLEIF_COMPANY_NUMBER_PAD_WIDTH.items():
        is_short_numeric = (
            filtered_expr.is_not_null()
            & filtered_expr.str.contains(r"^[0-9]+$")
            & (filtered_expr.str.len_chars() < width)
        )
        padded_expr = (
            pl.when(jurisdiction_expr == jurisdiction)
            .then(
                pl.when(is_short_numeric)
                .then(filtered_expr.str.zfill(width))
                .otherwise(padded_expr)
            )
            .otherwise(padded_expr)
        )

    scoped_expr = padded_expr
    for jurisdiction in GLEIF_COMPANY_NUMBER_AUTHORITY_SCOPED_JURISDICTIONS:
        # Whitespace is stripped from the scoped number so it compares equal to
        # offeneregister's raw (unspaced) register-id segment -- the two sides
        # otherwise format the same value differently ("HRB 150148" vs
        # "HRB150148"), which would silently defeat the join.
        scoped_expr = (
            pl.when(jurisdiction_expr == jurisdiction)
            .then(
                pl.when(padded_expr.is_not_null())
                .then(
                    pl.concat_str(
                        [authority_expr, padded_expr.str.replace_all(r"\s+", "")],
                        separator="|",
                    )
                )
                .otherwise(pl.lit(None, dtype=pl.Utf8))
            )
            .otherwise(scoped_expr)
        )

    return scoped_expr


def _resolve_offeneregister_authority_scoped_company_number_expr(
    *,
    columns: set[str],
    derived_number_expr: pl.Expr,
) -> pl.Expr:
    """Fold offeneregister's court-code prefix into an RA-scoped company_number.

    Offeneregister's raw ``company_number`` embeds a court (XJustizId) prefix
    (for example ``K1101R_HRB150148``) ahead of the register id. The last
    underscore-segment before that is taken as the current court -- Germany's
    Amtsgericht mergers can leave a former court segment ahead of it (for
    example ``H1101_H1101_HRB18423``), and the most recent one wins.

    The register id itself is taken from this same raw segment (``HRB150148``),
    not from ``derived_number_expr``'s separately shard-derived register_number
    -- that extraction drops a trailing court-merger disambiguation suffix
    (analogous to GLEIF's own "HRB 34542 HB" case), which silently collided
    genuinely different companies onto the same bare number once court-scoping
    made the rest of the key correct (found via real duplicate-key rows: five
    distinct companies at one court all reduced to "HRB 1162" once their
    "RZ"/"SB"/"OL"/... suffixes were dropped).

    Falls back to the unscoped ``derived_number_expr`` when there's no raw
    prefixed value to extract a court from at all (no scoping attempted); nulls
    out when a court prefix is present but not in the crosswalk (scoping was
    attempted and failed) -- an unresolved court must not silently degrade to a
    bare number that can collide with a different court's own numbering.

    Also nulls out when the register id's numeric part falls in
    ``OFFENEREGISTER_PLACEHOLDER_REGISTER_NUMBER_RANGE`` -- a confirmed
    upstream offeneregister.de defect where one historical scrape batch
    assigned the same placeholder number across unrelated companies (see that
    constant's docstring). No amount of court-scoping fixes a number that was
    never a real register id.
    """
    if "company_number" not in columns:
        return derived_number_expr

    raw_expr = coalesce_text(columns, ["company_number"])
    segments_expr = raw_expr.str.split("_")
    n_segments_expr = segments_expr.list.len()
    court_code_expr = (
        pl.when(n_segments_expr >= 2)
        .then(segments_expr.list.get(-2, null_on_oob=True))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )
    register_id_expr = (
        pl.when(n_segments_expr >= 2)
        .then(segments_expr.list.get(-1, null_on_oob=True))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )
    ra_code_expr = court_code_expr.replace_strict(
        get_crosswalk("xjustiz"), default=None, return_dtype=pl.Utf8
    )

    placeholder_min, placeholder_max = OFFENEREGISTER_PLACEHOLDER_REGISTER_NUMBER_RANGE
    register_number_value_expr = register_id_expr.str.extract(r"(\d+)", 1).cast(
        pl.Int64, strict=False
    )
    is_placeholder_expr = register_number_value_expr.is_between(
        placeholder_min, placeholder_max, closed="left"
    ).fill_null(False)

    return (
        pl.when(court_code_expr.is_null())
        .then(derived_number_expr)
        .when(is_placeholder_expr)
        .then(pl.lit(None, dtype=pl.Utf8))
        .when(ra_code_expr.is_not_null() & register_id_expr.is_not_null())
        .then(pl.concat_str([ra_code_expr, register_id_expr], separator="|"))
        .otherwise(pl.lit(None, dtype=pl.Utf8))
    )


def _resolve_lei_expr(*, columns: set[str], country: str) -> pl.Expr:
    """Resolve the canonical `lei` column.

    GLEIF's own `LEI` column is the authoritative source. Wikidata also
    carries a genuine LEI value via its P1278 claim -- sparse (~5.7% of
    company rows in the 2026-07-16 snapshot) but real, already extracted
    into a `lei` list column at Shard stage. `coalesce_text` already knows
    how to take the first non-empty string out of a list-typed column, the
    same way `company_number`'s own resolution reads that column today.
    Every other system has no LEI-shaped field at all.
    """
    if country == "gleif":
        return coalesce_text(columns, ["LEI"])
    if country == "wikidata":
        return coalesce_text(columns, ["lei"])
    return pl.lit(None, dtype=pl.Utf8)


def canonicalize_frame(
    frame: pl.DataFrame,
    *,
    country: str,
    source_company_type_col: str | None,
    exclude_company_type_values: set[str],
    canonical_source_column_aliases: tuple[tuple[str, str], ...] | None,
    system_field_candidates: dict[str, list[str]],
    canonical_derived_fields: tuple[tuple[str, tuple[str, ...], str], ...] | None,
    company_type_mapping_resolver: Callable[[str], dict[str, str]],
    project_identity_columns: list[str],
    opencorporates_columns: list[str],
    metadata_columns: list[str],
) -> pl.DataFrame:
    frame = apply_canonical_source_aliases(
        frame, aliases=canonical_source_column_aliases
    )

    if (
        source_company_type_col
        and exclude_company_type_values
        and source_company_type_col in frame.columns
    ):
        frame = frame.filter(
            ~pl.col(source_company_type_col)
            .cast(pl.Utf8)
            .str.strip_chars()
            .is_in(sorted(exclude_company_type_values))
        )

    # GLEIF records superseded by a merger/succession still carry their own
    # current registration data alongside an obsolete one; dropping any row with
    # a populated successor LEI removes the obsolete side while leaving the
    # successor's own row (which always exists independently in the same
    # extract) to represent the entity. General GLEIF behavior, not DE-specific.
    if "Entity_SuccessorEntity_SuccessorLEI" in frame.columns:
        frame = frame.filter(
            pl.col("Entity_SuccessorEntity_SuccessorLEI")
            .cast(pl.Utf8)
            .str.strip_chars()
            .fill_null("")
            == ""
        )

    source_columns = list(frame.columns)
    columns = set(source_columns)
    field_candidates = system_field_candidates
    system_config = get_system_canonical_config(country)

    name_expr, wikidata_name_bundle_expr = _resolve_name_expr(
        columns=columns,
        field_candidates=field_candidates,
        config_name_mode=system_config.name_mode,
    )

    try:
        country_type_mapping = company_type_mapping_resolver(country)
    except KeyError:
        country_type_mapping = {}

    raw_company_type_expr = coalesce_text(
        columns,
        _field_candidates_or_same_name(columns, field_candidates, "company_type"),
    )
    if country_type_mapping:
        company_type_expr = raw_company_type_expr.map_elements(
            lambda value: country_type_mapping.get(value, value) if value else None,
            return_dtype=pl.Utf8,
        )
    else:
        company_type_expr = raw_company_type_expr

    current_status_expr = coalesce_text(
        columns,
        _field_candidates_or_same_name(columns, field_candidates, "current_status"),
    )
    dissolution_date_expr = coalesce_date_iso(
        columns,
        _field_candidates_or_same_name(columns, field_candidates, "dissolution_date"),
    )

    if country == "wikidata":
        company_type_expr = (
            pl.when(company_type_expr == "item")
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(company_type_expr)
        )

        current_status_expr = (
            pl.when(current_status_expr == "item")
            .then(pl.lit(None, dtype=pl.Utf8))
            .otherwise(current_status_expr)
        )
        current_status_expr = (
            pl.when(dissolution_date_expr.is_not_null())
            .then(pl.lit("Dissolved"))
            .otherwise(pl.coalesce([current_status_expr, pl.lit("Active")]))
        )

    inactive_source_expr = coalesce_text(
        columns, _field_candidates_or_same_name(columns, field_candidates, "inactive")
    )
    inactive_expr = (
        pl.when(inactive_source_expr.is_not_null())
        .then(
            inactive_source_expr.str.to_lowercase().is_in(
                ["true", "t", "1", "yes", "y"]
            )
        )
        .otherwise(derive_inactive(current_status_expr))
    )
    if country == "wikidata":
        inactive_expr = (
            pl.when(dissolution_date_expr.is_not_null())
            .then(pl.lit(True))
            .otherwise(inactive_expr)
        )

    registered_address_expr = _resolve_registered_address_expr(
        columns=columns,
        field_candidates=field_candidates,
        mode=system_config.registered_address_mode,
    )

    jurisdiction_expr = _resolve_jurisdiction_expr(
        country=country,
        columns=columns,
        mode=system_config.jurisdiction_mode,
        candidates=system_config.jurisdiction_candidates,
    )
    registry_url_expr = _resolve_registry_url_expr(
        columns=columns,
        field_candidates=field_candidates,
        mode=system_config.registry_url_mode,
    )

    generated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    previous_names_expr = coalesce_name_list(
        columns,
        ["previous_names", "previous_names_list", "former_names", "old_names"],
    )
    if system_config.alternative_names_mode == "wikidata":
        assert wikidata_name_bundle_expr is not None, (  # nosec B101 - narrows Optional for mypy
            "alternative_names_mode='wikidata' requires name_mode='wikidata'"
        )
        alternative_names_expr = pl.coalesce(
            [
                pl.when(wikidata_name_bundle_expr.is_not_null())
                .then(wikidata_name_bundle_expr.struct.field("alternative_names"))
                .otherwise(pl.lit(None, dtype=pl.List(pl.Utf8))),
                coalesce_name_list(
                    columns, ["alternative_names", "aliases_en", "aliases"]
                ),
            ]
        )
    else:
        alternative_names_expr = coalesce_name_list(
            columns,
            [
                "alternative_names",
                "aliases_en",
                "aliases",
                "short_name",
                "official_name",
            ],
        )

    derived_columns = [
        jurisdiction_expr.alias("jurisdiction_code"),
        _resolve_company_number_expr(
            columns=columns,
            field_candidates=field_candidates,
            mode=system_config.company_number_mode,
            jurisdiction_expr=jurisdiction_expr,
            canonical_derived_fields=canonical_derived_fields,
        ).alias("company_number"),
        name_expr.alias("name"),
        alternative_names_expr.alias("alternative_names"),
        _resolve_lei_expr(columns=columns, country=country).alias("lei"),
        pl.lit(None, dtype=pl.Utf8).alias("vat"),
        pl.lit(None, dtype=pl.Utf8).alias("tax_id"),
        previous_names_expr.alias("previous_names"),
        company_type_expr.alias("company_type"),
        current_status_expr.alias("current_status"),
        coalesce_date_iso(
            columns,
            _field_candidates_or_same_name(
                columns, field_candidates, "incorporation_date"
            ),
        ).alias("incorporation_date"),
        dissolution_date_expr.alias("dissolution_date"),
        inactive_expr.alias("inactive"),
        coalesce_text(
            columns, _field_candidates_or_same_name(columns, field_candidates, "branch")
        ).alias("branch"),
        coalesce_text(
            columns,
            _field_candidates_or_same_name(columns, field_candidates, "branch_status"),
        ).alias("branch_status"),
        registered_address_expr.alias("registered_address_in_full"),
        registry_url_expr.alias("registry_url"),
        pl.lit(None, dtype=pl.Utf8).alias("opencorporates_url"),
        coalesce_text(
            columns,
            _field_candidates_or_same_name(columns, field_candidates, "match_uri")
            or ["match_uri"],
        ).alias("match_uri"),
        coalesce_text(columns, ["system_uri"]).alias("system_uri"),
        pl.lit(generated_at).alias("metadata_generated_utc"),
    ]

    canonical_names = [
        *project_identity_columns,
        *opencorporates_columns,
        *metadata_columns,
    ]

    return frame.with_columns(derived_columns).select(
        [pl.col(column) for column in canonical_names]
    )
