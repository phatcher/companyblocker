from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path

import polars as pl

from .canonical_system_config import _WIKIDATA_COUNTRY_QID_TO_ISO2

# QID -> country-QID closure, resolving sub-national/non-
# standard jurisdiction QIDs (a US state, a German Land, Hong Kong, Delaware)
# to a real country reachable by walking the Wikidata graph, rather than
# hand-enumerating more sovereign-state QIDs into
# _WIKIDATA_COUNTRY_QID_TO_ISO2. Built offline by
# scripts/build_wikidata_jurisdiction_closure.py against the Wikidata Query
# Service -- never fetched at runtime.
_WIKIDATA_QID_COUNTRY_CLOSURE_PATH = (
    Path(__file__).with_name("resources") / "wikidata_qid_country_closure.json"
)


@cache
def _load_wikidata_qid_country_closure() -> dict[str, str]:
    if not _WIKIDATA_QID_COUNTRY_CLOSURE_PATH.exists():
        return {}
    with _WIKIDATA_QID_COUNTRY_CLOSURE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_wikidata_qid_to_iso2(value: str) -> str | None:
    """Resolve one Wikidata country/jurisdiction QID to an ISO2 code.

    Checks the static sovereign-state table first, then the QID closure
    cache (a sub-national entity resolved to its country QID, which is then
    looked up in the same static table). Returns None -- never the raw QID
    -- for anything neither covers: an entity the closure fetch couldn't
    resolve, or one that resolved to a historical/defunct state or disputed
    territory not in the static table.
    """
    iso2 = _WIKIDATA_COUNTRY_QID_TO_ISO2.get(value)
    if iso2 is not None:
        return iso2
    resolved_country_qid = _load_wikidata_qid_country_closure().get(value)
    if resolved_country_qid is None:
        return None
    return _WIKIDATA_COUNTRY_QID_TO_ISO2.get(resolved_country_qid)


def clean_text(expr: pl.Expr) -> pl.Expr:
    text = expr.cast(pl.Utf8, strict=False).str.strip_chars()
    return pl.when(text == "").then(pl.lit(None, dtype=pl.Utf8)).otherwise(text)


def to_text_or_none(value: object) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    return stripped
        return None

    if isinstance(value, pl.Series):
        for item in value.to_list():
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    return stripped
        return None

    text = str(value).strip()
    return text or None


def _casefold_column_lookup(columns: set[str]) -> dict[str, str]:
    return {column.casefold(): column for column in columns}


def _resolve_column_name(columns: set[str], candidate: str) -> str | None:
    return _casefold_column_lookup(columns).get(candidate.casefold())


def _resolve_present_columns(columns: set[str], candidates: list[str]) -> list[str]:
    lookup = _casefold_column_lookup(columns)
    present: list[str] = []
    for candidate in candidates:
        resolved = lookup.get(candidate.casefold())
        if resolved is not None and resolved not in present:
            present.append(resolved)
    return present


def coalesce_text(columns: set[str], candidates: list[str]) -> pl.Expr:
    present = _resolve_present_columns(columns, candidates)
    if not present:
        return pl.lit(None, dtype=pl.Utf8)
    normalized = [
        pl.col(candidate).map_elements(to_text_or_none, return_dtype=pl.Utf8)
        for candidate in present
    ]
    return pl.coalesce(normalized)


def coalesce_date_iso(columns: set[str], candidates: list[str]) -> pl.Expr:
    present = _resolve_present_columns(columns, candidates)
    if not present:
        return pl.lit(None, dtype=pl.Utf8)

    date_exprs: list[pl.Expr] = []
    for candidate in present:
        source = clean_text(pl.col(candidate))
        date_exprs.extend(
            [
                source.str.strptime(pl.Date, format="%Y-%m-%d", strict=False),
                source.str.strptime(pl.Date, format="%Y-%m-%dT%H:%M:%SZ", strict=False),
                source.str.strptime(pl.Date, format="%d/%m/%Y", strict=False),
                source.str.strptime(pl.Date, format="%Y%m%d", strict=False),
            ]
        )

    return pl.coalesce(date_exprs).dt.strftime("%Y-%m-%d")


def compose_gb_registered_address(columns: set[str]) -> pl.Expr:
    return _compose_registered_address(
        columns,
        [
            "RegAddress.AddressLine1",
            "RegAddress.AddressLine2",
            "RegAddress.PostTown",
            "RegAddress.County",
            "RegAddress.PostCode",
        ],
    )


def _compose_registered_address(columns: set[str], parts: list[str]) -> pl.Expr:
    present = [
        resolved
        for part in parts
        if (resolved := _resolve_column_name(columns, part)) is not None
    ]
    if not present:
        return pl.lit(None, dtype=pl.Utf8)

    normalized_parts = [clean_text(pl.col(part)) for part in present]
    concatenated = pl.concat_str(normalized_parts, separator=", ", ignore_nulls=True)
    return clean_text(concatenated)


def derive_inactive(current_status: pl.Expr) -> pl.Expr:
    derived = current_status.str.to_lowercase().str.contains(
        r"dissolv|liquidat|struck off|removed|inactive|closed|cess",
        literal=False,
    )
    return pl.when(derived.is_null()).then(pl.lit(False)).otherwise(derived)


def compose_gleif_registered_address(columns: set[str]) -> pl.Expr:
    return _compose_registered_address(
        columns,
        [
            "Entity_LegalAddress_FirstAddressLine",
            "Entity_LegalAddress_AdditionalAddressLine",
            "Entity_LegalAddress_City",
            "Entity_LegalAddress_Region",
            "Entity_LegalAddress_PostalCode",
            "Entity_LegalAddress_Country",
        ],
    )


def resolve_wikidata_country_iso2_expr(columns: set[str]) -> pl.Expr:
    candidate_exprs: list[pl.Expr] = []
    country_column = _resolve_column_name(columns, "country")
    jurisdiction_column = _resolve_column_name(columns, "jurisdiction")
    if country_column is not None:
        candidate_exprs.append(
            pl.col(country_column).list.first().cast(pl.Utf8, strict=False)
        )
    if jurisdiction_column is not None:
        candidate_exprs.append(
            pl.col(jurisdiction_column).list.first().cast(pl.Utf8, strict=False)
        )

    if not candidate_exprs:
        return pl.lit(None, dtype=pl.Utf8)

    return pl.coalesce(candidate_exprs).map_elements(
        lambda value: _resolve_wikidata_qid_to_iso2(value) if value else None,
        return_dtype=pl.Utf8,
    )


def to_name_list_or_none(value: object) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None

    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    result.append(stripped)
                continue
            if isinstance(item, dict):
                candidate = item.get("company_name")
                if isinstance(candidate, str):
                    stripped = candidate.strip()
                    if stripped:
                        result.append(stripped)
        return result or None

    return None


def coalesce_name_list(columns: set[str], candidates: list[str]) -> pl.Expr:
    present = _resolve_present_columns(columns, candidates)
    if not present:
        return pl.lit(None, dtype=pl.List(pl.Utf8))

    normalized = [
        pl.col(candidate).map_elements(
            to_name_list_or_none, return_dtype=pl.List(pl.Utf8)
        )
        for candidate in present
    ]
    return pl.coalesce(normalized)


def normalize_name_list(value: object) -> list[str]:
    normalized = to_name_list_or_none(value)
    return normalized or []


def dedupe_names(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = value.strip()
        if not token:
            continue
        token_folded = token.casefold()
        if token_folded in seen:
            continue
        seen.add(token_folded)
        deduped.append(token)
    return deduped


TRANSLITERATION_NAME_TYPES: frozenset[str] = frozenset(
    {"transliteration_preferred", "transliteration_auto"}
)
"""name_type values that are a script rendering of the *same* name, not a
different one. Script-based primary-name override may only promote
one of these: a `previous`/`trading`/`alternative_language` row is a
genuinely different name, and promoting it just because it happens to pass
a Latin-script check would silently misrepresent the entity, not transliterate
it. Any caller-configured priority list is filtered to this set before use,
so a catalog mistake degrades to "no override for that type" rather than an
incorrect promotion or a crash.
"""


def has_latin_char(value: str) -> bool:
    """Return True if any character in `value` is a Latin-script codepoint.

    Source-neutral: used for GLEIF's non-Latin primary-name override
    as well as Wikidata's name-bundle selection below -- not specific
    to either source's data shape.
    """
    for ch in value:
        if "LATIN" in unicodedata.name(ch, ""):
            return True
    return False


def select_first_non_empty(candidates: Iterable[str | None]) -> str | None:
    """Return the first truthy candidate from a priority-ordered sequence.

    Source-neutral: the shared "priority list -> pick a winner" primitive
    behind both Wikidata's name-bundle selection and GLEIF's script-based
    primary-name override.
    """
    return next((candidate for candidate in candidates if candidate), None)


def select_wikidata_name_bundle(payload: Mapping[str, object]) -> dict[str, object]:
    label_en = normalize_name_list(payload.get("label_en"))
    official_name_en = normalize_name_list(payload.get("official_name_en"))
    short_name_en = normalize_name_list(payload.get("short_name_en"))
    official_name_all = normalize_name_list(payload.get("official_name"))
    short_name_all = normalize_name_list(payload.get("short_name"))
    aliases_en = normalize_name_list(payload.get("aliases_en"))
    official_name_latin = [name for name in official_name_all if has_latin_char(name)]

    primary_name = select_first_non_empty(
        (
            *official_name_en,
            *official_name_latin,
            *label_en,
            *short_name_en,
            *official_name_all,
            *short_name_all,
        )
    )

    alternative_candidates = [*official_name_all, *short_name_all, *aliases_en]
    alternatives = dedupe_names(alternative_candidates)
    if primary_name is not None:
        primary_folded = primary_name.casefold()
        alternatives = [
            name for name in alternatives if name.casefold() != primary_folded
        ]

    return {
        "name": primary_name,
        "alternative_names": alternatives or None,
    }


def resolve_wikidata_name_bundle_expr(columns: set[str]) -> pl.Expr:
    wikidata_name_columns = [
        "label_en",
        "official_name_en",
        "official_name",
        "short_name_en",
        "short_name",
        "aliases_en",
    ]
    present = [candidate for candidate in wikidata_name_columns if candidate in columns]

    return_dtype = pl.Struct(
        [
            pl.Field("name", pl.Utf8),
            pl.Field("alternative_names", pl.List(pl.Utf8)),
        ]
    )

    if not present:
        return pl.lit({"name": None, "alternative_names": None}, dtype=return_dtype)

    payload_expr = pl.struct(
        [pl.col(candidate).alias(candidate) for candidate in present]
    )
    return payload_expr.map_elements(
        select_wikidata_name_bundle, return_dtype=return_dtype
    )
