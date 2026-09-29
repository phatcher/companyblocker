#!/usr/bin/env python3
"""Generate ISO 20275 expansion artefacts using split-and-trim abbreviation rules.

Rule added:
- Split multi-value abbreviation fields on ';'
- Trim each token
- Drop empty tokens
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import TypedDict

import _bootstrap  # noqa: F401
from cli_common import (
    OUTPUT_CLEAR,
    PlannedOutput,
    add_declared_arguments,
    add_dry_run_arg,
    add_run_date_arg,
    add_workspace_roots_args,
    declared_settings,
    report_output_plan,
    report_resolved_settings,
    resolve_declared_settings,
    resolve_workspace_roots_from_args,
    resolved_setting_values,
    run_reporting_argument_errors,
)
from company_cleanse._cli_helper import SETTINGS as COMPANY_CLEANSE_SETTINGS
from company_cleanse.normalize import (
    normalize_company_type_value,
    normalize_tokens_with_operations,
)

from analysis.report_layout import metrics_dir, reports_dir
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots

REPORT_NAME = "iso20275_additions"


class IsoAbbreviationAnalysis(TypedDict):
    normalized_forms: list[str]
    family_targets: list[str]
    family_targets_normalized: list[str]
    proposed: str | None
    proposed_normalized: str | None
    alternate_forms: list[str]
    is_ambiguous: bool


class AuditSummary(TypedDict):
    existing_conflict_count: int
    proposed_conflict_count: int
    proposed_global_conflict_count: int
    high_diversity_source_count: int
    existing_conflicts: list[tuple[str, str, list[str]]]
    proposed_conflicts: list[tuple[str, str, str, str]]
    proposed_global_conflicts: list[tuple[str, str, str, str]]
    high_diversity_sources: list[tuple[str, int]]


RULES_PATH = files("company_cleanse.resources").joinpath("company_type_rules.json")
ISO_PATH = files("company_cleanse.resources").joinpath(
    "entity_legal_forms_iso20275.json"
)

OUT_JSON_ALL_NAME = "iso20275_additions_by_country.json"
OUT_JSON_ABBR_NAME = "iso20275_additions_by_country_abbr_only.json"
OUT_MD_NAME = "iso20275_decision_sheet_by_country.md"
OUT_AUDIT_MD_NAME = "iso20275_mapping_audit.md"

_DECLARATIONS = declared_settings(COMPANY_CLEANSE_SETTINGS)
_SURFACE = {"iso20275_countries": {"flag": "--countries"}}

# Proposal display policy is explicit by design:
# - Source labels preserve punctuation from ISO (e.g., "Co-operative").
# - Canonical proposals strip punctuation (e.g., "Co-operative" -> "Co operative").
SOURCE_DISPLAY_POLICY = "preserve-punctuation"
CANONICAL_DISPLAY_POLICY = "strip-punctuation"

CANONICAL_DISPLAY_OPERATIONS: tuple[str, ...] = (
    "punctuation",
    "singlespace",
    "singlechar",
)


def _collapse_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalized_surface_key(value: str) -> str:
    """Normalize key using the same legal-form normalizer as cleanse logic."""
    return normalize_company_type_value(value, transliterate=True)


def _strip_leading_article(value: str) -> str:
    """Drop leading English articles from normalized forms.

    This keeps ISO option variants like 'a limited partnership' and
    'an incorporated limited partnership' aligned with the base form.
    """
    if value.startswith("a "):
        return value[2:].strip()
    if value.startswith("an "):
        return value[3:].strip()
    return value


def _normalized_compact_key(value: str) -> str:
    """Space-agnostic key for already-normalized data variants (e.g. 'pty ltd' vs 'ptyltd')."""
    return _strip_leading_article(_normalized_surface_key(value)).replace(" ", "")


def _source_display_token(token: str) -> str:
    # Preserve source punctuation/casing; only trim/collapse spacing.
    return _collapse_spaces(token)


def _canonical_display_without_punctuation(value: str) -> str:
    """Return canonical display text with punctuation removed, preserving case."""
    return (
        normalize_tokens_with_operations(
            value,
            operations=CANONICAL_DISPLAY_OPERATIONS,
            and_tokens=(),
        )
        or ""
    )


def _strip_parenthetical_qualifiers(value: str) -> str:
    """Remove bracketed qualifiers from source display names (e.g. region labels)."""
    return _collapse_spaces(re.sub(r"\s*\([^)]*\)", "", value))


def _split_source_display_tokens(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [
        token
        for part in raw.split(";")
        if (token := _source_display_token(part.strip()))
    ]


def split_and_trim_multi(raw: str | None) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for token in _split_source_display_tokens(raw):
        key = _normalized_compact_key(token)
        if key in seen:
            continue
        seen.add(key)
        values.append(token)
    return values


def split_trim_raw_preserve_duplicates(raw: str | None) -> list[str]:
    return _split_source_display_tokens(raw)


def split_source_variants(raw: str | None) -> list[tuple[str, str | None]]:
    """Split source legal-form labels on '/' and ',' and trim each variant.

    ISO sometimes encodes synonymous labels in one field using separators.

    Returns tuples of:
    - display source (parenthetical qualifiers removed)
    - normalized parenthetical hint, when present (for target disambiguation)
    """
    if not raw:
        return []
    values: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for part in re.split(r"[/,]", raw):
        token = _source_display_token(part.strip())
        if not token:
            continue
        hint_match = re.search(r"\(([^)]+)\)", token)
        hint_norm = (
            _normalized_surface_key(hint_match.group(1).strip()) if hint_match else None
        )
        display = _strip_parenthetical_qualifiers(token)
        if not display:
            continue
        key = _normalized_surface_key(display)
        if key in seen:
            continue
        seen.add(key)
        values.append((display, hint_norm))
    return values


def _normalize_iso_token(token: str) -> tuple[str, str]:
    """Return (surface_key, compact_key) using the cleanse normalizer."""
    surface = _strip_leading_article(
        normalize_company_type_value(token, transliterate=True)
    )
    compact = surface.replace(" ", "")
    return surface, compact


def _family_key_from_compact(compact: str) -> str:
    """Group variants that only differ by trailing ltd/limited markers."""
    key = compact
    if key.endswith("limited"):
        key = key[: -len("limited")] + "ltd"
    key = key.removesuffix("ltd")
    return key


def _pick_preferred_compact(compacts: list[str]) -> str:
    """Pick one abbreviation candidate for a family.

    Preference:
    - key without trailing ltd/limited
    - then shortest key
    - then lexical order for determinism
    """
    ordered = sorted(compacts)
    no_ltd = [c for c in ordered if not (c.endswith(("ltd", "limited")))]
    pool = no_ltd if no_ltd else ordered
    pool.sort(key=lambda c: (len(c), c))
    return pool[0]


def analyse_iso_abbreviations(raw_tokens: list[str]) -> IsoAbbreviationAnalysis:
    """Analyse one ISO abbreviation field (already split on ';').

    Returns a case-insensitive, normalized representation that is suitable for
    deciding whether an ISO entry maps to one abbreviation target or multiple.
    """
    compact_to_surface: dict[str, str] = {}
    compact_to_display: dict[str, str] = {}
    compact_order: list[str] = []

    for token in raw_tokens:
        display = _source_display_token(token)
        surface, compact = _normalize_iso_token(token)
        if not compact:
            continue
        if compact not in compact_to_surface:
            compact_to_surface[compact] = surface
            compact_to_display[compact] = display
            compact_order.append(compact)

    if not compact_order:
        return {
            "normalized_forms": [],
            "family_targets": [],
            "family_targets_normalized": [],
            "proposed": None,
            "proposed_normalized": None,
            "alternate_forms": [],
            "is_ambiguous": False,
        }

    families: dict[str, list[str]] = {}
    for compact in compact_order:
        family = _family_key_from_compact(compact)
        families.setdefault(family, []).append(compact)

    family_target_compacts = [
        _pick_preferred_compact(compacts) for _, compacts in sorted(families.items())
    ]
    is_ambiguous = len(family_target_compacts) > 1

    normalized_forms = [compact_to_surface[c] for c in compact_order]
    family_targets = [compact_to_display[c] for c in family_target_compacts]
    family_targets_normalized = [compact_to_surface[c] for c in family_target_compacts]

    if is_ambiguous:
        proposed = None
        proposed_normalized = None
        alternate_forms: list[str] = []
    else:
        proposed = family_targets[0]
        proposed_normalized = compact_to_surface[family_target_compacts[0]]
        alternate_forms = [
            form for form in normalized_forms if form != proposed_normalized
        ]

    return {
        "normalized_forms": normalized_forms,
        "family_targets": family_targets,
        "family_targets_normalized": family_targets_normalized,
        "proposed": proposed,
        "proposed_normalized": proposed_normalized,
        "alternate_forms": alternate_forms,
        "is_ambiguous": is_ambiguous,
    }


def collect_existing_sources(rules: list[dict]) -> dict[str, set[str]]:
    by_country: dict[str, set[str]] = {}
    for row in rules:
        country = (row.get("country") or "").strip().lower()
        source = _normalized_surface_key((row.get("source") or "").strip())
        if not country or not source:
            continue
        by_country.setdefault(country, set()).add(source)
    return by_country


def _iter_normalized_rule_rows(
    rules: list[dict],
    include_countries: set[str] | None = None,
):
    for row in rules:
        country = (row.get("country") or "").strip().lower()
        if include_countries is not None and country not in include_countries:
            continue
        source_norm = _normalized_surface_key((row.get("source") or "").strip())
        canonical_norm = _normalized_surface_key((row.get("canonical") or "").strip())
        if country and source_norm and canonical_norm:
            yield country, source_norm, canonical_norm


def collect_unique_global_canonical_by_source(rules: list[dict]) -> dict[str, str]:
    """Return unique global canonical display values keyed by normalized source.

    If a source maps to multiple canonicals globally, it is excluded so the
    proposal flow can keep it for explicit review.
    """
    source_to_norm_cans: dict[str, set[str]] = defaultdict(set)
    source_norm_can_to_display: dict[tuple[str, str], str] = {}

    for row in rules:
        source_norm = _normalized_surface_key((row.get("source") or "").strip())
        canonical_display = _canonical_display_without_punctuation(
            (row.get("canonical") or "").strip()
        )
        canonical_norm = _normalized_surface_key(canonical_display)
        if not source_norm or not canonical_norm:
            continue
        source_to_norm_cans[source_norm].add(canonical_norm)
        source_norm_can_to_display.setdefault(
            (source_norm, canonical_norm), canonical_display
        )

    unique: dict[str, str] = {}
    for source_norm, can_norms in source_to_norm_cans.items():
        if len(can_norms) != 1:
            continue
        can_norm = next(iter(can_norms))
        unique[source_norm] = source_norm_can_to_display[(source_norm, can_norm)]
    return unique


def collect_global_canonical_display_by_normalized(rules: list[dict]) -> dict[str, str]:
    """Return a canonical display lookup keyed by normalized canonical value.

    This provides a stable, country-agnostic display style for proposal output.
    If multiple display variants exist for the same normalized canonical, the
    first encountered value is retained for deterministic behavior.
    """
    by_normalized: dict[str, str] = {}
    for row in rules:
        canonical_display = _canonical_display_without_punctuation(
            (row.get("canonical") or "").strip()
        )
        canonical_norm = _normalized_surface_key(canonical_display)
        if not canonical_norm or not canonical_display:
            continue
        by_normalized.setdefault(canonical_norm, canonical_display)
    return by_normalized


def _is_active_iso_entry_for_country(entry: dict, country: str) -> bool:
    return (entry.get("status") or "").strip().upper() == "ACTV" and (
        entry.get("country_code") or ""
    ).strip().lower() == country


def _pending_source_labels(
    source_variants: list[tuple[str, str | None]],
    existing_sources: set[str],
) -> list[str]:
    return [
        source
        for source, _ in source_variants
        if _normalized_surface_key(source) not in existing_sources
    ]


def _select_variant_tokens(
    tokens_all: list[str],
    source_variant_count: int,
    variant_index: int,
) -> list[str]:
    if source_variant_count > 1 and len(tokens_all) == source_variant_count:
        return [tokens_all[variant_index]]
    return tokens_all


def _build_ambiguous_multi_label_decision_row(
    *,
    pending_sources: list[str],
    country: str,
    local_analysis_all: IsoAbbreviationAnalysis,
    unique_targets_all: list[str],
    translit_raw_tokens_all: list[str],
) -> dict[str, object] | None:
    normalized_forms = local_analysis_all["normalized_forms"]
    if not normalized_forms:
        return None

    decision_row: dict[str, object] = {
        "source": "/".join(pending_sources),
        "source_variants": pending_sources,
        "canonical": None,
        "country": country,
        "reason": "multiple labels with multiple abbreviation targets",
        "abbreviations_local_normalized": normalized_forms,
        "canonical_choice_required": True,
        "canonical_target_candidates": unique_targets_all,
    }
    if unique_targets_all:
        decision_row["abbreviation_targets"] = unique_targets_all

    translit_analysis_all = analyse_iso_abbreviations(translit_raw_tokens_all)
    translit_normalized = translit_analysis_all["normalized_forms"]
    if translit_normalized:
        decision_row["abbreviations_transliterated_normalized"] = translit_normalized
    return decision_row


def _build_variant_row(
    *,
    country: str,
    source: str,
    local_analysis: IsoAbbreviationAnalysis,
    translit_analysis: IsoAbbreviationAnalysis,
    global_unique_canonical: dict[str, str],
    global_canonical_display: dict[str, str],
) -> tuple[str, dict[str, object]] | None:
    proposed_local = local_analysis["proposed"]
    proposed_local_normalized = local_analysis.get("proposed_normalized")
    local_is_ambiguous = bool(local_analysis["is_ambiguous"])
    local_normalized_forms = local_analysis["normalized_forms"]
    local_family_targets = local_analysis["family_targets_normalized"]
    local_alternates = local_analysis["alternate_forms"]

    if not local_normalized_forms:
        return None

    if proposed_local and not local_is_ambiguous:
        source_norm = _normalized_surface_key(source)
        canonical_value = global_unique_canonical.get(source_norm)
        if not canonical_value:
            canonical_value = global_canonical_display.get(
                proposed_local_normalized or "",
                _canonical_display_without_punctuation(proposed_local),
            )
        row: dict[str, object] = {
            "source": source,
            "canonical": canonical_value,
            "country": country,
        }
        if local_alternates:
            row["iso_alternate_forms_normalized"] = local_alternates
            row["notes"] = "Derived from ISO20275 (alternate form)"
        return ("simple", row)

    decision_row: dict[str, object] = {
        "source": source,
        "canonical": None,
        "country": country,
        "reason": "multiple local abbreviations",
        "abbreviations_local_normalized": local_normalized_forms,
    }
    if local_family_targets:
        decision_row["abbreviation_targets"] = local_family_targets
    translit_normalized = translit_analysis["normalized_forms"]
    if translit_normalized:
        decision_row["abbreviations_transliterated_normalized"] = translit_normalized
    return ("decision_needed", decision_row)


def _collect_country_additions(
    *,
    country: str,
    existing_sources: set[str],
    iso_data: dict[str, dict],
    global_unique_canonical: dict[str, str],
    global_canonical_display: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    simple: list[dict] = []
    decision_needed: list[dict] = []

    for entry in iso_data.values():
        if not _is_active_iso_entry_for_country(entry, country):
            continue

        source_raw = (entry.get("entity_legal_form_name") or "").strip()
        source_variants = split_source_variants(source_raw)
        if not source_variants:
            continue

        local_raw_tokens_all = split_trim_raw_preserve_duplicates(
            entry.get("abbreviations_local")
        )
        translit_raw_tokens_all = split_trim_raw_preserve_duplicates(
            entry.get("abbreviations_transliterated")
        )

        local_analysis_all = analyse_iso_abbreviations(local_raw_tokens_all)
        local_targets_all = local_analysis_all["family_targets_normalized"]
        unique_targets_all = list(dict.fromkeys(local_targets_all))
        pending_sources = _pending_source_labels(source_variants, existing_sources)
        if not pending_sources:
            continue

        if len(source_variants) > 1 and bool(local_analysis_all.get("is_ambiguous")):
            decision_row = _build_ambiguous_multi_label_decision_row(
                pending_sources=pending_sources,
                country=country,
                local_analysis_all=local_analysis_all,
                unique_targets_all=unique_targets_all,
                translit_raw_tokens_all=translit_raw_tokens_all,
            )
            if decision_row is not None:
                decision_needed.append(decision_row)
            continue

        for idx, (source, _) in enumerate(source_variants):
            if _normalized_surface_key(source) in existing_sources:
                continue

            local_raw_tokens = _select_variant_tokens(
                local_raw_tokens_all, len(source_variants), idx
            )
            translit_raw_tokens = _select_variant_tokens(
                translit_raw_tokens_all, len(source_variants), idx
            )
            local_analysis = analyse_iso_abbreviations(local_raw_tokens)
            translit_analysis = analyse_iso_abbreviations(translit_raw_tokens)
            classified_row = _build_variant_row(
                country=country,
                source=source,
                local_analysis=local_analysis,
                translit_analysis=translit_analysis,
                global_unique_canonical=global_unique_canonical,
                global_canonical_display=global_canonical_display,
            )
            if classified_row is None:
                continue
            kind, row = classified_row
            if kind == "simple":
                simple.append(row)
            else:
                decision_needed.append(row)

    return simple, decision_needed


def build_country_data(
    rules: list[dict],
    iso_data: dict[str, dict],
    include_countries: set[str] | None = None,
) -> tuple[dict, dict]:
    existing = collect_existing_sources(rules)
    global_unique_canonical = collect_unique_global_canonical_by_source(rules)
    global_canonical_display = collect_global_canonical_display_by_normalized(rules)
    # Master list is the current mapping table countries, not ISO coverage.
    all_countries = sorted(existing.keys())
    if include_countries is not None:
        all_countries = [c for c in all_countries if c in include_countries]

    additions_by_country: dict[str, dict[str, list[dict]]] = {}
    additions_abbr_only: dict[str, dict[str, list[dict]]] = {}

    for country in all_countries:
        simple, decision_needed = _collect_country_additions(
            country=country,
            existing_sources=existing.get(country, set()),
            iso_data=iso_data,
            global_unique_canonical=global_unique_canonical,
            global_canonical_display=global_canonical_display,
        )

        simple.sort(key=lambda r: (r["source"].casefold(), r["canonical"].casefold()))
        decision_needed.sort(key=lambda r: r["source"].casefold())

        additions_by_country[country] = {
            "simple": simple,
            "decision_needed": decision_needed,
        }
        additions_abbr_only[country] = {
            "simple": simple,
            "decision_needed": [
                d for d in decision_needed if d.get("abbreviations_local_normalized")
            ],
        }

    return additions_by_country, additions_abbr_only


def _build_existing_by_country_source(
    rules: list[dict],
    include_countries: set[str] | None,
) -> dict[str, dict[str, set[str]]]:
    existing_by_country_source: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for country, source_norm, canonical_norm in _iter_normalized_rule_rows(
        rules, include_countries
    ):
        existing_by_country_source[country][source_norm].add(canonical_norm)
    return existing_by_country_source


def _append_if_unseen(
    *,
    seen: set,
    unique_key,
    rows: list,
    row_value,
) -> bool:
    if unique_key in seen:
        return False
    seen.add(unique_key)
    rows.append(row_value)
    return True


def _collect_existing_conflicts(
    existing_by_country_source: dict[str, dict[str, set[str]]],
    add_blocking_error,
) -> list[tuple[str, str, list[str]]]:
    existing_conflicts: list[tuple[str, str, list[str]]] = []
    existing_conflict_seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for country, source_map in existing_by_country_source.items():
        for source, canonicals in source_map.items():
            if len(canonicals) <= 1:
                continue
            canonicals_sorted = tuple(sorted(canonicals))
            key = (country, source, canonicals_sorted)
            _append_if_unseen(
                seen=existing_conflict_seen,
                unique_key=key,
                rows=existing_conflicts,
                row_value=(country, source, list(canonicals_sorted)),
            )
            add_blocking_error(
                f"Existing conflict in rules: country={country}, source={source}, canonicals={list(canonicals_sorted)}"
            )
    return existing_conflicts


def _build_global_existing_by_source(rules: list[dict]) -> dict[str, set[str]]:
    global_existing_by_source: dict[str, set[str]] = defaultdict(set)
    for _, source_norm, canonical_norm in _iter_normalized_rule_rows(rules):
        global_existing_by_source[source_norm].add(canonical_norm)
    return global_existing_by_source


def _collect_proposed_conflicts(
    additions_by_country: dict[str, dict[str, list[dict]]],
    existing_by_country_source: dict[str, dict[str, set[str]]],
    global_existing_by_source: dict[str, set[str]],
    add_blocking_error,
) -> tuple[list[tuple[str, str, str, str]], list[tuple[str, str, str, str]]]:
    def _record_local_proposed_conflict(
        conflict_key: tuple[str, str, str, str], message: str
    ) -> None:
        _append_if_unseen(
            seen=proposed_conflict_seen,
            unique_key=conflict_key,
            rows=proposed_conflicts,
            row_value=conflict_key,
        )
        add_blocking_error(message)

    proposed_conflicts: list[tuple[str, str, str, str]] = []
    proposed_conflict_seen: set[tuple[str, str, str, str]] = set()
    proposed_seen: dict[tuple[str, str], str] = {}
    proposed_global_conflicts: list[tuple[str, str, str, str]] = []
    proposed_global_conflict_seen: set[tuple[str, str, str, str]] = set()

    for country, payload in additions_by_country.items():
        for row in payload.get("simple", []):
            source_norm = _normalized_surface_key((row.get("source") or "").strip())
            canonical_norm = _normalized_surface_key(
                (row.get("canonical") or "").strip()
            )
            if not source_norm or not canonical_norm:
                continue

            existing_canonicals = existing_by_country_source.get(country, {}).get(
                source_norm, set()
            )
            if existing_canonicals and canonical_norm not in existing_canonicals:
                existing_str = ",".join(sorted(existing_canonicals))
                conflict_key = (country, source_norm, existing_str, canonical_norm)
                _record_local_proposed_conflict(
                    conflict_key,
                    "Proposed mapping conflicts with existing rules: "
                    f"country={country}, source={source_norm}, existing={sorted(existing_canonicals)}, proposed={canonical_norm}",
                )

            proposed_key = (country, source_norm)
            previous = proposed_seen.get(proposed_key)
            if previous and previous != canonical_norm:
                conflict_key = (country, source_norm, previous, canonical_norm)
                _record_local_proposed_conflict(
                    conflict_key,
                    "Proposed mapping conflicts within proposal set: "
                    f"country={country}, source={source_norm}, first={previous}, second={canonical_norm}",
                )
            else:
                proposed_seen[proposed_key] = canonical_norm

            global_canonicals = global_existing_by_source.get(source_norm, set())
            if global_canonicals and canonical_norm not in global_canonicals:
                global_existing_str = ",".join(sorted(global_canonicals))
                global_conflict_key = (
                    country,
                    source_norm,
                    global_existing_str,
                    canonical_norm,
                )
                _append_if_unseen(
                    seen=proposed_global_conflict_seen,
                    unique_key=global_conflict_key,
                    rows=proposed_global_conflicts,
                    row_value=global_conflict_key,
                )
                add_blocking_error(
                    "Proposed mapping conflicts with global existing rules: "
                    f"country={country}, source={source_norm}, global_existing={sorted(global_canonicals)}, proposed={canonical_norm}"
                )

    return proposed_conflicts, proposed_global_conflicts


def _collect_high_diversity_sources(
    rules: list[dict],
    include_countries: set[str] | None,
) -> list[tuple[str, int]]:
    global_source_targets: dict[str, set[str]] = defaultdict(set)
    for _, source_norm, canonical_norm in _iter_normalized_rule_rows(
        rules, include_countries
    ):
        global_source_targets[source_norm].add(canonical_norm)
    return sorted(
        (source, len(canonicals))
        for source, canonicals in global_source_targets.items()
        if len(canonicals) >= 4
    )


def _dedupe_proposed_conflicts(
    proposed_conflicts: list[tuple[str, str, str, str]],
) -> list[tuple[str, str, str, str]]:
    unique_proposed_conflicts: list[tuple[str, str, str, str]] = []
    unique_proposed_seen: set[tuple[str, str, str, str]] = set()
    for country, source, left, right in proposed_conflicts:
        norm_key = (
            _collapse_spaces(country),
            _collapse_spaces(source),
            _collapse_spaces(left),
            _collapse_spaces(right),
        )
        _append_if_unseen(
            seen=unique_proposed_seen,
            unique_key=norm_key,
            rows=unique_proposed_conflicts,
            row_value=(country, source, left, right),
        )
    return unique_proposed_conflicts


def audit_mapping_table(
    rules: list[dict],
    additions_by_country: dict[str, dict[str, list[dict]]],
    include_countries: set[str] | None = None,
) -> tuple[AuditSummary, list[str]]:
    """Audit full mapping table and proposals for ambiguity/conflicts.

    Returns a summary dict and a list of blocking errors.
    """
    blocking_errors: list[str] = []
    blocking_error_seen: set[str] = set()

    def _add_blocking_error(message: str) -> None:
        if message not in blocking_error_seen:
            blocking_error_seen.add(message)
            blocking_errors.append(message)

    existing_by_country_source = _build_existing_by_country_source(
        rules, include_countries
    )
    existing_conflicts = _collect_existing_conflicts(
        existing_by_country_source, _add_blocking_error
    )
    global_existing_by_source = _build_global_existing_by_source(rules)
    proposed_conflicts, proposed_global_conflicts = _collect_proposed_conflicts(
        additions_by_country,
        existing_by_country_source,
        global_existing_by_source,
        _add_blocking_error,
    )
    high_diversity = _collect_high_diversity_sources(rules, include_countries)
    unique_proposed_conflicts = _dedupe_proposed_conflicts(proposed_conflicts)

    summary: AuditSummary = {
        "existing_conflict_count": len(existing_conflicts),
        "proposed_conflict_count": len(unique_proposed_conflicts),
        "proposed_global_conflict_count": len(proposed_global_conflicts),
        "high_diversity_source_count": len(high_diversity),
        "existing_conflicts": existing_conflicts,
        "proposed_conflicts": unique_proposed_conflicts,
        "proposed_global_conflicts": proposed_global_conflicts,
        "high_diversity_sources": high_diversity[:50],
    }
    return summary, blocking_errors


def _append_markdown_summary_metrics(lines: list[str], summary: AuditSummary) -> None:
    lines.append(
        f"- Existing in-country conflicts: {summary['existing_conflict_count']}"
    )
    lines.append(f"- Proposed conflicts: {summary['proposed_conflict_count']}")
    lines.append(
        f"- Proposed global conflicts: {summary['proposed_global_conflict_count']}"
    )
    lines.append(
        f"- High-diversity global sources (informational): {summary['high_diversity_source_count']}"
    )


def _append_markdown_list_section(
    lines: list[str], title: str, rows: list, render_row
) -> None:
    if not rows:
        return
    lines.append(title)
    for row in rows:
        lines.append(render_row(row))
    lines.append("")


def build_audit_markdown(summary: AuditSummary) -> str:
    lines: list[str] = []
    lines.append("# ISO 20275 Mapping Audit")
    lines.append("")
    _append_markdown_summary_metrics(lines, summary)
    lines.append("")

    existing_conflicts = summary["existing_conflicts"]
    _append_markdown_list_section(
        lines,
        "## Existing In-Country Conflicts",
        existing_conflicts,
        lambda row: f"- {row[0]}: `{row[1]}` -> {', '.join(row[2])}",
    )

    proposed_conflicts = summary["proposed_conflicts"]
    _append_markdown_list_section(
        lines,
        "## Proposed Conflicts",
        proposed_conflicts,
        lambda row: (
            f"- {row[0]}: `{row[1]}` existing/first={row[2]} proposed/second={row[3]}"
        ),
    )

    proposed_global_conflicts = summary["proposed_global_conflicts"]
    _append_markdown_list_section(
        lines,
        "## Proposed Global Conflicts",
        proposed_global_conflicts,
        lambda row: (
            f"- {row[0]}: `{row[1]}` global-existing={row[2]} proposed={row[3]}"
        ),
    )

    high_diversity = summary["high_diversity_sources"]
    _append_markdown_list_section(
        lines,
        "## High-Diversity Global Sources (Info)",
        high_diversity,
        lambda row: f"- `{row[0]}` maps to {row[1]} canonicals globally",
    )

    return "\n".join(lines).rstrip() + "\n"


def render_row_with_abbr(row: dict) -> str:
    source = row["source"]
    targets = row.get("abbreviation_targets") or []
    local = row.get("abbreviations_local_normalized") or []
    translit = row.get("abbreviations_transliterated_normalized") or []
    candidates = row.get("canonical_target_candidates") or []
    if row.get("canonical_choice_required") and candidates:
        alternates = [c for c in candidates if c != candidates[0]]
        if alternates:
            return (
                f"- {source}  # choose canonical: {candidates[0]} | "
                f"candidate alternates: {'; '.join(alternates)} | "
                f"targets: {'; '.join(candidates)}"
            )
        return f"- {source}  # choose canonical: {candidates[0]}"
    if local and targets:
        return f"- {source}  # targets: {'; '.join(targets)} | alternates(norm): {'; '.join(local)}"
    if local:
        return f"- {source}  # alternates(norm): {'; '.join(local)}"
    if translit:
        return f"- {source}  # alternates(translit,norm): {'; '.join(translit)}"
    return f"- {source}  # abbr: -"


def build_markdown(additions_by_country: dict[str, dict[str, list[dict]]]) -> str:
    lines: list[str] = []
    lines.append("# ISO 20275 -> company_type_rules decision sheet")
    lines.append("")
    lines.append("Use this per-country decision:")
    lines.append("- Option A: add **Simple** only now.")
    lines.append("- Option B: add **Simple + Decision-needed** now.")
    lines.append("- Option C: skip country for now.")
    lines.append("")

    for country in sorted(additions_by_country):
        simple = additions_by_country[country]["simple"]
        decision = additions_by_country[country]["decision_needed"]

        lines.append(f"## {country}")
        lines.append(f"- Decision point: choose A/B/C for `{country}`")
        lines.append(f"- Simple proposed values: {len(simple)}")
        lines.append(f"- Decision-needed values: {len(decision)}")
        lines.append("")

        if simple:
            lines.append("### Simple proposed values")
            for row in simple:
                lines.append(
                    f'- {{ "source": "{row["source"]}", "canonical": "{row["canonical"]}", "country": "{row["country"]}" }}'
                )
            lines.append("")

        if decision:
            show_all_for_country = country == "au"
            if show_all_for_country:
                lines.append("### Decision-needed values (full list)")
                rows_to_show = decision
            else:
                lines.append("### Decision-needed values (top 12 shown)")
                rows_to_show = decision[:12]

            for row in rows_to_show:
                lines.append(render_row_with_abbr(row))

            remaining = len(decision) - len(rows_to_show)
            if remaining > 0:
                lines.append(
                    f"- ... {remaining} more decision-needed values for `{country}`"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ISO 20275 mapping proposals and audits."
    )
    add_workspace_roots_args(parser)
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    add_run_date_arg(
        parser,
        detail="Names the analysis run directory the reports are written under.",
        default_behavior="today",
    )
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the scoped countries and the reports this run would "
            "write, without reading the rules table or writing anything."
        ),
    )
    return parser


def _include_countries(countries: str | None) -> set[str] | None:
    if not countries:
        return None
    return {token.strip().lower() for token in countries.split(",") if token.strip()}


def output_paths(roots: WorkspaceRoots, run_date: str) -> dict[str, Path]:
    """Where one run's four reports go: the JSON under `metrics/`, the markdown under `reports/`."""
    run_dir = analysis_report_run_dir(roots, REPORT_NAME, run_date)
    return {
        OUT_JSON_ALL_NAME: metrics_dir(run_dir) / OUT_JSON_ALL_NAME,
        OUT_JSON_ABBR_NAME: metrics_dir(run_dir) / OUT_JSON_ABBR_NAME,
        OUT_MD_NAME: reports_dir(run_dir) / OUT_MD_NAME,
        OUT_AUDIT_MD_NAME: reports_dir(run_dir) / OUT_AUDIT_MD_NAME,
    }


def _planned_outputs(paths: dict[str, Path]) -> list[PlannedOutput]:
    return [
        PlannedOutput(path, OUTPUT_CLEAR, note="rewritten from scratch every run")
        for path in paths.values()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    roots = resolve_workspace_roots_from_args(args)
    run_date = args.run_date or datetime.now(UTC).date().isoformat()
    paths = output_paths(roots, run_date)
    resolved = resolve_declared_settings(args, _DECLARATIONS, _SURFACE)
    include_countries = _include_countries(
        str(resolved_setting_values(resolved)["iso20275_countries"] or "") or None
    )

    if args.dry_run:
        report_resolved_settings(
            "generate_iso20275_additions",
            resolved,
            countries=sorted(include_countries) if include_countries else "all",
        )
        report_output_plan("[dry-run]", _planned_outputs(paths))
        return 0

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    out_json_all = paths[OUT_JSON_ALL_NAME]
    out_json_abbr = paths[OUT_JSON_ABBR_NAME]
    out_md = paths[OUT_MD_NAME]
    out_audit_md = paths[OUT_AUDIT_MD_NAME]

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    iso_data = json.loads(ISO_PATH.read_text(encoding="utf-8"))

    additions_by_country, additions_abbr_only = build_country_data(
        rules, iso_data, include_countries
    )
    audit_summary, blocking_errors = audit_mapping_table(
        rules, additions_by_country, include_countries
    )

    out_json_all.write_text(
        json.dumps(additions_by_country, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    out_json_abbr.write_text(
        json.dumps(additions_abbr_only, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    out_md.write_text(build_markdown(additions_by_country), encoding="utf-8")
    out_audit_md.write_text(build_audit_markdown(audit_summary), encoding="utf-8")

    print(f"Wrote {out_json_all}")
    print(f"Wrote {out_json_abbr}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_audit_md}")

    if blocking_errors:
        print("Blocking mapping audit failures detected:")
        for err in blocking_errors:
            print(f"- {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
