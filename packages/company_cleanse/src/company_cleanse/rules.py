"""Company-type rules: recognising a legal-form suffix, and mapping it to a canonical form.

`get_company_type_rules` recognises a suffix whichever country it came from, which is what
cleansing needs; `get_company_type_rules_for_country` restricts the rules to one country, for a
caller choosing a realistic replacement, and raises `UnknownCompanyTypeCountryError` for a
country with no rule data rather than returning a mapping that matches nothing.

Each entry in `resources/company_type_rules.json` has a `source` (the full legal form name), a
`canonical` ISO 20275 abbreviation, a `country` as ISO 3166-1 alpha-2, and optional `notes`. A
`canonical` value is stored pre-normalized, ASCII letters and digits separated by single spaces,
with `&` allowed only as a separator token (`GmbH & Co KG`), so matching never re-runs
punctuation or transliteration cleanup; `tests/test_rules.py` checks the format, conflicting
mappings and structure.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib.resources import files

from .normalize import _normalize_company_type_value

# Canonical values are already cleansed/transliterated/compressed so suffix matching
# does not need to re-apply punctuation or script-specific normalization.
CANONICAL_COMPANY_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9]+(?: (?:& )?[A-Za-z0-9]+)*$")

# Never matches anything -- returned for a country with no rule data, so callers get an
# empty-but-valid regex/mapping pair instead of a special "no data" sentinel to check for.
_UNMATCHABLE_REGEX = r"(?!x)x"


def _load_company_type_rules() -> list[tuple[str, str, str]]:
    """Load company type rules (source, canonical, country) from the packaged JSON data file."""
    rules_path = files("company_cleanse.resources").joinpath("company_type_rules.json")
    entries = json.loads(rules_path.read_text(encoding="utf-8"))
    for idx, entry in enumerate(entries, start=1):
        canonical = entry["canonical"]
        if not CANONICAL_COMPANY_TYPE_PATTERN.fullmatch(canonical):
            raise ValueError(
                "Invalid canonical company type format in company_type_rules.json "
                f"at entry #{idx}: '{canonical}'. Canonicals must match "
                f"{CANONICAL_COMPANY_TYPE_PATTERN.pattern}."
            )
    return [
        (entry["source"], entry["canonical"], entry["country"]) for entry in entries
    ]


COMPANY_TYPE_RULES: list[tuple[str, str, str]] = _load_company_type_rules()

_CANONICAL_FROM_RULES = {
    canonical for (_source, canonical, _country) in COMPANY_TYPE_RULES
}
CANONICAL_COMPANY_TYPES = _CANONICAL_FROM_RULES

COMPANY_TYPE_RULES = sorted(
    COMPANY_TYPE_RULES,
    key=lambda rule: len(rule[0]),
    reverse=True,
)


def _build_company_type_mapping(
    company_type_rules: list[tuple[str, str]],
    canonical_company_types: set[str],
) -> dict[str, str]:
    """Build normalized value->canonical mapping and fail on conflicting definitions."""
    mapping: dict[str, str] = {}
    provenance: dict[str, tuple[str, int]] = {}
    conflict_merges: list[tuple[str, str, str, int, int]] = []
    conflict_seen: set[tuple[str, str, str]] = set()
    canonical_values = {
        _normalize_company_type_value(canonical, transliterate=True)
        for canonical in canonical_company_types
        if canonical
    }

    def _register_mapping(
        value: str, canonical: str, rule_index: int, raw_value: str, raw_canonical: str
    ) -> None:
        _ = (raw_value, raw_canonical)
        if not value:
            return

        if value in mapping and mapping[value] != canonical:
            prev_canonical, prev_rule_index = provenance[value]
            conflict_key = (value, prev_canonical, canonical)
            if conflict_key not in conflict_seen:
                conflict_merges.append(
                    (value, prev_canonical, canonical, prev_rule_index, rule_index)
                )
                conflict_seen.add(conflict_key)
            return

        mapping[value] = canonical
        provenance[value] = (canonical, rule_index)

    def _register_variants(
        value: str, canonical: str, rule_index: int, raw_value: str, raw_canonical: str
    ) -> None:
        variants = {value, value.replace(" ", "")}
        for variant in variants:
            _register_mapping(variant, canonical, rule_index, raw_value, raw_canonical)

    for rule_index, (raw_value, raw_canonical) in enumerate(
        company_type_rules, start=1
    ):
        value = _normalize_company_type_value(raw_value, transliterate=False)
        transliterated_value = _normalize_company_type_value(
            raw_value, transliterate=True
        )
        canonical = _normalize_company_type_value(raw_canonical, transliterate=True)

        if not value:
            continue
        if canonical not in canonical_values:
            raise ValueError(
                f"Unknown canonical company type '{raw_canonical}' in rule #{rule_index} for value '{raw_value}'."
            )

        _register_variants(value, canonical, rule_index, raw_value, raw_canonical)
        _register_variants(
            transliterated_value, canonical, rule_index, raw_value, raw_canonical
        )

    if conflict_merges:
        details = "\n".join(
            (
                f"- '{value}': '{prev_canonical}' (rule #{prev_rule}) "
                f"vs '{new_canonical}' (rule #{new_rule})"
            )
            for value, prev_canonical, new_canonical, prev_rule, new_rule in conflict_merges
        )
        raise ValueError(
            "Conflicting company type mappings detected. "
            "Resolve the following conflicts before running:\n"
            f"{details}"
        )

    for canonical in canonical_values:
        mapping.setdefault(canonical, canonical)
        mapping.setdefault(canonical.replace(" ", ""), canonical)

    return mapping


def _build_company_type_regex(company_type_mapping: dict[str, str]) -> str:
    variants = list(company_type_mapping.keys())
    variants = sorted(set(variants), key=len, reverse=True)
    escaped_variants = [
        re.escape(variant).replace(r"\ ", r"\s+") for variant in variants
    ]
    return rf"(?i)(?:^|\s+)({'|'.join(escaped_variants)})$"


COMPANY_TYPE_MAPPING = _build_company_type_mapping(
    [(source, canonical) for source, canonical, _country in COMPANY_TYPE_RULES],
    CANONICAL_COMPANY_TYPES,
)

COMPANY_TYPE_TOKENS = set(COMPANY_TYPE_MAPPING.keys()) | {
    _normalize_company_type_value(value, transliterate=True)
    for value in COMPANY_TYPE_MAPPING.values()
}

COMPANY_TYPE_REGEX = _build_company_type_regex(COMPANY_TYPE_MAPPING)

_TRIE_MATCH_KEY = "__match__"


def _build_company_type_suffix_trie(
    company_type_mapping: dict[str, str],
) -> tuple[dict[str, object], int]:
    """Build a reversed-token trie for longest suffix matching."""
    trie: dict[str, object] = {}
    max_tokens = 0

    for variant in company_type_mapping:
        tokens = [token for token in variant.split() if token]
        if not tokens:
            continue

        max_tokens = max(max_tokens, len(tokens))
        node = trie
        for token in reversed(tokens):
            child = node.get(token)
            if not isinstance(child, dict):
                child = {}
                node[token] = child
            node = child
        node[_TRIE_MATCH_KEY] = variant

    return trie, max_tokens


def _match_company_type_from_suffix_surface(
    suffix_surface: str | None,
    suffix_trie: dict[str, object],
    max_tokens: int,
) -> str | None:
    """Return the longest company-type variant matching the end of a normalized suffix surface."""
    if not suffix_surface or max_tokens < 1:
        return None

    tokens = [token for token in suffix_surface.split() if token]
    if not tokens:
        return None

    node: dict[str, object] = suffix_trie
    best_match: str | None = None

    for depth, token in enumerate(reversed(tokens), start=1):
        child = node.get(token)
        if not isinstance(child, dict):
            break

        node = child
        maybe_match = node.get(_TRIE_MATCH_KEY)
        if isinstance(maybe_match, str):
            best_match = maybe_match

        if depth >= max_tokens:
            break

    return best_match


@cache
def _get_company_type_rules_for_country(country: str) -> tuple[str, dict[str, str]]:
    """Build a country-scoped regex + mapping, restricted to rules tagged with `country`.

    Internal cache layer for `get_company_type_rules_for_country`, which raises on the empty
    result this returns for a country with no rule data -- see that function's docstring for the
    public contract. Unlike the default (whole-dataset) rules, this excludes any variant whose
    only provenance is a *different* country's legal-form data -- so asking for "gb" never
    returns Australia's ISO 20275 classification label ("Public Company limited by guarantee")
    as if it were a real UK suffix.
    """
    country_rules = [
        (source, canonical)
        for source, canonical, rule_country in COMPANY_TYPE_RULES
        if rule_country == country
    ]
    if not country_rules:
        return _UNMATCHABLE_REGEX, {}

    country_canonicals = {canonical for _source, canonical in country_rules}
    mapping = _build_company_type_mapping(country_rules, country_canonicals)
    regex = _build_company_type_regex(mapping)
    return regex, mapping


class UnknownCompanyTypeCountryError(ValueError):
    """Raised by `get_company_type_rules_for_country` when `country` has no packaged rule data.

    Distinct from a plain `ValueError` so a caller that wants to treat "no data for this
    jurisdiction" as an expected, non-error outcome (rather than a caller mistake) can catch
    precisely this and nothing else.
    """


def get_company_type_rules() -> tuple[str, dict[str, str]]:
    """Return the packaged, full multi-jurisdiction company-type regex and canonical mapping.

    Covers all 41 countries `company_type_rules.json` currently tags -- broad enough to
    *recognize* a legal-form suffix regardless of which country it's from, which is what
    cleansing needs: narrowing this to one jurisdiction leaves most suffixes from every other
    jurisdiction unmatched. For a caller that instead needs to pick a *realistic replacement*
    suffix for a name already known to be from one jurisdiction, see
    `get_company_type_rules_for_country`, a separate entry point with its own contract -- this
    function does not take a `country` argument.
    """
    return COMPANY_TYPE_REGEX, COMPANY_TYPE_MAPPING.copy()


def get_company_type_rules_for_country(country: str) -> tuple[str, dict[str, str]]:
    """Return a jurisdiction-scoped company-type regex/mapping for `country`.

    Restricted to variants tagged with `country` in `company_type_rules.json`. For a caller
    that needs to pick a *realistic replacement* suffix for a name already known to
    be from `country` -- not for recognizing whether a suffix is present at all, which is what
    `get_company_type_rules()`'s full multi-jurisdiction default is for (see its docstring).
    Unlike that default, this excludes any variant whose only provenance is a *different*
    country's legal-form data -- so asking for "gb" never returns Australia's ISO 20275
    classification label ("Public Company limited by guarantee") as if it were a real UK suffix.

    Pass a real, lowercase country code matching `company_type_rules.json`'s own `country` field
    (e.g. `"gb"`, `"de"`, `"ie"`).

    Raises `UnknownCompanyTypeCountryError` if no packaged rule is tagged with `country`, rather
    than returning an empty mapping a caller could silently mistake for "this name has no
    suffix". A caller that wants to treat missing jurisdiction data as an expected, non-error
    outcome (for example, falling back to unscoped behavior) must catch this explicitly.
    """
    regex, mapping = _get_company_type_rules_for_country(country)
    if not mapping:
        raise UnknownCompanyTypeCountryError(
            f"No company-type rules are packaged for country {country!r}."
        )
    return regex, mapping.copy()
