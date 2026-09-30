"""Legal-suffix family: drop the trailing entity-form suffix, or swap it for another.

TAXONOMY.md family 7. There is at most *one* candidate site in a name, the trailing
suffix match or none, so this is a site finder returning zero or one site and needs
no special case anywhere: a step wanting the suffix touched asks for one attempt.

**Detection is deliberately jurisdiction-blind.** Whether a suffix is *present* is
recognized through `company_cleanse`'s whole-dataset regex, so a suffix is found
whatever country it came from. Only choosing a *replacement* is country-scoped.

**Replacement is curated, not arbitrary.** Always other spellings of the matched
suffix's own canonical — `Limited` for `Ltd`, the same legal form written another
way. A genuinely *different* canonical is offered only where a curated cluster says
that pairing occurs in reality: `gb`/`ie` `ltd`↔`plc` and `de` `gmbh`↔`ug` are real
conversions a company makes over its life, so two sources captured at different
times legitimately disagree on the same company. Offering every canonical within a
country instead was tried and produced `Ltd → CIO`, a Charity Commission structure
with no path from a company at all. The rule data has no regulatory-family field to
group canonicals by, which is why the clusters are curated rather than derived.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from functools import cache
from random import Random

from company_cleanse import (
    UnknownCompanyTypeCountryError,
    get_company_type_rules,
    get_company_type_rules_for_country,
)

from ..families import Family
from ..sited_operator import Site, SitedOperator, sited_registry
from .casing import match_case

_SUFFIX_REGEX, _ = get_company_type_rules()
_SUFFIX_PATTERN = re.compile(_SUFFIX_REGEX)

# Pairs of canonicals genuinely observed to be interchangeable for a country, whether
# through data confusion or through a real corporate conversion over a company's life.
# Not derived from the rule data: it carries no field to group canonicals by, and
# treating every canonical in a country as interchangeable produces nonsense.
_CONFUSABLE_CANONICAL_CLUSTERS: dict[str, tuple[frozenset[str], ...]] = {
    "gb": (frozenset({"ltd", "plc"}),),
    "ie": (frozenset({"ltd", "plc"}),),
    "de": (frozenset({"gmbh", "ug"}),),
}


@cache
def _country_mapping(country: str) -> tuple[tuple[str, str], ...]:
    """(variant, canonical) pairs known for `country`; empty for an unknown one.

    `get_company_type_rules_for_country` raises for a country with no packaged data.
    This is the one caller for which that is an expected outcome rather than an error,
    so it is the one place that turns it back into an empty result.
    """
    try:
        _regex, mapping = get_company_type_rules_for_country(country)
    except UnknownCompanyTypeCountryError:
        return ()
    return tuple(sorted(mapping.items()))


def _substitution_candidates(country: str, matched_text: str) -> tuple[str, ...]:
    """Replacement suffixes for `matched_text` within `country`.

    Empty when the match isn't in this country's own rule data — it was found by the
    permissive global regex, so there is nothing reliable to build a pool from.
    """
    country_mapping = dict(_country_mapping(country))
    # `company_cleanse`'s variant keys are lowercase; a register holding
    # "LIMITED" would otherwise miss and silently fall back to dropping.
    matched_text = matched_text.lower()
    matched_canonical = country_mapping.get(matched_text)
    if matched_canonical is None:
        return ()

    allowed = {matched_canonical}
    for cluster in _CONFUSABLE_CANONICAL_CLUSTERS.get(country, ()):
        if matched_canonical in cluster:
            allowed |= cluster

    return tuple(
        variant
        for variant, canonical in country_mapping.items()
        if canonical in allowed and variant != matched_text
    )


def _suffix_site(name: str, country: str | None) -> Sequence[Site]:
    """The trailing suffix match, or nothing. Never more than one."""
    match = _SUFFIX_PATTERN.search(name)
    if match is None:
        return ()
    start, end = match.span(1)
    return (Site(start, end),)


def _drop_suffix(name: str, site: Site, rng: Random, country: str | None) -> str:
    """Remove the suffix. Safe whatever jurisdiction it came from, so `country` is unused."""
    return name[: site.start].rstrip()


def _substitute_suffix(name: str, site: Site, rng: Random, country: str | None) -> str:
    """Swap the suffix for another realistic rendering within `country`.

    With no country, or none this suffix is known in, falls back to dropping it:
    that yields a name some registry could hold, where picking a suffix from the
    whole set would render a British company as a `GmbH`.
    """
    if country is None:
        return _drop_suffix(name, site, rng, country)

    matched = name[site.start : site.end]
    candidates = _substitution_candidates(country, matched)
    if not candidates:
        return _drop_suffix(name, site, rng, country)

    return (
        name[: site.start]
        + match_case(matched, rng.choice(candidates))
        + name[site.end :]
    )


for _operator in (
    SitedOperator(
        operator_id="legal_suffix.drop",
        family=Family.LEGAL_SUFFIX,
        sites=_suffix_site,
        change=_drop_suffix,
    ),
    SitedOperator(
        operator_id="legal_suffix.variant_substitution",
        family=Family.LEGAL_SUFFIX,
        sites=_suffix_site,
        change=_substitute_suffix,
    ),
):
    sited_registry.register(_operator)
