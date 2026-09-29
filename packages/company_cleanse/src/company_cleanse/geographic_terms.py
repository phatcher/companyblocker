"""Geographic terms that act as subsidiary markers in a company name, tiered by kind.

The `geographic_terms` operation is off unless a profile names it; once named, `country` is its
default tier, and `+region` and `+city` add to it. It can be selected in two places, which run at
different points. In the normalization profile it runs at the head of the chain, before
punctuation, so only its bracketed arm reaches a name still carrying its legal form
(`Siemens (UK) Ltd` to `siemens ltd`). In the short-name profile it runs after the legal form is
trimmed, so its trailing arm can reach the term (`Oracle Ireland Ltd` to `oracle`).

The `country` and `region` tiers are derived from `entity_legal_forms_iso20275.json`;
`resources/geographic_terms.json` adds only the colloquial terms that resource cannot supply,
with a reason per entry. The `city` tier has no entries yet.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from importlib.resources import files
from typing import Any

_ELF_RESOURCE_FILENAME = "entity_legal_forms_iso20275.json"
_CURATED_RESOURCE_FILENAME = "geographic_terms.json"

GEOGRAPHIC_TERM_KIND_COUNTRY = "country"
GEOGRAPHIC_TERM_KIND_REGION = "region"
GEOGRAPHIC_TERM_KIND_CITY = "city"

# Tier order, coarsest first. `country` is the default tier once the operation is
# selected at all; `region` and `city` are additive on top of it.
GEOGRAPHIC_TERM_KINDS: tuple[str, ...] = (
    GEOGRAPHIC_TERM_KIND_COUNTRY,
    GEOGRAPHIC_TERM_KIND_REGION,
    GEOGRAPHIC_TERM_KIND_CITY,
)

DEFAULT_GEOGRAPHIC_TERM_KINDS: tuple[str, ...] = (GEOGRAPHIC_TERM_KIND_COUNTRY,)

# Words that, sitting immediately before a matched term, mean the term is part of a
# larger place name this package does not carry rather than a subsidiary marker:
# `Bank of New England`, `Bank of New South Wales`. Multi-word terms that really are
# in the list (`New Jersey`, `West Virginia`, `South Carolina`) are matched
# longest-first, so they are consumed whole before this guard is ever consulted.
_PLACE_QUALIFIER_TOKENS: frozenset[str] = frozenset(
    {
        "new",
        "old",
        "great",
        "greater",
        "north",
        "northern",
        "south",
        "southern",
        "east",
        "eastern",
        "west",
        "western",
        "central",
        "upper",
        "lower",
    }
)

# A remainder ending on one of these is a fragment, not a name: `Bank of Ireland`
# must not become `Bank of`. Covers the English, French, German, Spanish, Italian and
# Dutch connectors that appear in the jurisdictions entity_legal_forms_iso20275.json
# already covers.
_DANGLING_CONNECTOR_TOKENS: frozenset[str] = frozenset(
    {
        "a",
        "al",
        "and",
        "as",
        "at",
        "da",
        "das",
        "de",
        "del",
        "della",
        "der",
        "des",
        "di",
        "die",
        "do",
        "dos",
        "du",
        "e",
        "el",
        "en",
        "et",
        "for",
        "fur",
        "het",
        "i",
        "in",
        "la",
        "las",
        "le",
        "les",
        "los",
        "of",
        "och",
        "on",
        "or",
        "the",
        "to",
        "und",
        "van",
        "von",
        "y",
        "zu",
    }
)

_BRACKETED_GROUP_PATTERN = re.compile(r"[(\[]([^()\[\]]*)[)\]]")
_NON_ALNUM_PATTERN = re.compile(r"[^0-9a-z]+")
_TRAILING_SEPARATORS = " \t,;:-–—/\\|."


def _fold(text: str) -> str:
    """Reduce text to the diacritic-free, punctuation-free lowercase form terms match on.

    The operation runs at the head of the normalization chain, before `punctuation`
    and `diacritics` have had a chance to fire, so both sides of a comparison have to
    be folded here rather than relying on the chain to have done it already. That is
    also what lets one entry (`Turkiye`) match a name spelling it `Türkiye`.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return _NON_ALNUM_PATTERN.sub(" ", without_marks.lower()).strip()


def _leading_segment(name: str) -> str:
    """Return the part of an ISO country name before its first parenthesis or comma.

    ISO 3166 formal names carry qualifiers a company name never repeats -- `Bolivia
    (Plurinational State of)`, `Korea (Republic of)`, `Bonaire, Sint Eustatius and
    Saba`. The full form is kept as its own term (harmless, since it never matches);
    this adds the part that actually appears in a name.
    """
    for separator in ("(", ","):
        index = name.find(separator)
        if index > 0:
            name = name[:index]
    return name.strip()


@lru_cache(maxsize=1)
def _load_elf_payload() -> dict[str, Any]:
    resource_path = files("company_cleanse.resources").joinpath(_ELF_RESOURCE_FILENAME)
    payload = json.loads(resource_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{_ELF_RESOURCE_FILENAME} must be an object.")
    return payload


def _require_geographic_terms_shape(
    payload: Any, *, source_label: str
) -> dict[str, Any]:
    """Validate the curated geographic-term payload's shape.

    Raises ``TypeError``/``ValueError`` describing the first problem found, and
    returns ``payload`` unchanged on success.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"{source_label} must be an object.")

    kinds = payload.get("kinds")
    if not isinstance(kinds, list) or not all(isinstance(kind, str) for kind in kinds):
        raise TypeError(f"{source_label} must include a 'kinds' list of strings.")

    unknown_kinds = sorted(set(kinds) - set(GEOGRAPHIC_TERM_KINDS))
    if unknown_kinds:
        raise ValueError(
            f"{source_label} declares unknown kinds {unknown_kinds}. "
            f"Supported kinds: {', '.join(GEOGRAPHIC_TERM_KINDS)}."
        )

    terms = payload.get("terms")
    if not isinstance(terms, list):
        raise TypeError(f"{source_label} must include a 'terms' list.")

    for index, entry in enumerate(terms):
        if not isinstance(entry, dict):
            raise TypeError(f"{source_label} terms[{index}] must be an object.")

        term = entry.get("term")
        if not isinstance(term, str) or not term.strip():
            raise ValueError(
                f"{source_label} terms[{index}] must include a non-empty 'term'."
            )

        kind = entry.get("kind")
        if kind not in GEOGRAPHIC_TERM_KINDS:
            raise ValueError(
                f"{source_label} terms[{index}] 'kind' must be one of "
                f"{', '.join(GEOGRAPHIC_TERM_KINDS)}, got {kind!r}."
            )

        note = entry.get("note")
        if note is not None and not isinstance(note, str):
            raise TypeError(f"{source_label} terms[{index}] 'note' must be a string.")

    return payload


@lru_cache(maxsize=1)
def _load_curated_payload() -> dict[str, Any]:
    resource_path = files("company_cleanse.resources").joinpath(
        _CURATED_RESOURCE_FILENAME
    )
    payload = json.loads(resource_path.read_text(encoding="utf-8"))
    return _require_geographic_terms_shape(
        payload, source_label=_CURATED_RESOURCE_FILENAME
    )


def derive_iso20275_geographic_terms() -> dict[str, tuple[str, ...]]:
    """Return the geographic terms carried by ``entity_legal_forms_iso20275.json``.

    ``country`` comes from that file's ``country`` field and ``region`` from its
    ``jurisdiction`` field with anything already a country removed, so `Canada` and
    `Guernsey` (which appear in both) stay countries. Case-folded duplicates collapse
    to their first spelling, which is what makes `Rhode Island`/`Rhode island` one
    term rather than two. Exposed rather than private so a test can re-derive it and
    assert the accessor has not drifted from the source resource.
    """
    countries: dict[str, str] = {}
    regions: dict[str, str] = {}

    for record in _load_elf_payload().values():
        if not isinstance(record, dict):
            continue

        country = str(record.get("country", "")).strip()
        if country:
            for candidate in (country, _leading_segment(country)):
                folded = _fold(candidate)
                if folded:
                    countries.setdefault(folded, candidate)

        jurisdiction = str(record.get("jurisdiction", "")).strip()
        if jurisdiction:
            folded = _fold(jurisdiction)
            if folded:
                regions.setdefault(folded, jurisdiction)

    for folded in countries:
        regions.pop(folded, None)

    return {
        GEOGRAPHIC_TERM_KIND_COUNTRY: tuple(sorted(countries.values())),
        GEOGRAPHIC_TERM_KIND_REGION: tuple(sorted(regions.values())),
        GEOGRAPHIC_TERM_KIND_CITY: (),
    }


def _validate_kind(kind: str) -> str:
    normalized = (kind or "").strip().lower()
    if normalized not in GEOGRAPHIC_TERM_KINDS:
        raise ValueError(
            f"Unknown geographic term kind {kind!r}. "
            f"Available kinds: {', '.join(GEOGRAPHIC_TERM_KINDS)}."
        )
    return normalized


@lru_cache(maxsize=len(GEOGRAPHIC_TERM_KINDS))
def get_geographic_terms(
    kind: str = GEOGRAPHIC_TERM_KIND_COUNTRY,
) -> tuple[str, ...]:
    """Return the packaged geographic terms for one tier, read-only.

    ``kind`` selects a single tier -- ``"country"`` (the default), ``"region"``, or
    ``"city"``. Each tier is the union of two sources: the terms derived from
    ``resources/entity_legal_forms_iso20275.json`` (its ``country`` and
    ``jurisdiction`` fields) and the curated colloquial layer in
    ``resources/geographic_terms.json``, which carries the short forms company names
    actually use and which ISO 3166's formal long forms cannot supply -- ``UK``,
    ``Britain``, ``USA``, ``Deutschland``. That file also records why each excluded
    candidate is excluded, including both ISO code lengths.

    ``"city"`` is wired end to end but carries no entries: the tier is selectable and
    strips nothing until a city source exists, so adding one later is a data change
    against the same resource rather than a second operation.

    There is no injection or override point, matching the contract
    ``get_corpus_noise_words()`` and ``get_manual_noise_words()`` already have: a
    caller selects tiers through a normalization profile and reads what the package
    holds.
    """
    normalized_kind = _validate_kind(kind)

    derived = derive_iso20275_geographic_terms()[normalized_kind]
    merged: dict[str, str] = {}
    for term in derived:
        folded = _fold(term)
        if folded:
            merged.setdefault(folded, term)

    for entry in _load_curated_payload()["terms"]:
        if entry["kind"] != normalized_kind:
            continue
        term = str(entry["term"]).strip()
        folded = _fold(term)
        if folded:
            merged.setdefault(folded, term)

    return tuple(sorted(merged.values()))


@lru_cache(maxsize=8)
def _geographic_term_index(
    kinds: tuple[str, ...],
) -> tuple[tuple[tuple[int, frozenset[str]], ...], int]:
    """Index the selected tiers' terms by token count, longest first.

    Longest-first is what keeps `New Jersey` from being read as a bare `Jersey` with
    a `New` in front of it, and `Saint Kitts and Nevis` from being read as `Nevis`.
    """
    by_length: dict[int, set[str]] = {}
    for kind in kinds:
        for term in get_geographic_terms(kind):
            folded = _fold(term)
            if not folded:
                continue
            by_length.setdefault(len(folded.split()), set()).add(folded)

    ordered = tuple(
        (length, frozenset(terms))
        for length, terms in sorted(by_length.items(), reverse=True)
    )
    return ordered, max(by_length, default=0)


def _strip_bracketed_geographic_terms(
    text: str,
    indexed_terms: tuple[tuple[int, frozenset[str]], ...],
) -> str:
    """Remove `(UK)`/`[Ireland]` groups whose whole content is a geographic term.

    Bracketing is itself the subsidiary marker, so this arm needs no positional test:
    a name does not put an integral part of itself in parentheses. It runs before the
    trailing arm and anywhere in the string, which is why `Siemens (UK) Ltd` resolves
    even though its trailing position is occupied by the legal form.
    """
    all_terms = {term for _, terms in indexed_terms for term in terms}
    if not all_terms:
        return text

    def _replace(match: re.Match[str]) -> str:
        folded = _fold(match.group(1))
        return " " if folded in all_terms else match.group(0)

    candidate = _BRACKETED_GROUP_PATTERN.sub(_replace, text)
    if candidate == text:
        return text

    if not _fold(candidate):
        # Everything the name had was inside the brackets; leave it alone.
        return text
    return re.sub(r"\s+", " ", candidate).strip()


def _strip_trailing_geographic_term(
    text: str,
    indexed_terms: tuple[tuple[int, frozenset[str]], ...],
    max_term_tokens: int,
    min_remaining_tokens: int,
) -> str:
    """Remove a trailing geographic term when it reads as a marker rather than a name part.

    The marker test is the whole of this function's caution, and it is what keeps
    `Air France`, `Bank of Ireland` and `Bank of America` intact while still
    resolving `Acme Systems Ireland`:

    - at least ``min_remaining_tokens`` tokens must survive. `Air France` fails here:
      one token is a fragment, not a shortened name. A caller that has already
      stripped a legal form knows the trailing term was a marker and passes 1;
      nothing else should.
    - the surviving remainder must not end on a connector. `Bank of Ireland` fails
      here, since `Bank of` is not a name.
    - the token before the term must not be a place-compounding qualifier.
      `Bank of New England` fails here, because `New England` is a place this package
      does not carry rather than `England` with a word in front of it.
    """
    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return text

    tokens = [match.group(0) for match in matches]
    folded_tokens = [_fold(token) for token in tokens]

    for length, terms in indexed_terms:
        if length > min(max_term_tokens, len(tokens)):
            continue

        remaining_count = len(tokens) - length
        if remaining_count < max(min_remaining_tokens, 1):
            continue

        candidate = " ".join(folded_tokens[-length:]).strip()
        if not candidate or candidate not in terms:
            continue

        preceding = folded_tokens[remaining_count - 1]
        if not preceding:
            # A punctuation-only token ('Acme & Ireland' folds '&' away entirely) is
            # as much of a fragment to end on as a spelled-out connector.
            return text
        if preceding in _DANGLING_CONNECTOR_TOKENS:
            return text
        if preceding in _PLACE_QUALIFIER_TOKENS:
            return text

        remainder = text[: matches[remaining_count - 1].end()]
        return remainder.rstrip(_TRAILING_SEPARATORS)

    return text


def strip_geographic_terms(
    text: str | None,
    *,
    kinds: tuple[str, ...] | None = None,
    min_remaining_tokens: int = 2,
) -> str | None:
    """Strip a trailing or bracketed geographic term from a name.

    Positional, never blanket: a geographic term is only removed when it is bracketed
    or trailing, because `Bank of Ireland`, `Air France`, `China Mobile` and `Bank of
    America` are real entities whose geographic term is integral. See
    ``_strip_trailing_geographic_term()`` for the additional marker test the trailing
    arm applies, which is what makes the first two of those survive.

    **Ordering constraint.** The trailing arm can only see the geographic term once
    the legal form is out of the way: `Oracle Ireland Ltd` presents `Ltd` in the
    trailing position first. Selected as the ``geographic_terms`` normalization
    operation this runs at the head of the chain, before punctuation stripping (which
    is what lets the bracketed arm see its brackets at all) and therefore before any
    legal-form handling -- so on that path a name still carrying its legal form is
    reached by the bracketed arm only. Selected as a short-name derivation stage it
    runs after company-type trimming, which is the path that resolves
    `Oracle Ireland Ltd` to `oracle`.

    ``kinds`` selects tiers; ``None`` is the default ``country`` tier alone.
    """
    if not text or not text.strip():
        return text

    resolved_kinds = resolve_geographic_term_kinds(kinds)
    indexed_terms, max_term_tokens = _geographic_term_index(resolved_kinds)
    if not indexed_terms:
        return text

    stripped = _strip_bracketed_geographic_terms(text, indexed_terms)
    return _strip_trailing_geographic_term(
        stripped,
        indexed_terms,
        max_term_tokens,
        min_remaining_tokens,
    )


def resolve_geographic_term_kinds(kinds: tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize a tier selection to validated, deduplicated, tier-ordered kinds.

    ``None`` or an empty selection resolves to the default ``country`` tier, so
    naming the operation without naming a tier gives the common path.
    """
    if not kinds:
        return DEFAULT_GEOGRAPHIC_TERM_KINDS

    resolved = {_validate_kind(kind) for kind in kinds}
    resolved.add(GEOGRAPHIC_TERM_KIND_COUNTRY)
    return tuple(kind for kind in GEOGRAPHIC_TERM_KINDS if kind in resolved)
