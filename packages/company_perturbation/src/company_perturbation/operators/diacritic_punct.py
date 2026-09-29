"""Diacritic and separator family: accented letters, and space/hyphen/ampersand swaps.

TAXONOMY.md family 6. Both operators are character substitutions, so a site is a
single character and the change picks a replacement for it.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from ..families import Family
from ..sited_operator import Site, SitedOperator, sited_registry
from .casing import match_case

# Latin diacritic variants of an ASCII base letter, restricted to characters that
# `company_cleanse.normalize.strip_diacritics` reduces cleanly back to that base, and
# covering the jurisdictions in entity_legal_forms_iso20275.json (French, Irish, Iberian,
# Nordic). Deliberately excludes c-cedilla, o-umlaut and u-umlaut: those are reachable
# through the homoglyph operator, and covering them twice would double two families over
# the same characters. The exclusion is asserted by a test, not left to this comment.
_DIACRITICS: dict[str, tuple[str, ...]] = {
    "a": ("á", "à", "â", "ä", "ã", "å"),
    "e": ("é", "è", "ê", "ë"),
    "i": ("í", "ì", "î", "ï"),
    "o": ("ó", "ò", "ô", "õ"),
    "u": ("ú", "ù", "û"),
    "y": ("ý", "ÿ"),
    "n": ("ñ",),
}

_SEPARATORS = (" ", "-", "&")


def _diacritic_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every character with a known accented form."""
    return tuple(
        Site(i, i + 1) for i, ch in enumerate(name) if ch.lower() in _DIACRITICS
    )


def _separator_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every position already holding one of the three interchangeable separators.

    Nothing is inserted where no separator exists, and apostrophes are untouched:
    those are word-internal (possessives, contractions) rather than inter-word
    separator noise, and treating them here would be a guess rather than a family.
    """
    return tuple(Site(i, i + 1) for i, ch in enumerate(name) if ch in _SEPARATORS)


def _add_diacritic(name: str, site: Site, rng: Random, country: str | None) -> str:
    original = name[site.start]
    accented = match_case(original, rng.choice(_DIACRITICS[original.lower()]))
    return name[: site.start] + accented + name[site.end :]


def _swap_separator(name: str, site: Site, rng: Random, country: str | None) -> str:
    current = name[site.start]
    alternatives = tuple(sep for sep in _SEPARATORS if sep != current)
    return name[: site.start] + rng.choice(alternatives) + name[site.end :]


for _operator in (
    SitedOperator(
        operator_id="diacritic_punct.diacritic_substitution",
        family=Family.DIACRITIC_PUNCT,
        sites=_diacritic_sites,
        change=_add_diacritic,
    ),
    SitedOperator(
        operator_id="diacritic_punct.separator_substitution",
        family=Family.DIACRITIC_PUNCT,
        sites=_separator_sites,
        change=_swap_separator,
    ),
):
    sited_registry.register(_operator)
