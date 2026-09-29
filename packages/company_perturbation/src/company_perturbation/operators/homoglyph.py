"""Confusable-character family: a visually similar character for an ASCII one.

TAXONOMY.md family 3, narrowed: a character-for-character confusable swap (Cyrillic
'а' for Latin 'a'), not the full script-variant rendering the rest of that family
still defers. That narrowing is what makes it buildable without a non-Latin company
corpus to validate against — correctness comes from character-shape identity, not
from matching real foreign-language names.

The table is `company_cleanse.get_ascii_homoglyphs()`, the inverse of its already
tested `transliterate_to_ascii` fold tables, rather than a vendored copy or a
general confusables library: a general table covers far more scripts than this needs
and would change the candidate set — and therefore output for a given generator
state — on every version bump.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from company_cleanse import get_ascii_homoglyphs

from ..families import Family
from ..sited_operator import Site, SitedOperator, sited_registry
from .casing import match_case

_HOMOGLYPHS: dict[str, tuple[str, ...]] = get_ascii_homoglyphs()


def _homoglyph_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every character with at least one known confusable."""
    return tuple(
        Site(i, i + 1) for i, ch in enumerate(name) if ch.lower() in _HOMOGLYPHS
    )


def _swap_homoglyph(name: str, site: Site, rng: Random, country: str | None) -> str:
    original = name[site.start]
    confusable = match_case(original, rng.choice(_HOMOGLYPHS[original.lower()]))
    return name[: site.start] + confusable + name[site.end :]


sited_registry.register(
    SitedOperator(
        operator_id="transliteration.homoglyph_substitution",
        family=Family.TRANSLITERATION,
        sites=_homoglyph_sites,
        change=_swap_homoglyph,
    )
)
