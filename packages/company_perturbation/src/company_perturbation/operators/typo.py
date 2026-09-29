"""Keying-error family: substitution, transposition, insertion, deletion, duplication.

TAXONOMY.md family 1. Each operator declares its candidate sites and how to make
*one* change at one of them; how many are attempted is the runner's business
(`sited_operator.apply_operator`), so nothing here counts edits.

That split is what makes these plausible as keying errors: a real typist makes a
small, absolute number of mistakes regardless of how long the name is, which is
what a step's own `attempts` states and what no operator here can inflate.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from ..families import Family
from ..sited_operator import Site, SitedOperator, sited_registry
from .casing import match_case

# Simplified QWERTY adjacency: lowercase letters only, immediate horizontal/vertical
# neighbours on a standard layout. Not a physical key-distance model (no diagonals, no
# digits or punctuation) -- TAXONOMY.md family 1 scoped this to a small hand-authored map.
_ADJACENT: dict[str, str] = {
    "q": "wa",
    "w": "qeas",
    "e": "wrds",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",
    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",
    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def _adjacent_letter_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every character with a known keyboard neighbour."""
    return tuple(Site(i, i + 1) for i, ch in enumerate(name) if ch.lower() in _ADJACENT)


def _any_character_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every character. A digit or punctuation mark is as easy to hit twice as a letter."""
    return tuple(Site(i, i + 1) for i in range(len(name)))


def _adjacent_pair_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every adjacent character pair, the unit a transposition swaps."""
    return tuple(Site(i, i + 2) for i in range(len(name) - 1))


def _deletable_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every character, unless the name is down to its last one.

    The floor keeps a name from being deleted away to nothing, a degenerate case
    with no value as a blocking input.
    """
    if len(name) <= 1:
        return ()
    return tuple(Site(i, i + 1) for i in range(len(name)))


def _substitute(name: str, site: Site, rng: Random, country: str | None) -> str:
    original = name[site.start]
    replacement = match_case(original, rng.choice(_ADJACENT[original.lower()]))
    return name[: site.start] + replacement + name[site.end :]


def _transpose(name: str, site: Site, rng: Random, country: str | None) -> str:
    first, second = name[site.start], name[site.start + 1]
    return name[: site.start] + second + first + name[site.end :]


def _insert(name: str, site: Site, rng: Random, country: str | None) -> str:
    original = name[site.start]
    neighbour = match_case(original, rng.choice(_ADJACENT[original.lower()]))
    return name[: site.end] + neighbour + name[site.end :]


def _delete(name: str, site: Site, rng: Random, country: str | None) -> str:
    return name[: site.start] + name[site.end :]


def _duplicate(name: str, site: Site, rng: Random, country: str | None) -> str:
    return name[: site.end] + name[site.start : site.end] + name[site.end :]


for _operator in (
    SitedOperator(
        operator_id="typo.keyboard_substitution",
        family=Family.TYPO,
        sites=_adjacent_letter_sites,
        change=_substitute,
    ),
    SitedOperator(
        operator_id="typo.transposition",
        family=Family.TYPO,
        sites=_adjacent_pair_sites,
        change=_transpose,
    ),
    SitedOperator(
        operator_id="typo.insertion",
        family=Family.TYPO,
        sites=_adjacent_letter_sites,
        change=_insert,
    ),
    SitedOperator(
        operator_id="typo.deletion",
        family=Family.TYPO,
        sites=_deletable_sites,
        change=_delete,
    ),
    SitedOperator(
        operator_id="typo.duplication",
        family=Family.TYPO,
        sites=_any_character_sites,
        change=_duplicate,
    ),
):
    sited_registry.register(_operator)
