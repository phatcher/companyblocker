"""Phonetic family: respell a word so it still sounds the same.

TAXONOMY.md family 2. A site is a word that has at least one respelling preserving
its Soundex code, and the change swaps in one of them.

The substitution table is hand-authored but not trusted on its own: every candidate
is kept only when `jellyfish.soundex(candidate) == jellyfish.soundex(word)`, so the
table is grounded in a real, tested phonetic algorithm rather than in judgement.
Non-alphabetic words are never eligible, since Soundex has no defined meaning for
digits or punctuation.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

import jellyfish

from ..families import Family
from ..sited_operator import Site, SitedOperator, sited_registry
from .casing import match_case

_SUBSTITUTION_PAIRS: tuple[tuple[str, str], ...] = (
    ("ph", "f"),
    ("ck", "k"),
    ("y", "i"),
    ("ie", "y"),
    ("s", "z"),
    ("c", "k"),
)


def _word_candidates(word: str) -> tuple[str, ...]:
    """Every single-substitution respelling of `word` that keeps its Soundex code."""
    if not word.isalpha():
        return ()

    # The substitution table is lowercase; fold for matching and let the caller put
    # the original capitalisation back.
    word = word.lower()
    original_code = jellyfish.soundex(word)
    candidates: set[str] = set()
    for first, second in _SUBSTITUTION_PAIRS:
        for old, new in ((first, second), (second, first)):
            start = 0
            while True:
                index = word.find(old, start)
                if index == -1:
                    break
                candidate = word[:index] + new + word[index + len(old) :]
                if candidate != word and jellyfish.soundex(candidate) == original_code:
                    candidates.add(candidate)
                start = index + 1

    return tuple(sorted(candidates))


def _respellable_word_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every word with at least one Soundex-preserving respelling."""
    sites: list[Site] = []
    position = 0
    for word in name.split(" "):
        if _word_candidates(word):
            sites.append(Site(position, position + len(word)))
        position += len(word) + 1
    return tuple(sites)


def _respell(name: str, site: Site, rng: Random, country: str | None) -> str:
    word = name[site.start : site.end]
    respelled = match_case(word, rng.choice(_word_candidates(word)))
    return name[: site.start] + respelled + name[site.end :]


sited_registry.register(
    SitedOperator(
        operator_id="phonetic.soundex_swap",
        family=Family.PHONETIC,
        sites=_respellable_word_sites,
        change=_respell,
    )
)
