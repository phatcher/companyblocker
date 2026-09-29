"""Word-order family: swap two neighbouring words.

TAXONOMY.md family 5. Assumes `name` is single-space-delimited text with no
leading, trailing or doubled spaces (`company_cleanse`'s `name_cleansed` shape);
behaviour on other input is undefined.

A site is an adjacent word *pair*, and one change swaps it: the same treatment
`typo.transposition` gets for characters. An adjacent swap is the elementary edit
real word-order variation takes, since company names differ by neighbouring words
trading places rather than by an arbitrary permutation. Repeated adjacent swaps
still reach any permutation when a profile asks for enough attempts.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from ..families import Family
from ..sited_operator import Site, SitedOperator, sited_registry


def _word_spans(name: str) -> list[Site]:
    spans: list[Site] = []
    position = 0
    for word in name.split(" "):
        spans.append(Site(position, position + len(word)))
        position += len(word) + 1
    return spans


def _adjacent_word_pair_sites(name: str, country: str | None) -> Sequence[Site]:
    """Every neighbouring word pair, spanning the first word's start to the second's end.

    A single-word name has no pair, so it offers no site and is never mutated.
    """
    words = _word_spans(name)
    if len(words) < 2:
        return ()
    return tuple(Site(words[i].start, words[i + 1].end) for i in range(len(words) - 1))


def _swap_words(name: str, site: Site, rng: Random, country: str | None) -> str:
    pair = name[site.start : site.end]
    first, _, second = pair.partition(" ")
    return name[: site.start] + f"{second} {first}" + name[site.end :]


sited_registry.register(
    SitedOperator(
        operator_id="word_order.word_shuffle",
        family=Family.WORD_ORDER,
        sites=_adjacent_word_pair_sites,
        change=_swap_words,
    )
)
