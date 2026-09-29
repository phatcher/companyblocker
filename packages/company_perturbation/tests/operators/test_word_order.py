"""Direct tests for `company_perturbation.operators.word_order`."""

from __future__ import annotations

from random import Random

from company_perturbation.operators.word_order import (
    _adjacent_word_pair_sites,
    _swap_words,
)
from company_perturbation.sited_operator import Site


def _spans(sites):
    return [(site.start, site.end) for site in sites]


def test_sites_are_the_neighbouring_word_pairs():
    assert _spans(_adjacent_word_pair_sites("acme holdings ltd", None)) == [
        (0, 13),
        (5, 17),
    ]


def test_a_single_word_name_has_no_pair_and_is_never_mutated():
    assert _adjacent_word_pair_sites("acme", None) == ()


def test_swapping_exchanges_the_two_words_and_leaves_the_rest():
    assert _swap_words("acme holdings ltd", Site(0, 13), Random(1), None) == (
        "holdings acme ltd"
    )
    assert _swap_words("acme holdings ltd", Site(5, 17), Random(1), None) == (
        "acme ltd holdings"
    )


def test_a_swap_preserves_every_word():
    original = "acme holdings group ltd"
    holdings_group = _adjacent_word_pair_sites(original, None)[1]
    mutated = _swap_words(original, holdings_group, Random(1), None)

    assert mutated == "acme group holdings ltd"
    assert sorted(mutated.split(" ")) == sorted(original.split(" "))
