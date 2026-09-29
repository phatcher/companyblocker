"""Direct tests for `company_perturbation.operators.phonetic`."""

from __future__ import annotations

from random import Random

import jellyfish
from company_perturbation.operators.phonetic import (
    _respell,
    _respellable_word_sites,
    _word_candidates,
)
from company_perturbation.sited_operator import Site


def test_every_candidate_keeps_the_word_s_soundex_code():
    """The table is hand-authored; Soundex is what keeps it honest."""
    for word in ("phoenix", "cisco", "system"):
        for candidate in _word_candidates(word):
            assert jellyfish.soundex(candidate) == jellyfish.soundex(word)


def test_a_non_alphabetic_word_has_no_candidates():
    """Soundex has no defined meaning for digits or punctuation."""
    assert _word_candidates("123") == ()
    assert _word_candidates("a-b") == ()


def test_sites_are_only_the_words_that_can_be_respelled():
    name = "phoenix 123"
    sites = _respellable_word_sites(name, None)

    assert [(site.start, site.end) for site in sites] == [(0, 7)]


def test_respelling_changes_the_word_but_not_how_it_sounds():
    mutated = _respell("phoenix ltd", Site(0, 7), Random(1), None)

    assert mutated != "phoenix ltd"
    assert mutated.endswith(" ltd")
    assert jellyfish.soundex(mutated.split(" ")[0]) == jellyfish.soundex("phoenix")
