"""Direct tests for `company_perturbation.operators.homoglyph`."""

from __future__ import annotations

from random import Random

from company_perturbation.operators.homoglyph import (
    _HOMOGLYPHS,
    _homoglyph_sites,
    _swap_homoglyph,
)
from company_perturbation.sited_operator import Site


def test_sites_are_only_characters_with_a_known_confusable():
    sites = _homoglyph_sites("acme 123", None)

    for site in sites:
        assert "acme 123"[site.start] in _HOMOGLYPHS


def test_a_name_with_nothing_confusable_offers_no_site():
    assert _homoglyph_sites("123", None) == ()


def test_swap_replaces_one_character_with_a_confusable_of_it():
    mutated = _swap_homoglyph("acme", Site(0, 1), Random(1), None)

    assert mutated[1:] == "cme"
    assert mutated[0] in _HOMOGLYPHS["a"]
    assert mutated[0] != "a"


def test_the_replacement_is_not_ascii():
    """A confusable that were itself ASCII would be a typo, not a transliteration."""
    mutated = _swap_homoglyph("acme", Site(0, 1), Random(3), None)

    assert not mutated[0].isascii()
