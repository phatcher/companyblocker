"""Direct tests for `company_perturbation.operators.diacritic_punct`."""

from __future__ import annotations

from random import Random

from company_cleanse import get_ascii_homoglyphs
from company_perturbation.operators.diacritic_punct import (
    _DIACRITICS,
    _add_diacritic,
    _diacritic_sites,
    _separator_sites,
    _swap_separator,
)
from company_perturbation.sited_operator import Site


def _spans(sites):
    return [(site.start, site.end) for site in sites]


def test_diacritic_sites_are_the_letters_with_an_accented_form():
    assert _spans(_diacritic_sites("acme", None)) == [(0, 1), (3, 4)]


def test_separator_sites_are_only_existing_separators():
    """Nothing is inserted where no separator is; apostrophes are left alone."""
    assert _spans(_separator_sites("o'brien & sons-ltd", None)) == [
        (7, 8),
        (8, 9),
        (9, 10),
        (14, 15),
    ]


def test_add_diacritic_replaces_one_letter_with_an_accented_form():
    mutated = _add_diacritic("acme", Site(0, 1), Random(1), None)

    assert mutated[1:] == "cme"
    assert mutated[0] in _DIACRITICS["a"]


def test_swap_separator_never_returns_the_separator_it_replaced():
    for seed in range(8):
        assert _swap_separator("a b", Site(1, 2), Random(seed), None) != "a b"


def test_the_diacritic_table_does_not_overlap_the_homoglyph_table():
    """Two families must not both own the same characters.

    c-cedilla, o-umlaut and u-umlaut are reachable through the homoglyph operator,
    so they are deliberately absent here; this asserts the exclusion rather than
    leaving it to a comment.
    """
    homoglyphs = get_ascii_homoglyphs()
    for base, accented in _DIACRITICS.items():
        overlap = set(accented) & set(homoglyphs.get(base, ()))
        assert not overlap, f"{base!r} is covered by both families: {sorted(overlap)}"
