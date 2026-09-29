"""Direct tests for `company_perturbation.operators.typo`.

Under the inverted contract these are plain equality assertions: a site finder is
a pure function of the name, and three of the five changes never touch the
generator at all. Only substitution and insertion draw, and then only to pick
which adjacent key, so a seeded generator pins them exactly.
"""

from __future__ import annotations

from random import Random

from company_perturbation.operators.typo import (
    _adjacent_letter_sites,
    _adjacent_pair_sites,
    _any_character_sites,
    _deletable_sites,
    _delete,
    _duplicate,
    _insert,
    _substitute,
    _transpose,
)
from company_perturbation.sited_operator import Site, sited_registry


def _spans(sites):
    return [(site.start, site.end) for site in sites]


def test_adjacent_letter_sites_skips_characters_with_no_keyboard_neighbour():
    """Spaces and punctuation have no neighbour; case is irrelevant.

    The adjacency map is lowercase, but a real register holds `MAGUIRE GROUP
    LIMITED`. Matching case-sensitively made typos fire on 378 of 817,761 Irish
    rows instead of roughly two thirds of them, so lookups fold and the
    replacement carries the original's case back.
    """
    assert _spans(_adjacent_letter_sites("Ab c-d", None)) == [
        (0, 1),
        (1, 2),
        (3, 4),
        (5, 6),
    ]


def test_any_character_sites_covers_everything():
    assert _spans(_any_character_sites("a b", None)) == [(0, 1), (1, 2), (2, 3)]


def test_adjacent_pair_sites_are_the_overlapping_pairs():
    assert _spans(_adjacent_pair_sites("abcd", None)) == [(0, 2), (1, 3), (2, 4)]


def test_a_single_character_name_has_no_pair_to_transpose():
    assert _adjacent_pair_sites("a", None) == ()


def test_a_single_character_name_offers_nothing_to_delete():
    """The floor that stops a name being deleted away to nothing."""
    assert _deletable_sites("a", None) == ()
    assert _spans(_deletable_sites("ab", None)) == [(0, 1), (1, 2)]


def test_transpose_swaps_the_pair_and_touches_nothing_else():
    assert _transpose("acme", Site(1, 3), Random(1), None) == "amce"


def test_delete_removes_exactly_the_site():
    assert _delete("acme", Site(1, 2), Random(1), None) == "ame"


def test_duplicate_repeats_the_site_in_place():
    assert _duplicate("acme", Site(1, 2), Random(1), None) == "accme"


def test_substitute_replaces_with_a_keyboard_neighbour():
    mutated = _substitute("acme", Site(0, 1), Random(1), None)

    assert len(mutated) == 4
    assert mutated[1:] == "cme"
    assert mutated[0] in "qwsz"


def test_insert_adds_a_keyboard_neighbour_after_the_site():
    mutated = _insert("acme", Site(0, 1), Random(1), None)

    assert len(mutated) == 5
    assert mutated.startswith("a")
    assert mutated[2:] == "cme"
    assert mutated[1] in "qwsz"


def test_the_family_registers_its_five_operators():
    registered = sited_registry.list_operators()

    for operator_id in (
        "typo.keyboard_substitution",
        "typo.transposition",
        "typo.insertion",
        "typo.deletion",
        "typo.duplication",
    ):
        assert operator_id in registered
