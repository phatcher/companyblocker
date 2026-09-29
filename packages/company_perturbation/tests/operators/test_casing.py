"""Direct tests for `company_perturbation.operators.casing`.

Every lookup table in this package is lowercase and real register data is not,
so this is what stands between a table match and an operator that quietly does
nothing on `MAGUIRE GROUP LIMITED`. The three patterns it carries over are
asserted separately from the case it deliberately declines to guess at.
"""

from __future__ import annotations

from company_perturbation.operators.casing import match_case


def test_an_upper_case_original_makes_the_replacement_upper_case():
    assert match_case("LIMITED", "ltd") == "LTD"


def test_a_lower_case_original_makes_the_replacement_lower_case():
    assert match_case("limited", "LTD") == "ltd"


def test_a_title_case_original_makes_the_replacement_title_case():
    assert match_case("Limited", "ltd") == "Ltd"


def test_a_multi_word_replacement_is_titled_word_by_word():
    assert match_case("Limited", "public limited company") == "Public Limited Company"


def test_a_mixed_case_original_is_left_as_the_table_gave_it():
    """There is no pattern to copy, and guessing one is worse than not trying."""
    assert match_case("LiMiTeD", "ltd") == "ltd"


def test_an_original_with_no_case_at_all_is_left_as_the_table_gave_it():
    assert match_case("2000", "ltd") == "ltd"


def test_an_empty_original_carries_nothing_over():
    assert match_case("", "ltd") == "ltd"


def test_an_empty_replacement_stays_empty():
    assert match_case("LIMITED", "") == ""
