"""Direct tests for `company_perturbation.operators.low_salience`."""

from __future__ import annotations

from random import Random

from company_cleanse import get_corpus_noise_words
from company_perturbation.operators.low_salience import _drop_token, _noise_word_sites
from company_perturbation.sited_operator import Site


def test_a_single_token_name_offers_no_site():
    """The floor that keeps at least one token alive, expressed as a site condition."""
    assert _noise_word_sites("acme", None) == ()


def test_only_noise_words_are_sites():
    noise_words = set(get_corpus_noise_words(None))
    name = " ".join(["zzqxacme", *sorted(noise_words)[:2]])

    sites = _noise_word_sites(name, None)

    for site in sites:
        assert name[site.start : site.end].lower() in noise_words
    assert all(name[site.start : site.end] != "zzqxacme" for site in sites)


def test_dropping_a_middle_token_removes_it_and_one_separator():
    assert _drop_token("acme systems ltd", Site(5, 12), Random(1), None) == "acme ltd"


def test_dropping_the_last_token_removes_the_separator_before_it():
    assert _drop_token("acme systems", Site(5, 12), Random(1), None) == "acme"


def test_a_dropped_name_is_a_subsequence_of_the_original():
    """Tokens are only ever removed, never replaced or reordered.

    This is what makes the family unable to manufacture a false same-entity claim
    between two different companies.
    """
    original = "acme systems ltd"
    mutated = _drop_token(original, Site(5, 12), Random(1), None)

    remaining = iter(original.split(" "))
    assert all(token in remaining for token in mutated.split(" "))
