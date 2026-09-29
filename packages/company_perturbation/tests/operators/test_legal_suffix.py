"""Direct tests for `company_perturbation.operators.legal_suffix`.

The family the inverted contract helps most: at most one site, which under the
previous shared edit-count formula degenerated into a threshold with a rounding
cliff and an unused seed. Here it is an ordinary site finder.
"""

from __future__ import annotations

from random import Random

from company_perturbation.operators.legal_suffix import (
    _drop_suffix,
    _substitute_suffix,
    _substitution_candidates,
    _suffix_site,
)


def test_a_name_has_at_most_one_suffix_site():
    sites = _suffix_site("acme holdings limited", None)

    assert len(sites) == 1


def test_a_name_with_no_recognised_suffix_offers_no_site():
    assert _suffix_site("acme holdings", None) == ()


def test_detection_ignores_the_country():
    """Recognising that a suffix is present stays permissive across jurisdictions.

    Only choosing a replacement is country-scoped.
    """
    assert _suffix_site("acme gmbh", "gb") == _suffix_site("acme gmbh", "de")


def test_dropping_removes_the_suffix_and_the_space_before_it():
    site = _suffix_site("acme holdings limited", None)[0]

    assert _drop_suffix("acme holdings limited", site, Random(1), None) == (
        "acme holdings"
    )


def test_substitution_candidates_exclude_the_matched_spelling():
    assert "ltd" not in _substitution_candidates("gb", "ltd")


def test_a_curated_cluster_widens_the_pool_to_a_real_conversion():
    """`ltd` and `plc` are a real corporate conversion, so sources disagree legitimately."""
    assert "plc" in _substitution_candidates("gb", "ltd")


def test_an_uncurated_canonical_is_never_offered():
    """`CIO` is a Charity Commission structure with no path from a company."""
    assert "cio" not in _substitution_candidates("gb", "ltd")


def test_an_unknown_country_has_no_candidates():
    assert _substitution_candidates("zz", "ltd") == ()


def test_substitution_replaces_the_suffix_within_the_country():
    site = _suffix_site("acme holdings ltd", None)[0]
    mutated = _substitute_suffix("acme holdings ltd", site, Random(3), "gb")

    assert mutated.startswith("acme holdings ")
    assert not mutated.endswith(" ltd")


def test_with_no_country_it_falls_back_to_dropping():
    """Better a real name than a British company rendered as a GmbH."""
    site = _suffix_site("acme holdings ltd", None)[0]

    assert _substitute_suffix("acme holdings ltd", site, Random(3), None) == (
        "acme holdings"
    )
