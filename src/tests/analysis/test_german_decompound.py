from __future__ import annotations

from analysis.german_decompound import split_compound


def test_splits_the_headline_example():
    # "Schwimm-Startgemeinschaft Leipzig" -> "SSG Leipzig" needs
    # "Startgemeinschaft" to split into "Start" + "Gemeinschaft".
    assert split_compound("Startgemeinschaft") == ["start", "gemeinschaft"]


def test_splits_krebsregister():
    assert split_compound("Krebsregister") == ["krebs", "register"]


def test_splits_vertrauensstelle_with_linking_element_removed():
    # "Vertrauensstelle" = "Vertrauen" + linking "s" + "Stelle"; the
    # linking element must be stripped from the left part, not kept.
    assert split_compound("Vertrauensstelle") == ["vertrauen", "stelle"]


def test_leaves_a_short_word_unsplit():
    assert split_compound("Bahn") == ["bahn"]


def test_leaves_an_ordinary_whole_word_unsplit():
    # A real, common standalone word should not be fragmented just because
    # some substring split happens to score nonzero.
    assert split_compound("Sparkasse") == ["sparkasse"]


def test_output_is_lowercase_even_when_unsplit():
    assert split_compound("BAHN") == ["bahn"]


def test_recursion_depth_is_bounded():
    # Every returned part is itself unsplittable (or depth-exhausted) --
    # this mostly guards against infinite/runaway recursion on adversarial
    # input rather than asserting a specific split shape.
    result = split_compound("Bundesverkehrsministeriumsangelegenheiten")
    assert isinstance(result, list)
    assert all(isinstance(part, str) and part for part in result)
