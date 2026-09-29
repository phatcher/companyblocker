"""Fixture-registry tests for the generalised jurisdiction selection rule.

No real checkpoint of any kind is loaded here -- `resolve_checkpoint_for_jurisdictions()` reads
only `coverage`/`jurisdictions` off whatever entries it is handed, so a lightweight fixture
entry exercises every fallback the shape supports.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from company_classify.checkpoint_selection import resolve_checkpoint_for_jurisdictions

_DEFAULT = "default-checkpoint"


@dataclass(frozen=True)
class _FixtureEntry:
    identifier: str
    coverage: str
    jurisdictions: tuple[str, ...]


_REGISTRY = (
    _FixtureEntry("en-checkpoint", "monolingual", ("gb", "ie")),
    _FixtureEntry("fr-checkpoint", "monolingual", ("fr",)),
    _FixtureEntry("de-checkpoint", "monolingual", ("de",)),
    _FixtureEntry("multi-checkpoint", "multilingual", ("gb", "ie", "fr", "de")),
)


def _resolve(jurisdictions: tuple[str, ...] | None) -> str:
    return resolve_checkpoint_for_jurisdictions(
        jurisdictions,
        _REGISTRY,
        identifier=lambda entry: entry.identifier,
        default=_DEFAULT,
    )


@pytest.mark.parametrize(
    ("jurisdictions", "expected"),
    [
        (("gb",), "en-checkpoint"),
        (("ie",), "en-checkpoint"),
        (("gb", "ie"), "en-checkpoint"),
        (("fr",), "fr-checkpoint"),
        (("de",), "de-checkpoint"),
    ],
)
def test_resolves_the_matched_monolingual_entry(jurisdictions, expected):
    assert _resolve(jurisdictions) == expected


@pytest.mark.parametrize("jurisdictions", [None, ()])
def test_falls_back_when_jurisdictions_are_unset(jurisdictions):
    assert _resolve(jurisdictions) == _DEFAULT


def test_falls_back_when_jurisdictions_span_no_single_monolingual_entry():
    # ("fr", "de") together match no single monolingual entry's coverage -- each covers
    # only one of the two -- so this is ambiguous rather than a match.
    assert _resolve(("fr", "de")) == _DEFAULT


def test_falls_back_when_jurisdictions_match_no_registered_entry():
    assert _resolve(("xx",)) == _DEFAULT


def test_never_returns_a_multilingual_entry_even_when_it_would_cover_the_jurisdictions():
    # The multilingual entry covers every jurisdiction in the fixture registry, but it must
    # never be picked as an inferred default -- only an explicit override reaches it.
    assert _resolve(("gb", "ie", "fr", "de")) == _DEFAULT
