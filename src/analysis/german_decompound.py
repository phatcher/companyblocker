"""German compound-word splitting via a Koehn & Knight (2003)-style frequency search.

Real German company names often compound legal-form/qualifier morphemes into
one agglutinated word with no internal hyphen or capital-letter boundary
(e.g. `"Krebsregister"` = `"Krebs"` + `"Register"`), which a hyphen-only or
whitespace-only tokenizer cannot see. `split_compound()` tries every split
point (with German linking-element removal at the boundary), scores each
candidate split by the geometric mean of `wordfreq` word frequency for its
parts, and keeps the best-scoring split only if it clearly beats leaving the
word whole -- recursing once more per resulting part so a three-morpheme
compound's later part can itself split again.

This lives here (not in `company_cleanse`) because it depends on `wordfreq`,
which `company_cleanse` is deliberately kept free of as a runtime dependency
so it stays consumable as a standalone, dependency-light package outside
this monorepo. A caller wires `split_compound` into
`company_cleanse.short_name_candidates`'s `decompound_fn` injection point
rather than the package taking on the dependency itself.
"""

from __future__ import annotations

from functools import lru_cache

import wordfreq

_MIN_PART_LENGTH = 3
# Common German linking elements ("Fugenelemente") inserted between compound
# parts, tried in longest-first order so "es" is preferred over "s" where
# both would leave a valid remainder.
_LINKING_ELEMENTS = ("es", "en", "er", "e", "s", "n")
# A split must beat the whole, unsplit word's own frequency score by this
# factor to be preferred -- otherwise a real (if less common) whole word
# would keep getting fragmented into coincidentally-frequent pieces.
_MIN_SCORE_RATIO = 1.5
_MAX_SPLIT_DEPTH = 2


@lru_cache(maxsize=16384)
def _zipf(word: str) -> float:
    # zipf_frequency returns 0.0 for unknown words -- never negative, so a
    # floor isn't needed before the geometric mean below.
    return wordfreq.zipf_frequency(word, "de")


def _candidate_left_forms(left: str) -> list[str]:
    """The raw left part plus every plausible linking-element-stripped form."""
    forms = [left]
    for linking in _LINKING_ELEMENTS:
        if left.endswith(linking) and len(left) - len(linking) >= _MIN_PART_LENGTH:
            forms.append(left[: -len(linking)])
    return forms


def _best_two_way_split(word: str) -> tuple[str, str] | None:
    """The highest-scoring two-way split of `word`, or `None` if none clears the bar."""
    best: tuple[str, str] | None = None
    best_score = -1.0
    for split_at in range(_MIN_PART_LENGTH, len(word) - _MIN_PART_LENGTH + 1):
        raw_left, right = word[:split_at], word[split_at:]
        right_score = _zipf(right)
        for left_candidate in _candidate_left_forms(raw_left):
            score = (_zipf(left_candidate) * right_score) ** 0.5
            if score > best_score:
                best_score = score
                best = (left_candidate, right)

    if best is None:
        return None
    if best_score < _zipf(word) * _MIN_SCORE_RATIO:
        return None
    return best


def split_compound(word: str, *, _depth: int = 0) -> list[str]:
    """Split one German word into its likely morphemes.

    Returns `[word.lower()]` (unsplit) when no candidate split's frequency
    score clearly beats the whole word's own, including for words too short
    to hold two `_MIN_PART_LENGTH`-length parts.
    """
    lowered = word.lower()
    if len(lowered) < _MIN_PART_LENGTH * 2 or _depth >= _MAX_SPLIT_DEPTH:
        return [lowered]

    split = _best_two_way_split(lowered)
    if split is None:
        return [lowered]

    left, right = split
    parts: list[str] = []
    for part in (left, right):
        parts.extend(split_compound(part, _depth=_depth + 1))
    return parts
