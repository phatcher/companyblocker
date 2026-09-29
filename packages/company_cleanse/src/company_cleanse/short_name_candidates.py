"""Layered deterministic short-name/significant-token candidate derivation.

Each layer below is a separate, independently testable pattern rather than a
fused distance function: given a full name, a layer either produces one
candidate string or declines (`None`). `derive_short_name_candidate()` runs a
caller-selected, caller-ordered subset of layers and returns the first hit,
tagged with which layer produced it, or an explicit "no confident
derivation" (`None`) -- never a forced low-confidence guess. Which layers run
and in what order is a caller decision (`DEFAULT_LAYER_ORDER` records the
order this package's own real-data measurement settled on); nothing here
picks that order for itself.

Layers:

- `exact_initials`: every token contributes its first letter, in order
  (`"International Business Machines"` -> `"IBM"`). With an injected
  `decompound_fn`, a compound token contributes one letter per sub-part
  instead of just its own first letter (`"Deutscher Skiverband"` ->
  `"DSV"`, once `decompound_fn` splits `"Skiverband"` into `"Ski"` +
  `"Verband"`; without it, the same name yields only `"DS"`).
- `subsequence_initials`: like `exact_initials`, but stopwords (articles,
  conjunctions, prepositions across a small closed en/fr/de vocabulary) and
  company-type tokens are skipped first, so a filler word doesn't break the
  acronym (`"National Aeronautics and Space Administration"` -> `"NASA"`).
  Only reported as a distinct candidate when it actually differs from
  `exact_initials`' own result (computed under the same `decompound_fn`), so
  the two layers' measured yields don't double-count the same names.
- `prefix_truncation`: trailing numeral and company-type tokens are dropped
  until the last remaining token is neither (`"Deutsche Bahn AG"` ->
  `"Deutsche Bahn"`, `"PSV Meiningen 90"` -> `"PSV Meiningen"`). Takes no
  `decompound_fn`: it never reduces a token to an initial in the first
  place, so there is nothing for one to change.
- `type_prefix_acronym_qualifier`: the leading token is a hyphenated (or,
  with an injected `decompound_fn`, an agglutinated) compound; each of its
  parts contributes an initial, and the remaining tokens (trailing
  company-type tokens dropped) are kept verbatim as a qualifier
  (`"Schwimm-Startgemeinschaft Leipzig"` -> `"SSG Leipzig"`, once
  `decompound_fn` further splits `"Startgemeinschaft"` into `"Start"` +
  `"Gemeinschaft"`; without a `decompound_fn` the same name yields `"SS
  Leipzig"` from the hyphen split alone).

`decompound_fn`, when given (directly to a layer, or via
`derive_short_name_candidate`, which forwards it to every layer that takes
one), is a plain `str -> Sequence[str]` callable; this module carries no
compounding logic or language-specific word lists of its own; a caller
supplies one (e.g. a German Koehn & Knight (2003)-style splitter) only where
measurement has shown it earns its keep.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .extract import _SHORT_NAME_COMPANY_TYPE_COMPONENT_TOKENS
from .normalize import _normalize_company_type_value

DecompoundFn = Callable[[str], Sequence[str]]

# Closed, small, hand-curated: articles/conjunctions/prepositions in the
# three languages this repo's real name data covers today. Deliberately not
# sourced from a general stopword list -- skipping too eagerly turns
# `subsequence_initials` into a guess rather than a deterministic pattern.
_STOPWORDS = frozenset(
    {
        # English
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        # French
        "le",
        "la",
        "les",
        "l",
        "de",
        "des",
        "du",
        "et",
        "ou",
        "pour",
        "dans",
        "en",
        # German
        "der",
        "die",
        "das",
        "und",
        "oder",
        "fur",
        "von",
        "zu",
        "im",
        "am",
        "zur",
        "zum",
    }
)

_RE_NUMERIC_TOKEN = re.compile(r"^\d+[a-z]?$", re.IGNORECASE)
_RE_TOKEN_SPLIT = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ShortNameCandidate:
    """One layer's derived candidate, tagged with the layer that produced it.

    Attributes:
        value: The derived short-name/significant-token candidate.
        pattern: The layer name that produced it (a key of `LAYERS`).
    """

    value: str
    pattern: str


def tokenize_name(name: str | None) -> list[str]:
    """Split a raw name into whitespace-delimited tokens, dropping empties."""
    if not name:
        return []
    return [token for token in _RE_TOKEN_SPLIT.split(name.strip()) if token]


def _normalized_token(token: str) -> str:
    return _normalize_company_type_value(token, transliterate=True)


def _is_company_type_token(token: str) -> bool:
    return _normalized_token(token) in _SHORT_NAME_COMPANY_TYPE_COMPONENT_TOKENS


def _is_stopword(token: str) -> bool:
    return token.strip(".,'’").lower() in _STOPWORDS


def _token_initial_letters(
    token: str, decompound_fn: DecompoundFn | None
) -> str | None:
    """One token's contribution to an initials string: one letter, or one per sub-part.

    Without `decompound_fn`, a token contributes its own first alphanumeric
    character. With one, the token is offered to it first (e.g. a German
    agglutinated compound like `"Skiverband"` splitting into `"Ski"` +
    `"Verband"`), and each resulting part contributes its own initial --
    `None` propagates if any part has no alphanumeric character at all.
    """
    parts = list(decompound_fn(token)) if decompound_fn is not None else [token]
    if not parts:
        parts = [token]
    letters: list[str] = []
    for part in parts:
        candidate_char = next((char for char in part if char.isalnum()), None)
        if candidate_char is None:
            return None
        letters.append(candidate_char.upper())
    return "".join(letters)


def _initials(
    tokens: Sequence[str], decompound_fn: DecompoundFn | None = None
) -> str | None:
    """Every token's initial letter(s), uppercased -- `None` if any token contributes none."""
    letters: list[str] = []
    for token in tokens:
        token_letters = _token_initial_letters(token, decompound_fn)
        if token_letters is None:
            return None
        letters.append(token_letters)
    return "".join(letters)


def pattern_exact_initials(
    tokens: Sequence[str], *, decompound_fn: DecompoundFn | None = None
) -> str | None:
    """Concatenate every token's initial, in order. Requires at least 2 tokens.

    With `decompound_fn`, a compound token contributes one initial per
    sub-part it splits into, rather than just its own first letter.
    """
    if len(tokens) < 2:
        return None
    return _initials(tokens, decompound_fn)


def pattern_subsequence_initials(
    tokens: Sequence[str], *, decompound_fn: DecompoundFn | None = None
) -> str | None:
    """Like `pattern_exact_initials`, skipping stopword/company-type tokens first."""
    content_tokens = [
        token
        for token in tokens
        if not _is_stopword(token) and not _is_company_type_token(token)
    ]
    if len(content_tokens) < 2:
        return None
    candidate = _initials(content_tokens, decompound_fn)
    if candidate is None:
        return None
    # Only a distinct signal when it actually differs from running the
    # unfiltered token list through the same computation.
    if candidate == _initials(tokens, decompound_fn):
        return None
    return candidate


def pattern_prefix_truncation(tokens: Sequence[str]) -> str | None:
    """Drop trailing numeral/company-type tokens; keep the rest verbatim."""
    end = len(tokens)
    while end > 0:
        token = tokens[end - 1]
        stripped = token.strip(".,")
        if _RE_NUMERIC_TOKEN.match(stripped) or _is_company_type_token(token):
            end -= 1
            continue
        break
    if end == 0 or end == len(tokens):
        return None
    return " ".join(tokens[:end])


def pattern_type_prefix_acronym_qualifier(
    tokens: Sequence[str],
    *,
    decompound_fn: DecompoundFn | None = None,
) -> str | None:
    """Acronym-of-compound-prefix + retained qualifier tokens.

    The leading token is split on hyphens; if `decompound_fn` is given, each
    hyphen-part is additionally offered to it for further (e.g. agglutinated
    German compound) splitting. Each resulting part contributes one initial.
    The remaining tokens (trailing company-type tokens dropped) are kept
    verbatim as the qualifier.
    """
    if len(tokens) < 2:
        return None

    head, *rest = tokens
    parts = [part for part in head.split("-") if part]
    if decompound_fn is not None:
        expanded: list[str] = []
        for part in parts:
            sub_parts = list(decompound_fn(part) or ())
            expanded.extend(sub_parts if sub_parts else [part])
        parts = expanded
    if len(parts) < 2:
        return None

    acronym = _initials(parts)
    if acronym is None:
        return None

    qualifier_end = len(rest)
    while qualifier_end > 0 and _is_company_type_token(rest[qualifier_end - 1]):
        qualifier_end -= 1
    qualifier = rest[:qualifier_end]
    if not qualifier:
        return acronym
    return f"{acronym} {' '.join(qualifier)}"


# Registry key -> layer callable. Every layer takes the tokenized name and
# an optional `decompound_fn` keyword (ignored by `prefix_truncation`, the
# only layer that has no use for it), so `derive_short_name_candidate()` can
# invoke any of them uniformly.
LAYERS: dict[str, Callable[..., str | None]] = {
    "exact_initials": (
        lambda tokens, decompound_fn=None, **_kwargs: pattern_exact_initials(
            tokens, decompound_fn=decompound_fn
        )
    ),
    "subsequence_initials": (
        lambda tokens, decompound_fn=None, **_kwargs: pattern_subsequence_initials(
            tokens, decompound_fn=decompound_fn
        )
    ),
    "prefix_truncation": lambda tokens, **_kwargs: pattern_prefix_truncation(tokens),
    "type_prefix_acronym_qualifier": (
        lambda tokens, decompound_fn=None, **_kwargs: (
            pattern_type_prefix_acronym_qualifier(tokens, decompound_fn=decompound_fn)
        )
    ),
}

# `subsequence_initials` only ever produces a candidate when it differs from
# `exact_initials`' own (see that function's docstring), so trying it first
# costs nothing and lets the more targeted pattern win when it applies,
# falling back to the blind whole-token-list initials otherwise.
# `exact_initials` fires for almost every 2+-token name (real-data
# measurement: 89-97% coverage per language, see
# `scripts/measure_short_name_pattern_layers.py`), so `prefix_truncation`
# and `type_prefix_acronym_qualifier` rarely get a turn in this combined
# order despite each having its own real, independently measured recovery
# rate -- that measurement is this module's real evidence, not the ordering
# below, which exists only as a single-candidate convenience default. A
# caller who wants every layer's own answer, not one combinator's pick,
# calls `LAYERS[name](tokens, ...)` directly per layer instead of going
# through `derive_short_name_candidate()`; a caller who wants a different
# priority (or a subset) passes its own `enabled_layers`.
DEFAULT_LAYER_ORDER: tuple[str, ...] = (
    "subsequence_initials",
    "exact_initials",
    "prefix_truncation",
    "type_prefix_acronym_qualifier",
)


def derive_short_name_candidate(
    name: str | None,
    *,
    enabled_layers: Sequence[str] = DEFAULT_LAYER_ORDER,
    decompound_fn: DecompoundFn | None = None,
) -> ShortNameCandidate | None:
    """Run layers in `enabled_layers` order; return the first hit, or `None`.

    `None` is a first-class result ("no confident derivation"), not an
    error -- most real short names are genuine rebrands with no recoverable
    signal in the string, and this never falls back to a low-confidence
    guess to avoid returning it.
    """
    tokens = tokenize_name(name)
    if not tokens:
        return None

    for layer_name in enabled_layers:
        layer_fn = LAYERS.get(layer_name)
        if layer_fn is None:
            raise ValueError(f"Unknown short-name derivation layer {layer_name!r}.")
        result = layer_fn(tokens, decompound_fn=decompound_fn)
        if result:
            return ShortNameCandidate(value=result, pattern=layer_name)
    return None
