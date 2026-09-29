"""Low-salience family: drop a corpus-derived noise word.

TAXONOMY.md's corpus-derived token operations. Unconditionally safe by
construction: tokens are only ever removed, never replaced or reordered, so the
result's tokens are always a sub-sequence of the original's. It cannot manufacture
a false same-entity claim between two different companies the way the deferred
generic-descriptor *substitution* family could.

Noise words come from `company_cleanse.get_corpus_noise_words(country)` — promoted,
corpus-derived reference data, never a hand-guessed list. `country` is forwarded
unchanged and never mapped here, so one language's noise words can't be applied to
another's names by construction rather than by convention. An unmapped or absent
country falls back to that function's own cross-language `global` list.

The "at least one token survives" floor used to need its own cap on the edit count.
Here it is just a condition on the site finder: a name down to a single token
offers no sites, and because sites are recomputed after every applied change, the
floor holds however many occurrences a profile asks for.
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

from company_cleanse import get_corpus_noise_words

from ..families import Family
from ..sited_operator import Site, SitedOperator, sited_registry


def _noise_word_sites(name: str, country: str | None) -> Sequence[Site]:
    tokens = name.split(" ")
    if len(tokens) < 2:
        return ()

    noise_words = set(get_corpus_noise_words(country))
    sites: list[Site] = []
    position = 0
    for token in tokens:
        if token.lower() in noise_words:
            sites.append(Site(position, position + len(token)))
        position += len(token) + 1
    return tuple(sites)


def _drop_token(name: str, site: Site, rng: Random, country: str | None) -> str:
    """Remove the token and the one separator it leaves behind."""
    if site.end < len(name):
        return name[: site.start] + name[site.end + 1 :]
    return name[: max(site.start - 1, 0)]


sited_registry.register(
    SitedOperator(
        operator_id="low_salience.token_drop",
        family=Family.LOW_SALIENCE,
        sites=_noise_word_sites,
        change=_drop_token,
    )
)
