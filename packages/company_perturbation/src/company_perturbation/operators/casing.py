"""Matching and replacing without caring how a source capitalises its names.

Every lookup table in this package is lowercase -- the keyboard adjacency map, the
diacritic and homoglyph tables, the phonetic substitution pairs, `company_cleanse`'s
company-type variants. Real source data is not: Ireland's register holds
`MAGUIRE GROUP LIMITED`, Companies House holds mixed case, and the materializer
perturbs the *raw* name deliberately, so that the cleanse pass afterwards is
measuring what it recovers.

Matching a lowercase table against raw names therefore silently does nothing. On a
real 817,761-row Irish run it made `typo.keyboard_substitution` fire on 378 rows
instead of roughly 64% of them, and made `legal_suffix.variant_substitution` fall
back to dropping every single time, because `"LIMITED"` was not a key. Both looked
like working operators producing quiet output rather than like a defect.

So: fold for the lookup, and put the original's capitalisation back on whatever the
table returned.
"""

from __future__ import annotations


def match_case(original: str, replacement: str) -> str:
    """Return `replacement` capitalised the way `original` was.

    All upper, all lower and Title each carry over. Anything else -- mixed case,
    digits, punctuation -- is left as the table gave it, since there is no obvious
    pattern to copy and guessing one would be worse than not trying.
    """
    if not original or not replacement:
        return replacement
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    if original.istitle():
        return replacement.title()
    return replacement
