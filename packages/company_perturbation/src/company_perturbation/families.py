from enum import StrEnum


class Family(StrEnum):
    """Operator families in scope for the v1 engine.

    Sourced from TAXONOMY.md's in-scope list. Deferred families (abbreviation/nickname
    substitution) deliberately have no slug here -- there is nothing to register an operator
    under until TAXONOMY.md moves them into scope.

    TRANSLITERATION is a narrower exception: TAXONOMY.md family 3 (script-variant
    transliteration) is still deferred as a whole -- no real non-Latin-script company *name*
    corpus exists in this repo to validate full name rendering against -- but
    homoglyph/confusable-character substitution is carved out as separately in scope, since it
    doesn't need a name corpus to validate: its correctness comes from character-shape identity
    (`company_cleanse.get_ascii_homoglyphs()`'s already-tested fold tables), not from matching
    real foreign-language names. See TAXONOMY.md family 3.

    LOW_SALIENCE is the other narrowed exception: TAXONOMY.md's "Corpus-derived token
    operations" section names two related families sharing one data source, but only the drop
    half is in scope -- the substitution half is still deferred, since
    swapping in a *different* low-salience word can manufacture a string that never existed,
    while dropping can only ever produce a substring of the input. See TAXONOMY.md's
    "Corpus-derived token operations" section.
    """

    WORD_ORDER = "word_order"
    TYPO = "typo"
    DIACRITIC_PUNCT = "diacritic_punct"
    LEGAL_SUFFIX = "legal_suffix"
    PHONETIC = "phonetic"
    TRANSLITERATION = "transliteration"
    LOW_SALIENCE = "low_salience"
