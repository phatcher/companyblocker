# company_perturbation Operator Taxonomy

Version: 2.1.0

## Purpose

This document catalogs the name-mutation operator families this package addresses: what each one is, why it exists, and what evidence supports it. It is the reference to know which noise categories are covered and on what basis.

## Evaluation criteria

A family earns a place here by being evaluated against four standing criteria, not by being convenient to build:

1. **Data availability.** Can the operator be built and tested against real reference data already in this repo, or would it ship validated only against hand-written fixtures with no real-corpus backing?
2. **Relevance.** Every system this repo processes is Latin-script company register data. A family whose value depends on data this repo doesn't have is lower priority than one addressing noise already observed in these systems.
3. **Dependency cost.** Standard-library-only code is preferred over reusing a resource already vendored in this repo, which is preferred over a new third-party library, which is preferred over a new library plus new reference-data acquisition.
4. **Determinism compatibility.** An operator's change must be byte-identical given the name, the site it acts at, and the generator state it is handed. It draws from that generator only for choices internal to the change, such as which adjacent key to substitute, and it draws a fixed number of values, since every step after it in a chain continues the same stream. A family that can only be implemented by calling a third-party library with its own internal, non-obviously-seedable RNG is a bigger risk to that contract than one built on code this package owns end to end.

## Families

### `word_order`: Implemented

Permutes space-delimited words within a name. Purely algorithmic, no reference data needed: the cheapest family to build first, which is why it proved out the operator interface and determinism contract before the others.

### `typo`: Implemented

Keyboard-adjacent substitution, adjacent-character transposition, single-character insertion and deletion, and character duplication (a later addition covering the fat-finger double-press case that omission alone didn't). This is the most common real-world data-entry noise category. Built on a small, hand-authored QWERTY adjacency map rather than a third-party typo-generator package: the determinism contract needs exact control over every random draw, which a dependency's internal RNG can't guarantee without an audit.

### `diacritic_punct`: Implemented

Two mechanisms: diacritic substitution (plain letters replaced with an accented variant, e.g. "e" to "é") and separator substitution (space, hyphen, and ampersand swapped at existing separator positions, never inserted where none exists, and never applied to apostrophes, which are a word-internal possessive/contraction pattern rather than inter-word separator noise). Directly relevant to data already in this repo: `fr` names carry accented Latin characters, and ampersand/punctuation normalization is an active concern in `company_cleanse`. The diacritic table deliberately excludes ç, ö, and ü even though they'd otherwise qualify, since those are already reachable through `transliteration`'s homoglyph substitution and covering them twice would double-count the same characters across two families; an automated test enforces the exclusion rather than relying on this note.

### `legal_suffix`: Implemented

Drops or varies legal-entity-form suffixes (Ltd, GmbH, SARL, Inc, and similar). Built on real, already-provenanced reference data: `company_cleanse`'s ISO 20275 entity-legal-form table, 3,600+ forms across 200+ jurisdictions. Variant substitution needs a jurisdiction signal to stay safe: an early design that pooled all jurisdictions' data for substitution let a UK "Ltd" become Australia's ISO 20275 classification label, and scoping to one country alone still wasn't enough, since a single country's own data mixes unrelated regulatory tracks (GB's Companies House types alongside Charity Commission types, letting "Ltd" become "CIO"). The shipped design instead draws from same-canonical spelling variants (always safe within a country) plus a small, explicitly curated table of real-world-evidenced confusable pairs per country. This is the one place in the package where the substitution pairing itself is curated rather than data-derived, since `company_cleanse`'s data expresses the variants but not which pairs are genuinely confusable.

### `phonetic`: Implemented

Replaces a word with a phonetically-equivalent respelling (e.g. "smith" to "smyth"), distinct from keystroke noise. Every candidate is drawn from a small, hand-authored table of spelling alternations but only kept if it shares the original word's Soundex code, verified against `jellyfish` rather than trusted on the table's own say-so. Deliberately excludes simple letter-doubling, since `typo`'s duplication and deletion operators already produce that noise.

### `transliteration`

**Homoglyph substitution: Implemented.** Swaps an ASCII letter for a visually similar non-Latin character (Cyrillic а for Latin a, and similar). Correctness comes from character-shape identity, verified against `company_cleanse`'s existing fold tables, so it doesn't need a real non-Latin company-name corpus to validate against.

**Full script-variant name rendering: Awaiting evidence.** No system this repo processes has non-Latin-script company names, so there's no real data to validate full rendering against; building it now would mean shipping it tested only against invented fixtures. Revisit if a system with real non-Latin-script names enters this repo (the in-progress Wikidata acquisition work is a plausible future source).

### Word-level substitution: Awaiting evidence

Two related, distinct families, tracked separately because they have different evidence gaps:

**Company-word abbreviation** (e.g. "Holdings" to "Hdgs", "International" to "Intl", outside the legal-suffix position). Real value for this domain, since it targets ordinary business words inside a name rather than the trailing legal form, but there's no harvested pairing data yet, and it shouldn't be built from a hand-guessed table, which would be indistinguishable from noise without a real source to check it against. Wikidata's name-variant sidecar rows (official/short-name variants, labels, and aliases) are a plausible source; whether they already contain usable abbreviation pairs hasn't been checked.

**Person-name nickname substitution** (e.g. "Robert" to "Bob"). No reference data exists anywhere in this repo, and every system here is company legal-entity data, not person names, so there's no concrete downstream need identified either. Lower priority than abbreviation; revisit if the domain expands to include person-name matching.

### Corpus-derived token operations

Two related families sharing the same data source (`company_tokenize`'s TF-IDF-derived noise-word data, promoted into `company_cleanse`) but different risk profiles.

**Generic low-salience-word drop: Implemented.** Drops one or more low-salience words from a name (e.g. "Cisco Systems Inc" to "Cisco"), drawn from `company_cleanse`'s corpus-derived noise-word data and scoped to the input's own jurisdiction, so one language's noise words are never applied to another's names. Unconditionally safe: dropping words can only ever produce a substring of the input, so it can never assert a false equivalence between two entities.

**Generic-descriptor substitution: Awaiting evidence.** Replaces a low-salience word with a different word from the same corpus-derived cluster (e.g. "Acme Holdings" to "Acme Systems"). Not simply the drop family with an extra step: substituting produces a string that never existed, and two words both being individually uninformative doesn't mean they're interchangeable descriptions of the same entity. Needs real evidence the substitution actually occurs for the same entity across sources, which the abbreviation family's own Wikidata finding doesn't transfer to, since that evidence was about abbreviation specifically.

### `ocr`: Planned

Same-script optical-character-recognition confusion (e.g. "0"/"O", "1"/"l"/"I", "rn"/"m", "5"/"S", "8"/"B"), distinct from `transliteration`'s cross-script homoglyphs and from `typo`'s keyboard-entry mechanics: a different real-world corruption source with its own confusion profile. Like homoglyph substitution, correctness comes from glyph-shape identity rather than needing a real corpus to validate against, so there's no evidence gap blocking this one, only the work of building it. Scan quality maps onto a chain step's own two numbers, a more degraded scan being a higher `probability` and a larger `attempts`, so no interface change is needed. Single-character confusions fit the same site-per-character mechanism every other table-driven operator in this package already uses; multi-character confusions (e.g. "rn" read as "m") need a variable-length substitution mechanism this package doesn't have yet, and can be scoped as a later extension once the single-character case is built.
