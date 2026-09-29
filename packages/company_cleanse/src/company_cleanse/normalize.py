"""String normalization, driven by a `|`-separated profile.

A profile starts from `default` and is modified by three sigils, shared with the short-name
profile:

- `-x` skips a stage that is on by default (`default|-lowercase`).
- `x` selects an operation that is off by default (`default|transliterate`).
- `+x` adds a tier to an operation the profile already selected
  (`default|geographic_terms|+region`); a tier with no `geographic_terms` in the profile raises,
  and `geographic` is shorthand for `geographic_terms`.

`casefold` shares `lowercase`'s slot, so a profile selects at most one of them
(`default|-lowercase|casefold`), and it must run after `diacritics`, since caseless matching is
`NFD(casefold(NFD(x)))`. `casefold` can lengthen a string (`ß` to `ss`). The default keeps
`lowercase`: batch cleansing forces `transliterate` ahead of the case step, so the text is ASCII
by then and the two agree, which held on every `offeneregister` and `gleif` name carrying a
case-fold-divergent character. They differ for direct callers of `normalize_tokens` and
`strip_company_suffix`, which get no forced transliteration.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from functools import lru_cache

from .geographic_terms import (
    DEFAULT_GEOGRAPHIC_TERM_KINDS,
    GEOGRAPHIC_TERM_KIND_COUNTRY,
    resolve_geographic_term_kinds,
    strip_geographic_terms,
)

# Core connector vocabulary used globally. system can override.
DEFAULT_AND_TOKENS: tuple[str, ...] = ("and", "et", "und")

NORMALIZATION_OPERATION_PUNCTUATION = "punctuation"
NORMALIZATION_OPERATION_PUNCTUATION_BANG = "punctuation_bang"
NORMALIZATION_OPERATION_SINGLESPACE = "singlespace"
NORMALIZATION_OPERATION_AND = "and"
NORMALIZATION_OPERATION_DIACRITICS = "diacritics"
NORMALIZATION_OPERATION_TRANSLITERATE = "transliterate"
NORMALIZATION_OPERATION_LOWERCASE = "lowercase"
NORMALIZATION_OPERATION_CASEFOLD = "casefold"
NORMALIZATION_OPERATION_SINGLECHAR = "singlechar"
NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS = "geographic_terms"

# `geographic` is accepted in a profile as a shorthand for `geographic_terms`, so
# `default|geographic|+region` and `default|geographic_terms|+region` are the same
# chain. The resolved operation tuple always carries the canonical spelling.
GEOGRAPHIC_TERMS_PROFILE_ALIAS = "geographic"

# `+x` adds an optional tier to an operation already selected. It is the third of the
# three profile sigils, alongside `-x` (skip a default-on stage) and a bare name
# (select an operation) -- see `parse_normalization_profile()`.
GEOGRAPHIC_TIER_SIGIL = "+"

DEFAULT_NORMALIZATION_PROFILE = "default"
DEFAULT_NORMALIZATION_OPERATIONS: tuple[str, ...] = (
    NORMALIZATION_OPERATION_PUNCTUATION,
    NORMALIZATION_OPERATION_SINGLESPACE,
    NORMALIZATION_OPERATION_AND,
    NORMALIZATION_OPERATION_SINGLECHAR,
    NORMALIZATION_OPERATION_DIACRITICS,
    NORMALIZATION_OPERATION_LOWERCASE,
)
SUPPORTED_NORMALIZATION_OPERATIONS: frozenset[str] = frozenset(
    {
        NORMALIZATION_OPERATION_PUNCTUATION,
        NORMALIZATION_OPERATION_PUNCTUATION_BANG,
        NORMALIZATION_OPERATION_SINGLESPACE,
        NORMALIZATION_OPERATION_AND,
        NORMALIZATION_OPERATION_DIACRITICS,
        NORMALIZATION_OPERATION_TRANSLITERATE,
        NORMALIZATION_OPERATION_LOWERCASE,
        NORMALIZATION_OPERATION_SINGLECHAR,
        # Registered but deliberately absent from DEFAULT_NORMALIZATION_OPERATIONS:
        # selectable per profile and off unless a profile names it, the way
        # `punctuation_bang` sits beside `punctuation`.
        NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS,
        # Also deliberately absent from the default chain: `str.casefold()` is the
        # operation Unicode defines for caseless *matching* (it unifies e.g. German
        # sharp s and final sigma the way `str.lower()` does not). Measured against
        # every real `offeneregister` (149,762 rows) and `gleif` (11,893 rows) name
        # containing a case-fold-divergent character (`ß`/`ẞ`/`ς`/`Σ`) when the caseless fold was added:
        # `name_cleansed` from `cleanse_lazyframe()` differs between `lowercase` and
        # `casefold` for 0 of those 161,655 rows, because `_resolve_runtime_normalization`
        # unconditionally forces `transliterate` ahead of the case-fold step for that
        # pipeline regardless of profile (see `with_transliteration_operation`), and
        # transliteration already folds `ß`->`ss`/`ς`->`s` to plain ASCII before either
        # case operation ever runs -- `str.lower()` and `str.casefold()` are identical
        # on ASCII input. `lowercase` staying the default is therefore not an
        # unexamined inheritance: there is no real-data case in the pipeline path for
        # disrupting it. The operation is real and does diverge for direct
        # `normalize_tokens()`/`strip_company_suffix()` callers who build a chain
        # without `transliterate` (`"Straße"` -> `"straße"` under `lowercase` vs.
        # `"strasse"` under `casefold`), which is exactly why it stays selectable.
        NORMALIZATION_OPERATION_CASEFOLD,
    }
)

_PUNCTUATION_OPERATION_FAMILY: frozenset[str] = frozenset(
    {
        NORMALIZATION_OPERATION_PUNCTUATION,
        NORMALIZATION_OPERATION_PUNCTUATION_BANG,
    }
)

# `lowercase` and `casefold` are mutually exclusive alternatives for the same slot in
# the chain (one caller-selected case operation, not both), the way `punctuation` and
# `punctuation_bang` are. Also the answer to "does this chain already fold case", used
# wherever a consumer needs to know that regardless of which of the two was selected
# (see `resolve_char_whitelist_for_operations`, `with_transliteration_operation`, and
# the polars step engine's `generate_cleansed_company_name`).
CASE_FOLDING_OPERATIONS: frozenset[str] = frozenset(
    {
        NORMALIZATION_OPERATION_LOWERCASE,
        NORMALIZATION_OPERATION_CASEFOLD,
    }
)

_DEFAULT_OPERATION_INSERTION_INDEX: dict[str, int] = {
    # Ahead of everything else, because `punctuation` turns `Siemens (UK)` into
    # `Siemens  UK ` and the bracketed arm needs to see the brackets. The cost of
    # running first is the ordering constraint documented on
    # `strip_geographic_terms()`: no legal-form stripping has happened yet on this
    # path, so a name still carrying `Ltd` is reached by the bracketed arm only.
    NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS: -1,
    NORMALIZATION_OPERATION_PUNCTUATION: 0,
    NORMALIZATION_OPERATION_PUNCTUATION_BANG: 0,
    NORMALIZATION_OPERATION_SINGLESPACE: 1,
    NORMALIZATION_OPERATION_AND: 2,
    NORMALIZATION_OPERATION_SINGLECHAR: 3,
    NORMALIZATION_OPERATION_DIACRITICS: 4,
    NORMALIZATION_OPERATION_TRANSLITERATE: 4,
    NORMALIZATION_OPERATION_LOWERCASE: 5,
    # Shares `lowercase`'s slot -- the two are alternatives, not sequential stages;
    # see `CASE_FOLDING_OPERATIONS`. Unicode's caseless-match recipe is
    # `NFD(casefold(NFD(x)))`: casefolding is not closed under normalization, so this
    # chain runs `casefold` *after* `diacritics` (which already applies an NFKD pass),
    # not before -- placing it earlier is not equivalent.
    NORMALIZATION_OPERATION_CASEFOLD: 5,
}

NormalizationOperationHandler = Callable[[str, re.Pattern[str] | None], str]

COMPANY_TYPE_NORMALIZATION_OPERATIONS: tuple[str, ...] = (
    NORMALIZATION_OPERATION_PUNCTUATION_BANG,
    NORMALIZATION_OPERATION_SINGLESPACE,
    NORMALIZATION_OPERATION_AND,
    NORMALIZATION_OPERATION_SINGLECHAR,
    NORMALIZATION_OPERATION_DIACRITICS,
    NORMALIZATION_OPERATION_LOWERCASE,
)

COMPANY_TYPE_TRANSLITERATION_OPERATIONS: tuple[str, ...] = (
    NORMALIZATION_OPERATION_PUNCTUATION_BANG,
    NORMALIZATION_OPERATION_SINGLESPACE,
    NORMALIZATION_OPERATION_AND,
    NORMALIZATION_OPERATION_SINGLECHAR,
    NORMALIZATION_OPERATION_TRANSLITERATE,
    NORMALIZATION_OPERATION_LOWERCASE,
)

# Modern Greek only (the 24-letter alphabet plus final sigma). Deliberately excludes
# archaic/historical letters (digamma, koppa, sampi, heta, san, sho, yot) and the Coptic
# letters that share the Greek Unicode block (U+03E2-03EF etc.) -- none of these appear in
# any company-register data this repo processes; adding them would be speculative coverage
# with no real corpus to validate against. Accented modern Greek (tonos/dialytika) needs no
# entry here: those characters have canonical NFKD decompositions, so `strip_diacritics`
# already reduces them to a base letter before this table is consulted.
GREEK_TO_LATIN = {
    "Α": "A",
    "Β": "B",
    "Γ": "G",
    "Δ": "D",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "I",
    "Θ": "TH",
    "Ι": "I",
    "Κ": "K",
    "Λ": "L",
    "Μ": "M",
    "Ν": "N",
    "Ξ": "X",
    "Ο": "O",
    "Π": "P",
    "Ρ": "R",
    "Σ": "S",
    "Τ": "T",
    "Υ": "Y",
    "Φ": "F",
    "Χ": "CH",
    "Ψ": "PS",
    "Ω": "O",
    "α": "a",
    "β": "b",
    "γ": "g",
    "δ": "d",
    "ε": "e",
    "ζ": "z",
    "η": "i",
    "θ": "th",
    "ι": "i",
    "κ": "k",
    "λ": "l",
    "μ": "m",
    "ν": "n",
    "ξ": "x",
    "ο": "o",
    "π": "p",
    "ρ": "r",
    "σ": "s",
    "ς": "s",
    "τ": "t",
    "υ": "y",
    "φ": "f",
    "χ": "ch",
    "ψ": "ps",
    "ω": "o",
}

# Covers the standard modern Russian alphabet plus the Serbian/Macedonian letters that
# extend the same Cyrillic block (both scripts are in scope: ISO 20275 carries `Russian
# Federation`, `Serbia`, and `North Macedonia` entries in
# resources/entity_legal_forms_iso20275.json). Deliberately excludes Ukrainian-specific
# letters (Ukrainian Е, І, Ї, Ґ) and Central Asian extensions (Kazakh/Bashkir/Abkhaz letters
# such as Ә, Ғ, Қ, Ң, Ө, Ұ, Һ) -- no Ukraine or Kazakhstan entries exist anywhere in that
# data today, so adding them would be coverage with nothing real to test against. Also
# excludes the archaic pre-1918 Russian orthography letters (Ѣ yat, Ѳ fita, Ѵ izhitsa) and
# other historical/liturgical Cyrillic (Ѡ omega, yus letters, etc.) for the same reason.
CYRILLIC_TO_LATIN = {
    "А": "A",
    "Б": "B",
    "В": "V",
    "Г": "G",
    "Д": "D",
    "Е": "E",
    "Ж": "ZH",
    "З": "Z",
    "И": "I",
    "Й": "Y",
    "К": "K",
    "Л": "L",
    "М": "M",
    "Н": "N",
    "О": "O",
    "П": "P",
    "Р": "R",
    "С": "S",
    "Т": "T",
    "У": "U",
    "Ф": "F",
    "Х": "H",
    "Ц": "TS",
    "Ч": "CH",
    "Ш": "SH",
    "Щ": "SHT",
    "Ъ": "A",
    "Ы": "Y",
    "Ь": "",
    "Э": "E",
    "Ю": "YU",
    "Я": "YA",
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sht",
    "ъ": "a",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
    # Serbian/Macedonian letters (same Cyrillic block, both jurisdictions present in
    # entity_legal_forms_iso20275.json). Mapped to match this table's existing digraph
    # style (zh/ch/sh/ts) rather than the accented-Latin forms Serbian's own Latin script
    # uses (Đ/Ć/Č/Š/Ž), since this table's output must stay plain ASCII throughout.
    "Ђ": "D",
    "Ѕ": "DZ",
    "Ј": "J",
    "Љ": "LJ",
    "Њ": "NJ",
    "Ћ": "C",
    "Џ": "DZH",
    "ђ": "d",
    "ѕ": "dz",
    "ј": "j",
    "љ": "lj",
    "њ": "nj",
    "ћ": "c",
    "џ": "dzh",
}

# Latin-alphabet letters that carry a stroke/bar or other non-decomposable mark, so
# `strip_diacritics`'s NFKD pass can't reduce them to a plain base letter on its own.
# Covers the jurisdictions/languages already present in entity_legal_forms_iso20275.json
# (Polish, Scandinavian, Turkish, Croatian, Maltese, German, French, Icelandic). Deliberately
# excludes the broader Latin Extended-B block (African/Skolt-Sami/click-language letters,
# e.g. Ƣ, Ʒ, Ǯ) -- none of those languages/jurisdictions appear in this repo's data.
LATIN_SPECIAL_TO_ASCII = {
    "ß": "ss",
    "Æ": "AE",
    "æ": "ae",
    "Œ": "OE",
    "œ": "oe",
    "Ø": "O",
    "ø": "o",
    "Ð": "D",
    "ð": "d",
    "Þ": "TH",
    "þ": "th",
    "Ł": "L",
    "ł": "l",
    "Đ": "D",
    "đ": "d",
    "İ": "I",
    "ı": "i",
    "Ş": "S",
    "ş": "s",
    "Ğ": "G",
    "ğ": "g",
    "Ç": "C",
    "ç": "c",
    "Ö": "O",
    "ö": "o",
    "Ü": "U",
    "ü": "u",
    "Ħ": "H",
    "ħ": "h",
}


# Fast-path probe for candidate runs like "S R L" or "P & S".
_SINGLE_CHAR_COLLAPSE_CANDIDATE_PATTERN = re.compile(
    r"(?<!\S)\w(?:\s+(?:&\s+)?\w)+(?!\S)"
)


def _is_single_alnum_token(token: str) -> bool:
    return len(token) == 1 and token.isalnum()


def _consume_single_char_run(tokens: list[str], start_index: int) -> tuple[str, int]:
    """Consume a collapsible run beginning at start_index and return (output, next_index)."""
    merged_parts = [tokens[start_index]]
    merged_alnum_count = 1
    index = start_index + 1

    while index < len(tokens):
        current = tokens[index]
        if _is_single_alnum_token(current):
            merged_parts.append(current)
            merged_alnum_count += 1
            index += 1
            continue

        if (
            current == "&"
            and index + 1 < len(tokens)
            and _is_single_alnum_token(tokens[index + 1])
        ):
            merged_parts.append("&")
            merged_parts.append(tokens[index + 1])
            merged_alnum_count += 1
            index += 2
            continue

        break

    if merged_alnum_count < 2:
        return tokens[start_index], start_index + 1

    return _reopen_ampersand_for_multichar_runs("".join(merged_parts)), index


def _reopen_ampersand_for_multichar_runs(word: str) -> str:
    """Reopen '&' spacing once a merged side is no longer a single character.

    Chaining plain single-character tokens (no '&' between them) inside a collapsed
    run can leave a merged fragment longer than one character sitting directly next
    to a later '&' join -- for example 'A T & T' collapses token-by-token into
    'AT&T', even though 'AT&T', 'AT & T', and 'A T & T' should all normalize the
    same way. Splitting the join back open whenever either side exceeds one
    character keeps '&' collapsing limited to genuine single-character initials
    (`P&S`, `H&M`) and avoids silently gluing a real abbreviation/word to '&'.
    """
    if "&" not in word:
        return word

    parts = word.split("&")
    pieces = [parts[0]]
    for index in range(1, len(parts)):
        left, right = parts[index - 1], parts[index]
        pieces.append(" & " if len(left) > 1 or len(right) > 1 else "&")
        pieces.append(parts[index])
    return "".join(pieces)


def strip_diacritics(text: str | None) -> str | None:
    """Remove diacritics so matching can be accent-insensitive."""
    if not text:
        return text

    # Hot-path: ASCII text has no diacritics to strip.
    if text.isascii():
        return text

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def transliterate_to_ascii(text: str | None) -> str | None:
    """Transliterate selected scripts/letters to ASCII."""
    if not text:
        return text

    # Hot-path: ASCII text needs no transliteration.
    if text.isascii():
        return text

    stripped = strip_diacritics(text) or ""
    if stripped.isascii():
        return stripped

    transliterated = "".join(GREEK_TO_LATIN.get(char, char) for char in stripped)
    transliterated = "".join(
        CYRILLIC_TO_LATIN.get(char, char) for char in transliterated
    )
    return "".join(LATIN_SPECIAL_TO_ASCII.get(char, char) for char in transliterated)


@lru_cache(maxsize=1)
def get_ascii_homoglyphs() -> dict[str, tuple[str, ...]]:
    """Inverse of the fold tables above: ASCII letter -> non-Latin characters that fold to it.

    `transliterate_to_ascii` is many-to-one (Greek/Cyrillic/Latin-special -> ASCII); this is the
    reverse direction, for callers that want to substitute an ASCII letter with something that
    visually resembles it (a homoglyph/confusable swap), not fold a whole non-Latin name down to
    ASCII. Built by inverting `GREEK_TO_LATIN`/`CYRILLIC_TO_LATIN`/`LATIN_SPECIAL_TO_ASCII` rather
    than maintained as a separate table, so it can never drift from the fold tables' own
    already-tested correspondences. Only single-character folds are included (digraphs like
    `Ж -> "ZH"` and empty-string folds like `Ь -> ""` are skipped): a homoglyph swap replaces one
    character with one character, so a digraph target has no single source character to report
    here even though it's a valid *fold* target.

    Each candidate's target is computed by running it through `transliterate_to_ascii` itself,
    not by reading the raw dict value it happens to carry -- some entries (e.g.
    `CYRILLIC_TO_LATIN["Й"] == "Y"`) are shadowed at runtime because `strip_diacritics`'s NFKD
    pass decomposes that character (Й -> И + combining breve) before the dict lookup ever runs,
    so the character actually folds to whatever `strip_diacritics` reduces it to ("I", via `И`),
    not its own raw dict entry. Deriving from the real pipeline output keeps this table
    self-consistent with `transliterate_to_ascii`'s actual behavior by construction, rather than
    silently drifting from it if a table entry is ever shadowed the same way.

    Iteration order over the source dicts is insertion order (fixed by the literal dict in source
    code, not hash-dependent), and the result is sorted before returning, so this is reproducible
    across runs/machines/processes without relying on Python's hash randomization -- consumers
    doing seeded random selection over these candidates (company_perturbation's determinism
    contract) need that guarantee.
    """
    homoglyphs: dict[str, list[str]] = {}
    for table in (GREEK_TO_LATIN, CYRILLIC_TO_LATIN, LATIN_SPECIAL_TO_ASCII):
        for source_char in table:
            ascii_target = transliterate_to_ascii(source_char)
            if not ascii_target or len(ascii_target) != 1:
                continue
            homoglyphs.setdefault(ascii_target, []).append(source_char)

    return {
        ascii_char: tuple(sorted(source_chars))
        for ascii_char, source_chars in sorted(homoglyphs.items())
    }


def collapse_single_char_sequences(text: str | None) -> str | None:
    """Collapse single-character token runs, including '&' joins: 'S R L' -> 'SRL', 'P & S' -> 'P&S'.

    A merged run stays glued to '&' only while both sides remain single characters;
    once chaining leaves a longer fragment next to '&' (e.g. 'A T & T' -> 'AT & T'),
    the space is reopened so `&`-joined abbreviations normalize the same way
    regardless of how the input happened to be spaced.

    This function owns '&' spacing end to end, so it is safe to call standalone
    (not only after the `and` normalization operation): a tightly glued '&' such as
    'AT&T' is widened to 'AT & T' before collapsing runs, so 'AT&T', 'AT & T', and
    'A T & T' all resolve to the same 'AT & T', matching what the full `normalize_tokens`
    pipeline (`and` then `singlechar`) produces.
    """
    if not text:
        return text

    if "&" in text and any(char.isalnum() for char in text):
        text = re.sub(r"\s*&\s*", " & ", text)

    if _SINGLE_CHAR_COLLAPSE_CANDIDATE_PATTERN.search(text) is None:
        return text

    token_matches = list(re.finditer(r"\S+", text))
    if not token_matches:
        return text

    tokens = [match.group(0) for match in token_matches]
    separators = [
        text[token_matches[index].end() : token_matches[index + 1].start()]
        for index in range(len(token_matches) - 1)
    ]
    leading = text[: token_matches[0].start()]
    trailing = text[token_matches[-1].end() :]

    out: list[str] = [leading]
    i = 0
    while i < len(tokens):
        next_index = i + 1
        if _is_single_alnum_token(tokens[i]):
            collapsed, next_index = _consume_single_char_run(tokens, i)
            out.append(collapsed)
        else:
            out.append(tokens[i])
        if next_index < len(tokens):
            out.append(separators[next_index - 1])
        else:
            out.append(trailing)
        i = next_index

    return "".join(out)


def operation_base_name(operation: str) -> str:
    """Return an operation's name without any `+tier` suffix it carries.

    Only `geographic_terms` is tiered today, so for every other operation this is the
    identity. Resolved chains carry the tiers inside the operation string
    (`geographic_terms+region`) rather than as a separate argument, because the chain
    tuple is also the normalization cache key -- two profiles differing only by tier
    have to be two different keys.
    """
    return operation.split(GEOGRAPHIC_TIER_SIGIL, 1)[0]


def geographic_tiers_for_operation(operation: str) -> tuple[str, ...]:
    """Return the tiers a resolved `geographic_terms` operation selects."""
    parts = operation.split(GEOGRAPHIC_TIER_SIGIL)
    return resolve_geographic_term_kinds(tuple(parts[1:]))


def _compose_geographic_operation(tiers: tuple[str, ...]) -> str:
    suffix = "".join(
        f"{GEOGRAPHIC_TIER_SIGIL}{tier}"
        for tier in tiers
        if tier != GEOGRAPHIC_TERM_KIND_COUNTRY
    )
    return f"{NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS}{suffix}"


def _canonicalize_operation_name(operation: str) -> str:
    if operation == GEOGRAPHIC_TERMS_PROFILE_ALIAS:
        return NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS
    return operation


def _canonicalize_operation(operation: str) -> str:
    """Resolve an operation's alias spelling and tier suffix to canonical form."""
    parts = operation.split(GEOGRAPHIC_TIER_SIGIL)
    base = _canonicalize_operation_name(parts[0])
    if len(parts) == 1:
        return base

    if base != NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS:
        raise ValueError(
            f"Operation '{operation}' uses a '{GEOGRAPHIC_TIER_SIGIL}' tier modifier, "
            f"which only '{NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS}' supports."
        )

    return _compose_geographic_operation(
        resolve_geographic_term_kinds(tuple(parts[1:]))
    )


def _insert_operation_using_default_order(
    operations: list[str],
    operation: str,
) -> None:
    target_index = _DEFAULT_OPERATION_INSERTION_INDEX.get(
        operation_base_name(operation), len(operations)
    )
    insert_at = len(operations)
    last_antecedent_index = -1
    for index, existing in enumerate(operations):
        existing_index = _DEFAULT_OPERATION_INSERTION_INDEX.get(
            operation_base_name(existing), len(operations)
        )
        if existing_index <= target_index:
            last_antecedent_index = index
        if existing_index > target_index:
            insert_at = index
            break
    if last_antecedent_index >= 0:
        insert_at = max(last_antecedent_index + 1, min(insert_at, len(operations)))
    operations.insert(insert_at, operation)


def _validate_supported_operation(operation: str, raw_profile: str) -> None:
    if operation_base_name(operation) in SUPPORTED_NORMALIZATION_OPERATIONS:
        return

    available = ", ".join(sorted(SUPPORTED_NORMALIZATION_OPERATIONS))
    raise ValueError(
        f"Unknown normalization operation '{operation}' in profile '{raw_profile}'. "
        f"Available operations: {available}."
    )


# Groups of operations where selecting one member of the group displaces whichever
# other member is already present, rather than the two coexisting in the chain.
_MUTUALLY_EXCLUSIVE_OPERATION_FAMILIES: tuple[frozenset[str], ...] = (
    _PUNCTUATION_OPERATION_FAMILY,
    CASE_FOLDING_OPERATIONS,
)


def _replace_family_operation(operations: list[str], operation: str) -> bool:
    """Swap `operation` in for whichever family member is already present, if any.

    Mirrors how `default|punctuation_bang` replaces the default chain's `punctuation`
    without needing an explicit `-punctuation` first; `default|casefold` replaces
    `lowercase` the same way.
    """
    family = next(
        (
            candidate
            for candidate in _MUTUALLY_EXCLUSIVE_OPERATION_FAMILIES
            if operation in candidate
        ),
        None,
    )
    if family is None:
        return False

    for index, existing in enumerate(operations):
        if existing in family:
            operations[index] = operation
            return True

    return False


def _add_profile_operation(operations: list[str], operation: str) -> None:
    if operation in operations:
        return

    if _replace_family_operation(operations, operation):
        return

    _insert_operation_using_default_order(operations, operation)


def _parse_default_profile_operations(
    parts: list[str], raw_profile: str
) -> tuple[str, ...]:
    operations = list(DEFAULT_NORMALIZATION_OPERATIONS)
    for modifier in parts[1:]:
        remove = modifier.startswith("-")
        operation = _canonicalize_operation_name(modifier[1:] if remove else modifier)
        _validate_supported_operation(operation, raw_profile)
        if remove:
            operations = [existing for existing in operations if existing != operation]
            continue
        _add_profile_operation(operations, operation)
    return tuple(operations)


def _upsert_explicit_profile_operation(operations: list[str], operation: str) -> None:
    if operation in operations:
        return

    if _replace_family_operation(operations, operation):
        return

    if operation in _PUNCTUATION_OPERATION_FAMILY:
        operations.append(operation)
        return

    operations.append(operation)


def _parse_explicit_profile_operations(
    parts: list[str], raw_profile: str
) -> tuple[str, ...]:
    operations: list[str] = []
    for part in parts:
        if part.startswith("-"):
            raise ValueError(
                f"Profile '{raw_profile}' uses remove modifiers without '{DEFAULT_NORMALIZATION_PROFILE}'."
            )
        operation = _canonicalize_operation_name(part)
        _validate_supported_operation(operation, raw_profile)
        _upsert_explicit_profile_operation(operations, operation)
    return tuple(operations)


def _extract_geographic_tier_modifiers(
    parts: list[str], raw_profile: str
) -> tuple[list[str], tuple[str, ...]]:
    remaining: list[str] = []
    tiers: list[str] = []
    for part in parts:
        if not part.startswith(GEOGRAPHIC_TIER_SIGIL):
            remaining.append(part)
            continue

        tier = part[1:].strip()
        if not tier:
            raise ValueError(
                f"Profile '{raw_profile}' has an empty "
                f"'{GEOGRAPHIC_TIER_SIGIL}' tier modifier."
            )
        tiers.append(tier)
    return remaining, tuple(tiers)


def _apply_geographic_tier_modifiers(
    operations: tuple[str, ...],
    tiers: tuple[str, ...],
    raw_profile: str,
) -> tuple[str, ...]:
    """Fold `+tier` modifiers into the geographic operation the profile selected.

    A tier modifier with no `geographic_terms` in the profile is rejected rather than
    silently ignored, so a typo fails loudly instead of quietly doing nothing.
    """
    if not tiers:
        return operations

    resolved_tiers = resolve_geographic_term_kinds(tiers)
    updated = list(operations)
    for index, operation in enumerate(updated):
        if operation_base_name(operation) == NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS:
            updated[index] = _compose_geographic_operation(resolved_tiers)
            return tuple(updated)

    listed = ", ".join(f"{GEOGRAPHIC_TIER_SIGIL}{tier}" for tier in tiers)
    raise ValueError(
        f"Profile '{raw_profile}' uses tier modifier(s) {listed} without selecting "
        f"'{NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS}'."
    )


def normalize_tokens(
    text: str | None,
    and_tokens: tuple[str, ...] | list[str] | None = None,
    normalization_profile: str = DEFAULT_NORMALIZATION_PROFILE,
) -> str | None:
    """Run ordered normalization operations for legal-form token surfaces.

    The default profile currently expands to:
    ``punctuation|singlespace|and|singlechar|diacritics|lowercase``

    Operation semantics:
    - ``punctuation``: remove non-word punctuation while preserving ``&``
    - ``punctuation_bang``: remove non-word punctuation while preserving ``&`` and ``!``
    - ``singlespace``: collapse repeated whitespace and trim outer whitespace
    - ``and``: normalize configured connector tokens and ``&`` spacing
    - ``singlechar``: collapse single-character token runs (for example ``S R L`` -> ``SRL``);
      stays spaced around ``&`` once either side is more than one character, so
      ``AT&T``/``AT & T``/``A T & T`` all normalize to ``at & t``
    - ``diacritics``: remove Unicode diacritics for accent-insensitive matching
    - ``lowercase``: lowercase input text (``str.lower()``, defined for display)
    - ``casefold``: fold case for caseless *matching* (``str.casefold()``); diverges
      from ``lowercase`` on real register text -- see ``CASE_FOLDING_OPERATIONS`` and
      ``_run_operation_casefold`` for the German/Greek examples. Mutually exclusive
      with ``lowercase``: selecting one displaces the other, the way
      ``punctuation_bang`` displaces ``punctuation``. Shares ``lowercase``'s slot in
      the chain, which is why it must run after ``diacritics`` rather than before --
      Unicode's caseless-match recipe is ``NFD(casefold(NFD(x)))`` because casefolding
      is not closed under normalization.
    - ``transliterate``: transliterate supported scripts/letters to ASCII
    - ``geographic_terms``: strip a trailing or bracketed geographic term
      (``Siemens (UK) Ltd`` -> ``siemens ltd``). Off unless a profile names it, and
      then ``country`` is the tier it applies unless ``+region``/``+city`` add more.
      Runs at the head of the chain so the bracketed arm can still see its brackets;
      see ``strip_geographic_terms()`` for the ordering constraint that follows from
      that, and for the guards that keep ``Air France`` intact.
    """
    return _normalize_tokens_with_resolved_operations(
        text,
        and_tokens=and_tokens,
        operations=parse_normalization_profile(normalization_profile),
    )


def normalize_tokens_with_operations(
    text: str | None,
    *,
    operations: Sequence[str],
    and_tokens: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    """Run normalization using an already-resolved ordered operation sequence."""
    return _normalize_tokens_with_resolved_operations(
        text,
        and_tokens=and_tokens,
        operations=operations,
    )


def _normalize_tokens_with_resolved_operations(
    text: str | None,
    *,
    operations: Sequence[str],
    and_tokens: tuple[str, ...] | list[str] | None = None,
) -> str | None:
    return _normalize_with_cached_operations(
        text,
        and_tokens=and_tokens,
        operations=operations,
        normalize_cached=_normalize_tokens_cached,
    )


def _normalize_with_and_tokens_key(
    text: str | None,
    and_tokens: tuple[str, ...] | list[str] | None,
    *,
    normalize_with_key,
) -> str | None:
    if not text:
        return text

    and_tokens_key = _normalize_and_tokens_key(and_tokens)
    return normalize_with_key(text, and_tokens_key)


def _normalize_with_cached_operations(
    text: str | None,
    *,
    and_tokens: tuple[str, ...] | list[str] | None,
    operations: Sequence[str],
    normalize_cached,
) -> str | None:
    resolved_operations = normalize_operations(operations)
    return _normalize_with_and_tokens_key(
        text,
        and_tokens,
        normalize_with_key=lambda value, key: normalize_cached(
            value,
            key,
            resolved_operations,
        ),
    )


def _normalize_and_tokens_key(
    and_tokens: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...] | None:
    if and_tokens is None:
        return None
    return tuple(and_tokens)


@lru_cache(maxsize=512)
def _resolve_and_tokens(and_tokens_key: tuple[str, ...] | None) -> tuple[str, ...]:
    if and_tokens_key is not None:
        resolved_and_tokens = [
            token.strip().lower() for token in and_tokens_key if token and token.strip()
        ]
    else:
        resolved_and_tokens = list(DEFAULT_AND_TOKENS)

    # De-duplicate while preserving order.
    deduped_and_tokens: list[str] = []
    seen: set[str] = set()
    for token in resolved_and_tokens:
        if token not in seen:
            deduped_and_tokens.append(token)
            seen.add(token)
    return tuple(deduped_and_tokens)


def parse_normalization_profile(normalization_profile: str | None) -> tuple[str, ...]:
    """Resolve normalization profile syntax to an ordered operation tuple.

    Three sigils, one meaning each:

    - ``-x`` skips a stage that is on by default (``default|-lowercase``, and in the
      short-name profiles, ``-company_type``/``-noise_words``).
    - a bare name selects an operation that is off by default
      (``default|transliterate``, ``default|geographic_terms``).
    - ``+x`` adds an optional tier to an operation the profile has already selected
      (``default|geographic_terms|+region|+city``). A tier modifier with no
      ``geographic_terms`` in the profile is an error, not a no-op.

    Supported forms:
    - ``default``
    - ``default|-lowercase`` (remove an operation from default)
    - ``default|geographic_terms`` (add an off-by-default operation; ``geographic`` is
      accepted as a shorthand for ``geographic_terms``)
    - ``default|geographic_terms|+region|+city`` (add tiers to it; ``country`` is the
      default tier and is always included)
    - ``punctuation|singlespace|and|singlechar|diacritics|transliterate|lowercase`` (fully specified chain)
    """
    raw_profile = (
        (normalization_profile or DEFAULT_NORMALIZATION_PROFILE).strip().lower()
    )
    if not raw_profile:
        raw_profile = DEFAULT_NORMALIZATION_PROFILE

    parts = [part.strip() for part in raw_profile.split("|") if part.strip()]
    if not parts:
        return DEFAULT_NORMALIZATION_OPERATIONS

    parts, tiers = _extract_geographic_tier_modifiers(parts, raw_profile)
    if not parts:
        return _apply_geographic_tier_modifiers(
            DEFAULT_NORMALIZATION_OPERATIONS, tiers, raw_profile
        )

    if parts[0] == DEFAULT_NORMALIZATION_PROFILE:
        operations = _parse_default_profile_operations(parts, raw_profile)
    else:
        operations = _parse_explicit_profile_operations(parts, raw_profile)

    return _apply_geographic_tier_modifiers(operations, tiers, raw_profile)


def normalize_operations(operations: Sequence[str]) -> tuple[str, ...]:
    """Normalize an ordered operation sequence to a validated lowercase tuple.

    Accepts the ``geographic`` alias and any ``+tier`` suffix, returning the canonical
    spelling in both cases, so a directly-supplied chain resolves to the same tuple
    (and therefore the same normalization cache key) as the profile that names it.
    """
    resolved_operations = tuple(
        _canonicalize_operation(str(operation).strip().lower())
        for operation in operations
        if str(operation).strip()
    )
    for operation in resolved_operations:
        if operation_base_name(operation) not in SUPPORTED_NORMALIZATION_OPERATIONS:
            available = ", ".join(sorted(SUPPORTED_NORMALIZATION_OPERATIONS))
            raise ValueError(
                f"Unknown normalization operation '{operation}'. "
                f"Available operations: {available}."
            )
    return resolved_operations


def with_transliteration_operation(operations: Sequence[str]) -> tuple[str, ...]:
    """Return operations with transliteration inserted before lowercase/casefold when absent."""
    resolved_operations = list(normalize_operations(operations))
    if NORMALIZATION_OPERATION_TRANSLITERATE in resolved_operations:
        return tuple(resolved_operations)

    case_index = next(
        (
            index
            for index, operation in enumerate(resolved_operations)
            if operation in CASE_FOLDING_OPERATIONS
        ),
        None,
    )
    if case_index is not None:
        resolved_operations.insert(case_index, NORMALIZATION_OPERATION_TRANSLITERATE)
    else:
        resolved_operations.append(NORMALIZATION_OPERATION_TRANSLITERATE)

    return tuple(resolved_operations)


def with_bang_preserving_punctuation_operation(
    operations: Sequence[str],
) -> tuple[str, ...]:
    """Replace plain punctuation operations with bang-preserving punctuation."""
    resolved_operations = list(normalize_operations(operations))
    return tuple(
        NORMALIZATION_OPERATION_PUNCTUATION_BANG
        if operation == NORMALIZATION_OPERATION_PUNCTUATION
        else operation
        for operation in resolved_operations
    )


def _run_operation_singlespace(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    del connector_pattern
    return re.sub(r"\s+", " ", text).strip()


def _normalization_profile_key(normalization_profile: str | None) -> str:
    return "|".join(parse_normalization_profile(normalization_profile))


def resolve_char_whitelist_for_profile(
    char_whitelist: str,
    normalization_profile: str = DEFAULT_NORMALIZATION_PROFILE,
) -> str:
    """Adjust char whitelist when lowercasing is disabled in the profile.

    The default whitelist historically assumes lowercase output. If callers remove
    `lowercase`, keep uppercase letters from being stripped by expanding `a-z`
    ranges to `A-Za-z`.
    """
    operations = parse_normalization_profile(normalization_profile)
    return resolve_char_whitelist_for_operations(
        char_whitelist,
        operations=operations,
    )


def resolve_char_whitelist_for_operations(
    char_whitelist: str,
    *,
    operations: Sequence[str],
) -> str:
    """Adjust char whitelist when case-folding is disabled in resolved operations.

    Treats `lowercase` and `casefold` alike: either already yields lowercase-only
    output, so the whitelist's `a-z` range needs no widening for either.
    """
    normalized_operations = set(normalize_operations(operations))
    if normalized_operations & CASE_FOLDING_OPERATIONS:
        return char_whitelist

    return char_whitelist.replace("a-z", "A-Za-z")


@lru_cache(maxsize=512)
def _connector_substitution_pattern(
    deduped_and_tokens: tuple[str, ...],
) -> re.Pattern[str] | None:
    connector_tokens = [token for token in deduped_and_tokens if token and token != "&"]
    if not connector_tokens:
        return None

    # Prefer longer alternatives first to avoid partial matches when tokens share prefixes.
    ordered = sorted(connector_tokens, key=len, reverse=True)
    alternation = "|".join(re.escape(token) for token in ordered)
    return re.compile(rf"(?<!\w)(?:{alternation})(?!\w)", flags=re.IGNORECASE)


def _run_operation_transliterate(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    del connector_pattern
    return transliterate_to_ascii(text) or ""


def _run_operation_lowercase(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    del connector_pattern
    return text.lower()


def _run_operation_casefold(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    """Run Unicode's caseless-match fold, not display lowercasing.

    Diverges from `lowercase` on real register text: `"Straße".casefold()` ==
    `"strasse"` (one codepoint expands to two) while `.lower()` leaves `"straße"`
    unchanged, and `.casefold()` unifies Greek final sigma with medial sigma
    (`"ΣΣΣς".casefold()` == `"σσσσ"`) where `.lower()` keeps them distinct.
    """
    del connector_pattern
    return text.casefold()


def _run_operation_diacritics(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    del connector_pattern
    return strip_diacritics(text) or ""


def _run_operation_and(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    normalized = re.sub(r"\s*&\s*", " & ", text)
    if connector_pattern is not None:
        normalized = connector_pattern.sub(" & ", normalized)
        normalized = re.sub(r"\s*&\s*", " & ", normalized)
    if normalized.strip() == "&":
        return "&"
    return normalized


def _run_operation_punctuation(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    del connector_pattern
    pattern = r"[^\w\s&]"
    normalized = re.sub(pattern, " ", text)
    normalized = normalized.replace("_", " ")
    return normalized


def _run_operation_punctuation_bang(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    del connector_pattern
    pattern = r"[^\w\s&!]"
    normalized = re.sub(pattern, " ", text)
    normalized = normalized.replace("_", " ")
    return normalized


def _run_operation_singlechar(
    text: str,
    connector_pattern: re.Pattern[str] | None,
) -> str:
    del connector_pattern
    return collapse_single_char_sequences(text) or ""


def _make_geographic_terms_handler(
    tiers: tuple[str, ...],
) -> NormalizationOperationHandler:
    def _run_operation_geographic_terms(
        text: str,
        connector_pattern: re.Pattern[str] | None,
    ) -> str:
        del connector_pattern
        return strip_geographic_terms(text, kinds=tiers) or ""

    return _run_operation_geographic_terms


NORMALIZATION_OPERATION_REGISTRY: dict[str, NormalizationOperationHandler] = {
    NORMALIZATION_OPERATION_TRANSLITERATE: _run_operation_transliterate,
    NORMALIZATION_OPERATION_LOWERCASE: _run_operation_lowercase,
    NORMALIZATION_OPERATION_CASEFOLD: _run_operation_casefold,
    NORMALIZATION_OPERATION_DIACRITICS: _run_operation_diacritics,
    NORMALIZATION_OPERATION_SINGLESPACE: _run_operation_singlespace,
    NORMALIZATION_OPERATION_AND: _run_operation_and,
    NORMALIZATION_OPERATION_PUNCTUATION: _run_operation_punctuation,
    NORMALIZATION_OPERATION_PUNCTUATION_BANG: _run_operation_punctuation_bang,
    NORMALIZATION_OPERATION_SINGLECHAR: _run_operation_singlechar,
    NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS: _make_geographic_terms_handler(
        DEFAULT_GEOGRAPHIC_TERM_KINDS
    ),
}


@lru_cache(maxsize=64)
def _resolve_operation_handler(operation: str) -> NormalizationOperationHandler:
    """Resolve a possibly-tiered operation name to the handler that runs it.

    The registry stays keyed on bare operation names, so it remains the single list of
    available operations for error messages; a tiered `geographic_terms+region` builds
    its own handler once and is cached here rather than on every row.
    """
    base = operation_base_name(operation)
    if base == NORMALIZATION_OPERATION_GEOGRAPHIC_TERMS:
        return _make_geographic_terms_handler(geographic_tiers_for_operation(operation))

    handler = NORMALIZATION_OPERATION_REGISTRY.get(base)
    if handler is None:
        available = ", ".join(sorted(NORMALIZATION_OPERATION_REGISTRY))
        raise ValueError(
            f"Unsupported normalization operation '{operation}'. "
            f"Available operations: {available}."
        )
    return handler


@lru_cache(maxsize=65536)
def _normalize_tokens_cached(
    text: str,
    and_tokens_key: tuple[str, ...] | None,
    operations: tuple[str, ...],
) -> str:
    deduped_and_tokens = _resolve_and_tokens(and_tokens_key)
    connector_pattern = _connector_substitution_pattern(deduped_and_tokens)
    normalized = text

    for operation in operations:
        normalized = _resolve_operation_handler(operation)(
            normalized, connector_pattern
        )

    return normalized


def normalize_company_type_value(text: str | None, transliterate: bool = False) -> str:
    """Normalize rule values and canonical values for stable matching and validation."""
    if not text:
        return ""

    operations = (
        COMPANY_TYPE_TRANSLITERATION_OPERATIONS
        if transliterate
        else COMPANY_TYPE_NORMALIZATION_OPERATIONS
    )
    return _normalize_company_type_value_cached(text, operations)


@lru_cache(maxsize=65536)
def _normalize_company_type_value_cached(text: str, operations: tuple[str, ...]) -> str:
    # Keep company-type normalization stable even if default profile changes.
    return (
        normalize_tokens_with_operations(
            text,
            operations=operations,
        )
        or ""
    )


# Suffix-surface normalization intentionally shares the same implementation.
normalize_suffix_surface = normalize_tokens
normalize_suffix_surface_with_operations = normalize_tokens_with_operations


_strip_diacritics = strip_diacritics
_transliterate_to_ascii = transliterate_to_ascii
_collapse_single_char_sequences = collapse_single_char_sequences
_normalize_tokens = normalize_tokens
_normalize_company_type_value = normalize_company_type_value
_normalize_suffix_surface = normalize_suffix_surface
