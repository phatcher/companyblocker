"""The supported entry points: `cleanse_lazyframe` for a batch, and its single-string equivalents."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from .config import CleanseConfig, get_effective_noise_words
from .extract import (
    _derive_short_name_from_cleansed,
    _extract_leading_quoted_tokens,
    _split_short_name_stage_flags,
)
from .normalize import DEFAULT_NORMALIZATION_PROFILE, normalize_suffix_surface
from .pipeline import _process_cleanse_lazyframe
from .rules import get_company_type_rules


def extract_quoted_name(company_name: str | None) -> str | None:
    """Return the leading quoted segment of a company name, if present.

    Single-string equivalent of the `quoted_name` column `cleanse_lazyframe()`
    produces: a name beginning with a `"..."` or `'...'` segment followed by more
    text returns that segment's normalized token form (for example `'"Acme" Ltd'`
    -> `"acme"`); a bare quoted string with nothing after it, or no leading quote
    at all, returns ``None``. Unlike `strip_company_suffix()`, this needs no
    company-type or noise-word configuration -- quoted-segment extraction is
    purely structural.
    """
    return _extract_leading_quoted_tokens(company_name)


def _resolve_company_type_rules(config: CleanseConfig) -> tuple[str, dict[str, str]]:
    if (
        config.company_type_regex is not None
        and config.company_type_mapping is not None
    ):
        return config.company_type_regex, config.company_type_mapping

    default_regex, default_mapping = get_company_type_rules()
    return (
        config.company_type_regex or default_regex,
        config.company_type_mapping or default_mapping,
    )


def cleanse_lazyframe(
    lf: pl.LazyFrame,
    config: CleanseConfig | None = None,
    *,
    file_columns: set[str] | list[str] | tuple[str, ...] | None = None,
) -> pl.LazyFrame:
    """Apply deterministic company-name cleansing to a Polars LazyFrame.

    The function uses packaged company-type rules unless they are overridden via
    ``CleanseConfig``. Every input column is kept, and these are appended:

    - ``name_cleansed``: The cleansed name tokenization reads, with the canonical company type.
    - ``name_cleansed_basic``: The name after the full default normalization chain and character
      filtering, without company-type replacement; ``singlechar`` has already collapsed spaced
      initials (``I B M`` to ``ibm``).
    - ``short_name``: A leading quoted or parenthesized name where one is extracted, otherwise
      derived from ``name_cleansed`` by stripping the company type and noise words.
    - ``quoted_name``: The leading quoted segment, when present and followed by more text.
    - ``acronym``: ``short_name`` when it matches the initials of the cleansed name body.
    - ``company_type``: The canonical company type, or default private handling.
    - ``company_type_source``: ``name_suffix``, ``source_mapping`` or ``default_private``.
    - ``company_type_missing``: Whether no explicit type was found before defaulting.
    - ``personal_owner``: An owner name split out where configured owner markers appear.

    A ``short_name`` that is a bare legal-form token is never an ``acronym``; when no ``acronym``
    was derived, it is prepended to ``name_cleansed`` instead. A ``quoted_name`` not equal to the
    ``acronym`` is prepended too. Neither is prepended when the cleansed name already starts with
    it. An input that is effectively symbols
    only, or leads with its legal form (``LTD ...``), can still yield ``short_name`` ``ltd``.
    """
    effective_config = config or CleanseConfig()
    company_type_regex, company_type_mapping = _resolve_company_type_rules(
        effective_config
    )
    resolved_input_columns: list[str] | None = None
    if file_columns is None:
        resolved_input_columns = lf.collect_schema().names()
    elif not isinstance(file_columns, set):
        resolved_input_columns = [str(column) for column in file_columns]

    company_col = effective_config.company_col

    source_company_type_col = effective_config.source_company_type_col
    source_company_type_mapping = effective_config.source_company_type_mapping
    use_source_company_type = (
        file_columns is not None
        and source_company_type_col is not None
        and source_company_type_col in file_columns
        and source_company_type_mapping is not None
    )

    return _process_cleanse_lazyframe(
        lf=lf,
        company_col=company_col,
        company_type_regex=company_type_regex,
        company_type_mapping=company_type_mapping,
        company_type_matcher=effective_config.company_type_matcher,
        normalization_profile=effective_config.normalization_profile,
        noise_words_profile=effective_config.noise_words_profile,
        noise_words_set_kind=effective_config.noise_words_set_kind,
        short_name_profile=effective_config.short_name_profile,
        and_tokens=effective_config.and_tokens,
        personal_owner_markers=effective_config.personal_owner_markers,
        char_whitelist=effective_config.char_whitelist,
        source_company_type_col=source_company_type_col,
        source_company_type_mapping=source_company_type_mapping,
        use_source_company_type=use_source_company_type,
        step_engines=effective_config.step_engines,
        input_columns=resolved_input_columns,
        jurisdiction_col=effective_config.jurisdiction_col,
    )


def _load_noise_words_override(
    noise_words_path: str | Path,
) -> tuple[str | dict[str, str], ...]:
    path = Path(noise_words_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        suffix_entries = payload.get("suffix", [])
        anywhere_entries = payload.get("anywhere", [])
        if not isinstance(suffix_entries, list) or not isinstance(
            anywhere_entries, list
        ):
            raise TypeError(
                "noise_words override must provide list values for 'suffix' and 'anywhere'."
            )

        entries: list[str | dict[str, str]] = []
        entries.extend(
            str(value).strip() for value in suffix_entries if str(value).strip()
        )
        entries.extend(
            {"token": str(value).strip(), "scope": "anywhere"}
            for value in anywhere_entries
            if str(value).strip()
        )
        return tuple(entries)

    if isinstance(payload, list):
        return tuple(str(value).strip() for value in payload if str(value).strip())

    raise ValueError(
        "noise_words override must be a list or an object with 'suffix'/'anywhere' keys."
    )


def strip_company_suffix(
    company_name: str | None,
    *,
    company_type: str | None = None,
    noise_words: tuple[str | dict[str, str], ...]
    | list[str | dict[str, str]]
    | None = None,
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "aggressive",
    noise_words_set_kind: str = "combined",
    normalization_profile: str = DEFAULT_NORMALIZATION_PROFILE,
    jurisdiction_code: str | None = None,
) -> str | None:
    """Strip trailing company-type suffix components from a company name.

    This reuses the same suffix/anywhere noise-word behavior as the cleanse pipeline
    and allows optional override tokens loaded from JSON.

    ``jurisdiction_code``, when given (for example ``"gb"``, ``"fr"``, ``"de"``),
    additionally pulls in the promoted, corpus-derived noise words for that
    jurisdiction as extra suffix-scope noise words -- this is how a standalone
    caller with no acquisition-pipeline context gets a better short name without
    needing to know `company_tokenize` exists. Falls back to a cross-language
    ``global`` word list when omitted or unmapped; never applies one language's
    words to another language's names.

    ``normalization_profile`` accepts the same ``normalize_tokens()`` profile syntax
    (for example ``default|-lowercase``) plus two additional toggles that control the
    short-name derivation stages run *after* normalizing, not the text normalization
    itself. Both are on by default, matching prior behavior:

    - ``default|-noise_words`` -- the common case: still trims the trailing
      company-type token (for example ``Plc``) but leaves other trailing noise
      words alone, e.g. ``"Acme Systems Plc"`` -> ``"acme systems"`` instead of the
      default ``"acme"``.
    - ``default|-company_type`` -- skips trailing company-type component trimming
      but still runs noise-word removal. Caveat: noise-word suffix trimming only
      works from the end of the name inward, so a company-type token left in place
      at the tail can block removal of noise words that would otherwise sit before
      it, e.g. ``"Acme Systems Plc"`` -> ``"acme systems plc"`` (nothing stripped),
      not ``"acme systems"`` -- ``"systems"`` is one token away from the end.
    - ``default|-company_type|-noise_words`` -- disables both, so nothing is
      stripped and the call reduces to a plain "tidy up" equivalent to
      ``normalize_suffix_surface``. Rarely what you want given the function's name;
      reach for ``normalize_suffix_surface``/``normalize_tokens`` directly instead
      if tidying up is all you need.

    A third stage is available, and unlike those two it is opt-*in*:

    - ``default|geographic_terms`` (or its ``geographic`` shorthand) additionally
      strips a trailing or bracketed geographic term once the company-type stage has
      removed the legal form, so ``"Oracle Ireland Ltd"`` -> ``"oracle"``. It runs
      between the company-type and noise-word stages. ``+region`` and ``+city`` add
      tiers on top of the default ``country`` tier
      (``"default|geographic_terms|+region"``); a tier with no ``geographic_terms``
      in the profile is rejected rather than ignored.
    """
    flags = _split_short_name_stage_flags(normalization_profile)
    include_noise_words = flags.include_noise_words

    normalized_name = normalize_suffix_surface(
        company_name, normalization_profile=flags.remaining_profile
    )
    if not normalized_name:
        return None

    effective_noise_words: (
        tuple[str | dict[str, str], ...] | list[str | dict[str, str]] | None
    ) = None
    if include_noise_words:
        effective_noise_words = noise_words
        if noise_words_path is not None:
            effective_noise_words = _load_noise_words_override(noise_words_path)
        if effective_noise_words is None:
            effective_noise_words = get_effective_noise_words(
                noise_words_profile=flags.noise_words_level or noise_words_profile,
                noise_words_set_kind=noise_words_set_kind,
            )

    return _derive_short_name_from_cleansed(
        normalized_name,
        company_type,
        effective_noise_words,
        include_company_type=flags.include_company_type,
        include_noise_words=include_noise_words,
        geographic_tiers=flags.geographic_tiers,
        jurisdiction_code=jurisdiction_code,
    )
