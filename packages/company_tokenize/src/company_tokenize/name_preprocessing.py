"""The one pre-processing step a name passes through before it is tokenized or vectorized.

A cleansed name keeps its company type, normalised to one spelling, because
the cleanse output is the name. What a tokenizer or a vectorizer should see is
a separate question with its own answer: `name_preprocessing` writes that text
to its own column beside the name it was derived from, and every caller that
turns names into tokens or vectors reads that column. What the step means can
then change here, in one place, without a caller or a recorded setting moving.

It also carries a guarantee the steps before it cannot. A run may choose a
cleanse profile that keeps capitals, or score the raw name, and a tokenizer
handed text in a form it was not trained on answers in unknown tokens. So
every profile here passes the name through cleanse's fixed normalization,
lowercase with punctuation and diacritics normalised, and no profile can opt
out of it: the least this step does is normalize, never nothing.

Today it means cleanse's own `strip_company_suffix`, so what a company type
is, and what a noise word is, stay decided by `company_cleanse`. A profile
string says which parts run, in the mini-language cleanse's profiles use:

    default                                  company type removed, noise words kept
    default|+noise_words:balanced            and noise words removed at that level
    default|-company_type                    company type kept: normalised, nothing removed
    default|-company_type|+noise_words:strict

`+noise_words` always names its level, so a recorded profile never rests on
an unstated default.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from company_cleanse import strip_company_suffix

NAME_PREPROCESSED_COLUMN = "name_preprocessed"
DEFAULT_NAME_PREPROCESSING_PROFILE = "default"
TRAINING_PREPROCESS_PROFILE = "default|-company_type"
"""The profile a tokenizer's training column passes through: normalised,
nothing removed, so it covers every text a run can hand the tokenizer. A
column cleansed under a profile that keeps capitals would otherwise train
a tokenizer on text no run hands it."""

_PROFILE_SEPARATOR = "|"
_DEFAULT_TOKEN = "default"
_KEEP_COMPANY_TYPE_TOKEN = "-company_type"
_NOISE_WORDS_PREFIX = "+noise_words"
_LEVEL_SEPARATOR = ":"


@dataclass(frozen=True)
class NamePreprocessing:
    """What a profile string asks for.

    Whether the company type goes, and the noise words level removed, or
    None to keep noise words.
    """

    remove_company_type: bool = True
    noise_words_level: str | None = None


def parse_name_preprocessing_profile(profile: str) -> NamePreprocessing:
    """Read a profile string, refusing anything it does not spell out."""
    tokens = [token.strip() for token in str(profile).split(_PROFILE_SEPARATOR)]
    if not tokens or tokens[0] != _DEFAULT_TOKEN:
        raise ValueError(
            f"A name preprocessing profile starts with '{_DEFAULT_TOKEN}', not {profile!r}."
        )
    remove_company_type = True
    noise_words_level: str | None = None
    for token in tokens[1:]:
        if token == _KEEP_COMPANY_TYPE_TOKEN:
            remove_company_type = False
        elif token.split(_LEVEL_SEPARATOR, 1)[0] == _NOISE_WORDS_PREFIX:
            _, _, level = token.partition(_LEVEL_SEPARATOR)
            if not level.strip():
                raise ValueError(
                    f"'{_NOISE_WORDS_PREFIX}' names its level, for example "
                    f"'{_NOISE_WORDS_PREFIX}{_LEVEL_SEPARATOR}balanced', in {profile!r}."
                )
            noise_words_level = level.strip()
        else:
            raise ValueError(
                f"Unknown name preprocessing step {token!r} in {profile!r}. Expected "
                f"'{_KEEP_COMPANY_TYPE_TOKEN}' or "
                f"'{_NOISE_WORDS_PREFIX}{_LEVEL_SEPARATOR}<level>'."
            )
    return NamePreprocessing(
        remove_company_type=remove_company_type, noise_words_level=noise_words_level
    )


def _cleanse_profile(settings: NamePreprocessing) -> str:
    """The `strip_company_suffix` profile that runs the parts `settings` asks for.

    The two grammars share their words and differ in one default: cleanse
    removes noise words unless told not to, and this step keeps them unless
    told to remove them.
    """
    toggles = [_DEFAULT_TOKEN]
    if not settings.remove_company_type:
        toggles.append(_KEEP_COMPANY_TYPE_TOKEN)
    toggles.append(
        "-noise_words"
        if settings.noise_words_level is None
        else f"{_NOISE_WORDS_PREFIX}{_LEVEL_SEPARATOR}{settings.noise_words_level}"
    )
    return _PROFILE_SEPARATOR.join(toggles)


def name_preprocessing(
    frame: pl.DataFrame,
    *,
    name_col: str,
    profile: str = DEFAULT_NAME_PREPROCESSING_PROFILE,
    jurisdiction_col: str | None = None,
    out_col: str = NAME_PREPROCESSED_COLUMN,
) -> pl.DataFrame:
    """Add `out_col`, the text of `name_col` a tokenizer or vectorizer should read.

    `name_col` is left as it was. A null name gives a null, as does a name
    with nothing left once normalised. `jurisdiction_col`,
    when given, lets noise word removal use that jurisdiction's own corpus
    words as cleanse does; it changes nothing while noise words are kept. Each
    distinct name is processed once, however often it repeats.
    """
    if name_col not in frame.columns:
        raise ValueError(f"DataFrame does not contain required column '{name_col}'.")
    settings = parse_name_preprocessing_profile(profile)
    names = frame.get_column(name_col).cast(pl.Utf8, strict=False)

    use_jurisdiction = (
        jurisdiction_col is not None
        and jurisdiction_col in frame.columns
        and settings.noise_words_level is not None
    )
    jurisdictions = (
        frame.get_column(str(jurisdiction_col)).cast(pl.Utf8, strict=False).to_list()
        if use_jurisdiction
        else [None] * frame.height
    )
    cleanse_profile = _cleanse_profile(settings)

    processed: dict[tuple[str | None, str | None], str | None] = {}
    values: list[str | None] = []
    for name, jurisdiction in zip(names.to_list(), jurisdictions, strict=True):
        key = (name, jurisdiction)
        if key not in processed:
            processed[key] = (
                None
                if name is None
                else strip_company_suffix(
                    name,
                    normalization_profile=cleanse_profile,
                    jurisdiction_code=jurisdiction,
                )
            )
        values.append(processed[key])
    return frame.with_columns(pl.Series(out_col, values, dtype=pl.Utf8))


__all__ = [
    "DEFAULT_NAME_PREPROCESSING_PROFILE",
    "NAME_PREPROCESSED_COLUMN",
    "NamePreprocessing",
    "TRAINING_PREPROCESS_PROFILE",
    "name_preprocessing",
    "parse_name_preprocessing_profile",
]
