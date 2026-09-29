"""Command-line settings this package's code reads, declared as plain data.

A script reads these to build its flags, name each default in its help, and
report which settings apply. Every tokenizer setting applies only to a run
reading the tokens text view. Nothing here builds a parser: the package stays a
library, and the declaration shape is a documented convention rather than a
type shared with other packages.
"""

from __future__ import annotations

from .training import SUPPORTED_TRAINERS

_TOKENS_VIEW = {"text_view": ("tokens",)}

SETTINGS: tuple[dict[str, object], ...] = (
    {
        "name": "tokenizer",
        "type": "str",
        "default": "wordpiece",
        "choices": SUPPORTED_TRAINERS,
        "help": "Tokenizer whose artifact tokenizes the names.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "tokenizer_encoding",
        "type": "str",
        "default": "bpe",
        "choices": ("bpe", "unigram"),
        "help": "SentencePiece encoding, promoted separately; wordpiece has none and ignores it.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "tokenizer_scope",
        "type": "str",
        "default": "country",
        "choices": ("country", "global"),
        "help": "Whether the tokenizer is resolved per country or globally.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "tokenizer_profile",
        "type": "str",
        "default": "promoted",
        "help": "Which of the scope's stored tokenizers to use, by its label: promoted or naive.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "tokenizer_path",
        "type": "str",
        "default": None,
        "help": "A tokenizer model file to use as it is, in place of the one the profile names.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "noise_words_profile",
        "type": "str",
        "default": "none",
        "help": "Noise-word profile removed before tokenizing; none removes no words.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "noise_words_set_kind",
        "type": "str",
        "default": "combined",
        "help": "Which noise-word set of that profile is removed.",
        "applies": _TOKENS_VIEW,
    },
)
"""Every tokenizer setting a blocking run takes."""
