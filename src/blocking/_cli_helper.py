"""Command-line settings this area's run reads, declared as plain data.

A script reads these through `scripts/cli_common.py` to build its flags, name
each default in its help, and report which settings a run took and where each
value came from. Nothing here builds a parser; the declaration shape is the
convention `cli_common` documents, and a script decides whether a setting is a
flag or a prefixed key and whether it overrides the default.
"""

from __future__ import annotations

from company_tokenize.name_preprocessing import DEFAULT_NAME_PREPROCESSING_PROFILE

from .name_transform import (
    DEFAULT_CLEANSE_PROFILE,
    DEFAULT_NAME_TRANSFORM,
    NAME_TRANSFORMS,
)
from .truth import MATCH_URI_COLUMN

SETTINGS: tuple[dict[str, object], ...] = (
    {
        "name": "source_system",
        "type": "str",
        "default": None,
        "help": (
            "Source system code, or a perturbed dataset's URI "
            "perturbed://<system>/<profile>/<version>/<seed> naming one materialized dataset."
        ),
    },
    {
        "name": "target_system",
        "type": "str",
        "default": None,
        "help": "Target system code.",
    },
    {
        "name": "countries",
        "type": "str",
        "default": None,
        "nargs": "+",
        "help": "Countries to score; unset scores every country the pairing carries.",
    },
    {
        "name": "match_col",
        "type": "str",
        "default": MATCH_URI_COLUMN,
        "help": (
            "Source column naming the target row each source row is equal to: "
            "match_uri for the source's recorded cross-system match, source_uri "
            "for a perturbed dataset or name sidecar scored against the system "
            "it came from."
        ),
    },
    {
        "name": "require_ground_truth",
        "type": "bool",
        "default": True,
        "help": (
            "Require the source to resolve a matched layer carrying ground truth; "
            "disabled, the source is read from its canonical snapshot with no truth "
            "scoring."
        ),
    },
    {
        "name": "name_transform",
        "type": "str",
        "default": DEFAULT_NAME_TRANSFORM,
        "choices": tuple(sorted(NAME_TRANSFORMS)),
        "help": (
            "The one function applied to both sides' names before comparison: "
            "identity scores each side's own column, the others score the form "
            "derived under the cleanse profile."
        ),
    },
    {
        "name": "preprocess_profile",
        "type": "str",
        "default": DEFAULT_NAME_PREPROCESSING_PROFILE,
        "help": (
            "Name preprocessing profile both sides' names pass through before they "
            "are tokenized or vectorized: default removes the company type and keeps "
            "noise words, default|+noise_words:<level> also removes noise words at "
            "that level, default|-company_type keeps the company type."
        ),
    },
    {
        "name": "cleanse_profile",
        "type": "str",
        "default": DEFAULT_CLEANSE_PROFILE,
        "help": "company_cleanse normalization profile both sides' name forms are derived under.",
    },
    {
        "name": "encoder",
        "type": "str",
        "default": None,
        "choices": ("fasttext",),
        "help": (
            "The encoder an encoder representation embeds names with: fasttext is "
            "a pretrained fastText checkpoint, each name the mean of its words' "
            "vectors. Required with the encoder representation, refused with any "
            "other."
        ),
        "applies": {"representation": ("encoder",)},
    },
    {
        "name": "fasttext_checkpoint",
        "type": "str",
        "default": None,
        "help": (
            "A registered fastText checkpoint slug; unset picks one from the run's "
            "countries."
        ),
        "applies": {"representation": ("encoder",)},
    },
    {
        "name": "top_k",
        "type": "int",
        "default": 20,
        "help": "Nearest neighbours retrieved per source row.",
    },
    {
        "name": "min_similarity",
        "type": "float",
        "default": 0.75,
        "help": "Minimum cosine similarity a candidate must reach.",
    },
    {
        "name": "max_candidates_per_source",
        "type": "int",
        "default": None,
        "help": "Cap on candidates kept per source row after thresholding.",
    },
    {
        "name": "max_candidates_per_target",
        "type": "int",
        "default": None,
        "help": "Cap on source rows that may claim one target row.",
    },
    {
        "name": "candidate_similarity_ratio",
        "type": "float",
        "default": None,
        "help": "Drop a source row's candidates below this ratio of its best match.",
    },
    {
        "name": "source_chunk_size",
        "type": "int",
        "default": 10_000,
        "help": "Source rows scored per batch against the target index.",
    },
    {
        "name": "target_neighbor_min_similarity",
        "type": "float",
        "default": None,
        "help": (
            "Minimum cosine similarity for a target-to-target near-neighbour "
            "edge, probed once per target index and unioned into clustering; "
            "must be set together with target_neighbor_max_per_target, and "
            "leaves target-neighbour linking off when both are unset."
        ),
    },
    {
        "name": "target_neighbor_max_per_target",
        "type": "int",
        "default": None,
        "help": (
            "Cap on near-neighbour edges kept per probed target row; must be "
            "set together with target_neighbor_min_similarity."
        ),
    },
)
"""Every setting a blocking run takes that this area reads."""
