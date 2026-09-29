"""Command-line settings this area's scripts read, declared as plain data.

A script reads these through `scripts/cli_common.py` to build its flags, name
each default in its help, and report which settings a run took and where each
value came from. Nothing here builds a parser; the declaration shape is the
convention `cli_common` documents, and a script decides whether a setting is a
flag or a prefixed key and whether it overrides the default.

A flag naming which run this is -- a system, a run date, an input or output
path, a column override -- is structural and stays an ordinary flag on the
script that takes it; it is not declared here.
"""

from __future__ import annotations

from company_tokenize.training import SUPPORTED_TRAINERS
from company_vectorize.clustering_contract import TFIDF_ANALYZERS

from .config import resolve_prepared_rows_per_file

_TOKENS_VIEW = {"representation": ("wordpiece", "sentencepiece")}

SETTINGS: tuple[dict[str, object], ...] = (
    {
        "name": "representation",
        "type": "str",
        "default": "tfidf",
        "help": "Clustering representation model.",
    },
    {
        "name": "similarity",
        "type": "str",
        "default": "cosine",
        "help": "Similarity metric.",
    },
    {
        "name": "similarity_backend",
        "type": "str",
        "default": "sklearn",
        "choices": ("sklearn", "sparse_dot_topn", "svd_rerank"),
        "help": (
            "Similarity execution backend. 'svd_rerank' projects to a dense "
            "low-dimensional space before reranking exactly -- a constant-factor "
            "speedup, not a sub-linear ANN index (see sparse_similarity.py); "
            "brute-force backends remain acceptable up to roughly ie's ~820K "
            "rows, but larger scenarios (for example fr's ~12.9M rows) need "
            "a sub-linear backend: lsh for sparse representations, hnsw for dense."
        ),
    },
    {
        "name": "clustering",
        "type": "str",
        "default": "knn_cc",
        "help": "Clustering/linkage strategy.",
    },
    {
        "name": "text_view",
        "type": "str",
        "default": "auto",
        "choices": ("auto", "name", "tokens"),
        "help": "Text view mode (defaults to representation-derived auto).",
    },
    {
        "name": "top_k",
        "type": "int",
        "default": 20,
        "help": "Top-k neighbours per source record.",
    },
    {
        "name": "min_similarity",
        "type": "float",
        "default": 0.75,
        "help": "Minimum cosine similarity threshold.",
    },
    {
        "name": "max_candidates_per_source",
        "type": "int",
        "default": None,
        "help": "Optional cap after thresholding.",
    },
    {
        "name": "progress_record_interval",
        "type": "int",
        "default": 10_000,
        "help": "Emit country progress at least every N source records scored.",
    },
    {
        "name": "source_chunk_size",
        "type": "int",
        "default": 1000,
        "help": "Source scoring batch size and progress checkpoint interval.",
    },
    {
        "name": "progress_time_interval_seconds",
        "type": "float",
        "default": 30.0,
        "help": "Emit country progress at least every N seconds when chunk events arrive.",
    },
    {
        "name": "prepared_rows_per_file",
        "type": "int",
        "default": resolve_prepared_rows_per_file(None),
        "help": "Maximum rows per prepared parquet file.",
    },
    {
        "name": "max_rows",
        "type": "int",
        "default": None,
        "help": (
            "Optional runtime cap on source rows scored per country (target "
            "country dataset remains full-size)."
        ),
    },
    {
        "name": "tfidf_ngram_min",
        "type": "int",
        "default": 2,
        "help": "Shortest n-gram the TF-IDF vectorizer emits.",
        "applies": {"representation": ("tfidf",)},
    },
    {
        "name": "tfidf_ngram_max",
        "type": "int",
        "default": 3,
        "help": "Longest n-gram the TF-IDF vectorizer emits.",
        "applies": {"representation": ("tfidf",)},
    },
    {
        "name": "tfidf_analyzer",
        "type": "str",
        "default": "char_wb",
        "choices": tuple(sorted(TFIDF_ANALYZERS)),
        "help": "What a TF-IDF n-gram is a run of.",
        "applies": {"representation": ("tfidf",)},
    },
    {
        "name": "tokenizer",
        "type": "str",
        "default": "wordpiece",
        "choices": SUPPORTED_TRAINERS,
        "help": "Tokenizer whose artifact tokenizes the names.",
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
        "default": "aggressive",
        "help": "Noise-word profile removed from the emitted tokens.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "noise_words_set_kind",
        "type": "str",
        "default": "combined",
        "help": "Which noise-word set of that profile is removed.",
        "applies": _TOKENS_VIEW,
    },
    {
        "name": "force_rebuild",
        "type": "bool",
        "default": False,
        "help": "Rebuild materialized output even if it already exists for this run.",
    },
)
"""Every setting this area's scripts take."""
