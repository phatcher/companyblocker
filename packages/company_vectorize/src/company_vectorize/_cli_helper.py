"""Command-line settings this package's code reads, declared as plain data.

A script reads these to build its flags, name each default in its help, and
report which settings apply to the representation and backend a run chose.
Nothing here builds a parser: the package stays a library, and the declaration
shape is a documented convention rather than a type shared with other packages.
Each backend option is declared beside the default the backend itself falls
back to, so the two cannot drift apart silently.
"""

from __future__ import annotations

from .clustering_contract import TFIDF_ANALYZERS
from .dense_brute_similarity import (
    DEFAULT_DENSE_BRUTE_BLOCK_BYTES,
    DEFAULT_DENSE_BRUTE_STORAGE_DTYPE,
)
from .dense_vocabulary_gate import (
    DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    DENSE_VOCABULARY_REPRESENTATIONS,
    EXHAUSTIVE_SPARSE_BACKENDS,
)
from .hnsw_similarity import (
    BUILD_THREADS_OPTION,
    DEFAULT_HNSW_BUILD_THREADS,
    DEFAULT_HNSW_CONNECTIVITY,
    DEFAULT_HNSW_EXPANSION_ADD,
    DEFAULT_HNSW_EXPANSION_SEARCH,
    HNSW_STORAGE_DTYPES,
)
from .lsh_similarity import (
    DEFAULT_LSH_NUM_BANDS,
    DEFAULT_LSH_NUM_PERM,
)
from .partition_similarity import (
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_HDBSCAN_SELECTION,
    DEFAULT_KMEANS_BATCH_RESEED_BELOW,
    DEFAULT_KMEANS_CLUSTERS,
    DEFAULT_KMEANS_FIT_ROWS,
    DEFAULT_KMEANS_MAX_PASSES,
    DEFAULT_KMEANS_RESTARTS,
    DEFAULT_KMEANS_SEED,
    DEFAULT_KMEANS_STALL_BATCHES,
    DEFAULT_KMEANS_START,
    DEFAULT_KMEANS_START_ROWS,
    DEFAULT_KMEANS_STOP_TOLERANCE,
    HDBSCAN_MIN_CLUSTER_SIZE_OPTION,
    HDBSCAN_MIN_SAMPLES_OPTION,
    HDBSCAN_SELECTION_CHOICES,
    HDBSCAN_SELECTION_OPTION,
    KMEANS_BATCH_RESEED_BELOW_OPTION,
    KMEANS_CLUSTERS_OPTION,
    KMEANS_FIT_ROWS_OPTION,
    KMEANS_RESTARTS_OPTION,
    KMEANS_SEED_OPTION,
    KMEANS_START_CHOICES,
    KMEANS_START_OPTION,
    KMEANS_START_ROWS_OPTION,
    MAX_PASSES_OPTION,
    STALL_BATCHES_OPTION,
    STOP_TOLERANCE_OPTION,
)
from .prefix_filter import PREFIX_FILTER_REPRESENTATIONS
from .sparse_similarity import (
    DEFAULT_SVD_RERANK_CANDIDATES,
    DEFAULT_SVD_RERANK_DIMENSIONS,
)

SIMILARITY_BACKENDS: tuple[str, ...] = (
    "sklearn",
    "sparse_dot_topn",
    "svd_rerank",
    "kmeans",
    "hdbscan",
    "lsh",
    "dense_brute",
    "hnsw",
)
"""The similarity backends a blocking run can select."""

DENSE_BRUTE_STORAGE_DTYPES: tuple[str, ...] = ("float16", "float32")
"""The `dense_brute` backend's `storage_dtype` option choices."""

TEXT_VIEWS: tuple[str, ...] = ("auto", "name", "tokens")
"""`auto` takes the representation's own text view."""

_DENSE_GATE = {
    "representation": tuple(sorted(DENSE_VOCABULARY_REPRESENTATIONS)),
    "backend": tuple(sorted(EXHAUSTIVE_SPARSE_BACKENDS)),
}

SETTINGS: tuple[dict[str, object], ...] = (
    {
        "name": "representation",
        "type": "str",
        "default": "tfidf",
        "help": "Clustering representation: tfidf, wordpiece, sentencepiece or sbert.",
    },
    {
        "name": "similarity_backend",
        "type": "str",
        "default": "sklearn",
        "choices": SIMILARITY_BACKENDS,
        "help": (
            "Similarity execution backend; kmeans and hdbscan partition the target "
            "index instead of scoring exhaustively, and hdbscan needs a dense "
            "representation."
        ),
    },
    {
        "name": "text_view",
        "type": "str",
        "default": "auto",
        "choices": TEXT_VIEWS,
        "help": "Text view the representation reads; auto takes the representation's own.",
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
        "help": "What a TF-IDF n-gram is a run of; word leaves a one-word name featureless.",
        "applies": {"representation": ("tfidf",)},
    },
    {
        "name": "sbert_model_name",
        "type": "str",
        "default": None,
        "help": (
            "A registry slug, hub checkpoint identifier or local checkpoint path; "
            "unset resolves one from the run's countries."
        ),
        "applies": {"representation": ("sbert",)},
    },
    {
        "name": "max_rows",
        "type": "int",
        "default": DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
        "help": (
            "Largest target side, rows per country, a dense-vocabulary "
            "representation may scan exhaustively before the run is refused; the "
            "largest scale benchmarked, not a proven ceiling."
        ),
        "applies": _DENSE_GATE,
    },
    {
        "name": "force",
        "type": "bool",
        "default": False,
        "help": "Run a combination the row gate would refuse, accepting its cost.",
        "applies": _DENSE_GATE,
    },
    {
        "name": "prefix_filter",
        "type": "bool",
        "default": False,
        "help": (
            "Prune candidate pairs before scoring by a prefix bound on the cosine "
            "the run keeps pairs by, sized from min_similarity; returns the "
            "exhaustive scan's candidates, measured slower than the scan itself."
        ),
        "applies": {
            "representation": tuple(sorted(PREFIX_FILTER_REPRESENTATIONS)),
            "backend": ("sklearn",),
        },
        "option": "prefix_filter",
    },
    {
        "name": "svd_candidates",
        "type": "int",
        "default": DEFAULT_SVD_RERANK_CANDIDATES,
        "help": "Candidates per source row kept from the projection for reranking.",
        "applies": {"backend": ("svd_rerank",)},
        "option": "svd_candidates",
    },
    {
        "name": "svd_dimensions",
        "type": "int",
        "default": DEFAULT_SVD_RERANK_DIMENSIONS,
        "help": "Dimensions of the reduced projection candidates are drawn from.",
        "applies": {"backend": ("svd_rerank",)},
        "option": "svd_dimensions",
    },
    {
        "name": "kmeans_clusters",
        "type": "int",
        "default": DEFAULT_KMEANS_CLUSTERS,
        "help": "Partitions the kmeans backend divides the target into.",
        "applies": {"backend": ("kmeans",)},
        "option": KMEANS_CLUSTERS_OPTION,
    },
    {
        "name": "kmeans_start",
        "type": "str",
        "default": DEFAULT_KMEANS_START,
        "choices": KMEANS_START_CHOICES,
        "help": (
            "How the kmeans fit picks its starting centroids from the start "
            "rows: spread picks each far from those already picked, at a cost "
            "growing with rows times clusters; random takes rows as they come."
        ),
        "applies": {"backend": ("kmeans",)},
        "option": KMEANS_START_OPTION,
    },
    {
        "name": "kmeans_start_rows",
        "type": "str",
        "default": DEFAULT_KMEANS_START_ROWS,
        "help": (
            "Target rows the kmeans start picks from: all, or a sample of that "
            "many, at least ten per cluster."
        ),
        "applies": {"backend": ("kmeans",)},
        "option": KMEANS_START_ROWS_OPTION,
    },
    {
        "name": "kmeans_fit_rows",
        "type": "str",
        "default": DEFAULT_KMEANS_FIT_ROWS,
        "help": (
            "Target rows each kmeans iteration reads: all for the full fit, or "
            "batches of that many for a mini-batch fit."
        ),
        "applies": {"backend": ("kmeans",)},
        "option": KMEANS_FIT_ROWS_OPTION,
    },
    {
        "name": "kmeans_restarts",
        "type": "int",
        "default": DEFAULT_KMEANS_RESTARTS,
        "help": "Independent kmeans fits, the tightest of which is kept.",
        "applies": {"backend": ("kmeans",)},
        "option": KMEANS_RESTARTS_OPTION,
    },
    {
        "name": "kmeans_batch_reseed_below",
        "type": "float",
        "default": DEFAULT_KMEANS_BATCH_RESEED_BELOW,
        "help": (
            "Fraction of the largest partition below which a mini-batch kmeans "
            "fit moves a partition's centroid to a fresh row."
        ),
        "applies": {"backend": ("kmeans",)},
        "option": KMEANS_BATCH_RESEED_BELOW_OPTION,
    },
    {
        "name": "max_passes",
        "type": "int",
        "default": DEFAULT_KMEANS_MAX_PASSES,
        "help": "Most passes a fit makes over the target before it stops.",
        "applies": {"backend": ("kmeans",)},
        "option": MAX_PASSES_OPTION,
    },
    {
        "name": "stop_tolerance",
        "type": "float",
        "default": DEFAULT_KMEANS_STOP_TOLERANCE,
        "help": "Centroid movement below which a fit stops early.",
        "applies": {"backend": ("kmeans",)},
        "option": STOP_TOLERANCE_OPTION,
    },
    {
        "name": "stall_batches",
        "type": "int",
        "default": DEFAULT_KMEANS_STALL_BATCHES,
        "help": "Batches in a row without improvement after which a mini-batch fit stops.",
        "applies": {"backend": ("kmeans",)},
        "option": STALL_BATCHES_OPTION,
    },
    {
        # Not `seed`, which a perturbed dataset's flag owns.
        "name": "backend_seed",
        "type": "int",
        "default": DEFAULT_KMEANS_SEED,
        "help": "Seed of the kmeans fit's start and sample, and of the lsh hash functions.",
        "applies": {"backend": ("kmeans", "lsh")},
        "option": KMEANS_SEED_OPTION,
    },
    {
        "name": "min_cluster_size",
        "type": "int",
        "default": DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
        "help": "Smallest partition the hdbscan backend keeps.",
        "applies": {"backend": ("hdbscan",)},
        "option": HDBSCAN_MIN_CLUSTER_SIZE_OPTION,
    },
    {
        "name": "hdbscan_min_samples",
        "type": "int",
        "default": DEFAULT_HDBSCAN_MIN_SAMPLES,
        "help": (
            "Neighbours a row needs to be a core point of an hdbscan cluster; "
            "unset is the minimum cluster size."
        ),
        "applies": {"backend": ("hdbscan",)},
        "option": HDBSCAN_MIN_SAMPLES_OPTION,
    },
    {
        "name": "hdbscan_selection",
        "type": "str",
        "default": DEFAULT_HDBSCAN_SELECTION,
        "choices": HDBSCAN_SELECTION_CHOICES,
        "help": "How hdbscan chooses clusters: eom the most stable, leaf the finest.",
        "applies": {"backend": ("hdbscan",)},
        "option": HDBSCAN_SELECTION_OPTION,
    },
    {
        "name": "num_perm",
        "type": "int",
        "default": DEFAULT_LSH_NUM_PERM,
        "help": "Length of the minhash signature the lsh backend hashes each row into.",
        "applies": {"backend": ("lsh",)},
        "option": "num_perm",
    },
    {
        "name": "num_bands",
        "type": "int",
        "default": DEFAULT_LSH_NUM_BANDS,
        "help": (
            "Bands the lsh backend splits a signature into; must divide num_perm, "
            "and more bands find more candidates."
        ),
        "applies": {"backend": ("lsh",)},
        "option": "num_bands",
    },
    {
        "name": "storage_dtype",
        "type": "str",
        "default": DEFAULT_DENSE_BRUTE_STORAGE_DTYPE,
        "choices": tuple(
            dict.fromkeys(DENSE_BRUTE_STORAGE_DTYPES + HNSW_STORAGE_DTYPES)
        ),
        "help": (
            "Precision a dense backend holds its target at: dense_brute takes "
            "float16 or float32 and converts each block to float32 for the "
            "multiply; hnsw takes float16 or int8."
        ),
        "applies": {"backend": ("dense_brute", "hnsw")},
        "option": "storage_dtype",
    },
    {
        "name": "block_bytes",
        "type": "int",
        "default": DEFAULT_DENSE_BRUTE_BLOCK_BYTES,
        "help": (
            "Target rows scanned per block by the dense_brute backend, sized "
            "in bytes so peak memory stays bounded by the block rather than "
            "the whole target."
        ),
        "applies": {"backend": ("dense_brute",)},
        "option": "block_bytes",
    },
    {
        "name": "connectivity",
        "type": "int",
        "default": DEFAULT_HNSW_CONNECTIVITY,
        "help": "hnsw graph fan-out per node (usearch Index(connectivity=...)).",
        "applies": {"backend": ("hnsw",)},
        "option": "connectivity",
    },
    {
        "name": "expansion_add",
        "type": "int",
        "default": DEFAULT_HNSW_EXPANSION_ADD,
        "help": "hnsw candidate list size while building the graph.",
        "applies": {"backend": ("hnsw",)},
        "option": "expansion_add",
    },
    {
        "name": "expansion_search",
        "type": "int",
        "default": DEFAULT_HNSW_EXPANSION_SEARCH,
        "help": "hnsw candidate list size while searching the built graph.",
        "applies": {"backend": ("hnsw",)},
        "option": "expansion_search",
    },
    {
        "name": "build_threads",
        "type": "int",
        "default": DEFAULT_HNSW_BUILD_THREADS,
        "help": (
            "Concurrent inserters building the hnsw graph; more are faster "
            "and vary the graph more between builds."
        ),
        "applies": {"backend": ("hnsw",)},
        "option": BUILD_THREADS_OPTION,
    },
)
"""Every setting this package's representations and backends read.

A backend setting names, as `option`, the `backend_options` key it fills;
every one that applies to a run is recorded with the run, given or defaulted.
"""
