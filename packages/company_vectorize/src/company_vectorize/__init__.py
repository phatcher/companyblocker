from .backends import HashVectorizer
from .base import EmbeddedRow, MultiEmbeddedRow, Vectorizer, l2_normalize
from .blocking_text_prep import (
    DEFAULT_BLOCKING_TEXT_SEPARATOR,
    compose_blocking_text,
)
from .clustering_contract import (
    ClusteringStrategy,
    EncoderHandle,
    EncoderTargetIndexBuildSettings,
    SbertTargetIndexBuildSettings,
    SentencepieceTargetIndexBuildSettings,
    SimilarityBackendIndex,
    TargetClusteringIndex,
    TargetIndexBuildSettings,
    TargetIndexStorableParts,
    TfidfTargetIndexBuildSettings,
    WordpieceTargetIndexBuildSettings,
)
from .clustering_factory import (
    resolve_clustering_strategy,
    resolve_target_index_build_settings,
)
from .clustering_metrics import (
    compute_cluster_shape_metrics,
    compute_directional_coverage,
)
from .dense_brute_similarity import (
    DEFAULT_DENSE_BRUTE_BLOCK_BYTES,
    DEFAULT_DENSE_BRUTE_STORAGE_DTYPE,
    TargetDenseBruteIndex,
    build_target_dense_brute_index,
)
from .dense_target_memory_gate import (
    SklearnDenseTargetMemoryError,
    ensure_sklearn_dense_target_memory_supported,
    estimate_sklearn_dense_target_memory_bytes,
)
from .dense_vocabulary_gate import (
    DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    FORCE_OPTION,
    MAX_ROWS_OPTION,
    DenseVocabularyScaleError,
)
from .duckdb_sql import (
    create_exact_knn_query_sql,
    create_table_sql_multi,
    create_upsert_sql_multi,
)
from .encoder_strategy import EncoderClusteringStrategy
from .graph_cluster import build_connected_components
from .hnsw_similarity import (
    DEFAULT_HNSW_CONNECTIVITY,
    DEFAULT_HNSW_EXPANSION_ADD,
    DEFAULT_HNSW_EXPANSION_SEARCH,
    DEFAULT_HNSW_STORAGE_DTYPE,
    HNSW_STORAGE_DTYPES,
    TargetHnswIndex,
    build_target_hnsw_index,
    hnsw_build_settings,
)
from .lsh_similarity import (
    DEFAULT_LSH_NUM_BANDS,
    DEFAULT_LSH_NUM_PERM,
    DEFAULT_LSH_SEED,
    MinHashLshParams,
    TargetLshIndex,
    build_target_lsh_index,
)
from .name_variant_index import expand_target_frame_with_name_variants
from .partition_similarity import (
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_KMEANS_CLUSTERS,
    TargetPartitionIndex,
    build_target_partition_index,
    partition_cluster_shape_frame,
)
from .pgvector import (
    create_ann_query_sql,
    create_extension_sql,
    create_hnsw_index_sql,
    create_table_sql,
    create_upsert_sql,
    vector_literal,
)
from .pipeline import vectorize_records, vectorize_records_multi
from .prefix_filter import (
    PrefixFilterIndex,
    build_prefix_filter_index,
)
from .sbert_model_registry import (
    SbertModelEntry,
    list_sbert_models,
    resolve_sbert_model_name,
    sbert_models_for_language,
)
from .sbert_pooling_gate import (
    UntrainedPoolingError,
    checkpoint_pooling_status,
    ensure_sentence_embedding_checkpoint,
)
from .sbert_strategy import DEFAULT_SBERT_MODEL_NAME, SbertClusteringStrategy
from .sparse_similarity import (
    TargetSimilarityBackendIndex,
    TargetSvdRerankIndex,
    build_target_nearest_neighbors,
)
from .tfidf_cluster import (
    build_clustering_text_view,
)

__all__ = [
    "DEFAULT_BLOCKING_TEXT_SEPARATOR",
    "DEFAULT_DENSE_BRUTE_BLOCK_BYTES",
    "DEFAULT_DENSE_BRUTE_STORAGE_DTYPE",
    "DEFAULT_DENSE_VOCABULARY_MAX_ROWS",
    "DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE",
    "DEFAULT_HNSW_CONNECTIVITY",
    "DEFAULT_HNSW_EXPANSION_ADD",
    "DEFAULT_HNSW_EXPANSION_SEARCH",
    "DEFAULT_HNSW_STORAGE_DTYPE",
    "DEFAULT_KMEANS_CLUSTERS",
    "DEFAULT_LSH_NUM_BANDS",
    "DEFAULT_LSH_NUM_PERM",
    "DEFAULT_LSH_SEED",
    "DEFAULT_SBERT_MODEL_NAME",
    "FORCE_OPTION",
    "HNSW_STORAGE_DTYPES",
    "MAX_ROWS_OPTION",
    "ClusteringStrategy",
    "DenseVocabularyScaleError",
    "EmbeddedRow",
    "EncoderClusteringStrategy",
    "EncoderHandle",
    "EncoderTargetIndexBuildSettings",
    "HashVectorizer",
    "MinHashLshParams",
    "MultiEmbeddedRow",
    "PrefixFilterIndex",
    "SbertClusteringStrategy",
    "SbertModelEntry",
    "SbertTargetIndexBuildSettings",
    "SentencepieceTargetIndexBuildSettings",
    "SimilarityBackendIndex",
    "SklearnDenseTargetMemoryError",
    "TargetClusteringIndex",
    "TargetDenseBruteIndex",
    "TargetHnswIndex",
    "TargetIndexBuildSettings",
    "TargetIndexStorableParts",
    "TargetLshIndex",
    "TargetPartitionIndex",
    "TargetSimilarityBackendIndex",
    "TargetSvdRerankIndex",
    "TfidfTargetIndexBuildSettings",
    "UntrainedPoolingError",
    "Vectorizer",
    "WordpieceTargetIndexBuildSettings",
    "build_clustering_text_view",
    "build_connected_components",
    "build_prefix_filter_index",
    "build_target_dense_brute_index",
    "build_target_hnsw_index",
    "build_target_lsh_index",
    "build_target_nearest_neighbors",
    "build_target_partition_index",
    "checkpoint_pooling_status",
    "compose_blocking_text",
    "compute_cluster_shape_metrics",
    "compute_directional_coverage",
    "create_ann_query_sql",
    "create_exact_knn_query_sql",
    "create_extension_sql",
    "create_hnsw_index_sql",
    "create_table_sql",
    "create_table_sql_multi",
    "create_upsert_sql",
    "create_upsert_sql_multi",
    "ensure_sentence_embedding_checkpoint",
    "ensure_sklearn_dense_target_memory_supported",
    "estimate_sklearn_dense_target_memory_bytes",
    "expand_target_frame_with_name_variants",
    "hnsw_build_settings",
    "l2_normalize",
    "list_sbert_models",
    "partition_cluster_shape_frame",
    "resolve_clustering_strategy",
    "resolve_sbert_model_name",
    "resolve_target_index_build_settings",
    "sbert_models_for_language",
    "vector_literal",
    "vectorize_records",
    "vectorize_records_multi",
]
