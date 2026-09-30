# company_vectorize

Candidate generation for company-name matching: turn a target's names into an index once, then score each source name against it and keep the nearest targets. The package knows nothing of the repository using it and holds no locations: where an index is kept is the caller's.

## Representations

A representation is how a name becomes a vector. `resolve_clustering_strategy(representation)` returns its `ClusteringStrategy`, and `resolve_target_index_build_settings()` its settings.

| Representation | Vector | Module |
| --- | --- | --- |
| `tfidf` | sparse tf-idf over character n-grams by default | `tfidf_strategy.py` |
| `wordpiece`, `sentencepiece` | sparse tf-idf over a trained tokenizer's subwords | `token_list_strategy.py` |
| `sbert` | dense sentence-transformer embedding, checkpoint named by registry slug or path | `sbert_strategy.py` |
| `encoder` | dense embedding from any encoder handle the caller already holds | `encoder_strategy.py` |

Every strategy builds the target index once and scores source rows a chunk at a time. `target_index_storable_parts()` and `rebuild_target_index()` hand back a built index as arrays and frames and rebuild one that scores identically, so a caller can persist it.

## Similarity backends

| Backend | Target matrix | Search | Module |
| --- | --- | --- | --- |
| `sklearn` | sparse or dense | exact, scans every target | `sparse_similarity.py` |
| `sparse_dot_topn` | sparse | exact, scans every target | `sparse_similarity.py` |
| `svd_rerank` | sparse | scans every target in a reduced space; can miss a true neighbour | `sparse_similarity.py` |
| `dense_brute` | dense | exact, one block of targets at a time | `dense_brute_similarity.py` |
| `kmeans`, `hdbscan` | `kmeans` either, `hdbscan` dense | approximate: scores only the partition a source row is routed to | `partition_similarity.py` |
| `lsh` | sparse | approximate: MinHash buckets, scored by estimated Jaccard, not cosine | `lsh_similarity.py` |
| `hnsw` | dense | approximate: `usearch` graph, saveable and reopened memory-mapped | `hnsw_similarity.py` |

An exhaustive scan is affordable at `ie`'s scale (about 820,000 targets) and not at `gb`'s or `fr`'s; `lsh` and `kmeans` are the sub-linear choices for a sparse representation, `hnsw` for a dense one. Each backend's options pass through `backend_options`, declared with their defaults in `_cli_helper.py`. `prefix_filter.py` adds an exact L2AP prune to `sklearn` over a sparse representation.

## Known Issues

- Two gates refuse a run rather than let it fail slowly: `dense_vocabulary_gate.py` refuses `wordpiece`/`sentencepiece` on an exhaustive backend past `DEFAULT_DENSE_VOCABULARY_MAX_ROWS` targets, and `dense_target_memory_gate.py` refuses `sklearn` on a dense target larger than the memory the caller reports.
- `hdbscan` routes a source row to the nearest cluster centroid, not by density membership, and target rows it labels noise are never reachable.
- `lsh` scores are Jaccard estimates, not comparable with the cosine every other backend reports; a shortlist needs an exact rerank before the two are read together.
- Neither `kmeans` nor `hdbscan` caps partition size; `partition_cluster_shape_frame()` reports the imbalance.
- `HashVectorizer`, `pgvector.py` and `duckdb_sql.py` are scaffolding: a deterministic test vectorizer and SQL templates, with execution left to the caller.

## Comparison Protocol

`comparison_protocol.py` fixes what every strategy comparison runs: the slices (`gleif -> gb` and `gleif -> ie`), a runtime budget per cell, the mandatory suite of sub-linear cells a comparison must complete, and the reference suite of exact cells their recall loss is priced against. `scripts/compare_blocking_strategies.py` enforces it.

```bash
# The mandatory suite over the fixed slices.
uv run python scripts/compare_blocking_strategies.py

# The reference suite instead, run once per slice.
uv run python scripts/compare_blocking_strategies.py --reference-column
```

## API

Everything below imports from the package root. Each module's opening docstring or comment gives its rules.

- `clustering_contract.py`: `ClusteringStrategy`, `TargetClusteringIndex`, `TargetIndexStorableParts`, `SimilarityBackendIndex`, `EncoderHandle`, and the build settings `TargetIndexBuildSettings`, `TfidfTargetIndexBuildSettings`, `WordpieceTargetIndexBuildSettings`, `SentencepieceTargetIndexBuildSettings`, `SbertTargetIndexBuildSettings`, `EncoderTargetIndexBuildSettings`. The strategy protocol and its persist-and-rehydrate pair.
- `clustering_factory.py`: `resolve_clustering_strategy`, `resolve_target_index_build_settings`.
- `sbert_strategy.py`: `SbertClusteringStrategy`, `DEFAULT_SBERT_MODEL_NAME`. `encoder_strategy.py`: `EncoderClusteringStrategy`.
- `sbert_model_registry.py`: `SbertModelEntry`, `resolve_sbert_model_name`, `list_sbert_models`, `sbert_models_for_language`. Checkpoints named by slug.
- `sbert_pooling_gate.py`: `ensure_sentence_embedding_checkpoint`, `checkpoint_pooling_status`, `UntrainedPoolingError`. Refusing a checkpoint with no trained pooling.
- `sparse_similarity.py`: `TargetSimilarityBackendIndex`, `TargetSvdRerankIndex`, `build_target_nearest_neighbors`.
- `dense_brute_similarity.py`: `TargetDenseBruteIndex`, `build_target_dense_brute_index`, `DEFAULT_DENSE_BRUTE_BLOCK_BYTES`, `DEFAULT_DENSE_BRUTE_STORAGE_DTYPE`.
- `partition_similarity.py`: `TargetPartitionIndex`, `build_target_partition_index`, `partition_cluster_shape_frame`, `DEFAULT_KMEANS_CLUSTERS`, `DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE`.
- `lsh_similarity.py`: `TargetLshIndex`, `build_target_lsh_index`, `MinHashLshParams`, `DEFAULT_LSH_NUM_PERM`, `DEFAULT_LSH_NUM_BANDS`, `DEFAULT_LSH_SEED`.
- `hnsw_similarity.py`: `TargetHnswIndex`, `build_target_hnsw_index`, `hnsw_build_settings`, `HNSW_STORAGE_DTYPES`, `DEFAULT_HNSW_STORAGE_DTYPE`, `DEFAULT_HNSW_CONNECTIVITY`, `DEFAULT_HNSW_EXPANSION_ADD`, `DEFAULT_HNSW_EXPANSION_SEARCH`.
- `prefix_filter.py`: `PrefixFilterIndex`, `build_prefix_filter_index`.
- `dense_vocabulary_gate.py`: `DenseVocabularyScaleError`, `DEFAULT_DENSE_VOCABULARY_MAX_ROWS`, `FORCE_OPTION`, `MAX_ROWS_OPTION`.
- `dense_target_memory_gate.py`: `SklearnDenseTargetMemoryError`, `ensure_sklearn_dense_target_memory_supported`, `estimate_sklearn_dense_target_memory_bytes`.
- `name_variant_index.py`: `expand_target_frame_with_name_variants`. Known name variants indexed as extra keys for the same target.
- `blocking_text_prep.py`: `compose_blocking_text`, `DEFAULT_BLOCKING_TEXT_SEPARATOR`. Several columns into one blocking-text column.
- `tfidf_cluster.py`: `build_clustering_text_view`. `graph_cluster.py`: `build_connected_components`. `clustering_metrics.py`: `compute_cluster_shape_metrics`, `compute_directional_coverage`.
- `base.py`: `Vectorizer`, `EmbeddedRow`, `MultiEmbeddedRow`, `l2_normalize`. `backends.py`: `HashVectorizer`. `pipeline.py`: `vectorize_records`, `vectorize_records_multi`.
- `pgvector.py`: `create_extension_sql`, `create_table_sql`, `create_upsert_sql`, `create_hnsw_index_sql`, `create_ann_query_sql`, `vector_literal`. `duckdb_sql.py`: `create_table_sql_multi`, `create_upsert_sql_multi`, `create_exact_knn_query_sql`.
