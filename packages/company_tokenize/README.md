# company_tokenize

Tokenizers for company names: train WordPiece or SentencePiece on a prepared name corpus, judge candidates in an optimize run, tokenize names with the result, and derive the noise words a name loses before it is tokenized. Building the corpus, choosing its files and deciding where a tokenizer is kept are the caller's; the package takes paths already resolved.

## Training and tokenizing

```python
from pathlib import Path

import polars as pl

from company_tokenize import tokenize_name_dataframe, train_wordpiece

train_wordpiece(corpus_path=Path("training_corpus.parquet"), tokenizer_path=Path("wordpiece/model.json"))
names = pl.DataFrame({"name": ["acme holdings", "globex corporation"]})
tokens = tokenize_name_dataframe(names, name_col="name", tokenizer_path=Path("wordpiece/model.json"))
```

Training is a greedy approximation and does not repeat exactly: two runs on one corpus differ in a handful of words. Keep the model that produced a result rather than expecting to train it again, and compare two vocabularies by their words, not their ids. SentencePiece's `model_type`, `character_coverage` and `byte_fallback` change what fertility and unknown-token rate mean, so a comparison holds them fixed.

Tokenization removes no words unless given a noise-word profile, and returns variable-length token lists with no padding.

## Noise words

Noise words are frequent words that carry little identity, derived from TF-IDF statistics over a training corpus and removed before tokenizing when asked. A set is resolved from an explicit list, then a profile of `none`, then a file, then `company_cleanse`'s packaged default, which is itself this package's output, checked in once reviewed. Lists can be pooled across systems by row share (`count`) or with each system weighted equally (`equal`).

## API

Each module's docstring gives its rules. Everything below imports from the package root.

- `training.py`: `train_wordpiece`, `train_sentencepiece`, `train_tokenizer_with_trainer`, `TrainerOptions`, `validate_trainer_options`, `SUPPORTED_TRAINERS`, `estimate_wordpiece_vocab_size`, `trainer_params_payload`, `normalize_optimize_min_frequencies_for_trainer`, `load_tokenizer_encoder`, `resolve_tokenizer_path_for_trainer`, `resolve_metadata_path_for_trainer`, `resolve_candidate_model_suffix_for_trainer`, `compute_corpus_content_hash`, and the optimize layout: `OptimizeSweepPaths`, `resolve_optimize_dir`, `resolve_optimize_sweep_paths`, `resolve_optimize_splits_dir`, `resolve_active_optimize_sweep_dir`, `resolve_optimize_canary_pointer_path`.
- `tokenization.py`: `tokenize_name_dataframe`, `tokenize_name_dataframe_dual`, `tokenize_name_dataframe_multi`, `tokenize_name_dataframe_with_request`. Names in memory.
- `ops.py`: `tokenize_name`, `tokenize_name_dual`, `tokenize_name_multi`, `tokenize_name_with_request`, `TokenizeFilesRequest`. Named parquet files.
- `tokenization_params.py`: `TokenizationRequest`, `TokenizerCalculationSpec`, `compile_tokenization_request`, `coerce_tokenization_request`.
- `vocabulary.py`: `load_tokenizer_vocabulary`, `TokenizerVocabulary`. The vocabulary and noise words as one unit.
- `tfidf.py`: `compute_corpus_token_tfidf_stats`, `compute_tokenizer_token_tfidf_stats`, `normalize_tfidf_use_case`, `expected_tfidf_namespace`, `validate_tfidf_namespace`, `select_stopword_candidates`, `select_rare_token_candidates`, `cumulative_document_frequency_mass_cutoff`, `count_tokens_above_document_frequency_pct`, `resolve_noise_words`, `trim_name_before_tokenization`, `trim_token_column`, `pool_token_tfidf_stats`, `SUPPORTED_POOLING_RULES`, `normalize_pooling_rule`, `infer_document_count_from_tfidf_stats`, `trim_pooled_corpus_by_system`.
- `noise_export.py`: `select_noise_word_candidates`, `build_noise_word_candidates_payload`, `validate_noise_word_candidates_payload`, `summarize_noise_word_candidates`, `DEFAULT_MAX_NOISE_WORD_CANDIDATES`. A scored candidate list for a consumer choosing its own cutoff.
- `optimize.py`: `candidate_key`, `is_candidate_excluded`, `scoped_history_for_trainer`, `build_optimize_canary_payload`, `build_optimize_summary_payload`, `compute_optimize_grid_hash`. How an optimize run judges and records candidates.
- `contracts.py`: `OptimizeCandidatePolicy`, `OptimizeCandidateResult`, `TrainingTarget`.
- `manifest.py`: `build_run_manifest`, `validate_run_manifest`, `RUN_MANIFEST_MANDATORY_FIELDS`, `RUN_MANIFEST_OPTIONAL_FIELDS`, `RUN_MANIFEST_MODES`, `load_run_manifest_if_present`, `validate_dual_tokenizer_artifacts`, `validate_dual_tokenizer_coherence`, `DUAL_TOKENIZER_ROLES`.
- `paths.py`: `TOKENIZER_IDS`, `TokenizerFields`, `tokenizer_id`, `tokenizer_directory`, `tokenizer_directory_files`, `TokenizerDirectoryFiles`, `scope_directory_files`, `ScopeDirectoryFiles`, `promoted_candidate_key`, `write_candidate_directory`, `resolve_tokenizer_paths`, `resolve_country_tokenizer_paths`, `resolve_global_tokenizer_paths`, `TokenizerArtifactPaths`, `TRAINING_CORPUS_FILENAME`, `to_portable_path_str`.
- `candidate_archive.py`: `resolve_archive_dir`, `read_archive_index`, `write_archive_index`, `resolve_candidate_model_path`, `resolve_archive_label`, `resolve_archived_tokenizer_path`, `ArchivedTokenizerSelection`, `archive_optimize_candidate`, `archive_trained_tokenizer`, `archive_promoted_tokenizer`, `archive_naive_tokenizer`, `compare_archive_entries`, `attach_report_to_archive_entry`.
