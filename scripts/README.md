# scripts

The repository's runnable scripts: pipeline stages, artifact producers, measurements, analyses and benchmarks. Each describes itself in its docstring and `--help`, and most offer `--dry-run`. What they share, the dry run, the overwrite report, argument errors, declared settings and how a run's inputs are named, is `cli_common.py`'s module docstring.

## Categories

Each script belongs to one category.

- **Pipeline entry points**: run a real pipeline stage against real data.
- **Repo gates**: check or regenerate a repository-level invariant.
- **Artifact producers**: write or promote an artifact, most of them a checked-in one; the only category that changes committed files.
- **Measurements**: answer one question, read-only; deleted once nothing still reads the answer.
- **Analysis and comparison**: exploratory and read-only.
- **Performance and profiling**: characterise the runtime or memory of an existing path.
- **Shared modules**: imported by the others, not run.

## Pipeline entry points

| Script | Purpose |
| --- | --- |
| `acquire_companies.py` | Acquire bulk company data for one or more systems. |
| `process_companies.py` | Run the processing stages: shard, canonical, match, cleanse, tokenize. |
| `pipeline_status.py` | Report how fresh each stage's output is. |
| `run_blocking.py` | Run a two-dataset blocking workflow, candidate generation through scoring. |
| `train_tokenizer.py` | Train or optimize a tokenizer. |
| `validate_clustering.py` | Run directional source-target validation. |
| `materialize_perturbations.py` | Materialize a perturbation profile as a dataset. |
| `materialize_public_benchmark.py` | Materialize a public two-table benchmark as two systems. |
| `compute_sif_word_probabilities.py` | Compute SIF word probabilities over one system's names. |
| `export_dataset_repo.py` | Copy the latest canonical, matched and prepare outputs into a dataset repo. |

## Repo gates

Repository-wide gates live in `tooling/`.

| Script | Purpose |
| --- | --- |
| `check_name_layer_identity.py` | Report whether each system's name layer carries per-row identity. |
| `check_stale_layout_generations.py` | Report a layer holding two layout generations of one partition. |
| `check_layer_path_literals.py` | Report hand-built `data/` layer paths, against the baseline `src/tests/workspace/test_layer_path_literals.py` pins. |
| `validate_catalog.py` | Validate the acquisition catalog. |
| `prune_artifacts.py` | Remove a keyed artifact, or stale temporary store directories, under `artifacts/store/`. |

## Artifact producers

| Script | Purpose |
| --- | --- |
| `generate_noise_words.py` | Generate TF-IDF token statistics and noise words from a tokenizer corpus. |
| `generate_iso20275_additions.py` | Propose ISO 20275 mappings per country, with a decision sheet and audit. |
| `generate_tokenizer_corpus_report.py` | Build and publish a scope's tokenizer and corpus report. |
| `generate_wikidata_benchmark_fixtures.py` | Generate the paired `.bz2`/`.gz` Wikidata benchmark fixtures. |
| `export_noise_word_candidates.py` | Export a scored per-system noise-word candidate list. |
| `promote_noise_words.py` | Promote corpus-derived noise words into `company_cleanse`. |
| `promote_noise_word_candidates.py` | Promote a noise-word candidate export into `company_tokenize`. |
| `promote_short_name_noise_words.py` | Promote short-name noise words into `company_cleanse`. |
| `archive_optimize_candidate.py` | Copy one optimize candidate into a durable, named location. |
| `build_wikidata_jurisdiction_closure.py` | Refresh the QID-to-country closure cache. |
| `extract_wikidata_fixture.py` | Create a small Wikidata JSONL fixture from a local dump. |
| `publish_reports.py` | Copy each country's report figures into `docs/reports/<country>/`. |
| `report_source_schemas.py` | Write each system's source-layer data dictionary under `docs/`. |

## Measurements

| Script | Question |
| --- | --- |
| `measure_gleif_jurisdiction_disparity.py` | Does GLEIF's pooled tokenizer show a per-jurisdiction quality disparity? |
| `measure_initialism_recall.py` | How often are real matched pairs initialism-shaped? |
| `measure_name_variant_recall.py` | What does a system's own name history recover? |
| `measure_prefix_suffix_divergence.py` | How often are real matched pairs prefix or suffix divergent? |
| `measure_short_name_recall.py` | What recall and precision does the short-name representation add? |
| `measure_pair_classifier_ceiling.py` | What is the pair classifier's ceiling on real name-variant pairs? |
| `measure_pooled_subword_encoder_ceiling.py` | What is the pooled-subword encoder's ceiling on the same pairs? |
| `measure_short_name_pattern_layers.py` | What does each short-name derivation layer recover, per language? |
| `measure_fasttext_alias_hit_rate.py` | What alias-pair hit rate does a pretrained fastText checkpoint reach? |

## Analysis and comparison

| Script | Purpose |
| --- | --- |
| `analyze_tokens.py` | Per-country and global token metrics. |
| `analyze_token_rarity.py` | The rare end of each corpus. |
| `analyze_token_zipf.py` | Zipf and OOV divergence per name tier. |
| `analyze_noise_layers.py` | What each noise layer removes, across corpora. |
| `analyze_vocab_shrinkage.py` | What a pooled vocabulary would drop per system. |
| `analyze_matches.py` | Match outcomes per source, target and country. |
| `analyze_gleif_entity_category.py` | GLEIF no-match recoverability by entity category. |
| `analyze_residual_pairs.py` | Four pictures over the pairs cleansing leaves unequal. |
| `analyze_cluster_shape.py` | Cluster sizes before and after a run's target-neighbour union. |
| `plot_pair_outcomes.py` | Which truth pairs each run of a pairing found. |
| `compare_blocking_strategies.py` | Run several blocking strategies for one pairing and compare them. |
| `report_strategy_comparison.py` | Build a pairing's strategy comparison from finished runs. |
| `aggregate_strategy_comparisons.py` | Combine every pairing's comparison into one scaling table. |
| `report_recall_curve.py` | A run's recall against comparisons spent, and a ranking of runs by it. |
| `remeasure_blocking_truth.py` | Re-measure a finished run's ground truth without re-running it. |
| `build_promotion_report.py` | Combine a validation run and a classifier's metrics into a promotion report. |
| `compare_tokenizer_candidates.py` | Compare two archived tokenizer candidates. |
| `compare_tokenizer_scopes.py` | Compare every scope's operational tokenizer. |
| `plot_optimize_elbow.py` | Plot an optimize run's fertility distance against vocabulary size. |
| `plot_country_global_overlay.py` | Overlay each scope's per-country fertility and unknown-token rate. |
| `profile_analysis_schema.py` | Profile a system's parquet schema for analysis. |
| `compare_wikidata_rust_wikisieve.py` | Diff a `wikisieve` run against a stored reference. |
| `iso_expand.py` | Per-country ISO 20275 coverage. |

## Performance and profiling

| Script | Purpose |
| --- | --- |
| `benchmark_analysis_loader.py` | Benchmark the analysis load modes. |
| `benchmark_validate_clustering.py` | Benchmark `validate_clustering.py`'s similarity backends. |
| `benchmark_wikidata_line_filter.py` | Benchmark Wikidata line filtering. |
| `profile_wikidata.py` | Profile the Wikidata bz2 read path. |
| `diagnostic_indexed_bzip2_scanner.py` | Scan a dump with `indexed_bzip2` to localise a crash. |

## PowerShell scripts

Windows-only, since they drive MSVC build tooling and 7-Zip directly.

| Script | Purpose |
| --- | --- |
| `build_wikisieve.ps1` | Run a cargo subcommand against `src/rust/wikisieve`. |
| `deploy_wikisieve.ps1` | Build and copy the `wikisieve` binary into `tools/bin/`. |
| `extract_wikidata_properties.ps1` | Build a property graph from the raw dump with `wdgrep`. |

## Shared modules

| Module | Purpose |
| --- | --- |
| `cli_common.py` | Arguments, dry runs and reporting shared by every script. |
| `pipeline_runner.py` | Per-stage execution for `process_companies.py`. |
| `wikidata_profile_common.py` | Helpers for the Wikidata profiling scripts. |
| `_bootstrap.py` | Put the repository's source roots on `sys.path`, once. |
| `_polars_threads.py` | Cap Polars' thread pool before anything imports Polars. |
