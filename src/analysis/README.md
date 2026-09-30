# analysis

Measurements and pictures over the pipeline's outputs: token behaviour per corpus, match outcomes, and what each blocking run found. Every module reads what another area already wrote and changes none of it, and every artifact lands under `artifacts/analysis/<analysis>/runs/<run_date>/`.

## Modules

Each module's docstring describes it, and each script's `--help` its arguments.

- Loading and profiling
    - [data_loader.py](data_loader.py): Column-pruned, filtered parquet reads.
    - [benchmark_loader.py](benchmark_loader.py): The load modes compared on speed and memory.
    - [system_discovery.py](system_discovery.py): Which systems have tokenized output.
    - [io_contract.py](io_contract.py): Column presence, null rates and dtypes per system (`scripts/profile_analysis_schema.py`).
    - [run_manifest.py](run_manifest.py): One manifest per run date, one entry per phase.
    - [report_layout.py](report_layout.py): A report run's `metrics/`, `viz/` and `reports/` directories.
- Tokens and vocabulary
    - [token_metrics.py](token_metrics.py): Per-system token metrics and reports.
    - [token_rarity.py](token_rarity.py): The rare end of each corpus against general-language frequency (`scripts/analyze_token_rarity.py`).
    - [token_zipf.py](token_zipf.py): Zipf and OOV divergence per system and name tier (`scripts/analyze_token_zipf.py`).
    - [noise_layers.py](noise_layers.py): What each noise layer removes, across corpora (`scripts/analyze_noise_layers.py`).
    - [vocab_shrinkage.py](vocab_shrinkage.py): What a pooled vocabulary would drop from each corpus (`scripts/analyze_vocab_shrinkage.py`).
    - [german_decompound.py](german_decompound.py): German compound splitting.
    - [sif_word_probabilities.py](sif_word_probabilities.py): The word probabilities SIF weights a name by.
- Matching and blocking outcomes
    - [match_metrics.py](match_metrics.py): Match-outcome KPIs, no-match triage and recoverability (`scripts/analyze_matches.py`).
    - [match_metrics_gleif.py](match_metrics_gleif.py): GLEIF's unrecoverable no-matches by entity category (`scripts/analyze_gleif_entity_category.py`).
    - [residual_pictures.py](residual_pictures.py): Four pictures over the pairs cleansing leaves unequal (`scripts/analyze_residual_pairs.py`).
    - [cluster_shape_pictures.py](cluster_shape_pictures.py): Cluster sizes before and after a run's target-neighbour union (`scripts/analyze_cluster_shape.py`).
    - [pair_outcomes.py](pair_outcomes.py): Which truth pairs each run found, read across a pairing's runs (`scripts/plot_pair_outcomes.py`).
    - [promotion_report.py](promotion_report.py): Classifier and blocking metrics in one row per strategy and operating point (`scripts/build_promotion_report.py`).
- [_cli_helper.py](_cli_helper.py): The settings this area's scripts read.

Two read-only measurement scripts over the matched layer sit beside these, `scripts/measure_initialism_recall.py` and `scripts/measure_prefix_suffix_divergence.py`, each described by its own docstring.

## Notebooks

- `notebooks/analyse_matches.ipynb`: One match-analysis scenario, interactively.
- `notebooks/analyse_residual_pairs.ipynb`: The residual pictures, drawn inline.
- `notebooks/analyse_pair_outcomes.ipynb`: Which pairs each run found.
- `notebooks/analyse_perturbation_operators.ipynb`: Every registered perturbation operator run over sample names.
- `notebooks/analyse_perturbation_profile.ipynb`: A perturbation profile's scenarios run over a sample, with what each step changed.
- `notebooks/analyse_corpus_zipf.ipynb`: One system's rank-frequency curves across the `raw`, `basic` and `cleansed` name tiers, through `token_zipf`.
- `notebooks/analyse_tfidf.ipynb`: Each tokenizer scope's token TF-IDF statistics, `global` included.
- `notebooks/analyse_tokens.ipynb`: Tokenizer optimize runs across systems: winning parameters and how the candidate search went.
- `notebooks/data_analyse.ipynb`: A finished `scripts/analyze_tokens.py` run read back: country and global token statistics and its summary report.

## Findings

- [token-zipf.md](../../docs/findings/token-zipf.md): What the Zipf runs showed across the live systems.
- [gleif_no_match_investigation.md](gleif_no_match_investigation.md): Why GLEIF names go unmatched.
