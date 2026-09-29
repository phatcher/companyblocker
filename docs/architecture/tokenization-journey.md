# Design Journey: Tokenization → Blocking Keys

Part of [CompanyBlocker Architecture](../architecture.md): detail behind [§5](../architecture.md#5-design-journey-tokenization--blocking-keys).

Tokenizer optimization is where [§1](../architecture.md#1-executive-summary)'s "don't assert, measure" pillar has been applied most extensively, including to the measurement approach itself.

## The training/optimize pipeline, briefly

`train_tokenizer.py` (**Train/Optimize** stage, README) trains WordPiece or SentencePiece tokenizers per system or globally. `--mode optimize` runs a grid search over `vocab_size` and `min_frequency`, scoring candidates against fertility, unknown-token-rate and token-count-distribution targets before `--finalize` promotes a winner. `generate_noise_words.py` separately computes TF-IDF statistics over the trained corpus and derives three noise-word exclusion profiles (`strict`, `balanced`, `aggressive`) that Tokenize can apply. Both are in active use across systems.

The training corpus is one canonical name per entity, never an entity's recorded name variants or historical name chain alongside it. An entity with several recorded variants would contribute the same tokens to a document-frequency count more than once, inflating the rare and distinctive tokens a hapax-legomena-based rarity analysis depends on. Name-variant and alias data is reserved for evaluation, which makes it a structural holdout population rather than one carved out by random sampling.

## Four defects found in the search methodology

`src/training/optimize_search_methodology.md` is a dated investigation log (2026-08-26, IE corpus) recording what the search got wrong and how each was fixed.

1. **The seed-independence bug.** Running five seeds is meant to buy independent train/validation splits for a noise estimate. The split assignment used a hand-rolled weak LCG formula with the seed folded in additively. On the real 817,766-row IE corpus, seed pair (42, 43) produced validation-set overlap of **76.4%** against an expected ~20% under independence, so seeds 42 to 46 were mostly duplicates rather than five independent samples. Fixed by replacing the formula with Polars' built-in hash (`pl.col("row_nr").hash(seed=seed)`), verified to bring overlap back to ~20% across multiple seed pairs. The first version of the investigation had to be redone: it had run against a stale binary, since Python does not hot-reload an already-imported module, and this was caught only by checking the corrected re-run's own seed-42/seed-43 overlap directly.
2. **A fixed discrete grid, not a continuous search.** `--vocab-sizes` and `--min-frequencies` define a lattice, 60 cells by default, and a two-phase pilot/expansion design covers every cell once before giving the frontier candidates a full five seeds. On the real IE run, 60 of 60 cells got at least one seed but only 33 (55%) were promoted to the five-seed expansion. The other 27 remain single-seed estimates with no variance bound, and can exclude a competitive candidate that drew one noisy pilot sample.
3. **A calculated noise floor.** For the 33 fully-seeded cells, measured seed-to-seed standard deviation of `fertility_distance` gives a minimum-distinguishable gap of roughly `2 × SE_diff ≈ 0.0013`. The IE winner (vocab=25,000) beat its nearest rejected competitors by 21 to 110 times that gap. The number is IE-corpus-specific and **does not transfer** to FR (16 times more rows), GLEIF, GB or OffeneRegister without re-measuring per system.
4. **A selection-metric bug.** The optimize sweep picked its winner on held-out validation-split `fertility_distance`, but the deployed tokenizer is retrained on the full corpus with no held-out split, since the pipeline retrains and tokenizes the same snapshot in the same run and serves no live stream of novel names. Held-out and in-sample fertility disagreed systematically, their minima sitting ~2,000 vocab points apart on the real IE data. `compute_candidate_selection_score` now optimizes the in-sample (train-split) fertility term, matching the operating condition; `unk_rate` and the generalization-delta term stayed validation-sourced, since those are about held-out behaviour. After the fix the sweep's reported metric and the full-corpus retrain's metric agreed to within ~8%, against the earlier approach being off by 11 times.

A follow-up densified sweep over a promising vocab=23,000 region, derived from measured local slope and noise floor, rejected every candidate in the band on `token_count_p95`, a hard deterministic gate rather than seed noise. The earlier "~3.8x better" claim for that band had been computed against raw metrics, bypassing the rejection-gate function, so it never tested whether those candidates survive the real pipeline.

## The still-open question

`fertility_target=1.25`, `token_count_p95_max=6.0` and the other gate thresholds trace back to their first commit as bare literals with no comment or discussion. Five parts of increasingly careful statistics have made the proxy metrics (fertility distance, token-count percentiles) more precise without validating that those proxies correlate with what they proxy for: downstream blocking and matching quality. Closing it needs `validate_clustering.py` run against already-archived candidates that differ meaningfully on these proxy metrics (`naive_v40000_mf1`, `v25000_mf8_prior_operational` and `v25000_mf1_train_split_fix_winner` span a useful range), checking whether real match and cluster quality moves across them. **In Progress.**

## TF-IDF noise-word profiles

Separately from vocabulary selection, `generate_noise_words.py` computes per-system TF-IDF statistics over the trained corpus and derives three named exclusion-strength profiles (`strict`, `balanced`, `aggressive`) applied at Tokenize time. Country and global noise-word sets differ because they are derived from different corpora, which the README states as a design expectation. A repository-wide threshold-profile contract unifying how those three are chosen and validated across systems is planned.

## The wordfreq/Zipf comparison

`analyze_token_rarity.py` extracts each system's rare-token tail (hapax legomena by default, `--max-document-frequency 1`) and, for systems with a mapped reference language, compares those tokens against `wordfreq`'s general-language frequency data, quantifying per system how much real corpus vocabulary a general-text tokenizer would fail to represent. `wordfreq`'s frequency data was frozen by its author in 2020, so the baseline predates large-scale LLM-generated web text. `global` is excluded, its corpus spanning every system's language with no single reference language to compare against. Measuring per system means a system whose company names do not have a distinct vocabulary shows up as such rather than being hidden by a global average.

## How this feeds Vectorize and Blocking

The TF-IDF representation bypasses this pipeline: it scores Cleanse's raw `name` column directly, with no trained tokenizer. The WordPiece and SentencePiece representations depend on Train/Optimize's promoted artifact rather than the Tokenize stage's persisted output, and Blocking tokenizes on the fly. The seed-independence fix, the in-sample selection fix and the open proxy-metric question therefore flow into candidate quality for two of `src/blocking`'s three deterministic baseline representations, and into the pooled-subword encoder built over these same trained tokens (see [DeepBlocker inspiration](deepblocker.md)). The proxy-metric question is the largest untested assumption under that half of the blocking-strategy comparison space ([§1](../architecture.md#1-executive-summary)).

## Memory, not CPU, constrains the optimize workload

Peak single-worker memory nearly tripled between two otherwise-identical IE optimize runs, 3.1GB to 8.9GB at `--max-workers 6`, because different vocab-size candidates landed concurrently on the same worker slot, while median system CPU stayed around 29%. Worker count and memory are therefore re-measured per system (FR, GB, GLEIF, OffeneRegister) rather than extrapolated linearly from one system's measurement, the same reasoning as the noise floor above applied to infrastructure sizing.
