# Blocking scripts

The scripts that generate candidate pairs for a source and target pairing and compare one strategy against another.

- `run_blocking.py`: Executes one blocking run for a source and target pairing and writes a run directory.
- `compare_blocking_strategies.py`: Runs several representations over the same pairing and reports them side by side.

# run_blocking.py

Runs the two-dataset blocking workflow, meaning candidate generation, scoring and clustering, for one source and target pair.

```
uv run python scripts/run_blocking.py --source gleif --target ie --representation tfidf
```

Only the two system arguments are required. Everything else defaults, so the command above is the same as spelling out the defaults it relies on:

```
uv run python scripts/run_blocking.py \
  --source gleif --target ie \
  --representation tfidf \
  --similarity-backend sklearn \
  --text-view auto \
  --ground-truth true
```

`--source` and `--target` name the pairing, and the direction matches the match stage's: the multi-hive system is the source, the registry is the target. `--countries` narrows the run to particular hive partitions.

`--dry-run` resolves the source and target datasets, the strategy and the output directory, reports them, and stops without executing the run.

## The text that is scored

Three arguments decide the text both sides are compared on, and the same value applies to the source and the target.

- `--name-transform`: Which form of the name is scored: `cleanse` (default), `short_name`, `acronym`, or `identity` for a deliberate raw-name run.
- `--cleanse-profile`: The `company_cleanse` profile the name forms are derived under.
- `--preprocess-profile`: What is removed from the scored name before it is tokenized or vectorized. `default` removes the company type and keeps noise words, `default|+noise_words:<level>` also removes noise words at that level (`strict`, `balanced` or `aggressive`), and `default|-company_type` keeps the company type. Every profile lowercases and normalises punctuation and diacritics.

The derived forms are stored per system and country under the two profiles, so a second run under another backend derives nothing and a new profile derives only its own.

## Strategies

A strategy is a pair of choices: a **representation**, which decides how a name becomes a vector, and a **similarity backend**, which decides how neighbours are found in that space. They are chosen independently, but not every combination is permitted.

Five representations are available:

- `tfidf`: N-gram term weighting over the name, character n-grams within word boundaries by default (`tfidf.analyzer`). The default representation, and the deterministic baseline the others are measured against.
- `wordpiece`: The trained WordPiece tokenizer's subword units.
- `sentencepiece`: The trained SentencePiece tokenizer's subword units.
- `sbert`: Dense sentence embeddings. `--sbert-model` takes a registry slug, a hub checkpoint identifier or a local checkpoint path; omitted, it resolves a registered monolingual checkpoint matching every country given, when exactly one matches, and otherwise the pretrained English default.
- `encoder`: Dense vectors from an encoder the caller supplies. `--encoder fasttext` is the one `run_blocking.py` offers: a pretrained fastText checkpoint, each name the unweighted mean of its words' vectors. The checkpoint is the registered one matching every country given, or `--fasttext-checkpoint <slug>`, read from the pretrained-vectors folder under its registered checksum; a run whose file is absent is refused with the command that downloads it. The run records the encoder as `fasttext:<slug>:<checksum>`, so runs differing only in checkpoint are separate runs. `--encoder` is required with this representation and refused with any other. A library caller passes the loaded encoder as `BlockingRunConfig.encoder` and its name as `BlockingStrategyConfig.encoder_name`.

`wordpiece` and `sentencepiece` are the dense-vocabulary representations, and are the pair the row-count gate applies to.

Five similarity backends are available:

- `sklearn`: Exhaustive scoring. The default.
- `sparse_dot_topn`: Exhaustive scoring through a sparse top-n kernel.
- `svd_rerank`: Reduced-dimension retrieval followed by a rerank.
- `kmeans`: Partitions the target index and routes each source row to one partition rather than scoring exhaustively. Tuned with `--additional-args backend.kmeans_clusters=<int>` and the fit's other settings, which `company_vectorize`'s README lists; `backend.kmeans_fit_rows=<int>` below the target size fits in mini-batches.
- `hdbscan`: The same partitioning approach with density-based clusters, tuned with `--additional-args backend.min_cluster_size=<int>`, `backend.hdbscan_min_samples=<int>` and `backend.hdbscan_selection=eom|leaf`.

`hdbscan` requires a dense representation, so it runs against `sbert` only: a sparse representation raises a `ValueError` rather than degrading.

`--max-rows` bounds the largest target side, in rows per country, that a dense-vocabulary representation may run against an exhaustive backend before the run is refused. Its default is the largest scale that combination has actually been benchmarked at rather than a proven safe ceiling, so raising it belongs behind a measurement run. It does not apply to `tfidf` or `sbert`, nor to the `svd_rerank`, `kmeans` and `hdbscan` backends. `--force true` runs a combination the gate would refuse, accepting its runtime and memory cost.

## Candidate window

Three arguments bound what survives scoring: `--top-k` neighbours per source record, `--min-similarity` as a cosine floor, and `--max-candidates-per-source` as a cap applied after thresholding. `--source-chunk-size` sets the scoring batch size and the progress checkpoint interval.

A run directory records the computed scores and not the arguments that produced them, so a figure that depends on the window cannot be recovered from the directory afterwards. Two runs are comparable only if both were scored over the same window.

## Ground truth

`--ground-truth true` (the default) requires the source dataset to resolve a layer carrying its truth column -- `match_uri` on the `matched` layer under the default rule, or whichever column `--match-col` names. `--ground-truth false` reads the source from its latest canonical snapshot and produces no truth scoring, so a run made that way carries candidates but no precision or recall; its `summary.json` records `evaluation_skipped: true` so that is visible after the fact, rather than only implied by an absent `countries` block.

A row's cross-system match is resolved by walking its `source_uri` chain one hop at a time until it reaches a row whose URI carries a system scheme, then looking that up in the matched layer. Depth follows the lineage: an entity is one hop, a name row two, a perturbed alias three. Nothing is read from a cached column on the sidecar, so a derived dataset stays valid when its source system is re-matched.

## Tuning arguments and how they group

`--additional-args` takes `stage.key=value` pairs, grouped by a prefix naming the part of the run each belongs to, so one flag surface covers every representation without adding a flag per combination:

- `tfidf.ngram_min`, `tfidf.ngram_max`: The n-gram range, for the `tfidf` representation.
- `tfidf.analyzer`: What an n-gram is a run of: `char_wb` (default), `char` or `word`. Under `word` with the default range a one-word name has no features and can match only through the exact-name fast path.
- `tokenizer.name`, `tokenizer.scope`, `tokenizer.profile`, `tokenizer.path`: Which trained tokenizer the subword representations use, named by its label for the scope or by a file path.
- `pruning.max_candidates_per_target`, `pruning.candidate_similarity_ratio`: Pruning applied from the target side, complementing the per-source window above.
- `metric.f_beta`: The beta weight the evaluation is scored with.
- `backend.<key>`: Accumulates into the backend's own options, which is how `kmeans_clusters` and `min_cluster_size` reach the partitioning backends. Every declared setting of the run's backend is recorded in the run's identity whether given or defaulted, so a changed default never gives a new run the key of one made under the old.

## Where a run is written

The source reads `<system>/matched` under the data root and the target its latest dated snapshot under `<system>/canonical`; no run reads a cleansed layer, since it derives every name form itself from the raw `name`; `--data-dir`, `--output-dir` and `--temp-dir` move the three workspace roots and are the only paths the script takes. A run lands at `blocking/data/<target>/<kind>/<source>/<representation>/<key>` under the artifacts root, holding `matched_edges.parquet`, `pair_truth_eval_detail.parquet`, `clusters.parquet`, `summary.json`, `manifest.json` and the other per-run artifacts.

The config hash identifies the **configuration**. It is necessary to tell two configurations of the same pairing apart, and it is deliberately the same across different pairings that share a configuration, so the same identifier appears under more than one pairing. It carries no information about whether a run is current or whether it produced anything: identity, freshness and outcome are three separate questions. Read freshness from `pipeline_status.py`, and outcome from the run's own `summary.json`, whose `candidate_pair_count` is zero for a run that completed without producing candidates.

# compare_blocking_strategies.py

Runs several representations over one pairing and reports them side by side, so a strategy choice is made against a measurement rather than in isolation.

```
uv run python scripts/compare_blocking_strategies.py --source gleif --target gb --representations tfidf wordpiece sentencepiece
```

`--representations` accepts `tfidf`, `wordpiece`, `sentencepiece` and `sbert`, running each as its own leg over the same inputs. The pairing, candidate-window and backend arguments match `run_blocking.py`'s.

`--prefix-filter true` turns on the candidate prefilter for the sparse representations on `sklearn`, exact for the cosine it scores and off by default because the compiled scan is faster, `--kmeans-clusters` and `--min-cluster-size` are promoted to top-level flags for the partitioning backends, and `--runtime-budget-seconds` bounds the whole comparison.

A leg that a gate declines to run is recorded as a failed cell rather than aborting the comparison, so one refused combination does not cost the results of the others.
