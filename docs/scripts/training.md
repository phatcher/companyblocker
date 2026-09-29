# Training scripts

The scripts that train a tokenizer over cleansed names and derive the noise-word sets used to reduce noise in blocking keys.

- `train_tokenizer.py`: Trains a tokenizer, or sweeps hyperparameters to select one.
- `generate_noise_words.py`: Derives noise-word sets from a trained tokenizer's corpus.

# train_tokenizer.py

```
uv run python scripts/train_tokenizer.py --systems gb --tokenizer wordpiece
```

`--systems` names the systems whose cleansed names form the training corpus, and `--name-col` selects the column within them. The column is normalised before the corpus is written, lowercase with punctuation and diacritics normalised and nothing removed, so a column that keeps capitals trains on the text a blocking run hands the tokenizer; the tokenizer's metadata records the column and that preprocess profile. `--tokenizer` chooses between `wordpiece` and `sentencepiece`.

`--mode` selects between `train`, which produces one tokenizer from the arguments given, and `optimize`, which sweeps a preset's parameter grid and reports the results for a promotion decision. `--optimize-preset` names the grid, matching the trainer.

An optimize run with `--finalize` reuses every candidate already trained, and skips itself when its finalized result is newer than the corpus and every logged trial and selection over the logged trials still picks the recorded winner. `--reoptimize true` reselects and refines under the current code regardless, still reusing every trained candidate; `--force` discards them and retrains.

## Vocabulary arguments

`--vocab-size` and `--min-frequency` set the vocabulary the trainer targets. For SentencePiece, `--encoding` selects `unigram` or `bpe`, `--sp-character-coverage` sets how much of the observed character set the model must cover, and `--sp-byte-fallback` controls whether unseen characters decompose to bytes rather than becoming unknown tokens.

`--profile` names a global tokenizer profile, defaulting to `auto`. `--validation-fraction` holds back part of the corpus for evaluation during an optimize run.

## Corpus handling

The prepared training corpus and the trained model are written to the scope's working tree, `artifacts/tokenizers/work/<scope>/`; `--corpus-filename` names the corpus file there. Resampling the training corpus discards the existing one, so `--optimize-wipe` is required as an explicit acknowledgement before an optimize run will do it.

# generate_noise_words.py

```
uv run python scripts/generate_noise_words.py --system gb
```

Reads a prepared corpus and produces the token statistics and noise-word sets that downstream noise reduction consumes. `--system` targets one system code, or `global` for the cross-system profile named by `--profile`. `--systems` takes several, and `--pooling` chooses whether they are pooled by document `count` or weighted `equal`.

## Selection arguments

A token's eligibility comes from its document frequency and inverse document frequency: `--min-document-frequency-pct` sets the floor a token must reach to be considered, `--max-idf` sets an optional ceiling before bucketing, and `--min-token-length` excludes short tokens outright.

Three sets are produced at different aggressions, sized by `--strict-token-count`, `--balanced-token-count` and `--aggressive-token-count`, so a consumer picks a level rather than a threshold. `--seed-noise-words-path` supplies a hand-curated starting list that the derived sets extend.

Outputs are written to `--token-stats-out`, `--token-set-out` and `--noise-words-out`, under `artifacts/tokenizers` by default.

# Where the trained artifacts are used

A newly trained tokenizer reaches two consumers by different routes, and only one of them is immediate.

Blocking tokenizes on the fly from the promoted tokenizer artifact rather than reading the persisted `tokenized/` data layer, so promotion is enough for a blocking run to use the new tokenizer.

The `tokenize` stage calls `company_tokenize` during the run, so `data/<system>/tokenized/` records the tokenizer as it stood when that layer was last written. Promotion does not change what is already there. Anything computed from that layer, which is the training and analysis route rather than blocking, describes the older tokenizer until the affected system is re-tokenized. See the acquisition scripts document for the same property on the cleanse side.
