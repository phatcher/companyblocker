# DeepBlocker Inspiration

Part of [CompanyBlocker Architecture](../architecture.md): detail behind [§4](../architecture.md#4-related-systems-deepblocker-and-pyjedai).

## What DeepBlocker does

DeepBlocker's paper (Table 3) evaluates eight representative deep-learning blocking solutions. All eight share one pipeline: concatenate a tuple's attribute values into a single string, embed each word of that string (pretrained fastText vectors for six solutions, BPE subwords for SBERT and Trans-encoder), combine the word vectors into one tuple vector, and retrieve top-K candidates by cosine similarity over the tuple vectors, in place of hand-built blocking keys (shared token, shared prefix, phonetic code). They differ in the combining step, how the word vectors become one tuple vector; the word vectors themselves are an input every solution needs before it starts.

The concatenation is not attribute-aware: the paper's example turns three attributes into `"Daniel Smith LA 18"`. This repo's single `name` field is already one string, so it enters that pipeline unchanged.

## The eight solutions

The paper's design space (Table 1) has three modules: Word Embedding (word or character granularity, pretrained or learned), Tuple Embedding (aggregation, or self-supervision through self-reproduction, cross-tuple training, triplet loss minimization or a hybrid), and Vector Pairing (hash, similarity-based or composite). The table's columns summarise the first two.

| Solution | Tuple embedding | Training signal | Word-level input |
| --- | --- | --- | --- |
| SIF | Frequency-weighted average of word vectors, first principal component removed | None | Pretrained fastText |
| Seq2seq | LSTM encoder and decoder over the word sequence | Self-reproduction: predicting each next word, cross-entropy | Pretrained fastText |
| Autoencoder | Feed-forward encoder and decoder over the SIF aggregate | Reconstruction, self-supervised | Pretrained fastText |
| SBERT | Pretrained BERT with Sentence-BERT aggregation | Triplet loss on generated triplets as described; the scheme measured is the pretrained checkpoint, with no training time reported | BPE subwords |
| CTT | Siamese branches over the SIF aggregate, feed-forward classifier over their absolute difference | Match/non-match prediction on synthetic labelled pairs | Pretrained fastText |
| Hybrid | The autoencoder's trained encoder in place of SIF as the CTT aggregator, feeding CTT's Siamese summarizer | Reconstruction, then cross-tuple pairs | Pretrained fastText |
| Trans-encoder | Seq2seq with a standard transformer in place of the LSTMs, the paper's one sentence on it | Self-reproduction, by analogy with Seq2seq; no result of its own is reported | BPE subwords |
| CTT-cosine | Siamese branches over the SIF aggregate, cosine between them as the score | Cosine objective on synthetic labelled pairs | Pretrained fastText |

Each solution is a family here, named for the scheme. Every family composes into the shared resolver and appears as a row in the cross-area strategy matrix.

## Order of building

The corpus is structured by nature and carries a single attribute, the name, so the paper's structured-data results are the ones that apply, and its textual-data results do not. On structured data the paper finds the Autoencoder the best scheme and the cheapest to train, Hybrid no better than the Autoencoder alone, and Seq2seq significantly behind every other scheme. The families are therefore built in this order: SIF, then the Autoencoder over it; CTT and CTT-cosine; Hybrid once both its halves exist; Seq2seq and Trans-encoder last. The last two are not prioritised because a sequence-oriented scheme has the least to act on over a short name, but they stay buildable and take a matrix row like the rest, since capability is compared and never gated on a prior result. Each epic's `Decision` records its own place in that order.

The Autoencoder's place is on cost, not on expected recall. The later pretrained-embeddings analysis ([external alignment](../external_alignment.md)) found a pretrained sentence encoder beating DeepBlocker's Autoencoder over fastText by 15% recall on eight of ten datasets, at a vectorization cost that was 99% of its blocking time. The SBERT family's pretrained row exists already, so the Autoencoder is the cheapest trained family to put beside it, and the comparison between the two on company names is the one that analysis predicts and does not measure.

## Mechanism, component, state

Each row is one mechanic of that pipeline and the component here that implements it. `Partial` means some of the mechanic is delivered and the rest is not.

| DeepBlocker mechanic | Component here | State |
| --- | --- | --- |
| Serialize a tuple to one string | The single `name` field | `Completed` |
| Embed the string | A `representation` value plus its `TargetIndexBuildSettings` subclass behind the `ClusteringStrategy` protocol (`clustering_contract.py`). TF-IDF, WordPiece, SentencePiece and pretrained S-BERT exist; a seam taking any trained encoder's vectors does not yet | `Partial` |
| Build a target index over the embeddings | `ClusteringStrategy.build_target_index()`/`build_backend_index()` | `Completed` |
| Retrieve top-K by cosine similarity | `ClusteringStrategy.score_source_chunk()`, which thresholds and ranks in one call | `Completed` |
| Sub-linear candidate generation. The eight solutions all pair by top-K cosine through FAISS; hash and composite pairing are in the design space, and LSH did much worse | `"kmeans"`/`"hdbscan"` partition backends (`partition_similarity.py`) and the `"lsh"` sparse-signature backend (`lsh_similarity.py`), and the dense `"hnsw"` backend (`hnsw_similarity.py`) | `Partial` |
| Generate self-supervised training pairs | `company_classify`'s three pair producers, drawing mutations from `company_perturbation`'s operator taxonomy. The paper's own generator, a random word subset keeping at least 60% of the words with p random negatives per tuple, joins them as a run option of the trainable Siamese branch | `Partial` |
| CTT: a Siamese network and feed-forward classifier over the absolute difference of two tuple vectors | A trainable Siamese branch: the pooled-subword contrastive encoder (`PooledSubwordContrastiveEncoder`) has landed, while the contract that makes its architecture swappable and trains it under a classifier over the absolute difference, and the indexing of its embeddings, are still to come. `TfidfPairMlpClassifier` is kept as a baseline rescorer, with no trained branch and no extractable embedding, its ceiling on real name pairs measured | `Partial` |
| Supply the static word vectors six of the eight solutions need | Word vectors trained per target system on its own corpus and a pretrained fastText baseline, behind one token-to-vector lookup that records which it served | `Planned` |
| Choose the pretrained transformer (SBERT) | A language-keyed registry (`resources/sbert_models.json`) resolved from a run's `--countries`, overridden by `--sbert-model`. The three-model comparison run is owed | `Completed` |
| Score a blocking strategy | `recall_at_k` and `candidate_set_size_ratio` on the strategy-comparison report, the two metrics DeepBlocker reports, computed for the deterministic baselines on the same slices. A real comparison run carrying both columns is owed | `Completed` |
| Orchestrate a run | `execute_blocking_run()`, which takes the representation as configuration | `Completed` |

## Differences from a straight reimplementation

1. **Comparison, not replacement.** Every representation, deterministic or learned, runs through the same comparison report on the same slices under the same two metrics. Which one a run uses is a per-run setting and can differ by slice.
2. **The trained Siamese branch has a swappable architecture.** DeepBlocker fixes its Siamese summarizer as a two-layer feed-forward network over the SIF aggregate. Here the branch is a contract any branch family implements: the pooled-subword contrastive encoder over this repo's trained tokenizers first, then the paper's feed-forward branch over a vector input, then convolutional and recurrent branches, compared in the matrix with the pretrained sentence encoder. The paper's own configuration is kept as a row of the CTT family, as the pretrained ingredient is a row of the SIF family, so the matrix says whether a branch over this repo's tokens beats the paper's summarizer. CTT, CTT-cosine and Hybrid are that one branch trained under a classifier over the absolute difference, or over the concatenation the paper's ablation found slightly stronger, under a cosine objective, or behind the frozen autoencoder encoder, and each yields an indexable embedding. The TF-IDF pair rescorer stays as the baseline they are measured against. The branches reuse the tokenizers [the tokenization design journey](tokenization-journey.md) documents.
3. **Self-supervision uses the existing mutation taxonomy.** `company_classify`'s pair producers generate positive and negative pairs from `company_perturbation`'s operators (token drop and shuffle, legal-suffix swap, phonetic swap, transliteration), with random or lexically hard negatives. DeepBlocker generates one positive per tuple by keeping a random subset of at least 60% of its words and p random negatives. That generator is a run option of the trainable Siamese branch, beside the taxonomy, so the taxonomy's advantage is measured rather than assumed.
4. **The static-vector ingredient is domain-trained, with the pretrained one as a baseline row.** A general pretrained fastText model is a poor fit for multi-jurisdiction, multi-script, legal-suffix-heavy company names, so one is trained per target system on that system's own cleansed name corpus, never on a blend of systems: `ltd` is 95% of `gb` names and 15% of `gleif`'s, so blended frequencies misweight every jurisdiction in the blend. The pretrained vectors are served through the same lookup, so both appear in the matrix.
5. **Encoder selection is language-matched per run.** The corpus is majority non-English and a country-scoped run carries the language signal, so a run resolves a checkpoint from its countries, falling back to the English default when the countries are unset, span more than one registered entry, or match none. A checkpoint's tokenizer segments with a vocabulary trained on that language, so each registry entry is to record its own tokenizer.
6. **Seq2seq and Trans-encoder read this repo's trained tokenizers.** DeepBlocker's Seq2seq reads pretrained fastText word vectors and its Trans-encoder BPE subwords; here Seq2seq runs over an already-trained tokenizer's token ids and Trans-encoder over the trained SentencePiece tokens. The paper specifies Trans-encoder in one sentence and reports no result for it, so its specification here is this repository's own.
7. **SBERT is fine-tuned on generated pairs with a selectable metric loss, beyond the paper's measurement.** DeepBlocker describes triplet training but measures the pretrained checkpoint; the untuned row is the paper's SBERT as measured, and fine-tuning under a contrastive, triplet or multiple-negatives-ranking loss chosen per run is this repo's extension.
8. **SIF's weights and principal component are frozen with the target index.** DeepBlocker counts word frequencies over the two tables being blocked and its code refits the component on every batch it embeds; here the frequencies are counted once from the target system's corpus, persists both with the index, and weights inbound names by the same distribution.
9. **Every model is trained on the target corpus alone.** DeepBlocker trains each scheme on the union of the two tables being blocked, so the Autoencoder reconstructs the very tuples it will retrieve. Here the target system is the canonical reference an inbound name is resolved against, so the static vectors, the autoencoder, the sequence trainers and the Siamese branches are each trained on the target system's own corpus and pairs, one model per target, never blended and never including the inbound side. An inbound name is embedded by a model that has not seen it, which is what the frozen index in difference 8 already implies for SIF.

## The reference implementation

The published code at [saravanan-thirumuruganathan/DeepBlocker](https://github.com/saravanan-thirumuruganathan/DeepBlocker) ships three of the eight schemes, Autoencoder, CTT and Hybrid, and differs from the paper's description in ways that matter when a row here claims to be the paper's configuration. Read from the code, not the paper:

- Every scheme's `preprocess` runs over the concatenation of both tables, so the SIF frequencies, the principal component and every trained model see the tuples they will later retrieve. Difference 9 above is the departure from this.
- SIF uses the SIF paper's default weighting constant `a = 1e-3`, a mean rather than a sum of the weighted word vectors, and a token below `min_freq` gets weight `1.0`. The first component is refitted by truncated SVD on every call to embed a batch, so the component removed at training time is not the one removed from a queried batch. Difference 8's frozen component is the fix.
- The Autoencoder reconstructs the SIF vector with the component removed, in 300 dimensions, through a 300-300-150 encoder and its mirror, with ReLU where the paper says Tanh, mean-squared-error loss, Adam at `1e-3`, 50 epochs, batch 256.
- CTT's Siamese summarizer is 300-300-150 with ReLU, and its classifier is a single linear layer with a sigmoid over the absolute difference, not the two-layer network the paper describes, trained with binary cross-entropy.
- The published CTT and Hybrid return the aggregator's vector as the tuple embedding, the SIF vector for CTT and the autoencoder's for Hybrid; the trained summarizer is never applied at embedding time. Whether the paper's CTT and Hybrid figures came from this code is not stated. The paper's description, the summarizer output as the embedding, is what the CTT row here implements; the code's behaviour is not a second row, since its candidate sets are the SIF and Autoencoder rows' already, and its trained head is a candidate pair rescorer for the resolver's scorer seam instead.
- Pair generation makes five positives and five negatives per tuple. A positive removes a uniformly random count of up to 40% of the tuple's tokens, so a positive can equal its anchor; a negative is a uniformly random tuple with the anchor itself not excluded.
- Vector pairing is an exact full cosine matrix through `scipy` with `argsort`, not FAISS, and the tokenizer is `torchtext`'s `basic_english`.

## Reference figures on Amazon-Google

The figure conformance work is held to is a local run of DeepBlocker's own reference code rather than the published number, since the run depends on none of this repo's code and can be made before anything is built to match it. Amazon-Google at K of 50 over `title`, `manufacturer` and `price`, the left table's 1,363 tuples posed as queries against the right table's 3,226 indexed tuples, with ground truth of 1,167 pairs derived by the reference code's own `blocking_utils.process_files` from the rows labelled `1` across `train`, `valid` and `test`.

| Representation | Recall | Candidate set | `candidate_set_size_ratio` |
| --- | --- | --- | --- |
| SIF | 0.9597 | 68,150 | 0.015499 |
| Autoencoder | 0.8955 (sd 0.0048, n=5) | 68,150 | 0.015499 |

The candidate set is 1,363 x 50 in every run, as exact top-K requires. The published code labels three schemes and produces these two representations: `CTTTupleEmbedding` returns the SIF vector and `HybridTupleEmbedding` the autoencoder's, each discarding a Siamese head it trained. The SIF row is exact, holding to sixteen decimal places across three runs, two torch versions and two tokenizer implementations, because nothing trained reaches its output. The Autoencoder row is a mean over five draws spanning 0.8903 to 0.9023: `torch.manual_seed` is never called and both trainers shuffle, so no single autoencoder run is reproducible, and any comparison closer than about one point needs repetitions. Two of those five draws were produced under the `Hybrid` label, which runs only under the repair named below and is therefore this repository's output rather than the reference code's.

### Against the paper

Table 6 reports 97.1 for this dataset at the same K and the same 68.2k candidate set. The local Autoencoder figure is 7.6 points below it, about sixteen times the measured spread, so the difference is not sampling. Table 4 reports 1,300 matches where the reference code's own converter derives 1,167 from the archive its README links.

### Checkpoint and archive

fastText `wiki.en.bin`, 8,493,673,445 bytes, from `https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip`. Benchmark `amazon_google_exp_data.zip`, sha256 `9b19c85c98c6289f38969e4961cb944bdf986123fc42f6bf17283ac6eb502e61`, from the DeepMatcher datasets page under `Structured/Amazon-Google`.

### The diff from upstream

Two branches in a local checkout of the reference code, both off `upstream/main`. `update` carries three stack-compatibility changes: `scikit-learn` in place of the `sklearn` stub, the obsolete `pathlib` requirement dropped, and `basic_english` reimplemented locally. `update-torchtext` carries the first two and keeps upstream's `torchtext`; the two tokenizers agree over all 18,356 strings and 110,588 tokens of both tables, so the branches differ in provenance and not in result. Both then carry one repair, `HybridTupleEmbedding.input_dimension` corrected from the word-vector width to the autoencoder's output width, without which that path raises.

### Environment, none of it in the diff

`requirements.txt` is unpinned, so a reproduction fails without these. `pandas<3`, since pandas 3 rejects `deep_blocker.py`'s string `fillna` over the float64 `price` column. `pybind11<3` for the fastText build, since 3.x removes the `ssize_t` its pybind source uses. `CL=/std:c++17 /Dssize_t=ptrdiff_t`, since fastText's `setup.py` passes MSVC neither. A build path short enough for `MAX_PATH`, since the linker otherwise fails and leaves a zero-byte `.pyd` that a rerun reports as success, so the `.pyd` size is what says the build worked rather than the exit code. `update-torchtext` additionally pins `torch==2.3.1`, the ABI `torchtext` 0.18.0 is built against. All four build settings are MSVC artefacts and do not arise on Linux.

## Two tracks

- **The embedding track**: a learned vector space, self-supervised via pairs, with one family per DeepBlocker scheme. On the representation side: the pretrained sentence encoder, the partition, LSH and HNSW candidate backends, the SIF representation, and the strategy that indexes any trained encoder's output. On the training side: the pair producers, the trainable Siamese branch with its swappable architecture, the fine-tuned sentence encoder, the domain-trained and pretrained static vectors, and the Autoencoder, Seq2seq and Trans-encoder trainers. The pair producers' real alias-pair source (the GLEIF and Wikidata `*-names-*` sidecars) is a second self-supervision signal alongside synthetic pairs.
- **The classical/deterministic track**: the TF-IDF, WordPiece and SentencePiece baselines, comparison-cleaning, and the items from the blocking-taxonomy audit against Papadakis et al. arXiv:1905.06167. These are mechanical derivations or pruning steps with no new dependency. Its live members are the multi-key name-variant target index and derived-key expansion.

Both tracks feed the same comparison. Three results so far bound what the classical track can recover: the short-name stem bought precision and no recall; the recorded alias and variant keys recovered 1.7 to 2.1pp of real recall on divergent-name pairs; a cross-record initialism signal was declined, since 0.0128% of 391,684 real GLEIF matched pairs are initialism-shaped.

## The other related system

pyJedAI implements the same embed, index and retrieve top-K shape through its own vector-based workflow, and is covered separately in [pyJedAI](pyjedai.md).
