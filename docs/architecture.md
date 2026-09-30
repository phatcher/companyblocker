# CompanyBlocker Architecture

This document is the detail layer behind [`README.md`](../README.md). The README says how to run the system; this document says why it is shaped as it is. Sections 3 to 8 are summaries with a link to a standalone detail document each. This document names capabilities and decisions, not their tracking state.

## 1. Executive Summary

CompanyBlocker is a research platform for blocking on company names: the candidate-generation step that precedes pairwise comparison in entity resolution. Blocking is well studied for people records. For companies the published work is thin, often not reproducible, or embedded in commercial matching software where only the output is visible. The aim here is a reproducible, inspectable implementation, taking its design from DeepBlocker and pyJedAI (§4).

Three constraints shape it.

- **Public data.** The commercial datasets that would settle whether a technique works on companies are priced for enterprises, so the platform runs on public registries (national registers, GLEIF, Wikidata; §7) on a single machine, with a new source onboarded through metadata rather than code (§3).
- **Claims are tested.** Statements such as "company names need their own tokenizer" are treated as hypotheses. Standard tooling is tried first and a bespoke path adopted only on evidence, which is kept. The token-rarity analysis, which measures each corpus's rare-token tail against `wordfreq`'s pre-2020 general-language frequencies, is the running example (§5), and every optimisation is logged with a before-and-after number, rejected ones included (§8).
- **Build order follows the data.** Blocking needs clean, tokenized names, so acquisition, cleansing and tokenization came first and have been through several rounds of real-data iteration. Blocking is the current focus: the strategy-comparison harness is one documented command, and the embedding track (§4) is largely still planned.

## 2. System Overview & Goals

The platform's goals, from the README: acquire and normalise company registry data from several systems; build stable intermediate datasets for matching and learning; support repeatable training, analysis and validation of entity-resolution quality.

A **system** is one data source: a country registry (`gb`, `fr`, `ie` as bulk downloads; `dk`, `ee`, `fi` through APIs) or a non-country one (`gleif`, `offeneregister`, `wikidata`). A system is onboarded by adding catalog metadata (see [design principles](architecture/design-principles.md#metadata-first-catalog-driven-onboarding)).

### Processing pipelines

Three workflows share one set of stages.

**Primary data flow**

```
Acquire → Shard → Canonical → Cleanse → Match → Vectorize/Blocking → Analyse
```

| Stage | Purpose | Output layer |
| --- | --- | --- |
| Acquire | Download source datasets, with metadata-driven freshness gating | `data/<system>/acquire/<date>` |
| Shard | Convert acquired payloads to Parquet shards | `data/<system>/source/<date>` |
| Canonical | Map shards into the shared canonical schema | `data/<system>/canonical/<date>` |
| Cleanse | Deterministic company-name and company-type normalisation | `data/<system>/cleansed/` |
| Match | Join records by jurisdiction and company number to produce ground truth | `data/<system>/matched/` |
| Vectorize/Blocking | Generate and score candidate-match strategies against ground truth | `artifacts/blocking/` |
| Analyse | Evaluate clustering, blocking and match quality | `artifacts/validation/`, `artifacts/analysis/` |

Match is the only source of ground truth: it joins two systems on jurisdiction and company number to produce `match_uri`. A blocking run can execute against a system's canonical snapshot alone, but without a matched layer behind it there is nothing to score against, so a run is scoreable only once its source system has been through Match.

Canonical establishes `system_uri` uniqueness. The collisions it absorbs come from the source, not from chunked acquisition: a measurement over real snapshots found 321 duplicate `system_uri` groups in GLEIF and none in `fr`, `gb` or `offeneregister`. No stage merges distinct records found to be the same company; that is downstream of blocking.

**Product path and instruments.** The product path is Canonical to Match to Blocking. The cleansed and tokenized layers are instruments: a materialised snapshot of the current cleanse rules over a whole corpus, used for inspection and as the tokenizer training corpus. Blocking does not read them. It derives both sides' name forms live from the raw `name` under one named cleanse profile per run, so the profile is a comparison axis and identity is its zero point. Match still reads the cleansed layer, since that is Cleanse's output; moving it to read Canonical, so a system can be matched before any cleanse configuration exists, is planned. A stale cleansed layer is therefore a stale reading of the rules, never a broken product path.

**Tokenizer-artifact loop**

```
Train/Optimize → tokenizer + noise-word artifacts
                    ├→ Tokenize (persists tokenized/ columns) → Analyse
                    └→ Blocking's WordPiece/SentencePiece representations (tokenize on the fly)
```

| Stage | Purpose | Output layer |
| --- | --- | --- |
| Train/Optimize | Build or grid-search WordPiece and SentencePiece tokenizers, derive TF-IDF noise-word profiles | `artifacts/tokenizers/work/<system>/`; promoted candidates under `artifacts/tokenizers/data/` |
| Tokenize | Apply a trained tokenizer to persist token columns for consumers that want them at rest | `data/<system>/tokenized/` |

Blocking's tokenizer-based representations depend on the trained artifact, not on the Tokenize stage having run: they tokenize on the fly. The TF-IDF representation uses no trained tokenizer at all.

Each stage is a contract boundary with a stable schema, so onboarding a system, swapping a tokenizer or changing a blocking strategy can be reasoned about and tested on its own. That separation is the central design choice.

## 3. Architecture & Design Principles

Metadata-first onboarding that confines source-specific code to two stages; a deliberately narrow canonical schema, since name-only is the hardest blocking case and every extra field is acquisition and cleansing surface for every system; stage-oriented execution with contract boundaries; deterministic baselines before learned variants; package and area boundaries enforced by `tach`; Polars and Parquet for single-machine scale, with a stated rule for when another engine is allowed; C4 diagrams in Structurizr; and one deterministic-seeding convention.

**Full detail:** [Architecture & Design Principles](architecture/design-principles.md)

## 4. Related systems: DeepBlocker and pyJedAI

**DeepBlocker.** Its pipeline, serialise a tuple, embed each word, combine the word vectors into one tuple vector, retrieve top-K by cosine, fits the `ClusteringStrategy` contract every representation here implements, so a scheme enters as a representation value rather than a new orchestration layer. Present today: the pretrained sentence encoder, the partition and LSH backends, an exact dense scan, the self-supervised pair producers, a TF-IDF pair rescorer kept as a baseline, and a trained pooled-subword contrastive encoder. Planned: the dense HNSW backend, the static word-vector ingredients, the trainable Siamese branch that CTT, CTT-cosine and Hybrid share, and the remaining scheme trainers. Each of the paper's eight schemes is a family with its own epic, scored by the same recall-at-k and candidate-set-size ratio as the deterministic baselines. Where the adaptation departs from the paper, the departure is recorded in the detail document, nine so far, including that every model is trained on the target corpus alone. The corpus is structured and carries one attribute, so the families are built in the order the paper's structured-data results support, Autoencoder first and the sequence-oriented schemes last, with none gated on a prior result. The classical track is the baseline the families are judged against, as the paper's non-learned blockers are in its own comparison.

**pyJedAI.** An end-to-end entity-resolution library whose vector-based block building embeds records with pretrained encoders and retrieves neighbours through a FAISS index, the same shape as the blocking here. Its key-based blocking corresponds to the deterministic representations; its block-cleaning and meta-blocking stages have no counterpart, since candidate generation and scoring are one call here. No run has been scored against it.

**Full detail:** [DeepBlocker Inspiration](architecture/deepblocker.md), [pyJedAI](architecture/pyjedai.md), and [External Alignment](external_alignment.md) for the standing against both, the pretrained-embeddings analysis and progressive ER.

## 5. Design Journey: Tokenization → Blocking Keys

Tokenizer optimisation refined its proxy metrics (fertility distance, token-count percentiles) through four corrections: a seed-independence bug that produced 76% correlated splits against about 20% expected; a fixed discrete search grid with a computed noise floor; a selection metric read on a held-out split although the deployed tokenizer is retrained on the full corpus; and a promising vocabulary region whose candidates the real gates rejected. Whether those proxies predict blocking or matching quality has not been tested, and that is the largest open assumption under the strategy comparison.

**Full detail:** [Tokenization Design Journey](architecture/tokenization-journey.md)

## 6. Package Deep-Dive

- **`company_cleanse`**: ISO 20275-grounded normalisation; a leaf package.
- **`company_tokenize`**: WordPiece and SentencePiece training, TF-IDF noise-word derivation.
- **`company_vectorize`**: the `ClusteringStrategy` contract and the similarity backends, over TF-IDF, WordPiece, SentencePiece and the sentence encoder.
- **`company_classify`**: single-text baselines, the TF-IDF pair rescorer and the pooled-subword contrastive encoder; the rest of the encoder track is planned.
- **`company_perturbation`**: deterministic name-mutation operators in seven families, with profile resolution and per-record orchestration.

Short-name and divergent-name matching (acronyms, truncation, unrelated brand names) are three different problems, and a blended recall over them is dominated by trivial identity matches. The metrics therefore report precision and recall over the cleansed-different residual, with raw-identical and cleanse-absorbed pairs carried as counts.

**Full detail:** [Package Deep-Dive](architecture/packages.md)

## 7. Per-System Acquisition Journeys

Nine systems are onboarded: `gb`, `fr` and `ie` by bulk download, `dk`, `ee` and `fi` by API, and `gleif`, `offeneregister` and `wikidata`. The bulk country registries were straightforward. GLEIF needed schema-index caching, a reversed-token trie for company-type matching and several data-quality fixes. OffeneRegister produced the most defects: an unusable raw company number, an extraction bug, an upstream placeholder-number defect and a tokenizer regex bug. Wikidata needed a Rust extractor for its 145 GB compressed dump and had two production incidents, both root-caused. DBpedia was tried and stopped on data quality.

**Full detail:** [Per-System Acquisition Journeys](architecture/acquisition-journeys.md)

## 8. Cross-Cutting Optimization Log Highlights

`docs/optimizations.md` is the append-only benchmark log. Two patterns recur. Small, individually benchmarked changes compound: six of them took the cleanser from 9,372 to 15,327 rows per second. A refactor that isolates tangled code surfaces latent bugs: splitting a 1,400-line module into five found two unrelated to the reason for the split.

**Full detail:** [Optimization Log Highlights](architecture/optimization-highlights.md)

## 9. Quality Gates

The gates keep the code's structure honest against what the documents claim of it. `bandit`, `mypy`, `ruff check` and `tach` run as pre-commit hooks against changed files only; a full-repository run of each is a separate step. `pytest`, `pyscn` and `graphify` run only when invoked.

| Tool | What it checks |
| --- | --- |
| `bandit` | Static security analysis |
| `graphify` | A queryable knowledge graph of the repository, refreshed with `graphify update .`; the JSON graph is a local artifact, the Markdown report is committed |
| `mypy` | Static types |
| `pyscn` | Coupling, cohesion, dependency cycles, dead and duplicate code |
| `pytest` | The test suite |
| `ruff check` | Bugs, unused code, style |
| `tach` | The module dependency boundaries in `tach.toml` |

## 10. Current State

The acquisition, canonicalisation, cleansing and tokenization pipeline, the package boundaries and the repository-wide quality gates are built. The blocking comparison harness is built and is the current focus. The embedding track is mostly planned: the pretrained sentence encoder and the exact dense scan exist, and the sub-linear dense index, the static-vector ingredients and the trained encoders do not.
## 11. Appendix

### Systems reference

| Code | Source | Acquisition shape | State |
| --- | --- | --- | --- |
| `gb` | Companies House (UK) | Bulk monthly ZIP/CSV | Supported |
| `fr` | INSEE SIRENE (France) | Bulk monthly Parquet | Supported |
| `ie` | CRO (Ireland) | Bulk daily ZIP/CSV | Supported |
| `gleif` | GLEIF Global LEI | Bulk ZIP/XML | Supported |
| `offeneregister` | Community aggregator (Germany) | Bulk JSONL | Supported |
| `wikidata` | Wikidata | Bulk compressed dump, Rust extractor | Acquired through cleanse; no tokenizer has been fit on its own vocabulary |
| `dk` / `ee` / `fi` | National APIs (Denmark, Estonia, Finland) | API | Supported, credential-gated |
| `dbpedia` | DBpedia Databus | Bulk RDF/TTL | Stopped on data quality |
| `nl` / `edgar` | | | Not onboarded, research required |

### Detail documents

- [Architecture & Design Principles](architecture/design-principles.md): §3
- [DeepBlocker Inspiration](architecture/deepblocker.md) and [pyJedAI](architecture/pyjedai.md): §4
- [Tokenization Design Journey](architecture/tokenization-journey.md): §5
- [Package Deep-Dive](architecture/packages.md): §6
- [Per-System Acquisition Journeys](architecture/acquisition-journeys.md): §7
- [Wikidata Extraction Design Journey](architecture/wikidata-extraction.md): §7
- [Optimization Log Highlights](architecture/optimization-highlights.md): §8

### Related documents

- Operational runbook: `README.md`
- External alignment: `docs/external_alignment.md`
- Canonical schema contract: `docs/canonical_schema.md`
- Each system's own source columns: `docs/source_schemas.md`
- Full optimisation log: `docs/optimizations.md`
- Structurizr/C4 model source: `docs/structurizr/`
- Source-specific data-quality adjustments: `docs/source_data_adjustments.md`