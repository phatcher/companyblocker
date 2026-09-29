# pyJedAI

Part of [CompanyBlocker Architecture](../architecture.md): detail behind [§4](../architecture.md#4-related-systems-deepblocker-and-pyjedai).

## What pyJedAI is

pyJedAI is an end-to-end entity-resolution library from the AI team at the University of Athens, the same group as the blocking taxonomy (Papadakis et al. arXiv:1905.06167) this repo's classical track was audited against. It composes a pipeline from five stages: block building, block cleaning, comparison cleaning, entity matching, and entity clustering.

It offers two families of candidate generation. The key-based family builds blocks from the record text (`StandardBlocking`, `QGramsBlocking`, `ExtendedQGramsBlocking`, `SuffixArraysBlocking`, `ExtendedSuffixArraysBlocking`). The vector-based family, `EmbeddingsNNBlockBuilding`, embeds records with pretrained PyTorch word or sentence embeddings and retrieves the k nearest neighbours through a FAISS index.

That second family is the one that matters for positioning. It has the same shape as DeepBlocker's pipeline and as the blocking here: turn a record into one string, embed it, index the embeddings, retrieve top-K by vector similarity. pyJedAI therefore covers the same layer, which is what makes it a candidate external baseline rather than a toolkit for a different problem.

## Stage, counterpart, state

Each row is one pyJedAI pipeline stage and what stands in its place here. `Partial` means some of it is delivered and the rest is not.

| pyJedAI stage | Counterpart here | State |
| --- | --- | --- |
| Key-based block building over record text | The deterministic representations behind `ClusteringStrategy`: TF-IDF, WordPiece, SentencePiece, plus the multi-key name-variant index and its derived keys | `Partial` |
| Vector-based block building: pretrained embeddings indexed in FAISS | The pretrained sentence-encoder strategy, with HNSW as the sub-linear index | `Partial` |
| Block cleaning: drop or shrink blocks by size | No counterpart. Candidate generation and scoring are one call, `score_source_chunk()`, so no block object exists to act on | `Planned` |
| Comparison cleaning (meta-blocking): weight and prune candidate pairs by how blocks co-occur | No counterpart. Multi-key indexing now gives a record several keys, which is the setting this stage was built for | `Planned` |
| Entity matching: score candidate pairs | A pair scorer composed with candidate generation in one resolver: the TF-IDF pair rescorer baseline and the trained pooled-subword encoder, with the swappable Siamese branch still to come | `Partial` |
| Entity clustering: partition matched pairs into equivalence sets | Not built, and out of scope: this platform stops at candidate generation and its evaluation | `Planned` |

## Where this repository differs

1. **One attribute, not many.** pyJedAI resolves multi-attribute records and vectorizes either per attribute or by concatenating attributes into one sentence. This repo's records carry a single company name, so the concatenation step is a no-op and every result is scored on name evidence alone.
2. **Company names, not general records.** The cleansing, company-type resolution and legal-suffix handling that `company_cleanse` performs against ISO 20275 have no equivalent in a general-purpose ER toolkit, and the divergent-name problems this corpus presents are company-specific.
3. **A comparison harness is the deliverable.** The strategy-comparison report scores every representation on the same slices under `recall_at_k` and `candidate_set_size_ratio`, so strategies are ranked against each other on this corpus rather than one being configured and run.
4. **Blocking only.** pyJedAI carries the pipeline through matching and clustering to resolved entities. This platform stops at candidate generation and its evaluation, which is why its false-positive tolerance is deliberately loose: a spurious candidate costs one discarded downstream comparison.

## The orchestration credit

`src/blocking/README.md` credits pyJedAI for `src/blocking`'s workflow shape: load two datasets, generate candidate blocks, and surface evaluation metrics, with the algorithm layer not owning orchestration.

## Not yet a measured comparison

pyJedAI is not a dependency here and no run has been scored against it. Its vector-based workflow is the closest existing implementation of what the family epics build, so it is the natural external baseline for them, but that comparison has not been made and no claim about relative quality or runtime is available.
