# External Alignment

How the platform stands against the external systems it takes design influence from, and where that rests on evidence. [`architecture.md`](architecture.md) describes what the system is; mechanism-level mappings live in [DeepBlocker Inspiration](architecture/deepblocker.md) and [pyJedAI](architecture/pyjedai.md).

Status vocabulary is `Implemented`, `Planned` and `Awaiting evidence` (a capability exists but no run has produced a number for it).

## The four reference points

pyJedAI and the pretrained-embeddings analysis come from the same Athens group but are independent: a library and a measurement.

| System | What it is | Axis it covers | Detail here |
| --- | --- | --- | --- |
| DeepBlocker | An experimental comparison of eight deep-learning blocking schemes over one shared pipeline: serialize a tuple, embed each word, combine into one vector, retrieve top-K by cosine similarity | How a record becomes a vector | [DeepBlocker Inspiration](architecture/deepblocker.md) |
| Pretrained embeddings for ER | An experimental comparison of twelve pretrained models (three static, five BERT-family, four sentence-transformer) as nearest-neighbour blockers and as matchers, with DeepBlocker's Autoencoder as baseline | Which pretrained vectorizer to start from, and what it costs | This document |
| pyJedAI | A library composing block building, block cleaning, comparison cleaning, matching and clustering, with key-based and vector-based block building | What stages an ER pipeline is made of | [pyJedAI](architecture/pyjedai.md) |
| PER | A framework decomposing progressive ER into filtering, weighting, scheduling and matching, emitting results in pay-as-you-go order under a budget | In what order candidates are emitted, and what that costs | This document |

### What the pretrained-embeddings analysis found

The analysis (Zeakis, Papadakis, Skoutas and Koubarakis, PVLDB 16) vectorizes each record once per model and blocks by nearest-neighbour search, the same shape as this repo's sentence-encoder strategy.

| Finding | Detail |
| --- | --- |
| Recall by family | Sentence-transformers highest on every dataset but noisiest; S-GTR-T5 best. Static: GloVe > fastText > word2vec. Untuned BERT-family below the static models |
| Against DeepBlocker | S-GTR-T5 beat the Autoencoder over fastText by 15% recall on eight of ten datasets; the other two were near-perfect for both |
| Cost | Vectorization was 99% of the sentence encoder's blocking time; DeepBlocker was faster at small k, losing that lead only where it scaled poorly with k or input size |

This is the closest external prior for the SBERT family against the Autoencoder family: expect the untuned sentence encoder to win recall and lose on encoding cost. The Autoencoder is built first on cost. None of the analysis's datasets contain company names, so it is a prior, not a number; its public sets serve as a known-answer test of the implementation.

## Where the alignment holds

- **DeepBlocker:** The `ClusteringStrategy` protocol in `clustering_contract.py` admits an embedding scheme as a representation value inside the existing contract. Departures from the paper (input, loss, Siamese-branch architecture) are recorded in [DeepBlocker Inspiration](architecture/deepblocker.md). Self-supervised pairs come from `company_perturbation`'s operator taxonomy, and encoder selection is keyed by a run's countries, which the majority-non-English corpus requires.
- **pyJedAI:** The divergence is scope, not shape: this platform stops at candidate generation and its evaluation, and ranks representations against each other on the same slices under the same metrics.

## Where the alignment is unmeasured

### Short sequences narrow DeepBlocker's aggregation dimension

A serialized multi-attribute tuple runs to tens of mixed words; a cleansed company name is a few words from one vocabulary. The aggregation dimension (how a word sequence becomes one vector) has little to act on over a few words.

| Scheme type | Exposure on short names |
| --- | --- |
| Frequency-weighted average and schemes built on it | Unaffected in mechanism; removing the dominant principal component acts on the legal-suffix direction, a substantive effect |
| Recurrent or transformer encoder trained from scratch | Exposed directly; a reconstruction objective has less to reconstruct |
| Pretrained sentence encoder | Unaffected |
| Pairwise-score output | Serves reranking, not blocking: no index can be built over it |
| Cosine-trained output | Directly indexable, so addresses candidate generation |

DeepBlocker's own finding that the simplest self-supervised schemes are competitive points the same way: expect a narrower spread across the eight than in the paper.

### Static word vectors gate most of the embedding families

Six of the eight schemes need static word vectors. Both the domain-trained ingredient and the pretrained baseline are `Planned`, so most of the embedding track waits on one component. The sub-linear dense index sits ahead of it because the strategy matrix runs on the sub-linear backends and the sentence-encoder row, which needs no static vectors, is in the first matrix.

### Block cleaning and comparison cleaning have no counterpart

Candidate generation and scoring are one call, `score_source_chunk()`, so no block object exists for a cleaning stage to act on. Comparison cleaning (meta-blocking) is the highest-yield technique in the blocking taxonomy the classical track was audited against, and multi-key name-variant indexing now gives each record several keys, its precondition. Opening that stage is an architecture question about the fused call.

| Classical-track result | Outcome |
| --- | --- |
| Short-name stem | Precision gained, no recall |
| Cross-record initialism signal | Declined at 0.0128% of real matched pairs |
| Recorded alias and variant keys | +1.7 to 2.1pp on divergent-name pairs |

### Progressive evaluation is absent

| PER step | Counterpart here |
| --- | --- |
| Filtering | Candidate generation |
| Weighting | `similarity` column |
| Scheduling | None, and no metric for it |
| Matching | Pair classifier |

The platform's economics are already progressive: it tolerates loose precision because a spurious candidate costs one downstream comparison. The metrics are batch: `recall_at_k` and `candidate_set_size_ratio` at one fixed `top_k`, which makes strategies retrieving at different costs incomparable. Progressive recall (recall against comparisons spent) removes the choice of k. PER schedules globally across the candidate stream; ranking here is per source record, so easy and hard records get the same budget.

`matched_edges` already carries `similarity` and `rank` per pair, so a progressive-recall curve and its normalized area are derivable from existing files with no new run or dependency. `compute_pair_truth_eval()` already supports recall within the top 1 or top 5 of a wider run.

### No external comparison run exists

pyJedAI is not a dependency and no run has been scored against it, though its vector-based block building is the closest existing implementation. No DeepBlocker scheme has been scored either. Every family epic is measured against the classical track, whose own strength is unestablished. `Awaiting evidence` applies to the comparison as a whole.

### What the reference artifacts can be held to

A conformance target has to come from published code, not the paper. DeepBlocker's reference implementation was run locally on Amazon-Google at K = 50 (figures in `docs/architecture/deepblocker.md`).

| Axis | Finding |
| --- | --- |
| Coverage | Four of the paper's eight solutions: the frequency-weighted average and the three built on it. No LSTM, transformer or BERT solution; one pairing strategy; none of the three non-DL baselines. Four of Table 6's five column groups cannot be reproduced |
| Scale | Materialises the full source-by-target cosine matrix and sorts every row. Tens of MB at 1,363 × 3,226; terabytes at the paper's 1M × 1M. No sub-linear index |
| Reproducibility | Paper: 97.1 recall; code: 0.8955. Paper: 1,300 matches; the code's converter derives 1,167 from the linked archive |
| Stability | Neither the tensor library nor data loaders are seeded; runs differ by up to 1.2pp. The published figure is about sixteen measured standard deviations from the local one |

The frequency-weighted average reproduces exactly, to sixteen decimal places across two tensor-library versions and two tokenizer implementations, because nothing trained reaches its output. The reference is dependable where it learns nothing and not where it trains. Conformance therefore covers recall on small public benchmarks for representations the reference can compute, and nothing about scale: the sub-linear backends are established against this repository's own exact scan.

## The calibration problem

The standard ER benchmark suite (products, publications, restaurants, music, people) contains no company names, nor does the seventeen-dataset universal blocking benchmark (which adds notebooks and census records). Long-text company datasets pair an encyclopedia article with a corporate web page and exercise a document encoder, not short-name blocking.

- **Quality does not transfer.** Vocabulary, error modes and match boundary all differ, so no public number stands in for a company-name number.
- **Harness correctness is still testable,** against a figure the reference code produces locally, not one its paper publishes (see the reproducibility row above). Where the local figure is exact the test can compare vectors; where the reference's run-to-run spread exceeds the difference being read, it settles nothing. Either way it produces no number citable about companies.

`company_perturbation` supplies an internal oracle: a perturbed corpus has exact ground truth and a known difficulty ladder, so it locates where a strategy breaks and against which operator family. Absorption and break-point evaluation, and family-specific robustness metrics, are `Planned`. Precondition: baseline rows are scored from a materialized cleansed column while perturbed rows are cleansed live, so the two arms are computed differently.

## A company-name blocking benchmark

[`architecture.md`](architecture.md) §1 notes company ER is thin, not reproducible outside its papers, or proprietary. GLEIF joined against the single-country registries on jurisdiction and company number is a public, at-scale, ground-truth company-name linkage set, and its split into raw-identical, cleanse-absorbed and cleansed-different pairs stops trivial exact matches dominating a blended figure. The `match_uri` ground truth comes from GLEIF's mandatory registration-authority and company-number fields, so confidence in its validity is high; the residual risk is timing, where a registry and GLEIF snapshot disagree on a record's state.

## Wikidata dump extraction

A different axis: acquisition rather than blocking. [Wikidata Extraction Design Journey](architecture/wikidata-extraction.md) describes `wikisieve`, which scans the full Wikidata JSON dump for companies. The nearest controlled comparison is a peer-reviewed evaluation of four subsetting tools on one server and one dump; the rest are self-reported.

### Published figures

| Tool | Language and parallelism | Task | Dump | Time | Entities/s | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `wikisieve` | Rust, parallel batches | Companies by exact `P31` roots or a `P1454` value in a 49,176-class closure, 612 MB projection | 2026-07-16, 120.9M lines, 154.8 GB | 37.3 min | 54,014 | Full-dump run |
| wikibase-dump-filter (WDF) | Node.js, one process | Life-science subset by `P31`, whole entities written (36 GB) | 2022-01-03, 95.9M items, 102 GB | 3.9 h | about 6,900 | Subsetting evaluation |
| wikidata-cache | Rust, reader thread and parallel workers | Music entities to CSV | About 110M entities, about 130 GB | About 3 h | about 10,000, stated | README |
| KGTK | Python, six threads | Same subset as WDF, import then query | 2022-01-03 | 4.8 h | about 5,600 | Subsetting evaluation |
| WDumper | Java | Same subset, RDF output | 2022-01-03 | 6.5 h | about 4,100 | Subsetting evaluation |
| wikidata-filter | Rust | Labels from N-Triples | Not stated | About 6 h | about 4,200, stated | README |
| WDSub | Scala, ShEx schemas | Same subset, RDF output | 2022-01-03 | 12 h | about 2,200 | Subsetting evaluation |
| wd2sql | Rust, SIMD JSON | Whole dump into SQLite, a conversion rather than a filter | Stated as 1.5 TB | Under 12 h on a 2015 laptop | Not comparable | README |

Rates are each tool's line count over its reported time: an order of magnitude, not a ranking. The full `wikisieve` run averaged 2.2 of 32 available cores. Its prefilter matches each line against a marker's values as well as its property names, rejecting 91.87% of lines before parsing, and it parses only the claims a spec names.

| Comparison | `wikisieve` ratio | Caveat |
| --- | --- | --- |
| wikidata-cache (fastest self-reported Rust) | About 5.4× | Different task and dump |
| WDumper, WDSub | 12 to 25× | Their times include RDF serialisation |
| WDF | About 7.8× | Published rates, WDF's on 2019 Zen 2 cores and `wikisieve`'s on 2014 Haswell cores. The controlled run below measures 9.7 to 16.0× on one machine, with per-core speed unseparated |

| Difference from the evaluation | Detail |
| --- | --- |
| Hardware | Evaluation: 2 AMD EPYC 7302 (64 threads), 320 GB, spinning disk; WDF and WDumper use one core. `wikisieve`: 32-thread workstation, SSD at 1,259 MB/s sequential, 2.2 cores and 1.4 GB used |
| Task | WDF writes 36 GB of whole entities; `wikisieve` about 600 MB of projected fields. The evaluation's filter is exact `P31` on four classes; `wikisieve` also resolves a 49,176-entry closure. The two pull in opposite directions |
| Dump | 95.9M items (2022) against 120.9M lines (2026), with larger entities in the later dump |
| Output format | All read the same Wikibase JSON. WDumper writes gzipped N-Triples, WDSub gzipped Turtle, KGTK includes import and graph-cache build. Only WDF (NDJSON) times selection without a format conversion |
| Decompression | The dump is one gzip member behind a 22-byte stub, inflated on one thread by every tool including `wikisieve`: `rapidgzip-core`'s admission screen decodes the stub, hits end-of-stream and picks the sequential decoder (652 MB slice: 6.53 s CPU at `-P 1`, 6.62 s at `-P 16`). Overriding that choice decodes 1.61× faster with identical output |

Feature support against the same tools is tabulated in [Wikidata Extraction Design Journey](architecture/wikidata-extraction.md#capability-against-other-dump-tools).

### This repository's retired engines

| Engine | Full-dump time |
| --- | --- |
| Python scanner, bz2 | About 34 h (about 1,000 lines/s) |
| Python scanner, gzip | At least 7.6 h (4,398 lines/s, inside the published 2,200 to 6,900 band) |
| First Rust extractor | 91.3 min |
| Spec-driven port with claim cache | 77.9 min |
| Decompression off the main thread and value prefilter | 37.3 min |

### Controlled run against WDF

Both tools on this machine, on one `ext4` volume inside WSL, against the evaluation's own dump and filter, WDF running the command behind its published time. Both select identical entity sets (3,434,538 entities).

| Run | Time (s) | Spread over 3 runs | Speed-up over WDF | Ratio to WDF (evaluation scale) |
| --- | --- | --- | --- | --- |
| WDF, local | 22,731.5 | Single run; an earlier run over `/mnt/c` took 24,819 (8.4% away) | 1 | 1 |
| WDF, published | 13,876 | Evaluation's hardware | 1 | 1 |
| `wikisieve`, whole entities | 1,423.5 | 6.0% | 9.7 to 16.0× | 0.063 to 0.103 |
| `wikisieve`, projection | 1,299.5 | 10.1% | 10.7 to 17.5× | 0.057 to 0.094 |

Each range is bounded by the published and the local WDF time. Neither bound holds beyond this machine: the published time comes from the evaluation's Zen 2 cores and `wikisieve`'s from this machine's Haswell cores, and how far per-core speed moves the ratio is unmeasured. The figure to quote is the published-rate ratio above, about 7.8×, until both tools run on one newer machine. On the evaluation's scale KGTK is 1.24, WDumper 1.69 and WDSub 3.10. Whole entities is the subsetting comparison, since only it writes what WDF writes.

WDF here took 1.6× its published time. Storage is ruled out (WDF consumed 4.8 MB/s of compressed input, 5% of the slowest path); per-core speed is the likeliest remainder, since WDF is single-threaded end to end and this machine's 2014 Haswell cores are slower per core than the evaluation's 2019 Zen 2. That is unmeasured, which is why the speed-up is a range.

| Floor | Measured |
| --- | --- |
| Sequential read, `ext4` volume (64 KiB to 8 MiB blocks) | 97 MB/s |
| Sequential read, NVMe from Windows | 2,090 MB/s |
| Sequential read, NVMe from WSL via `/mnt/c` | 118 MB/s |
| Decompression, `rapidgzip-rust` | 1,121.7 s |
| Decompression, `gzip -d` | 6,080.7 s |

| Tool | Time over its own decompression floor | Output written |
| --- | --- | --- |
| `wikisieve` projection | 1.16× | 927 MB |
| WDF | 3.74× | 37.6 GB |

Single-threaded leg, 200,000-line sample behind the same `gzip -d`, both keeping 886 entities: WDF 73.3 s CPU, `wikisieve` (`RAYON_NUM_THREADS=1`) 3.8 s, about 19× less work per entity.

| Environment | Detail |
| --- | --- |
| CPU and memory | 2 Intel Xeon E5-2640 v3 (16 cores, 32 threads), 64 GB |
| OS | Windows 11 26200, WSL 2.7.14, Ubuntu 22.04.5, kernel 6.18.33.2 |
| Storage | `ext4` vhdx on a Storage Spaces pool over two WD Red WD40EFRX spinning drives |
| Software | `wikisieve` 0.1.0, wikibase-dump-filter 6.1.1 on Node v22.23.2, `gzip` 1.10, `rapidgzip-rust` 0.3.1 |

The crate README records per-leg numbers and commands.

## Priorities

Ordered by evidence gained against effort, given that external quality comparison is unavailable.

1. Progressive recall and its normalized area from existing `matched_edges` artifacts. No new run or dependency; removes the choice of k from every strategy comparison.
2. Absorption and break-point evaluation across perturbation strength, once the live-cleanse precondition is settled.
3. A pyJedAI vector-based block-building run on one country slice.
4. The architecture decision on whether the fused generate-and-score call opens to admit comparison cleaning.
5. One public benchmark as a harness known-answer test: Abt-Buy and Amazon-Google under the untuned sentence encoder against the pretrained-embeddings analysis, then Amazon-Google under SIF and the Autoencoder against a local run of DeepBlocker's reference code.
6. The sub-linear dense index, which the strategy matrix's cells run on (exact scan as reference column) and the sentence-encoder row needs at gb scale. Then the static word-vector ingredient, which gates SIF, Autoencoder and CTT.
7. Within the embedding track: SIF as the untrained control, the Autoencoder over it, then CTT and CTT-cosine, Hybrid after both its halves, sequence-oriented schemes last. The control is the substrate the reconstruction-based families build on, so its result makes their spread measurable.

A published company-name benchmark sits outside this ordering and is decided on its own terms.

## References

- DeepBlocker: [Deep Learning for Blocking in Entity Matching: A Design Space Exploration](https://www.vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf) (PVLDB 14)
- pyJedAI: [AI-team-UoA/pyJedAI](https://github.com/AI-team-UoA/pyJedAI)
- PER: [Progressive Entity Resolution: A Design Space Exploration](https://arxiv.org/abs/2503.08298) (arXiv:2503.08298), [PACMMOD](https://dl.acm.org/doi/10.1145/3709715), [reference implementation](https://github.com/JacobMaciejewski/PER-Design-Space-Exploration)
- Pretrained embeddings for ER: [Pre-trained Embeddings for Entity Resolution: An Experimental Analysis](https://arxiv.org/abs/2304.12329) (arXiv:2304.12329, PVLDB 16), [reference implementation](https://github.com/alexZeakis/Embedings4ER)
- Blocking taxonomy: [Blocking and Filtering Techniques for Entity Resolution: A Survey](https://arxiv.org/abs/1905.06167) (arXiv:1905.06167)
- Universal dense blocking benchmark: [Towards Universal Dense Blocking for Entity Resolution](https://arxiv.org/abs/2404.14831) (arXiv:2404.14831)
- Wikidata subsetting evaluation: [Wikidata subsetting: Approaches, tools, and evaluation](https://www.semantic-web-journal.net/system/files/swj3491.pdf) (Semantic Web 15(6), [publisher page](https://content.iospress.com/articles/semantic-web/sw233491))
- Wikidata dump tools: [wikibase-dump-filter](https://github.com/maxlath/wikibase-dump-filter), [wikidata-cache](https://github.com/WXYC/wikidata-cache), [wikidata-filter](https://github.com/alexkreidler/wikidata-filter), [wd2sql](https://github.com/p-e-w/wd2sql)
