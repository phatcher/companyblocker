# Blocking methods on `gleif -> ie`

What the finished runs on disk say about which truth pairs each method finds, what finding them costs, and how far a difference between two runs can be trusted. Every figure here was read on 2026-09-21 from the tables `scripts/report_strategy_comparison.py` writes, through `analysis.pair_outcomes` and `notebooks/analyse_pair_outcomes.ipynb`, and each table names the runs it read. Artifacts under `artifacts/` are regenerated, so a figure is evidence for the run key beside it and is re-read rather than trusted once that run is remade.

All runs share `top_k` 20, `min_similarity` 0.75, the `cleanse` name transform under the `default` profiles and the exact-name filter. `tfidf`, `wordpiece` and `sentencepiece` ran on the `kmeans` backend and `sbert` (`all-MiniLM-L6-v2`) on `dense_brute`, which scans every target, so an `sbert` miss is never a routing miss and a `kmeans` miss may be. One run per method is read unless a table says otherwise.

## The `never` population is where every miss is

The exact-name joins resolve every truth pair whose names are equal at some form before any scan runs, so recall is 1.0 at each level but the last, and the 626 misses of `tfidf/dcbb9a02a255` all sit in `never`.

| Level | Truth pairs | Found | Recall |
| --- | --- | --- | --- |
| `raw` | 12,237 | 12,237 | 1.000 |
| `basic` | 4,824 | 4,824 | 1.000 |
| `cleansed` | 1,002 | 1,002 | 1.000 |
| `preprocessed` | 267 | 267 | 1.000 |
| `never` | 775 | 149 | 0.192 |
| All | 19,105 | 18,479 | 0.967 |

About half of `never` is out of any name-based method's reach. `random_pair_level` scores each source's cleansed name against other pairs' target names by character 3-gram Jaccard, and its 99th percentile over 20 shuffles is 0.20 to 0.22 across six seeds. Between 362 and 370 of the 775 truth pairs score no higher than that against their own target, so by name they are no different from a non-match, and the four methods below found one or two of them between them. The population a method can be held to is therefore about 410 pairs, and a `never` recall of 0.53 is the ceiling, not 1.0. The level is drawn from these 775 names and belongs to them, not to the method; a single shuffle is too few scores to set it, giving 0.17 to 0.26 across the same seeds.

## Methods at their own settings

| Run | Tokenizer | Candidates | `never` found | `never` precision | `never` recall | `never` F2 | Overall precision | Overall recall | Overall F2 | Mean cluster | Largest cluster | Nodes in clusters over 50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `sbert/b6c9d6c4ef3a` | none | 124,751 | 232 | 0.072 | 0.299 | 0.184 | 0.158 | 0.972 | 0.478 | 7.32 | 18,112 | 30.2% |
| `tfidf/dcbb9a02a255` | none | 52,784 | 149 | 0.141 | 0.192 | 0.179 | 0.362 | 0.967 | 0.725 | 3.36 | 479 | 7.0% |
| `wordpiece/fad36375762f` | `naive` | 35,766 | 94 | 0.194 | 0.121 | 0.131 | 0.533 | 0.964 | 0.830 | 2.69 | 155 | 1.0% |
| `sentencepiece/418502f5dbb7` | `promoted` | 38,564 | 96 | 0.168 | 0.124 | 0.131 | 0.497 | 0.964 | 0.812 | 2.79 | 137 | 1.1% |
| `sentencepiece/ec58527f270c` | `naive` | 33,212 | 78 | 0.167 | 0.101 | 0.109 | 0.575 | 0.964 | 0.849 | 2.60 | 128 | 0.9% |

F2 is computed here from each row's precision and recall and is not a column of the comparison. Overall recall is the same for every method because the joins supply it, so the overall ranking is a precision ranking: the token methods lead, `tfidf` follows and `sbert` is last. On `never` the order reverses on recall and `sbert` and `tfidf` tie on F2, `sbert`'s extra recall being paid for in precision. The comparison's `tokenizer` column reads `wordpiece:promoted` for the `sbert` and `tfidf` runs, neither of which reads a tokenizer.

Cluster size is the connected components of each run's candidate graph, from `blocking.inspection.compute_cluster_size_distribution`. Against a working band in which a block of 10 is conservative, 10 to 50 is reasonable and thousands is not, `sbert` at this threshold fails outright, 30% of its nodes sitting in clusters over 50 and one cluster holding 18,112, and `tfidf` leaves 7% there.

## One threshold is not one cost

Cosine is on a different scale per representation, so runs made at one `min_similarity` do not spend the same comparisons, and a run that keeps more candidates finds more pairs for that reason alone. `cutoffs_for_candidates` reads every run at one budget, and at `wordpiece`'s 35,766 candidates the ranking on `never` changes.

| Read at | `sbert` | `tfidf` | `wordpiece` | `sentencepiece` (`naive`) |
| --- | --- | --- | --- | --- |
| As made, 0.75 for every run | 232 | 149 | 94 | 78 |
| Every run within 35,766 candidates | 69 | 119 | 94 | 78 |
| Cutoff that budget gives | 0.929 | 0.807 | 0.75 | 0.75 |

At equal cost `sbert` finds the fewest `never` pairs of the four and `tfidf` the most, so `sbert`'s lead as made is a candidate set 3.5 times `wordpiece`'s and 2.4 times `tfidf`'s, not harder pairs found. A run is only ever read more tightly than it was made, since the pairs below its threshold were never kept, so a run meant to be read at several cutoffs is made at a low one.

## Which pairs, and how far a difference can be trusted

Read as made over the latest run of each representation, 519 of the 775 pairs are found by no method, 67 by `sbert` alone, 66 by `sbert` and `tfidf`, 35 by all four and 10 by `tfidf` alone; `count_found_by` gives the rest.

Runs of one method differ only in the backend, so every disagreement between them is routing or seed and none is the representation. Thirteen `tfidf` runs on `kmeans`, differing in its start, cluster count and seed, each find 146 to 156 `never` pairs, which reads as stable, but they do not find the same pairs: 91 are found by all thirteen, 92 by some and not others, and 592 by none, the union being 183. Two of them differing only in seed (`tfidf/102b06b7c9db`, `tfidf/2ea742235c5e`) disagree on 25 pairs, and over all couples `run_disagreement` gives a median of 30 and a maximum of 58. A difference between two methods of fewer pairs than that is not a finding: the 10 pairs `tfidf` alone finds above are inside it, and the 67 `sbert` alone finds are outside it. Each `kmeans` run also loses at least 27 of the 183 pairs `tfidf` demonstrably reaches, so its `never` recall understates the representation by 15% or more.

## Across corpora, a cap per source holds cost and a cutoff does not

The same `tfidf` settings on three targets, one run each.

| Target | Run | Target rows | Source rows scanned | `never` pairs | Candidates per source row | Sources at the `top_k` of 20 | `never` recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ie` | `tfidf/dcbb9a02a255` | 817,761 | 23,094 | 775 | 2.29 | 1.5% | 0.192 |
| `gb` | `tfidf/7bbb57cc20ae` | 5,698,275 | 220,039 | 2,535 | 2.76 | 4.6% | 0.073 |
| `offeneregister` | `tfidf/5e5446c0a333` | 5,305,727 | 236,555 | 9,312 | 5.12 | 12.5% | 0.132 |

One cutoff costs more than twice as much per source on `offeneregister` as on `ie`, and the share of sources filling their `top_k` rises eightfold, while the two large targets, near in size, differ by nearly two to one. A cutoff cannot be tuned per corpus in operation, since the corpus is not known in advance. A cap on candidates per source needs no such knowledge, and `pair_found_matrix`'s `max_rank` reads what it would keep.

| Rule, over 0.75 | `ie` per source row | `gb` per source row | `offeneregister` per source row | `never` pairs kept, `ie` / `gb` / `offeneregister` |
| --- | --- | --- | --- | --- |
| As made | 2.29 | 2.76 | 5.12 | 149 / 184 / 1,229 |
| Cap of 5 per source | 1.71 | 1.58 | 2.43 | 149 / 176 / 1,180 |
| Cap of 3 per source | 1.43 | 1.20 | 1.78 | 146 / 173 / 1,138 |

A cap of 3 brings the three corpora within a factor of 1.5 of each other in cost and keeps 98%, 94% and 93% of the `never` pairs each run found. It cannot recover a pair `top_k` already crowded out, and on `ie` a cap of 3 cuts `sbert` from 124,751 candidates to 47,849 for 209 of its 232 pairs. The cap bounds what a source hands on, not the size of a cluster, since sources chain through the targets they share; what a cap does to cluster size is not measured here, because it needs runs made with `max_candidates_per_source` set rather than a re-reading of runs made without it.

## Limits

- `gb` and `offeneregister` hold one `tfidf` run each and `fr` none, so the cross-corpus reading is one method on three targets.
- Every sparse run is on `kmeans`, so routing loss sits under each of their `never` figures and only the within-method reading above bounds it.
- `tfidf/28e7cccfca3b` is left out throughout: it classified 1,042 pairs as `never` where every other run has 775, so it scored a different population.

## Reading it again

```
uv run python scripts/report_strategy_comparison.py
```

That rebuilds every pairing's comparison and pair tables from the runs on disk and runs nothing. `notebooks/analyse_pair_outcomes.ipynb` then reads any pairing's runs as made, at a cutoff per run, at one budget of candidates, or under a cap per source.
