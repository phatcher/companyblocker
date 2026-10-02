# Candidate generation findings

Measured with `scripts/run_blocking.py` unless a section names another script.

## The `kmeans` backend on `gleif -> ie`

All runs `tfidf`, `--min-similarity 0.75`, `--top-k 20`, one 25,000-row chunk, counted under the `preprocessed` name-equality level, which leaves 775 true pairs at `never`. Target 817,761 rows. The runs predate the current setting names and recorded the start as `kmeans_init`: `smart`, `sample` or `random`.

| Clusters, start, start rows | Run | Fit | Fit peak memory | `never` found | Recall | Precision | `recall_area_never` |
|---|---|---|---|---|---|---|---|
| 50, `spread`, all | `699aca9d1759` | 99.4 s | not recorded | 150 | 0.194 | 0.158 | 0.1221 |
| 300, `spread`, 50,000 | `2ed5a1661d59` | 220.7 s | 3.21 GiB | 156 | 0.201 | 0.157 | 0.1318 |
| 900, `random`, all, seed 42 | `102b06b7c9db` | 885.2 s | 3.73 GiB | 146 | 0.188 | 0.142 | 0.1271 |
| 900, `random`, all, seed 1 | `848e80067cbe` | 854.7 s | not recorded | 150 | 0.194 | 0.149 | 0.1278 |
| 900, `random`, all, seed 2 | `2ea742235c5e` | 704.7 s | 3.82 GiB | 147 | 0.190 | 0.146 | 0.1271 |
| 900, `spread`, 50,000 | `444fdad44e8e` | 906.6 s | 3.68 GiB | 150 | 0.194 | 0.152 | 0.1279 |
| 900, `spread`, all | `dcbb9a02a255` | 2,104.4 s | 3.64 GiB | 149 | 0.192 | 0.141 | 0.1307 |
| 2,400, `spread`, 50,000 | `aa02fa4992b2` | 2,517.6 s | 5.19 GiB | 146 | 0.188 | 0.134 | 0.1296 |
| `lsh`, defaults | `2f6c2bad0ed8` | 124.6 s | not recorded | 114 | 0.147 | 0.141 | 0.1036 |

- The start does not change what is found: the three starts at 900 clusters found 149, 146 and 150, inside the 146-150 spread that three seeds of one start give. It changes the fit's cost. The spread start over a 50,000-row sample reaches the inertia of the spread start over every row in under half the time, and `random` is cheapest and fits loosest.
- More clusters found no more pairs: 300 found more than 900 or 2,400 and had the best `recall_area_never`. Scoring takes 5-10 s at every count while the fit grows with the count.
- Every `kmeans` configuration beat `lsh`: 146-156 pairs against 114, and `recall_area_never` 0.122-0.132 against 0.104.
- The mini-batch fit is not measured.

## How the `kmeans` fit's cost grows with target size

300 clusters, the spread start over a 50,000-row sample, on each target's own `gleif` pairing.

| Target | Target rows | Fit | Per row per cluster | Fit peak memory | Per million rows |
|---|---|---|---|---|---|
| `ie` | 817,761 | 220.7 s | 0.90 us | 3.21 GiB | 3.9 GiB |
| `gb` | 5,698,275 | 1,588.9 s | 0.93 us | 21.28 GiB | 3.7 GiB |
| `offeneregister` | 5,305,727 | 1,773.2 s | 1.11 us | 23.26 GiB | 4.4 GiB |

- Fit time and peak memory both grow about linearly with target rows, to within the roughly 20% a target's own vocabulary moves them.
- `fr` (12,930,798 rows) is not measured. At about 52 GiB of peak memory on a 64 GiB machine the run would page.
- At `gb` and `offeneregister` size the fit is not the only cost: `name_forms` took 693 s and 800 s there, against 0.4 s on `ie`, where it came from a cache.

## Prefix filtering on `gleif -> ie`

The L2AP bound admits 2.6% of the pairs, but scoring them pair by pair in Python is slower than the exhaustive scan's compiled sparse multiply, which touches only the columns two rows share. The filter is off by default for that reason. The filtered and the exhaustive run returned the same 35,939 candidate pairs. The filtered run took 21 s to generate candidates and about 200 s to score them, against 339 s for the exhaustive scan.

## A short name on the source side

Reading `short_name` on the source side while the target stays on `name` leaves recall flat, between -0.0001 and +0.0004, and raises precision by 5 to 7 points, from 25% to 35% fewer false-positive candidates.

## Initialisms across records

Measured with `scripts/measure_initialism_recall.py`. A matched pair where one name is the initials of the other is 50 of 391,684 real GLEIF matched pairs, about 0.013%, and 0.77% of the 6,462 pairs sharing no token at all.

## Word-level prefix and suffix divergence

Measured with `scripts/measure_prefix_suffix_divergence.py`. A matched pair whose token sequences differ by a bounded leading or trailing run (`Cisco Systems Inc` against `Cisco`) is about 0.6% of real GLEIF matched pairs, lowest for `gb` at about 0.2%. Cosine similarity over shared tokens recovers almost none of them.
