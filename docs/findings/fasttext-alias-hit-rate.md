# fastText alias hit rate

Measured with `scripts/measure_fasttext_alias_hit_rate.py` on the `en-cc-300` checkpoint over Wikidata's GB name variants.

- A 37.5% top-1 hit rate: 4,989 of 13,291 anchors, in a pool of 19,111 distinct names.
- Scoring anchors in batches holds `chunk_size x pool_size` floats at a time. A pool-by-pool matrix would be quadratic in the pool: roughly 1.5 TiB for Wikidata's FR jurisdiction (456,699 distinct names) and 16 GiB for DE's 46,586.
- At the default `chunk_size` of 500, GB's pool holds well under 0.3 GiB, DE's under 0.6 GiB and FR's under 5.5 GiB. These footprints are computed from the pool sizes and the arrays the probe holds (the similarity batch, the same-shape partition index array and the pool's 300-dimensional vectors), not measured from a run.
