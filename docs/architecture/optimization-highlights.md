# Cross-Cutting Optimization Log Highlights

Part of [CompanyBlocker Architecture](../architecture.md): detail behind [§8](../architecture.md#8-cross-cutting-optimization-log-highlights).

`docs/optimizations.md` is the full, append-only log every benchmarked change is recorded to (see [§1](../architecture.md#1-executive-summary)). This section covers two patterns from it that recur across areas. Entries covered per-system in [acquisition journeys](acquisition-journeys.md), including GLEIF's LRU cache, the trie matcher and the rejected schema-hint attempt, are not repeated here.

## Six measured changes compounding to 34.1%

The cleanser's `name_cleanse` path (`src/acquisition/cleanser.py` at the time; the engine now lives in the standalone `company_cleanse` package, see below) improved from **9,372 rows/s to 15,327 rows/s, a 34.1% cumulative gain, across six separately benchmarked changes** over 2026-06-19/20:

1. Removing a duplicate diacritic-strip pass in the transliteration path: 9,372 → 11,298 rows/s (+20.5%).
2. A lazy-ASCII-fallback attempt measuring as a wash (9,492 → 9,372 rows/s, within noise), kept as groundwork for later changes: a neutral result on its own sample is not a harmful one.
3. Single-pass quoted-parenthesized extraction, removing three redundant calls to the same extraction logic: 9,771 → 10,637 rows/s (+8.9%).
4. Reusing an already-computed normalized name for suffix surface instead of a second normalization pass: 11,429 → 11,652 rows/s (+2.0%).
5. Vectorizing two `map_elements` Python-UDF token-validity checks into native Polars string expressions: 11,429 → 11,947 rows/s (+4.5%).
6. Single-pass lead-token extraction, struct-returning and replacing two separate UDF passes, plus extracting the lead-token decision logic into one unit-testable step: 11,429 → 15,327 rows/s (+34.1% combined), the largest jump in the chain.

Every step was validated against the same deterministic 20%-sample real-files regression baseline before being kept, and the baseline was re-promoted afterwards with a recorded dataset fingerprint (`selected_file_list_sha256`), so later regression checks compare like-for-like runs.

## Bugs surfaced by the module split

The 2026-06-20 session that landed the trie company-type matcher (see [acquisition journeys](acquisition-journeys.md)) also split the ~1,400 line `cleanser.py` monolith into five modules (`cleanser_normalize`, `cleanser_rules`, `cleanser_extract`, `cleanser_pipeline`, `cleanser_ops`) for maintainability, with no algorithmic change and the benchmark logged as not applicable.

That split has since been superseded: the cleanse engine was extracted into the standalone `company_cleanse` package, where `normalize.py`, `rules.py`, `extract.py` and `pipeline.py` now live, with `src/acquisition/cleanser_ops.py` and `cleanser_orchestrate.py` remaining as the acquisition-side orchestration wrapper.

The refactor surfaced two latent bugs unrelated to its purpose: a recursive `name_tokenize` wrapper, and a broken rule import in `private_analysis.py` that had silently disabled duplicate-rule detection in proposal generation. Both became visible once the code sat in isolated, individually-testable modules.
