# Optimizations Log

This document tracks performance and code optimizations over time.

## Entry Template
- Date: YYYY-MM-DD
- Area: file or subsystem
- Change: short description of what changed
- Why: hotspot or motivation
- Validation: tests or checks run
- Benchmark: before/after metrics (include command/config)
- Outcome: impact summary
- Notes: follow-up ideas

## 2026-07-08 - GLEIF Schema Index Caching Optimization (LRU Cache)

- Date: 2026-07-08
- Area: `src/acquisition/sharding_gleif.py` (`_gleif_company_schema_indexes()`)
- Change: Added `@lru_cache(maxsize=1)` decorator to `_gleif_company_schema_indexes()` to cache the XSD-derived allowed_paths frozenset and sanitization map, avoiding per-record dict reconstruction overhead. This function was being called 200K+ times per shard pipeline with identical outputs.
- Why: Baseline profiling (2026-07-08 entry) identified schema index rebuilding as a micro-hotspot; each of 200K records was triggering expensive set/frozenset construction from the same XSD paths. LRU cache trades minimal memory (one cached frozenset + one cached dict) for per-record computational savings.
- Validation:
  - `src/tests/acquisition/test_sharding_gleif.py` (14 passed, cache fixture clearing confirms no cache leakage between tests)
  - Canonical tests: `test_canonicalize_system_shards_maps_gleif_fields`, `test_canonicalize_system_shards_uses_latest_available_snapshot_when_run_date_omitted` (both PASSED)
- Benchmark:
  - Command/config: `.venv\Scripts\python.exe scripts/process_companies.py --systems gleif --processes shard --force --additional-args "shard.max_rows=200000"` (same as baseline)
  - Input: 200,000 LEI records (first 2 chunks from gleif-lei-cdf-2026-06-17.zip, 497 MB source)
  - Before (baseline, no caching): 200,000 rows in 42.35s (4,723 rows/s median)
  - After (with @lru_cache): 200,000 rows in 30.30s (6,601 rows/s median)
  - Delta: -12.05s (-28.5% wall time), +1,878 rows/s (+39.8% throughput)
  - Chunk-level throughput (with caching):
    - Chunk 1: 6,990 rows/s
    - Chunk 2: 6,317 rows/s
    - Variance: -9.6% (vs baseline 40% degradation)
  - Phase breakdown (with caching, total 30.30s):
    - Extract: 12.77s (42%)
    - Write: 2.73s (9%)
    - Other: 14.80s (49%)
  - Phase improvements (all ~28%):
    - Extract: 12.77s vs 17.73s (-28.0%)
    - Write: 2.73s vs 3.92s (-30.4%)
    - Other: 14.80s vs 20.70s (-28.5%)
- Outcome: Improvement across all phases with no algorithmic change. Per-chunk variance dropped from 40% to ~10%. Full 1.8M-row GLEIF ZIP would now take ~4.55 hours (vs ~6 hours baseline) based on this 200K sample.
- Notes:
  - Cache hit rate is effectively 100% for a single shard pipeline since `_gleif_company_schema_indexes()` is deterministic (depends only on XSD content, not input data).
  - Two remaining high-impact optimizations: (1) "Other" phase (49% of cost) could benefit from batch ZIP iteration or parallel record extraction; (2) XML extraction already uses C-based ElementTree but could explore streaming parsers or lxml for larger documents.
  - Chunk throughput stability improvement suggests this fix removes a cumulative cost pattern; next profile would focus on whether Extract phase can be parallelized.

## 2026-07-08 - GLEIF DataFrame Schema Hint Attempt (REJECTED - Caused Regression)

- Date: 2026-07-08
- Area: `src/acquisition/sharding_gleif.py` (DataFrame construction)
- Change: Attempted optimization #2: pre-define Polars DataFrame schema from XSD-derived allowed fields to skip type inference. Added `_gleif_polars_schema()` function to build `dict[str, pl.DataType]` mapping all sanitized field names to `pl.String`, then passed `schema=polars_schema` parameter to `pl.DataFrame()` instead of `infer_schema_length=None`.
- Why: Type inference requires scanning all rows to determine column types; pre-specifying schema should skip this step, especially for sparse/dynamic GLEIF records where not all fields are present in each chunk.
- Validation:
  - Tests passed (14/14)
  - But benchmark showed **21% performance regression**
- Benchmark:
  - Command/config: Same as baseline (200K rows, shard.max_rows=200000)
  - Before (schema caching baseline): 200,000 rows in 30.30s (6,601 rows/s)
  - After (with pre-defined schema): 200,000 rows in 36.68s (5,453 rows/s)
  - Delta: +6.38s (+21% slower)
  - Phase breakdown (with schema):
    - Extract: 15.45s (42%)
    - Write: 3.44s (9%)
    - Other: 17.78s (49%)
- Outcome: **Rejected**, on the measured regression. Pre-defining all ~200+ XSD fields adds more overhead than skipping inference saves, since Polars' type inference is already suited to sparse dataframes.
- Notes:
  - Root cause: GLEIF records are sparse (most fields are missing per record), so dynamic schema inference is actually faster than building and matching against a comprehensive pre-defined schema.
  - Learning: Avoid pre-optimization of sparse/dynamic schemas; let Polars infer from actual data.
  - Optimization #6 (country lookup frozenset) was kept and validated: 30.45s (6,569 rows/s), within 0.5% variance of baseline.

## 2026-07-03 - Wikidata Raw vs Raw-NoSplit Retest Policy Note
- Date: 2026-07-03
- Area: `scripts/benchmark_wikidata_line_filter.py`, `src/acquisition/wikidata_pipeline_helpers.py`
- Change: Added an explicit benchmark note and inline matcher comment recording that `raw` and `raw_nosplit` are near-parity on the current 200K stream benchmark slice.
- Why: Avoid spending benchmark cycles repeatedly validating a delta that is currently negligible.
- Validation:
  - `.venv\Scripts\python.exe scripts/benchmark_wikidata_line_filter.py --mode two-pass --input raw/wikidata/acquire/2026-07-02/wikidata-all-2026-07-02.json.bz2 --max-lines 200000 --runs 1 --scan-mode full --wiring-modes stream --candidate-matcher raw`
  - `.venv\Scripts\python.exe scripts/benchmark_wikidata_line_filter.py --mode two-pass --input raw/wikidata/acquire/2026-07-02/wikidata-all-2026-07-02.json.bz2 --max-lines 200000 --runs 1 --scan-mode full --wiring-modes stream --candidate-matcher raw_nosplit`
- Benchmark:
  - raw: 55.07s, rows_written=1,458, output_bytes=872,822
  - raw_nosplit: 54.92s, rows_written=1,458, output_bytes=872,822
  - Delta: -0.15s (~0.27%)
- Outcome: Treat `raw` and `raw_nosplit` as perf-equivalent for routine work; do not re-run this comparison unless matcher internals or stream line-iteration logic changes.
- Notes: Keep using script-parity benchmarks for claims that affect end-to-end runtime decisions.

## 2026-07-03 - Wikidata Stream Matcher No-Split Variant
- Date: 2026-07-03
- Area: `src/acquisition/wikidata_pipeline_helpers.py`, `scripts/benchmark_wikidata_line_filter.py`, `src/tests/acquisition/test_downloader_wikidata.py`
- Change: Added a new candidate matcher mode `raw_nosplit` that keeps the existing raw-byte semantics but avoids the `any(map(...))` path and exposes the mode through benchmark CLI. Added a parity regression test to confirm `raw_nosplit` output matches `raw` in stream two-pass mode.
- Why: 200K-line stream profile showed candidate scanning as the dominant CPU hotspot; targeted a low-risk micro-optimization on the matcher path.
- Validation:
  - `src/tests/acquisition/test_downloader_wikidata.py` (13 passed)
  - Benchmark parity run in stream mode at 200K lines with same source and config
- Benchmark:
  - Command/config (raw): `.venv\Scripts\python.exe -m cProfile -o artifacts/perf/wikidata_stream_raw_200k_new.pstats scripts/benchmark_wikidata_line_filter.py --input <latest wikidata bz2> --mode two-pass --wiring-modes stream --candidate-matcher raw --scan-mode full --max-lines 200000 --runs 1`
  - Command/config (raw_nosplit): `.venv\Scripts\python.exe -m cProfile -o artifacts/perf/wikidata_stream_raw_nosplit_200k.pstats scripts/benchmark_wikidata_line_filter.py --input <latest wikidata bz2> --mode two-pass --wiring-modes stream --candidate-matcher raw_nosplit --scan-mode full --max-lines 200000 --runs 1`
  - Before (raw): elapsed 56.20s
  - After (raw_nosplit): elapsed 55.11s
  - Delta: -1.09s (~1.9% faster)
  - Matcher hotspot cumtime: 26.04s (`raw`) -> 25.57s (`raw_nosplit`)
- Outcome: Small but consistent gain with no behavioral change in the tested scenario; candidate matching remains a major hotspot and I/O iteration is still dominant overall.
- Notes: Next profiling/optimization focus should be `io_eta_helpers._iter_raw_lines_from_handle` and line handling in the streaming loop.

## 2026-07-02 - Wikidata Two-Pass Optimization (Fast P31 Filter + Full Projection)
- Date: 2026-07-02
- Area: `src/acquisition/downloader_wikidata.py`, `src/acquisition/pipeline.py`, `src/acquisition/sharding.py`
- Change: Introduced `_fast_p31_filter_wikidata()` (Pass 1) and `extract_wikidata_company_projection_two_pass()` orchestrator. Pass 1 filters for P31 (instance_of) property at decompression speed (~12k lines/sec) before full JSON parsing. Pass 2 reads pre-filtered intermediate and performs full entity extraction and projection (~4-5k lines/sec). Updated pipeline.py and sharding.py to call two-pass version.
- Why: Single-pass achieve only 3.8k lines/sec due to CPU-bound JSON parsing/filtering on every line; decompression can sustain 12k+ lines/sec. Separating concerns allows Pass 1 to run at decompression speed, with Pass 2 operating on much smaller dataset (30-40% of raw lines have P31).
- Validation:
  - Syntax validation: `python -m py_compile` on all updated files (passed)
  - Full run test on 5.9M lines (no crash, stable throughput 3.8-7k lines/sec)
  - indexed_bzip2 v1.7.0 confirmed working at production scale
- Benchmark:
  - Diagnostic scanner baseline: 1M lines in 77.2s = 12,958 lines/sec (decompression only, no parsing)
  - Single-pass shard on 5.9M lines: 77.1s elapsed, 3.8k lines/sec (full parsing/filtering)
  - Expected two-pass: ~2.8 hours for Pass 1 (120M lines at 12k/sec) + ~3 hours for Pass 2 (50M filtered at 4.5k/sec) = ~5.8 hours total
  - Estimated improvement: **31% faster than single-pass** (vs 8.7 hours single-pass)
- Outcome: Ready for production validation. Two-pass design separates I/O-bound decompression from CPU-bound parsing, allowing both to run at optimal speed. By-product: will discover actual Wikidata company count (~40-50K expected from 120M entities).
- Notes: Intermediate file cleaned up after Pass 2 completes (configurable). If Pass 2 crashes, intermediate is preserved for retry. Next: run full 102GB file and measure actual speedup and final company count from Wikidata.

## 2026-07-02 - Wikidata Raw-Line Candidate Prefilter
- Date: 2026-07-02
- Area: `src/acquisition/downloader_wikidata.py`, `src/acquisition/sharding.py`, `src/tests/acquisition/test_downloader_wikidata.py`, `src/tests/acquisition/test_sharding.py`
- Change: Added a cheap raw-line candidate filter for Wikidata prepare before JSON decoding, while keeping raw-line counting and `max_rows` semantics intact. Also kept the strict prepare-only shard source contract and added a regression test for the `shard.max_rows` pass-through.
- Why: Profiled Wikidata prepare runs showed JSON parsing and full claim traversal were still being paid on too many discarded rows; the raw-line heuristic reduces that cost before decode.
- Validation:
  - `src/tests/acquisition/test_downloader_wikidata.py` (passed)
  - `src/tests/acquisition/test_sharding.py::test_shard_system_source_passes_max_lines_to_wikidata_prepare` (passed)
  - `src/tests/acquisition/test_sharding_wikidata.py` (passed)
- Benchmark:
  - Command/config: `.venv\Scripts\python.exe scripts/process_companies.py --systems wikidata --processes shard --allow-research --force --additional-args shard.max_rows=100000`
  - Before: 100,000 lines in 125.6s, line_rate=796/s
  - After: 100,000 lines in 99.9s, line_rate=1,001/s
- Outcome: Prepared Wikidata throughput improved by about 26% on the same bounded slice while preserving the strict stage boundary behavior.
- Notes: The next bottleneck is still the cost of parsing and scanning the surviving company candidates; a deeper raw-text P31 precheck may be possible, but it needs careful semantics validation.

## 2026-06-20 - Failed Candidate: Native Replace For Company-Type Mapping
- Date: 2026-06-20
- Area: `packages/company_cleanse/src/company_cleanse/pipeline.py` (`_effective_company_type` -> `company_type` mapping step)
- Change: Replaced Python UDF mapping
  - from: `map_elements(lambda s: company_type_mapping.get(s, s) if s else None)`
  - to: native Polars `replace_strict(company_type_mapping, default=pl.col("_effective_company_type"))`
- Why: Remove one remaining Python UDF stage in optimized plan and reduce row-wise Python overhead.
- Validation:
  - `packages/company_cleanse/tests/test_pipeline.py` (passed)
  - `packages/company_cleanse/tests/test_api.py` (passed)
  - `src/tests/test_cleanser_pipeline.py` (passed)
  - `src/tests/test_cleanser_orchestrate.py` (passed)
- Benchmark:
  - Command/config: `PERF_BENCHMARK=1`, `PERF_SYSTEM=gb`, `PERF_REAL_INPUT_LAYER=canonical`, `PERF_REAL_INPUT_GLOB=raw/gb/canonical/**/*.parquet`, `PERF_REAL_MAX_FILES=2`, `PERF_RUNS=3`, `PERF_COMPANY_TYPE_MATCHER=trie`
  - Plan-level change: python UDF count decreased from 10 -> 9
  - Throughput regressed to 11,728 rows/s (median elapsed 17.053s)
  - Prior trie run on same 2-file slice was ~14.9k rows/s
- Outcome: **Failed optimization**. One fewer Python UDF in the plan, and end-to-end throughput worsened.
- Notes: Reverted immediately (`git checkout -- packages/company_cleanse/src/company_cleanse/pipeline.py`). Do not re-attempt this exact substitution without deeper engine-level profiling to explain the regression.

## 2026-06-20 - Add Trie Company-Type Matcher And Set As Default
- Date: 2026-06-20
- Area: `packages/company_cleanse/src/company_cleanse/{rules.py,pipeline.py,config.py}`, `src/acquisition/cleanser_orchestrate.py`
- Change: Added a reversed-token trie matcher for company-type suffix matching, kept the existing regex matcher in parallel, introduced a switchable matcher mode (`regex|trie`), and changed runtime default to `trie`.
- Why: The large suffix regex is expensive in tight loops; trie lookup is a simpler longest-suffix-match primitive over pre-normalized tokens.
- Validation:
  - `packages/company_cleanse/tests/test_rules.py` (passed)
  - `packages/company_cleanse/tests/test_pipeline.py` (passed)
  - `packages/company_cleanse/tests/test_api.py` (passed)
  - `src/tests/test_cleanser_pipeline.py` (passed)
  - `src/tests/test_cleanser_orchestrate.py` (passed)
  - Added low-level matcher parity tests and in-memory perf test (`packages/company_cleanse/tests/test_company_type_matcher_performance.py`).
- Benchmark:
  - Primitive matcher cost (in-memory, no parquet/pipeline I/O):
    - Command/config: `PERF_BENCHMARK=1`, `PERF_ROWS=200000`, `pytest packages/company_cleanse/tests/test_company_type_matcher_performance.py -q -s`
    - Regex: 27,123 rows/s (7.374s)
    - Trie: 430,288 rows/s (0.465s)
    - Speedup: 15.864x
  - End-to-end pipeline (real files, GB canonical, 2 files, 200k rows, 3 runs):
    - Regex: 14,182.99 rows/s, median elapsed 14.101s
    - Trie: 14,912.99 rows/s, median elapsed 13.411s
    - Improvement: +730 rows/s (+5.15%)
- Outcome: The trie cuts primitive matcher cost 15.9x; the end-to-end gain is smaller because of non-matcher pipeline costs.
- Notes: Suffix matching requires pre-normalized surfaces (case/diacritic/punctuation/whitespace normalization) before invoking either matcher.

## 2026-06-19 - Lazy ASCII Fallback For Suffix Path
- Date: 2026-06-19
- Area: `src/acquisition/cleanser.py` (`name_cleanse` suffix classification path)
- Change: Compute `_suffix_surface_ascii` and `_raw_company_type_ascii` only when `_raw_company_type` is null, instead of unconditionally.
- Why: Profiling showed suffix normalization/transliteration as a major hotspot; reduce unnecessary per-row transliteration work.
- Validation:
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_tokenize_harness.py` (passed)
- Benchmark:
  - Command/config: `PERF_BENCHMARK=1`, `PERF_SYSTEM=gb`, `PERF_REAL_INPUT_LAYER=source`, `PERF_REAL_MAX_FILES=3`, `PERF_RUNS=1`, no profiler
  - Before: 9,492 rows/s, median elapsed 31.604s
  - After: 9,372 rows/s, median elapsed 32.011s
- Outcome: No improvement on this sample; slightly slower and likely within run-to-run variance.
- Notes: Next likely optimization is to parse quoted parenthesized parts once and reuse tuple fields instead of calling extraction logic multiple times.

## 2026-06-19 - Remove Duplicate Diacritic Strip In Transliteration Path
- Date: 2026-06-19
- Area: `src/acquisition/cleanser.py` (`_normalize_suffix_surface`)
- Change: In transliteration mode, call `_transliterate_to_ascii` directly instead of first calling `_strip_diacritics`; this removes a redundant full-string pass.
- Why: Profiling showed normalization/transliteration as a hotspot; transliteration path was stripping diacritics twice.
- Validation:
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_tokenize_harness.py` (passed)
- Benchmark:
  - Command/config: `PERF_BENCHMARK=1`, `PERF_SYSTEM=gb`, `PERF_REAL_INPUT_LAYER=source`, `PERF_REAL_MAX_FILES=3`, `PERF_RUNS=1`, no profiler
  - Before: 9,372 rows/s, median elapsed 32.011s
  - After: 11,298 rows/s, median elapsed 26.553s
- Outcome: Throughput gain on the same sample and config.
- Notes: Keep this as the current baseline candidate and continue profiling around Python UDF-heavy map_elements paths.

## 2026-06-19 - Single-Pass Quoted Parenthesized Extraction
- Date: 2026-06-19
- Area: `src/acquisition/cleanser.py` (`name_cleanse` special-case extraction path)
- Change: Replaced three separate calls to `_extract_quoted_type_parenthesized_parts` with one call that returns a struct and reused its fields for `_special_short_name`, `_special_company_type`, and `_special_full_name`.
- Why: Profiling showed repeated Python UDF work in map_elements lambdas; this removed duplicate per-row regex/normalization/parsing for the same input.
- Validation:
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_cleanser_replacement_sequence.py` (passed)
- Benchmark:
  - Command/config: canonical real-files regression sample, deterministic 20% random selection (`seed=20260619`), `PERF_BENCHMARK=1`, `PERF_SYSTEM=gb`, `PERF_RUNS=1`, `PERF_UPDATE_BASELINE=0`
  - Before (canonical baseline): 9,771 rows/s, median elapsed 122.640s
  - After: 10,637 rows/s, median elapsed 112.655s
- Outcome: Throughput improved by 8.86% and elapsed time reduced by 8.14% on the same sampling method.
- Notes: Keep profiling map_elements-heavy paths (suffix normalization and acronym/short-name derivation remain prominent).

## 2026-06-20 - Reuse Normalized Name For Suffix Surface
- Date: 2026-06-20
- Area: `src/acquisition/cleanser.py` (`name_cleanse` suffix extraction path)
- Change: Removed the second `_normalize_suffix_surface(... keep_bang=False)` Python UDF pass and derived `_suffix_surface` from `_normalized_name_body` using Polars string operations.
- Why: Profiling identified duplicate normalization work in adjacent map_elements stages.
- Validation:
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_cleanser_replacement_sequence.py` (passed)
  - `src/tests/test_cleanser_normalize_tokens.py` (passed)
- Benchmark:
  - Command/config: deterministic canonical real-files sample (20%, `seed=20260619`), `PERF_BENCHMARK=1`, `PERF_BENCHMARK_MODE=regression`, `PERF_SYSTEM=gb`, `PERF_RUNS=3`, `PERF_UPDATE_BASELINE=0`
  - Before (baseline): 11,429 rows/s, median elapsed 104.842s
  - After: 11,652 rows/s, median elapsed 102.843s
- Outcome: Throughput improved by 1.95% and elapsed time reduced by 1.91% while preserving behaviour.
- Notes: Next target is replacing additional map_elements token checks with native Polars expressions.

## 2026-06-20 - Vectorised Lead Token Validity Checks
- Date: 2026-06-20
- Area: `src/acquisition/cleanser.py` (`name_cleanse` lead-token validation path)
- Change: Replaced Python `map_elements` calls for `_is_valid_short_name_token` and `_is_acronym_like_token` with native Polars `str.replace_all`, `str.len_chars`, and `str.contains` expressions.
- Why: Reduce high-frequency Python UDF overhead in short-name and quoted-name gating logic.
- Validation:
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_cleanser_replacement_sequence.py` (passed)
  - `src/tests/test_cleanser_normalize_tokens.py` (passed)
  - `src/tests/test_cleanser_acronym.py` (passed)
- Benchmark:
  - Command/config: deterministic canonical real-files sample (20%, `seed=20260619`), `PERF_BENCHMARK=1`, `PERF_BENCHMARK_MODE=regression`, `PERF_SYSTEM=gb`, `PERF_RUNS=3`, `PERF_UPDATE_BASELINE=0`
  - Before (baseline): 11,429 rows/s, median elapsed 104.842s
  - After: 11,947 rows/s, median elapsed 100.298s
- Outcome: Throughput improved by 4.53% and elapsed time reduced by 4.33% while preserving behaviour.
- Notes: Next likely win is reducing `_extract_leading_delimited_tokens`/`_extract_leading_quoted_tokens` Python extraction passes where Polars regex extraction can preserve semantics.

## 2026-06-20 - Single-Pass Lead Token Extraction
- Date: 2026-06-20
- Area: `src/acquisition/cleanser.py` (`name_cleanse` lead-token extraction path)
- Change: Replaced two separate `_stripped_name` UDF passes (`_extract_leading_delimited_tokens` and `_extract_leading_quoted_tokens`) with one struct-returning pass (`_extract_leading_token_struct`) and projected both lead fields from that struct.
- Why: Remove duplicated parsing/regex work while preserving existing extraction semantics.
- Validation:
  - `src/tests/test_cleanser_lead_tokens.py` (passed)
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_cleanser_acronym.py` (passed)
  - Query plan snapshot: `src/tests/test_cleanser_performance.py::test_name_cleanse_query_plan_snapshot_real_files` (passed)
- Benchmark:
  - Command/config: deterministic canonical real-files sample (20%, `seed=20260619`), `PERF_BENCHMARK=1`, `PERF_BENCHMARK_MODE=regression`, `PERF_SYSTEM=gb`, `PERF_RUNS=3`, `PERF_UPDATE_BASELINE=0`
  - Before (baseline): 11,429 rows/s, median elapsed 104.842s
  - After: 14,394 rows/s, median elapsed 83.251s
- Outcome: Throughput improved by 25.94% and elapsed time reduced by 20.60% while preserving behaviour.
- Notes: Next likely target is reducing `_name_body_from_leading_token` and `_move_trailing_the_to_front` Python UDF passes with equivalent Polars transformations if plan snapshots do not show adverse allocation expansion.

## 2026-06-20 - Extract Lead Token Decision Step UDF
- Date: 2026-06-20
- Area: `src/acquisition/cleanser.py` (`name_cleanse` lead-token decision path)
- Change: Extracted short/quoted lead-token decision logic into `_derive_lead_token_decisions` and replaced inline Polars branching with one unit-testable step-UDF call returning a structured decision payload.
- Why: Reduce drift risk between inline pipeline logic and Python helper semantics by centralizing this stage into one testable function.
- Validation:
  - `src/tests/test_cleanser_lead_tokens.py` (passed)
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_cleanser_acronym.py` (passed)
  - Query plan snapshot: `src/tests/test_cleanser_performance.py::test_name_cleanse_query_plan_snapshot_real_files` (passed)
- Benchmark:
  - Command/config: deterministic canonical real-files sample (20%, `seed=20260619`), `PERF_BENCHMARK=1`, `PERF_BENCHMARK_MODE=regression`, `PERF_SYSTEM=gb`, `PERF_RUNS=3`, `PERF_UPDATE_BASELINE=0`
  - Before (baseline): 11,429 rows/s, median elapsed 104.842s
  - After: 15,327 rows/s, median elapsed 78.180s
- Outcome: Throughput improved by 34.10% and elapsed time reduced by 25.43% while preserving behaviour.
- Notes: Next extraction candidate is `_name_body_from_leading_token`/`_move_trailing_the_to_front` as a unified step-UDF with similar parity/performance gates.

## 2026-06-20 - Baseline Refresh After Lead-Token Decision Extraction
- Date: 2026-06-20
- Area: `artifacts/perf/cleanse_real_files_baseline.json` (performance governance)
- Change: Promoted a new real-files regression baseline after the lead-token decision extraction shipped and correctness/perf gates remained green.
- Why: Keep future regressions anchored to current code and a reproducible dataset definition, rather than comparing against a pre-optimisation baseline.
- Validation:
  - `src/tests/test_cleanser_performance.py::test_name_cleanse_benchmark_real_files` (passed, baseline updated)
  - Deterministic file sampling: canonical layer, 20% sample, `seed=20260619`
- Benchmark:
  - Command/config: `PERF_BENCHMARK=1`, `PERF_BENCHMARK_MODE=regression`, `PERF_SYSTEM=gb`, explicit `PERF_REAL_INPUT_FILES` from deterministic sample, `PERF_RUNS=3`, `PERF_UPDATE_BASELINE=1`
  - New baseline: 12,551 rows/s, median elapsed 95.474s, rows 1,198,275
  - Dataset fingerprint: `selected_file_list_sha256=8360bf8bd881561628b51408361b9ced3e6d3a76e2eefebe9e2b9278cbf8d272`
- Outcome: Baseline has been refreshed with reproducible dataset metadata, so subsequent regression checks compare like-for-like runs.
- Notes: Throughput varies across host load; baseline promotion was based on governance criteria (mode isolation + deterministic dataset + passing checks), not on a single best run.

## 2026-06-20 - Polars Name-Body Expression Refactor
- Date: 2026-06-20
- Area: `src/acquisition/cleanser.py` (`name_cleanse` name-body and trailing-THE reorder path)
- Change: Replaced two Python `map_elements` stages with native Polars string expressions:
  - removed UDF-based lead-delimited prefix stripping for `_name_body`
  - removed UDF-based trailing `(THE)` reorder for `_name_body_reordered`
  - removed now-unused helpers `_strip_leading_delimited_prefix`, `_name_body_from_leading_token`, and `_move_trailing_the_to_front`
- Why: Continue reducing Python UDF overhead in the hottest cleanse path while preserving established casing and `(THE)` semantics.
- Validation:
  - `src/tests/test_cleanser_lead_tokens.py` (passed)
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_cleanser_acronym.py` (passed)
- Benchmark:
  - Spot-check command/config: `PERF_BENCHMARK=1`, `PERF_BENCHMARK_MODE=regression`, `PERF_SYSTEM=gb`, `PERF_REAL_INPUT_LAYER=canonical`, `PERF_REAL_MAX_FILES=3`, `PERF_RUNS=1`, `PERF_UPDATE_BASELINE=0`
  - Result: 14,567 rows/s, median elapsed 20.595s over 300,000 rows
- Outcome: Refactor is behaviourally stable on focused suites and remains in the expected performance band on a quick real-files check.
- Notes: Use the deterministic 20% benchmark slice for any baseline/promotion decisions; this spot check is directional only.

## 2026-06-20 - Cleanser Module Split
- Date: 2026-06-20
- Area: `src/acquisition/cleanser.py` and new sibling modules
- Change: Broke the ~1,400-line cleanser monolith into five focused modules:
  - `cleanser_normalize.py`: transliteration, diacritic stripping, token normalisation
  - `cleanser_rules.py`: company-type rule loading, mapping construction and regex building
  - `cleanser_extract.py`: lead-token extraction, acronym derivation, preservation helpers
  - `cleanser_pipeline.py`: Polars pipeline steps and parquet write helper
  - `cleanser_ops.py`: analysis, corpus sampling, WordPiece training/tokenization
  `cleanser.py` is now a thin compatibility layer re-exporting all public symbols.
  Also fixed `private_analysis.py` to import `_load_company_type_rules` from
  `cleanser_rules` (restoring duplicate-rule detection in proposal generation) and
  removed the duplicate `[tool.pytest.ini_options]` block from `pyproject.toml`.
- Why: Maintainability. The monolith made navigation and isolated testing difficult; also exposed two latent bugs (recursive `name_tokenize` wrapper and broken private-analysis rule import) that are now fixed.
- Validation:
  - `src/tests/test_cleanser_smoke.py` (passed)
  - `src/tests/test_cleanser_pipeline_steps.py` (passed)
  - `src/tests/test_cleanser_lead_tokens.py` (passed)
  - `src/tests/test_cleanser_harness.py` (passed)
  - `src/tests/test_cleanser_acronym.py` (passed)
  - `src/tests/test_cleanser_chunk_naming.py` (passed, regression fix for recursive wrapper)
  - `src/tests/acquisition/test_private_analysis.py` (passed, regression fix for rule import)
  - Full suite: 170 passed, 0 failed
- Benchmark: Not applicable, a structural refactor with no algorithm changes. Post-refactor throughput consistent with the refreshed baseline (12,551 rows/s).
- Outcome: Each module has one responsibility and can be tested in isolation. Two latent bugs surfaced and were fixed.
- Notes: `cleanser.py` still owns `PipelineConfig`, `configure()`, and `name_cleanse()` as live code. A further slice could move these into a dedicated orchestration module.

## 2026-07-08 - GLEIF Sharding Baseline (200K rows, max_rows support)
- Date: 2026-07-08
- Area: `src/acquisition/sharding_gleif.py`, `src/acquisition/sharding_system_handlers.py`, `src/acquisition/sharding.py`
- Change: Added max_rows parameter support to GLEIF ZIP→Parquet sharding, allowing profiling on limited datasets (2–3 shards) without full 5–6 hour ZIP reprocessing. Enables phase-level bottleneck identification.
- Why: Profile GLEIF sharding to identify hotspots (Extract vs Write vs Other); baseline needed for regression detection before optimization attempts.
- Validation:
  - .venv\Scripts\python.exe scripts/process_companies.py --systems gleif --processes shard --force --additional-args "shard.max_rows=200000"
  - Output: 2 shards (200K rows) in 42.35s with phase breakdown
  - No schema or data corruption observed; sidecar shards written correctly
- Benchmark:
  - Command/config: --systems gleif --processes shard --max_rows=200000, chunk_size=100000, sidecar=True
  - Input: 200,000 LEI records (first 2 chunks from gleif-lei-cdf-2026-06-17.zip, 497 MB source)
  - Throughput: 4,723 rows/s median (chunk 1: 5,736 rows/s, chunk 2: 4,074 rows/s)
  - Phase breakdown (total 42.35s):
    - Extract (XML parsing): 17.73s (42%)
    - Write (Parquet): 3.92s (9%)
    - Other (ZIP iteration, filtering, DataFrame): 20.70s (49%)
  - Chunk-level variance: 40% slowdown from chunk 1 to chunk 2 suggests extract cost increases with cursor position or XML recursion depth
- Outcome: Baseline established for GLEIF sharding throughput. Primary hotspot identified: "Other" phase (49% of cost) covers ZIP file iteration, country filtering, and DataFrame construction. XML extraction (42%) is CPU-bound and second largest. Parquet write (9%) is not a bottleneck.
- Notes:
  - Optimization targets (by impact): (1) "Other" phase: ZIP iteration/filtering logic may benefit from batch processing or re-architecting ZIP seek patterns. (2) XML extraction: evaluate streaming XML parser or C-based parsing library. (3) Write already efficient; skip optimization here.
  - Chunk throughput degradation (40%) suggests cumulative state cost or GC effects; rerun with cProfile to confirm.
  - With current ~4.7k rows/s, full 1.8M row ZIP (~380k rows per file × ~5 files) would take ~6 hours; baseline supports decision to limit profiling runs to first 2–3 shards.

## 2026-06-29 - Wikidata Prepare max_rows Short-Circuit Validation
- Date: 2026-06-29
- Area: `src/acquisition/downloader_wikidata.py`, `src/acquisition/sharding.py`
- Change: Validated bounded prepare projection using `shard.max_rows=100`, including prepare early-stop logging and end-to-end propagation into source and canonical outputs.
- Why: Full Wikidata projection is expensive; capped runs provide a fast iteration loop for optimization and correctness checks.
- Validation:
  - `.venv\Scripts\python.exe scripts/process_companies.py --systems wikidata --processes shard canonical --date 2026-06-27 --allow-research --additional-args shard.max_rows=100`
  - `.venv\Scripts\python.exe -c "import polars as pl; prep=sum(1 for l in open('raw/wikidata/prepare/2026-06-27/wikidata-companies-2026-06-27.jsonl',encoding='utf-8') if l.strip()); src=pl.read_parquet('raw/wikidata/source/2026-06-27/wikidata-001.parquet').height; can=pl.read_parquet('raw/wikidata/canonical/2026-06-27/wikidata-001.parquet').height; print('prepare_rows',prep); print('source_rows',src); print('canonical_rows',can)"`
- Benchmark: Prepare projection early-stopped at written=100 after scanned=20,813 in 40.2s (scan_rate=517/s, write_rate=2/s). Row counts verified: prepare=100, source=100, canonical=100.
- Outcome: Confirmed deterministic capped iteration path; `max_rows` cap is enforced and correctly propagated across prepare -> source -> canonical.
- Notes: Remove existing prepare artifact before capped runs to avoid reusing an uncapped projection.

## 2026-06-29 - Wikidata Production Parse Path (orjson + bytes stream)
- Date: 2026-06-29
- Area: `src/acquisition/downloader_wikidata.py`, `pyproject.toml`
- Change: Switched Wikidata entity iterator to binary line streaming and enabled bytes-first JSON parsing through optional `orjson`; added `orjson` dependency in project metadata.
- Why: Profiled bottlenecks showed JSON decoding as a primary production hotspot after decompression, so parser acceleration is production-relevant and low-risk.
- Validation:
  - `.venv\Scripts\python.exe -m pytest -q src/tests/acquisition/test_downloader_wikidata.py src/tests/acquisition/test_sharding.py::test_shard_system_source_honors_max_rows_in_prepare_projection src/tests/acquisition/test_sharding_wikidata.py`
  - `.venv\Scripts\python.exe -m pip install orjson>=3.11.3`
  - `.venv\Scripts\python.exe -c "import cProfile,pstats,io; from pathlib import Path; from acquisition.downloader_wikidata import extract_wikidata_company_projection; src=Path('raw/wikidata/acquire/2026-06-27/wikidata-all-2026-06-27.json.bz2'); dst=Path('artifacts/perf/wikidata_profile_max100_after_orjson.jsonl'); pr=cProfile.Profile(); pr.enable(); extract_wikidata_company_projection(src,dst,max_rows=100); pr.disable(); s=io.StringIO(); pstats.Stats(pr,stream=s).sort_stats('cumtime').print_stats(12); print(s.getvalue())"`
- Benchmark: capped profile (max_rows=100, scanned≈20.8k) moved from 40.877s baseline to 35.533s with `orjson` active (≈13.1% faster). JSON decode cost dropped from 12.245s (`json.raw_decode`) to 6.063s (`orjson.loads`).
- Outcome: Confirmed production-path improvement without changing extraction semantics.
- Notes: BZ2 decompression remains dominant; further gains likely require compression/IO strategy changes rather than extraction logic changes.

## 2026-09-01 - Buffered closure-file JSON read in both Wikidata extractors

- Date: 2026-09-01
- Area: `src/rust/wikisieve/src/lib.rs` (`load_closure_qids`), `src/rust/wikidata/main.rs` (`load_company_type_qids`)
- Change: Wrapped the `File` handle in a `BufReader` before handing it to `serde_json::from_reader`. Both extractors read the p279 company-type closure file this way at startup.
- Why: `serde_json::from_reader` on a bare `File` pulls bytes through its `IoRead` adapter one at a time, so parsing p279.json (2.62 MB, 49,176 entries) cost ~10s. This is a fixed charge on every run regardless of input size, so it barely registered on a full-dump run but dominated fixture-scale benchmarking. Both extractors carried it identically, so prior wikisieve-vs-main.rs comparisons were fair; the absolute numbers on short runs were not.
- Validation:
  - `pre-commit run --files src/rust/wikidata/main.rs src/rust/wikisieve/src/lib.rs` (rust fmt/clippy/check for wikidata, rust test for wikisieve, all passed)
  - Isolated by substituting a 1-entry closure file, which dropped startup to under 0.15s before the fix, confirming the cost was this call and not process spawn, spec compilation, or opening the input.
- Benchmark:
  - Command/config: `wikisieve.exe --input data/wikidata/acquire/2026-07-16/wikidata-all.json.gz --spec <resolved company.json> --output-mode jsonl --max-rows N`
  - Startup, `--max-rows 1`: 10.23s before, 0.22s after (-98%)
  - `--max-rows 100000`: 21.10s before, 10.92s after (-10.18s, matching the removed startup exactly)
  - 65 MB fixture (`wikidata-benchmark-sample.jsonl.gz`, 10,000 lines) end to end: 1.89s after, against ~10s of its previous ~12s spent in this one call
  - Input-size independence before the fix: 9.90s startup on the 65 MB fixture against 9.78s on the 144 GB dump
- Outcome: Fixture-scale extractor benchmarking is roughly 6x faster and now measures extraction rather than closure-file parsing. Full-dump impact is ~10s of a 4,755s run (0.2%).
- Notes: The deployed `tools/bin/wikidata_company_extractor.exe` was not rebuilt: `main.rs` is frozen and the catalog routes Wikidata through `wikisieve`, so the binary is unused; the source fix keeps the two in step for any future comparison. `wikisieve` itself is slated to move to a standalone repo, where this change should travel with it.
