# Per-System Acquisition Journeys

Part of [CompanyBlocker Architecture](../architecture.md): detail behind [§7](../architecture.md#7-per-system-acquisition-journeys).

Each system's acquisition and canonicalization code (see [design principles](design-principles.md)'s two stages of source-specific code) carries its own record of problems found against real data and fixed. Depth here follows how much trouble a system's data caused, not source importance: the single-country registries are simple; OffeneRegister and Wikidata took most of the engineering effort.

## GB (Companies House, United Kingdom)

- **Acquisition**: bulk monthly ZIP/CSV download (`BasicCompanyDataAsOneFile-<date>.zip`) from Companies House, refreshed on a 90-day threshold.
- **Identity**: a single national registry, so `company_number` is already nationally unique with no scoping or derivation. GB's company-type exclusion list exists but is empty.
- **Cross-system wrinkle**: GLEIF reports some GB registration numbers without leading zeros. GB's canonicalization zero-pads to its fixed 8-character format (`GLEIF_COMPANY_NUMBER_PAD_WIDTH`), and a registration-authority whitelist nulls out non-Companies-House registry numbers before matching. This lives on the GLEIF side of the join, documented in `docs/source_data_adjustments.md`.
- **Scale**: ~5.7M rows. **Status**: **Completed**, stable.

## FR (SIRENE, France)

- **Acquisition**: bulk monthly Parquet (`stockunitelegale.parquet`) from INSEE/data.gouv.fr. `StockUniteLegaleHistorique`, the separate temporal name-change file INSEE also publishes, is not part of the pipeline.
- **Identity**: single national registry, `siren` nationally unique, no scoping needed.
- **Name variants**: the FR loader extracts a name-variant sidecar (`fr-names-*.parquet`) from alternate-name columns already in the main file (`sigleUniteLegale`, `denominationUsuelle1/2/3UniteLegale`, `nomUsageUniteLegale`, `pseudonymeUniteLegale`), the shape GLEIF, Wikidata and GB already produce. Population after filtering INSEE's `"[ND]"` redaction sentinel: 2.65% (`sigleUniteLegale`), 3.09% (`denominationUsuelle1`), 9.09% (`nomUsageUniteLegale`, the most populated of the five), 0.19% (`pseudonymeUniteLegale`).
- **Sole traders are excluded.** SIRENE's raw feed includes `catégorie juridique` code `1000` (*entrepreneur individuel*, sole trader), a natural person rather than a legal entity. FR is the only one of GB, FR and IE with an active company-type exclusion (`fr-exclude.json` = `["1000"]`). **GDPR**: a sole trader's registry record is a named individual's personal data, and this platform has no lawful basis or purpose to process it. **Relevance**: the entity-resolution problem here is matching legal entities, and no other system would produce a comparable match against a sole trader.
- **Scale**: ~12.9M rows, the largest single-country corpus here, and the system whose tokenizer-optimization noise floor and memory profile [the tokenization design journey](tokenization-journey.md) records as **not yet re-measured** against IE's numbers. **Status**: **Completed**, stable.

## IE (CRO, Ireland)

- **Acquisition**: bulk **daily** ZIP/CSV (`companies.csv.zip`) from the CRO's open-data portal, the only one of the three single-country registries refreshed daily rather than monthly, and keyed by `run_date` rather than `month_start`.
- **Identity**: single national registry, nationally unique, no scoping needed, empty exclusion list.
- IE is the corpus behind [the tokenization design journey](tokenization-journey.md)'s search-methodology investigation.
- **Scale**: ~818K rows. **Status**: **Completed**.

## GLEIF (Global LEI Foundation)

A multi-hive global reference dataset of 1.8M+ LEI records rather than a country registry, and typically the source side in cross-system matching (see [§2](../architecture.md#2-system-overview--goals)) because it is not scoped to one jurisdiction.

- **Acquisition**: bulk ZIP (`gleif-lei-cdf-<date>.zip`) containing per-record XML, requiring XSD-schema-driven extraction, an exception to metadata-only onboarding (see [design principles](design-principles.md)).
- **Performance**: a 200K-row baseline (2 shards) ran at 4,723 rows/s, split Extract 42%, Write 9%, Other (ZIP iteration, filtering, DataFrame construction) 49%, implying ~6 hours for the full 1.8M-row ZIP. Caching the XSD-derived schema-index frozenset (`@lru_cache(maxsize=1)`, since it was rebuilt identically 200K+ times per run) cut that to 30.30s for the same 200K rows (6,601 rows/s, **+39.8% throughput**), with phase variance dropping from 40% to ~10% and implying ~4.55 hours for the full ZIP. Also pre-specifying the Polars DataFrame schema to skip type inference was benchmarked and **rejected**: it regressed throughput 21%, because GLEIF's records are sparse enough that dynamic inference beats a forced comprehensive schema. It is kept as a documented rejected optimization (`docs/optimizations.md`, 2026-07-08).
- **Company-type matching**: replacing a large suffix regex with a reversed-token trie (`regex|trie` switchable, `trie` default) gave a **15.9x speedup on the matcher primitive** (27,123 → 430,288 rows/s in isolation) and **+5.15% end to end** (14,183 → 14,913 rows/s) once real parquet I/O and the rest of the pipeline are included. A further attempt to remove a UDF stage via `replace_strict` was benchmarked, found to regress, and reverted.
- **Data-quality problems found and fixed against real GLEIF data**: `previous_names` is traced through GLEIF's successor-entity chain (34,001 records carry a `SuccessorLEI`; 23,777 chain-derived previous-name rows folded onto 18,391 surviving entities from the real 2026-06-21 snapshot). GLEIF's own curated ASCII transliteration is promoted into canonical `name` for non-Latin-script entities where one exists (15,787 entities affected), leaving the ~95.6% of non-Latin entities with no curated transliteration as an accepted gap with no fallback substituted. A name-variant sidecar carries multi-valued official and short-name claims (20,048 rows from a real 1M-row extraction, zero null language codes).
- **Status**: **Completed**, actively maintained.

## OffeneRegister (Germany)

The system whose data produced the most distinct defects, documented in `docs/source_data_adjustments.md`.

- **Acquisition**: a single line-delimited JSON export (`de_companies_ocdata.jsonl.bz2`, ~5.3M rows) from a community aggregator rather than an official government API: a static 2018 extract from OpenCorporates data that never updates, hence `refresh_if_older_than_days: null`.
- **`company_number` is not usable verbatim**: the raw ID is an OpenCorporates-style composite embedding a court prefix (for example `K1101R_HRB150148`). The court-to-registration-authority crosswalk is built from GLEIF's own official authorities list, cross-referenced against OffeneRegister's address-city field. An earlier attempt to build it from an LLM-summarized web fetch was caught fabricating rows and discarded. Coverage is partial: 89 of 427 distinct court codes mapped, covering 77.1% of rows (4,088,328 of 5,305,727).
- **An extraction bug, found and fixed**: shard-stage register-number extraction dropped a trailing court-merger disambiguation suffix (`HRB1162RZ` → `1162`, losing `RZ`), collapsing five distinct companies at one court onto the same key. Fixed by keeping the raw segment intact in the composite key.
- **An upstream data defect, confirmed rather than assumed**: the historical 2018 scrape batch (Mannheim/Baden-Baden) assigned the same placeholder register number across unrelated companies, including across register types, with HRA, HRB and VR sharing one literal number at the same court, affecting ~4% of rows (215,039 of 5,305,727). Confirmed as upstream rather than local corruption by inspecting the raw JSONL and re-acquiring the source to check it was byte-identical. These rows are excluded from `company_number` derivation rather than force-scoped.
- **A tokenizer bug found via Zipf analysis**: the word-token regex was ASCII-only, fragmenting accented German words (`beschränkter` → `BESCHR` and `NKTER` as two spurious tokens). Fixed to a Unicode-aware pattern. On real data, raw-tier occurrence-weighted OOV dropped from 12.7% to 8.6% and names-with-OOV from 50.3% to 36.7%, which reversed which corpus tier looked worst. The bug was in the measurement tool, not the data.
- **Known residual**: legitimate reuse of German company numbers decades apart after full deregistration produces 141 groups and 282 duplicate-key rows even under correct court scoping. Confirmed benign, not filtered, recorded as a limitation.
- **Linguistic outlier**: OffeneRegister's Zipf slope (-1.15/-1.16) is steeper than its own German reference corpus (-1.06), the only mapped-language system where that holds. It is attributed to `GMBH` dominating 49-60% of documents and to German spelling out legal forms as full words rather than short suffix codes.
- **Status**: the defects above are fixed; the Zipf/OOV divergence investigation that surfaced the tokenizer bug has landed in `src/analysis`.

## Wikidata

The one source too large for a Python pipeline on a single machine: a 154.8 GB gzip dump of 120.9M entities, of which 884,070 are companies. Extraction went through Python over bz2, Python over gzip, a Rust extractor and the spec-driven `wikisieve`, which scans the full dump in about 79 minutes and matches the frozen Rust extractor byte for byte. [Wikidata Extraction Design Journey](wikidata-extraction.md) has the engines, their measurements, the equivalence checks and the schema discovery results.

Acquisition, shard, canonical and cleanse run at real scale, and regenerated canonical and cleansed output has been verified by row-by-row diff. No tokenizer or noise-word profile has been fit on Wikidata's own vocabulary. Wikidata is multi-jurisdiction internally, so its profiles need stratifying per jurisdiction, with a separate pooled profile for its unknown-jurisdiction population.

## DBpedia

Tried and stopped.

- **What it was**: a bulk Databus-sourced extraction (instance-types, mapping-based literals, English generic labels; ~350MB compressed monthly TTL dump) with a narrow, line-oriented streaming parser rather than full RDF/Turtle graph materialization, since loading the full graph in-process would be expensive at that file size. The cost is fragility against unusual Turtle constructs as the ontology evolves.
- **Why it stopped**: the catalog metadata (`src/acquisition/catalog/systems/dbpedia.json`) records acquisition as discontinued for poor upstream data quality, replaced by `wikidata` as the equivalent global cross-jurisdiction source. Its specific defects are not catalogued the way OffeneRegister's are.
- **Status**: **Stopped**, excluded from normal acquisition runs. The catalog's `status` field was corrected to `stopped` from an earlier `research_required` value that read as still-pending.
