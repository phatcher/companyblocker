# Architecture & Design Principles

Part of [CompanyBlocker Architecture](../architecture.md): detail behind [§3](../architecture.md#3-architecture--design-principles).

The README states four design strategies. Each maps to something concrete in the codebase.

## Metadata-first, catalog-driven onboarding

Onboarding a new system is not metadata-only end to end. Acquiring a source needs a download and extraction handler for its format, which is code. The field mapping from its native columns into the canonical schema is not: it is `system_field_candidates` in the system's own catalog entry (`gb`'s Companies House fields, `fr`'s SIRENE fields, `ie`'s CRO fields), typed in `models_plan.py` and resolved by `canonical.py`; see `docs/canonical_schema.md`.

That code is confined to two stages, Acquire and Canonical. Every stage after them (Cleanse, Tokenize, Match, Validate, Blocking) is system-neutral: it operates on canonical columns and catalog metadata (freshness thresholds, sharding handler selection, schema mapping, modelled in `src/acquisition/catalog/` JSON and typed plan models in `models_plan.py`, read by `registry.py`), never on source-specific branching. The exceptions are acquisition-side (GLEIF's XSD-driven schema traversal, Wikidata's two-pass projection), never downstream of Canonical. Maintaining that boundary is ongoing work.

## Canonical schema scope

The canonical schema (`docs/canonical_schema.md`) is what makes the system-neutral claim above possible. Its core is identity (`jurisdiction_code`, `company_number`, `system_uri`) plus name variants (`name`, `alternative_names`, `previous_names`), with the minimal lifecycle and status fields needed to judge whether a match candidate is valid (`current_status`, `inactive`, `incorporation_date`, `dissolution_date`). It carries no financials, ownership structure, SIC or industry codes, directors, or filings.

Two reasons:

1. **Name-only is the worst-case blocking problem, and the one this project studies.** A record carrying richer structured attributes (registered address normalized to parts, industry code, ownership graph) lets blocking lean on those instead of on two records with no shared identifier judged only by name. Building the schema around that constraint fixes what the blocking-strategy comparisons ([DeepBlocker inspiration](deepblocker.md), [package deep-dive](packages.md)) are scored on. The name-only case occurs downstream of a named entity recogniser, in ecommerce systems given no VAT or registration identifier, and when cross-referencing instrument or issuer data from two equity data providers.
2. **Every canonical attribute is acquisition and cleansing surface for every system.** A field is only as good as its weakest source's mapping for it, so adding one requires every current and future system to supply a credible mapping or an explicit null before the field is trustworthy. A narrow schema keeps the per-system onboarding cost low enough for the "plug in another public dataset in a few steps" goal ([§1](../architecture.md#1-executive-summary)) to hold as sources are added.

## Sharding and the Canonical relayout

The Shard stage writes many small files per system: work parallelises across them, progress is visible as they land, and a failure loses one file rather than a run. Those are producer-side properties. For readers, a few hundred fragments of a few megabytes each defeat parquet's row-group and partition-pruning economics.

Canonical changes the layout, because it is already rewriting every row to normalise a system-specific schema onto the neutral one: it consolidates fragments into `jurisdiction_code=`-partitioned files sized for scanning, and splits name variants out into name rows at the same time. Both happen once, at the boundary where the data stops being source-shaped and starts being pipeline-shaped.

No downstream stage therefore knows how a source chunked itself or which layout a system uses. Cleanse, tokenize, analysis and blocking read one shape, partitioned the way they query it. Where Canonical leaves the producer-side layout in place, the cost is paid by every consumer on every read.

## Entity and name corpora are kept independent

An entity row carries one name, its own; every variant a source supplies lives as a name row. The two populations are disjoint by construction: the primary name is never duplicated into name rows, and variants are never reachable from an entity row.

That independence is what lets the two serve different purposes. Tokenizer training and the Zipf and rarity analysis read the entity corpus, one row per company, so a train/validation split over it cannot place a company on one side and its own alias on the other. Blocking's multi-key target index (`expand_target_frame_with_name_variants`) reads name rows, because it wants every string an entity is known by.

Merging them fails silently: a held-out unknown-token rate computed over a corpus containing both "IBM" and "International Business Machines" for the same company reads better than it should, and that number selects the operational tokenizer.

A statistic computed over one corpus is therefore not comparable with one computed over the other, and an analysis wanting both reports them separately rather than pooled.

## Stage-oriented execution, contract-driven artifacts

Covered in [§2](../architecture.md#2-system-overview--goals). Each stage reads one declared input layer and writes one declared output layer (`constants_pipeline.py`'s `PIPELINE_STAGE_INPUT_LAYERS`), so a stage cannot reach across the pipeline for data it was not handed.

## Deterministic baseline first

Explainable, rule-based flows land before higher-complexity model variants, and a higher-tier variant clears a benchmark and quality-parity bar before promotion. This is the shape of the tokenizer-trainer story (WordPiece baseline, then SentencePiece as a compared alternative) and of the blocking-strategy story: TF-IDF, WordPiece and SentencePiece clustering are tested, wired-in candidate-generation paths scored against ground truth, the pretrained S-BERT embedding strategy is wired in on the same terms, and DeepBlocker's remaining families are built to be scored beside them (see [DeepBlocker Inspiration](deepblocker.md)). Individual higher-tier variants are at various stages, tagged where they appear below.

## Package and area boundaries, mechanically enforced

The repository splits into two kinds of unit:

- **`packages/*`**: standalone, independently-versionable libraries with their own `pyproject.toml` and no dependency on `src/`: `company_cleanse`, `company_tokenize`, `company_vectorize`, `company_classify`, and `company_perturbation`.
- **`src/*`**: pipeline orchestration areas that consume those packages: `acquisition`, `analysis`, `training`, `validation`, `blocking`, plus `workspace`, which is not an orchestration area but the shared leaf owning every data-layer and artifact path rule.

The dependency direction is declared in `tach.toml`, and `tach check` runs as a pre-commit hook and a quality gate. The package rows below are stated from what each package actually imports, which its `pyproject.toml` requirements match. An import crossing between `packages/` and `src/` is not rejected mechanically, so those edges are declared rather than enforced until a boundary checker covering them is adopted.

| Module | Depends on |
| --- | --- |
| `packages/company_cleanse` | *(none, leaf package)* |
| `packages/company_tokenize` | `company_cleanse` |
| `packages/company_vectorize` | `company_cleanse` |
| `packages/company_classify` | `company_perturbation` |
| `packages/company_perturbation` | `company_cleanse` |
| `packages/company_resolvers` | *(none, leaf package)* |
| `src/workspace` | *(none, leaf)*: every `src` module may depend on it |
| `src/acquisition` | `workspace`, `company_cleanse`, `company_tokenize`, `company_vectorize` |
| `src/analysis` | `src/acquisition`, `company_tokenize` |
| `src/training` | `src/acquisition`, `src/analysis`, `company_tokenize` |
| `src/validation` | `src/acquisition`, `company_cleanse`, `company_perturbation`, `company_tokenize`, `company_vectorize` |
| `src/blocking` | `company_vectorize`, `company_tokenize`, `src/validation` |
| `scripts` | all of the above except `company_perturbation`, which it reaches through `src/validation` |

The packages sit on `company_cleanse` rather than in a chain: `company_tokenize`, `company_vectorize` and `company_perturbation` each depend on it directly, and only `company_classify` is two hops out, through `company_perturbation`. `company_tokenize` and `company_vectorize` are independent of each other, since a trained tokenizer reaches blocking as packaged or supplied token data rather than as an import. Every `src/` area builds on packages, never the reverse, and `src/blocking` depends on `src/validation` rather than duplicating its scoring logic (`compute_pair_truth_eval`).

**Areas compose through artefacts, not through each other's code.** Each area computes what its own responsibility covers and writes the result under a declared schema; an area needing another's results reads that artefact rather than importing the producer. The import graph and the reading graph therefore differ legitimately: `src/analysis` declares no dependency on `src/validation` or `src/blocking` yet reports over both, because it reads their promoted outputs. Cross-area synthesis belongs to `analysis`, since reading promoted outputs across areas is the only responsibility spanning them and needs no import edge. A package is one step stricter: it produces its own metrics and never reads another area's outputs, which makes a cross-area report filed inside a package a boundary violation.

`src/workspace` owns the path rules, so "where an artefact lives" has one implementation rather than one per consumer, and a reader cannot drift from its writer by resolving a path differently. Before it existed the storage contract lived inside `src/acquisition`, so every other area reached into the writer to learn the shape of shared storage, and `src/blocking`, which has no acquisition edge, carried a hand-copied glob whose failure mode was a silent `None`. The same grounding extends from `data/` to `artifacts/`, so no area re-derives an artifact layout of its own.

Packages are standalone deliverables: their own `pyproject.toml`, no dependency on `src/`, and a documented root import surface (`company_cleanse`'s README lists its supported entrypoints). A project can depend on `company_cleanse` or `company_tokenize` for deterministic company-name cleansing or trained tokenization without knowing about `src/acquisition`'s stage orchestration.

Where a package makes a normalization claim, that claim is grounded in an external authority. `company_cleanse`'s company-type resolution (`reference_data.py`) uses the ISO 20275 Entity Legal Forms Code List, the GLEIF-maintained international standard for legal-form codes, covering 3,600+ forms across 200+ jurisdictions.

Coverage of that list is incremental and data-driven. `company_type_rules.json`'s `{source, canonical, country}` mappings are built and verified against the real country data acquired and cleansed so far, each jurisdiction added once real data justified it, including `normalize.py`'s documented exclusion of forms not evidenced in the data on hand. The standard supplies the ground truth a mapping is checked against; it does not substitute for verifying against real records.

Where one leaf package would benefit from another's derived data without depending on it at runtime, the mechanism is a checked-in artifact snapshot rather than a code dependency. A data-driven improvement to `company_cleanse` computed by `company_tokenize`, such as a statistically-derived word list, moves as a reviewed, versioned file copied into the consuming package, the shape `company_type_rules.json` and `entity_legal_forms_iso20275.json` already take. The lineage lives in commit history, and installing the consuming package alone works with no runtime awareness of the other.

## Jurisdiction is a proxy for language, not the same thing

`jurisdiction_code`, where a company is legally registered, is an always-available signal for which language a name is likely written in, but it is not the same fact: a German-registered subsidiary can carry an English name, and a UK-registered company a German one. Any process fitting language-sensitive statistics (which tokens are generic versus distinctive, which subword vocabulary fits a corpus) treats jurisdiction as a proxy with an unmeasured error rate.

Per-system tokenizer training is therefore per-jurisdiction training standing in for per-language training. It holds for the single-country systems (`gb`, `fr`, `ie`, each overwhelmingly one language) and breaks down for GLEIF and Wikidata, both multi-jurisdiction internally, which cannot be treated as one flat corpus without blending distinct languages' vocabularies.

The repository-wide `global` scope is the fallback used when jurisdiction is unknown, the one case with no other signal. Balancing training rows equally per system prevents one system's raw size from dominating, but does not give equal representation per language, since a multi-jurisdiction system's internal language mix can still skew the pool. A `global` profile performing best on whichever language is best-represented in the systems feeding it defeats the reason it exists.

## Technology & execution model: Polars + Parquet, not Pandas

`polars` is a first-class runtime dependency (`pyproject.toml`); `pandas` appears only in the `dev` dependency group alongside notebook and plotting tools, and is never a runtime path. Every stage's storage format is Parquet.

Polars' lazy, columnar, multi-threaded execution model lets a single machine process multi-million-row, multi-gigabyte registry sources (GLEIF's 1.8M-record ZIP, Wikidata's ~145GB compressed dump) without a distributed cluster.

A faster Polars-native execution path, whether native expressions or a `duckdb` SQL escape hatch, can cost clean, unit-testable library code. That tradeoff is resolved area-locally rather than as one repo-wide pattern, because the two existing approaches solve different-shaped problems. `company_cleanse`'s step-engine split (`CleanseStepEngine`, `stepengine_base.py`) swaps the implementation flavour of one per-step transform, selected per step behind a shared ABC interface, entirely within Polars' `LazyFrame -> LazyFrame` vocabulary, and every variant is unit-testable against a small in-memory frame with no external engine or file I/O (`test_stepengine_base.py`). `analysis.token_metrics`'s `engine=` toggle swaps the entire query engine for one expensive cross-system aggregation, embedding raw SQL behind a documented, measured trigger (`token_performance_plan.md`'s Phase 2.5 storage decision gate: >20min wall-clock, >10M rows/20GB, >5 repeated cross-system joins). An ABC wrapping DuckDB SQL would not make the SQL more unit-testable, and `company_cleanse`'s step-level swaps need no non-Polars engine.

Four rules generalize, and any new performance-sensitive Polars module is judged against them:

1. Default to pure-Polars, `LazyFrame`-native implementations, unit-testable against small in-memory frames with no external engine or file I/O.
2. Introduce a non-Polars engine only behind an explicit, opt-in parameter, never a silent heuristic switch, and only once a measured trigger is hit and documented. `token_performance_plan.md`'s Phase 2.5 thresholds are the template; adapt the numbers per module.
3. An alternate-engine implementation is covered by at least one test exercising that codepath. `token_metrics`'s single DuckDB integration test is the floor.
4. An alternate-engine implementation returns the same shape and contract as the default path, so callers stay engine-agnostic.

`analysis.token_zipf` is the confirmation case: a pure-Polars module at real scale (`fr` at 12.9M rows) that finished its full multi-system sweep in about 6 minutes with no engine-variant split. Rule 2 is expected to be reached first on Wikidata-derived corpora, whose acquisition-side processing times are already large.

The Rust Wikidata extractor (see [acquisition journeys](acquisition-journeys.md)) covers the case where Polars-in-Python hits a wall, single-pass CPU-bound JSON parsing over a 145GB stream: a purpose-built native tool behind the same pipeline contract, not a cluster.

## Architecture diagrams: Structurizr / C4

The authoritative diagram source is `docs/structurizr/` (the DSL, not the rendered diagrams): `model.dsl` defines the C4 model (System Context, Containers, Components) and `views.dsl` the rendered views. It is edited as text and rendered via Structurizr Lite locally.

Some edges in `model.dsl` are commented as **aspirational**: drawn to show intended architecture the code does not call yet, such as `training → vectorize` (nothing in `src/training` calls `company_vectorize`) and `validation → recognize` (`company_classify` is standalone, not called from any pipeline stage). This document follows the same convention, tagging status throughout rather than in a final status section.

## Deterministic seeding: scalar-Python and vectorized-Polars conventions

Two seeded-randomness patterns exist in this repository, ratified as two conventions, one per execution context. They solve different problems and share no code.

**Vectorized-Polars convention**, for row-level seeding over a dataframe such as a reproducible train/validation split across millions of rows: `company_tokenize/optimize.py`'s `write_seed_split_artifacts` uses `pl.col("row_nr").hash(seed=seed) % split_resolution`, mixing `seed` and row index through Polars' built-in hash. A hand-rolled linear congruential generator using the ANSI-C `rand()` constants, with the seed folded in as an additive offset, was rejected: nearby seeds produced strongly correlated splits, with seeds 42 and 43 overlapping 76% in validation-set membership against an expected ~20% under independence, which defeats evaluating multiple seeds for a stable estimate. `hash(seed=seed)` measured at that ~20% independence across seed pairs. Any new Polars row-level seeded operation follows this pattern rather than a hand-rolled formula.

**Scalar-Python convention**, for one record or call at a time such as an operator mutating a single name: plain `random.Random(seed)`, CPython's Mersenne Twister, seeded once and drawn from directly. `company_classify/split.py`'s `random.Random(seed).shuffle(...)` follows it, as does `company_perturbation`, where one generator is seeded per record from `(run seed, local_id)` and then carried through that record's whole operator chain rather than reseeded per step. The pairing is what makes a record's outcome independent of processing order and of how a corpus was sliced: a run seed alone would make a record depend on how many preceded it, and a record identifier alone would give it exactly one possible perturbation.

The scalar convention was measured to the same bar the LCG rejection was, against both a downstream consumption pattern (`rng.sample(eligible, num_edits)`, a selection of `num_edits` positions from a per-record list of eligible ones) and the raw output stream:

1. **Selection overlap**, the direct analogue of the tokenizer methodology: the fraction of overlap between two seeds' selections against the value expected under independence. For `eligible` lists of size 10, 20 and 40 at intensities 0.3, 0.5 and 0.7, 2,000 seed pairs were sampled at four seed gaps: adjacent (`s`, `s+1`), `+5`, `+100`, and a control gap (`s`, `s+1,000,003`), plus seeds 42, 43 and 44 which this repository uses elsewhere. At `eligible_n=20`, `intensity=0.5` (`k=10`), adjacent-seed mean overlap was 0.500 (stdev 0.117) against an independence-expected 0.500 (`k/n`); the `+5`, `+100` and large-gap conditions measured 0.501, 0.502 and 0.501. The pattern held at every combination tested (10/0.5, 40/0.5, 20/0.3, 20/0.7), with adjacent-seed overlap tracking the control to within sampling noise and none of the LCG's elevation, which measured 3.8x the independence-expected value.
2. **Raw-draw correlation**: Pearson correlation between `random.Random(s).random()`'s first draw and `random.Random(s+gap).random()`'s, across 5,000 seeds at gaps of 1, 2, 5, 100 and 1,000,003. All five gaps produced correlations within ±0.02, noise level for a 5,000-sample estimate (standard error ≈ 0.014), with the adjacent-gap value (`r = -0.0103`) no larger in magnitude than the control (`r = -0.0189`). The second and third draws after seeding gave the same result.

CPython's Mersenne Twister seeding does not reproduce the nearby-seed correlation the LCG had, for either the downstream sampling usage or the raw stream. The two are different algorithms, so the vectorized rejection does not transfer to the scalar case, and the measurements above are what establish that. CPython's integer seeding routes through an array-based Mersenne Twister initialization (`_randommodule.c`'s `random_seed()`, which `Lib/random.py`'s `Random.seed()` delegates to via `super().seed(a)` for an int `a`) rather than folding the seed in as a single linear term; that is supporting context, not verified against the C source here, and the measurement is the load-bearing evidence.

A future measurement at a different list size, selection count or consumption pattern that finds a correlation this check did not is grounds to revisit the convention.

**Composing a seed from more than one identifying value** is a distinct step from the PRNG primitive. `company_perturbation.generation`'s `_record_seed()` is the caller, combining a run's seed with a record's own identifier. The rule as implemented:

1. Take the components in a fixed order: the run seed, then the record's `local_id`.
2. Hash each as its own length-prefixed segment, writing its byte length (`len(encoded).to_bytes(8, "big")`) before its encoded bytes, so segment boundaries are unambiguous whatever a segment contains. A separator-delimited join is vulnerable to delimiter collision, since `local_id` is externally sourced and this module does not control its charset: one containing the separator would let two different pairs produce the same seed. The digest is `hashlib.blake2b`, not the built-in `hash()`, which is salted per process by `PYTHONHASHSEED` unless pinned and so is not reproducible across runs or machines.
3. Convert the digest to an integer (`int.from_bytes(digest, "big")`) and seed one `random.Random` with it.

That generator is then carried through the record's whole operator chain. Each trial consumes a fixed number of draws whether or not it fires, so a step landing no changes and a step landing several advance the stream identically, and changing one step does not reshuffle every step after it.

`acquisition.name_variant_uri` uses the same length-prefixed segment hashing for the same reason, over a different field tuple.
