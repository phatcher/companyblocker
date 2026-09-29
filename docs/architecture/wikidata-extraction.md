# Design Journey: Wikidata Extraction

Part of [CompanyBlocker Architecture](../architecture.md): detail behind the Wikidata row of [§7](../architecture.md#7-per-system-acquisition-journeys) and [Per-System Acquisition Journeys](acquisition-journeys.md#wikidata).

Wikidata is the one source whose raw input is too large for a Python pipeline on a single machine, and the case behind [design principles](design-principles.md)'s point that a purpose-built native tool beats reaching for a cluster. The company records are found by scanning the whole dump, and four engines have done that scan.

| Engine | Input | Design | Measured throughput | Full dump | Role today |
| --- | --- | --- | --- | --- | --- |
| Python over bz2 | `.json.bz2` | Standard-library bz2 stream, `orjson` parsing, raw-line prefilter, one process | About 800 lines/s decompressing, about 1,000 lines/s with the prefilter (100,000-line bounded runs) | About 34 h estimated at 1,001 lines/s | Retired |
| Python over gzip | `.json.gz` | Standard-library gzip stream, `orjson` parsing, single scan, `P1454` closure match | 4,398 lines/s decompressing and iterating (10,000-line fixture), against 3,211 for the best bz2 reader | At least 7.6 h estimated at 4,398 lines/s, before parsing | Selectable engine |
| Rust extractor (`main.rs`) | `.json.gz` | Streaming gzip, byte prefilter, shallow zero-copy parsing, parallel batches, resumable chunks | 22,078 lines/s | 91.3 min | Frozen reference for equivalence checks; selectable |
| `wikisieve`, shallow-parse core | `.json.gz` | The same core driven by a declarative spec, with schema discovery, chunk merging and a raw candidate cache | 25,884 lines/s | 77.9 min | Superseded by the row below |
| `wikisieve` | `.json.gz` | The above, plus gzip decompression on its own thread, single-threaded on this dump, and a prefilter that searches each line for a marker's values as well as its property names | 54,014 lines/s | 37.3 min | Production default |

The two middle Rust figures come from one back-to-back run of both engines on the same 2026-07-16 dump. The current `wikisieve` figure is a later full run of the same dump and spec, 2,238.4 s wall for 4,923.1 s of CPU, whose output is byte-identical to the run above it. Neither Python engine has run the full dump: their times are the measured line rates applied to its 120,905,360 lines. The bz2 estimate covers the whole pipeline on a bounded run; the gzip estimate covers reading alone, so it is a floor.

## The problem

- The 2026-07-16 dump is 154,805,551,356 bytes of gzip (144.2 GiB), 120,905,360 lines, one entity per line.
- 884,070 lines are companies, 0.73% of the dump. The share varies with position in the dump, so an early slice overstates it: 1.54% in the first 500,000 lines, 0.37% in the first 8 million.
- The gzip file is a single DEFLATE stream, so it offers no free parallelism: a reader cannot be handed a range without inflating everything before it. Splitting it is still possible by speculatively indexing DEFLATE block boundaries inside the one member and inflating those concurrently, which is what the current reader does; every published tool measured against this one inflates on a single thread.
- Parsing, not decompression alone, is the main cost once a fast decompressor is in place: on the same 1M-line slice, the Rust extractor took 97.2 to 126.4 s with JSON parsing and 44.7 to 69.8 s with parsing skipped.

## How a company is recognised

Every engine applies the same rule, and the equivalence checks below hold them to it. An entity is selected when either holds:

- **Direct instance of a root:** its `P31` (instance of) is exactly one of six classes: company (`Q783794`), enterprise (`Q6881511`), organization (`Q43229`), corporation (`Q167037`), public company (`Q891723`) and multinational corporation (`Q161726`). Subclasses do not count on this path.
- **Legal form in the closure:** its `P1454` (legal form) is one of the 49,176 classes in `p279.json`, an array of `{"subclass": ...}` entries (2.6 MB) kept beside the dump.

The closure is broader than company legal forms. It holds the six roots, company types such as bank, airline, software company and holding company, legal forms such as GmbH and SA, and organisations that are not companies, such as universities, football clubs, government agencies and political parties, which is the shape of the subclass tree of organization. Nothing in this repository builds the file or records which root it was built from.

Two consequences follow. An instance of a company subclass, a bank whose `P31` is bank, is selected only if it records a legal form, since the closure is applied to `P1454` and not to `P31`; how many companies that leaves out is unmeasured. And because the frozen extractor applies the same rule, the equivalence checks confirm the engines agree with each other, not that the population is complete.

Reading the closure file was a fixed cost of about 10 s per run in both Rust engines, because an unbuffered reader parsed it one byte at a time. Buffering the read took startup from 10.23 s to 0.22 s, which matters for bounded and fixture runs and is negligible on a full dump.

## 1. Python over bz2

The first engine read the `.json.bz2` dump through the standard-library `bz2` module in one process.

- **Parsing:** binary line streaming with `orjson` halved JSON decode time (12.2 s to 6.1 s) and made a capped profile 13% faster.
- **Prefilter:** a cheap raw-line check before JSON decode rejected most non-candidate lines, 26% faster on a 100,000-line bounded run (796 to 1,001 lines/s).
- **Decompression was the wall:** standard-library bz2 decompressed about 800 lines/s. Parallel bz2 decompression (`indexed_bzip2`) reached 6,755 lines/s on 100,000 lines but failed on the full file, and `rapidgzip` reads gzip only.
- **Two-pass split:** a byte-level `P31` filter feeding a full parse of the survivors was designed here. It is the origin of the byte prefilter every later engine uses.
- **Matching gap:** this engine matched only direct instances of the six root classes, with no `P1454` closure match.

No full-dump run was completed on bz2. At the measured 1,001 lines/s the full dump would take about 34 hours, and about 42 hours at the standard-library decompression rate alone.

## 2. Python over gzip

Acquisition moved to the `.json.gz` dump because a paired benchmark on a 10,000-line fixture measured the gzip line reader at 4,398 lines/s against 3,211 lines/s for the best bz2 reader, with near-zero startup against 0.12 s. That was a single run, so it gives a direction rather than a precise ratio.

- **Design:** standard-library gzip, `orjson` parsing, and a single scan that filters and projects in one read. The two-pass wiring and the alternative matchers were retired.
- **`P1454` closure match:** the equivalence harness against the Rust extractor showed this engine missing every company qualifying through `P1454` alone. It now loads `p279.json` beside the dump and falls back to the six roots, with a warning, when the file is absent. It matches the Rust output byte for byte at 10,000 and 100,000 rows.
- **Role:** still a selectable engine, and the engine used when the catalog names none.

## 3. The Rust extractor

`src/rust/wikidata/main.rs` built `wikidata_company_extractor.exe`, selected as the `rust_cli` engine. Both were removed once `wikisieve` was held to this repository's own conformance test; its full-dump extract of 2026-09-01 is kept as a stored reference.

- **Decompression:** streaming `flate2` gzip behind a 1 MiB buffer.
- **Prefilter:** a byte scan for `"P31"` or `"P1454"` before any parsing. On the full dump 98.1% of lines pass it, since both properties are common; the saving is on the rest.
- **Shallow parsing:** each entity is read as borrowed `serde_json::RawValue` spans, and only the claims the match and projection use are parsed. This made parsing 3.2 to 3.6x faster, verified byte-identical at 500,000, 1M and 2M rows.
- **Parallel batches:** 100,000 scanned lines are read and prefiltered in sequence, then projected across cores with `rayon`, preserving order. This measured 2.11x faster than sequential projection, close to the single-stream ceiling above.
- **Resumable output:** records are written to a chunk directory whose chunks finalise on the 100,000-line checkpoint, with a state file and rotated backups, so an interrupted run resumes instead of rescanning. A forced kill mid-run left the in-progress chunk empty before this was aligned, and the aligned version was verified under a forced kill.
- **Full dump:** 5,476 s (91.3 min), 22,078 lines/s, 884,070 records.

The extractor is now frozen: it gains no features and serves as the fixed reference the spec-driven engine is checked against.

## 4. wikisieve

`src/rust/wikisieve/` keeps the Rust extractor's core (streaming gzip, shallow `RawValue` parsing, 100,000-line `rayon` batches) and replaces its hardcoded field list with a declarative spec compiled once at startup. See the [wikisieve README](../../src/rust/wikisieve/README.md) for the spec and CLI.

- **Spec:** `markers` decide which entities are companies (a `P31` QID set and a `P1454` closure file, combined with `any`), and `projected_fields` decide the output, in four shapes (`text_list`, `entity_id_list`, `time`, `multi_lang_text`) with `take` and `match_only` modifiers. This repository's company spec is `src/acquisition/catalog/projections/wikidata-company.json`.
- **Speed:** the spec-driven port first ran at about 72% of the extractor's speed. A claim cache that parses a property used by both a marker and a field once, and precompiled `memchr` substring searches built from the spec's marker properties, turned that into a lead: about 9% faster on an interleaved 2M-line window, and 17% more throughput on the full dump (25,884 against 22,078 lines/s, 77.9 against 91.3 min).
- **Resumable chunks and merging:** the same chunk and resume model as the extractor, plus a `merge-chunks` subcommand that joins chunks in order into a temporary file and renames it only when the row count matches a required expected count. On a real bounded run it matched the Python merge byte for byte (19,372 rows), and the required count caught a chunk-prefix mismatch that would otherwise have produced an empty file.
- **Raw candidate cache:** the pipeline writes the raw dump line behind every matched record, gzipped, beside the projection. A new field can then be reprojected from the cache (about 2.2 GB at full scale) instead of rescanning the 154.8 GB dump. Capture costs 4 to 5% of wall time.

### Equivalence at full-dump scale

Both Rust engines ran the full 2026-07-16 dump back to back on the same machine and inputs, and the comparison found them byte-for-byte equivalent across every field the extractor produces:

| Measure | Rust extractor | wikisieve |
| --- | --- | --- |
| Records emitted | 884,070 | 884,070 |
| Missing, extra or differing records against the extractor | | 0, 0, 0 |
| Elapsed | 5,476 s (91.3 min) | 4,671 s (77.9 min) |
| Throughput | 22,078 lines/s | 25,884 lines/s |

Smaller comparisons run on every change: this repository's conformance test (`src/tests/acquisition/test_wikisieve_conformance.py`) runs the deployed binary with the production spec over the tracked 10,000-line benchmark sample against a checked-in, reviewed expected output, and `scripts/compare_wikidata_rust_wikisieve.py` is the same diff by hand against any stored reference; the crate keeps its own integration test on five real entities; and the 10,000 and 100,000-row Python against Rust harness remains.

### Schema discovery

`wikisieve profile` is a separate diagnostic pass over the same scan machinery. It classifies every claim property on the matched companies by shape, or as `complex` (quantities, coordinates, no-value claims), and reports as gaps the simple-shaped properties seen often enough that the spec does not yet project.

A run over the first 8M lines of the dump (6.6%, 376 s, gap threshold 40 companies) matched 29,244 companies carrying 2,359 distinct properties, and reported 417 gaps. The dump is ordered roughly by QID, so this is not a uniform sample. The identifier properties among the 50 most common gaps:

| Property | Identifier | Companies carrying it |
| --- | --- | --- |
| `P2671` | Google Knowledge Graph ID | 39.2% |
| `P646` | Freebase ID | 25.8% |
| `P214` | VIAF cluster ID | 24.2% |
| `P1320` | OpenCorporates organization ID | 22.8% |
| `P4156` | Czech Registration ID (IČO) | 13.1% |
| `P2088` | Crunchbase organization ID | 10.7% |
| `P213` | ISNI | 10.3% |
| `P2427` | GRID ID | 9.6% |
| `P3225` | Corporate Number (Japan) | 6.1% |
| `P4264` | LinkedIn company or organization ID | 5.4% |
| `P1297` | IRS Employer Identification Number | 3.8% |
| `P3608` | EU VAT number | 3.4% |

The D-U-N-S number (`P2771`) is not among the 50. The full report lists every property and is regenerated by the same command over a larger window.

## Capability against other dump tools

A published evaluation of Wikidata subsetting tools (see [External Alignment](../external_alignment.md#wikidata-dump-extraction) for the speed comparison and sources) found flexibility and speed in tension: the most expressive tool was the slowest, because its filters run against every entity. `wikisieve` sits with the fast, item-based tools, with two differences that matter for companies: subclass-aware matching through a closure file, and built-in analytics.

| Capability | WDF | WDumper | KGTK | WDSub | `wikisieve` |
| --- | --- | --- | --- | --- | --- |
| Output | NDJSON | N-Triples | TSV or RDF | JSON or RDF | JSONL projection or ids, plus the matched raw entity lines |
| Subset definition | Command-line filters | JSON spec | Kypher | ShEx | JSON spec (`markers`, `projected_fields`) |
| Runs on a PC | Yes | Yes | Yes | Yes | Yes |
| Needs the full dump locally | Yes | No, online demo | Yes | Yes | Yes |
| Live subsetting | No | No | No | No | No |
| Massive data | Yes | Yes | Yes | Yes | Yes |
| Qualifiers | Yes | Yes | Yes | Yes | No: main snak values only |
| References | Yes | Yes | In progress | Yes | No |
| Graph traversal | No | No | Yes | No | Partial: membership in a precomputed transitive closure |
| Further output transforms | No | No | Yes | No | No |
| Analytics | No | No | Yes | Yes | Yes: `profile` shape and fill rate per property, and run summary counts |

The evaluation's three flexibility cases, against `wikisieve`:

- **Instances of given classes:** supported, as by every tool. A closure file lets a marker match any class in a precomputed subclass tree, which WDF and WDumper cannot do in a single pass; the company spec applies its closure to legal form, not to `P31` (see [How a company is recognised](#how-a-company-is-recognised)).
- **Selection by label or description language or value:** not supported. Labels and aliases are not addressable in a spec, and only the English label, description and aliases are emitted.
- **Selection on references:** not supported.

Joined conditions across entities, such as compounds related to extracted diseases, are outside a single pass for `wikisieve`, as for WDF and WDumper.

### Why the cost follows the spec

Only the properties a spec names are parsed, a property used by both a marker and a field is parsed once, and the byte-level prefilter looks only for marker properties. Adding a projected field or a closure costs little, which is why closure-aware matching still runs at several times the evaluation's fastest rate. What would cost speed is a selection condition the prefilter cannot use: every entity carries labels, so selecting on language or label value in the scan would force every one of the 120.9M lines through JSON parsing.

This is why `wikisieve` is a projection engine and why the raw candidate cache is kept. The scan selects only on properties the prefilter can see and writes each matched entity's original line beside the projection. The cache holds 884,070 of the dump's 120,905,360 lines (0.73%) and is itself a valid `--input`, so a new field, or a condition on language, label value, qualifiers or references, runs as a second pass over the cache and parses a small fraction of the dump instead of all of it.

## Where it goes next

- **A controlled comparison** against the published subsetting evaluation's fastest tool, both on one volume with its own decompressor and a single-threaded leg beside the parallel one, which is what turns a rates comparison into a measured ratio.
- **A designed library API**, then a repository of its own for `wikisieve`, consumed here as a pinned build while the frozen extractor and the comparison script stay here as the conformance check.
- **Entity-level fields** in the spec, so labels and aliases in every language can be projected, not only property claims.
- **Building the closure file** from a SPARQL query instead of a hand-supplied file.
- **An all-simple-fields mode** that projects every simple-shaped property the profile classifies, for exploration.
- **More identifier fields** in this repository's company spec, chosen from the profile's fill rates above.
