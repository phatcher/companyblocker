# wikisieve

A fast, spec-driven line filter and claim projector for the Wikidata JSON dump. It streams a gzip-compressed dump, cheaply rejects lines that can't possibly match, and deep-parses only the claim properties a spec references, instead of the whole entity.

## Performance

On the 2022-01-03 dump used by the Wikidata subsetting evaluation (Hosseini Beghaeiraveri et al., Semantic Web 15(6)), `wikisieve` selects the same entities per class as the evaluation's counts and as wikibase-dump-filter, and runs 16 to 18 times faster than wikibase-dump-filter on the same machine. The decode is the floor: on this dump `rapidgzip-core` inflates on one thread, and `wikisieve` takes 1.16 to 1.27 times as long as that decode alone. A full scan of the 2026-07-16 dump, 120,905,360 lines and 144 GiB gzipped, took 2,238 s on a shared machine and peaked at 1.4 GB of memory. [BENCHMARKS.md](BENCHMARKS.md) has the runs, the machines and the commands.

## Usage

```
wikisieve --input <dump.json.gz> --spec <spec.json> --output <output.jsonl> \
  [--summary-json <summary.json>] [--max-rows N] [--max-records N] \
  [--output-mode ids|jsonl] [--raw-candidate-output <path>]
```

- `--input`: gzip-compressed Wikidata JSON dump (one JSON array element per line), or `-` to read the same entities uncompressed from standard input. `profile` takes a file path only.
- `--spec`: a spec file (see below) saying which claims make a line a match and which claims to project.
- `--output`: destination JSONL path, or `-` for stdout.
- `--output-mode`: `jsonl` (default, the projected record) or `ids` (the matched entity's `id` only).
- `--max-rows`: stop after this many scanned lines. This bounds cost: scan time is roughly linear in lines scanned.
- `--max-records`: stop after this many emitted records. This bounds yield, not cost, since a sparse spec can scan most of the input to reach N. For a cheap preview pass both.
- `--summary-json`: also write run statistics (lines scanned, candidates, emitted, skipped) and the build's provenance as JSON.
- `--raw-candidate-output`: also write the dump line behind each emitted record, gzipped. The cache is itself a valid `--input`, so adding a projected field later is a run over a few gigabytes rather than the whole dump.

`wikisieve --version` prints `wikisieve <crate version> (commit <git commit>[, dirty])`. The same fields are written into every `--summary-json` file. The dirty flag is read when `build.rs` reruns, so a binary meant to be traced is built from a committed tree.

### Resumable chunked output

```
wikisieve --input <dump.json.gz> --spec <spec.json> --output <destination.jsonl> --resume \
  --chunk-dir <chunk-dir> --state-path <state.json> [--chunk-prefix <prefix>]
```

With `--resume`, `--chunk-dir` and `--state-path` are required. A run writes numbered chunk files (prefix `wikisieve-part-` by default) into `--chunk-dir` and records progress in the state file; `--output` is still required but not written, since `merge-chunks` assembles the flat file. An interrupted scan resumes from the last completed chunk. A run bounded by `--max-rows` or `--max-records` keeps its chunks so a later run with a higher limit can extend it; only an unbounded run is complete. On a successful merge the state file is renamed to `.completed.json`.

Every chunk, raw-candidate companion, merge output and state write is fsynced before the rename that makes it visible, so the state never counts a chunk a restart would not find. A stop loses at most the batch in flight.

A flat run truncates `--output` and `--raw-candidate-output`, and keeps no backup. To replace a live file, point the run at staging paths and rename them over the live ones after a clean exit.

### Assembling chunks: `merge-chunks`

```
wikisieve merge-chunks --chunk-dir <chunk-dir> --output <destination.jsonl> \
  (--state-path <state.json> | --summary-json <summary.json> | --expected-rows N) \
  [--chunk-prefix <prefix>] [--raw-candidate-output <path>]
```

Concatenates a completed run's chunks in index order and removes them. An expected row count is required: the merge is written to a temporary file and renamed over the destination only once its count matches, so a failed merge never touches a good destination. Pass `--raw-candidate-output` whenever the run captured raw lines, or the merge discards them with the chunk directory. Pass `--chunk-prefix` whenever the run used one; a mismatch matches no chunks and the count check refuses.

### Schema discovery: `profile`

```
wikisieve profile --input <dump.json.gz> --spec <spec.json> --output <report.json> \
  [--max-rows N] [--gap-min-count N]
```

A slower diagnostic pass. For every matched candidate it classifies every claim property present, not only the projected ones, into one of the four `projected_fields` shapes, or `complex` for anything else (quantity, globe-coordinate, `novalue`/`somevalue` snaks, or claims disagreeing on shape). The report gives per-property frequency and shape counts, and a gap report of simple-shaped properties seen on at least `--gap-min-count` candidates (default 1) that no `projected_fields` entry names. It does not resume: it is meant for a bounded sample.

## Spec format

```json
{
  "markers": [
    { "property": "P31", "match": { "type": "qid_set", "qids": ["Q783794"] } },
    { "property": "P1454", "match": { "type": "qid_closure_file", "path": "p279.json" } }
  ],
  "match_logic": "any",
  "projected_fields": [
    { "property": "P1278", "field": "lei" }
  ]
}
```

A line matches if any marker's claim property, at any rank, resolves to a QID in that marker's match set (`match_logic: "any"` is the only supported value). A match set is an inline `qid_set` or a `qid_closure_file` of `[{"subclass": "Q..."}, ...]` entries, resolved relative to the spec file unless absolute.

Each `projected_fields` entry names a claim property and the output field to project it into, in one of four shapes:

- `text_list` (default): every claim's text value (or `.text` of a monolingual value), as an array.
- `entity_id_list`: every claim's entity id (`"Q..."`), as an array.
- `time`: the first claim's `time` value, or `null`.
- `multi_lang_text`: three fields, the full text list (`field`), the English subset (`field_en`) and the full list tagged with each value's language (`field_variants`). The entry names all three.

Any other `"shape"` is a spec-load error. Two modifiers apply on top of a shape:

- `"take": "first"` (`text_list`/`entity_id_list` only): the first value, or `null`, instead of the array. A property named by several entries is still parsed once.
- `"match_only": true` (`entity_id_list` only, and only with a marker on the same property): keeps only the ids in that marker's match set.

```json
{ "property": "P1278", "field": "company_number", "take": "first" },
{ "property": "P1454", "field": "matched_company_type_qids", "shape": "entity_id_list", "match_only": true }
```

## Output record shape

Every matched line produces a JSON object with a fixed envelope, `id`, `entity_type`, `modified`, `label_en`, `description_en`, `aliases_en`, `instance_of`, `sitelinks_count`, plus the fields its `projected_fields` entries produce.

## Development

`scripts/build_wikisieve.ps1` wraps cargo, and its header says how to set the MSVC linker environment when cargo picks up an incomplete Visual Studio install.

Line coverage, with [`cargo-llvm-cov`](https://github.com/taiki-e/cargo-llvm-cov) (`rustup component add llvm-tools-preview`, `cargo install cargo-llvm-cov --locked`):

```powershell
cargo llvm-cov --summary-only
```

`tests/extract_company.rs` runs the built binary over five real entities in `tests/fixtures/` with a copy of the production company spec, and holds the output of the hardcoded extractor this crate replaced. `tests/known_answer_company.rs` runs the real production spec over a 10,000-entity sample from outside the crate against a reviewed expected output; its doc comment says how that output was reviewed and how to regenerate it.
