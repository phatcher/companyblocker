# DBpedia Sharding Design Note

This module intentionally implements a focused, streaming extraction path instead of a full RDF graph workflow.

## Why it currently parses RDF as text

- The sharding step only needs a narrow subset of triples:
  - organization type detection
  - English labels
  - selected literal predicates (jurisdiction and company number candidates)
- A line-oriented streaming parser allows single-pass processing over large `ttl.bz2` dumps.
- This keeps memory usage predictable and minimizes additional runtime dependencies.

## Tradeoffs

- This is not a full Turtle/RDF semantic parser.
- It is less robust than a full semantic RDF parser for edge-case Turtle constructs.
- Future ontology expansion increases the risk of regex/parser drift.
- We intentionally avoid full in-process RDF graph parsing for the current workflow because the monthly DBpedia source artifact is large (about 350 MB compressed `.bz2`) and  significantly larger when decompressed, which makes full graph materialization expensive for local pipeline runs.

## Current policy in this module

- Keep extraction deterministic and explicit by predicate allow-lists.
- Do not fabricate identifiers.
- Leave values null when not derivable from source triples.
- Prefer additive schema changes over breaking schema rewrites.

## Provenance column strategy

We control the pipeline, so additive columns are acceptable when they improve
auditability and do not break deterministic output.

Recommended additive columns for DBpedia sharding output:

- `inclusion_basis`: primary inclusion reason (`ontology_type` or `link_inferred`)
- `inclusion_flags`: compact evidence tags (for example `ontology_type,redirect`)
- `canonical_dbpedia_uri`: resolved canonical subject used for inclusion
- `inference_method`: null for direct ontology inclusion; otherwise method name
- `evidence_confidence`: `high` for direct ontology type, `medium` for link inference

Contract guidance:

- Existing columns remain unchanged and retain current semantics.
- New columns are additive and nullable where not applicable.
- Deterministic precedence must be documented (direct ontology evidence wins over
  inferred link evidence when both exist).
- Any new inference method must set `inference_method` and `inclusion_flags`.

## When to prefer a triple store instead

If the requirement shifts from focused extraction to comprehensive RDF querying,
prefer a dedicated triple-store workflow (for example Apache Jena) over an
in-process parser migration.

Indicators that a triple store is the better fit:

- repeated multi-hop graph joins across large DBpedia snapshots
- broad predicate exploration beyond allow-list extraction
- need for reproducible SPARQL query plans over full datasets
- operational need to separate ingestion/indexing from pipeline execution
