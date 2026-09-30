# acquisition

The catalog-driven pipeline that turns each source system's published data into the layers everything else reads: acquire, shard, canonical, match, cleanse and tokenize. Orchestration is generic and source-specific behaviour is isolated per system, so a source file can be traced through to its canonical and cleansed rows.

```
uv run python scripts/process_companies.py --systems gleif --processes match --additional-args match.target-system=ie
```

## Stages

Each stage's module docstring describes what it reads, writes and guarantees.

1. Acquire ([pipeline.py](pipeline.py), `scripts/acquire_companies.py`): download a system's source into a dated run folder, with filesystem-backed freshness checks.
2. Shard ([sharding.py](sharding.py) and a `sharding_<system>.py` per source): split the source into parquet chunks with harmonised schemas.
3. Canonical ([canonical.py](canonical.py)): map to the OpenCorporates-style schema, with a name-variant sidecar per system.
4. Match ([match_ops.py](match_ops.py)): label one source's canonical rows against a target's.
5. Cleanse ([cleanser_orchestrate.py](cleanser_orchestrate.py)): `company_cleanse` over canonical rows, into a partitioned `cleansed/` layer.
6. Tokenize ([tokenizer_ops.py](tokenizer_ops.py)): tokens over `cleansed/`, into `tokenized/`.

`scripts/process_companies.py` runs any of them; `scripts/pipeline_status.py` reports which are stale and prints the command that refreshes each; `scripts/check_name_layer_identity.py` checks the name-variant sidecars carry per-row identity. [docs/scripts/acquisition.md](../../docs/scripts/acquisition.md) holds the invocations.

## Systems

- The catalog, one JSON file per system under [catalog/systems](catalog/systems), is loaded and validated by [catalog/\_\_init\_\_.py](catalog/__init__.py) against [catalog/systems.schema.json](catalog/systems.schema.json).
- A `research_required` system runs only when research is explicitly enabled, and only for the stages it allows ([constants_status.py](constants_status.py)).
- Company-type values map to a canonical form per country ([company_type_mappings.py](company_type_mappings.py), `company_type_mappings/`).
- DBpedia's parser: [sharding_dbpedia.README.md](sharding_dbpedia.README.md).

### Wikidata

The dump is projected to a company extract by [downloader_wikidata.py](downloader_wikidata.py), through `wikisieve` by default, which must be built and deployed to `tools/bin` first (`scripts/build_wikisieve.ps1`, `scripts/deploy_wikisieve.ps1`). [sharding_wikidata.py](sharding_wikidata.py) shards the extract, and [wikidata_jurisdiction_closure.py](wikidata_jurisdiction_closure.py) resolves jurisdictions the static table misses. `src/tests/acquisition/test_wikisieve_conformance.py` holds the deployed binary to this repository's spec and sample, and `scripts/compare_wikidata_rust_wikisieve.py` is the same diff by hand, at any scale. Extraction speed and how it was reached are in [docs/architecture/wikidata-extraction.md](../../docs/architecture/wikidata-extraction.md).

## Adding a system

1. Add a system JSON under [catalog/systems](catalog/systems) and validate it against the schema.
2. Add a download adapter only if the catalog's download templates are not enough.
3. Add a source-specific sharding module only if the generic readers are not enough.
4. Add canonical field candidates and mapping tests.
5. Add pipeline tests for its gating and stage behaviour.

Prefer a bulk endpoint to an API where both exist and the data quality is sufficient.

## Diagrams

- [Acquisition flow and freshness](../../docs/plantuml/acquisition-flow-sequence.puml)
- [Sharding dispatch](../../docs/plantuml/sharding-dispatch-sequence.puml)
- [Wikidata runtime](../../docs/plantuml/wikidata-runtime-sequence.puml)
- [Cleanse flow](../../docs/plantuml/cleanse-sequence.puml), [company-type resolution](../../docs/plantuml/cleanse-company-type-decision-flow.puml), [short-name derivation](../../docs/plantuml/cleanse-short-name-derivation-activity.puml)
- [Canonical to cleanse to tokenize handoff](../../docs/plantuml/pipeline-handoff-canonical-cleanse-tokenize-sequence.puml)
- [Match stage](../../docs/plantuml/match-stage-sequence.puml)
