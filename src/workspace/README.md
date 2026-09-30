# workspace

The on-disk shape of this repository's stored data and authored configuration: `data/`, `artifacts/` and `config/`, which every other area reads or writes. It gives every area one answer to "where is this, and what shape is it in", keeps that answer out of the areas that produce the data, and keeps it out of `packages/` entirely so they stay consumable outside this monorepo. It is the code counterpart of the Workspace Storage container in `docs/structurizr/model.dsl`.

## Contracts

Each is stated in full in the module named.

- **The roots** (`roots.py`, `repository.py`, `kind_layout.py`): where a run reads and writes, five roots resolved once from flags, environment or defaults, anchored on the checkout a file sits in, and passed to every layout function; a directory is resolved by kind, so no caller learns which tree it lives in.
- **The layer contract** (`layer_layout.py`, `data_layout.py`, `data_file_naming.py`): a layer's directories, families, partitions and file names; downstream layers mirror their input's shape rather than deciding it; a write is staged and swapped into place whole.
- **The reference contract** (`reference.py`, `derived_uri.py`, `run_inputs.py`): a script names what it reads and writes as a typed reference, never a path, and every layout is registered here before any area can build one. **Flags in, reference out** is `scripts/cli_common.add_reference_args` and `require_reference_from_args`.
- **Records** (`records.py`): everything produced carries a record of what it came from, written last, so a location is complete exactly when it holds one.
- **The artifacts contract** (`artifact_layout.py`, `reference_inputs.py`): the roots under `artifacts/`, which is generated and untracked, and the tracked reference inputs under `config/reference/`.
- **The archive contract** (`artifact_archive.py`): a content-addressed archive keyed by facets and a settings signature, and the append-only keyed store shared by worktrees and sessions.
- **The input-identity contract** (`identity.py`): the four keys a run is identified by, hashed from what it consumed.

## Implemented capabilities

Beside the contracts: resolving a row's cross-system match from its URI (`match_resolution.py`), per-phase run telemetry (`telemetry.py`), a population's name forms stored once (`name_forms_store.py`), the promoted-tokenizer pointer and store (`pointer.py`, `tokenizer_store.py`), published reports in the checkout (`published_reports.py`), the canonical snapshot Cleanse reads (`cleanse_inputs.py`), finding the directories under a root that hold parquet output (`parquet_discovery.py`), and a DuckDB-backed layer resolver (`duckdb_catalog.py`).

`scripts/check_layer_path_literals.py` enforces the reference contract and partition paths at the pre-commit stage, against the baseline `src/tests/baselines/workspace_paths.txt`; `scripts/check_stale_layout_generations.py` reports a layer holding two layout generations of one partition; `scripts/prune_artifacts.py` is the only thing that deletes from the keyed store.

## Boundaries

- Nothing here knows how the data was produced: catalogs, plans, per-system rules and stage orchestration are `src/acquisition`'s.
- Nothing under `packages/` depends on this area. A package takes a root, a pattern or a resolved file list, and the caller resolves it.
- How the stored data is enumerated belongs here too. A caller receives only paths, so `duckdb_catalog.py` changes what enumerates the files without any caller seeing it.
