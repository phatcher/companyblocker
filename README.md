# CompanyBlocker

A data platform for acquiring company registry data from many sources, normalising it, and evaluating blocking strategies for entity resolution against ground truth built from the data itself. A system is a source: a country register (`gb`, `fr`, `ie`) or a cross-border one (`gleif`, `wikidata`, `offeneregister`).

## Pipeline

```
Acquire -> Shard -> Canonical -> Match -> Blocking
                        |
                        +-> Cleanse -> Train/Optimize -> Tokenize
```

- **Acquire, Shard, Canonical** turn each source's published data into one canonical schema.
- **Match** labels one source's records against one target's, producing the `match_uri` ground truth blocking is scored against.
- **Blocking** runs a candidate-generation strategy over a source and target and measures what it found.
- **Cleanse, Train/Optimize, Tokenize** are the refinement loop: they materialise the current cleansing rules and tokenizers for inspection, and train the tokenizers a strategy uses. Blocking tokenizes on the fly from a trained tokenizer rather than reading the Tokenize layer.

`docs/architecture.md` has the full picture and the reasoning behind it.

## Quick start

Requires Windows PowerShell and Python 3.12 or later.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\recreate_venv.ps1
. .\.venv\Scripts\Activate.ps1

uv run python scripts\acquire_companies.py --systems gb gleif
uv run python scripts\process_companies.py --systems gb gleif --processes shard canonical cleanse
uv run python scripts\process_companies.py --systems gleif --processes match --additional-args match.target-system=gb
uv run python scripts\run_blocking.py --source gleif --target gb --dry-run
```

Data lands under `data/<system>/` and everything a run produces under `artifacts/`. Refresh is metadata-driven per system; `--date` works on a specific historical snapshot instead of the latest.

## Stages

Every script describes itself with `--help`, and `scripts/README.md` lists them by category.

- **Acquire** (`acquire_companies.py`): downloads into `data/<system>/acquire/<date>`. `--systems` takes codes, a comma list or `all`; a `research_required` system also needs `--allow-research`.
- **Shard, Canonical, Cleanse, Tokenize, Match** (`process_companies.py --processes ...`): the stages the runner accepts, not a dependency chain. A match names one source (`--systems`) and one target (`--additional-args match.target-system=<code>`); the target is usually a single-country register and the source a cross-border one.
- **Train/Optimize** (`train_tokenizer.py`): trains or searches tokenizers per system or for the pooled `global` scope; `generate_noise_words.py` then derives TF-IDF noise words from the training corpus.
- **Blocking** (`run_blocking.py`, `compare_blocking_strategies.py`): runs and compares candidate-generation strategies; `src/blocking/README.md`.
- **Validate and analyse** (`validate_clustering.py`, `analyze_*.py`, `measure_*.py`): evaluation and diagnostics under `artifacts/validation/` and `artifacts/analysis/`.

Notebooks under `notebooks/` are committed with their executed outputs, so the figures read without re-running anything. Re-execute one in place after its inputs change:

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace notebooks\analyse_residual_pairs.ipynb
```

## Systems and credentials

Supported: `gb`, `fr`, `ie` (bulk download), `dk`, `ee` (API contract), `fi` (open API), `gleif`, `offeneregister`, `wikidata` (bulk download). `nl`, `dbpedia` and `edgar` are `research_required`. The catalog under `src/acquisition/catalog/systems/` is the full list.

Two need credentials in the environment:

```powershell
$env:DK_CVR_AUTH_HEADER_VALUE = "Basic <base64-username-password>"   # Denmark CVR
$env:EE_ARIREGISTER_API_KEY = "<api-key>"                            # Estonia e-Business Register
```

## Quality Gates

Tests first, then `pre-commit`. Tests are unit-tier by default; a broad stage-level test carries `@pytest.mark.integration`, a long benchmark `performance`, and one that renders a picture `graphics`. `docs/TESTSTYLE.md` says which tier a test belongs to. One test drives the stages end to end over a small fixture corpus under `src/tests/pipeline/`, each stage reading only what the previous one wrote, so the contracts between stages are asserted rather than assumed.

```powershell
uv run pytest -p no:tach -p check_test_tier_markers -n 8                                # full suite
uv run pytest -p no:tach -m "not integration and not performance and not graphics"    # fast iteration
uv run pytest -p no:tach -n 8 --cov=src/<area> --cov-report=term --cov-fail-under=0   # an area's coverage, not gated
```

Both `-p` flags are optimisations and live on the invocation, not in `pytest.ini`: `no:tach` skips the `tach` plugin's collection, and `check_test_tier_markers` lets the tier check read this run's collection instead of starting its own. `--cov-fail-under=0` is required, since `pyproject.toml` sets `fail_under` and any `--cov` run would otherwise be a gate.

`.pre-commit-config.yaml` is the only list of checks, and each hook declares its stage. File-scoped checks run on every commit, whole-tree and Rust checks at pre-push, and network-dependent audits manually:

```powershell
pre-commit run --files <paths you changed>
pre-commit run --all-files --hook-stage pre-push
pre-commit run --all-files --hook-stage manual
```

A dependency fix is a version bump for a direct dependency, and an exact pin under `[tool.uv] override-dependencies` for a transitive one. Adopt a fix once it has been out 30 days; a major-version bump also needs the full suite.

## Layout

- `src/acquisition`: ingestion, sharding, canonical mapping, match and the stage orchestration.
- `src/blocking`: the blocking workflow and strategy comparison.
- `src/validation`: directional source-target validation and the datasets it scores against.
- `src/analysis`: measurements and pictures over the pipeline's outputs.
- `src/training`: tokenizer training and optimize orchestration.
- `src/workspace`: the on-disk shape of `data/`, `artifacts/` and `config/`, references, run identity and production records.
- `src/rust/wikisieve`: the spec-driven Rust extractor over the Wikidata dump.
- `packages/company_cleanse`, `company_tokenize`, `company_vectorize`, `company_classify`, `company_resolvers`, `company_perturbation`: standalone libraries for cleansing, tokenization, vectorization and similarity, classification, resolution and name perturbation.
- `scripts`: the command-line entry points over the areas, listed by category in `scripts/README.md`.
- `tooling`: scripts that serve the repository itself, its gates and generators.
- `config`: checked-in tokenizer promotions, perturbation profiles and reference data.
- `notebooks`: analysis notebooks, committed with their outputs.
- `database`: a local Postgres with pgvector, for the SQL `company_vectorize.pgvector` generates.
## Further reading

- `docs/architecture.md`: architecture and design.
- `docs/canonical_schema.md`, `docs/source_schemas.md`: the canonical schema and each system's source columns.
- `docs/cli_conventions.md`: command-line conventions.
- `docs/diagrams/README.md`: the C4 and UML diagrams.
- `docs/findings/`: measured results.
- `docs/optimizations.md`: the optimisation log.
