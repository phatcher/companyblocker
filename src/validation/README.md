# validation

Directional source-to-target validation: runs a blocking representation over labelled systems, scores what it found against ground truth, and materializes the perturbed and public-benchmark datasets it is scored on.

```
uv run --no-sync python scripts/validate_clustering.py --sources gleif --targets gb --dry-run
```

## Design contract

- Validation stays source-target generalized and never hardcodes a specific source system.
- Coverage runs source to target, with country scoping applied before edge generation.
- Lexical TF-IDF with connected components is the deterministic baseline for parity and diagnostics.
- Scoring and clustering primitives live in `company_vectorize`; run orchestration lives here.
- Every artifact is held to its schema in `contracts.ARTIFACT_SCHEMAS`.

## Modules

Each module's docstring describes it, and each script's `--help` its arguments.

- Running and scoring
    - [runner.py](runner.py): The validation matrix, and pair-truth, run and robustness scoring.
    - [recall_curve.py](recall_curve.py): Recall against comparisons spent, and its area.
    - [target_index_cache.py](target_index_cache.py): Target indexes and neighbour edges cached in the artifact store, shared with `blocking`.
    - [name_forms.py](name_forms.py): Every cleanse-derived form of a name.
    - [gleif_name_variant_collapse.py](gleif_name_variant_collapse.py): GLEIF's name-variant self-join, scored.
- Inputs and outputs
    - [config.py](config.py): A run's configuration, and each representation's text column.
    - [input_contract.py](input_contract.py): The input row contract, baseline and perturbed.
    - [loader.py](loader.py), [prepared_dataset.py](prepared_dataset.py): Loading a snapshot and its prepared per-country copy.
    - [contracts.py](contracts.py), [reporting.py](reporting.py): Artifact schemas, and writing them.
- Datasets to score against
    - [perturbation_profiles.py](perturbation_profiles.py): Where perturbation profiles are kept, and loading one.
    - [perturbation_materializer.py](perturbation_materializer.py), [perturbation_dataset_contracts.py](perturbation_dataset_contracts.py): A profile's perturbations written as a dataset (`scripts/materialize_perturbations.py`).
    - [public_benchmark_materializer.py](public_benchmark_materializer.py): A public two-table benchmark written as two systems (`scripts/materialize_public_benchmark.py`).
- [_cli_helper.py](_cli_helper.py): The settings this area's scripts read.

The DeepBlocker reference figures on Amazon-Google, which conformance work here is held to, are in `docs/architecture/deepblocker.md`.
