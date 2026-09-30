# blocking

The two-dataset blocking workflow: load a labelled source and a target's canonical snapshot, derive both sides' name forms, generate candidate pairs, cluster them, and score the result against ground truth. The workflow's shape follows pyJedAI: load two datasets, build blocks, prune or score candidate pairs, cluster the similarity graph, and emit evaluation summaries, without the algorithm layer owning orchestration.

- This area owns orchestration: running, storing, reading back and comparing runs.
- `packages/company_vectorize` owns the candidate-generation algorithms and similarity backends.
- `src/validation` owns truth scoring and metrics; a run calls its `compute_pair_truth_eval()` rather than re-deriving them.

## Running

```
uv run --no-sync python scripts/run_blocking.py --source gleif --target ie --representation tfidf --similarity-backend kmeans --top-k 20 --min-similarity 0.75
```

A run scores one source against one target under one method, and is filed at `artifacts/blocking/data/<target>/<kind>/<source>/<representation>/<key>/`, the key being a digest of what the run consumed and was configured with. A run whose location already holds a finished record is reused rather than scored again. `--help` names every setting with its default, and `--dry-run` lists those that apply to the chosen method.

Every exact-name match is taken before any scan, so recall is 1.0 at every name-equality level but `never`, where the names differ even after cleansing. `never` is the figure a method is judged on, and `run_blocking.py` prints it on its own line.

## Modules

Each module's docstring describes it.

- [workflow.py](workflow.py): One run, country by country, and its identity.
- [contracts.py](contracts.py): The run's configuration, result and artifact schemas.
- [loader.py](loader.py): Which layer each side reads.
- [name_transform.py](name_transform.py): The name forms a run derives, and which one it compares.
- [truth.py](truth.py): Which target row a source row is equal to.
- [run_layout.py](run_layout.py): Where a run sits and what it holds.
- [reporting.py](reporting.py): Writing a run, and making it a production.
- [stored_runs.py](stored_runs.py): Finished runs read back from disk.
- [comparison.py](comparison.py): Runs compared side by side and pair by pair.
- [audit.py](audit.py): A run's misses checked against the exact scan.
- [inspection.py](inspection.py): Plot-ready frames for notebooks.

## Scripts

Under `scripts/` at the repository root, each documented by its `--help`:

- `run_blocking.py`: One run.
- `compare_blocking_strategies.py`: The fixed comparison protocol over several methods.
- `report_strategy_comparison.py`: A comparison and per-pair tables from the runs on disk, running nothing.
- `aggregate_strategy_comparisons.py`: Every pairing's comparison combined, with runtime against target size.
- `report_recall_curve.py`: A finished run's recall against comparisons spent.
- `remeasure_blocking_truth.py`: A finished run's truth re-resolved, its candidates untouched.
- `measure_short_name_recall.py`, `measure_name_variant_recall.py`: What scoring the short name, or recorded name variants, recovers.

`run_blocking.py` caps Polars at four threads, since a run's memory follows its thread count; `POLARS_MAX_THREADS` overrides it.

## Further reading

- `docs/findings/blocking-methods-gleif-ie.md`: What the finished runs on `gleif -> ie` say about which pairs each method finds and at what cost.
- `docs/findings/polars-memory.md`: Why a run's memory is set by its threads rather than its data.
- [blocking-run-activity.puml](../../docs/plantuml/blocking-run-activity.puml): One run, load to written run.
- [truth-resolution-activity.puml](../../docs/plantuml/truth-resolution-activity.puml): Truth resolution through the `source_uri` chain.
- [strategy-comparison-sequence.puml](../../docs/plantuml/strategy-comparison-sequence.puml): Finished runs read back into a strategy comparison, with audits.
