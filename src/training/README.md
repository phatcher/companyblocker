# training

Orchestration for tokenizer training: the train and optimize workloads, the optimize sweep's phases, its concurrency and retry policy, promotion of a trained candidate, and the reports over promoted candidates. Only tokenizer training is orchestrated here.

## Modules

Each module's docstring describes it.

- [workloads.py](workloads.py): The train and optimize workloads over a run's targets.
- [optimize_execution.py](optimize_execution.py): One target end to end, and the optimize sweep's phases.
- [optimize_retry.py](optimize_retry.py): Which candidate failures are retried.
- [optimize_concurrency.py](optimize_concurrency.py): Memory-aware sizing of optimize waves.
- [optimize_expansion_strategy.py](optimize_expansion_strategy.py): Which vocabulary sizes get more seeds.
- [runtime.py](runtime.py): State shared across one optimize run.
- [promotion.py](promotion.py): Storing a candidate and pointing a profile at it.
- [tokenizer_corpus_report.py](tokenizer_corpus_report.py): One report per scope's promoted candidate.
- [scope_comparison.py](scope_comparison.py): Every scope's promoted candidate side by side.

## Scripts

Under `scripts/` at the repository root, each documented by its `--help`:

- `train_tokenizer.py`: The train and optimize modes.
- `archive_optimize_candidate.py`: Copies one optimize candidate out of its sweep into a durable, named location.
- `generate_tokenizer_corpus_report.py`: Builds and publishes a scope's report.
- `compare_tokenizer_scopes.py`: Writes the cross-scope comparison.
- `plot_country_global_overlay.py`: Plots each country's fertility and unknown-token rate against `global`.

## Further reading

- [optimize_search_methodology.md](optimize_search_methodology.md): What the optimize grid covers and how much confidence that coverage gives.
- [tokenizer-optimize-activity.puml](../../docs/plantuml/tokenizer-optimize-activity.puml): The optimize and promotion flow.
