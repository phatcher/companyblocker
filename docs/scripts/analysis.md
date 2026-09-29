# Analysis scripts

The scripts that report on what the pipeline produced: how well two systems matched, and how a clustering configuration behaves over a pairing.

- `analyze_matches.py`: Reports match coverage and quality for a source, target and country scenario.
- `validate_clustering.py`: Runs a clustering configuration over one or more pairings and reports the result.

# analyze_matches.py

```
uv run python scripts/analyze_matches.py --source gleif --target gb --country gb
```

`--source` and `--target` name the pairing and `--country` filters to a hive partition, defaulting to the target system's own code. `--target-display` sets the label the report uses for the target where the system code reads poorly.

The script reads the source system's `matched/` layer, so it depends on the match stage having run for that pairing. A scenario that has never been run is reported as such by `pipeline_status.py` rather than silently absent.

`--scenarios-file` runs a set of scenarios defined in a file instead of the one named on the command line, and `--date` groups scenarios sharing a run. `--max-rows` caps rows per input dataset for a fast pass over a large system.

# validate_clustering.py

```
uv run python scripts/validate_clustering.py --sources gleif --targets ie --countries ie
```

Takes lists rather than single values: `--sources`, `--targets` and `--countries` each accept several, and the script runs the resulting combinations.

## Configuration arguments

`--representation` and `--similarity-backend` select the vector space and the neighbour search, with `sklearn`, `sparse_dot_topn` and `svd_rerank` available. `--clustering` and `--similarity` name the clustering method and the similarity measure. `--text-view` chooses between the cleansed name and its tokenized form.

The candidate window uses the same three arguments as a blocking run: `--top-k`, `--min-similarity` and `--max-candidates-per-source`. `--source-name-col` and `--name-col` select the text columns on each side.

## Progress and batching

`--source-chunk-size` sets how many source records are processed at a time, and `--progress-record-interval` and `--progress-time-interval-seconds` control how often progress is reported. Progress is emitted on a fixed interval independent of how the caller batches work, so a large input reports while it is still in flight rather than only on completion.

# Reading a result

A comparison between two configurations is only meaningful when both were computed over the same candidate window, since precision and recall both move with `--top-k`. A run's own output records the computed scores rather than the arguments that produced them, so the configuration has to be carried alongside the result rather than recovered from it.
