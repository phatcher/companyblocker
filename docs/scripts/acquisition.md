# Acquisition and pipeline scripts

The scripts that bring a system's data in and move it through the shared layers, plus the report that says which of those layers is out of date.

- `acquire_companies.py`: Downloads source datasets for one or more systems.
- `process_companies.py`: Runs one or more post-acquisition pipeline stages for one or more systems.
- `pipeline_status.py`: Reports the latest artifact timestamp per system per stage and prints the command that would refresh each stale one.

# acquire_companies.py

```
uv run python scripts/acquire_companies.py --systems gb fr ie
```

Downloads source datasets into `data/<system>/acquire/<snapshot_date>`, applying metadata-driven freshness gates that decide whether each source needs refreshing. Refresh is not tied to a fixed date: the freshness check comes from each system's own metadata. `--date` overrides the snapshot folder for working on a historic dataset.

`--systems` accepts space-separated or comma-separated values, and `all` selects every enabled system. A system marked `research_required` is one not yet fully adopted, and needs `--allow-research` before it will run. `--fail-on-unsupported` turns a system the catalog cannot handle into an error rather than a skip.

`--dry-run` prints the planned actions without downloading anything.

```
uv run python scripts/acquire_companies.py --systems gb --dry-run
```

`process_companies.py` also exposes `acquire` in `--processes`, which runs the same stage as part of a longer stage selection. Combining it there with `--allow-research` stops the run after acquisition rather than continuing into the later stages.

# process_companies.py

One script drives every post-acquisition stage. `--systems` takes one or more system codes and `--processes` takes the stages to run, defaulting to `shard canonical cleanse tokenize`.

```
uv run python scripts/process_companies.py --systems gb --processes shard canonical cleanse
```

The stages run in a fixed order: `acquire`, `shard`, `canonical`, `cleanse`, `match`, `tokenize`. Selecting a stage does not run the ones before it, so a stage whose input is stale produces stale output without complaint. `pipeline_status.py` is what reports that condition.

## Stage arguments

Stage-specific options are passed through `--additional-args` in `stage.key=value` form rather than as top-level flags, so a key belongs to exactly one stage:

```
uv run python scripts/process_companies.py --systems gleif --processes match --additional-args match.target-system=ie
```

`--force` makes selected stages run even when their outputs appear current, which is needed after a change that alters what the output should contain without changing what the freshness check compares.

## Match direction

Match is the one stage whose arguments encode a direction. A match is a statement of intent, this source labelled against that target, so both are named and the direction given is the direction run.

`--systems` names the **source**, and `match.target-system` names the **target**. The match relationship is held source-side only: `data/<source>/matched/` is keyed by the source's own `system_uri` with `match_uri` naming the target, so a target system never carries one. A target system having no `matched/` directory is therefore correct, not a gap.

A multi-hive system such as `gleif` is the source side; single-hive registries such as `gb`, `ie` and `fr` are the reference side. One run writes one thing: the source's `matched/` partition for the jurisdiction it shares with the target, so `gleif`'s `de`, `fr`, `gb` and `ie` partitions are four runs. A run with no `match.target-system`, or with more than one system in `--systems`, is refused before any stage starts: no system is matched because it happens to be on disk.

Match reads the latest canonical snapshot for both sides, so it does not depend on Cleanse having run, and a newer canonical on either side means the match layer is behind it.

## The text the tokenize stage reads

The tokenize stage is a run of the tokenizer to see what it does, and it tokenizes the text a blocking run would. `--name-col` picks the column, and `--preprocess-profile`, named as a blocking run names it, passes that column through the same preprocessing first: `default` removes the company type, `default|-company_type` keeps it, and every profile normalises. `--noise-words-profile` defaults to `none`, which removes no words; naming a profile removes that profile's words. The stage checks and refuses nothing about either choice.

## Previewing a run

`--dry-run` reports, per selected stage and system, the skip-versus-run decision, the resolved input and output directories, and what existing output would be cleared or migrated, without touching the filesystem. Since `data/` is shared storage that takes hours to regenerate, this is worth running first for anything that writes.

```
uv run python scripts/process_companies.py --systems fr --processes cleanse --dry-run
```

# pipeline_status.py

```
uv run python scripts/pipeline_status.py --root .
```

Reports the latest artifact timestamp per system for `acquire`, `shard`, `canonical`, `cleanse` and `tokenize`, flagging a stage with `!` when it is older than its upstream stage's latest output. It also covers match-analysis freshness per source, target and country scenario, and blocking run freshness. Each stale entry is followed by the exact command that would refresh it, derived from `process_companies.py`'s own stage choices so a renamed stage cannot leave a dead recipe behind.

`match` is deliberately absent from the per-system table: it is a per-scenario concern that the match-analysis section covers, including by omission for scenarios never run.

# A stage's output is a snapshot of its package, not of its input

Neither the cleanse nor the tokenize stage carries its own copy of what it applies. `cleanser_orchestrate.py` calls `company_cleanse`'s `cleanse_lazyframe` during the run, and `tokenizer_ops.py` calls `company_tokenize`'s `tokenize_name` the same way, so each stage applies whatever the package does at the moment it executes.

The consequence is that `data/<system>/cleansed/` shows the state of the cleanser when that layer was last written, and `data/<system>/tokenized/` the state of the tokenizer when that layer was last written. A change to a cleanse rule or a newly promoted tokenizer therefore has no effect on anything already on disk, and nothing about the existing layer indicates that it now describes an older set of rules: the files are present, well-formed and correctly timestamped against their own inputs. Only re-running the stage brings the layer into line with the package.

This is why a rule change and a data refresh are separate acts, and why the freshness report compares a layer against its upstream layer rather than against the code that produced it.

# Layers that are deliberately not maintained

The persisted `tokenized/` layer is training and analysis input. Blocking tokenizes on the fly from the promoted tokenizer artifact and never reads it, so it is left behind by a canonical and cleanse refresh rather than regenerated, that being the most expensive stage by file count. A number computed from it describes the pre-refresh names until the affected system is re-tokenized.
