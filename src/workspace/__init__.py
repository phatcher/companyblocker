"""The shape of this repository's workspace storage.

`docs/structurizr/model.dsl` has modelled **Workspace Storage** as a
container since before this package existed -- "repository-local data and
artifact folders: data/, artifacts/analysis/, artifacts/tokenizers/, and
manifest outputs" -- with a documented edge to it from `orchestration`,
`analysis`, `training`, `validation`, `blocking`, the CLI, the notebooks and
downstream consumers. Every container reads or writes it; one container
happens to write it first.

The code had no counterpart, so the container's contract lived inside
`src/acquisition` and every other area had to reach into the *writer* to
learn the shape of shared *storage*. `src/blocking` could not even do that
(`tach.toml` gives it no acquisition edge), so
`blocking.loader.load_name_variant_frame` carried a hand-copied glob whose
failure mode is a silent `None`. This package is that missing counterpart:
a leaf every area may depend on, so the model's edges are expressible in
code.

**What belongs here.** How to reach this repository's stored data, and what
shape it is in: the `data/<system>/<layer>` root every layer lives under
(`data_layout`), layer and family directories beneath it (`layer_layout`),
the filename families within them (`data_file_naming`), which snapshot of a
layer to read (`cleanse_inputs`), and discovery over those
(`parquet_discovery`). The same for what the stages produce rather than
consume: the `artifacts/` roots every area writes its reports, tokenizers
and benchmark output into (`artifact_layout`).

The container is *Workspace Storage*, not "the filesystem" -- its recorded
technology is Parquet on disk today, but that is the current answer to
"where is the data", not the question. Anything that answers the same
question through a different backing -- a DuckDB catalog over these files,
a warehouse connection -- belongs here too, so that a change of backing is
a change inside this package rather than an edit to every reader. That is
the practical payoff of the module existing at all: the areas above it ask
for a system's rows, not for a glob.

**What does not.** Anything that knows how the data was *produced* --
catalogs, plans, per-system rules, stage orchestration -- stays in
`src/acquisition`. And nothing under `packages/` may depend on this: those
are meant to be consumable outside this monorepo, so they take a root path,
a file pattern, or an already-resolved file list, and the caller (a
top-level script, normally) does the resolving.
"""
