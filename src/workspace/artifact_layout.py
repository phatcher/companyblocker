"""Sole owner of the on-disk shape of this repository's build artifacts.

`data/` is what every stage reads or writes as its subject matter;
`artifacts/` is what a stage produces *about* that data along the way -- a
trained tokenizer, an analysis report, a benchmark result -- and every area
used to derive its own corner of it by hand: `company_tokenize/paths.py`
derives `artifacts/tokenizers` for itself, and `src/analysis`,
`src/training`, `src/validation` and several scripts each spell
`project_root / "artifacts" / <area> / ...` inline. Every one of those
inline constructions agrees with its siblings by convention only, the same
drift `layer_layout` exists to close for `data/`.

**The shapes.** Real `artifacts/` holds three today, plus a fourth root
reserved here for the first trained-model archive to use it:

    artifacts/
        analysis/<report>/runs/<run_date>/[...]    one dated run per report, area-shaped beneath that
        tokenizers/work/<system-code-or-global>/   one tokenizer working tree per scope
        tokenizers/data/<scope>/<tokenizer_id>/<key>/   one promoted candidate per reference
        perf/                                      benchmark output
        validation/                                validation run output
        models/<model-family>/...                  one trained-model archive per family, shaped by `artifact_archive`
        pretrained_vectors/<checksum>/<filename>    one downloaded checkpoint file per checksum

Deliberately smaller than `layer_layout`'s family split: `layer_layout`'s
`primary`/`names` are two competing row shapes that must never share a
directory, where an analysis report or a tokenizer scope has no such family
to keep apart. This module reflects what is actually on disk rather than
importing that split speculatively. What is beneath a report's run
directory or a validation run is still that area's own decision, the same
way a layer's own file names are `data_file_naming`'s business, not
`layer_layout`'s: this module only resolves the roots that every one of
those areas already agrees on.

**The tokenizer split.** `tokenizers/work/<system-code>/` and
`tokenizers/work/global/` mirror
`company_tokenize.paths`'s existing system-versus-global split exactly
(`tokenizer_scope_dir`), because that package already got the shape right
for its own artifacts; this module gives every other caller -- `src/analysis`
reading a system's `noise_words.json`, `src/training` writing a tokenizer's
model file -- the same answer instead of re-deriving it.
"""

from __future__ import annotations

from pathlib import Path

from .roots import WorkspaceRoots

ANALYSIS_DIR_NAME = "analysis"

TOKENIZERS_DIR_NAME = "tokenizers"

TOKENIZER_WORK_DIR_NAME = "work"
"""The leaf of the tokenizer root that `company_tokenize` lays out for itself,
beside the `data` leaf holding the promoted candidates `workspace.reference`
locates, so no scope's name can collide with that leaf."""

MODELS_DIR_NAME = "models"

PRETRAINED_VECTORS_DIR_NAME = "pretrained_vectors"
"""Downloaded, never-trained-here checkpoints (the pretrained fastText baseline), kept
apart from `models/`: a checkpoint here is identified by the checksum of what was downloaded,
not by `artifact_archive`'s settings-plus-content signature, since there is no training
configuration to sign -- the download either is or is not the file the registry names."""

STORE_DIR_NAME = "store"

PERF_DIR_NAME = "perf"

VALIDATION_DIR_NAME = "validation"

BLOCKING_DIR_NAME = "blocking"

RUNS_DIR_NAME = "runs"

GLOBAL_TOKENIZER_SCOPE = "global"
"""The tokenizer scope directory name for a scope with no system code, matching
`company_tokenize.paths.resolve_global_tokenizer_paths`."""


def artifact_root(roots: WorkspaceRoots) -> Path:
    """Where this repository keeps everything a stage produces, as opposed to
    `data/`'s everything a stage consumes: the resolved artifacts root,
    `artifacts/` under the checkout unless overridden. The root for an
    artifact with no area of its own -- a one-off cross-cutting file, not a
    report or a tokenizer -- lives directly here."""
    return roots.artifacts


def analysis_artifact_root(roots: WorkspaceRoots) -> Path:
    """Where every analysis report keeps its output, one subdirectory per
    report."""
    return artifact_root(roots) / ANALYSIS_DIR_NAME


def analysis_report_runs_root(roots: WorkspaceRoots, report: str) -> Path:
    """Where one analysis report keeps every run it has ever produced, one
    dated subdirectory per run."""
    return analysis_artifact_root(roots) / report / RUNS_DIR_NAME


def analysis_report_run_dir(roots: WorkspaceRoots, report: str, run_date: str) -> Path:
    """Where one dated run of one analysis report keeps its own output.

    The report's further shape beneath this directory (`metrics/`, `viz/`,
    `reports/`, or a scenario subdirectory) is that report's own decision,
    the same way a layer's file names are `data_file_naming`'s business and
    not `layer_layout`'s.
    """
    return analysis_report_runs_root(roots, report) / run_date


def tokenizer_artifact_root(roots: WorkspaceRoots) -> Path:
    """The working tree handed to `company_tokenize`, which lays out one
    subdirectory per scope (a system code, or `global`) beneath it: sweeps,
    resume state and unpromoted candidates. A promoted candidate is not here
    but at its `tokenizer://` reference's location."""
    return artifact_root(roots) / TOKENIZERS_DIR_NAME / TOKENIZER_WORK_DIR_NAME


def tokenizer_scope_dir(roots: WorkspaceRoots, *, system: str | None = None) -> Path:
    """One tokenizer scope's own subtree: a system code's directory, or
    `tokenizers/global/` when `system` is None.

    Mirrors `company_tokenize.paths`'s system-versus-global split, so a
    caller outside that package (`src/analysis` reading a system's
    `noise_words.json`, `src/training` writing a tokenizer's
    model file) gets the same scope directory that package's own
    tokenizer-path resolvers would build for it.
    """
    scope = (
        GLOBAL_TOKENIZER_SCOPE
        if system is None
        else _normalize_tokenizer_system(system)
    )
    return tokenizer_artifact_root(roots) / scope


def trained_model_artifact_root(roots: WorkspaceRoots) -> Path:
    """Where every trained-model archive lives, one subdirectory per model family.

    A model family's own candidate and promoted paths beneath this root are resolved
    through `artifact_archive.resolve_candidate_dir`, the same shape `tokenizer_scope_dir`
    already gives the tokenizer archive's root: this function resolves only the shared
    root, not what a family keeps beneath it.
    """
    return artifact_root(roots) / MODELS_DIR_NAME


def pretrained_vector_artifact_root(roots: WorkspaceRoots) -> Path:
    """Where a downloaded pretrained-checkpoint file lives, one subdirectory per checksum.

    A caller resolves `<this root>/<sha256 checksum>/<filename>` and writes there once, never
    over an existing checksum directory -- the checksum, not the settings that produced a
    trained-here model, is this tree's whole key.
    """
    return artifact_root(roots) / PRETRAINED_VECTORS_DIR_NAME


def artifact_store_root(roots: WorkspaceRoots) -> Path:
    """Where keyed, append-only artifacts shared by sessions are stored."""
    return artifact_root(roots) / STORE_DIR_NAME


def perf_artifact_root(roots: WorkspaceRoots) -> Path:
    """Where benchmark output lives, one file per benchmark."""
    return artifact_root(roots) / PERF_DIR_NAME


def validation_artifact_root(roots: WorkspaceRoots) -> Path:
    """Where validation run output lives. A run's further shape beneath this
    directory is `src/validation`'s own decision."""
    return artifact_root(roots) / VALIDATION_DIR_NAME


def blocking_artifact_root(roots: WorkspaceRoots) -> Path:
    """Where blocking run output lives.

    Under `artifacts/` rather than `data/`: a run's outputs are derived
    artefacts keyed by the configuration and the inputs that produced them,
    not a data layer, so they belong in the same tree as the tokenizer and
    trained-model archives that one resolver already answers for. The shape
    beneath this directory is `src/blocking`'s own decision, resolved through
    `artifact_archive`.
    """
    return artifact_root(roots) / BLOCKING_DIR_NAME


def _normalize_tokenizer_system(system: str) -> str:
    """Match `company_tokenize.paths._normalize_system` exactly: stripped,
    lowercased, and required to be non-empty."""
    normalized = system.strip().lower()
    if not normalized:
        raise ValueError(
            "A system code is required for a system-scoped tokenizer directory."
        )
    return normalized
