"""Fixed comparison protocol every strategy-comparison sweep follows by default.

`gleif -> gb`/`ie`/`fr` are already used repeatedly by name as this package's
real-data comparison slices, but until now only as ad hoc convention --
nothing fixed them, or a runtime budget, as a protocol every comparison run
follows. This module is that fixed point: the slices, the per-cell runtime
budget, and the failure-status vocabulary `scripts/compare_blocking_strategies.py`
uses to enforce both. It lives in `company_vectorize`, not `src/blocking`,
because packages are meant to be usable independently of the repo -- `src`
may depend on packages, never the reverse.

`fr` (12,930,798 target rows) is deliberately excluded from `FIXED_SLICES`.
Its ground truth produces a precision collapse to 0.121 (vs 0.879/0.842 on
`gb`/`ie`), a likely SIRENE `[ND]`-redaction artifact, still uninvestigated --
baking a known-distorted ground truth into the standard fairness comparison
would be premature. It stays available as an explicit extended/exploratory
slice (pass `--source gleif --target fr` directly to
`compare_blocking_strategies.py`), caveat stated, once that redaction
artifact is actually investigated.

A "cell" is one `(strategy, country)` combination -- one representation run
against one country partition of one `(source_system, target_system)` slice.
`RUNTIME_BUDGET_SECONDS` bounds each cell independently via subprocess
isolation (`subprocess.run(..., timeout=...)` around a `run_blocking.py`
invocation), not an in-process thread timeout: a thread can only stop
waiting, not kill a GIL-bound or BLAS-heavy runaway, and unbounded memory
growth has separately been observed in this exact codepath, so only
process-level isolation actually bounds worst-case resource use.

`MANDATORY_SUITE` and `REFERENCE_SUITE` are the two combination sets within
that fixed protocol, replacing the single `BASELINE_SUITE` tuple this module
carried before them: every classical representation
(`tfidf`/`wordpiece`/`sentencepiece`) scored through a sub-linear backend
(`kmeans`, the partition backend that accepts a sparse matrix -- `hdbscan`
requires a dense one and so never applies here; `lsh`, the sparse-signature
MinHash/LSH-banding backend) plus the sentence encoder (`sbert`) through the
dense ANN backend (`hnsw`) is `MANDATORY_SUITE`: what a comparison sweep must
complete, for every country in a slice, before that slice's comparison
outputs can stand in for "the baseline was run". `sbert` needs the optional
`sentence-transformers` extra, which is now declared and synced (see
`pyproject.toml`'s `sbert` group), so it is no longer excluded from the
default sweep.

The sub-linear backends are the production shape a comparison exists to
measure -- what does not scale is the exact comparison every residual source
row pays against every target row, not the embedding pass itself. Making
that exact scan the workload measured a `gb` cell at three hours
unaccelerated, and the dense-vocabulary gate refuses `wordpiece`/
`sentencepiece` on it above `DENSE_VOCABULARY_MAX_ROWS` target rows, so an
all-exact suite could not complete on either fixed slice.

`REFERENCE_SUITE` is what each mandatory cell's recall loss is priced
against, one exact cell per representation and slice rather than a cell per
`(representation, backend)` pair: prefix-filtered `sklearn` (`prefix_filter.py`'s
L2AP bound, an exact prune) for the three sparse representations, and the
chunked exact dense scan (`dense_brute`) for `sbert`. It is a named option on
the comparison script, not part of the default sweep -- it is the expensive
part, and is meant to run once per slice, not on every sweep. The
dense-vocabulary gate can still refuse `wordpiece`/`sentencepiece`'s reference
cell at a slice past its row-count ceiling; that refusal is recorded as
refused rather than forced past with `--force`, and a slice can be complete
on `MANDATORY_SUITE` with its reference column partly refused -- the two sets
are reported, and gated, independently. `svd_rerank` and `sparse_dot_topn` sit
in neither set: a comparison *against* the reference, like every backend
outside `MANDATORY_SUITE` that is not itself the reference.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, NamedTuple

import polars as pl

from company_vectorize.dense_vocabulary_gate import (
    DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    DENSE_VOCABULARY_REPRESENTATIONS,
    EXHAUSTIVE_SPARSE_BACKENDS,
)

# `(source_system, target_system)` pairs the fixed protocol sweeps by
# default. Full population, no sampling. Order matters only for print/log
# readability -- nothing downstream depends on it.
FIXED_SLICES: tuple[tuple[str, str], ...] = (
    ("gleif", "gb"),
    ("gleif", "ie"),
)

# `fr`'s exclusion is a decision, not an oversight -- see the module
# docstring. Recorded here as data (rather than only in prose) so a caller
# building a message that explains the omission can read it from one place.
EXCLUDED_SLICES: tuple[tuple[str, str], ...] = (("gleif", "fr"),)

# Reference to the dense-vocabulary ceiling gating wordpiece/sentencepiece
# without --force -- the same default `scripts/compare_blocking_strategies.py`'s
# own `--max-rows` flag already uses. Re-exported under this module's name
# so the protocol's constants are readable from one import rather than two.
DENSE_VOCABULARY_MAX_ROWS = DEFAULT_DENSE_VOCABULARY_MAX_ROWS

# Per-`(strategy, country)`-cell runtime budget, in seconds. Headroom above
# the largest known-good accelerated run (`gb` at 203s); well below the known
# pathological case (one unaccelerated `gb` cell measured at 10,986s, ~3
# hours). A conservative starting budget from the one real measurement
# available, not a proven ceiling.
RUNTIME_BUDGET_SECONDS = 600

# What a cell's subprocess invocation resolved to. "completed" is the only
# status build_strategy_comparison() ever sees -- "timeout"/"error" cells are
# recorded to the sidecar failures log instead and excluded from the
# strategy_comparison artifact, whose schema this protocol leaves untouched.
CellStatus = Literal["completed", "timeout", "error"]

CELL_STATUSES: tuple[CellStatus, ...] = ("completed", "timeout", "error")


class BaselineSuiteCell(NamedTuple):
    """One mandatory `(representation, similarity_backend)` combination.

    Deliberately narrower than `BlockingStrategyConfig` -- the baseline suite
    fixes representation and backend only, the same two columns
    `blocking.comparison.build_strategy_comparison()` already keys its
    `strategy_comparison` rows on alongside `country`. `top_k`/`min_similarity`/
    `accelerator_settings` are run-configuration choices the fixed protocol
    already pins elsewhere (`scripts/compare_blocking_strategies.py`'s CLI
    defaults), not part of what makes a cell "baseline" or not.
    """

    representation: str
    similarity_backend: str


# The mandatory baseline suite (replacing the earlier single all-`sklearn`
# tuple): every classical representation on the sparse-capable partition
# backend (`kmeans` -- `hdbscan` requires a dense matrix and so never applies
# to these three) and on the sparse-signature LSH backend, plus the sentence
# encoder on the dense ANN backend. These are the production shape a
# comparison exists to measure -- see the module docstring.
MANDATORY_SUITE: tuple[BaselineSuiteCell, ...] = (
    BaselineSuiteCell("tfidf", "kmeans"),
    BaselineSuiteCell("tfidf", "lsh"),
    BaselineSuiteCell("wordpiece", "kmeans"),
    BaselineSuiteCell("wordpiece", "lsh"),
    BaselineSuiteCell("sentencepiece", "kmeans"),
    BaselineSuiteCell("sentencepiece", "lsh"),
    BaselineSuiteCell("sbert", "hnsw"),
)

# The reference column: one exact cell per representation, priced
# once per slice rather than as one of the mandatory backends -- see the
# module docstring for why an exact scan is a reference and not the
# workload. `sklearn` here runs with its exact prefix filter on by default;
# `dense_brute` is the chunked exact scan for a dense matrix.
REFERENCE_SUITE: tuple[BaselineSuiteCell, ...] = (
    BaselineSuiteCell("tfidf", "sklearn"),
    BaselineSuiteCell("wordpiece", "sklearn"),
    BaselineSuiteCell("sentencepiece", "sklearn"),
    BaselineSuiteCell("sbert", "dense_brute"),
)

# The `REFERENCE_SUITE` cells the dense-vocabulary gate can ever
# refuse outright: a dense-vocabulary representation on an exhaustive sparse
# backend, past `DENSE_VOCABULARY_MAX_ROWS` target rows. Whether a given
# slice's cell actually crosses that ceiling is real-data-dependent and not
# knowable here; this only names which cells the gate can reach at all, so a
# caller checking a real refusal against this set is checking the right
# cells rather than re-deriving the gate's own representation/backend pairing.
GATED_REFERENCE_CELLS: frozenset[BaselineSuiteCell] = frozenset(
    cell
    for cell in REFERENCE_SUITE
    if cell.representation in DENSE_VOCABULARY_REPRESENTATIONS
    and cell.similarity_backend in EXHAUSTIVE_SPARSE_BACKENDS
)


class BaselineSuiteStatus(NamedTuple):
    """`MANDATORY_SUITE`/`REFERENCE_SUITE` completeness for one `comparison` frame.

    The two sets are reported, and gated, independently (module docstring):
    `missing_mandatory_cells` is what `require_baseline_suite_complete()`
    raises on, `missing_reference_cells` never does. `refused_reference_cells`
    is disjoint from `missing_reference_cells` -- a refused cell is a named,
    reported outcome, not an absence.
    """

    missing_mandatory_cells: tuple[tuple[str, str, str], ...]
    missing_reference_cells: tuple[tuple[str, str, str], ...]
    refused_reference_cells: tuple[tuple[str, str, str], ...]


def missing_baseline_suite_cells(
    comparison: pl.DataFrame,
    *,
    countries: Sequence[str],
    refused_reference_cells: Sequence[tuple[str, str, str]] = (),
) -> BaselineSuiteStatus:
    """Return `MANDATORY_SUITE`/`REFERENCE_SUITE`'s completeness against `comparison`.

    `comparison` is a `strategy_comparison`-shaped frame (`build_strategy_comparison()`'s
    output, or one read back off disk) carrying at least `representation`,
    `similarity_backend`, `country`, and `stage` -- this function reads only
    those four columns and has no dependency on `blocking`, matching this
    module's own "packages never depend on `src`" boundary. A cell is missing
    unless it has a completed (`stage == "pruned"`) row. `refused_reference_cells`
    names `(representation, similarity_backend, country)` triples the caller
    already knows the dense-vocabulary gate refused (real-data-dependent, so
    this function cannot derive it itself); a triple named there is reported
    as refused rather than missing, and only when it is actually one of
    `REFERENCE_SUITE`'s own cells for one of `countries`. An empty result on
    every field means both suites are complete for `countries`, not that
    `comparison` itself is non-empty; an empty `comparison` frame trivially
    returns every cell as missing (refusals aside).
    """
    if comparison.height == 0:
        completed: set[tuple[str, str, str]] = set()
    else:
        completed = set(
            comparison.filter(pl.col("stage") == "pruned")
            .select("representation", "similarity_backend", "country")
            .unique()
            .iter_rows()
        )

    reference_triples = {
        (cell.representation, cell.similarity_backend, country)
        for cell in REFERENCE_SUITE
        for country in countries
    }
    refused = {
        triple for triple in refused_reference_cells if triple in reference_triples
    }

    missing_mandatory = tuple(
        (cell.representation, cell.similarity_backend, country)
        for cell in MANDATORY_SUITE
        for country in countries
        if (cell.representation, cell.similarity_backend, country) not in completed
    )
    missing_reference = tuple(
        triple
        for triple in reference_triples
        if triple not in completed and triple not in refused
    )

    return BaselineSuiteStatus(
        missing_mandatory_cells=missing_mandatory,
        missing_reference_cells=missing_reference,
        refused_reference_cells=tuple(sorted(refused)),
    )


class BaselineSuiteIncompleteError(ValueError):
    """Raised by `require_baseline_suite_complete()` when a mandatory cell is missing.

    `missing_cells` carries the same `(representation, similarity_backend,
    country)` triples `missing_baseline_suite_cells().missing_mandatory_cells`
    returned, so a caller that catches this can report or retry the specific
    gap rather than re-deriving it from the message string. The reference
    column never raises this -- a slice can be complete on its mandatory
    cells with its reference column partly refused or missing.
    """

    def __init__(self, missing_cells: Sequence[tuple[str, str, str]]) -> None:
        self.missing_cells = list(missing_cells)
        rendered = ", ".join(
            f"{representation}/{similarity_backend}@{country}"
            for representation, similarity_backend, country in missing_cells
        )
        super().__init__(
            "baseline suite is incomplete -- missing "
            f"{len(self.missing_cells)} cell(s): {rendered}"
        )


def require_baseline_suite_complete(
    comparison: pl.DataFrame,
    *,
    countries: Sequence[str],
    refused_reference_cells: Sequence[tuple[str, str, str]] = (),
) -> None:
    """Raise `BaselineSuiteIncompleteError` unless `MANDATORY_SUITE` is complete.

    This is the gate a strategy-adoption decision is meant to call before
    treating a strategy comparison as evidence: the suite existing is not
    enough on its own, every mandatory cell in it must actually have
    completed. `refused_reference_cells` is forwarded to
    `missing_baseline_suite_cells()` only so a caller can pass one call's
    worth of state through both checks; it never affects whether this raises.
    """
    status = missing_baseline_suite_cells(
        comparison,
        countries=countries,
        refused_reference_cells=refused_reference_cells,
    )
    if status.missing_mandatory_cells:
        raise BaselineSuiteIncompleteError(status.missing_mandatory_cells)
