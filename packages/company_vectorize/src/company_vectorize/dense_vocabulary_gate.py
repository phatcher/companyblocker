"""Default-fail gate on dense-vocabulary representations at scale.

The gate applies when such a representation is run against an exhaustive similarity
backend.

`sklearn` and `sparse_dot_topn` are exhaustive source-by-target scans. They are
affordable for `tfidf` only because its large character-n-gram vocabulary means
most row pairs share no feature at all, so almost every pair costs nothing to
reject. `wordpiece` and `sentencepiece` emit a small, dense subword vocabulary
and get none of that for free: nearly every pair shares something, and the same
backend degrades from "fast enough" to "does not finish" as the target side
grows. Prefix filtering (`prefix_filter.py`) narrows the constant but not the
shape -- its documented dense-mask fallback returns a candidate set approaching
`n_targets` for exactly the rows that make this combination expensive.

The failure mode is a silent burn, not an error: a real `gleif -> gb` run of
`wordpiece`/`sklearn`/`prefix_filter` spent about five minutes building indexes
with no output, then scored on a single core with RSS climbing past 17GB before
being killed, and four separate `gb` evaluation attempts were lost the same way
before the pattern was recognized. Nothing about it is subtle once you know;
this gate exists so the machine knows, and refuses in a second instead of
twenty minutes.

Why a scale threshold rather than a flat refusal: a synthetic benchmark
measured this combination at parity-or-slower up to 20,000 target rows, and the
real crash was at 5,698,275 rows. Nothing in between has been measured, so
`DEFAULT_DENSE_VOCABULARY_MAX_ROWS` is a deliberately conservative
floor -- the largest scale anyone has actually benchmarked -- and not a proven
safe ceiling. Raise it only behind a measurement run.

Imported by module path (`company_vectorize.dense_vocabulary_gate`) rather than
re-exported from the package root, as `graph_cluster`/`sparse_similarity`/
`tfidf_cluster` already are; nothing here needs the shorter spelling.

`svd_rerank` is deliberately not gated: it projects to a fixed
low-dimensional dense representation before scoring, so its per-query cost is
bounded by `svd_dimensions` rather than by vocabulary density. `kmeans`/
`hdbscan` (`partition_similarity.py`) are not gated either -- they route each
source row to one partition instead of scanning the whole target side.
"""

from __future__ import annotations

from collections.abc import Mapping

# Representations whose vocabulary is small and dense enough that an exhaustive
# scan gets no free rejections from sparsity. Both are TF-IDF over emitted
# subword tokens (`token_list_strategy.py`); it is the token vocabulary, not
# the weighting, that makes them dense.
DENSE_VOCABULARY_REPRESENTATIONS: frozenset[str] = frozenset(
    {"wordpiece", "sentencepiece"}
)

# Backends that score every source row against every target row. `svd_rerank`
# is excluded on purpose -- see the module docstring.
EXHAUSTIVE_SPARSE_BACKENDS: frozenset[str] = frozenset({"sklearn", "sparse_dot_topn"})

DEFAULT_DENSE_VOCABULARY_MAX_ROWS = 20_000

FORCE_OPTION = "force"
MAX_ROWS_OPTION = "max_rows"

# The fixed substring `_refusal_message()` always emits, exported so a caller
# reading this gate's error back out of a subprocess's captured stderr (as
# `scripts/compare_blocking_strategies.py` does for a reference-column cell)
# can tell a scale refusal apart from any other `ValueError`
# without duplicating the sentence it is looking for.
REFUSAL_MESSAGE_MARKER = "is refused against the"

_TRUE_LITERALS = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSE_LITERALS = frozenset({"0", "false", "f", "no", "n", "off"})


class DenseVocabularyScaleError(ValueError):
    """A dense-vocabulary representation was paired with an exhaustive backend.

    Raised when that pairing happens at a target scale past `max_rows`, without `force`.
    """


def _coerce_force(value: object) -> bool:
    """Parse the `force` backend option strictly.

    Deliberately not `bool(value)`: these options arrive from a CLI's
    `--additional-args backend.force=false` as the *string* `"false"`, which is
    truthy, and silently bypassing a default-fail gate is the one outcome this
    module exists to prevent.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        literal = value.strip().lower()
        if literal in _TRUE_LITERALS:
            return True
        if literal in _FALSE_LITERALS:
            return False
    if isinstance(value, int):
        return bool(value)
    raise DenseVocabularyScaleError(
        f"backend option '{FORCE_OPTION}' must be a boolean "
        f"(or one of {sorted(_TRUE_LITERALS | _FALSE_LITERALS)}), got {value!r}."
    )


def _coerce_max_rows(value: object) -> int:
    if isinstance(value, bool):
        raise DenseVocabularyScaleError(
            f"backend option '{MAX_ROWS_OPTION}' must be a non-negative integer "
            f"row count, got {value!r}."
        )
    if isinstance(value, (int, float, str)):
        try:
            max_rows = int(value)
        except (TypeError, ValueError):
            max_rows = -1
        if max_rows >= 0:
            return max_rows
    raise DenseVocabularyScaleError(
        f"backend option '{MAX_ROWS_OPTION}' must be a non-negative integer "
        f"row count, got {value!r}."
    )


def resolve_dense_vocabulary_gate_options(
    backend_options: Mapping[str, object] | None,
) -> tuple[bool, int]:
    """Read this gate's two options out of a `backend_options` mapping.

    Args:
        backend_options: The call's backend options, or `None`.

    Returns:
        `(force, max_rows)`, defaulting to `(False,
        DEFAULT_DENSE_VOCABULARY_MAX_ROWS)` when the keys are absent.

    Raises:
        DenseVocabularyScaleError: Either option is present but not a value
            this gate can read.
    """
    options = backend_options or {}
    force = _coerce_force(options.get(FORCE_OPTION, False))
    max_rows = _coerce_max_rows(
        options.get(MAX_ROWS_OPTION, DEFAULT_DENSE_VOCABULARY_MAX_ROWS)
    )
    return force, max_rows


def _refusal_message(
    *, representation: str, backend: str, target_rows: int, max_rows: int
) -> str:
    return (
        f"The '{representation}' representation {REFUSAL_MESSAGE_MARKER} "
        f"'{backend}' similarity backend at {target_rows:,} target rows "
        f"(max_rows={max_rows:,}). '{backend}' scores every source row against "
        "every target row, which is only affordable for 'tfidf': its large "
        "character-n-gram vocabulary means most row pairs share no feature and "
        f"cost nothing to reject. '{representation}' emits a small, dense "
        "subword vocabulary, so nearly every pair shares something and the scan "
        "degrades from slow to non-terminating as the target side grows "
        "(prefix_filter narrows the constant, not the shape). A real "
        "gleif -> gb run of this combination burned about twenty minutes and "
        "over 17GB of RSS with no output before being killed. Pick one of: "
        f"(1) a target side of at most {max_rows:,} rows, which is what "
        "--max-rows caps -- raise it only behind an actual measurement, since "
        "the default is the largest scale ever benchmarked, not a proven safe "
        "ceiling; (2) a different backend -- 'svd_rerank' bounds per-query cost "
        "by its fixed low-dimensional projection, and 'kmeans' routes each "
        "source row to one target partition instead of scanning all of them; "
        "(3) the 'tfidf' representation, whose sparse vocabulary is what this "
        "backend assumes; or (4) --force (backend_options "
        f"{{'{FORCE_OPTION}': True}}) to run this combination anyway, accepting "
        "the runtime and memory cost."
    )


def ensure_dense_vocabulary_scale_supported(
    *,
    representation: str,
    backend: str,
    target_rows: int,
    backend_options: Mapping[str, object] | None = None,
) -> None:
    """Reject a dense-vocabulary representation on an exhaustive backend at scale.

    A no-op for any combination outside the gate: a sparse-vocabulary
    representation, a backend that is not an exhaustive scan, a target side
    within `max_rows`, or an explicit `force`.

    Args:
        representation: The clustering representation about to run (for
            example `"wordpiece"`).
        backend: The similarity backend about to run (for example
            `"sklearn"`).
        target_rows: How many rows the target side carries. Callers may pass
            either the loaded target frame's height or the built index's
            `len(target_ids)` -- this is a scale check, and the two differ
            only by empty and variant rows.
        backend_options: The call's backend options. `force` bypasses the gate
            outright; `max_rows` moves the threshold (default
            `DEFAULT_DENSE_VOCABULARY_MAX_ROWS`).

    Raises:
        DenseVocabularyScaleError: The combination is gated, `target_rows`
            exceeds `max_rows`, and `force` is not set -- or one of the two
            options could not be read.
    """
    if representation not in DENSE_VOCABULARY_REPRESENTATIONS:
        return
    if backend not in EXHAUSTIVE_SPARSE_BACKENDS:
        return

    force, max_rows = resolve_dense_vocabulary_gate_options(backend_options)
    if force or target_rows <= max_rows:
        return

    raise DenseVocabularyScaleError(
        _refusal_message(
            representation=representation,
            backend=backend,
            target_rows=target_rows,
            max_rows=max_rows,
        )
    )
