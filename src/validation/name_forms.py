"""Every cleanse-derived form of a name, recomputed live, and what a stored copy is keyed by.

A layer's own `name_cleansed` reflects whichever ruleset and profile were in
force when that system's Cleanse stage last ran, two systems cleansed at
different times disagree on the same name, and a perturbed row never passed
the Cleanse stage at all. So a run derives the forms from the raw `name`
under its own profile, through `workspace.name_forms_store`, which keeps them
for every later run over the same names. `blocking` and this area both score
names, so what is derived, which columns a stored entry keeps and the digest
of the rules it was derived under are declared here once, where both import.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.resources import files

import polars as pl
from company_cleanse import CleanseConfig, cleanse_lazyframe

RAW_NAME_COLUMN = "name"

# The cleanse pipeline derives each form through Python callbacks
# (`company_cleanse.pipeline`'s `map_elements`), so one pass holds the GIL and
# runs on one core whatever Polars' thread count is: 693 s over `gb`'s 5.9M
# rows at 122% CPU. A row's forms depend on that row alone, so the frame is
# split and the chunks derived in worker processes, which is the one way
# around the GIL without rewriting every callback as an expression.
#
# Four by default, the Polars cap's own figure
# (`docs/findings/polars-memory.md`): each worker holds its chunk and the
# forms it derives, so the
# cost of another is memory, not cores. `NAME_FORMS_WORKERS` overrides it,
# and 1 derives in this process as before.
_WORKERS_ENV_VAR = "NAME_FORMS_WORKERS"
_DEFAULT_WORKERS = 4
# Below this a pool costs more than it saves: a worker pays for its own
# interpreter and imports (seconds on Windows, where processes are spawned).
_PARALLEL_MIN_ROWS = 250_000


def name_forms_workers() -> int:
    """How many processes `derive_name_forms` splits a large frame across."""
    raw = os.environ.get(_WORKERS_ENV_VAR)
    if raw is None:
        return _DEFAULT_WORKERS
    try:
        workers = int(raw)
    except ValueError:
        return _DEFAULT_WORKERS
    return max(1, workers)


def _derive(frame: pl.DataFrame, *, profile: str) -> pl.DataFrame:
    """One `cleanse_lazyframe` pass over `frame`, in this process."""
    config = CleanseConfig(company_col=RAW_NAME_COLUMN, normalization_profile=profile)
    return cleanse_lazyframe(frame.lazy(), config).collect()


def derive_name_forms(
    frame: pl.DataFrame, *, profile: str, workers: int | None = None
) -> pl.DataFrame:
    """`frame` with every cleanse-derived name form recomputed live from its
    raw `name` under `profile`, replacing any the frame carried.

    One `cleanse_lazyframe` pass, split across `workers` processes for a
    frame past `_PARALLEL_MIN_ROWS` rows and derived in this process
    otherwise. The split is row-wise and the chunks are concatenated in
    order, so the frame returned is the one pass's own: a row's forms are
    derived from that row alone. `workers` defaults to `name_forms_workers()`.

    A frame with no `name` column cannot be derived from and is refused
    rather than passed through, since a run that silently kept a
    materialized column would be reading exactly what this exists to stop.
    """
    if RAW_NAME_COLUMN not in frame.columns:
        raise ValueError(
            f"name forms are derived from the {RAW_NAME_COLUMN!r} column, which "
            f"the frame does not carry; columns are {frame.columns!r}"
        )
    if frame.height == 0:
        return frame

    resolved = name_forms_workers() if workers is None else max(1, workers)
    if resolved == 1 or frame.height < _PARALLEL_MIN_ROWS:
        return _derive(frame, profile=profile)

    chunk_rows = -(-frame.height // resolved)
    chunks = [
        frame.slice(start, chunk_rows) for start in range(0, frame.height, chunk_rows)
    ]
    with ProcessPoolExecutor(max_workers=resolved) as pool:
        derived = list(pool.map(_derive_chunk, ((chunk, profile) for chunk in chunks)))
    return pl.concat(derived, how="vertical")


def _derive_chunk(work: tuple[pl.DataFrame, str]) -> pl.DataFrame:
    """One chunk's forms, as a worker process derives them. Top level and
    single-argument so it pickles for `ProcessPoolExecutor.map`."""
    chunk, profile = work
    return _derive(chunk, profile=profile)


NAME_FORM_COLUMNS: tuple[str, ...] = (
    "short_name",
    "quoted_name",
    "acronym",
    "name_cleansed_basic",
    "name_cleansed",
    "personal_owner",
    "company_type_source",
    "company_type_missing",
    "company_type",
)
"""Every column `derive_name_forms` adds, which is what a stored entry keeps."""


@lru_cache(maxsize=1)
def name_forms_ruleset_digest() -> str:
    """Digest of the rule files `company_cleanse` packages: company types, legal
    forms, geographic terms and noise words. A stored entry is keyed on it, so
    a changed rule derives the forms again. The package's version does not
    move with its rules, and a change to its code is a run's commit, as it is
    for the run keys."""
    digest = hashlib.blake2b(digest_size=8)
    for resource in sorted(
        files("company_cleanse").joinpath("resources").iterdir(), key=lambda r: r.name
    ):
        if resource.name.endswith(".json"):
            digest.update(resource.name.encode("utf-8"))
            digest.update(resource.read_bytes())
    return digest.hexdigest()


__all__ = [
    "NAME_FORM_COLUMNS",
    "RAW_NAME_COLUMN",
    "derive_name_forms",
    "name_forms_ruleset_digest",
    "name_forms_workers",
]
