"""A population's derived name forms, derived once and read by every run after.

A derived name form is a function of the raw names and of the profiles that
shaped it: the cleansed name of the cleanse profile, the preprocessed name of
the cleanse profile and the preprocessing profile both. A column a layer
carries holds one profile's answer from whenever that layer was last written,
so a run under another profile cannot use it, and two areas scoring the same
names must not each derive their own and hope to agree. One entry per
population and pair of profiles settles both: the first run to ask derives
and stores it, every later run, in any area, reads the same columns back.

An entry sits on the keyed, append-only store `artifact_archive` provides,
under `artifacts/store/name-forms/<system>/<country>/<key>/`, as one parquet
of the id column and the derived columns. Its key digests the population,
`rows_digest` over the id and the raw name, with the two profiles and the
deriver's version, so rewriting a layer with the same rows moves nothing and
a changed name, profile or ruleset gives a new entry.

This module knows nothing about how a form is derived: `workspace` imports
neither cleanse nor tokenize, so the caller hands the derivation in.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import polars as pl

from .artifact_archive import (
    artifact_is_complete,
    begin_artifact_write,
    commit_artifact_write,
    resolve_candidate_dir,
)
from .artifact_layout import artifact_store_root
from .identity import rows_digest
from .roots import WorkspaceRoots

NAME_FORMS_STORE_FACET = "name-forms"
NAME_FORMS_FILENAME = "name_forms.parquet"

Derive = Callable[[pl.DataFrame], pl.DataFrame]


def name_forms_settings(
    *,
    population_key: str,
    cleanse_profile: str,
    preprocess_profile: str,
    scored_col: str,
    deriver_version: str,
) -> dict[str, object]:
    """What an entry is keyed by, and the manifest written beside it.
    `scored_col` is the column the preprocessed name was made from, which a
    run's name transform chooses."""
    return {
        "population_key": population_key,
        "cleanse_profile": cleanse_profile,
        "preprocess_profile": preprocess_profile,
        "scored_col": scored_col,
        "deriver_version": deriver_version,
    }


def _facet(value: str) -> str:
    """`value` as one directory name: a perturbed dataset names its system by a URI."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "_"


def resolve_name_forms(
    roots: WorkspaceRoots,
    frame: pl.DataFrame,
    *,
    system: str,
    country: str,
    columns: Sequence[str],
    cleanse_profile: str,
    preprocess_profile: str,
    scored_col: str,
    deriver_version: str,
    derive: Derive,
    id_col: str = "system_uri",
    name_col: str = "name",
    population_key: str | None = None,
) -> tuple[pl.DataFrame, bool]:
    """`frame` with `columns` attached, and whether they were read from the store.

    `derive` takes `frame` and returns it carrying `columns`; it runs only
    when the store holds no entry for these rows under these profiles, and
    what it returns is stored before anything is handed back. The columns
    attached are always the stored ones, read or just written, so a first run
    and a later one hand back the same values. Any of `columns` that `frame`
    already carried is replaced, its other columns and its row order are
    kept, and an empty frame is derived and never stored. `population_key`
    is `rows_digest` over `id_col` and `name_col` when the caller already has
    it, and is computed here otherwise.
    """
    if frame.height == 0:
        return derive(frame), False

    settings = name_forms_settings(
        population_key=population_key
        or rows_digest(frame, id_col=id_col, value_col=name_col),
        cleanse_profile=cleanse_profile,
        preprocess_profile=preprocess_profile,
        scored_col=scored_col,
        deriver_version=deriver_version,
    )
    entry = resolve_candidate_dir(
        artifact_store_root(roots),
        NAME_FORMS_STORE_FACET,
        _facet(system),
        _facet(country),
        settings=settings,
    )
    stored = artifact_is_complete(entry)
    if not stored:
        derived = derive(frame)
        missing = [name for name in (id_col, *columns) if name not in derived.columns]
        if missing:
            raise ValueError(
                f"The name forms derivation returned no {missing!r} column."
            )
        staging = begin_artifact_write(entry)
        derived.select(id_col, *columns).unique(
            subset=[id_col], keep="first", maintain_order=True
        ).write_parquet(staging / NAME_FORMS_FILENAME)
        commit_artifact_write(entry, staging, manifest=settings)

    forms = pl.read_parquet(entry / NAME_FORMS_FILENAME)
    carried = [name for name in columns if name in frame.columns]
    return (
        frame.drop(carried).join(forms, on=id_col, how="left", maintain_order="left"),
        stored,
    )


__all__ = [
    "NAME_FORMS_FILENAME",
    "NAME_FORMS_STORE_FACET",
    "name_forms_settings",
    "resolve_name_forms",
]
