"""The uniqueness check over a system's name-variant rows.

A name row's own `system_uri`, `name://<system>/<id>/<hash>`, is composed by
`workspace.derived_uri.name_variant_uri` from the entity it is a name of and the
name's type and value, so identity is a pure function of content and stable
across reruns. What stays here is the check that no two rows of one system share
the triple that identity is made from.
"""

from __future__ import annotations

import polars as pl

_MAX_REPORTED_DUPLICATES = 20


class NameVariantUniquenessError(ValueError):
    pass


def check_name_variant_uniqueness(frame: pl.DataFrame, *, system_code: str) -> None:
    """Hard-fail if `frame` (all of one system's own name-variant sidecar
    rows, across every shard) has more than one row sharing the same
    `(source_uri, name_type, name)` triple.

    Scoped per system's own sidecar, not globally cross-entity: different
    entities legitimately sharing a former/trading name is real and must
    stay valid, so this only ever compares rows that already share the same
    `source_uri`. Aggregates every violation into one error (rather than
    failing on the first bad row) so a batch of duplicates can be
    investigated in one pass -- mirrors
    `validation.perturbation_materializer.PerturbationCollisionError`'s
    aggregated reporting."""

    duplicate_keys = (
        frame.group_by(["source_uri", "name_type", "name"])
        .agg(pl.len().alias("_count"))
        .filter(pl.col("_count") > 1)
        .sort(["source_uri", "name_type", "name"])
    )
    if duplicate_keys.height == 0:
        return

    all_keys = duplicate_keys.select(["source_uri", "name_type", "name"]).rows()
    preview = all_keys[:_MAX_REPORTED_DUPLICATES]
    preview_text = ", ".join(
        f"(source_uri={source_uri!r}, name_type={name_type!r}, value={value!r})"
        for source_uri, name_type, value in preview
    )
    suffix = (
        f" (and {len(all_keys) - _MAX_REPORTED_DUPLICATES} more)"
        if len(all_keys) > _MAX_REPORTED_DUPLICATES
        else ""
    )
    raise NameVariantUniquenessError(
        f"{system_code}: {len(all_keys)} duplicate (source_uri, name_type, value) "
        f"triple(s) in the name-variant sidecar: {preview_text}{suffix}"
    )
