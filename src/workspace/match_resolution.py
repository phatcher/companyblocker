"""Resolve a row's cross-system match from its URI alone.

`data/<code>/matched/` holds the only place a cross-system match is ever
recorded, keyed by the *source* row's own `system_uri` with `match_uri`
naming the target -- a target system never carries the relationship itself.
An entity row's own `system_uri` is that key directly. A derived row, a name
variant (`name://gb/123/<hash>`) or a perturbed copy of either
(`perturbed://gb/en-lite/v1/42/123`), carries no entry of its own: its URI is
its derivation and the URI it was derived from, so `workspace.derived_uri`
decomposes it to its entity with no dataset read, and the entity's match is the
row's match.

Every caller with the same need (`blocking`, `validation`,
`company_vectorize`) used to re-derive this hop, or skip it and read
`system_uri` directly -- which resolves nothing for a row that has been
given its own per-row identity, and does so *silently*: the join simply
returns no rows rather than failing. `resolve_cross_system_match` is the one
mechanism instead: decompose to the entity, then one `matched/` lookup, and a
URI that names no row of any kind fails (`UnknownUriSchemeError`) rather than
reading as "no match".

**Depth is not hardcoded.** The walk follows `DerivedUri.source` until it
reaches the underived case, so an entity resolves in no step, a name row in
one and a perturbed name row in two, with no depth parameter or bound.

**No system is passed in.** The URI says whose row it is, so the system whose
`matched/` layer is read is the entity's own.
"""

from __future__ import annotations

import polars as pl

from .data_layout import MATCHED_LAYER_NAME, system_layer_dir
from .derived_uri import DerivedUri, parse_uri
from .layer_layout import resolve_primary_files
from .reference import InvalidReferenceError
from .roots import WorkspaceRoots


class MatchResolutionError(ValueError):
    """Base class for every way `resolve_cross_system_match` can fail."""


class UnknownUriSchemeError(MatchResolutionError):
    """A `system_uri` that names no entity, name variant or perturbed row."""


class MatchLayerNotFoundError(MatchResolutionError):
    """A system's `matched/` layer has no primary files at all -- the system
    is not participating in cross-system matching, detected rather than
    assumed silently absent."""


def entity_of(uri: str) -> DerivedUri:
    """The entity a row's URI decomposes to, the row itself where it is one."""
    try:
        row = parse_uri(uri)
    except InvalidReferenceError as error:
        raise UnknownUriSchemeError(
            f"system_uri {uri!r} names no row: {error}"
        ) from error
    if not row.is_row:
        raise UnknownUriSchemeError(f"system_uri {uri!r} names a dataset, not a row.")
    while row.source is not None:
        row = row.source
    return row


def _lookup_match_uri(*, roots: WorkspaceRoots, system: str, uri: str) -> str | None:
    matched_dir = system_layer_dir(roots, system, layer=MATCHED_LAYER_NAME)
    files = resolve_primary_files(matched_dir, system_code=system)
    if not files:
        raise MatchLayerNotFoundError(
            f"System {system!r} has no 'matched/' layer under {matched_dir} -- it is not "
            "participating in cross-system matching."
        )
    frame = pl.concat([pl.read_parquet(path) for path in files], how="vertical_relaxed")
    if "system_uri" not in frame.columns:
        raise MatchResolutionError(
            f"System {system!r}'s 'matched/' layer has no 'system_uri' column."
        )
    matches = frame.filter(pl.col("system_uri") == uri)
    if matches.height == 0:
        return None
    if "match_uri" not in matches.columns:
        raise MatchResolutionError(
            f"System {system!r}'s 'matched/' layer has no 'match_uri' column."
        )
    match_uri = matches.row(0, named=True)["match_uri"]
    return None if match_uri is None else str(match_uri)


def resolve_cross_system_match(roots: WorkspaceRoots, *, uri: str) -> str | None:
    """A row's cross-system match, or `None` when the row's entity has no
    recorded match.

    `uri` is the row's own `system_uri`. It is decomposed to its entity, whose
    system's `matched/` layer is the one lookup made.
    """
    entity = entity_of(uri)
    return _lookup_match_uri(roots=roots, system=entity.system, uri=entity.uri)
