"""Which reference of a kind a profile name stands for: a tracked file under `config/`.

An artifact store keeps every candidate a producer ever wrote, each under its
own reference, and none of them is tracked. Which one a reader should load is
a decision, so it is small, authored by a promotion and tracked in git: one
JSON file per kind, `config/<kind>s.json`, mapping a profile name and a
selection to the complete reference the name stands for and the reference
that one replaced.

    {"promoted": {"tokenizer://ie/wordpiece": {
        "reference": "tokenizer://ie/wordpiece/<key>",
        "replaced": null}},
     "naive": {"tokenizer://ie/wordpiece": {...}}}

The selection is the slot: a tokenizer's is its scope and tokenizer id. A
profile is a short name meaning the same thing for every slot, `promoted`
unless another is given, so each slot has one reference per profile and each
profile keeps its own record of what it replaced. This module knows nothing
about what a reference's directory holds, which is its producer's, and nothing
tokenizer-specific: any kind registered in `workspace.reference` can keep a
pointer the same way.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from .kind_layout import Kind
from .reference import Reference, locate, parse_reference
from .roots import WorkspaceRoots

PROMOTED_PROFILE = "promoted"
REFERENCE_FIELD = "reference"
REPLACED_FIELD = "replaced"

_PROFILE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

Entries = dict[str, dict[str, dict[str, str | None]]]


class PointerError(LookupError):
    """A pointer names no reference for a profile and selection, or names one
    whose directory does not exist."""


class NothingPromotedError(PointerError):
    """A profile names nothing for a selection. Told apart from a named
    reference that is absent, since a caller with a default of its own may
    fall back on the first and must never on the second."""


def pointer_path(roots: WorkspaceRoots, kind: Kind) -> Path:
    """The file holding `kind`'s promoted references: `config/<kind>s.json`."""
    return roots.config / f"{kind.value}s.json"


def _covers(selection: Reference, ref: Reference) -> bool:
    chosen = ref.fields
    return (
        selection.kind is ref.kind
        and selection.side is ref.side
        and all(chosen.get(name) == value for name, value in selection.values)
    )


def _checked_profile(profile: str) -> str:
    if not _PROFILE_PATTERN.fullmatch(profile):
        raise ValueError(
            f"A pointer profile is a short lowercase name, not {profile!r}."
        )
    return profile


def _read(path: Path) -> Entries:
    if not path.is_file():
        return {}
    return {
        str(profile): {str(slot): dict(entry) for slot, entry in slots.items()}
        for profile, slots in json.loads(path.read_text(encoding="utf-8")).items()
    }


def promoted_reference(
    roots: WorkspaceRoots, selection: Reference, *, profile: str = PROMOTED_PROFILE
) -> Reference:
    """The reference `profile` names for `selection`, whose directory exists.

    Raises `PointerError` naming the profile and the selection when the
    profile names nothing for it, and naming the reference when its directory
    is absent, which is what a checkout sees before the candidate has been
    trained into its artifact store.
    """
    path = pointer_path(roots, selection.kind)
    entry = _read(path).get(_checked_profile(profile), {}).get(selection.uri)
    if entry is None or not entry.get(REFERENCE_FIELD):
        raise NothingPromotedError(
            f"{path} names no {profile} reference for {selection.uri}."
        )
    ref = parse_reference(str(entry[REFERENCE_FIELD]))
    if not ref.complete or not _covers(selection, ref):
        raise PointerError(
            f"{path} names {ref.uri} as {profile} for {selection.uri}, "
            f"which it does not name."
        )
    location = locate(roots, ref)
    if not location.is_dir():
        raise PointerError(
            f"{path} names {ref.uri} as {profile} for {selection.uri}, "
            f"which is absent at {location}."
        )
    return ref


def promote(
    roots: WorkspaceRoots,
    selection: Reference,
    ref: Reference,
    *,
    profile: str = PROMOTED_PROFILE,
) -> Reference | None:
    """Make `ref` the reference `profile` names for `selection`, recording the
    one it replaced, and return that one, or None when there was none.

    Refuses a reference `selection` does not name or whose directory is
    absent, so a pointer never names what a reader cannot load. Naming the
    reference the profile already names changes nothing and keeps what it
    replaced. Another profile's entry for the same selection is left alone.
    """
    _checked_profile(profile)
    if not ref.complete or not _covers(selection, ref):
        raise PointerError(f"{selection.uri} does not name {ref.uri}.")
    location = locate(roots, ref)
    if not location.is_dir():
        raise PointerError(f"{ref.uri} is absent at {location}; nothing to promote.")
    path = pointer_path(roots, selection.kind)
    entries = _read(path)
    slots = entries.get(profile, {})
    current = (slots.get(selection.uri) or {}).get(REFERENCE_FIELD)
    if current == ref.uri:
        replaced = slots[selection.uri].get(REPLACED_FIELD)
        return parse_reference(replaced) if replaced else None
    slots[selection.uri] = {REFERENCE_FIELD: ref.uri, REPLACED_FIELD: current}
    entries[profile] = slots
    _write(path, entries)
    return parse_reference(current) if current else None


def _write(path: Path, entries: Entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(entries, indent=2, sort_keys=True) + "\n"
    handle, staging = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        os.replace(staging, path)
    except BaseException:
        Path(staging).unlink(missing_ok=True)
        raise


__all__ = [
    "PROMOTED_PROFILE",
    "NothingPromotedError",
    "PointerError",
    "pointer_path",
    "promote",
    "promoted_reference",
]
