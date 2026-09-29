"""Sole owner of the on-disk root for every system's data, the `data/`
counterpart to `artifact_layout`'s `artifacts/`.

`data/` is what every stage reads or writes as its subject matter, one
subdirectory per system: `data/<system>/<layer>/<snapshot>/`, where a snapshot
is an acquisition date or the one `current` a derived layer holds. Three of a
system's datasets are named by a `data://<system>/<dataset>` reference, which
`workspace.reference` registers and `dataset_reference_at` reads back. No
function used to take a project root and return one of these paths -- unlike
`artifact_layout`'s `artifacts/` side, whose roots already had real call
sites before this module existed -- so every caller composed
`project_root / "data" / <system> / <layer>` inline instead, including this
package's own `cleanse_inputs` and `match_resolution`, four times between
them. This module is that missing half, shaped on the same signature
`artifact_layout` already proved: a root function per pattern, taking the
resolved `WorkspaceRoots` and returning the path, so a caller wanting one has
something to call and no caller decides for itself where `data/` is.

**Scope.** This module resolves only the root a layer lives under. What is
beneath that root -- the `primary`/`names` family split, partition
directories, which files hold the real data -- is `layer_layout`'s decision,
the same way a layer's own file names are `data_file_naming`'s business and
neither is this module's.
"""

from __future__ import annotations

import re
from pathlib import Path

from .kind_layout import Kind
from .layer_layout import primary_family_dir
from .reference import Reference, Side, locate, reference, reference_at
from .roots import DATA_DIR_NAME, WorkspaceRoots

__all__ = [
    "CANONICAL_LAYER_NAME",
    "CLEANSED_LAYER_NAME",
    "DATA_DIR_NAME",
    "MATCHED_LAYER_NAME",
    "TOKENIZED_LAYER_NAME",
    "canonical_snapshot_dir",
    "data_root",
    "dataset_reference_at",
    "latest_canonical_snapshot_dir",
    "layer_directory",
    "layer_system",
    "perturbed_dataset_dir",
    "perturbed_dataset_reference",
    "system_layer_dir",
]

CANONICAL_LAYER_NAME = "canonical"
"""Dated: a canonical directory holds one subdirectory per snapshot
(`canonical_snapshot_dir`), where every other layer below is undated."""

CLEANSED_LAYER_NAME = "cleansed"

MATCHED_LAYER_NAME = "matched"

TOKENIZED_LAYER_NAME = "tokenized"

_CURRENT_SNAPSHOT = "current"
"""The one snapshot directory a derived layer holds."""

_DERIVED_LAYERS = (CLEANSED_LAYER_NAME, MATCHED_LAYER_NAME, TOKENIZED_LAYER_NAME)
"""Layers holding one `current` snapshot, derived from the dated ones."""

_SNAPSHOT_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def data_root(roots: WorkspaceRoots) -> Path:
    """Where every system keeps its data, one subdirectory per system: the
    resolved data root, `data/` under the checkout unless overridden."""
    return roots.data


def layer_directory(system_root: Path, layer: str) -> Path:
    """Where one layer sits beneath a system's own directory, whichever tree
    that directory is in: `data/<system>`, a perturbed dataset's directory, or a
    configured prepared base's `<system>`.

    A derived layer (`cleansed`, `matched`, `tokenized`) holds one `current`
    snapshot, so this is `<system_root>/<layer>/current`, where its families and
    partitions sit. A dated layer (`acquire`, `prepare`, `source`, `canonical`)
    holds one directory per acquisition date, so this is the directory those sit
    in; `latest_canonical_snapshot_dir` picks the current one. Every layer is
    therefore two levels beneath its system root once a snapshot is chosen.
    """
    root = Path(system_root) / layer
    return root / _CURRENT_SNAPSHOT if layer in _DERIVED_LAYERS else root


def layer_system(layer_dir: Path) -> str:
    """The system a layer directory belongs to: the reverse of
    `layer_directory`, for a caller handed a layer's directory rather than
    its system. A snapshot directory, `current` or a date, sits two levels
    beneath its system; a dated layer's own directory, one."""
    layer_dir = Path(layer_dir)
    if layer_dir.name == _CURRENT_SNAPSHOT or _SNAPSHOT_DATE.fullmatch(layer_dir.name):
        return layer_dir.parent.parent.name
    return layer_dir.parent.name


def system_layer_dir(roots: WorkspaceRoots, *scope: str, layer: str) -> Path:
    """Where one layer of one system's data is read and written:
    `layer_directory` beneath `data/<system>`.

    `scope` is almost always a single system code. A caller with no system
    code of its own -- a synthetic dataset generated under a nested scope like
    `perturbed/<profile-id>` -- passes every segment of that scope in order,
    and gets the same shape beneath it.
    """
    return layer_directory(data_root(roots).joinpath(*scope), layer)


def dataset_reference_at(roots: WorkspaceRoots, snapshot_dir: Path) -> Reference:
    """The `data://` reference whose rows a snapshot directory holds: a dated
    `canonical` snapshot or `matched`'s one. A caller holding the directory it
    read records what it consumed through this, without knowing which family
    directory beneath the snapshot the reference locates."""
    return reference_at(roots, primary_family_dir(Path(snapshot_dir)))


def perturbed_dataset_reference(
    *, source_system: str, profile_id: str, version: str, seed: int
) -> Reference:
    """The reference naming one materialized perturbed dataset: the system it
    perturbs, the profile and that profile's version, and the seed."""
    return reference(
        Kind.PERTURBATION,
        Side.DATA,
        source=source_system,
        profile=profile_id,
        version=version,
        seed=str(seed),
    )


def perturbed_dataset_dir(
    roots: WorkspaceRoots,
    *,
    source_system: str,
    profile_id: str,
    version: str,
    seed: int,
) -> Path:
    """The directory one materialized perturbed dataset sits in, the location
    of `perturbed_dataset_reference`:
    `<source_system>/<profile_id>/<version>/<seed>` beneath the perturbation
    data root. Its layers sit beneath it in the same shape as a system's,
    through `layer_directory`."""
    return locate(
        roots,
        perturbed_dataset_reference(
            source_system=source_system,
            profile_id=profile_id,
            version=version,
            seed=seed,
        ),
    )


def canonical_snapshot_dir(
    roots: WorkspaceRoots, *, system: str, run_date: str
) -> Path:
    """Where one dated canonical snapshot lives for one system."""
    return system_layer_dir(roots, system, layer=CANONICAL_LAYER_NAME) / run_date


def latest_canonical_snapshot_dir(canonical_dir: Path) -> Path | None:
    """The current snapshot beneath one canonical layer directory: its
    lexicographically latest dated subdirectory, or None when it holds none.

    ISO `YYYY-MM-DD` names sort, and real data has more than one snapshot for
    at least one system, so "the only one" is never assumed. Takes the layer
    directory rather than the roots so a caller reading under an
    overridden base directory resolves the same way as one under `data/`;
    every reader of the current canonical rows makes this choice here rather
    than walking the directory itself.
    """
    if not canonical_dir.is_dir():
        return None
    dated = sorted(child for child in canonical_dir.iterdir() if child.is_dir())
    return dated[-1] if dated else None
