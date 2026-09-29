"""Resolves a directory by *kind*, so no caller learns which tree it lives in.

`data_layout` and `artifact_layout` each own one tree and name it in their own
signatures: a caller reaching for `artifact_root` has already decided its
subject is an artifact, and one reaching for `system_layer_dir` has already
decided the shape is `<scope>/<layer>`. That is the right contract for
something whose tree is settled, and the wrong one for something whose tree is
still a question, because the answer leaks into every call site and moving it
later is a sweep rather than an edit.

Here a caller names the *concern* and what it wants of it, and this module
decides where that is. The tree, the segments, and whether an instance nests
beneath a root are all internal and changeable without touching a caller.

    config/
        <kind>/profile/              authored profiles for one concern
    artifacts/
        <segments...>/data/          that concern's generated data

This module places each side's root; beneath it, the directory each reference
occupies is the kind's layout in `workspace.reference`, registered there before
any area can build one.

A kind's segments are a path, not a name: one is the common case, and a kind
that needs to sit beneath another concern, or to split by a second axis, adds
them without a signature change here or at any call site.

**Which tree a kind's data lives in is part of the kind.** `data/` is what a
stage reads or writes as its subject matter, one subdirectory per acquired
system; `artifacts/` is what a stage produces *about* that data. Perturbation
data is derived from a named profile rather than acquired, so it resolves
under `artifacts/`. That is recorded per kind in `_LOCATIONS`, not assumed
here, so a kind whose data really is acquired resolves under `data/` without
this module growing a special case.

**Why two functions, not one.** They hide the same kind of decision but answer
different questions: authored input that is hand-edited and tracked, against
generated output that is large and regenerable. Different lifecycles, different
`.gitignore` treatment, and folding them into one keyed space would re-couple
them. Keeping them apart also makes the function the namespace, so a kind stays
short and needs no qualifier of its own: `profile_directory(root, Kind.CLEANSE)`
and `data_directory(root, Kind.CLEANSE)` are unambiguous and mean different
things, where a bare `cleanse` directory would not be.

**One shared kind vocabulary.** `Kind` serves both functions, so a concern
means the same thing on either side. A kind may legitimately exist on only one:
a benchmark kind would be data-only, a threshold profile profile-only. Kinds
are added when they get a real first caller, not in advance -- the rule
`artifact_layout` states for its own roots, reflecting what is on disk rather
than importing a split speculatively.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .artifact_layout import BLOCKING_DIR_NAME, TOKENIZERS_DIR_NAME, artifact_root
from .roots import WorkspaceRoots

PROFILE_DIR_NAME = "profile"

DATA_DIR_NAME = "data"
"""The leaf under a concern's own directory holding its generated datasets,
separating them from anything else that concern writes beside them."""


class Kind(StrEnum):
    """A concern that owns authored profiles, generated data, or both."""

    PERTURBATION = "perturbation"
    BLOCKING = "blocking"
    BLOCKING_AUDIT = "blocking-audit"
    DATA = "data"
    TOKENIZER = "tokenizer"


@dataclass(frozen=True)
class _Location:
    """Where one kind sits: its path on disk, which tree its data lives in, and
    the leaf its datasets sit beneath, or None when they fill the tree itself."""

    segments: tuple[str, ...]
    data_root: Callable[[WorkspaceRoots], Path]
    leaf: str | None = DATA_DIR_NAME


def _data_root(roots: WorkspaceRoots) -> Path:
    return roots.data


_LOCATIONS: dict[Kind, _Location] = {
    # The concern is named `perturbation`; the directory is `perturbed`, which
    # is what materialization already writes and what the `perturbed://` URI
    # scheme says. Held here so neither name has to bend to the other.
    Kind.PERTURBATION: _Location(segments=("perturbed",), data_root=artifact_root),
    # Blocking runs are produced about the data, keyed by configuration, so
    # they sit under `artifacts/`; the segment is the same one
    # `artifact_layout.blocking_artifact_root` names, so the concern's own
    # root and its data leaf cannot drift apart. What sits beneath the leaf,
    # the target/kind/source hierarchy and the per-run key, is the `blocking`
    # layout registered in `workspace.reference`.
    Kind.BLOCKING: _Location(segments=(BLOCKING_DIR_NAME,), data_root=artifact_root),
    # A run's audit sits beside its data, under the same `blocking`
    # segment, in its own `audit` leaf -- never inside a run's own directory
    # (the audit's own decision), and never inside `data/` either, so a glob
    # over `data/` still finds only runs. One layout per kind of source could
    # be registered here later, the way `_PAIRING_LAYOUTS` does for runs
    # themselves; today only a plain (`BlockingSourceKind.DATA`) run's audit
    # is wired (`blocking.run_layout.audit_reference_for`).
    Kind.BLOCKING_AUDIT: _Location(
        segments=(BLOCKING_DIR_NAME, "audit"), data_root=artifact_root
    ),
    # A system's datasets are the data root's own subject matter, one directory
    # per system, so they fill the tree with no segment and no leaf.
    Kind.DATA: _Location(segments=(), data_root=_data_root, leaf=None),
    # Promoted tokenizer candidates are one leaf under the tokenizer artifact
    # root, beside the working tree `company_tokenize` lays out for itself.
    Kind.TOKENIZER: _Location(segments=(TOKENIZERS_DIR_NAME,), data_root=artifact_root),
}


def profile_directory(roots: WorkspaceRoots, kind: Kind) -> Path:
    """Where `kind`'s authored profiles live: `config/<kind>/profile`.

    Input rather than output: hand-edited, small, tracked in git, so under the
    config root with the checkout's other configuration rather than under
    `artifacts/`, which holds only what runs generate. Named by the concern
    itself, the same word as a reference's scheme, and deliberately independent
    of `data_directory`, so one can move without dragging the other. Drafts sit
    in untracked `draft/` directories beneath it.
    """
    return roots.config / kind.value / PROFILE_DIR_NAME


def data_directory(roots: WorkspaceRoots, kind: Kind) -> Path:
    """Where `kind`'s generated data lives: `<tree>/<segments...>/data`.

    The root only. The directories beneath it, one per reference, are the
    kind's layout registered in `workspace.reference`, never the producing
    area's; the files inside one reference's directory are the area's own.

    Grouped by concern first and `data` second, the mirror of
    `profile_directory`, so everything one concern generates sits together and
    its datasets are one leaf among them rather than the whole of it.
    """
    location = _LOCATIONS[kind]
    root = location.data_root(roots).joinpath(*location.segments)
    return root if location.leaf is None else root / location.leaf


__all__ = [
    "DATA_DIR_NAME",
    "PROFILE_DIR_NAME",
    "Kind",
    "data_directory",
    "profile_directory",
]
