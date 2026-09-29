"""Direct tests for `workspace.kind_layout`.

The point of this module is that a caller never learns which tree a kind sits
in, so what is pinned here is the resolved shape rather than any caller's use
of it. Two of these assertions are deliberately about what is *not* shared:
a kind's profile directory and its data directory are independent locations,
and the concern's name is independent of its directory's, so neither can be
quietly re-coupled without a test saying so.
"""

from __future__ import annotations

from pathlib import Path

from workspace import kind_layout
from workspace.artifact_layout import artifact_root, blocking_artifact_root
from workspace.kind_layout import (
    Kind,
    _Location,
    data_directory,
    profile_directory,
)
from workspace.roots import WorkspaceRoots, default_workspace_roots


def test_profile_directory_resolves_under_the_config_root_by_concern(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """Authored input is configuration, so it sits under `config/`, named by the
    concern as a reference's scheme is, never under `artifacts/`."""
    assert profile_directory(workspace_roots, Kind.PERTURBATION) == (
        tmp_path / "config" / "perturbation" / "profile"
    )


def test_data_directory_groups_by_concern_then_data(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert data_directory(workspace_roots, Kind.PERTURBATION) == (
        tmp_path / "artifacts" / "perturbed" / "data"
    )


def test_blocking_data_directory_is_the_leaf_of_the_blocking_artifact_root(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """Blocking's runs are one leaf under the concern's own root, beside the
    cross-pairing aggregate that root also holds, so the two resolvers must
    agree on the segment rather than each spelling `blocking`."""
    assert data_directory(workspace_roots, Kind.BLOCKING) == (
        blocking_artifact_root(workspace_roots) / "data"
    )


def test_profile_and_data_directories_are_independent_locations(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """Neither contains the other, so moving one cannot drag the other with it.

    Authored profiles are tracked and hand-edited; generated data is large and
    regenerable. Nesting either inside the other is what would force a
    `.gitignore` carve-out and couple the two lifecycles.
    """
    profiles = profile_directory(workspace_roots, Kind.PERTURBATION)
    data = data_directory(workspace_roots, Kind.PERTURBATION)

    assert profiles != data
    assert data not in profiles.parents
    assert profiles not in data.parents


def test_kind_names_the_concern_not_the_data_directory():
    """`perturbation` is the concern; `perturbed` is its data directory on disk.

    The divergence is deliberate and held in `_LOCATIONS`, so neither name has
    to bend to the other. Asserting the enum value alone would pass if the
    mapping were dropped and the kind renamed to match the directory.
    """
    assert Kind.PERTURBATION.value == "perturbation"
    assert (
        "perturbation"
        not in data_directory(
            default_workspace_roots(Path("/x")), Kind.PERTURBATION
        ).parts
    )


def test_a_kind_may_span_several_segments(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    """A kind's location is a path, not a name.

    No shipped kind needs more than one segment yet, so the capability is
    exercised through the private table rather than left unasserted until one
    does: a caller that later sits beneath another concern must not need a
    signature change here.
    """
    monkeypatch.setitem(
        kind_layout._LOCATIONS,
        Kind.PERTURBATION,
        _Location(segments=("outer", "inner"), data_root=artifact_root),
    )

    assert data_directory(workspace_roots, Kind.PERTURBATION) == (
        tmp_path / "artifacts" / "outer" / "inner" / "data"
    )
