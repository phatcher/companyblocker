"""Direct tests for `workspace.data_layout`.

Each assertion pins a root this module exposes to the real
`data/<system>/<layer>/<snapshot>/` shape stated in
`src/workspace/README.md`'s layer contract -- see
`src/workspace/data_layout.py`'s module docstring for why no such function
existed before this one. A caller composing its own `"data"` path is exactly
the drift this module exists to close, so these are pinned independently of
any one caller's test.
"""

from __future__ import annotations

from pathlib import Path

from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    canonical_snapshot_dir,
    data_root,
    dataset_reference_at,
    latest_canonical_snapshot_dir,
    layer_directory,
    layer_system,
    perturbed_dataset_dir,
    perturbed_dataset_reference,
    system_layer_dir,
)
from workspace.reference import locate
from workspace.roots import WorkspaceRoots


def test_data_root(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert data_root(workspace_roots) == tmp_path / "data"


def test_system_layer_dir_is_a_derived_layer_s_current_snapshot(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert (
        system_layer_dir(workspace_roots, "gb", layer=CLEANSED_LAYER_NAME)
        == tmp_path / "data" / "gb" / "cleansed" / "current"
    )


def test_system_layer_dir_is_the_directory_a_dated_layer_s_snapshots_sit_in(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert (
        system_layer_dir(workspace_roots, "gb", layer=CANONICAL_LAYER_NAME)
        == tmp_path / "data" / "gb" / "canonical"
    )


def test_system_layer_dir_for_a_nested_scope(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert (
        system_layer_dir(
            workspace_roots, "perturbed", "noise-mid", layer=CLEANSED_LAYER_NAME
        )
        == tmp_path / "data" / "perturbed" / "noise-mid" / "cleansed" / "current"
    )


def test_canonical_snapshot_dir(tmp_path: Path, workspace_roots: WorkspaceRoots):
    assert (
        canonical_snapshot_dir(workspace_roots, system="gb", run_date="2026-06-15")
        == tmp_path / "data" / "gb" / "canonical" / "2026-06-15"
    )


def test_canonical_snapshot_dir_is_under_its_own_layer_dir(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert canonical_snapshot_dir(
        workspace_roots, system="gb", run_date="2026-06-15"
    ).parent == (system_layer_dir(workspace_roots, "gb", layer=CANONICAL_LAYER_NAME))


def test_latest_canonical_snapshot_dir_picks_the_latest_date(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    """ISO dates sort, so the last dated subdirectory is the current
    snapshot; a file beside them is not a candidate."""
    canonical_dir = system_layer_dir(workspace_roots, "gb", layer=CANONICAL_LAYER_NAME)
    for date in ("2026-03-01", "2026-01-01"):
        (canonical_dir / date).mkdir(parents=True)
    (canonical_dir / "_dedupe_report.json").write_text("{}", encoding="utf-8")

    assert latest_canonical_snapshot_dir(canonical_dir) == canonical_dir / "2026-03-01"


def test_latest_canonical_snapshot_dir_is_none_without_a_snapshot(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    canonical_dir = system_layer_dir(workspace_roots, "gb", layer=CANONICAL_LAYER_NAME)

    assert latest_canonical_snapshot_dir(canonical_dir) is None
    canonical_dir.mkdir(parents=True)
    assert latest_canonical_snapshot_dir(canonical_dir) is None


def test_a_snapshot_directory_reads_back_as_the_dataset_it_holds(
    workspace_roots: WorkspaceRoots,
):
    canonical = canonical_snapshot_dir(
        workspace_roots, system="gb", run_date="2026-06-01"
    )
    matched = system_layer_dir(workspace_roots, "gleif", layer=MATCHED_LAYER_NAME)

    assert dataset_reference_at(workspace_roots, canonical).uri == (
        "data://gb/canonical/2026-06-01"
    )
    assert dataset_reference_at(workspace_roots, matched).uri == "data://gleif/matched"


def test_a_perturbed_dataset_sits_where_its_reference_locates(
    workspace_roots: WorkspaceRoots,
):
    dataset = perturbed_dataset_reference(
        source_system="ie", profile_id="en-lite", version="v2", seed=42
    )

    assert dataset.uri == "perturbed://ie/en-lite/v2/42"
    assert perturbed_dataset_dir(
        workspace_roots, source_system="ie", profile_id="en-lite", version="v2", seed=42
    ) == locate(workspace_roots, dataset)


def test_layer_directory_gives_every_tree_the_same_shape(tmp_path: Path):
    dataset = tmp_path / "artifacts" / "perturbed" / "data" / "ie" / "p1" / "1"

    assert (
        layer_directory(dataset, CLEANSED_LAYER_NAME)
        == dataset / "cleansed" / "current"
    )
    assert layer_directory(dataset, CANONICAL_LAYER_NAME) == dataset / "canonical"


def test_layer_system_reads_the_system_back_from_a_layer_directory(
    workspace_roots: WorkspaceRoots,
):
    assert (
        layer_system(system_layer_dir(workspace_roots, "gb", layer=CLEANSED_LAYER_NAME))
        == "gb"
    )
    assert (
        layer_system(
            canonical_snapshot_dir(workspace_roots, system="ie", run_date="2026-06-21")
        )
        == "ie"
    )
    assert (
        layer_system(
            system_layer_dir(workspace_roots, "fr", layer=CANONICAL_LAYER_NAME)
        )
        == "fr"
    )
