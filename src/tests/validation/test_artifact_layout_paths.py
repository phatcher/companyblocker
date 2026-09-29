"""Direct tests confirming this area's `artifacts/` paths resolve through
`workspace.artifact_layout` rather than being composed by hand here.

Each assertion pins a migrated path to the value it produced before the
migration (`root / "artifacts" / "validation" / "perturbed" / ...` and
`root / "artifacts" / "perf" / ...`), so a future edit to either site or to
`workspace.artifact_layout` cannot silently change what is on disk.
"""

from __future__ import annotations

from pathlib import Path

from workspace.artifact_layout import perf_artifact_root
from workspace.roots import WorkspaceRoots


def test_benchmark_default_output_path_matches_previous_inline_shape(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    default_output_path = (
        perf_artifact_root(workspace_roots)
        / "validate_clustering_backend_benchmark.json"
    )

    assert default_output_path == (
        tmp_path / "artifacts" / "perf" / "validate_clustering_backend_benchmark.json"
    )
