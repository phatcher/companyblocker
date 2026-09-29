from __future__ import annotations

import pytest

from blocking.run_layout import resolve_comparison_location
from scripts import plot_pair_outcomes
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def _roots_argv(roots: WorkspaceRoots) -> list[str]:
    return [
        "--data-dir",
        str(roots.data),
        "--output-dir",
        str(roots.artifacts),
        "--temp-dir",
        str(roots.temp),
    ]


def test_dry_run_names_the_pairing_and_the_figures_it_would_write(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = plot_pair_outcomes.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "ie",
            "--level",
            "basic",
            "--date",
            "2026-09-29",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pairings=[('gleif', 'ie')]" in out
    location = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="ie"
    )
    run_dir = analysis_report_run_dir(workspace_roots, "pair_outcomes", "2026-09-29")
    for directory, name in (
        ("viz", plot_pair_outcomes.FOUND_BY_UPSET),
        ("viz", plot_pair_outcomes.RUN_OVERLAP),
        ("viz", plot_pair_outcomes.NAME_SIMILARITY),
        ("viz", plot_pair_outcomes.CLUSTER_SIZES),
        ("metrics", plot_pair_outcomes.LEVEL_METRICS_TABLE),
    ):
        assert str(run_dir / directory / f"gleif__ie_basic_{name}") in out
    assert not location.directory.exists()
    assert not run_dir.exists()


def test_a_pairing_with_no_per_pair_outcomes_is_reported_as_an_error(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = plot_pair_outcomes.main(
        [*_roots_argv(workspace_roots), "--source", "gleif", "--target", "ie"]
    )

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "no per-pair outcomes" in err


def test_a_source_given_without_a_target_is_refused(
    workspace_roots: WorkspaceRoots,
) -> None:
    with pytest.raises(SystemExit):
        plot_pair_outcomes.main([*_roots_argv(workspace_roots), "--source", "gleif"])
