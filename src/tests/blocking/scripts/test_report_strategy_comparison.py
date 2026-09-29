from __future__ import annotations

import pytest

from blocking.run_layout import ComparisonArtefact, resolve_comparison_location
from scripts import report_strategy_comparison
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


def test_dry_run_names_the_pairing_and_the_report_it_would_write(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = report_strategy_comparison.main(
        [
            *_roots_argv(workspace_roots),
            "--source",
            "gleif",
            "--target",
            "gb",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "pairings=[('gleif', 'gb')]" in out
    location = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="gb"
    )
    assert str(location.path(ComparisonArtefact.REPORT)) in out
    assert str(location.path(ComparisonArtefact.REPORT_CSV)) in out
    assert str(location.path(ComparisonArtefact.PAIR_OUTCOME_RUNS)) in out
    assert str(location.path(ComparisonArtefact.PAIR_OUTCOME_RUNS_CSV)) in out
    assert str(location.path(ComparisonArtefact.PAIR_OUTCOMES)) in out
    assert str(location.path(ComparisonArtefact.PAIR_OUTCOME_CANDIDATES)) in out
    assert not location.directory.exists()


def test_a_pairing_with_no_finished_run_is_reported_as_an_error(
    workspace_roots: WorkspaceRoots, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = report_strategy_comparison.main(
        [*_roots_argv(workspace_roots), "--source", "gleif", "--target", "gb"]
    )

    assert exit_code == 1
    assert "no finished run" in capsys.readouterr().err


def test_a_source_given_without_a_target_is_refused(
    workspace_roots: WorkspaceRoots,
) -> None:
    with pytest.raises(SystemExit):
        report_strategy_comparison.main(
            [*_roots_argv(workspace_roots), "--source", "gleif"]
        )
