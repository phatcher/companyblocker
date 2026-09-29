from __future__ import annotations

from pathlib import Path

import pytest

from blocking.run_layout import ComparisonArtefact, resolve_comparison_location
from scripts import plot_pair_outcomes, publish_reports
from workspace.artifact_layout import tokenizer_artifact_root
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


def _touch(path: Path, content: bytes = b"png") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_figures_are_copied_under_the_country_with_their_report_names(
    workspace_roots: WorkspaceRoots, tmp_path: Path
) -> None:
    location = resolve_comparison_location(
        workspace_roots, source_system="gleif", target_system="ie"
    )
    _touch(location.path(ComparisonArtefact.REPORT))
    _touch(location.path(ComparisonArtefact.PAIR_OUTCOMES))
    assert location.pairing is not None
    _touch(
        plot_pair_outcomes.output_path(
            workspace_roots,
            "2026-09-01",
            location.pairing,
            "never",
            plot_pair_outcomes.FOUND_BY_UPSET,
        ),
        b"older",
    )
    _touch(
        plot_pair_outcomes.output_path(
            workspace_roots,
            "2026-09-29",
            location.pairing,
            "never",
            plot_pair_outcomes.FOUND_BY_UPSET,
        ),
        b"upset",
    )
    optimize = (
        tokenizer_artifact_root(workspace_roots)
        / "ie"
        / "sentencepiece_bpe"
        / "optimize"
    )
    _touch(
        optimize / "344aa760" / "8a65013c" / "elbow_curve.sentencepiece.png", b"elbow"
    )
    reports = tmp_path / "reports"

    exit_code = publish_reports.main(
        [*_roots_argv(workspace_roots), "--reports-dir", str(reports)]
    )

    assert exit_code == 0
    assert (
        reports / "ie" / "gleif_residual_found_by_upset.png"
    ).read_bytes() == b"upset"
    assert (
        reports / "ie" / "tokenizer_elbow_sentencepiece_bpe.png"
    ).read_bytes() == b"elbow"
    assert not (reports / "ie" / "gleif_residual_run_overlap.png").exists()


def test_a_figure_not_yet_drawn_is_named_with_the_command_that_draws_it(
    workspace_roots: WorkspaceRoots, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    optimize = (
        tokenizer_artifact_root(workspace_roots) / "gb" / "wordpiece" / "optimize"
    )
    optimize.mkdir(parents=True)

    exit_code = publish_reports.main(
        [*_roots_argv(workspace_roots), "--reports-dir", str(tmp_path / "reports")]
    )

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "gb/tokenizer_elbow_wordpiece.png" in out
    assert "plot_optimize_elbow.py --systems gb --tokenizer wordpiece" in out
