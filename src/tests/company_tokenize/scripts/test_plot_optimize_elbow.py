from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from company_tokenize import (
    compute_corpus_content_hash,
    resolve_optimize_canary_pointer_path,
    resolve_optimize_sweep_paths,
    resolve_tokenizer_paths,
)
from company_tokenize import resolve_optimize_dir as resolve_tokenizer_optimize_dir

from scripts import plot_optimize_elbow as plot_module
from scripts.plot_optimize_elbow import (
    NAIVE_BASELINE_VOCAB,
    plot_optimize_elbow,
    resolve_optimize_dir,
)
from workspace.artifact_layout import tokenizer_artifact_root
from workspace.roots import WorkspaceRoots, default_workspace_roots


def _write_active_sweep_dir(
    *, root: Path, system: str, trainer: str = "wordpiece"
) -> Path:
    """Simulate a real completed sweep: a real corpus file (so
    `compute_corpus_content_hash` has something to hash), a pointer file
    recording a `grid_hash`, and returns that sweep's leaf directory for the
    test to populate with a run log/summary.
    """
    tokenizer_paths = resolve_tokenizer_paths(
        tokenizer_root=tokenizer_artifact_root(default_workspace_roots(root)),
        scope="country",
        system=system,
        profile="default",
    )
    tokenizer_paths.corpus_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name": ["alpha ltd"]}).write_parquet(tokenizer_paths.corpus_path)
    optimize_dir = resolve_tokenizer_optimize_dir(
        scope_directory=tokenizer_paths.directory,
        trainer=trainer,
        tokenizer_encoding=None,
    )
    pointer_path = resolve_optimize_canary_pointer_path(optimize_dir=optimize_dir)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps({"grid_hash": "testgrid"}), encoding="utf-8")
    sweep_paths = resolve_optimize_sweep_paths(
        optimize_dir=optimize_dir,
        corpus_content_hash=compute_corpus_content_hash(tokenizer_paths.corpus_path),
        grid_hash="testgrid",
    )
    return sweep_paths.sweep_dir


def _write_run_log(optimize_dir: Path) -> None:
    rows = []
    for min_freq in (1, 2):
        for seed in (1, 2):
            for vocab in (8000, 10000, 12000, 14000):
                # Fertility distance improves quickly then flattens -- an elbow
                # shape, with the 12000/14000 steps representing post-elbow
                # confirmation steps that shouldn't change the winner.
                distance = {8000: 0.08, 10000: 0.02, 12000: 0.019, 14000: 0.0185}[vocab]
                rows.append(
                    {
                        "min_frequency": min_freq,
                        "seed": seed,
                        "vocab_size_requested": vocab,
                        "vocab_size_resolved": vocab,
                        "fertility_distance": distance,
                        "rejected": vocab == 8000,
                    }
                )
    optimize_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(optimize_dir / "optimize_runs.parquet")


def _write_summary(optimize_dir: Path) -> None:
    summary = {
        "elbow_min_points": 3,
        "elbow_min_improvement": 0.002,
        "effective_elbow_extra_steps": 2,
        "fertility_tolerance": 0.05,
        "winner_candidate": {
            "vocab_size_resolved": 10000,
            "min_frequency": 1,
            "median_fertility_distance": 0.02,
            "median_selection_score": 0.9,
        },
    }
    (optimize_dir / "optimize_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def _write_run_log_with_naive_point(optimize_dir: Path) -> None:
    """Same shape as `_write_run_log`, but with the naive operating point
    (`vocab_size=40000`, `min_frequency=1`) actually on the grid.
    """
    rows = []
    for min_freq in (1, 2):
        for seed in (1, 2):
            for vocab in (10000, NAIVE_BASELINE_VOCAB):
                rows.append(
                    {
                        "min_frequency": min_freq,
                        "seed": seed,
                        "vocab_size_requested": vocab,
                        "vocab_size_resolved": vocab,
                        "fertility_distance": 0.02 if vocab == 10000 else 0.05,
                        "rejected": False,
                    }
                )
    optimize_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(optimize_dir / "optimize_runs.parquet")


def _capture_axes(monkeypatch) -> dict:
    captured: dict = {}
    real_subplots = plot_module.plt.subplots

    def fake_subplots(*args, **kwargs):
        fig, ax = real_subplots(*args, **kwargs)
        captured["ax"] = ax
        return fig, ax

    monkeypatch.setattr(plot_module.plt, "subplots", fake_subplots)
    return captured


def _legend_labels(captured: dict) -> list[str]:
    return [text.get_text() for text in captured["ax"].get_legend().get_texts()]


@pytest.mark.integration
@pytest.mark.graphics
def test_plot_optimize_elbow_marks_the_naive_baseline_when_it_is_on_the_grid(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    sweep_dir = _write_active_sweep_dir(root=tmp_path, system="gb")
    _write_run_log_with_naive_point(sweep_dir)
    _write_summary(sweep_dir)
    captured = _capture_axes(monkeypatch)

    plot_optimize_elbow(
        roots=workspace_roots,
        scope="country",
        system="gb",
        profile="default",
        trainer="wordpiece",
    )

    assert any("naive baseline" in label for label in _legend_labels(captured))


@pytest.mark.integration
@pytest.mark.graphics
def test_plot_optimize_elbow_omits_the_naive_marker_when_the_grid_lacks_it(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    """The marker is read from the sweep's own run log, so a grid that never
    trained the naive point gets no marker rather than a value borrowed from
    a differently-measured source.
    """
    sweep_dir = _write_active_sweep_dir(root=tmp_path, system="gb")
    _write_run_log(sweep_dir)
    _write_summary(sweep_dir)
    captured = _capture_axes(monkeypatch)

    plot_optimize_elbow(
        roots=workspace_roots,
        scope="country",
        system="gb",
        profile="default",
        trainer="wordpiece",
    )

    assert not any("naive baseline" in label for label in _legend_labels(captured))


def _write_run_log_with_token_count_p95(optimize_dir: Path) -> None:
    """Same shape as `_write_run_log`, plus a `token_count_p95` column
    whose step lands mid-grid -- the "cliff" a twin-axis plot should surface.
    """
    rows = []
    for min_freq in (1,):
        for seed in (1, 2):
            for vocab in (8000, 10000, 12000, 14000):
                distance = {8000: 0.08, 10000: 0.02, 12000: 0.019, 14000: 0.0185}[vocab]
                p95 = 7 if vocab <= 10000 else 6
                rows.append(
                    {
                        "min_frequency": min_freq,
                        "seed": seed,
                        "vocab_size_requested": vocab,
                        "vocab_size_resolved": vocab,
                        "fertility_distance": distance,
                        "token_count_p95": p95,
                        "rejected": vocab == 8000,
                    }
                )
    optimize_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(optimize_dir / "optimize_runs.parquet")


@pytest.mark.integration
@pytest.mark.graphics
def test_plot_optimize_elbow_adds_a_token_count_p95_twin_axis_when_present(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    sweep_dir = _write_active_sweep_dir(root=tmp_path, system="gb")
    _write_run_log_with_token_count_p95(sweep_dir)
    _write_summary(sweep_dir)
    captured = _capture_axes(monkeypatch)

    plot_optimize_elbow(
        roots=workspace_roots,
        scope="country",
        system="gb",
        profile="default",
        trainer="wordpiece",
    )

    assert any("token_count_p95" in label for label in _legend_labels(captured))
    assert len(captured["ax"].figure.axes) == 2


@pytest.mark.integration
@pytest.mark.graphics
def test_plot_optimize_elbow_omits_the_p95_axis_when_the_run_log_lacks_it(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    sweep_dir = _write_active_sweep_dir(root=tmp_path, system="gb")
    _write_run_log(sweep_dir)
    _write_summary(sweep_dir)
    captured = _capture_axes(monkeypatch)

    plot_optimize_elbow(
        roots=workspace_roots,
        scope="country",
        system="gb",
        profile="default",
        trainer="wordpiece",
    )

    assert not any("token_count_p95" in label for label in _legend_labels(captured))
    assert len(captured["ax"].figure.axes) == 1


def test_resolve_optimize_dir_matches_country_tokenizer_layout(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    optimize_dir = resolve_optimize_dir(
        roots=workspace_roots,
        scope="country",
        system="gb",
        profile="default",
        trainer="wordpiece",
    )
    assert optimize_dir == (
        tokenizer_artifact_root(workspace_roots) / "gb" / "wordpiece" / "optimize"
    )


@pytest.mark.integration
@pytest.mark.graphics
def test_plot_optimize_elbow_writes_png_from_run_log_and_summary(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    sweep_dir = _write_active_sweep_dir(root=tmp_path, system="gb")
    _write_run_log(sweep_dir)
    _write_summary(sweep_dir)

    output_path = plot_optimize_elbow(
        roots=workspace_roots,
        scope="country",
        system="gb",
        profile="default",
        trainer="wordpiece",
    )

    assert output_path == sweep_dir / "elbow_curve.wordpiece.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_optimize_elbow_handles_missing_run_log(workspace_roots: WorkspaceRoots):
    output_path = plot_optimize_elbow(
        roots=workspace_roots,
        scope="country",
        system="ie",
        profile="default",
        trainer="wordpiece",
    )

    assert output_path is None


@pytest.mark.integration
@pytest.mark.graphics
def test_plot_optimize_elbow_handles_missing_summary(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    sweep_dir = _write_active_sweep_dir(root=tmp_path, system="fr")
    _write_run_log(sweep_dir)
    # No summary.json written -- the chart should still render from the run log alone.

    output_path = plot_optimize_elbow(
        roots=workspace_roots,
        scope="country",
        system="fr",
        profile="default",
        trainer="wordpiece",
    )

    assert output_path is not None
    assert output_path.exists()
