from __future__ import annotations

import json
import sys

import pytest

from scripts import plot_country_global_overlay as overlay_script
from scripts.plot_country_global_overlay import plot_country_global_overlay
from tests.promoted_tokenizers import promoted_tokenizer_files
from workspace.published_reports import tokenizer_reports_dir
from workspace.roots import WorkspaceRoots

TRAINER = "wordpiece"


def _promote_swept_scope(
    *,
    roots: WorkspaceRoots,
    system: str | None,
    vocab_size_resolved: int,
    per_system: list[dict[str, object]],
) -> None:
    files = promoted_tokenizer_files(roots, system=system, trainer=TRAINER)
    files.metrics.write_text(json.dumps({"fertility_distance": 0.01}), encoding="utf-8")
    summary = {
        "winner_candidate": {"vocab_size_resolved": vocab_size_resolved},
        "winner_output": {"per_system": per_system},
        "candidate_stats": {},
        "job_totals": {},
        "wall_seconds": 1.0,
    }
    files.optimize_summary.write_text(json.dumps(summary), encoding="utf-8")


@pytest.mark.graphics
def test_plot_country_global_overlay_writes_png_for_swept_scopes(
    workspace_roots: WorkspaceRoots,
):
    _promote_swept_scope(
        roots=workspace_roots,
        system="ie",
        vocab_size_resolved=25000,
        per_system=[{"system": "ie", "rows": 100, "unk_rate": 0.01, "fertility": 1.2}],
    )
    _promote_swept_scope(
        roots=workspace_roots,
        system=None,
        vocab_size_resolved=40000,
        per_system=[
            {"system": "ie", "rows": 100, "unk_rate": 0.02, "fertility": 1.3},
            {"system": "gb", "rows": 200, "unk_rate": 0.03, "fertility": 1.4},
        ],
    )

    output_path = plot_country_global_overlay(
        roots=workspace_roots, trainer=TRAINER, systems=["ie"]
    )

    assert output_path is not None
    assert output_path.exists()
    assert output_path.name == f"scope_comparison_overlay.{TRAINER}.png"


def test_plot_country_global_overlay_returns_none_when_nothing_swept(
    workspace_roots: WorkspaceRoots,
):
    output_path = plot_country_global_overlay(
        roots=workspace_roots, trainer=TRAINER, systems=["ie"]
    )
    assert output_path is None


@pytest.mark.parametrize("off_switch", [False, True])
def test_main_publishes_the_overlay_unless_switched_off(
    tmp_path, workspace_roots: WorkspaceRoots, monkeypatch, off_switch: bool
):
    rendered = tmp_path / f"scope_comparison_overlay.{TRAINER}.png"
    rendered.write_bytes(b"png bytes")
    monkeypatch.setattr(
        overlay_script, "plot_country_global_overlay", lambda **kwargs: rendered
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot_country_global_overlay.py", "--root", str(tmp_path)]
        + (["--no-publish"] if off_switch else []),
    )

    assert overlay_script.main() == 0

    published = tokenizer_reports_dir(workspace_roots) / rendered.name
    if off_switch:
        assert not published.exists()
    else:
        assert published.read_bytes() == b"png bytes"
