from __future__ import annotations

import json

from company_tokenize import tokenizer_directory_files

from training.scope_comparison import (
    BENCHMARK_COLUMNS,
    NEVER_SWEPT_STATUS,
    NOTHING_PROMOTED_STATUS,
    OK_STATUS,
    QUALITY_SUMMARY_COLUMNS,
    build_scope_comparison,
    render_scope_comparison_markdown,
    scope_comparison_as_dict,
)
from workspace.kind_layout import Kind
from workspace.pointer import promote
from workspace.reference import Reference, Side, locate, reference
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import tokenizer_selection

TRAINER = "wordpiece"


def _store(
    roots: WorkspaceRoots,
    *,
    system: str | None,
    key: str,
    metrics: dict[str, float],
    optimize_summary: dict[str, object] | None = None,
) -> Reference:
    """Write a candidate directory: its `metrics`, and an optimize summary if given."""
    selection = tokenizer_selection(system=system, tokenizer_id="wordpiece")
    candidate = reference(Kind.TOKENIZER, Side.DATA, **selection.fields, key=key)
    files = tokenizer_directory_files(locate(roots, candidate), trainer=TRAINER)
    files.directory.mkdir(parents=True)
    files.metrics.write_text(json.dumps(metrics), encoding="utf-8")
    if optimize_summary is not None:
        files.optimize_summary.write_text(
            json.dumps(optimize_summary), encoding="utf-8"
        )
    return candidate


def _promote(roots: WorkspaceRoots, candidate: Reference, *, system: str | None):
    promote(
        roots,
        tokenizer_selection(system=system, tokenizer_id="wordpiece"),
        candidate,
    )


def _optimize_summary(
    *,
    vocab_size_resolved: int,
    min_frequency: int,
    per_system: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "winner_candidate": {
            "vocab_size_requested": vocab_size_resolved,
            "vocab_size_resolved": vocab_size_resolved,
            "min_frequency": min_frequency,
        },
        "candidate_stats": {
            "total_candidates": 20,
            "accepted_candidates": 18,
            "rejected_candidates": 2,
            "elapsed_seconds": {"median": 1.5, "p95": 3.2},
        },
        "job_totals": {"completed_jobs": 20, "submitted_jobs": 20},
        "wall_seconds": 123.4,
        "winner_output": {"per_system": per_system},
    }


def test_build_scope_comparison_full_data_for_every_scope(
    workspace_roots: WorkspaceRoots,
):
    """A scope that was optimize-swept and promoted appears fully in both
    tables plus the overlay, at its own winning vocab size, read from the
    directory of the candidate the pointer names."""
    metrics = {
        "rows": 100.0,
        "unk_rate": 0.0,
        "single_char_token_pct": 5.0,
        "fertility": 1.2,
        "fertility_distance": 0.01,
        "token_count_mean": 3.0,
        "token_count_median": 3.0,
        "token_count_p95": 5.0,
    }
    candidate = _store(
        workspace_roots,
        system="ie",
        key="swept",
        metrics=metrics,
        optimize_summary=_optimize_summary(
            vocab_size_resolved=25000,
            min_frequency=8,
            per_system=[
                {"system": "ie", "rows": 100, "unk_rate": 0.0, "fertility": 1.2}
            ],
        ),
    )
    _promote(workspace_roots, candidate, system="ie")

    build = build_scope_comparison(
        roots=workspace_roots, trainer=TRAINER, systems=["ie"]
    )

    assert len(build.quality_rows) == 2  # "ie" + "global"
    assert len(build.benchmark_rows) == 2
    ie_quality = next(row for row in build.quality_rows if row.scope == "ie")
    assert ie_quality.status == OK_STATUS
    assert ie_quality.candidate == "swept"
    assert ie_quality.as_record()["fertility_distance"] == 0.01

    ie_benchmark = next(row for row in build.benchmark_rows if row.scope == "ie")
    assert ie_benchmark.status == OK_STATUS
    record = ie_benchmark.as_record()
    assert record["vocab_size_resolved"] == 25000
    assert record["min_frequency"] == 8
    assert record["wall_seconds"] == 123.4

    assert len(build.overlay_points) == 1
    assert build.overlay_points[0].scope == "ie"
    assert build.overlay_points[0].vocab_size_resolved == 25000


def test_a_promoted_candidate_without_a_optimize_summary_reports_never_swept(
    workspace_roots: WorkspaceRoots,
):
    """A promoted candidate with metrics but no optimize summary: the quality row
    is ok, the benchmark row is flagged rather than fabricated, and it
    contributes no overlay point."""
    candidate = _store(
        workspace_roots,
        system=None,
        key="unswept",
        metrics={"rows": 50.0, "unk_rate": 0.01, "fertility": 1.3},
    )
    _promote(workspace_roots, candidate, system=None)

    build = build_scope_comparison(
        roots=workspace_roots, trainer=TRAINER, systems=["ie"]
    )

    global_quality = next(row for row in build.quality_rows if row.scope == "global")
    assert global_quality.status == OK_STATUS

    global_benchmark = next(
        row for row in build.benchmark_rows if row.scope == "global"
    )
    assert global_benchmark.status == NEVER_SWEPT_STATUS
    assert global_benchmark.as_record()["wall_seconds"] is None

    ie_quality = next(row for row in build.quality_rows if row.scope == "ie")
    assert ie_quality.status == NOTHING_PROMOTED_STATUS

    assert build.overlay_points == []


def test_build_scope_comparison_nothing_promoted_reports_a_row_per_scope(
    workspace_roots: WorkspaceRoots,
):
    """A scope the pointer names nothing for still gets a row in both tables,
    flagged rather than dropped -- the fixed-shape contract."""
    build = build_scope_comparison(
        roots=workspace_roots, trainer=TRAINER, systems=["ie", "gb"]
    )

    assert {row.scope for row in build.quality_rows} == {"ie", "gb", "global"}
    assert all(row.status == NOTHING_PROMOTED_STATUS for row in build.quality_rows)
    assert all(row.status == NOTHING_PROMOTED_STATUS for row in build.benchmark_rows)
    assert build.overlay_points == []


def test_only_the_candidate_the_pointer_names_is_reported(
    workspace_roots: WorkspaceRoots,
):
    """Another stored candidate, a naive one or one an optimize run trained,
    is not reported because it is there: the pointer decides."""
    chosen = _store(
        workspace_roots, system="ie", key="chosen", metrics={"fertility_distance": 0.01}
    )
    _store(
        workspace_roots, system="ie", key="other", metrics={"fertility_distance": 0.03}
    )
    _promote(workspace_roots, chosen, system="ie")

    build = build_scope_comparison(
        roots=workspace_roots, trainer=TRAINER, systems=["ie"]
    )

    ie_quality = next(row for row in build.quality_rows if row.scope == "ie")
    assert ie_quality.candidate == "chosen"
    assert ie_quality.as_record()["fertility_distance"] == 0.01


def test_render_scope_comparison_markdown_shape_is_stable(
    workspace_roots: WorkspaceRoots,
):
    """Regression check that the comparison output's column shape holds
    regardless of which scopes have real data."""
    build = build_scope_comparison(
        roots=workspace_roots, trainer=TRAINER, systems=["ie"]
    )
    markdown = render_scope_comparison_markdown(build)

    assert "| " + " | ".join(QUALITY_SUMMARY_COLUMNS) + " |" in markdown
    assert "| " + " | ".join(BENCHMARK_COLUMNS) + " |" in markdown
    assert NOTHING_PROMOTED_STATUS in markdown

    payload = scope_comparison_as_dict(build)
    assert set(payload.keys()) == {"trainer", "quality_summary", "benchmark", "overlay"}
    assert {row["scope"] for row in payload["quality_summary"]} == {"ie", "global"}
    for row in payload["quality_summary"]:
        assert set(row.keys()) == set(QUALITY_SUMMARY_COLUMNS)
    for row in payload["benchmark"]:
        assert set(row.keys()) == set(BENCHMARK_COLUMNS)
