"""Cross-scope tokenizer comparison: one standardised shape across every scope.

Pure presentation/consolidation over what each scope's promoted candidate
already holds, mirroring `tokenizer_corpus_report.py`'s house style -- no new
metric is computed here, only assembled and reported in one fixed shape
regardless of which scopes actually have data. Lives here (not in
`packages/company_tokenize`, which must stay dependency-free per
`tach.toml`) because it reads the pointer and the store, which are
`workspace`'s, the same reason `tokenizer_corpus_report.py` lives here.

Two tables, one row per scope (the country/system scopes this repo has swept
plus `global`), each scope resolved independently so a scope with no data at
all never drops a row:

- **Quality-summary table**: the metrics each scope's promoted candidate was
  stored with (`rows`, `unk_rate`, `single_char_token_pct`, `fertility`,
  `fertility_distance`, `token_count_mean/median/p95`).
- **Benchmark table**: the optimize-sweep evidence (candidate counts, wall
  time, elapsed-time percentiles, the winning vocab/min_frequency) from that
  candidate's optimize summary, which only an `--mode optimize --finalize` run
  stores; a candidate without one reports `status="never swept"` rather than
  fabricated numbers.

Which tokenizer is promoted is the pointer's to say: the reference
`config/tokenizers.json` names as `promoted` for the scope and tokenizer id.
The row reports what that candidate's directory holds, and names it by its
key. A scope the pointer names nothing for reports `nothing promoted`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from company_tokenize import tokenizer_directory_files, tokenizer_id

from workspace.pointer import NothingPromotedError
from workspace.reference import Reference, locate
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import promoted_tokenizer

# The scopes the WordPiece elbow-optimize sweep covered -- not `acquisition.pipeline_selection`'s
# `all_target_code_lists()`, which answers a different question (who
# onboarding should regenerate) and currently includes `wikidata`, which has
# no tokenizer artifacts at all.
DEFAULT_COUNTRY_SYSTEMS: tuple[str, ...] = ("fr", "gb", "gleif", "ie", "offeneregister")

NOTHING_PROMOTED_STATUS = "nothing promoted"
NEVER_SWEPT_STATUS = "never swept"
OK_STATUS = "ok"

QUALITY_SUMMARY_COLUMNS: tuple[str, ...] = (
    "scope",
    "trainer",
    "status",
    "candidate",
    "rows",
    "unk_rate",
    "single_char_token_pct",
    "fertility",
    "fertility_distance",
    "token_count_mean",
    "token_count_median",
    "token_count_p95",
)

BENCHMARK_COLUMNS: tuple[str, ...] = (
    "scope",
    "trainer",
    "status",
    "candidate",
    "vocab_size_requested",
    "vocab_size_resolved",
    "min_frequency",
    "total_candidates",
    "accepted_candidates",
    "rejected_candidates",
    "elapsed_seconds_median",
    "elapsed_seconds_p95",
    "completed_jobs",
    "submitted_jobs",
    "wall_seconds",
)


@dataclass(frozen=True)
class ScopeQualityRow:
    """One quality-summary table row -- always present, one per scope."""

    scope: str
    trainer: str
    status: str
    candidate: str | None = None
    metrics: dict[str, Any] | None = None

    def as_record(self) -> dict[str, Any]:
        metrics = self.metrics or {}
        return {
            "scope": self.scope,
            "trainer": self.trainer,
            "status": self.status,
            "candidate": self.candidate,
            "rows": metrics.get("rows"),
            "unk_rate": metrics.get("unk_rate"),
            "single_char_token_pct": metrics.get("single_char_token_pct"),
            "fertility": metrics.get("fertility"),
            "fertility_distance": metrics.get("fertility_distance"),
            "token_count_mean": metrics.get("token_count_mean"),
            "token_count_median": metrics.get("token_count_median"),
            "token_count_p95": metrics.get("token_count_p95"),
        }


@dataclass(frozen=True)
class ScopeBenchmarkRow:
    """One benchmark table row -- always present, one per scope."""

    scope: str
    trainer: str
    status: str
    candidate: str | None = None
    winner_candidate: dict[str, Any] | None = None
    candidate_stats: dict[str, Any] | None = None
    job_totals: dict[str, Any] | None = None
    wall_seconds: float | None = None

    def as_record(self) -> dict[str, Any]:
        winner = self.winner_candidate or {}
        stats = self.candidate_stats or {}
        elapsed = stats.get("elapsed_seconds") or {}
        job_totals = self.job_totals or {}
        return {
            "scope": self.scope,
            "trainer": self.trainer,
            "status": self.status,
            "candidate": self.candidate,
            "vocab_size_requested": winner.get("vocab_size_requested"),
            "vocab_size_resolved": winner.get("vocab_size_resolved"),
            "min_frequency": winner.get("min_frequency"),
            "total_candidates": stats.get("total_candidates"),
            "accepted_candidates": stats.get("accepted_candidates"),
            "rejected_candidates": stats.get("rejected_candidates"),
            "elapsed_seconds_median": elapsed.get("median"),
            "elapsed_seconds_p95": elapsed.get("p95"),
            "completed_jobs": job_totals.get("completed_jobs"),
            "submitted_jobs": job_totals.get("submitted_jobs"),
            "wall_seconds": self.wall_seconds,
        }


@dataclass(frozen=True)
class ScopeOverlayPoint:
    """One scope's per-country breakdown at its own winning vocab size.

    `per_system` is `winner_output.per_system` verbatim -- one entry for a
    country scope (itself), one entry per constituent country for `global`.
    Deliberately not normalised across scopes onto a shared vocab axis: two
    scopes can, and typically do, resolve to different vocab sizes, and that
    difference is exactly what the comparison exists to show.
    """

    scope: str
    vocab_size_resolved: int | None
    per_system: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class ScopeComparisonBuild:
    trainer: str
    quality_rows: list[ScopeQualityRow]
    benchmark_rows: list[ScopeBenchmarkRow]
    overlay_points: list[ScopeOverlayPoint]


def _iter_scope_specs(
    systems: list[str],
) -> list[tuple[str, str, str | None]]:
    """`(scope_label, scope, system)` for every row to build.

    `global` is always last and always included -- it is what every other
    scope in this table is being compared against.
    """
    specs: list[tuple[str, str, str | None]] = [
        (code, "country", code) for code in systems
    ]
    specs.append(("global", "global", None))
    return specs


def _promoted_reference(
    roots: WorkspaceRoots, *, system: str | None, tokenizer_id: str
) -> Reference | None:
    """The reference the pointer names as `promoted`, or None when it names none."""
    try:
        return promoted_tokenizer(roots, system=system, tokenizer_id=tokenizer_id)
    except NothingPromotedError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_scope_comparison(
    *,
    roots: WorkspaceRoots,
    trainer: str,
    tokenizer_encoding: str | None = None,
    systems: list[str] | None = None,
) -> ScopeComparisonBuild:
    """Build both comparison tables plus the per-country overlay data.

    One row per scope in every output, regardless of what real data exists
    for it -- the fixed shape this exists to guarantee. `systems` defaults to
    `DEFAULT_COUNTRY_SYSTEMS`; `global` is always appended.
    """
    resolved_systems = (
        list(systems) if systems is not None else list(DEFAULT_COUNTRY_SYSTEMS)
    )
    resolved_tokenizer_id = tokenizer_id(trainer, tokenizer_encoding)

    quality_rows: list[ScopeQualityRow] = []
    benchmark_rows: list[ScopeBenchmarkRow] = []
    overlay_points: list[ScopeOverlayPoint] = []

    for scope_label, _scope, system in _iter_scope_specs(resolved_systems):
        promoted = _promoted_reference(
            roots, system=system, tokenizer_id=resolved_tokenizer_id
        )
        if promoted is None:
            quality_rows.append(
                ScopeQualityRow(
                    scope=scope_label, trainer=trainer, status=NOTHING_PROMOTED_STATUS
                )
            )
            benchmark_rows.append(
                ScopeBenchmarkRow(
                    scope=scope_label, trainer=trainer, status=NOTHING_PROMOTED_STATUS
                )
            )
            continue

        candidate = str(promoted.fields["key"])
        files = tokenizer_directory_files(locate(roots, promoted), trainer=trainer)
        quality_rows.append(
            ScopeQualityRow(
                scope=scope_label,
                trainer=trainer,
                status=OK_STATUS,
                candidate=candidate,
                metrics=_read_json(files.metrics),
            )
        )

        if not files.optimize_summary.exists():
            benchmark_rows.append(
                ScopeBenchmarkRow(
                    scope=scope_label,
                    trainer=trainer,
                    status=NEVER_SWEPT_STATUS,
                    candidate=candidate,
                )
            )
            continue
        summary = _read_json(files.optimize_summary)

        benchmark_rows.append(
            ScopeBenchmarkRow(
                scope=scope_label,
                trainer=trainer,
                status=OK_STATUS,
                candidate=candidate,
                winner_candidate=summary.get("winner_candidate") or {},
                candidate_stats=summary.get("candidate_stats") or {},
                job_totals=summary.get("job_totals") or {},
                wall_seconds=summary.get("wall_seconds"),
            )
        )
        winner_output = summary.get("winner_output") or {}
        winner_candidate = summary.get("winner_candidate") or {}
        overlay_points.append(
            ScopeOverlayPoint(
                scope=scope_label,
                vocab_size_resolved=winner_candidate.get("vocab_size_resolved"),
                per_system=list(winner_output.get("per_system") or []),
            )
        )

    return ScopeComparisonBuild(
        trainer=trainer,
        quality_rows=quality_rows,
        benchmark_rows=benchmark_rows,
        overlay_points=overlay_points,
    )


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _render_table(columns: tuple[str, ...], records: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in records:
        lines.append(
            "| " + " | ".join(_fmt_cell(record.get(col)) for col in columns) + " |"
        )
    return "\n".join(lines)


def render_scope_comparison_markdown(build: ScopeComparisonBuild) -> str:
    """Render the one standardised comparison document every scope is reported in."""
    lines = [
        f"# Tokenizer Scope Comparison -- trainer={build.trainer}",
        "",
        "Pure consolidation over what each scope's promoted candidate was",
        "stored with -- nothing here is computed fresh. A scope with no",
        f"promoted tokenizer reports `{NOTHING_PROMOTED_STATUS}`, and a",
        "candidate with no optimize summary reports",
        f"`{NEVER_SWEPT_STATUS}` in the benchmark table -- none is",
        "dropped or filled with fabricated numbers.",
        "",
        "## Quality Summary",
        "",
        _render_table(
            QUALITY_SUMMARY_COLUMNS,
            [row.as_record() for row in build.quality_rows],
        ),
        "",
        "## Benchmark",
        "",
        _render_table(
            BENCHMARK_COLUMNS,
            [row.as_record() for row in build.benchmark_rows],
        ),
        "",
    ]
    return "\n".join(lines) + "\n"


def scope_comparison_as_dict(build: ScopeComparisonBuild) -> dict[str, Any]:
    """The same data as `render_scope_comparison_markdown`, as plain JSON."""
    return {
        "trainer": build.trainer,
        "quality_summary": [row.as_record() for row in build.quality_rows],
        "benchmark": [row.as_record() for row in build.benchmark_rows],
        "overlay": [
            {
                "scope": point.scope,
                "vocab_size_resolved": point.vocab_size_resolved,
                "per_system": point.per_system,
            }
            for point in build.overlay_points
        ],
    }
