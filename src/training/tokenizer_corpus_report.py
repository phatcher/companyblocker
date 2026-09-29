"""Consolidated tokenizer/corpus report for a scope's promoted candidate.

Pure presentation/consolidation -- reads what the store holds for each
candidate, the optimize run's elbow picture and the corpus Zipf outputs already
produced elsewhere, and assembles one markdown document that embeds them,
rather than computing any new metric itself. Lives here (not in
`packages/company_tokenize`, which must stay dependency-free per `tach.toml`,
and not in `src/analysis`, which does not depend on `src/training`) because
this module is the only one in the dependency graph allowed to import all of
what it consolidates: `company_tokenize`'s file names, `workspace`'s store and
pointer, and `analysis.token_zipf`'s Zipf outputs.

Grain: one report per scope and tokenizer id, for the candidate the pointer
names as `promoted`. Its own tokenizer's section comes first, and every other
tokenizer with stored candidates for the scope follows, so the candidate is
read beside what it was chosen over.

The report is published to `docs/reports/tokenizers/<scope>/` under fixed
names, `report.md` and its pictures, replacing what was there, so git history
keeps the earlier versions. A candidate that is not promoted has no report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from company_tokenize import (
    TOKENIZER_IDS,
    compute_corpus_content_hash,
    resolve_active_optimize_sweep_dir,
    scope_directory_files,
    tokenizer_directory_files,
    tokenizer_id,
)

from analysis.token_zipf import find_latest_zipf_summary_for_system
from training.promotion import NAIVE_PROFILE
from workspace.artifact_layout import (
    GLOBAL_TOKENIZER_SCOPE,
    tokenizer_artifact_root,
    tokenizer_scope_dir,
)
from workspace.pointer import PROMOTED_PROFILE, NothingPromotedError
from workspace.published_reports import publish, tokenizer_reports_dir
from workspace.records import read_record
from workspace.reference import Reference, locate, select_references
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import promoted_tokenizer, tokenizer_selection

PUBLISHED_REPORT_FILENAME = "report.md"


@dataclass(frozen=True)
class TokenizerCorpusReportBuild:
    markdown: str
    image_files: dict[str, Path]
    corpus_content_hash: str
    tokenizers_included: list[str] = field(default_factory=list)
    zipf_run_date: str | None = None


@dataclass(frozen=True)
class PublishedTokenizerCorpusReport:
    """The report built for a promoted candidate, and where it was published, if it was."""

    candidate: Reference
    build: TokenizerCorpusReportBuild
    published_dir: Path | None


def _fmt_num(value: object, *, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (int, float)) else "n/a"


def _fmt_pct(value: object) -> str:
    return f"{float(value):.1%}" if isinstance(value, (int, float)) else "n/a"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _scope_system(candidate: Reference) -> str | None:
    scope = str(candidate.fields["scope"])
    return None if scope == GLOBAL_TOKENIZER_SCOPE else scope


def _live_elbow_image_path(
    *, roots: WorkspaceRoots, system: str | None, identifier: str
) -> Path | None:
    """The elbow picture currently on disk in the folder of the latest
    optimize run for one tokenizer, if there is one.

    That folder is gitignored scratch, so a caller that wants a fresh picture
    renders one first with `scripts/plot_optimize_elbow.py`, which this module
    cannot import: `tach.toml` grants `src/training` no edge to `scripts`.
    """
    fields = TOKENIZER_IDS[identifier]
    run_dir = resolve_active_optimize_sweep_dir(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope="global" if system is None else "country",
        system=system,
        profile="default",
        trainer=fields.tokenizer,
        tokenizer_encoding=fields.tokenizer_encoding,
    )
    if run_dir is None:
        return None
    elbow_path = run_dir / f"elbow_curve.{fields.tokenizer}.png"
    return elbow_path if elbow_path.exists() else None


def _named_by(
    roots: WorkspaceRoots, *, system: str | None, identifier: str
) -> dict[str, str]:
    """Which profile names which stored candidate, by the candidate's uri."""
    named: dict[str, str] = {}
    for profile in (PROMOTED_PROFILE, NAIVE_PROFILE):
        try:
            pointed = promoted_tokenizer(
                roots, system=system, tokenizer_id=identifier, profile=profile
            )
        except NothingPromotedError:
            continue
        named[pointed.uri] = (
            f"{named[pointed.uri]}, {profile}" if pointed.uri in named else profile
        )
    return named


def _candidate_rows(
    roots: WorkspaceRoots, *, system: str | None, identifier: str
) -> list[str]:
    """One table row per stored candidate of a scope and tokenizer id, oldest first."""
    fields = TOKENIZER_IDS[identifier]
    named = _named_by(roots, system=system, identifier=identifier)
    rows: list[tuple[str, str]] = []
    for stored in select_references(
        roots, tokenizer_selection(system=system, tokenizer_id=identifier)
    ):
        record = read_record(roots, stored)
        parameters = record.parameters if record is not None else {}
        files = tokenizer_directory_files(
            locate(roots, stored), trainer=fields.tokenizer
        )
        metrics = _read_json(files.metrics) or {}
        stored_at = record.finished_at if record is not None else "n/a"
        rows.append(
            (
                stored_at,
                "| {key} | {named} | {mode} | {vocab} | {mf} | {fert} | {fd} | {unk} | {sc} | {ts} |".format(
                    key=stored.fields["key"],
                    named=named.get(stored.uri, ""),
                    mode=parameters.get("mode", "n/a"),
                    vocab=parameters.get("vocab_size", "n/a"),
                    mf=parameters.get("min_frequency", "n/a"),
                    fert=_fmt_num(metrics.get("fertility")),
                    fd=_fmt_num(metrics.get("fertility_distance"), digits=6),
                    unk=_fmt_num(metrics.get("unk_rate"), digits=6),
                    sc=_fmt_num(metrics.get("single_char_token_pct")),
                    ts=stored_at,
                ),
            )
        )
    return [row for _, row in sorted(rows)]


def build_tokenizer_corpus_report(
    *, roots: WorkspaceRoots, candidate: Reference
) -> TokenizerCorpusReportBuild:
    """Build the report's markdown text and the image files it references for
    `candidate`, a stored tokenizer's reference. Pure builder -- reads what is
    on disk and returns data; `publish_tokenizer_corpus_report` persists it.
    """
    system = _scope_system(candidate)
    own_identifier = str(candidate.fields["tokenizer_id"])
    corpus_path = scope_directory_files(
        tokenizer_scope_dir(roots, system=system)
    ).corpus
    corpus_content_hash = (
        compute_corpus_content_hash(corpus_path) if corpus_path.exists() else "unknown"
    )

    rows_by_identifier = {
        identifier: _candidate_rows(roots, system=system, identifier=identifier)
        for identifier in TOKENIZER_IDS
    }
    # The report's own tokenizer first: a reader scanning top-down should hit
    # the candidate's section before any other tokenizer's, not after.
    identifiers = [
        identifier
        for identifier in (
            own_identifier,
            *(name for name in TOKENIZER_IDS if name != own_identifier),
        )
        if rows_by_identifier[identifier]
    ]

    image_files: dict[str, Path] = {}
    report_scope_label = system or GLOBAL_TOKENIZER_SCOPE

    lines: list[str] = [
        f"# Tokenizer & Corpus Report -- {report_scope_label}",
        "",
        f"- Candidate: `{candidate.uri}`",
        f"- Generated: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        (
            f"- Built for the promoted `{own_identifier}` tokenizer of this scope. It "
            "lists every stored candidate of every tokenizer for the scope, not "
            f"`{own_identifier}`'s alone."
        ),
        f"- Corpus content hash: `{corpus_content_hash}`",
        "- Pure consolidation: every number and image below is read from what each",
        "  candidate was stored with, the optimize run's folder and the corpus",
        "  Zipf/OOV analysis outputs already produced -- nothing here is computed fresh.",
        "",
        "## Stored Tokenizer Candidates",
        "",
    ]

    if not identifiers:
        lines += ["_No stored tokenizer candidates found for this scope yet._", ""]

    for identifier in identifiers:
        marker = " (this report's candidate)" if identifier == own_identifier else ""
        lines += [f"### Tokenizer: {identifier}{marker}", ""]
        lines += [
            (
                "| candidate | named by | mode | vocab_size | min_frequency | fertility | "
                "fertility_distance | unk_rate | single_char_token_pct | stored |"
            ),
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            *rows_by_identifier[identifier],
            "",
        ]

        if identifier == own_identifier:
            files = tokenizer_directory_files(
                locate(roots, candidate), trainer=TOKENIZER_IDS[identifier].tokenizer
            )
            winner = (_read_json(files.optimize_summary) or {}).get("winner_candidate")
            if winner:
                lines += [
                    (
                        "- The optimize run that chose this candidate picked "
                        f"vocab={winner.get('vocab_size_requested')}, "
                        f"min_frequency={winner.get('min_frequency')}, "
                        "median_fertility_distance="
                        f"{_fmt_num(winner.get('median_fertility_distance'), digits=6)}, "
                        f"pass_rate={winner.get('pass_rate')} "
                        f"({winner.get('accepted_runs')}/{winner.get('total_runs')} seeds)."
                    ),
                    "",
                ]

        elbow_path = _live_elbow_image_path(
            roots=roots, system=system, identifier=identifier
        )
        if elbow_path is not None:
            image_name = f"elbow.{identifier}.png"
            image_files[image_name] = elbow_path
            lines += [f"![{identifier} elbow curve]({image_name})", ""]
        else:
            lines += [
                (
                    f"_No elbow-curve image available for tokenizer={identifier} -- the "
                    "stored metrics table above is real, complete evidence either way; "
                    "only the plot itself wasn't available to capture._"
                ),
                "",
            ]

    lines += ["## Corpus Zipf / OOV Divergence", ""]

    zipf_run_date: str | None = None
    if system is None:
        lines += [
            (
                "_Zipf/OOV evidence is computed per country system, not per global "
                "tokenizer scope -- not applicable here._"
            ),
            "",
        ]
    else:
        zipf_match = find_latest_zipf_summary_for_system(roots, system)
        if zipf_match is None:
            lines += [
                (
                    f"_No token-Zipf run found for system={system} "
                    f"(run `scripts/analyze_token_zipf.py --systems {system}` to produce one)._"
                ),
                "",
            ]
        else:
            zipf_run_date, rows_df = zipf_match
            lines += [
                f"- Source run: `artifacts/analysis/token_zipf/runs/{zipf_run_date}/`",
                "",
                (
                    "| tier | names | unique tokens | corpus slope | reference slope | "
                    "type OOV% | occurrence-weighted OOV% | names w/ hapax% | names w/ OOV% |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
            for row in rows_df.iter_rows(named=True):
                lines.append(
                    "| {tier} | {names:,} | {tokens:,} | {slope} | {ref} | {toov} | {ooov} | {hap} | {noov} |".format(
                        tier=row["tier"],
                        names=row["total_docs"],
                        tokens=row["unique_tokens"],
                        slope=_fmt_num(row["corpus_zipf_slope"], digits=2),
                        ref=_fmt_num(row["reference_zipf_slope"], digits=2),
                        toov=_fmt_pct(row["type_level_unseen_in_wordfreq_pct"]),
                        ooov=_fmt_pct(
                            row["occurrence_weighted_unseen_in_wordfreq_pct"]
                        ),
                        hap=_fmt_pct(row["names_with_hapax_pct"]),
                        noov=_fmt_pct(row["names_with_unseen_in_wordfreq_pct"]),
                    )
                )
            lines.append("")

            for row in rows_df.iter_rows(named=True):
                plot_rel = row.get("plot_path")
                if not plot_rel:
                    continue
                source_path = roots.checkout / str(plot_rel)
                if not source_path.exists():
                    continue
                tier = row["tier"]
                image_name = f"zipf.{tier}.png"
                image_files[image_name] = source_path
                lines += [
                    f"### {tier} tier",
                    "",
                    f"![{system} {tier} zipf curve]({image_name})",
                    "",
                ]
                # The curve and the head of the same distribution as named
                # words are a pair, and publishing the pair is the point of
                # copying them in here at all. `.get` because a summary
                # parquet written before `head_words_plot_path` existed has
                # no such column, and those runs keep rendering the single
                # picture.
                head_words_rel = row.get("head_words_plot_path")
                if not head_words_rel:
                    continue
                head_words_source_path = roots.checkout / str(head_words_rel)
                if not head_words_source_path.exists():
                    continue
                head_words_image_name = f"zipf.{tier}.head_words.png"
                image_files[head_words_image_name] = head_words_source_path
                lines += [
                    f"![{system} {tier} most frequent words]({head_words_image_name})",
                    "",
                ]

    return TokenizerCorpusReportBuild(
        markdown="\n".join(lines) + "\n",
        image_files=image_files,
        corpus_content_hash=corpus_content_hash,
        tokenizers_included=identifiers,
        zipf_run_date=zipf_run_date,
    )


def publish_tokenizer_corpus_report(
    *, roots: WorkspaceRoots, build: TokenizerCorpusReportBuild, candidate: Reference
) -> Path:
    """Publish `build` to its scope's directory under `docs/reports/tokenizers/`,
    replacing the report and pictures an earlier promotion left there."""
    directory = tokenizer_reports_dir(roots, scope=str(candidate.fields["scope"]))
    publish(
        directory,
        texts={PUBLISHED_REPORT_FILENAME: build.markdown},
        files=build.image_files,
        replace=True,
    )
    return directory


def generate_tokenizer_corpus_report(
    *,
    roots: WorkspaceRoots,
    system: str | None,
    trainer: str,
    tokenizer_encoding: str | None = None,
    publish_report: bool = True,
) -> PublishedTokenizerCorpusReport:
    """Build the report of the candidate the pointer names as `promoted` for a
    scope and tokenizer, and publish it unless `publish_report` is False, which
    is the off switch. Raises `pointer.NothingPromotedError` when the pointer
    names none: a candidate that is not promoted has no report.
    """
    candidate = promoted_tokenizer(
        roots, system=system, tokenizer_id=tokenizer_id(trainer, tokenizer_encoding)
    )
    build = build_tokenizer_corpus_report(roots=roots, candidate=candidate)
    published_dir = (
        publish_tokenizer_corpus_report(roots=roots, build=build, candidate=candidate)
        if publish_report
        else None
    )
    return PublishedTokenizerCorpusReport(
        candidate=candidate, build=build, published_dir=published_dir
    )
