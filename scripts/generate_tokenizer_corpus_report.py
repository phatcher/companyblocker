from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import add_root_arg, run_reporting_argument_errors
from company_tokenize import tokenizer_id
from plot_optimize_elbow import plot_optimize_elbow

from training.tokenizer_corpus_report import (
    PublishedTokenizerCorpusReport,
    generate_tokenizer_corpus_report,
)
from workspace.artifact_layout import GLOBAL_TOKENIZER_SCOPE
from workspace.kind_layout import Kind
from workspace.pointer import PointerError
from workspace.reference import Side, reference, select_references
from workspace.roots import WorkspaceRoots, default_workspace_roots
from workspace.tokenizer_store import scope_system


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the consolidated tokenizer/corpus report (the optimize elbow "
            "curve + metrics, plus the corpus Zipf/OOV curve + metrics) for the "
            "candidate config/tokenizers.json names as promoted for a scope and "
            "tokenizer, and publish it to docs/reports/tokenizers/<scope>/. Renders "
            "a fresh elbow curve for that tokenizer immediately before capturing "
            "it, so the report does not depend on someone having separately run "
            "plot_optimize_elbow.py first. A candidate that is not promoted has no "
            "report."
        )
    )
    add_root_arg(parser, help_text="Project root containing artifacts/tokenizers.")
    parser.add_argument(
        "--scope", choices=["country", "global"], default="country", help=""
    )
    parser.add_argument(
        "--system",
        default=None,
        help="System code, required for --scope country. Pass 'all' to report on "
        "every system with a stored candidate for this tokenizer, skipping any "
        "with nothing promoted.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["wordpiece", "sentencepiece"],
        default="wordpiece",
        help="Tokenizer whose promoted candidate is reported on.",
    )
    parser.add_argument(
        "--encoding",
        dest="tokenizer_encoding",
        choices=["bpe", "unigram"],
        default="bpe",
        help="SentencePiece encoding, when --tokenizer sentencepiece. Ignored for "
        "--tokenizer wordpiece.",
    )
    parser.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        help="Build the report without publishing it to docs/reports/tokenizers/<scope>/.",
    )
    return parser


def _discover_country_systems(roots: WorkspaceRoots, *, identifier: str) -> list[str]:
    """Every system with a stored candidate for this tokenizer id, so a caller
    never has to enumerate systems by hand."""
    stored = select_references(
        roots, reference(Kind.TOKENIZER, Side.DATA, tokenizer_id=identifier)
    )
    return sorted(
        {
            str(candidate.fields["scope"])
            for candidate in stored
            if candidate.fields["scope"] != GLOBAL_TOKENIZER_SCOPE
        }
    )


def _generate_one_report(
    *,
    roots: WorkspaceRoots,
    scope: str,
    system: str | None,
    trainer: str,
    tokenizer_encoding: str | None,
    publish: bool,
) -> PublishedTokenizerCorpusReport:
    # Render a fresh elbow curve right before the report captures it. A missing
    # run log (no optimize run for this tokenizer) is not fatal: the report
    # still generates, just without an elbow image.
    plot_optimize_elbow(
        roots=roots,
        scope=scope,
        system=system,
        profile="default",
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
    )
    return generate_tokenizer_corpus_report(
        roots=roots,
        system=scope_system(scope, system),
        trainer=trainer,
        tokenizer_encoding=tokenizer_encoding,
        publish_report=publish,
    )


def _report_line(report: PublishedTokenizerCorpusReport) -> str:
    where = (
        f"published to {report.published_dir}"
        if report.published_dir is not None
        else "built, not published"
    )
    return f"{report.candidate.uri} {where}"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.scope == "country" and not args.system:
        parser.error("--system is required when --scope country (or pass --system all)")

    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)

    if args.scope == "country" and args.system == "all":
        identifier = tokenizer_id(args.tokenizer, args.tokenizer_encoding)
        systems = _discover_country_systems(roots, identifier=identifier)
        if not systems:
            print(f"No stored {identifier} candidates found for any system")
            return 0
        for system in systems:
            try:
                report = _generate_one_report(
                    roots=roots,
                    scope=args.scope,
                    system=system,
                    trainer=args.tokenizer,
                    tokenizer_encoding=args.tokenizer_encoding,
                    publish=args.publish,
                )
                print(f"{system}: {_report_line(report)}")
            except PointerError as exc:
                print(f"{system}: skipped -- {exc}")
        return 0

    try:
        report = _generate_one_report(
            roots=roots,
            scope=args.scope,
            system=args.system,
            trainer=args.tokenizer,
            tokenizer_encoding=args.tokenizer_encoding,
            publish=args.publish,
        )
    except PointerError as exc:
        parser.error(str(exc))
    print(f"Report: {_report_line(report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
