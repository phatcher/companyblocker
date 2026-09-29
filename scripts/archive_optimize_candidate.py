from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import add_root_arg, run_reporting_argument_errors
from company_tokenize import archive_optimize_candidate

from workspace.artifact_layout import tokenizer_artifact_root
from workspace.roots import default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy one optimize candidate out of its optimize-sweep leaf into a "
            "durable, named location. --tokenizer/--encoding must match the "
            "sweep this candidate was trained in."
        )
    )
    add_root_arg(parser, help_text="Project root containing artifacts/tokenizers.")
    parser.add_argument(
        "--scope", choices=["country", "global"], default="country", help=""
    )
    parser.add_argument(
        "--system", default=None, help="System code, required for --scope country."
    )
    parser.add_argument(
        "--profile", default="default", help="Global tokenizer profile, if applicable."
    )
    parser.add_argument(
        "--tokenizer", choices=["wordpiece", "sentencepiece"], default="wordpiece"
    )
    parser.add_argument(
        "--encoding",
        dest="tokenizer_encoding",
        choices=["bpe", "unigram"],
        default="bpe",
        help=(
            "SentencePiece encoding the candidate was swept under, when "
            "--tokenizer sentencepiece. Ignored for --tokenizer wordpiece."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--vocab-size",
        type=int,
        required=True,
        help="Requested vocab size for this candidate (-1 for the 'auto' candidate).",
    )
    parser.add_argument("--min-frequency", type=int, required=True)
    parser.add_argument(
        "--label",
        required=True,
        help="Name for the archived candidate (for example 'naive_baseline' or "
        "'v25000_mf8_winner'). Written as <label>.json and <label>.metadata.json.",
    )
    parser.add_argument("--notes", default=None, help="Optional free-text note.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.scope == "country" and not args.system:
        parser.error("--system is required when --scope country")

    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)
    dest_path = archive_optimize_candidate(
        tokenizer_root=tokenizer_artifact_root(roots),
        scope=args.scope,
        system=args.system,
        profile=args.profile,
        trainer=args.tokenizer,
        tokenizer_encoding=args.tokenizer_encoding,
        seed=args.seed,
        vocab_requested=args.vocab_size,
        min_frequency=args.min_frequency,
        label=args.label,
        notes=args.notes,
    )
    metadata_path = dest_path.parent / f"{args.label}.metadata.json"
    print(f"Archived candidate: {dest_path}")
    print(f"Metadata: {metadata_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
