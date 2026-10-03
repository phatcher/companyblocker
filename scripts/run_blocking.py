from __future__ import annotations

import _polars_threads  # noqa: F401  (first: Polars reads it at import)

# isort: split
import argparse
import json
import sys
import time
from collections.abc import Collection, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    OUTPUT_EXTEND,
    PlannedOutput,
    add_declared_arguments,
    add_dry_run_arg,
    add_perturbed_source_args,
    add_workspace_roots_args,
    applicable_settings_record,
    backend_option_surface,
    backend_options_from_settings,
    declared_keys_help,
    declared_settings,
    parse_stage_key_value_args,
    report_output_plan,
    report_resolved_settings,
    resolve_declared_settings,
    resolve_workspace_roots_from_args,
    resolved_setting_values,
    run_reporting_argument_errors,
    setting_was_given,
    source_system_from_args,
)
from company_classify import (
    MeanPooledTokenVectorEncoder,
    TokenVectorProvenance,
    load_pretrained_fasttext_vectors,
    resolve_fasttext_checkpoint_entry,
    resolve_fasttext_slug_for_jurisdictions,
)
from company_tokenize._cli_helper import SETTINGS as TOKENIZE_SETTINGS
from company_vectorize._cli_helper import SETTINGS as VECTORIZE_SETTINGS
from company_vectorize.clustering_policy import resolve_sbert_model_for_jurisdictions
from company_vectorize.dense_vocabulary_gate import (
    DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
)

from blocking._cli_helper import SETTINGS as BLOCKING_SETTINGS
from blocking.contracts import (
    BlockingRunConfig,
    BlockingRunResult,
    BlockingStrategyConfig,
    build_blocking_run_identity,
    validate_blocking_run_config,
)
from blocking.loader import load_dataset_descriptor
from blocking.reporting import (
    produce_blocking_run,
    write_blocking_report,
    write_run_manifest,
)
from blocking.run_layout import (
    is_finished_run,
    resolve_pairing_dir,
    resolve_run_location_for,
)
from blocking.truth import source_truth_for_column
from blocking.workflow import execute_blocking_run, resolve_blocking_run_keys
from validation.config import resolve_text_view_for_representation
from validation.contracts import POPULATION_UNIVERSE
from validation.runner import NAME_EQUALITY_NEVER, pair_truth_eval_row
from workspace.artifact_layout import pretrained_vector_artifact_root
from workspace.identity import current_commit
from workspace.roots import WorkspaceRoots
from workspace.run_inputs import DERIVED_TRUTH_COLUMN

_DECLARATIONS = declared_settings(
    BLOCKING_SETTINGS, VECTORIZE_SETTINGS, TOKENIZE_SETTINGS
)

# Where this script takes each declared setting: a flag, or a prefixed
# `--additional-args` key, and the one default it sets for itself.
_SURFACE: dict[str, dict[str, object]] = {
    "source_system": {"flag": "--source", "required": True},
    "target_system": {"flag": "--target", "required": True},
    "match_col": {"flag": "--match-col"},
    "countries": {"flag": "--countries"},
    "require_ground_truth": {"flag": "--ground-truth"},
    "representation": {"flag": "--representation"},
    "similarity_backend": {"flag": "--similarity-backend"},
    "max_rows": {"flag": "--max-rows"},
    "force": {"flag": "--force"},
    "text_view": {"flag": "--text-view"},
    "name_transform": {"flag": "--name-transform"},
    "preprocess_profile": {"flag": "--preprocess-profile"},
    "cleanse_profile": {"flag": "--cleanse-profile"},
    "sbert_model_name": {"flag": "--sbert-model"},
    "encoder": {"flag": "--encoder"},
    "fasttext_checkpoint": {"flag": "--fasttext-checkpoint"},
    "top_k": {"flag": "--top-k"},
    "min_similarity": {"flag": "--min-similarity"},
    "max_candidates_per_source": {"flag": "--max-candidates-per-source"},
    # Smaller than a library run's batch, for more frequent progress lines.
    "source_chunk_size": {"flag": "--source-chunk-size", "default": 1000},
    "tfidf_ngram_min": {"key": "tfidf.ngram_min"},
    "tfidf_ngram_max": {"key": "tfidf.ngram_max"},
    "tfidf_analyzer": {"key": "tfidf.analyzer"},
    "tokenizer": {"key": "tokenizer.name"},
    "tokenizer_encoding": {"key": "tokenizer.encoding"},
    "tokenizer_scope": {"key": "tokenizer.scope"},
    "tokenizer_profile": {"key": "tokenizer.profile"},
    "tokenizer_path": {"key": "tokenizer.path"},
    "max_candidates_per_target": {"key": "pruning.max_candidates_per_target"},
    "candidate_similarity_ratio": {"key": "pruning.candidate_similarity_ratio"},
    "target_neighbor_min_similarity": {"flag": "--target-neighbor-min-similarity"},
    "target_neighbor_max_per_target": {"flag": "--target-neighbor-max-per-target"},
    # Every backend setting on its `backend.<option>` key.
    **backend_option_surface(_DECLARATIONS),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a two-dataset blocking workflow (candidate generation, "
        "scoring, clustering) for one source/target system pair.",
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    add_perturbed_source_args(parser)
    add_workspace_roots_args(parser)
    parser.add_argument(
        "--additional-args",
        nargs="+",
        action="append",
        metavar="STAGE.KEY=VALUE",
        help=(
            "Settings taken as prefixed keys: "
            f"{declared_keys_help(_DECLARATIONS, _SURFACE)}. Any other "
            "backend.<key> is passed to the backend as given."
        ),
    )
    # Plumbing for compare_blocking_strategies.py, which launches this script
    # and reads back where the run's outputs went: a path it makes under the
    # temp root, never one a user chooses, so it is kept out of the help.
    parser.add_argument("--run-record", default=None, help=argparse.SUPPRESS)
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run config (source/target datasets, strategy, output "
            "directory) and report the settings that apply, each with its value "
            "and whether it was given or defaulted, without executing the run. "
            "A run lands under --output-dir's "
            "blocking/data/<target>/<kind>/<source>/<representation>/<key>."
        ),
    )
    return parser


def _fasttext_checkpoint_slug(values: Mapping[str, Any]) -> str:
    """The `--fasttext-checkpoint` slug, or the one the run's `--countries` pick."""
    given = values["fasttext_checkpoint"]
    if given is not None and str(given).strip():
        return str(given).strip()
    countries = values["countries"]
    return resolve_fasttext_slug_for_jurisdictions(
        tuple(str(country).strip().lower() for country in countries)
        if countries
        else None
    )


def _download_command(slug: str) -> str:
    return (
        "download it with: uv run python scripts/measure_fasttext_alias_hit_rate.py "
        f"--download --slug {slug}"
    )


def _fasttext_encoder_name(slug: str) -> str:
    """What a run using fastText checkpoint `slug` records as its encoder: the
    slug and the registered checksum of the file, so a checkpoint swapped under
    one slug is a different run."""
    checksum = resolve_fasttext_checkpoint_entry(slug).checksum
    if checksum is None:
        raise ValueError(
            f"fastText checkpoint '{slug}' has no registered checksum, so there is "
            f"no verified file to name the run by; {_download_command(slug)}, then "
            "record the checksum it prints in fasttext_checkpoints.json"
        )
    return f"fasttext:{slug}:{checksum}"


def _fasttext_checkpoint_path(roots: WorkspaceRoots, slug: str) -> Path:
    """Where the decompressed checkpoint for `slug` sits, in the pretrained-vectors
    folder under its registered checksum.

    Raises:
        FileNotFoundError: The file is not there; the message carries the command
            that downloads it.
    """
    entry = resolve_fasttext_checkpoint_entry(slug)
    filename = entry.source_url.rsplit("/", maxsplit=1)[-1].removesuffix(".gz")
    path = pretrained_vector_artifact_root(roots) / str(entry.checksum) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"fastText checkpoint '{slug}' is not at {path}; {_download_command(slug)}"
        )
    return path


def _load_fasttext_lookup(path: Path, provenance: TokenVectorProvenance):
    return load_pretrained_fasttext_vectors(path, provenance=provenance)


class _FastTextEncoder:
    """The fastText encoder a run hands its blocking configuration: each name is
    the mean of its words' vectors, the checkpoint read on first use so a dry run
    or a refused configuration never loads several gigabytes."""

    def __init__(self, path: Path, provenance: TokenVectorProvenance) -> None:
        self._path = path
        self._provenance = provenance
        self._encoder: MeanPooledTokenVectorEncoder | None = None

    def embed(self, name: str):
        if self._encoder is None:
            self._encoder = MeanPooledTokenVectorEncoder(
                _load_fasttext_lookup(self._path, self._provenance)
            )
        return self._encoder.embed(name)


def _encoder_for_run(
    args: argparse.Namespace, strategy: BlockingStrategyConfig, roots: WorkspaceRoots
) -> _FastTextEncoder | None:
    """The encoder `--encoder` names, for a run that scores through one. Any other
    representation is left without it, for the configuration to refuse."""
    if strategy.encoder_name is None or strategy.representation != "encoder":
        return None
    resolved, _ = _resolve_settings(args)
    slug = _fasttext_checkpoint_slug(resolved_setting_values(resolved))
    entry = resolve_fasttext_checkpoint_entry(slug)
    return _FastTextEncoder(
        _fasttext_checkpoint_path(roots, slug),
        TokenVectorProvenance(
            slug=slug, source=entry.source_url, checksum=entry.checksum
        ),
    )


def _resolve_settings(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Every declared setting this script takes, resolved, with whether each
    applies to the representation, backend and text view the run chose; and the
    parsed `--additional-args`."""
    additional = parse_stage_key_value_args(
        args.additional_args,
        invalid_message="Invalid --additional-args token '{token}'. Expected STAGE.KEY=VALUE format.",
        empty_message="Invalid --additional-args token '{token}'. Stage and key must be non-empty.",
    )
    values = resolved_setting_values(
        resolve_declared_settings(args, _DECLARATIONS, _SURFACE, additional=additional)
    )
    representation = str(values["representation"]).strip().lower()
    try:
        text_view: str | None = resolve_text_view_for_representation(
            representation=representation,
            requested_text_view=str(values["text_view"]),
        )
    except (KeyError, ValueError):
        # An invalid pairing is refused when the configuration is validated;
        # here it only means no text view narrows what is reported.
        text_view = None
    context = {
        "representation": representation,
        "backend": str(values["similarity_backend"]).strip().lower(),
        "text_view": text_view,
    }
    resolved = resolve_declared_settings(
        args, _DECLARATIONS, _SURFACE, additional=additional, context=context
    )
    return resolved, additional


def _strategy_from_settings(
    values: Mapping[str, Any],
    additional: Mapping[str, Mapping[str, object]],
    *,
    applicable: Collection[str],
) -> BlockingStrategyConfig:
    # Backend options are the backend.<key> values the caller passed; every
    # declared backend setting that applies, given or defaulted, so a run never
    # shares the identity of one that ran its backend differently; and the row
    # gate's two only when they move off the default, so an invocation that
    # never touches them keeps the run identity, and the run directory, it had.
    backend_options: dict[str, object] = dict(additional.get("backend") or {})
    backend_options.update(
        backend_options_from_settings(
            _DECLARATIONS,
            {
                "representation": str(values["representation"]).strip().lower(),
                "backend": str(values["similarity_backend"]).strip().lower(),
            },
            {name: values[name] for name in applicable},
        )
    )
    if values["force"]:
        backend_options["force"] = True
    if int(values["max_rows"]) != DEFAULT_DENSE_VOCABULARY_MAX_ROWS:
        backend_options["max_rows"] = int(values["max_rows"])

    text_view = str(values["text_view"]).strip().lower()
    sbert_model_name = values["sbert_model_name"]
    if sbert_model_name is not None:
        sbert_model_name = str(sbert_model_name).strip() or None

    encoder_name = None
    if values["encoder"] is not None:
        # Only fasttext is declared, so naming an encoder is naming its checkpoint.
        encoder_name = _fasttext_encoder_name(_fasttext_checkpoint_slug(values))

    return BlockingStrategyConfig(
        representation=str(values["representation"]).strip().lower(),
        top_k=int(values["top_k"]),
        min_similarity=float(values["min_similarity"]),
        max_candidates_per_source=values["max_candidates_per_source"],
        similarity_backend=str(values["similarity_backend"]).strip().lower(),
        backend_options=backend_options or None,
        tfidf_ngram_min=int(values["tfidf_ngram_min"]),
        tfidf_ngram_max=int(values["tfidf_ngram_max"]),
        tfidf_analyzer=str(values["tfidf_analyzer"]).strip().lower(),
        text_view=None if text_view in ("", "auto") else text_view,
        preprocess_profile=str(values["preprocess_profile"]).strip(),
        tokenizer=str(values["tokenizer"]).strip().lower(),
        tokenizer_encoding=str(values["tokenizer_encoding"]).strip().lower(),
        tokenizer_scope=str(values["tokenizer_scope"]).strip().lower(),
        tokenizer_profile=str(values["tokenizer_profile"]).strip(),
        tokenizer_path=(
            None
            if values["tokenizer_path"] is None
            else str(values["tokenizer_path"]).strip()
        ),
        max_candidates_per_target=values["max_candidates_per_target"],
        candidate_similarity_ratio=values["candidate_similarity_ratio"],
        sbert_model_name=sbert_model_name,
        encoder_name=encoder_name,
        name_transform=str(values["name_transform"]).strip().lower(),
        cleanse_profile=str(values["cleanse_profile"]).strip(),
        target_neighbor_min_similarity=values["target_neighbor_min_similarity"],
        target_neighbor_max_per_target=values["target_neighbor_max_per_target"],
    )


def _resolve_strategy(args: argparse.Namespace) -> BlockingStrategyConfig:
    resolved, additional = _resolve_settings(args)
    return _strategy_from_settings(
        resolved_setting_values(resolved),
        additional,
        applicable=applicable_settings_record(resolved),
    )


def _build_config(
    args: argparse.Namespace, *, strategy: BlockingStrategyConfig
) -> BlockingRunConfig:
    roots = resolve_workspace_roots_from_args(args)
    # A perturbed source's truth is the row it was made from, so nobody types it.
    match_col = (
        DERIVED_TRUTH_COLUMN
        if args.perturbed is not None and not setting_was_given(args, "match_col")
        else str(args.match_col)
    )
    truth = source_truth_for_column(match_col)

    source = load_dataset_descriptor(
        roots=roots,
        system=source_system_from_args(args, roots),
        require_ground_truth=bool(args.require_ground_truth),
        truth=truth,
    )
    target = load_dataset_descriptor(
        roots=roots,
        system=args.target_system,
        require_ground_truth=False,
    )

    countries = (
        tuple(str(country).strip().lower() for country in args.countries)
        if args.countries
        else None
    )

    return BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=source,
        target=target,
        countries=countries,
        strategy=strategy,
        truth=truth,
        encoder=_encoder_for_run(args, strategy, roots),
    )


def _run_configuration(
    resolved: list[dict[str, object]], config: BlockingRunConfig
) -> dict[str, dict[str, object]]:
    """The applicable settings a run records, with the checkpoint an unset
    `--sbert-model` resolves to named beside it, and the encoder, checkpoint
    included, `--encoder` names."""
    record = applicable_settings_record(resolved)
    sbert = record.get("sbert_model_name")
    if sbert is not None and sbert["value"] is None:
        sbert["resolved"] = resolve_sbert_model_for_jurisdictions(config.countries)
    encoder = record.get("encoder")
    if encoder is not None and config.strategy.encoder_name is not None:
        encoder["resolved"] = config.strategy.encoder_name
    return record


def _format_bytes(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "unknown"
    return f"{float(value) / (1024**3):.2f}GiB"


# The first rows-scored line seen for each country, so a later one can say how
# fast rows have gone since and how long the rest should take.
_first_progress: dict[object, tuple[float, int]] = {}


def _scoring_pace(country: object, scored: object, total: object) -> str:
    if not isinstance(scored, int) or not isinstance(total, int):
        return ""
    began, scored_then = _first_progress.setdefault(country, (time.monotonic(), scored))
    elapsed = time.monotonic() - began
    if scored <= scored_then or elapsed <= 0:
        return ""
    per_row = elapsed / (scored - scored_then)
    return f" {1 / per_row:.0f} rows/s remaining~{(total - scored) * per_row:.0f}s"


def _print_progress(event: dict[str, object]) -> None:
    phase = event.get("phase")
    country = event.get("country")
    stamp = f"{datetime.now().astimezone():%H:%M:%S}"
    if phase == "phase_start":
        print(
            f"{stamp} [blocking] country={country} {event.get('name')} starting "
            f"(rows={event.get('rows')})"
        )
    elif phase == "phase_complete":
        elapsed = event.get("elapsed_seconds")
        seconds = f"{float(elapsed):.1f}s" if isinstance(elapsed, (int, float)) else "?"
        print(
            f"{stamp} [blocking] country={country} {event.get('name')} done in {seconds} "
            f"(rss={_format_bytes(event.get('rss_bytes'))} "
            f"peak={_format_bytes(event.get('peak_rss_bytes'))})"
        )
    elif phase == "country_start":
        print(
            f"{stamp} [blocking] country={country} starting "
            f"(source_rows={event.get('source_rows')} "
            f"target_rows={event.get('target_rows')})"
        )
    elif phase == "country_progress":
        print(
            f"{stamp} [blocking] country={country} "
            f"rows_scored={event.get('rows_scored')}/{event.get('source_rows')}"
            + _scoring_pace(country, event.get("rows_scored"), event.get("source_rows"))
        )
    elif phase == "country_complete":
        print(
            f"{stamp} [blocking] country={country} done "
            f"(matched_edges={event.get('matched_edges')})"
        )


def _print_eval_summary(result: BlockingRunResult) -> None:
    pair_eval = result.pair_truth_eval
    if pair_eval is None or pair_eval.height <= 0:
        return
    for country in pair_eval.get_column("country").unique(maintain_order=True):
        rows = pair_eval.filter(pl.col("country") == country)
        row = pair_truth_eval_row(rows, POPULATION_UNIVERSE)
        print(
            "[blocking] eval-summary "
            f"country={row.get('country')} "
            f"labelled_sources={row.get('labelled_sources')} "
            f"tp={row.get('tp')} fp={row.get('fp')} fn={row.get('fn')} "
            f"precision={row.get('precision')} recall={row.get('recall')} "
            # recall_at_k/candidate_set_size_ratio, the harness's
            # ranking-quality/candidate-volume metrics, alongside the
            # blended figures above.
            f"recall_at_k={row.get('recall_at_k')} "
            f"candidate_set_size_ratio={row.get('candidate_set_size_ratio')}"
        )
        # The universe line above is dominated by pairs already equal
        # (raw or after cleansing) -- this corpus's residual, the population
        # blocking actually had to earn, is the `never` population. It has its
        # own whole matrix, so its precision is printed beside its recall.
        # Printed as its own line rather than folded into the line above so the
        # headline figure a reader takes away is this one, not the universe's.
        # A run given no name forms has no levels and prints no residual line.
        if rows.filter(pl.col("population") == NAME_EQUALITY_NEVER).height == 0:
            continue
        residual = pair_truth_eval_row(rows, NAME_EQUALITY_NEVER)
        print(
            "[blocking] residual-summary "
            f"country={residual.get('country')} "
            f"residual_truth_pairs={residual.get('truth_pairs')} "
            f"tp={residual.get('tp')} fp={residual.get('fp')} "
            f"fn={residual.get('fn')} "
            f"precision={residual.get('precision')} "
            f"recall={residual.get('recall')}"
        )


def run_blocking(args: argparse.Namespace) -> int:
    resolved, additional = _resolve_settings(args)
    strategy = _strategy_from_settings(
        resolved_setting_values(resolved),
        additional,
        applicable=applicable_settings_record(resolved),
    )
    config = _build_config(args, strategy=strategy)
    validate_blocking_run_config(config)
    configuration = _run_configuration(resolved, config)

    if args.dry_run:
        sbert = configuration.get("sbert_model_name", {})
        encoder = configuration.get("encoder", {})
        report_resolved_settings(
            "run_blocking",
            resolved,
            **(
                {"sbert_model_resolved": sbert["resolved"]}
                if "resolved" in sbert
                else {}
            ),
            **(
                {"encoder_resolved": encoder["resolved"]}
                if "resolved" in encoder
                else {}
            ),
            source_dir=config.source.system_dir,
            source_layer=config.source.layer,
            has_ground_truth=config.source.has_ground_truth,
            target_dir=config.target.system_dir,
            polars_threads=pl.thread_pool_size(),
        )
        report_output_plan(
            "[dry-run]  ",
            [
                PlannedOutput(
                    resolve_pairing_dir(
                        config.roots,
                        source_system=config.source.system,
                        target_system=config.target.system,
                    ),
                    OUTPUT_EXTEND,
                    note=f"a new run under {config.strategy.representation}/, keyed "
                    "once its inputs are read, or the finished run with the same keys "
                    "reused",
                )
            ],
        )
        return 0

    print(
        f"[blocking] starting {config.source.system} -> {config.target.system} "
        f"({config.strategy.representation}, {pl.thread_pool_size()} Polars threads)"
    )

    # The run's keys come from reading what it would consume, before any
    # scoring, so a finished run with the same keys is reused instead of
    # scored again.
    keys = resolve_blocking_run_keys(config, progress_callback=_print_progress)
    location = resolve_run_location_for(config, keys=keys)
    run_dir = location.directory
    if is_finished_run(config.roots, location):
        print(f"[blocking] reusing the finished run at {run_dir}")
        _write_run_record(args.run_record, run_dir=run_dir, reused=True)
        return 0

    # The run is staged and put in place only once everything is written,
    # its production record last, so a run that fails leaves nothing behind.
    with produce_blocking_run(config, keys=keys, invocation=sys.argv) as staged:
        started = time.perf_counter()
        result = execute_blocking_run(
            config,
            source_chunk_size=int(args.source_chunk_size),
            progress_callback=_print_progress,
            expected_keys=keys,
        )
        elapsed = time.perf_counter() - started
        print(f"[blocking] completed in {elapsed:.1f}s")

        _print_eval_summary(result)

        paths = write_blocking_report(staged, result)
        paths.append(
            write_run_manifest(
                staged,
                identity=build_blocking_run_identity(config, keys=keys),
                configuration=configuration,
                roots=config.roots,
                commit=current_commit(config.roots.checkout),
            )
        )
    print(f"[blocking] run directory: {run_dir}")
    for path in paths:
        print(f"[blocking] {path.name}: {run_dir / path.name}")
    _write_run_record(args.run_record, run_dir=run_dir, reused=False)

    return 0


def _write_run_record(path: str | None, *, run_dir: Path, reused: bool) -> None:
    """Record where the run's outputs are and whether they were reused, for the
    caller that passed `--run-record`."""
    if path is None:
        return
    Path(path).write_text(
        json.dumps({"run_dir": str(run_dir), "reused": reused}), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_blocking(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[blocking] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
