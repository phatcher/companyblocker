from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import cast

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    add_declared_arguments,
    add_dry_run_arg,
    add_perturbed_source_args,
    add_root_arg,
    declared_keys_help,
    declared_settings,
    parse_stage_key_value_args,
    perturbed_source_from_args,
    report_resolved_settings,
    resolve_declared_settings,
    resolved_setting_values,
    run_reporting_argument_errors,
)

from acquisition.constants_pipeline import PROCESS_STAGE_CLEANSE
from acquisition.constants_status import (
    STATUS_LIVE,
    STATUS_RESEARCH_REQUIRED,
    is_research_stage_allowed,
)
from acquisition.pipeline import resolve_system_selection
from acquisition.plan_registry import COUNTRY_REGISTRY, SYSTEM_REGISTRY, get_system_plan
from validation._cli_helper import SETTINGS as VALIDATION_SETTINGS
from validation.config import (
    ValidationRunConfig,
    resolve_prepared_rows_per_file,
    resolve_text_view_for_representation,
    validate_run_config,
)
from validation.contracts import ARTIFACT_SCHEMAS, POPULATION_UNIVERSE
from validation.prepared_dataset import prepare_system_dataset
from validation.reporting import write_validation_artifacts
from validation.runner import pair_truth_eval_row, run_validation_matrix
from workspace.artifact_archive import resolve_candidate
from workspace.artifact_layout import validation_artifact_root
from workspace.roots import WorkspaceRoots, default_workspace_roots

_DECLARATIONS = declared_settings(VALIDATION_SETTINGS)

# Where this script takes each declared setting: a flag, or a prefixed
# `--additional-args` key. A flag naming which run this is (the systems, the
# countries, a column override, an output path) is not declared -- it stays
# an ordinary argument below.
_SURFACE: dict[str, dict[str, object]] = {
    "representation": {"flag": "--representation"},
    "similarity": {"flag": "--similarity"},
    "similarity_backend": {"flag": "--similarity-backend"},
    "clustering": {"flag": "--clustering"},
    "text_view": {"flag": "--text-view"},
    "top_k": {"flag": "--top-k"},
    "min_similarity": {"flag": "--min-similarity"},
    "max_candidates_per_source": {"flag": "--max-candidates-per-source"},
    "progress_record_interval": {"flag": "--progress-record-interval"},
    "source_chunk_size": {"flag": "--source-chunk-size"},
    "progress_time_interval_seconds": {"flag": "--progress-time-interval-seconds"},
    "prepared_rows_per_file": {"flag": "--prepared-rows-per-file"},
    "max_rows": {"flag": "--max-rows"},
    "tfidf_ngram_min": {"key": "tfidf.ngram_min"},
    "tfidf_ngram_max": {"key": "tfidf.ngram_max"},
    "tfidf_analyzer": {"key": "tfidf.analyzer"},
    "tokenizer": {"key": "tokenizer.name"},
    "tokenizer_scope": {"key": "tokenizer.scope"},
    "tokenizer_profile": {"key": "tokenizer.profile"},
    "tokenizer_path": {"key": "tokenizer.path"},
    "noise_words_profile": {"key": "noise_words.profile"},
    "noise_words_set_kind": {"key": "noise_words.set_kind"},
}


def _live_system_codes() -> tuple[str, ...]:
    live = sorted(
        {
            code
            for code, plan in {**COUNTRY_REGISTRY, **SYSTEM_REGISTRY}.items()
            if plan.status == STATUS_LIVE
        }
    )
    return tuple(live)


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    remaining = max(0, round(seconds))
    hours, rem = divmod(remaining, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _to_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, (int, float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _to_float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    return default


def _parse_additional_args(
    additional_args: list[list[str]] | None,
) -> dict[str, dict[str, object]]:
    return parse_stage_key_value_args(
        additional_args,
        invalid_message="Invalid --additional-args token '{token}'. Expected STAGE.KEY=VALUE format.",
        empty_message="Invalid --additional-args token '{token}'. Stage and key must be non-empty.",
    )


def _slugify_segment(value: object) -> str:
    text = str(value).strip().lower()
    if not text:
        return "default"
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "default"


def _filter_live_systems(
    system_codes: tuple[str, ...],
    *,
    allow_research: bool,
) -> tuple[tuple[str, ...], list[tuple[str, str]]]:
    live: list[str] = []
    skipped: list[tuple[str, str]] = []
    for system_code in system_codes:
        try:
            plan = get_system_plan(system_code)
        except KeyError as exc:
            known_live = ", ".join(_live_system_codes())
            raise ValueError(
                f"Unknown system '{system_code}'. Live systems: {known_live}."
            ) from exc
        if plan.status == STATUS_LIVE:
            live.append(system_code)
            continue

        if plan.status == STATUS_RESEARCH_REQUIRED and allow_research:
            if is_research_stage_allowed(plan, stage=PROCESS_STAGE_CLEANSE):
                live.append(system_code)
            else:
                skipped.append(
                    (
                        system_code,
                        f"{plan.status} (policy denies stage '{PROCESS_STAGE_CLEANSE}')",
                    )
                )
            continue

        skipped.append((system_code, plan.status))
    return tuple(live), skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run directional source-target clustering validation.",
    )
    parser.add_argument(
        "--sources",
        dest="source_systems",
        nargs="+",
        required=True,
        help="One or more source systems.",
    )
    add_perturbed_source_args(parser)
    parser.add_argument(
        "--targets",
        dest="target_systems",
        nargs="+",
        required=True,
        help="One or more target systems.",
    )
    parser.add_argument(
        "--countries", nargs="+", default=None, help="Optional country filter list."
    )
    parser.add_argument(
        "--date",
        dest="run_date",
        default=None,
        help="Run date context (reserved for future date-partitioned inputs).",
    )
    parser.add_argument(
        "--source-name-col",
        default=None,
        help="Cleansed source name column used for source-side matching input. Defaults to --name-col.",
    )
    parser.add_argument(
        "--name-col",
        default="name_cleansed",
        help="Cleansed name column to use as primary text.",
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for artifacts. Defaults to artifacts/validation/clustering/<source>__<target>/<representation-family>/<comparison>/<family-config>/<key>.",
    )
    parser.add_argument(
        "--prepared-base-dir",
        default=None,
        help=(
            "Prepared system dataset base directory. "
            "Defaults to data/<system>/matched when available, otherwise data/<system>/cleansed."
        ),
    )
    parser.add_argument(
        "--additional-args",
        nargs="+",
        action="append",
        default=[],
        metavar="STAGE.KEY=VALUE",
        help=(
            "Method-specific options taken as prefixed keys: "
            f"{declared_keys_help(_DECLARATIONS, _SURFACE)}."
        ),
    )
    parser.add_argument(
        "--include-reverse-direction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run target-to-source direction (reserved for future use).",
    )
    parser.add_argument(
        "--allow-research",
        action="store_true",
        default=False,
        help="Include research_required systems when research policy allows cleanse-stage runtime.",
    )
    parser.add_argument(
        "--exception-policy", default=None, help="Optional exception policy JSON path."
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only materialize prepared parquet partitions (jurisdiction_code hive layout) and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding prepared jurisdiction partitions even when existing partitions are present.",
    )
    add_root_arg(parser, help_text="Project root path.")
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve the run configuration (settings, resolved output "
            "directory) and report it, without scoring or writing anything."
        ),
    )
    return parser


def _resolve_selected_live_systems(
    args: argparse.Namespace,
) -> tuple[WorkspaceRoots, tuple[str, ...], tuple[str, ...]]:
    root = Path(args.root).resolve()
    roots = default_workspace_roots(root)
    source_selection = resolve_system_selection(args.source_systems)
    target_selection = resolve_system_selection(args.target_systems)

    if "all" in source_selection.requested:
        print(
            "[info] Resolved systems for --sources all: "
            + ", ".join(source_selection.concrete)
        )
    if "all" in target_selection.requested:
        print(
            "[info] Resolved systems for --targets all: "
            + ", ".join(target_selection.concrete)
        )

    source_systems_requested = tuple(source_selection.concrete)
    target_systems_requested = tuple(target_selection.concrete)
    source_systems, source_skipped = _filter_live_systems(
        source_systems_requested,
        allow_research=bool(args.allow_research),
    )
    target_systems, target_skipped = _filter_live_systems(
        target_systems_requested,
        allow_research=bool(args.allow_research),
    )

    for system_code, status in source_skipped:
        print(
            f"[info] skipping non-live source system '{system_code}' (status={status})"
        )
    for system_code, status in target_skipped:
        print(
            f"[info] skipping non-live target system '{system_code}' (status={status})"
        )

    if not source_systems:
        raise ValueError("No live source systems remain after filtering.")
    if not target_systems:
        raise ValueError("No live target systems remain after filtering.")
    # `--perturbed <profile>` reads every source as perturbed under that profile:
    # each becomes its dataset, pinned to the version and seed on disk.
    source_systems = tuple(
        dataset.uri
        if (dataset := perturbed_source_from_args(args, roots, source=system))
        is not None
        else system
        for system in source_systems
    )
    return roots, source_systems, target_systems


def _resolve_representation_runtime_settings(
    *,
    values: dict[str, object],
    additional: dict[str, dict[str, object]],
) -> tuple[str, int, int, str, str, str, str, str]:
    tfidf_options_present = bool(additional.get("tfidf"))
    tokenizer_options = additional.get("tokenizer", {})

    # Already coerced to int by `resolve_declared_settings`'s declared type;
    # `cast` only satisfies mypy, which sees a plain `object` here.
    tfidf_ngram_min = cast(int, values["tfidf_ngram_min"])
    tfidf_ngram_max = cast(int, values["tfidf_ngram_max"])
    tokenizer = str(values["tokenizer"]).strip().lower()
    tokenizer_scope = str(values["tokenizer_scope"]).strip().lower()
    tokenizer_profile = str(values["tokenizer_profile"]).strip()
    noise_words_profile = str(values["noise_words_profile"]).strip()
    noise_words_set_kind = str(values["noise_words_set_kind"]).strip()

    requested_representation = str(values["representation"]).strip().lower()
    if requested_representation == "":
        requested_representation = "tfidf"
    if requested_representation not in {"tfidf", "wordpiece", "sentencepiece"}:
        raise ValueError(
            "representation must be one of: tfidf, wordpiece, sentencepiece"
        )

    if requested_representation == "tfidf":
        if tokenizer_options:
            raise ValueError(
                "tokenizer.* additional args are only valid for token representations"
            )
        representation_family = "tfidf"
    else:
        if tfidf_options_present:
            raise ValueError(
                "tfidf.* additional args are only valid for representation=tfidf"
            )
        representation_family = requested_representation
        if "name" not in tokenizer_options:
            tokenizer = representation_family
        elif tokenizer != representation_family:
            raise ValueError(
                f"representation '{representation_family}' requires tokenizer.name='{representation_family}'"
            )

    return (
        representation_family,
        tfidf_ngram_min,
        tfidf_ngram_max,
        tokenizer,
        tokenizer_scope,
        tokenizer_profile,
        noise_words_profile,
        noise_words_set_kind,
    )


def _resolve_output(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    source_systems: tuple[str, ...],
    target_systems: tuple[str, ...],
    representation_family: str,
    comparison: str,
    family_config_slug: str,
    method_config: dict[str, object],
) -> tuple[Path, str]:
    """Where this run writes, and the digest the workspace keyed it by.

    Both come from the one resolver, so the digest recorded in the run's
    manifest is the one that named the directory rather than a second hash of
    the same settings. An explicit `--output-dir` relocates where the run
    lands; it does not change the digest that identifies it.
    """
    # Slugify each system code before joining: a perturbed source is its dataset's URI,
    # whose ':' and '/' are illegal in a Windows path segment.
    # A no-op for every ordinary system code (already [a-z0-9-]+).
    source_label = "-".join(_slugify_segment(system) for system in source_systems)
    target_label = "-".join(_slugify_segment(system) for system in target_systems)
    resolved = resolve_candidate(
        validation_artifact_root(roots) / "clustering",
        f"{source_label}__{target_label}",
        _slugify_segment(representation_family),
        comparison,
        family_config_slug,
        settings=method_config,
    )
    if args.output_dir:
        return (roots.checkout / Path(args.output_dir)).resolve(), resolved.signature
    return resolved.directory.resolve(), resolved.signature


def _resolve_settings(
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    """Every declared setting this script takes, resolved, with whether each
    applies to the representation the run chose; and the parsed
    `--additional-args`."""
    additional = _parse_additional_args(args.additional_args)
    values = resolved_setting_values(
        resolve_declared_settings(args, _DECLARATIONS, _SURFACE, additional=additional)
    )
    representation = str(values["representation"]).strip().lower()
    try:
        text_view: str | None = resolve_text_view_for_representation(
            representation=representation,
            requested_text_view=str(values["text_view"]).strip().lower(),
        )
    except (KeyError, ValueError):
        # An invalid pairing is refused once the configuration is validated;
        # here it only means no text view narrows what is reported.
        text_view = None
    context = {"representation": representation, "text_view": text_view}
    resolved = resolve_declared_settings(
        args, _DECLARATIONS, _SURFACE, additional=additional, context=context
    )
    return resolved, additional


def _build_validation_runtime_context(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    source_systems: tuple[str, ...],
    target_systems: tuple[str, ...],
) -> tuple[
    ValidationRunConfig,
    str,
    str,
    str,
    str,
    dict[str, object],
    str,
]:
    resolved, additional = _resolve_settings(args)
    values = resolved_setting_values(resolved)
    backend_options = additional.get("backend", {})

    (
        representation_family,
        tfidf_ngram_min,
        tfidf_ngram_max,
        tokenizer,
        tokenizer_scope,
        tokenizer_profile,
        noise_words_profile,
        noise_words_set_kind,
    ) = _resolve_representation_runtime_settings(values=values, additional=additional)
    tfidf_analyzer = str(values["tfidf_analyzer"]).strip().lower()

    text_view = resolve_text_view_for_representation(
        representation=representation_family,
        requested_text_view=str(values["text_view"]).strip().lower(),
    )

    comparison = _slugify_segment(f"{values['similarity']}-{values['clustering']}")
    if representation_family == "tfidf":
        family_arg_parts = [
            _slugify_segment(f"ngram-{tfidf_ngram_min}-{tfidf_ngram_max}"),
            _slugify_segment(f"analyzer-{tfidf_analyzer}"),
            _slugify_segment(f"backend-{values['similarity_backend']}"),
        ]
    else:
        family_arg_parts = [
            _slugify_segment(f"scope-{tokenizer_scope}"),
            _slugify_segment(f"stop-{noise_words_profile}-{noise_words_set_kind}"),
            _slugify_segment(f"backend-{values['similarity_backend']}"),
        ]
    family_config_slug = "_".join(family_arg_parts)
    method_id = "_".join(
        [
            _slugify_segment(representation_family),
            comparison,
            family_config_slug,
        ]
    )

    # Opt-in only: absent means the tokenizer the profile names.
    tokenizer_path = (
        None
        if values["tokenizer_path"] is None
        else str(values["tokenizer_path"]).strip()
    )

    effective_method_config: dict[str, object] = {
        "representation": representation_family,
        "similarity": str(values["similarity"]).strip(),
        "similarity_backend": str(values["similarity_backend"]).strip(),
        "clustering": str(values["clustering"]).strip(),
        "text_view": text_view,
        "top_k": int(values["top_k"]),
        "min_similarity": float(values["min_similarity"]),
        "max_candidates_per_source": values["max_candidates_per_source"],
        "backend": backend_options,
    }
    if representation_family == "tfidf":
        effective_method_config["tfidf"] = {
            "ngram_min": tfidf_ngram_min,
            "ngram_max": tfidf_ngram_max,
            "analyzer": tfidf_analyzer,
        }
    else:
        tokenizer_config: dict[str, object] = {
            "name": tokenizer,
            "scope": tokenizer_scope,
            "profile": tokenizer_profile,
            "noise_words_profile": noise_words_profile,
            "noise_words_set_kind": noise_words_set_kind,
        }
        # Keyed into the config hash only when given, so an ordinary run keeps
        # the hash (and therefore the output directory) it has today.
        if tokenizer_path:
            tokenizer_config["path"] = tokenizer_path
        effective_method_config["tokenizer"] = tokenizer_config

    # The workspace derives the key; this script hands over the settings and
    # composes nothing. The digest it used names the resolved directory and is
    # what the run records.
    resolved_output_dir, config_hash = _resolve_output(
        args=args,
        roots=roots,
        source_systems=source_systems,
        target_systems=target_systems,
        representation_family=representation_family,
        comparison=comparison,
        family_config_slug=family_config_slug,
        method_config=effective_method_config,
    )

    config = ValidationRunConfig(
        roots=roots,
        run_date=args.run_date,
        source_systems=source_systems,
        target_systems=target_systems,
        countries=tuple(
            str(value).strip().lower() for value in args.countries if str(value).strip()
        )
        if args.countries
        else None,
        name_col=str(args.name_col).strip(),
        representation=representation_family,
        similarity=str(values["similarity"]).strip(),
        clustering=str(values["clustering"]).strip(),
        text_view=text_view,
        similarity_backend=str(values["similarity_backend"]).strip(),
        backend_options=backend_options,
        tokenizer=tokenizer,
        tokenizer_scope=tokenizer_scope,
        tokenizer_profile=tokenizer_profile,
        tokenizer_path=tokenizer_path,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
        top_k=int(values["top_k"]),
        min_similarity=float(values["min_similarity"]),
        max_candidates_per_source=values["max_candidates_per_source"],
        output_dir=resolved_output_dir,
        prepared_base_dir=(
            Path(args.prepared_base_dir).resolve() if args.prepared_base_dir else None
        ),
        prepared_rows_per_file=resolve_prepared_rows_per_file(
            values["prepared_rows_per_file"]
        ),
        include_reverse_direction=bool(args.include_reverse_direction),
        exception_policy_path=Path(args.exception_policy).resolve()
        if args.exception_policy
        else None,
        source_name_col=(
            str(args.source_name_col).strip()
            if args.source_name_col
            else str(args.name_col).strip()
        ),
        tfidf_ngram_min=tfidf_ngram_min,
        tfidf_ngram_max=tfidf_ngram_max,
        tfidf_analyzer=tfidf_analyzer,
    )
    validate_run_config(config)
    return (
        config,
        method_id,
        representation_family,
        comparison,
        family_config_slug,
        effective_method_config,
        config_hash,
    )


def _resolve_max_rows_arg(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        raise TypeError("max_rows must be an integer")
    try:
        parsed_max_rows = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_rows must be an integer") from exc
    if parsed_max_rows <= 0:
        raise ValueError("max_rows must be greater than 0")
    return parsed_max_rows


def _run_prepare_only_mode(
    *, config: ValidationRunConfig, force: bool, max_rows: int | None
) -> int:
    prepared_by_system: dict[str, object] = {}

    def _prepare_and_log(system: str) -> None:
        if system in prepared_by_system:
            return
        stats = prepare_system_dataset(
            config=config,
            system=system,
            force_rebuild=force,
        )
        prepared_by_system[system] = stats
        print(
            "[validation] prepared "
            f"system={system} "
            f"countries={len(stats.kept_rows_by_country)} "
            f"dir={stats.system_dir}"
        )

    for system in sorted(set(config.target_systems)):
        _prepare_and_log(system)

    for system in sorted(set(config.source_systems)):
        _prepare_and_log(system)

    if max_rows is not None:
        print(
            "[info] --max-rows is a runtime scoring cap and is ignored for --prepare-only."
        )
    return 0


def _write_run_manifest(
    *,
    config: ValidationRunConfig,
    manifest_path: Path,
    method_id: str,
    representation_family: str,
    comparison: str,
    family_config_slug: str,
    config_hash: str,
    effective_method_config: dict[str, object],
) -> None:
    manifest_path.write_text(
        json.dumps(
            {
                "method_id": method_id,
                "path_segments": {
                    "representation_family": representation_family,
                    "comparison": comparison,
                    "family_config": family_config_slug,
                    "config_hash": config_hash,
                },
                "config_hash": config_hash,
                "effective_method_config": effective_method_config,
                "source_systems": list(config.source_systems),
                "target_systems": list(config.target_systems),
                "countries": list(config.countries) if config.countries else None,
            },
            ensure_ascii=True,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_placeholder_artifacts(config: ValidationRunConfig) -> None:
    # Emit placeholder artifacts immediately so filesystem observers can see run activity early.
    for artifact_name in (
        "source_outcomes",
        "clusters",
        "directional_coverage",
        "cluster_shape",
        "pair_truth_eval",
        "exceptions",
        "run_metrics",
    ):
        schema = ARTIFACT_SCHEMAS[artifact_name]
        placeholder = pl.DataFrame({column: [] for column in schema}, schema=schema)
        placeholder.write_parquet(config.output_dir / f"{artifact_name}.parquet")


def _write_running_status(*, config: ValidationRunConfig, status_path: Path) -> None:
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "started_at_unix": time.time(),
                "source_systems": list(config.source_systems),
                "target_systems": list(config.target_systems),
                "output_dir": str(config.output_dir),
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_progress_emitter(progress_path: Path):
    last_country_log_rows: dict[tuple[str, str, str], int] = {}
    last_country_log_ts: dict[tuple[str, str, str], float] = {}

    def _emit_progress(event: dict[str, object]) -> None:
        phase = str(event.get("phase", ""))
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")

        def _pair_scope() -> str:
            return (
                f"{event.get('source_system')}->{event.get('target_system')} "
                f"country={event.get('country')}"
            )

        def _log_pair_phase(label: str, extra: str = "") -> None:
            suffix = f" {extra}" if extra else ""
            print(f"[info] {label} {_pair_scope()}{suffix}")

        if phase == "pair_start":
            _log_pair_phase(
                "pair-start",
                f"source_rows={event.get('source_rows')} target_rows={event.get('target_rows')}",
            )
            return

        if phase == "target_cache_check":
            _log_pair_phase("target-cache-check")
            return

        if phase in {"source_token_cache_check", "target_token_cache_check"}:
            side = "source" if phase.startswith("source") else "target"
            _log_pair_phase(f"{side}-token-cache-check")
            return

        if phase in {
            "source_token_cache_hit",
            "source_token_cache_miss",
            "target_token_cache_hit",
            "target_token_cache_miss",
        }:
            side = "source" if phase.startswith("source") else "target"
            status = "hit" if phase.endswith("_hit") else "miss"
            build_elapsed = _to_float(event.get("build_elapsed_seconds", 0.0))
            _log_pair_phase(
                f"{side}-token-cache-{status}",
                " ".join(
                    [
                        f"rows={event.get(f'{side}_token_rows')}",
                        f"cache_dir={event.get(f'{side}_token_cache_dir')}",
                        f"elapsed={build_elapsed:.1f}s",
                    ]
                ),
            )
            return

        if phase in {"target_cache_hit", "target_cache_miss"}:
            cache_status = "hit" if phase.endswith("_hit") else "miss"
            parts = [
                f"rows={event.get('target_rows')}",
                f"cache_dir={event.get('target_cache_dir')}",
            ]
            if cache_status == "miss":
                build_elapsed = _to_float(event.get("build_elapsed_seconds", 0.0))
                parts.append(f"elapsed={build_elapsed:.1f}s")
            _log_pair_phase(f"target-cache-{cache_status}", " ".join(parts))
            return

        if phase == "target_nn_ready":
            _log_pair_phase("target-nn-ready", f"rows={event.get('target_rows')}")
            return

        if phase == "country_start":
            key = (
                str(event.get("source_system", "")),
                str(event.get("target_system", "")),
                str(event.get("country", "")),
            )
            last_country_log_rows[key] = 0
            last_country_log_ts[key] = time.perf_counter()
            _log_pair_phase(
                "country-start",
                f"source_rows={event.get('source_rows')} target_rows={event.get('target_rows')}",
            )
            return

        if phase == "country_chunk":
            rows_scored = _to_int(event.get("source_rows_scored", 0))
            rows_total = _to_int(event.get("source_rows_total", 0))
            elapsed = _to_float(event.get("elapsed_seconds", 0.0))
            _log_pair_phase(
                "country-chunk",
                (
                    f"source_rows_scored={rows_scored}/{rows_total} "
                    f"chunk_edges_found={event.get('chunk_edges_found')} "
                    f"elapsed={elapsed:.1f}s"
                ),
            )
            return

        if phase == "system_prepared":
            elapsed = _to_float(event.get("elapsed_seconds", 0.0))
            print(
                "[info] system-prepared "
                f"system={event.get('system')} "
                f"countries={event.get('countries')} "
                f"elapsed={elapsed:.1f}s"
            )
            return

        if phase == "pair":
            elapsed = _to_float(event.get("elapsed_seconds", 0.0))
            eta_raw = event.get("eta_seconds")
            eta = _format_eta(eta_raw if isinstance(eta_raw, (int, float)) else None)
            print(
                "[info] progress "
                f"{event.get('source_system')}->{event.get('target_system')} "
                f"units={event.get('completed_units')}/{event.get('total_units')} "
                f"target_rows={event.get('target_rows')} "
                f"source_rows_processed={event.get('source_rows_processed')} "
                f"source_rows_found={event.get('source_rows_with_match')} "
                f"edges_found={event.get('edges_found')} "
                f"cumulative_target_rows={event.get('target_rows_cumulative')} "
                f"cumulative_source_rows_processed={event.get('source_rows_processed_cumulative')} "
                f"cumulative_source_rows_found={event.get('source_rows_with_match_cumulative')} "
                f"cumulative_edges_found={event.get('edges_found_cumulative')} "
                f"elapsed={elapsed:.1f}s eta={eta}"
            )

    return _emit_progress


def _write_completed_status(
    *,
    config: ValidationRunConfig,
    status_path: Path,
    total_elapsed: float,
    timings: list[dict[str, object]],
) -> None:
    status_path.write_text(
        json.dumps(
            {
                "state": "completed",
                "started_at_unix": time.time() - total_elapsed,
                "completed_at_unix": time.time(),
                "elapsed_seconds": total_elapsed,
                "output_dir": str(config.output_dir),
                # Each phase's wall-clock, resident set size and CPU share
                # (`workspace.telemetry`), so the run's cost is read here.
                "timings": timings,
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _print_pair_eval_summary(result_frames: dict[str, pl.DataFrame]) -> None:
    pair_eval = result_frames.get("pair_truth_eval")
    if pair_eval is None or pair_eval.height <= 0:
        return

    # The cross-country rollup when there is one, else the first country, and
    # within it the universe selected by name: each country holds one row per
    # population, so taking a row by position would read an arbitrary one.
    countries = pair_eval.get_column("country").unique(maintain_order=True).to_list()
    country = "__all__" if "__all__" in countries else countries[0]
    summary_row = pair_truth_eval_row(
        pair_eval.filter(pl.col("country") == country), POPULATION_UNIVERSE
    )
    labelled_sources = summary_row.get("labelled_sources")
    if not isinstance(labelled_sources, int) or labelled_sources <= 0:
        return

    print(
        "[validation] eval-summary "
        f"source={summary_row.get('source_system')} "
        f"target={summary_row.get('target_system')} "
        f"country={summary_row.get('country')} "
        f"labelled_sources={summary_row.get('labelled_sources')} "
        f"tp={summary_row.get('tp')} "
        f"fp={summary_row.get('fp')} "
        f"fn={summary_row.get('fn')} "
        f"precision={summary_row.get('precision')} "
        f"recall={summary_row.get('recall')} "
        f"f1={summary_row.get('f1')} "
        f"reduction_ratio={summary_row.get('reduction_ratio')}"
    )


def run_validation(args: argparse.Namespace) -> int:
    roots, source_systems, target_systems = _resolve_selected_live_systems(args)
    (
        config,
        method_id,
        representation_family,
        comparison,
        family_config_slug,
        effective_method_config,
        config_hash,
    ) = _build_validation_runtime_context(
        args=args,
        roots=roots,
        source_systems=source_systems,
        target_systems=target_systems,
    )
    # A second, cheap resolution of the same declared settings -- purely
    # informational here, for the dry-run report and the runtime knobs
    # (`max_rows`, `source_chunk_size`) `ValidationRunConfig` does not carry.
    resolved, _ = _resolve_settings(args)
    values = resolved_setting_values(resolved)

    max_rows = _resolve_max_rows_arg(values["max_rows"])

    if args.dry_run:
        report_resolved_settings(
            "validate_clustering",
            resolved,
            source_systems=source_systems,
            target_systems=target_systems,
            method_id=method_id,
            prepare_only=bool(args.prepare_only),
            output_dir=str(config.output_dir),
        )
        return 0

    if bool(args.prepare_only):
        return _run_prepare_only_mode(
            config=config,
            force=bool(args.force),
            max_rows=max_rows,
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = config.output_dir / "progress.jsonl"
    status_path = config.output_dir / "run_status.json"
    manifest_path = config.output_dir / "run_manifest.json"

    _write_run_manifest(
        config=config,
        manifest_path=manifest_path,
        method_id=method_id,
        representation_family=representation_family,
        comparison=comparison,
        family_config_slug=family_config_slug,
        config_hash=config_hash,
        effective_method_config=effective_method_config,
    )
    _write_placeholder_artifacts(config)
    _write_running_status(config=config, status_path=status_path)

    run_started = time.perf_counter()
    source_chunk_size = max(1, int(values["source_chunk_size"]))

    print(
        "[info] artifact-write-mode incremental+final "
        f"(incremental checkpoints: {config.output_dir / '_incremental'})"
    )

    emit_progress = _build_progress_emitter(progress_path)

    result_frames = run_validation_matrix(
        config,
        progress_callback=emit_progress,
        source_chunk_size=source_chunk_size,
        prepare_force=bool(args.force),
        runtime_max_rows=max_rows,
        incremental_output_dir=config.output_dir,
    )
    paths = write_validation_artifacts(
        output_dir=config.output_dir,
        source_outcomes=result_frames["source_outcomes"],
        clusters=result_frames["clusters"],
        directional_coverage=result_frames["directional_coverage"],
        shape_metrics=result_frames["cluster_shape"],
        pair_truth_eval=result_frames["pair_truth_eval"],
        exceptions=result_frames["exceptions"],
        run_metrics=result_frames.get("run_metrics"),
    )

    total_elapsed = time.perf_counter() - run_started
    _write_completed_status(
        config=config,
        status_path=status_path,
        total_elapsed=total_elapsed,
        timings=result_frames["timings"].to_dicts(),
    )
    print(f"[validation] completed in {total_elapsed:.1f}s")
    _print_pair_eval_summary(result_frames)
    for key, value in sorted(paths.items()):
        print(f"[validation] {key}: {value}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_validation(args)


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
