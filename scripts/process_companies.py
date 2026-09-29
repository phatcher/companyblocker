from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    OUTPUT_CLEAR,
    OUTPUT_EXTEND,
    OUTPUT_MIGRATE,
    PlannedOutput,
    add_char_whitelist_arg,
    add_chunk_size_arg,
    add_company_col_arg,
    add_country_token_col_arg,
    add_global_token_col_arg,
    add_input_file_arg,
    add_name_col_arg,
    add_root_arg,
    add_run_date_arg,
    add_systems_arg,
    add_token_col_arg,
    add_token_mode_arg,
    add_tokenizer_path_arg,
    add_tokenizer_profile_arg,
    add_tokenizer_scope_arg,
    parse_key_value_pairs,
    parse_optional_bool,
    parse_stage_key_value_args,
    report_output_plan,
    run_reporting_argument_errors,
    supported_codes_epilog,
)
from company_tokenize import (
    SUPPORTED_TRAINERS,
    TokenizerCalculationSpec,
    TrainerOptions,
    coerce_tokenization_request,
    compile_tokenization_request,
    to_portable_path_str,
    tokenize_name,
    tokenize_name_dual,
    tokenize_name_multi,
    validate_trainer_options,
)
from pipeline_runner import execute_acquire as _execute_acquire
from pipeline_runner import execute_canonical as _execute_canonical
from pipeline_runner import execute_cleanse as _execute_cleanse
from pipeline_runner import execute_match as _execute_match
from pipeline_runner import execute_shard as _execute_shard
from pipeline_runner import (
    execute_tokenize_for_system as _execute_tokenize_for_system,
)
from pipeline_runner import resolve_promoted_tokenizer_path

from acquisition.canonical import (
    canonicalize_system_name_rows,
    canonicalize_system_primary_name_override,
    canonicalize_system_shards,
    canonicalize_system_successor_chain,
    derive_system_name_rows,
    finalize_canonical_name_rows,
)
from acquisition.cleanser_orchestrate import cleanse_canonical_view, configure
from acquisition.company_type_registry import get_system_company_type_mapping
from acquisition.constants_pipeline import (
    DEFAULT_PROCESS_STAGES,
    OPTIONAL_PROCESS_STAGES,
)
from acquisition.constants_status import STATUS_RESEARCH_REQUIRED
from acquisition.pipeline import (
    resolve_system_selection,
    run_acquisition,
    validate_requested_systems,
)
from acquisition.plan_registry import get_system_plan
from acquisition.sharding import shard_system_source
from workspace.cleanse_inputs import resolve_input_dir
from workspace.data_file_naming import (
    COMPANION_NAMES,
    companion_data_file_glob,
    primary_data_file_glob,
)
from workspace.data_layout import (
    CANONICAL_LAYER_NAME,
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    TOKENIZED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import (
    is_partitioned_layer,
    partition_values,
    resolve_primary_files,
)
from workspace.roots import WorkspaceRoots, default_workspace_roots

PROCESS_CHOICES = [stage for stage in DEFAULT_PROCESS_STAGES if stage != "match"]
PROCESS_CHOICES_WITH_ACQUIRE = [*OPTIONAL_PROCESS_STAGES, *PROCESS_CHOICES, "match"]


@dataclass
class _ProcessPipelineTotals:
    cleansed_rows: int = 0
    matched_rows: int = 0
    tokenized_rows: int = 0
    tokenized_systems: int = 0


def _run_process_systems(
    *,
    execution_system_codes: list[str],
    system_runner: Callable[[str], tuple[bool, _ProcessPipelineTotals]],
) -> tuple[_ProcessPipelineTotals, list[str]]:
    totals = _ProcessPipelineTotals()
    failed_systems: list[str] = []

    for index, system_code in enumerate(execution_system_codes, start=1):
        print(
            f"[info] [{index}/{len(execution_system_codes)}] Preparing system '{system_code}'"
        )
        stage_failed, system_totals = system_runner(system_code)
        totals.cleansed_rows += system_totals.cleansed_rows
        totals.matched_rows += system_totals.matched_rows
        totals.tokenized_rows += system_totals.tokenized_rows
        totals.tokenized_systems += system_totals.tokenized_systems
        if stage_failed:
            failed_systems.append(system_code)

    return totals, failed_systems


def _emit_process_pipeline_summary(
    *,
    selected_processes: set[str],
    totals: _ProcessPipelineTotals,
    execution_system_count: int,
) -> None:
    if "cleanse" in selected_processes:
        print(f"[info] Cleanse complete. Total rows cleansed: {totals.cleansed_rows:,}")
    if "match" in selected_processes:
        print(
            f"[info] Match complete. Total rows written to matched artifacts: {totals.matched_rows:,}"
        )
    if "tokenize" in selected_processes:
        print(
            "[info] Tokenization complete. "
            f"Systems tokenized: {totals.tokenized_systems}/{execution_system_count}. "
            f"Total rows tokenized: {totals.tokenized_rows:,}"
        )


def _finish_process_pipeline(*, failed_systems: list[str]) -> int:
    if failed_systems:
        print(
            "[error] Process pipeline failed for system(s): "
            + ", ".join(failed_systems)
        )
        return 1

    print("[info] Process pipeline complete")
    return 0


def _parse_tokenizer_specs(
    tokenizer_specs: list[str] | None,
    *,
    project_root: Path,
) -> list[TokenizerCalculationSpec]:
    if not tokenizer_specs:
        return []

    parsed_specs: list[TokenizerCalculationSpec] = []
    seen_labels: set[str] = set()
    seen_cols: set[str] = set()

    for raw_spec in tokenizer_specs:
        fields = parse_key_value_pairs(
            tokens=raw_spec.split(","),
            invalid_message=(
                f"Invalid --tokenizer-spec '{raw_spec}'. Expected comma-separated key=value entries."
            ),
            empty_message=(
                f"Invalid --tokenizer-spec '{raw_spec}'. Empty key or value is not allowed."
            ),
        )

        trainer = fields.get("tokenizer", "wordpiece").strip().lower()
        if trainer not in SUPPORTED_TRAINERS:
            allowed = ", ".join(SUPPORTED_TRAINERS)
            raise ValueError(
                f"Invalid tokenizer '{trainer}' in --tokenizer-spec '{raw_spec}'. Expected one of: {allowed}."
            )

        path_value = fields.get("path")
        col_value = fields.get("col")
        if path_value is None or col_value is None:
            raise ValueError(
                f"Invalid --tokenizer-spec '{raw_spec}'. Required keys are path and col."
            )

        tokenizer_path = Path(path_value)
        if not tokenizer_path.is_absolute():
            tokenizer_path = (project_root / tokenizer_path).resolve()

        noise_words_path_value = fields.get("noise_words_path", fields.get("stop_path"))
        noise_words_path: Path | None = None
        if noise_words_path_value is not None:
            noise_words_path = Path(noise_words_path_value)
            if not noise_words_path.is_absolute():
                noise_words_path = (project_root / noise_words_path).resolve()
        noise_words_profile = fields.get("noise_words_profile", "none").strip().lower()
        noise_words_set_kind = (
            fields.get("noise_words_set_kind", "combined").strip().lower()
        )

        token_col = col_value.strip()
        label = fields.get("label", token_col).strip()

        if not label:
            raise ValueError(
                f"Invalid --tokenizer-spec '{raw_spec}'. label cannot be empty."
            )
        if not token_col:
            raise ValueError(
                f"Invalid --tokenizer-spec '{raw_spec}'. col cannot be empty."
            )
        if label in seen_labels:
            raise ValueError(f"Duplicate tokenizer spec label '{label}'.")
        if token_col in seen_cols:
            raise ValueError(f"Duplicate tokenizer spec output column '{token_col}'.")

        seen_labels.add(label)
        seen_cols.add(token_col)
        parsed_specs.append(
            TokenizerCalculationSpec(
                token_col=token_col,
                tokenizer_path=tokenizer_path,
                trainer=trainer,
                label=label,
                noise_words_path=noise_words_path,
                noise_words_profile=noise_words_profile,
                noise_words_set_kind=noise_words_set_kind,
            )
        )
    return parsed_specs


def _write_tokenizer_specs_sidecar(
    *,
    tokenized_dir: Path,
    tokenizer_specs: list[TokenizerCalculationSpec],
    name_col: str,
    noise_words_profile: str,
    preprocess_profile: str,
    noise_words_set_kind: str,
    project_root: Path | None = None,
) -> None:
    if not tokenizer_specs:
        return

    compiled_request = compile_tokenization_request(
        coerce_tokenization_request(
            tokenizer_specs=tokenizer_specs,
            name_col=name_col,
            noise_words_profile=noise_words_profile,
            preprocess_profile=preprocess_profile,
            noise_words_set_kind=noise_words_set_kind,
        )
    )

    payload = {
        "schema_version": "2",
        "request": {
            "name_col": compiled_request.name_col,
            "noise_words_profile": compiled_request.noise_words_profile,
            "preprocess_profile": compiled_request.preprocess_profile,
            "noise_words_set_kind": compiled_request.noise_words_set_kind,
        },
        "tokenizer_specs": [
            {
                "label": spec.label,
                "tokenizer": spec.trainer,
                "token_col": spec.token_col,
                "tokenizer_path": to_portable_path_str(
                    spec.tokenizer_path, project_root=project_root
                )
                if spec.tokenizer_path is not None
                else "",
                "noise_words_path": to_portable_path_str(
                    spec.noise_words_path, project_root=project_root
                )
                if spec.noise_words_path is not None
                else "",
                "noise_words_profile": spec.noise_words_profile,
                "noise_words_set_kind": spec.noise_words_set_kind,
            }
            for spec in compiled_request.tokenizer_calculations
        ],
    }
    sidecar_path = tokenized_dir / "tokenizer_specs.json"
    sidecar_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_shard_or_canonical_stage(
    *,
    stage: str,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    input_file: str | None,
    force: bool,
    chunk_size: int,
    sidecar: bool,
    allow_research: bool,
    stage_options: dict[str, dict[str, object]],
) -> bool:
    if stage not in {"shard", "canonical"}:
        raise ValueError(f"Unsupported stage for shared executor: {stage}")

    if not force and _is_stage_up_to_date(
        stage=stage,
        system_code=system_code,
        roots=roots,
        run_date=run_date,
        input_file=input_file,
    ):
        print(f"[info] [{system_code}] Stage {stage} up-to-date; skipping")
        return False

    if stage == "shard":
        effective_stage_options = stage_options
        if force:
            # `--force` alone (no `--additional-args shard.force=...`) is the
            # documented way to make a wikidata re-run re-extract a stale
            # projected artifact. Threaded in here rather than
            # at CLI parse time so `--additional-args shard.force=true`
            # keeps working as an explicit alternative, and only the shard
            # stage's own options are touched.
            effective_stage_options = dict(stage_options)
            shard_options = dict(effective_stage_options.get("shard") or {})
            shard_options["force"] = True
            effective_stage_options["shard"] = shard_options

        stage_failed, _ = _execute_shard(
            system_code=system_code,
            roots=roots,
            run_date=run_date,
            chunk_size=chunk_size,
            sidecar=sidecar,
            allow_research=allow_research,
            stage_options=effective_stage_options,
            shard_system_source_fn=shard_system_source,
        )
        return stage_failed

    stage_failed, _ = _execute_canonical(
        system_code=system_code,
        roots=roots,
        run_date=run_date,
        chunk_size=chunk_size,
        allow_research=allow_research,
        derive_system_name_rows_fn=derive_system_name_rows,
        canonicalize_system_shards_fn=canonicalize_system_shards,
        canonicalize_system_name_rows_fn=canonicalize_system_name_rows,
        canonicalize_system_primary_name_override_fn=canonicalize_system_primary_name_override,
        canonicalize_system_successor_chain_fn=canonicalize_system_successor_chain,
        finalize_canonical_name_rows_fn=finalize_canonical_name_rows,
    )
    return stage_failed


def _resolve_effective_and_tokens(
    *,
    system_plan,
) -> tuple[str, ...] | None:
    if getattr(system_plan, "and_tokens", None):
        cleaned = tuple(
            token.strip().upper()
            for token in system_plan.and_tokens
            if token and token.strip()
        )
        if cleaned:
            return cleaned

    # None means "use package defaults" (AND/ET/UND). Empty tuple would disable replacement.
    return None


def _parse_additional_args(
    additional_args: list[list[str]] | None,
) -> dict[str, dict[str, object]]:
    error_message = (
        "Invalid --additional-args token '{token}'; expected stage.key=value"
    )
    return parse_stage_key_value_args(
        additional_args,
        invalid_message=error_message,
        empty_message=error_message,
        parse_group=lambda group: parse_key_value_pairs(
            tokens=group,
            invalid_message=error_message,
            empty_message=error_message,
        ),
    )


def _resolve_match_target_system(
    *, stage_options: dict[str, dict[str, object]]
) -> str | None:
    match_options = stage_options.get("match", {})
    raw_value = match_options.get("target-system", match_options.get("target_system"))
    if raw_value is None:
        return None
    resolved = str(raw_value).strip().lower()
    return resolved or None


def _latest_snapshot_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    snapshot_dirs = sorted(path for path in root.glob("*") if path.is_dir())
    if not snapshot_dirs:
        return None
    return snapshot_dirs[-1]


def _stage_files(
    *,
    root: Path,
    pattern: str,
    exclude_names: set[str] | None = None,
) -> list[Path]:
    if not root.exists():
        return []
    files = [path for path in root.glob(pattern) if path.is_file()]
    if exclude_names:
        files = [path for path in files if path.name not in exclude_names]
    return sorted(files)


def _outputs_are_fresh(*, input_files: list[Path], output_files: list[Path]) -> bool:
    if not input_files or not output_files:
        return False
    latest_input_mtime = max(path.stat().st_mtime for path in input_files)
    oldest_output_mtime = min(path.stat().st_mtime for path in output_files)
    return oldest_output_mtime >= latest_input_mtime


def _resolve_snapshot_name(*, root: Path, run_date: str | None) -> str | None:
    if run_date is not None:
        return run_date
    latest = _latest_snapshot_dir(root)
    return None if latest is None else latest.name


def _is_stage_up_to_date(
    *,
    stage: str,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    input_file: str | None,
) -> bool:
    if stage == "shard" and system_code == "wikidata":
        return False

    source_root = system_layer_dir(roots, system_code, layer="source")
    acquire_root = system_layer_dir(roots, system_code, layer="acquire")
    canonical_root = system_layer_dir(roots, system_code, layer=CANONICAL_LAYER_NAME)

    if stage == "shard":
        snapshot_name = _resolve_snapshot_name(root=acquire_root, run_date=run_date)
        if snapshot_name is None:
            return False

        acquire_dir = acquire_root / snapshot_name
        source_dir = source_root / snapshot_name
        input_files = _stage_files(root=acquire_dir, pattern="*")
        output_files = _stage_files(
            root=source_dir, pattern=primary_data_file_glob(system_code)
        )
        return _outputs_are_fresh(input_files=input_files, output_files=output_files)

    if stage == "canonical":
        snapshot_name = _resolve_snapshot_name(root=source_root, run_date=run_date)
        if snapshot_name is None:
            return False

        source_dir = source_root / snapshot_name
        canonical_dir = canonical_root / snapshot_name
        input_files = _stage_files(
            root=source_dir, pattern=primary_data_file_glob(system_code)
        )
        # The entity view specifically -- a `*.parquet` glob at the snapshot
        # top level sees only `sample.parquet` and the name-variant sidecar
        # once the entity view is partitioned, and would read a
        # snapshot with no entity output at all as up-to-date.
        output_files = resolve_primary_files(
            canonical_dir, system_code=system_code
        ) + _stage_files(
            root=canonical_dir,
            pattern=companion_data_file_glob(
                system_code=system_code, family=COMPANION_NAMES
            ),
        )
        return _outputs_are_fresh(input_files=input_files, output_files=output_files)

    if stage == "cleanse":
        try:
            canonical_dir = resolve_input_dir(
                roots=roots, run_date=run_date, system=system_code
            )
        except (FileNotFoundError, ValueError):
            return False

        # cleansed/cleansed_merge collapsed into one directory --
        # the merged view lives under cleansed_root's `primary/` family (or,
        # for a pre-family-split system, cleansed_root's own top level),
        # flat {code}-NNN.parquet or jurisdiction_code=*/ partitioned; its
        # chunks/ subdirectory (raw, not-yet-merged output) must never count
        # as "output" here, so this can't be a single blind **/*.parquet glob.
        # resolve_primary_files already knows both locations and both shapes,
        # and never returns sample.parquet or anything under chunks/.
        cleansed_root = system_layer_dir(roots, system_code, layer=CLEANSED_LAYER_NAME)

        def _cleansed_view_output_files() -> list[Path]:
            return resolve_primary_files(cleansed_root, system_code=system_code)

        if input_file is None:
            input_files = resolve_primary_files(canonical_dir, system_code=system_code)
            output_files = _cleansed_view_output_files()
        else:
            input_path = canonical_dir / input_file
            input_files = [input_path] if input_path.exists() else []
            output_files = _cleansed_view_output_files()

        return _outputs_are_fresh(input_files=input_files, output_files=output_files)

    if stage == "match":
        return False

    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the post-acquisition processing pipeline for one or more systems: "
            "shard, canonicalize, cleanse, match, and tokenize."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=supported_codes_epilog(),
    )
    add_systems_arg(parser, action="process")
    parser.add_argument(
        "--processes",
        nargs="+",
        choices=PROCESS_CHOICES_WITH_ACQUIRE,
        default=PROCESS_CHOICES,
        help=f"Pipeline stages to run. Defaults to: {' '.join(PROCESS_CHOICES)}.",
    )
    add_run_date_arg(
        parser,
        subject="Acquisition/canonical run date",
        default_behavior="latest_supported_snapshots",
    )
    add_root_arg(parser, help_text="Project root containing data/ and artifacts/.")
    add_chunk_size_arg(parser, help_text="Target rows per shard/canonical file.")
    add_company_col_arg(parser)
    add_input_file_arg(parser)
    add_char_whitelist_arg(parser)
    add_name_col_arg(parser)
    add_token_mode_arg(parser)
    add_token_col_arg(parser)
    add_country_token_col_arg(parser)
    add_global_token_col_arg(parser)
    add_tokenizer_path_arg(parser)
    add_tokenizer_scope_arg(parser)
    add_tokenizer_profile_arg(parser)
    parser.add_argument(
        "--tokenizer-spec",
        action="append",
        default=[],
        metavar="KEY=VALUE,...",
        help=(
            "Tokenizer comparison spec for tokenize stage. Repeat for multiple tokenizers. "
            "Required keys: path, col. Optional keys: tokenizer, label, noise_words_path (or stop_path), "
            "noise_words_profile, noise_words_set_kind. "
            "Example: --tokenizer-spec tokenizer=sentencepiece,path=artifacts/tokenizers/work/ie/sentencepiece_bpe/model.model,col=tokens_sp_bpe,label=sp_bpe,noise_words_path=artifacts/tokenizers/ie/noise_words.json,noise_words_profile=aggressive"
        ),
    )
    parser.add_argument(
        "--noise-words-path",
        default=None,
        help=(
            "Optional noise-word artifact path used by tokenize stage. "
            "If omitted, packaged global defaults are used."
        ),
    )
    parser.add_argument(
        "--noise-words-profile",
        default="none",
        help=(
            "Noise words removed before tokenizing: none (the default) removes no "
            "words, a profile name removes that profile's words from the packaged "
            "list or from --noise-words-path."
        ),
    )
    parser.add_argument(
        "--preprocess-profile",
        default="default",
        help=(
            "What is removed from --name-col before it is tokenized, named as a "
            "blocking run names it: default removes the company type, "
            "default|-company_type keeps it. Every profile normalises."
        ),
    )
    parser.add_argument(
        "--noise-words-set-kind",
        default="combined",
        help="Noise-word set kind within the selected profile (for example: combined, tokens, seed_legal).",
    )
    parser.add_argument(
        "--trimmed-name-col",
        default=None,
        help="Optional output column to persist pre-tokenize trimmed names.",
    )
    parser.add_argument(
        "--additional-args",
        nargs="+",
        action="append",
        default=[],
        metavar="STAGE.KEY=VALUE",
        help="Stage-specific additional arguments in space-separated stage.key=value form.",
    )
    parser.add_argument(
        "--allow-research",
        action="store_true",
        help=(
            "Allow acquisition for systems marked research_required. "
            "When used with the acquire stage, process_companies stops after acquisition."
        ),
    )
    parser.add_argument(
        "--sidecar",
        nargs="?",
        const=True,
        type=lambda value: parse_optional_bool(value, argument_name="--sidecar"),
        default=True,
        help=(
            "Enable sidecar shard outputs (default: true). "
            "Use --sidecar=false for wide-only shard output."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force selected stages to run even when outputs appear up to date.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report what each selected stage would do for each system -- "
            "skip-vs-run decision, resolved input/output directories, and "
            "what existing on-disk output would be cleared or migrated -- "
            "without executing any stage or touching the filesystem."
        ),
    )
    return parser


def _count_parquet_files(directory: Path, *, pattern: str = "*.parquet") -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.glob(pattern) if path.is_file())


def _report_wikidata_prepare_reuse(
    *, roots: WorkspaceRoots, snapshot_name: str | None, force: bool, prefix: str
) -> None:
    """The 3 final wikidata-*.parquet shard files always get cleared and
    rewritten (see the caller) because _is_stage_up_to_date hardcodes
    shard+wikidata as never up to date -- but that's the cheap materialize
    step, not the expensive multi-hour full-dump extraction. That extraction
    is gated by its own separate, already-resume-aware check inside
    sharding_wikidata.py's prepare_shard_input (has_projected_file /
    has_chunked_output / force), independent of this dry-run's skip/run
    decision. This reports what that check would actually find, read-only,
    so "would RUN" here is never mistaken for "would re-extract the whole
    dump".
    """
    if snapshot_name is None:
        print(
            f"{prefix}   wikidata prepare: cannot resolve an acquire snapshot to "
            "check for existing prepared data"
        )
        return

    prepare_dir = system_layer_dir(roots, "wikidata", layer="prepare") / snapshot_name
    projected_path = prepare_dir / "wikidata-companies.jsonl"
    chunk_dir = prepare_dir / "wikidata-companies.chunks"
    has_projected_file = projected_path.exists() and projected_path.stat().st_size > 0
    has_chunked_output = chunk_dir.exists() and any(chunk_dir.rglob("*.jsonl"))

    if has_projected_file or has_chunked_output:
        found = projected_path if has_projected_file else chunk_dir
        size_note = ""
        if has_projected_file:
            size_gb = projected_path.stat().st_size / (1024**3)
            size_note = f" ({size_gb:.1f} GiB)"
        if force:
            print(
                f"{prefix}   wikidata prepare: found existing prepared data at "
                f"{found}{size_note} -- --force is set, so the expensive "
                "full-dump extraction WOULD RUN and overwrite it (backed up "
                "first)"
            )
        else:
            print(
                f"{prefix}   wikidata prepare: found existing prepared data at {found}{size_note} "
                "-- the expensive full-dump extraction would be SKIPPED and this "
                "reused; only the cheap final-shard materialize step above "
                "actually reruns"
            )
    else:
        print(
            f"{prefix}   wikidata prepare: no existing prepared data found under "
            f"{prepare_dir} -- this WOULD trigger the expensive full-dump "
            "extraction (multi-hour), not just the cheap materialize step"
        )


def _report_dry_run_shard_or_canonical(
    *,
    stage: str,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    input_file: str | None,
    force: bool,
    prefix: str,
) -> None:
    up_to_date = (not force) and _is_stage_up_to_date(
        stage=stage,
        system_code=system_code,
        roots=roots,
        run_date=run_date,
        input_file=input_file,
    )
    if up_to_date:
        print(f"{prefix} Stage {stage}: would SKIP (output already up to date)")
        return

    root_name = "source" if stage == "shard" else "canonical"
    stage_root = system_layer_dir(roots, system_code, layer=root_name)
    snapshot_name = _resolve_snapshot_name(
        root=(
            system_layer_dir(roots, system_code, layer="acquire")
            if stage == "shard"
            else system_layer_dir(roots, system_code, layer="source")
        ),
        run_date=run_date,
    )
    output_dir = stage_root / snapshot_name if snapshot_name else stage_root
    print(f"{prefix} Stage {stage}: would RUN -> output {output_dir}")
    report_output_plan(
        f"{prefix}  ",
        [
            PlannedOutput(
                output_dir,
                OUTPUT_CLEAR,
                pattern="*.parquet",
                note="rewritten from scratch (unconditional, not force-gated)",
            )
        ],
    )
    if stage == "shard" and system_code == "wikidata":
        _report_wikidata_prepare_reuse(
            roots=roots,
            snapshot_name=snapshot_name,
            force=force,
            prefix=prefix,
        )


def _dry_run_cleanse_partition_count(directory: Path) -> int:
    # partition_values checks both the current primary/-family location and
    # the pre-family-split top-level one, so a real, already-migrated
    # cleansed_dir is counted correctly rather than reading as empty.
    return len(partition_values(directory))


def _dry_run_cleanse_flat_count(directory: Path, *, system_code: str) -> int:
    # A partitioned layer's rows are already counted by
    # _dry_run_cleanse_partition_count; only count flat output for a layer
    # that isn't partitioned, so the two never double-count the same run.
    if is_partitioned_layer(directory):
        return 0
    return len(resolve_primary_files(directory, system_code=system_code))


def _report_dry_run_cleanse(
    *,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    input_file: str | None,
    force: bool,
    prefix: str,
) -> None:
    up_to_date = (not force) and _is_stage_up_to_date(
        stage="cleanse",
        system_code=system_code,
        roots=roots,
        run_date=run_date,
        input_file=input_file,
    )
    if up_to_date:
        print(f"{prefix} Stage cleanse: would SKIP (output already up to date)")
        return

    try:
        canonical_dir = resolve_input_dir(
            roots=roots, run_date=run_date, system=system_code
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"{prefix} Stage cleanse: would FAIL to find input -- {exc}")
        return

    cleansed_dir = system_layer_dir(roots, system_code, layer=CLEANSED_LAYER_NAME)
    chunks_dir = cleansed_dir / "chunks"
    chunks_count = _count_parquet_files(chunks_dir)
    partition_count = _dry_run_cleanse_partition_count(cleansed_dir)
    flat_count = _dry_run_cleanse_flat_count(cleansed_dir, system_code=system_code)

    print(
        f"{prefix} Stage cleanse: would RUN -> input {canonical_dir}, output {cleansed_dir}"
    )
    outputs: list[PlannedOutput] = []
    if chunks_count == 0 and flat_count > 0:
        outputs.append(
            PlannedOutput(
                cleansed_dir,
                OUTPUT_MIGRATE,
                into=chunks_dir,
                holds=lambda _path: (
                    f"{flat_count} legacy flat file(s) at its top level"
                ),
                note="moved, not deleted",
            )
        )
    if force and chunks_count:
        outputs.append(
            PlannedOutput(
                chunks_dir,
                OUTPUT_CLEAR,
                pattern="*.parquet",
                note="--force clears existing chunks before re-cleansing",
            )
        )
    if partition_count or flat_count:
        outputs.append(
            PlannedOutput(
                cleansed_dir,
                OUTPUT_CLEAR,
                holds=lambda _path: (
                    f"merged view: {partition_count} partition(s), "
                    f"{flat_count} flat file(s)"
                ),
                note="cleanse rebuilds its merged view on every real run",
            )
        )
    report_output_plan(f"{prefix}  ", outputs)


def _report_dry_run_match(
    *, system_code: str, target_system: str | None, roots: WorkspaceRoots, prefix: str
) -> None:
    matched_dir = system_layer_dir(roots, system_code, layer=MATCHED_LAYER_NAME)
    print(
        f"{prefix} Stage match: would RUN {system_code} -> {target_system}, "
        f"output {matched_dir} "
        "(match has no freshness check -- always executes when selected)"
    )
    report_output_plan(
        f"{prefix}  ",
        [
            PlannedOutput(
                matched_dir,
                OUTPUT_EXTEND,
                pattern="**/*.parquet",
                note="the partition for the jurisdiction the source shares with the "
                "target is replaced, others are kept",
            )
        ],
    )


def _report_dry_run_tokenize(
    *, system_code: str, roots: WorkspaceRoots, prefix: str
) -> None:
    cleansed_dir = system_layer_dir(roots, system_code, layer=CLEANSED_LAYER_NAME)
    tokenized_dir = system_layer_dir(roots, system_code, layer=TOKENIZED_LAYER_NAME)

    print(
        f"{prefix} Stage tokenize: would RUN -> input {cleansed_dir}, "
        f"output {tokenized_dir} (tokenize has no freshness check -- "
        "always fully regenerates when selected)"
    )
    report_output_plan(
        f"{prefix}  ",
        [
            PlannedOutput(
                tokenized_dir,
                OUTPUT_CLEAR,
                pattern="**/*.parquet",
                note="regenerated from scratch",
            ),
        ],
    )


def _report_dry_run_stage(
    *,
    stage: str,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    input_file: str | None,
    force: bool,
    match_target_system: str | None,
) -> None:
    """Read-only report of what `stage` would do for `system_code` -- no
    filesystem writes, no stage execution. Mirrors the real skip/run
    decisions and cleanup targets in _is_stage_up_to_date /
    _execute_non_train_strict_stage / execute_cleanse / prepare_tokenized_output_dir  # noqa: E501
    without calling any of them.
    """
    prefix = f"[dry-run] [{system_code}]"

    if stage in {"shard", "canonical"}:
        _report_dry_run_shard_or_canonical(
            stage=stage,
            system_code=system_code,
            roots=roots,
            run_date=run_date,
            input_file=input_file,
            force=force,
            prefix=prefix,
        )
        return

    if stage == "cleanse":
        _report_dry_run_cleanse(
            system_code=system_code,
            roots=roots,
            run_date=run_date,
            input_file=input_file,
            force=force,
            prefix=prefix,
        )
        return

    if stage == "match":
        _report_dry_run_match(
            system_code=system_code,
            target_system=match_target_system,
            roots=roots,
            prefix=prefix,
        )
        return

    if stage == "tokenize":
        _report_dry_run_tokenize(system_code=system_code, roots=roots, prefix=prefix)
        return

    print(f"{prefix} Stage {stage}: no dry-run reporting implemented, would be a no-op")


def _execute_non_train_strict_stage(
    *,
    stage: str,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    chunk_size: int,
    sidecar: bool,
    allow_research: bool,
    stage_options: dict[str, dict[str, object]],
    input_file: str | None,
    company_col: str,
    char_whitelist: str,
    explicit_tokenizer_path: Path | None,
    tokenizer_scope: str,
    tokenizer_profile: str,
    shared_tokenizer_path: Path | None,
    token_mode: str,
    name_col: str,
    token_col: str,
    country_token_col: str,
    global_token_col: str,
    noise_words_path: Path | None,
    noise_words_profile: str,
    preprocess_profile: str,
    noise_words_set_kind: str,
    trimmed_name_col: str | None,
    tokenizer_specs: list[TokenizerCalculationSpec],
    trainer: str,
    force: bool,
    match_target_system: str | None,
    dry_run: bool = False,
) -> tuple[bool, dict[str, int]]:
    deltas = {
        "cleansed_rows": 0,
        "matched_rows": 0,
        "tokenized_rows": 0,
        "tokenized_systems": 0,
    }

    if dry_run:
        _report_dry_run_stage(
            stage=stage,
            system_code=system_code,
            roots=roots,
            run_date=run_date,
            input_file=input_file,
            force=force,
            match_target_system=match_target_system,
        )
        return False, deltas

    if stage in {"shard", "canonical"}:
        stage_failed = _run_shard_or_canonical_stage(
            stage=stage,
            system_code=system_code,
            roots=roots,
            run_date=run_date,
            input_file=input_file,
            force=force,
            chunk_size=chunk_size,
            sidecar=sidecar,
            allow_research=allow_research,
            stage_options=stage_options,
        )
        return stage_failed, deltas

    if stage == "cleanse":
        if not force and _is_stage_up_to_date(
            stage="cleanse",
            system_code=system_code,
            roots=roots,
            run_date=run_date,
            input_file=input_file,
        ):
            print(f"[info] [{system_code}] Stage cleanse up-to-date; skipping")
            return False, deltas

        stage_failed, cleansed_rows = _execute_cleanse(
            system_code=system_code,
            roots=roots,
            run_date=run_date,
            input_file=input_file,
            company_col=company_col,
            char_whitelist=char_whitelist,
            resolve_effective_and_tokens=_resolve_effective_and_tokens,
            configure_fn=configure,
            resolve_input_dir_fn=resolve_input_dir,
            get_system_plan_fn=get_system_plan,
            get_system_company_type_mapping_fn=get_system_company_type_mapping,
            cleanse_canonical_view_fn=cleanse_canonical_view,
        )
        deltas["cleansed_rows"] = cleansed_rows
        return stage_failed, deltas

    if stage == "match":
        stage_failed, matched_rows = _execute_match(
            system_code=system_code,
            roots=roots,
            target_system=match_target_system,
        )
        deltas["matched_rows"] = matched_rows
        return stage_failed, deltas

    if stage == "tokenize":
        stage_failed, rows = _execute_tokenize_for_system(
            system_code=system_code,
            roots=roots,
            input_file=input_file,
            explicit_tokenizer_path=explicit_tokenizer_path,
            tokenizer_scope=tokenizer_scope,
            tokenizer_profile=tokenizer_profile,
            shared_tokenizer_path=shared_tokenizer_path,
            token_mode=token_mode,
            name_col=name_col,
            token_col=token_col,
            country_token_col=country_token_col,
            global_token_col=global_token_col,
            noise_words_path=noise_words_path,
            noise_words_profile=noise_words_profile,
            preprocess_profile=preprocess_profile,
            noise_words_set_kind=noise_words_set_kind,
            trimmed_name_col=trimmed_name_col,
            tokenizer_specs=tokenizer_specs,
            trainer=trainer,
            resolve_tokenizer_path_fn=resolve_promoted_tokenizer_path,
            tokenize_name_fn=tokenize_name,
            tokenize_name_dual_fn=tokenize_name_dual,
            tokenize_name_multi_fn=tokenize_name_multi,
            write_tokenizer_specs_sidecar_fn=_write_tokenizer_specs_sidecar,
        )
        if not stage_failed:
            deltas["tokenized_rows"] = rows
            deltas["tokenized_systems"] = 1
        return stage_failed, deltas

    return False, deltas


def _run_non_train_strict_flow(
    *,
    execution_system_codes: list[str],
    selected_process_list: list[str],
    selected_processes: set[str],
    roots: WorkspaceRoots,
    run_date: str | None,
    chunk_size: int,
    sidecar: bool,
    allow_research: bool,
    stage_options: dict[str, dict[str, object]],
    input_file: str | None,
    company_col: str,
    char_whitelist: str,
    explicit_tokenizer_path: Path | None,
    tokenizer_scope: str,
    tokenizer_profile: str,
    shared_tokenizer_path: Path | None,
    token_mode: str,
    name_col: str,
    token_col: str,
    country_token_col: str,
    global_token_col: str,
    noise_words_path: Path | None,
    noise_words_profile: str,
    preprocess_profile: str,
    noise_words_set_kind: str,
    trimmed_name_col: str | None,
    tokenizer_specs: list[TokenizerCalculationSpec],
    trainer: str,
    force: bool,
    dry_run: bool = False,
) -> int:
    match_target_system = _resolve_match_target_system(stage_options=stage_options)

    def _run_system(system_code: str) -> tuple[bool, _ProcessPipelineTotals]:
        system_totals = _ProcessPipelineTotals()
        system_failed = False

        for stage in selected_process_list:
            stage_failed, deltas = _execute_non_train_strict_stage(
                stage=stage,
                system_code=system_code,
                roots=roots,
                run_date=run_date,
                chunk_size=chunk_size,
                sidecar=sidecar,
                allow_research=allow_research,
                stage_options=stage_options,
                input_file=input_file,
                company_col=company_col,
                char_whitelist=char_whitelist,
                explicit_tokenizer_path=explicit_tokenizer_path,
                tokenizer_scope=tokenizer_scope,
                tokenizer_profile=tokenizer_profile,
                shared_tokenizer_path=shared_tokenizer_path,
                token_mode=token_mode,
                name_col=name_col,
                token_col=token_col,
                country_token_col=country_token_col,
                global_token_col=global_token_col,
                noise_words_path=noise_words_path,
                noise_words_profile=noise_words_profile,
                preprocess_profile=preprocess_profile,
                noise_words_set_kind=noise_words_set_kind,
                trimmed_name_col=trimmed_name_col,
                tokenizer_specs=tokenizer_specs,
                trainer=trainer,
                force=force,
                match_target_system=match_target_system,
                dry_run=dry_run,
            )
            system_totals.cleansed_rows += deltas["cleansed_rows"]
            system_totals.matched_rows += deltas["matched_rows"]
            system_totals.tokenized_rows += deltas["tokenized_rows"]
            system_totals.tokenized_systems += deltas["tokenized_systems"]
            if stage_failed:
                system_failed = True
                print(
                    f"[error] [{system_code}] Stopping remaining stages for this system"
                )
                break

        return system_failed, system_totals

    totals, failed_systems = _run_process_systems(
        execution_system_codes=execution_system_codes,
        system_runner=_run_system,
    )
    _emit_process_pipeline_summary(
        selected_processes=selected_processes,
        totals=totals,
        execution_system_count=len(execution_system_codes),
    )
    return _finish_process_pipeline(failed_systems=failed_systems)


def run_pipeline(
    *,
    systems: str | list[str],
    processes: list[str] | None = None,
    run_date: str | None = None,
    root: str | Path = ".",
    chunk_size: int = 100_000,
    company_col: str = "name",
    input_file: str | None = None,
    char_whitelist: str = r"[^a-z0-9\s!&]",
    name_col: str = "name_cleansed",
    token_mode: str = "single",
    token_col: str = "name_tokens",
    country_token_col: str = "country_tokens",
    global_token_col: str = "global_tokens",
    noise_words_path: str | Path | None = None,
    noise_words_profile: str = "none",
    preprocess_profile: str = "default",
    noise_words_set_kind: str = "combined",
    trimmed_name_col: str | None = None,
    tokenizer_path: str | Path | None = None,
    tokenizer_scope: str = "country",
    tokenizer_profile: str = "default",
    tokenizer_specs: list[str] | None = None,
    train: bool = False,
    global_tokenizer: bool = False,
    sample_fraction: float = 1.0,
    global_balance_mode: str = "equal",
    global_system_target_rows: int = 300000,
    global_system_min_rows: int = 50000,
    global_system_max_rows: int | None = None,
    global_file_min_rows: int = 500,
    global_balance_seed: int = 42,
    vocab_size: int | None = None,
    trainer: str = "wordpiece",
    min_frequency: int = 1,
    tokenizer_encoding: str = "bpe",
    sp_character_coverage: float = 1.0,
    sp_byte_fallback: bool = False,
    additional_args: list[list[str]] | None = None,
    allow_research: bool = False,
    sidecar: bool = True,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    selection = resolve_system_selection(
        systems,
        add_global_for_all=True,
        global_expands_to_all=True,
    )
    validate_requested_systems(selection.requested)
    execution_system_codes = selection.concrete

    if not execution_system_codes:
        raise ValueError("No concrete systems resolved from --systems.")

    stage_options = _parse_additional_args(additional_args)
    selected_process_list = PROCESS_CHOICES if processes is None else processes
    selected_processes = set(selected_process_list)

    # A match is one named source labelled against one named target. Both
    # are refused here, before any stage runs, rather than resolved from
    # what happens to be on disk.
    if "match" in selected_processes:
        if len(execution_system_codes) != 1:
            raise ValueError(
                "The match stage takes exactly one source system in --systems, got "
                f"{', '.join(execution_system_codes)}."
            )
        if _resolve_match_target_system(stage_options=stage_options) is None:
            raise ValueError(
                "The match stage needs its target named: add "
                "--additional-args match.target-system=<target>."
            )

    if train:
        raise ValueError(
            "Training-enabled company pipeline runs are handled by train_tokenizer.py."
        )

    project_root = Path(root).resolve()
    roots = default_workspace_roots(project_root)
    parsed_tokenizer_specs = _parse_tokenizer_specs(
        tokenizer_specs, project_root=project_root
    )
    normalized_trainer = trainer.strip().lower()
    if normalized_trainer not in SUPPORTED_TRAINERS:
        allowed = ", ".join(SUPPORTED_TRAINERS)
        raise ValueError(
            f"Unsupported trainer '{trainer}'. Expected one of: {allowed}."
        )
    if min_frequency <= 0:
        raise ValueError("min_frequency must be greater than zero.")

    trainer_options = TrainerOptions(
        tokenizer_encoding=tokenizer_encoding,
        sp_character_coverage=sp_character_coverage,
        sp_byte_fallback=sp_byte_fallback,
    )
    validate_trainer_options(trainer=normalized_trainer, options=trainer_options)

    tokenizer_scope = "global" if global_tokenizer else tokenizer_scope

    print(
        f"[info] Starting process pipeline for {len(execution_system_codes)} system(s) "
        f"with stages: {', '.join(selected_process_list)}"
    )
    if "all" in selection.requested:
        print(
            "[info] Resolved systems for --systems all: "
            + ", ".join(execution_system_codes)
        )

    if dry_run:
        print(
            "[dry-run] No stage will actually execute; every action below is a report only."
        )

    should_stop_after_acquire = False
    if "acquire" in selected_processes:
        if dry_run:
            print(
                "[dry-run] Stage acquire: would download/refresh source data for "
                + ", ".join(execution_system_codes)
                + " (acquire has no freshness check -- always attempts a refresh "
                "when selected; the acquisition module itself decides per-source "
                "whether a re-download is actually needed)."
            )
        else:
            should_stop_after_acquire = _execute_acquire(
                execution_system_codes=execution_system_codes,
                roots=roots,
                run_date=run_date,
                allow_research=allow_research,
                get_system_plan_fn=get_system_plan,
                research_required_status=STATUS_RESEARCH_REQUIRED,
                run_acquisition_fn=run_acquisition,
            )
    if should_stop_after_acquire:
        return 0

    explicit_tokenizer_path = Path(tokenizer_path).resolve() if tokenizer_path else None
    resolved_noise_words_path = (
        Path(noise_words_path).resolve() if noise_words_path else None
    )
    shared_tokenizer_path: Path | None = explicit_tokenizer_path
    if explicit_tokenizer_path is None and tokenizer_scope == "global" and not train:
        shared_tokenizer_path = resolve_promoted_tokenizer_path(
            roots=roots, scope="global", trainer=normalized_trainer
        )

    return _run_non_train_strict_flow(
        execution_system_codes=execution_system_codes,
        selected_process_list=selected_process_list,
        selected_processes=selected_processes,
        roots=roots,
        run_date=run_date,
        chunk_size=chunk_size,
        sidecar=sidecar,
        allow_research=allow_research,
        stage_options=stage_options,
        input_file=input_file,
        company_col=company_col,
        char_whitelist=char_whitelist,
        explicit_tokenizer_path=explicit_tokenizer_path,
        tokenizer_scope=tokenizer_scope,
        tokenizer_profile=tokenizer_profile,
        shared_tokenizer_path=shared_tokenizer_path,
        token_mode=token_mode,
        name_col=name_col,
        token_col=token_col,
        country_token_col=country_token_col,
        global_token_col=global_token_col,
        noise_words_path=resolved_noise_words_path,
        noise_words_profile=noise_words_profile,
        preprocess_profile=preprocess_profile,
        noise_words_set_kind=noise_words_set_kind,
        trimmed_name_col=trimmed_name_col,
        tokenizer_specs=parsed_tokenizer_specs,
        trainer=normalized_trainer,
        force=force,
        dry_run=dry_run,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return run_pipeline(
        systems=args.systems,
        processes=args.processes,
        run_date=args.run_date,
        root=args.root,
        chunk_size=args.chunk_size,
        company_col=args.company_col,
        input_file=args.input_file,
        char_whitelist=args.char_whitelist,
        name_col=args.name_col,
        token_mode=args.token_mode,
        token_col=args.token_col,
        country_token_col=args.country_token_col,
        global_token_col=args.global_token_col,
        noise_words_path=args.noise_words_path,
        noise_words_profile=args.noise_words_profile,
        preprocess_profile=args.preprocess_profile,
        noise_words_set_kind=args.noise_words_set_kind,
        trimmed_name_col=args.trimmed_name_col,
        tokenizer_path=args.tokenizer_path,
        tokenizer_scope=args.tokenizer_scope,
        tokenizer_profile=args.tokenizer_profile,
        tokenizer_specs=args.tokenizer_spec,
        train=getattr(args, "train", False),
        global_tokenizer=getattr(args, "global_tokenizer", False),
        sample_fraction=getattr(args, "sample_fraction", 1.0),
        global_balance_mode=getattr(args, "global_balance_mode", "equal"),
        global_system_target_rows=getattr(args, "global_system_target_rows", 300000),
        global_system_min_rows=getattr(args, "global_system_min_rows", 50000),
        global_system_max_rows=getattr(args, "global_system_max_rows", None),
        global_file_min_rows=getattr(args, "global_file_min_rows", 500),
        global_balance_seed=getattr(args, "global_balance_seed", 42),
        vocab_size=getattr(args, "vocab_size", None),
        trainer=getattr(args, "tokenizer", "wordpiece"),
        min_frequency=getattr(args, "min_frequency", 1),
        tokenizer_encoding=getattr(args, "tokenizer_encoding", "bpe"),
        sp_character_coverage=getattr(args, "sp_character_coverage", 1.0),
        sp_byte_fallback=getattr(args, "sp_byte_fallback", False),
        additional_args=args.additional_args,
        allow_research=args.allow_research,
        sidecar=args.sidecar,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
