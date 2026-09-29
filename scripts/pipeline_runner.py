from __future__ import annotations

import shutil
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from company_tokenize import tokenizer_directory_files, tokenizer_id

from acquisition.match_ops import materialize_match_uri_artifact
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    MATCHED_LAYER_NAME,
    TOKENIZED_LAYER_NAME,
    system_layer_dir,
)
from workspace.layer_layout import (
    PARTITION_COLUMN,
    primary_family_dir,
    resolve_primary_files,
)
from workspace.pointer import NothingPromotedError, PointerError
from workspace.roots import WorkspaceRoots
from workspace.tokenizer_store import promoted_tokenizer_directory, scope_system


def _run_stage_and_report(
    *,
    system_code: str,
    stage_name: str,
    output_label: str,
    stage_runner,
    handled_exceptions: tuple[type[Exception], ...],
) -> tuple[bool, int]:
    print(f"[info] [{system_code}] Stage {stage_name}")
    try:
        output_paths = stage_runner()
        print(f"[{system_code}] wrote {len(output_paths)} {output_label}")
        return False, 0
    except handled_exceptions as exc:
        print(f"[error] [{system_code}] Stage {stage_name} failed: {exc}")
        return True, 0


def execute_acquire(
    *,
    execution_system_codes: list[str],
    roots: WorkspaceRoots,
    run_date: str | None,
    allow_research: bool,
    get_system_plan_fn,
    research_required_status: str,
    run_acquisition_fn,
) -> bool:
    has_research_targets = any(
        get_system_plan_fn(system_code).status == research_required_status
        for system_code in execution_system_codes
    )
    acquisition_results = run_acquisition_fn(
        systems=execution_system_codes,
        roots=roots,
        run_date=run_date,
        skip_unsupported=True,
        allow_research=allow_research,
        progress=lambda message: print(f"[info] {message}"),
    )
    for result in acquisition_results:
        print(f"[{result.status}] {result.message}")

    if allow_research and has_research_targets:
        print(
            "[info] Stopping after acquire because --allow-research is set; "
            "downstream DBpedia processing is not enabled yet."
        )
        return True

    return False


def execute_shard(
    *,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    chunk_size: int,
    sidecar: bool,
    allow_research: bool,
    stage_options: dict[str, dict[str, object]],
    shard_system_source_fn,
) -> tuple[bool, int]:
    return _run_stage_and_report(
        system_code=system_code,
        stage_name="shard",
        output_label="shard(s)",
        stage_runner=lambda: shard_system_source_fn(
            system_code,
            roots=roots,
            run_date=run_date,
            chunk_size=chunk_size,
            sidecar=sidecar,
            allow_research=allow_research,
            stage_options=stage_options.get("shard"),
            progress=lambda message: print(f"[info] {message}"),
        ),
        handled_exceptions=(FileNotFoundError,),
    )


def execute_canonical(
    *,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    chunk_size: int,
    allow_research: bool,
    derive_system_name_rows_fn,
    canonicalize_system_shards_fn,
    canonicalize_system_name_rows_fn,
    canonicalize_system_primary_name_override_fn,
    canonicalize_system_successor_chain_fn,
    finalize_canonical_name_rows_fn,
) -> tuple[bool, int]:
    # The entity pass (canonicalize_system_shards: reads the main per-entity
    # Shard-stage file, writes the canonical entity view) and the name-row
    # chain (derive_system_name_rows -> canonicalize_system_name_rows) touch
    # disjoint outputs, which the family split makes structural
    # rather than a matter of filename convention: the entity pass owns
    # data/<code>/canonical/<snapshot>/primary/ outright, and the name-row
    # chain owns .../names/ plus its Shard-layer inputs at
    # data/<code>/source/<snapshot>/<code>-names-*.parquet. Both sides do
    # read the same main Shard-stage file, but concurrent reads of the same
    # file are safe. So the two run concurrently here; the three steps after
    # them all read both families and so need both to have finished --
    # canonicalize_system_primary_name_override,
    # canonicalize_system_successor_chain, and finalize_canonical_name_rows,
    # which additionally has to follow the successor chain because that step
    # appends rows to the sidecar it consolidates.
    def _run_entity_pass() -> tuple[bool, int]:
        return _run_stage_and_report(
            system_code=system_code,
            stage_name="canonical",
            output_label="canonical shard(s)",
            stage_runner=lambda: canonicalize_system_shards_fn(
                system_code,
                run_date=run_date,
                roots=roots,
                chunk_size=chunk_size,
                allow_research=allow_research,
            ),
            handled_exceptions=(FileNotFoundError, RuntimeError),
        )

    def _run_name_row_pass() -> tuple[bool, int]:
        # Derives Shard-shaped name-variant rows from already-written shard
        # output for systems whose naming fields survive into the main shard
        # parquet (Wikidata today; GLEIF emits its own during Shard). A
        # no-op for any system with no registered deriver, so this is safe
        # to always run. canonicalize_system_name_rows reads this step's own
        # output, so within this chain it must still run first.
        derive_failed, _ = _run_stage_and_report(
            system_code=system_code,
            stage_name="shard name-row derivation",
            output_label="derived shard name-row file(s)",
            stage_runner=lambda: derive_system_name_rows_fn(
                system_code,
                run_date=run_date,
                roots=roots,
                allow_research=allow_research,
            ),
            handled_exceptions=(FileNotFoundError, RuntimeError),
        )
        if derive_failed:
            return True, 0

        # Maps Shard-stage native name-form rows to the shared name_type
        # vocabulary. A no-op for any system without name_variant_type_map
        # configured, so this is safe to always run.
        return _run_stage_and_report(
            system_code=system_code,
            stage_name="canonical name rows",
            output_label="canonical name-row shard(s)",
            stage_runner=lambda: canonicalize_system_name_rows_fn(
                system_code,
                run_date=run_date,
                roots=roots,
                allow_research=allow_research,
            ),
            handled_exceptions=(FileNotFoundError, RuntimeError),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        entity_future = executor.submit(_run_entity_pass)
        name_row_future = executor.submit(_run_name_row_pass)
        # .result() re-raises any exception the worker thread didn't itself
        # handle (see handled_exceptions above); the executor's own context
        # manager still blocks on shutdown until both submitted tasks finish
        # running before that propagates, so neither pass's real
        # success/failure is ever left unresolved or unreported even when
        # the other one is the one that raises.
        entity_failed, count = entity_future.result()
        name_rows_failed, _ = name_row_future.result()

    if entity_failed or name_rows_failed:
        # count is always 0 here: _run_stage_and_report returns a literal 0
        # on both its success and failure branches (pre-existing, unrelated
        # to this change -- execute_canonical's only caller already discards
        # this second tuple element). Propagating count as-is keeps the
        # return shape identical to the prior strictly-sequential code.
        return True, count

    # Promotes a script-appropriate name-variant row into canonical `name`
    # when the primary name has no Latin-script character. A no-op for any
    # system without primary_name_override_type_priority configured, so
    # this is safe to always run.
    override_failed, _ = _run_stage_and_report(
        system_code=system_code,
        stage_name="canonical primary-name override",
        output_label="canonical shard(s) with an overridden primary name",
        stage_runner=lambda: canonicalize_system_primary_name_override_fn(
            system_code,
            run_date=run_date,
            roots=roots,
            allow_research=allow_research,
        ),
        handled_exceptions=(FileNotFoundError, RuntimeError),
    )
    if override_failed:
        return override_failed, count

    # Folds a predecessor's name onto its successor(s) via GLEIF's
    # SuccessorLEI edges. A no-op without Shard-stage successor sidecar
    # files on disk (only GLEIF produces these today), so this is safe to
    # always run.
    successor_chain_failed, _ = _run_stage_and_report(
        system_code=system_code,
        stage_name="canonical successor chain",
        output_label="canonical shard(s) with folded successor-chain names",
        stage_runner=lambda: canonicalize_system_successor_chain_fn(
            system_code,
            run_date=run_date,
            roots=roots,
            allow_research=allow_research,
        ),
        handled_exceptions=(FileNotFoundError, RuntimeError),
    )
    if successor_chain_failed:
        return successor_chain_failed, count

    # Consolidates the name-variant sidecar, joins each row's
    # jurisdiction_code in from the entity view, and gives every row its own
    # unique system_uri/source_uri identity -- the layer's last
    # write, so no later stage (Match, previously) needs to rewrite it. Must
    # run last: it needs the entity view the first pass writes, and the
    # successor-chain step above appends rows to the very file it
    # consolidates. A no-op for any system with no sidecar, so this is safe
    # to always run.
    finalize_names_failed, _ = _run_stage_and_report(
        system_code=system_code,
        stage_name="canonical name finalization",
        output_label="consolidated canonical name-row file(s)",
        stage_runner=lambda: finalize_canonical_name_rows_fn(
            system_code,
            run_date=run_date,
            roots=roots,
            allow_research=allow_research,
        ),
        handled_exceptions=(FileNotFoundError, RuntimeError),
    )

    return finalize_names_failed, count


def execute_cleanse(
    *,
    system_code: str,
    roots: WorkspaceRoots,
    run_date: str | None,
    input_file: str | None,
    company_col: str,
    char_whitelist: str,
    resolve_effective_and_tokens,
    configure_fn,
    resolve_input_dir_fn,
    get_system_plan_fn,
    get_system_company_type_mapping_fn,
    cleanse_canonical_view_fn,
) -> tuple[bool, int]:
    print(f"[info] [{system_code}] Stage cleanse")
    try:
        cfg = configure_fn(country=system_code, roots=roots)
        # Cleanse has no chunks/ staging pass, so any chunks/ still on disk
        # is an older run's leftovers, not this run's input. It is
        # cleared unconditionally rather than only under --force: leaving it
        # would keep a stale, undeduped copy of the cleansed rows beside the
        # real view, which is exactly the double-count trap
        # perturbation_materializer._read_merged_cleansed_frame documents.
        legacy_chunks_dir = cfg.cleansed_dir / "chunks"
        if input_file is None and legacy_chunks_dir.exists():
            shutil.rmtree(legacy_chunks_dir, ignore_errors=True)
            print(
                f"[info] [{system_code}] Removed legacy cleanse staging "
                f"{legacy_chunks_dir}"
            )
        cfg.cleansed_dir.mkdir(parents=True, exist_ok=True)
        input_dir = resolve_input_dir_fn(
            roots=roots,
            run_date=run_date,
            system=system_code,
        )
        system_plan = get_system_plan_fn(system_code)
        effective_and_tokens = resolve_effective_and_tokens(
            system_plan=system_plan,
        )
        source_company_type_col = system_plan.company_type_column
        personal_owner_markers = getattr(system_plan, "personal_owner_markers", None)
        source_company_type_mapping = get_system_company_type_mapping_fn(
            system_plan.code
        )

        # The system's own declared partition column, resolved the same way
        # canonicalize_system_shards resolves it, so Cleanse can tell a
        # legitimately-flat system from a canonical snapshot that predates
        # the partitioned write.
        plan_resources = getattr(system_plan, "resources", None)
        source_resource = plan_resources[0] if plan_resources else None
        resolve_partition_by = getattr(
            source_resource, "resolve_output_partition_by", None
        )
        partition_by = (
            resolve_partition_by(default_partition_by=PARTITION_COLUMN)
            if callable(resolve_partition_by)
            else PARTITION_COLUMN
        )

        # No company_type_regex/company_type_mapping here: this system's own
        # jurisdiction narrows the packaged rules to that one country's
        # variants, and a name can carry a suffix from anywhere.
        # Leaving both out of cleanse_config_kwargs lets CleanseConfig's own
        # default resolve the full, multi-jurisdiction rule set instead --
        # the one place that decides which rules cleansing uses.
        cleanse_kwargs: dict[str, object] = {
            "input_dir": input_dir,
            "output_dir": cfg.cleansed_dir,
            "system": system_code,
            "partition_by": partition_by,
            "company_col": company_col,
            "and_tokens": effective_and_tokens,
            "char_whitelist": char_whitelist,
            "source_company_type_col": source_company_type_col,
            "source_company_type_mapping": source_company_type_mapping,
        }
        if personal_owner_markers is not None:
            cleanse_kwargs["personal_owner_markers"] = personal_owner_markers
        if input_file is not None:
            cleanse_kwargs["input_file"] = input_file

        # No merge pass follows: Canonical writes the deduped,
        # jurisdiction_code=-partitioned view and this maps it 1:1.
        cleansed_rows = cleanse_canonical_view_fn(
            **cleanse_kwargs,
        )
        print(
            f"[{system_code}] cleansed {cleansed_rows:,} row(s) into {cfg.cleansed_dir}"
        )
        return False, cleansed_rows
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] [{system_code}] Stage cleanse failed: {exc}")
        return True, 0


def execute_match(
    *,
    system_code: str,
    roots: WorkspaceRoots,
    target_system: str | None,
) -> tuple[bool, int]:
    if target_system is None:
        raise ValueError(
            f"[{system_code}] Stage match needs its target system named; a match "
            "is one source labelled against one target."
        )
    print(f"[info] [{system_code}] Stage match -> {target_system}")
    try:
        summary = materialize_match_uri_artifact(
            roots=roots,
            source_system=system_code,
            target_system=target_system,
        )
        output_path = system_layer_dir(roots, system_code, layer=MATCHED_LAYER_NAME)
        print(
            f"[{system_code}] {system_code} -> {summary.target_system} "
            f"({summary.source_country}): wrote {summary.rows_written:,} row(s) with "
            f"{summary.rows_matched:,} matched key(s) to {output_path} "
            f"(files={summary.output_file_count}, source_valid={summary.source_valid_keys:,}/{summary.source_rows_total:,}, "
            f"target_valid={summary.target_valid_keys:,}/{summary.target_rows_total:,}, "
            f"source_dupes={summary.source_duplicate_keys:,}, target_dupes={summary.target_duplicate_keys:,}, "
            f"elapsed={summary.elapsed_seconds:.2f}s)"
        )
        if summary.source_duplicate_keys or summary.target_duplicate_keys:
            print(
                f"[warn] [{system_code}] {summary.source_duplicate_keys:,} source / "
                f"{summary.target_duplicate_keys:,} target ambiguous key(s) excluded "
                "from the join -- see "
                f"{output_path}/_duplicate_keys.parquet"
            )

        # Match no longer touches the name-variant sidecar:
        # `system_uri`/`source_uri` are given to every name row where
        # Canonical writes `.../names/` (`finalize_canonical_name_rows`), so
        # nothing here needs to rewrite that already-written layer. A
        # consumer that needs a name row's `match_uri` resolves it itself by
        # joining the row's `source_uri` against this stage's own
        # `data/<code>/matched/` output.
        return False, summary.rows_written
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] [{system_code}] Stage match failed: {exc}")
        return True, 0


def _has_parquet_input(
    source_dir: Path, *, system_code: str, input_file: str | None
) -> bool:
    # source_dir's own primary/ family (or, for a pre-family-split layer,
    # its own top level) may hold either flat {code}-NNN.parquet files or
    # jurisdiction_code=*/ partitioned output (cleansed/cleansed_merge were
    # collapsed into one directory) -- chunks/ (raw,
    # not-yet-merged content) is deliberately not a valid input here.
    # resolve_primary_files already knows both locations and both shapes.
    if not source_dir.exists():
        return False
    if input_file is None:
        return bool(resolve_primary_files(source_dir, system_code=system_code))
    if (source_dir / input_file).exists():
        return True
    if (primary_family_dir(source_dir) / input_file).exists():
        return True
    return any(source_dir.glob(f"{PARTITION_COLUMN}=*/**/{input_file}"))


def _handle_missing_stage_input(
    *,
    system_code: str,
    on_missing: str,
    warn_message: str,
    error_message: str,
    strict_on_missing: bool,
) -> tuple[bool, int]:
    if on_missing == "warn":
        print(warn_message)
        return False, 0
    if strict_on_missing and on_missing != "error":
        raise ValueError(f"Unsupported on_missing mode: {on_missing}")
    print(error_message)
    return True, 0


def _validate_stage_parquet_input(
    *,
    system_code: str,
    source_dir: Path,
    input_file: str | None,
    on_missing: str,
    stage_name: str,
    input_label: str,
    strict_on_missing: bool,
) -> tuple[bool, int] | None:
    if _has_parquet_input(source_dir, system_code=system_code, input_file=input_file):
        return None
    return _handle_missing_stage_input(
        system_code=system_code,
        on_missing=on_missing,
        warn_message=(
            f"[warn] Skipping '{system_code}' {stage_name}: "
            f"no {input_label} parquet shards found in {source_dir}"
        ),
        error_message=(
            f"[error] [{system_code}] Stage {stage_name} failed: "
            f"no {input_label} parquet shards found in {source_dir}"
        ),
        strict_on_missing=strict_on_missing,
    )


def prepare_tokenized_output_dir(
    *,
    system_code: str,
    tokenized_dir: Path,
    input_file: str | None,
) -> None:
    if tokenized_dir.exists():
        stale_chunks_dir = tokenized_dir / "chunks"
        if stale_chunks_dir.exists():
            shutil.rmtree(stale_chunks_dir, ignore_errors=True)
            print(
                f"[info] [{system_code}] Removed stale scratch chunks from {stale_chunks_dir}"
            )

        # No name-based "merge mode" to key off any more -- a fresh run's
        # actual shape (flat vs jurisdiction_code=*/ partitioned) is decided
        # freshly by the tokenizer from cleansed_dir's current shape, so
        # clear stale output of *either* shape unconditionally rather than
        # guessing which one applies before the run has even started.
        # company_tokenize mirrors its cleansed/ input's own relative path
        # so real tokenized output lands under a primary/ family
        # directory, not directly at tokenized_dir's own top level -- both
        # locations need clearing, the same dual-check is_partitioned_layer
        # already does for a reader.
        if input_file is None:
            removed_count = 0
            primary_dir = primary_family_dir(tokenized_dir)
            if primary_dir.exists():
                shutil.rmtree(primary_dir)
                removed_count += 1
            for partition_dir in tokenized_dir.glob(f"{PARTITION_COLUMN}=*"):
                if partition_dir.is_dir():
                    shutil.rmtree(partition_dir)
                    removed_count += 1
            for stale_file in tokenized_dir.glob(f"{system_code}-*.parquet"):
                stale_file.unlink()
                removed_count += 1
            if removed_count:
                print(
                    f"[info] [{system_code}] Cleared {removed_count} stale tokenized output(s) "
                    f"from {tokenized_dir}"
                )
        else:
            for sample_output in tokenized_dir.rglob(input_file):
                if sample_output.exists():
                    sample_output.unlink()
                    print(
                        f"[info] [{system_code}] Cleared stale tokenized sample: {sample_output}"
                    )

    tokenized_dir.mkdir(parents=True, exist_ok=True)


def _resolve_tokenize_io_dirs(
    *,
    roots: WorkspaceRoots,
    system_code: str,
    input_file: str | None,
) -> tuple[Path, Path]:
    # cleansed/cleansed_merge collapsed into one cleansed/ directory
    # -- there's only one input layer to prefer now, so no more branching on
    # which of two directories has parquet input.
    return (
        system_layer_dir(roots, system_code, layer=CLEANSED_LAYER_NAME),
        system_layer_dir(roots, system_code, layer=TOKENIZED_LAYER_NAME),
    )


def resolve_promoted_tokenizer_path(
    *,
    roots: WorkspaceRoots,
    scope: str,
    system: str | None = None,
    trainer: str = "wordpiece",
) -> Path | None:
    """The promoted tokenizer's model for a scope and tokenizer id, or None when
    nothing is promoted for it, which leaves Tokenize on the packaged default.
    A promoted candidate missing from this machine's store is not that case
    and raises: tokenizing with another tokenizer would be a silent wrong
    result."""
    try:
        directory = promoted_tokenizer_directory(
            roots,
            system=scope_system(scope, system),
            tokenizer_id=tokenizer_id(trainer),
        )
    except NothingPromotedError:
        return None
    return tokenizer_directory_files(directory, trainer=trainer).model


def _resolve_single_tokenizer_path(
    *,
    explicit_tokenizer_path: Path | None,
    tokenizer_scope: str,
    shared_tokenizer_path: Path | None,
    roots: WorkspaceRoots,
    system_code: str,
    tokenizer_profile: str,
    single_trainer: str,
    resolve_tokenizer_path_fn,
) -> Path | None:
    if explicit_tokenizer_path is not None:
        return explicit_tokenizer_path
    if tokenizer_scope == "global":
        return shared_tokenizer_path
    return resolve_tokenizer_path_fn(
        roots=roots,
        scope="country",
        system=system_code,
        trainer=single_trainer,
    )


def _build_dual_tokenize_kwargs(
    *,
    cleansed_dir: Path,
    tokenized_dir: Path,
    source_files: tuple[Path, ...],
    country_tokenizer_path: Path,
    global_tokenizer_path: Path,
    name_col: str,
    country_token_col: str,
    global_token_col: str,
    noise_words_path: Path | None,
    noise_words_profile: str,
    preprocess_profile: str,
    noise_words_set_kind: str,
    trimmed_name_col: str | None,
    input_file: str | None,
    country_trainer: str,
    global_trainer: str,
) -> dict[str, object]:
    dual_tokenize_kwargs: dict[str, object] = {
        "cleansed_dir": cleansed_dir,
        "tokenized_dir": tokenized_dir,
        "source_files": source_files,
        "country_tokenizer_path": country_tokenizer_path,
        "global_tokenizer_path": global_tokenizer_path,
        "name_col": name_col,
        "country_token_col": country_token_col,
        "global_token_col": global_token_col,
        "noise_words_path": noise_words_path,
        "noise_words_profile": noise_words_profile,
        "preprocess_profile": preprocess_profile,
        "noise_words_set_kind": noise_words_set_kind,
        "trimmed_name_col": trimmed_name_col,
        **({"input_file": input_file} if input_file is not None else {}),
    }

    non_default_trainers = {
        "country_trainer": country_trainer,
        "global_trainer": global_trainer,
    }
    for key, trainer_name in non_default_trainers.items():
        if trainer_name != "wordpiece":
            dual_tokenize_kwargs[key] = trainer_name

    return dual_tokenize_kwargs


def execute_tokenize(
    *,
    system_code: str,
    roots: WorkspaceRoots,
    cleansed_dir: Path,
    tokenized_dir: Path,
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
    input_file: str | None,
    tokenizer_specs: list[object],
    single_trainer: str,
    country_trainer: str,
    global_trainer: str,
    allow_auto_dual: bool,
    resolve_tokenizer_path_fn,
    tokenize_name_fn,
    tokenize_name_dual_fn,
    tokenize_name_multi_fn,
) -> int:
    def _spec_value(spec: object, key: str, default: object = "") -> object:
        if isinstance(spec, Mapping):
            return spec.get(key, default)
        return getattr(spec, key, default)

    # Which files under cleansed/ are this system's real data is a workspace
    # fact, so it is resolved here and handed over. company_tokenize is meant
    # to be consumable outside this repository and so cannot know it -- when
    # it was left to guess, a missed shape silently fell back to whatever
    # parquet sat at the root, which for `gb` was a one-row diagnostic file.
    source_files = tuple(resolve_primary_files(cleansed_dir, system_code=system_code))

    if tokenizer_specs:
        rows = tokenize_name_multi_fn(
            cleansed_dir=cleansed_dir,
            tokenized_dir=tokenized_dir,
            source_files=source_files,
            tokenizer_specs=tokenizer_specs,
            name_col=name_col,
            noise_words_path=noise_words_path,
            noise_words_profile=noise_words_profile,
            preprocess_profile=preprocess_profile,
            noise_words_set_kind=noise_words_set_kind,
            trimmed_name_col=trimmed_name_col,
            input_file=input_file,
        )
        col_list = ", ".join(
            str(_spec_value(spec, "token_col", "")) for spec in tokenizer_specs
        )
        print(f"[tokenize] mode=multi columns={col_list}")
        return rows

    if token_mode == "dual":
        country_tokenizer_path = _resolve_single_tokenizer_path(
            explicit_tokenizer_path=explicit_tokenizer_path,
            tokenizer_scope="country",
            shared_tokenizer_path=None,
            roots=roots,
            system_code=system_code,
            tokenizer_profile=tokenizer_profile,
            single_trainer=country_trainer,
            resolve_tokenizer_path_fn=resolve_tokenizer_path_fn,
        )

        global_tokenizer_path = resolve_tokenizer_path_fn(
            roots=roots,
            scope="global",
            trainer=global_trainer,
        )

        if country_tokenizer_path is None:
            raise FileNotFoundError(
                f"Country tokenizer unavailable for '{system_code}' in dual mode. "
                "Train or provide a country tokenizer before running dual tokenization."
            )
        if global_tokenizer_path is None:
            raise FileNotFoundError(
                f"Global tokenizer unavailable for profile '{tokenizer_profile}' in dual mode. "
                "Train or provide a global tokenizer before running dual tokenization."
            )

        dual_tokenize_kwargs = _build_dual_tokenize_kwargs(
            cleansed_dir=cleansed_dir,
            tokenized_dir=tokenized_dir,
            source_files=source_files,
            country_tokenizer_path=country_tokenizer_path,
            global_tokenizer_path=global_tokenizer_path,
            name_col=name_col,
            country_token_col=country_token_col,
            global_token_col=global_token_col,
            noise_words_path=noise_words_path,
            noise_words_profile=noise_words_profile,
            preprocess_profile=preprocess_profile,
            noise_words_set_kind=noise_words_set_kind,
            trimmed_name_col=trimmed_name_col,
            input_file=input_file,
            country_trainer=country_trainer,
            global_trainer=global_trainer,
        )

        rows = tokenize_name_dual_fn(
            **dual_tokenize_kwargs,
        )
        print(
            f"[tokenize] mode=dual country_col={country_token_col} "
            f"global_col={global_token_col}"
        )
        print(
            "[tokenize] country tokenizer: "
            + (
                str(country_tokenizer_path)
                if country_tokenizer_path is not None
                else "package bundled default"
            )
        )
        print(
            "[tokenize] global tokenizer: "
            + (
                str(global_tokenizer_path)
                if global_tokenizer_path is not None
                else "package bundled default"
            )
        )
        return rows

    auto_country_tokenizer_path: Path | None = None
    auto_global_tokenizer_path: Path | None = None
    if allow_auto_dual and explicit_tokenizer_path is None:
        auto_country_tokenizer_path = resolve_tokenizer_path_fn(
            roots=roots,
            scope="country",
            system=system_code,
            trainer=country_trainer,
        )
        auto_global_tokenizer_path = resolve_tokenizer_path_fn(
            roots=roots,
            scope="global",
            trainer=global_trainer,
        )

    if (
        allow_auto_dual
        and explicit_tokenizer_path is None
        and auto_country_tokenizer_path is not None
        and auto_global_tokenizer_path is not None
    ):
        dual_tokenize_kwargs = _build_dual_tokenize_kwargs(
            cleansed_dir=cleansed_dir,
            tokenized_dir=tokenized_dir,
            source_files=source_files,
            country_tokenizer_path=auto_country_tokenizer_path,
            global_tokenizer_path=auto_global_tokenizer_path,
            name_col=name_col,
            country_token_col=country_token_col,
            global_token_col=global_token_col,
            noise_words_path=noise_words_path,
            noise_words_profile=noise_words_profile,
            preprocess_profile=preprocess_profile,
            noise_words_set_kind=noise_words_set_kind,
            trimmed_name_col=trimmed_name_col,
            input_file=input_file,
            country_trainer=country_trainer,
            global_trainer=global_trainer,
        )

        rows = tokenize_name_dual_fn(
            **dual_tokenize_kwargs,
        )
        print(
            f"[tokenize] mode=auto-dual country_col={country_token_col} "
            f"global_col={global_token_col}"
        )
        print(f"[tokenize] country tokenizer: {auto_country_tokenizer_path}")
        print(f"[tokenize] global tokenizer: {auto_global_tokenizer_path}")
        return rows

    resolved_tokenizer_path = _resolve_single_tokenizer_path(
        explicit_tokenizer_path=explicit_tokenizer_path,
        tokenizer_scope=tokenizer_scope,
        shared_tokenizer_path=shared_tokenizer_path,
        roots=roots,
        system_code=system_code,
        tokenizer_profile=tokenizer_profile,
        single_trainer=single_trainer,
        resolve_tokenizer_path_fn=resolve_tokenizer_path_fn,
    )

    tokenize_kwargs: dict[str, object] = {
        "cleansed_dir": cleansed_dir,
        "tokenized_dir": tokenized_dir,
        "source_files": source_files,
        "tokenizer_path": resolved_tokenizer_path,
        "name_col": name_col,
        "token_col": token_col,
        "noise_words_path": noise_words_path,
        "noise_words_profile": noise_words_profile,
        "preprocess_profile": preprocess_profile,
        "noise_words_set_kind": noise_words_set_kind,
        "trimmed_name_col": trimmed_name_col,
    }
    if input_file is not None:
        tokenize_kwargs["input_file"] = input_file
    if single_trainer != "wordpiece":
        tokenize_kwargs["trainer"] = single_trainer

    rows = tokenize_name_fn(
        **tokenize_kwargs,
    )
    if resolved_tokenizer_path is None:
        print("[tokenize] tokenizer: package bundled default")
    else:
        print(f"[tokenize] tokenizer: {resolved_tokenizer_path}")
    return rows


def execute_tokenize_for_system(
    *,
    system_code: str,
    roots: WorkspaceRoots,
    input_file: str | None,
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
    tokenizer_specs: list[object],
    trainer: str,
    resolve_tokenizer_path_fn,
    tokenize_name_fn,
    tokenize_name_dual_fn,
    tokenize_name_multi_fn,
    write_tokenizer_specs_sidecar_fn,
    allow_auto_dual: bool = True,
    on_missing: str = "error",
    catch_errors: bool = True,
) -> tuple[bool, int]:
    print(f"[info] [{system_code}] Stage tokenize")
    cleansed_dir, tokenized_dir = _resolve_tokenize_io_dirs(
        roots=roots,
        system_code=system_code,
        input_file=input_file,
    )
    input_label = CLEANSED_LAYER_NAME
    missing_result = _validate_stage_parquet_input(
        system_code=system_code,
        source_dir=cleansed_dir,
        input_file=input_file,
        on_missing=on_missing,
        stage_name="tokenize",
        input_label=input_label,
        strict_on_missing=False,
    )
    if missing_result is not None:
        return missing_result

    prepare_tokenized_output_dir(
        system_code=system_code,
        tokenized_dir=tokenized_dir,
        input_file=input_file,
    )

    if catch_errors:
        try:
            rows = execute_tokenize(
                system_code=system_code,
                roots=roots,
                cleansed_dir=cleansed_dir,
                tokenized_dir=tokenized_dir,
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
                input_file=input_file,
                tokenizer_specs=tokenizer_specs,
                single_trainer=trainer,
                country_trainer=trainer,
                global_trainer=trainer,
                allow_auto_dual=allow_auto_dual,
                resolve_tokenizer_path_fn=resolve_tokenizer_path_fn,
                tokenize_name_fn=tokenize_name_fn,
                tokenize_name_dual_fn=tokenize_name_dual_fn,
                tokenize_name_multi_fn=tokenize_name_multi_fn,
            )
        except (FileNotFoundError, PointerError, ValueError) as exc:
            print(f"[error] [{system_code}] Stage tokenize failed: {exc}")
            return True, 0
    else:
        rows = execute_tokenize(
            system_code=system_code,
            roots=roots,
            cleansed_dir=cleansed_dir,
            tokenized_dir=tokenized_dir,
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
            input_file=input_file,
            tokenizer_specs=tokenizer_specs,
            single_trainer=trainer,
            country_trainer=trainer,
            global_trainer=trainer,
            allow_auto_dual=allow_auto_dual,
            resolve_tokenizer_path_fn=resolve_tokenizer_path_fn,
            tokenize_name_fn=tokenize_name_fn,
            tokenize_name_dual_fn=tokenize_name_dual_fn,
            tokenize_name_multi_fn=tokenize_name_multi_fn,
        )

    write_tokenizer_specs_sidecar_fn(
        tokenized_dir=tokenized_dir,
        tokenizer_specs=tokenizer_specs,
        name_col=name_col,
        noise_words_profile=noise_words_profile,
        preprocess_profile=preprocess_profile,
        noise_words_set_kind=noise_words_set_kind,
        project_root=roots.checkout,
    )
    print(f"[{system_code}] tokenized {rows:,} row(s) into {tokenized_dir}")
    return False, rows
