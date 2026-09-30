"""Run tfidf/wordpiece/sentencepiece blocking strategies for one or more
source/target pairs and fold the results into one
`blocking.comparison.build_strategy_comparison()` report per pair.

An orchestration script, not a new capability: `write_blocking_report()`
and `build_strategy_comparison()`/`write_strategy_comparison_report()` already
exist and are tested; this just chains three CLI-shaped runs together the way
`scripts/run_blocking.py` runs one, then feeds them into the existing
comparison builder.

This is the canonical entrypoint for the fixed comparison protocol
(`company_vectorize.comparison_protocol`). With no `--source`/
`--target` given, it sweeps `comparison_protocol.FIXED_SLICES`
(`gleif -> gb`/`ie`). Each `(representation, similarity_backend, country)`
cell runs `scripts/run_blocking.py` as its own subprocess, bounded by
`--runtime-budget-seconds` (default `comparison_protocol.RUNTIME_BUDGET_SECONDS`).
A cell that times out or exits non-zero is recorded to a sidecar
`strategy_comparison_failures.jsonl` next to that pair's comparison report,
and the sweep continues rather than aborting -- the existing
`strategy_comparison` artifact and schema carry only completed cells and are
otherwise unchanged.

The default sweep (`--representations` given or not, `--similarity-backend`
omitted) runs `comparison_protocol.MANDATORY_SUITE`'s cells: `--reference-column`
switches it to `REFERENCE_SUITE`'s exact cells instead, run once per slice
rather than on every sweep. Giving `--similarity-backend`
explicitly overrides either set with one backend applied uniformly across
`--representations`, the ad hoc narrower-run shape this script has always
supported. `sbert` is part of the default sweep now that its optional
`sentence-transformers` dependency is declared and synced
(`comparison_protocol.py`'s module docstring); `--sbert-model` still selects
its checkpoint. Running two or three sbert-labelled invocations at different
`--sbert-model` values under distinct `--output-dir`s, and
combining the resulting `strategy_comparison` files with
`blocking.comparison.combine_strategy_comparisons()`, is how an
encoder-checkpoint comparison is built -- this script's own per-invocation
`strategy_comparison` output always labels every `sbert` cell `"sbert"`
regardless of which checkpoint it used, since `label` here is the
representation name, not the resolved checkpoint.

`--cross-cleanser <profile>` runs every representation and country twice,
under `--cleanse-profile` as the baseline and under `<profile>` as the
variant, so the comparison carries both side by side on its `name_transform`
column. `--preprocess-profiles` runs every cell once per preprocess profile
named, `default` alone unless given, carried on that column too, and
`--tokenizer-profile` names which stored tokenizer every cell uses.
A cell whose run has already finished is reused rather than re-run:
each cell's `run_blocking.py` computes its run's keys before scoring and
reuses a finished run holding the same keys, and records where the run's
outputs are for this script to read back. Every cell runs under this
invocation's resolved roots, handed to it through the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 - dev-tooling script; see nosec B603 at its call site
import sys
import tempfile
import time
from pathlib import Path

import _bootstrap
import polars as pl
from cli_common import (
    OUTPUT_CLEAR,
    OUTPUT_EXTEND,
    PlannedOutput,
    add_declared_arguments,
    add_dry_run_arg,
    add_perturbed_source_args,
    add_workspace_roots_args,
    backend_options_from_settings,
    declared_settings,
    report_output_plan,
    report_resolved_settings,
    resolve_declared_settings,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
    setting_was_given,
    source_flags,
    source_system_from_args,
)
from company_tokenize._cli_helper import SETTINGS as TOKENIZE_SETTINGS
from company_tokenize.name_preprocessing import DEFAULT_NAME_PREPROCESSING_PROFILE
from company_vectorize._cli_helper import SETTINGS as VECTORIZE_SETTINGS
from company_vectorize.comparison_protocol import (
    FIXED_SLICES,
    MANDATORY_SUITE,
    REFERENCE_SUITE,
    RUNTIME_BUDGET_SECONDS,
    BaselineSuiteCell,
    CellStatus,
    missing_baseline_suite_cells,
)
from company_vectorize.dense_vocabulary_gate import (
    DEFAULT_DENSE_VOCABULARY_MAX_ROWS,
    DENSE_VOCABULARY_REPRESENTATIONS,
    REFUSAL_MESSAGE_MARKER,
)
from company_vectorize.prefix_filter import PREFIX_FILTER_REPRESENTATIONS

from blocking._cli_helper import SETTINGS as BLOCKING_SETTINGS
from blocking.comparison import (
    StrategyRunEntry,
    build_strategy_comparison,
    diff_name_equality_levels,
    tally_name_equality_transitions,
)
from blocking.contracts import (
    BlockingDatasetDescriptor,
    BlockingRunConfig,
    BlockingStrategyConfig,
    validate_blocking_run_config,
)
from blocking.loader import load_dataset_descriptor
from blocking.reporting import (
    write_name_equality_attribution_report,
    write_strategy_comparison_report,
)
from blocking.run_layout import (
    BlockingComparisonLocation,
    ComparisonArtefact,
    resolve_comparison_location,
    resolve_pairing_dir,
)
from blocking.stored_runs import load_blocking_run_result as _load_blocking_run_result
from blocking.truth import source_truth_for_column
from workspace.roots import (
    ENV_DATA_DIR,
    ENV_OUTPUT_DIR,
    ENV_TEMP_DIR,
    WorkspaceRoots,
)

# Sourced from MANDATORY_SUITE rather than listed again here, so
# this script's default `--representations` sweep cannot drift from the
# protocol's own definition of it. `dict.fromkeys` dedupes while preserving
# MANDATORY_SUITE's order (more than one cell shares a representation over
# different backends -- tfidf/wordpiece/sentencepiece each appear twice).
# `sbert` is included here now that its optional dependency is declared and
# synced (comparison_protocol.py's module docstring) -- it is no longer held
# out of the default sweep the way BASELINE_SUITE held it out.
REPRESENTATIONS = tuple(dict.fromkeys(cell.representation for cell in MANDATORY_SUITE))

_DECLARATIONS = declared_settings(
    BLOCKING_SETTINGS, VECTORIZE_SETTINGS, TOKENIZE_SETTINGS
)

_FORWARDED = "Forwarded to every cell's run_blocking.py."

# Where this script takes each declared setting, and what the setting means to
# a sweep that runs several representations per slice.
_SURFACE: dict[str, dict[str, object]] = {
    "source_system": {
        "flag": "--source",
        "note": "Omit together with --target to sweep the fixed protocol "
        "slices (company_vectorize.comparison_protocol.FIXED_SLICES).",
    },
    "target_system": {
        "flag": "--target",
        "note": "Omit together with --source to sweep the fixed protocol slices.",
    },
    "countries": {"flag": "--countries", "note": "Applied to every slice swept."},
    "top_k": {"flag": "--top-k"},
    "min_similarity": {"flag": "--min-similarity"},
    "similarity_backend": {
        "flag": "--similarity-backend",
        "note": "The default sweep's representations are sparse, so hdbscan needs "
        "--representations narrowed to sbert.",
    },
    "prefix_filter": {"flag": "--prefix-filter"},
    "max_rows": {
        "flag": "--max-rows",
        "note": "Two of the default sweep's three representations are gated by it; "
        "a refused cell is recorded as failed and the sweep continues.",
    },
    "force": {"flag": "--force"},
    "kmeans_clusters": {"flag": "--kmeans-clusters"},
    "min_cluster_size": {"flag": "--min-cluster-size"},
    "source_chunk_size": {"flag": "--source-chunk-size"},
    "name_transform": {"flag": "--name-transform", "note": _FORWARDED},
    "cleanse_profile": {"flag": "--cleanse-profile", "note": _FORWARDED},
    "tokenizer_profile": {"flag": "--tokenizer-profile", "note": _FORWARDED},
    "match_col": {"flag": "--match-col", "note": _FORWARDED},
    "sbert_model_name": {
        "flag": "--sbert-model",
        "note": "Forwarded to every sbert cell's run_blocking.py.",
    },
}


def _tokenizer_for(representation: str) -> str:
    """The `tokenizer.name` value a cell's tokenizer-backed representation
    needs -- `wordpiece`/`sentencepiece` name themselves, `tfidf` and `sbert`
    both fall back to `wordpiece` (`tfidf` because any trained tokenizer
    works equally for it, `sbert` because it has no tokenizer stage at all
    and the fallback is simply never read for it).
    """
    if representation in ("wordpiece", "sentencepiece"):
        return representation
    return "wordpiece"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tfidf/wordpiece/sentencepiece blocking strategies for one "
        "or more source/target pairs and write one strategy_comparison report per "
        "pair. Defaults to the fixed comparison protocol's gb/ie slices when "
        "--source/--target are omitted."
    )
    add_declared_arguments(parser, _DECLARATIONS, _SURFACE)
    add_perturbed_source_args(parser)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=REPRESENTATIONS,
        default=list(REPRESENTATIONS),
        help=(
            "Representations to compare, in order. Defaults to all of "
            f"{'/'.join(REPRESENTATIONS)} (the mandatory baseline suite, "
            "sbert included now that its optional dependency is declared and "
            "synced). Narrowing it no longer discards earlier work on a later "
            "refusal (each cell runs in its own subprocess and a refusal "
            "is recorded as a failed cell, not an abort), but still limits which "
            "cells run at all -- see --sbert-model for sbert's own option."
        ),
    )
    parser.add_argument(
        "--reference-column",
        action="store_true",
        default=False,
        help=(
            "Run REFERENCE_SUITE's exact cells instead of MANDATORY_SUITE's "
            "sub-linear ones -- the reference each mandatory cell's recall loss "
            "is priced against. A named option rather than part of the "
            "default sweep, since it is the expensive part and is meant to run "
            "once per slice. sklearn's exact prefix filter defaults on "
            "for this option's sparse cells unless --prefix-filter is given "
            "explicitly."
        ),
    )
    parser.add_argument(
        "--runtime-budget-seconds",
        type=float,
        default=RUNTIME_BUDGET_SECONDS,
        help="Per-(strategy, country)-cell wall-clock budget enforced via "
        "subprocess timeout. A cell that exceeds it is killed, recorded "
        "as 'timeout' in that pair's strategy_comparison_failures.jsonl, and the "
        f"sweep continues. Defaults to comparison_protocol.RUNTIME_BUDGET_SECONDS "
        f"({RUNTIME_BUDGET_SECONDS:.0f}s).",
    )
    parser.add_argument(
        "--cross-cleanser",
        default=None,
        metavar="PROFILE",
        help="Run every representation and country again under this cleanse "
        "profile, beside the --cleanse-profile baseline, so the comparison "
        "reads what the cleanser moved. Must differ from --cleanse-profile.",
    )
    parser.add_argument(
        "--preprocess-profiles",
        nargs="+",
        default=[DEFAULT_NAME_PREPROCESSING_PROFILE],
        metavar="PROFILE",
        help="Name preprocessing profiles every cell runs under, one run each, "
        "side by side in the comparison's name_transform column: default "
        "removes the company type, default|-company_type keeps it. "
        f"(default: {DEFAULT_NAME_PREPROCESSING_PROFILE})",
    )
    # The one profile a cell runs under, set per cell by
    # `_with_preprocess_profile()`.
    parser.set_defaults(preprocess_profile=DEFAULT_NAME_PREPROCESSING_PROFILE)
    add_workspace_roots_args(parser)
    add_dry_run_arg(
        parser,
        help_text=(
            "Resolve which slices, representations and countries this run "
            "would sweep and report them, without running any cell's "
            "run_blocking.py subprocess."
        ),
    )
    return parser


def _resolve_slices(
    args: argparse.Namespace, roots: WorkspaceRoots
) -> list[tuple[str, str]]:
    """Resolve which `(source_system, target_system)` pairs this run sweeps.

    An explicit pair (`--source` and `--target` both given)
    always wins. Omitting both defaults to `FIXED_SLICES`. Giving only one of
    the two is rejected rather than silently guessed at.
    """

    if args.source_system and args.target_system:
        return [(source_system_from_args(args, roots), args.target_system)]
    if args.source_system or args.target_system:
        raise ValueError(
            "--source and --target must both be given, or both "
            "omitted to sweep the fixed protocol slices "
            f"({', '.join(f'{source}->{target}' for source, target in FIXED_SLICES)})"
        )
    return list(FIXED_SLICES)


def _resolve_cleanse_profiles(args: argparse.Namespace) -> tuple[str, ...]:
    """The cleanse profiles every cell runs under: `--cleanse-profile`, the
    baseline, then `--cross-cleanser` when given.

    A cross profile naming the baseline is refused, since its runs would be
    the baseline's runs again.
    """

    baseline = str(args.cleanse_profile).strip()
    if args.cross_cleanser is None:
        return (baseline,)
    cross = str(args.cross_cleanser).strip()
    if cross == baseline:
        raise ValueError(
            f"--cross-cleanser {cross!r} is the --cleanse-profile baseline; name "
            "a different profile to compare against"
        )
    return (baseline, cross)


def _with_cleanse_profile(
    args: argparse.Namespace, cleanse_profile: str
) -> argparse.Namespace:
    """`args` with `cleanse_profile` in place of `--cleanse-profile`, so one
    cell's configuration and argv are both built under that profile."""
    return argparse.Namespace(**{**vars(args), "cleanse_profile": cleanse_profile})


def _resolve_preprocess_profiles(args: argparse.Namespace) -> tuple[str, ...]:
    """The preprocess profiles every cell runs under, in the order given, a
    profile named twice run once."""
    return tuple(dict.fromkeys(str(p).strip() for p in args.preprocess_profiles))


def _with_preprocess_profile(
    args: argparse.Namespace, preprocess_profile: str
) -> argparse.Namespace:
    """`args` with `preprocess_profile` as the one profile a cell runs under,
    so its configuration and argv are both built under it."""
    return argparse.Namespace(
        **{**vars(args), "preprocess_profile": preprocess_profile}
    )


def _resolve_countries(
    *, roots: WorkspaceRoots, target_system: str, explicit_countries: list[str] | None
) -> tuple[str, ...]:
    """Resolve which countries a slice's cells sweep.

    An explicit `--countries` override applies as given. Otherwise this reads
    the target system's own discoverable country partitions -- for the fixed
    protocol's `gb`/`ie` slices that is one country each, matching the
    target system code itself, but this stays correct for any future slice
    whose target spans more than one.
    """

    if explicit_countries:
        return tuple(explicit_countries)

    target_descriptor = load_dataset_descriptor(
        roots=roots, system=target_system, require_ground_truth=False
    )
    countries = target_descriptor.available_countries
    if not countries:
        raise ValueError(
            f"target system '{target_system}' has no discoverable country "
            "partitions to sweep -- pass --countries explicitly"
        )
    return countries


def _resolve_sweep_cells(args: argparse.Namespace) -> list[BaselineSuiteCell]:
    """Which `(representation, similarity_backend)` cells this invocation sweeps.

    An explicit `--similarity-backend` always wins and applies uniformly to
    every representation in `--representations` -- the ad hoc narrower-run
    shape this script has always supported. Omitting it derives one backend
    per representation from `MANDATORY_SUITE`, or from `REFERENCE_SUITE`
    under `--reference-column`, so the *default* sweep is a two-sets split
    rather than one hand-picked backend applied to every
    representation alike -- a representation with more than one mandatory
    backend (`tfidf`/`wordpiece`/`sentencepiece` each carry `kmeans` and
    `lsh`) sweeps both, one cell each.
    """
    if setting_was_given(args, "similarity_backend"):
        return [
            BaselineSuiteCell(representation, args.similarity_backend)
            for representation in args.representations
        ]
    suite = REFERENCE_SUITE if args.reference_column else MANDATORY_SUITE
    return [cell for cell in suite if cell.representation in args.representations]


def _resolve_prefix_filter(args: argparse.Namespace) -> bool:
    """Whether a cell's `sklearn` leg runs prefix-filtered.

    `--reference-column` defaults this on: `prefix_filter.py`'s L2AP bound
    makes the filter exact, and `comparison_protocol.REFERENCE_SUITE`'s own
    docstring argues for running its sparse cells filtered. An explicit
    `--prefix-filter` always wins over that default.
    """
    if args.reference_column and not setting_was_given(args, "prefix_filter"):
        return True
    return bool(args.prefix_filter)


def _resolve_strategy_config(
    representation: str, similarity_backend: str, args: argparse.Namespace
) -> BlockingStrategyConfig:
    # Every backend setting that applies joins the identity given or
    # defaulted, as `run_blocking.py` resolves them for the cell it runs, so a
    # changed default never hands a new cell a finished cell's key. This script
    # sets the prefix filter, the cluster count and the minimum cluster size;
    # the rest are their declared defaults.
    resolved_backend_options = backend_options_from_settings(
        _DECLARATIONS,
        {"representation": representation, "backend": similarity_backend},
        {
            "prefix_filter": _resolve_prefix_filter(args),
            "kmeans_clusters": int(args.kmeans_clusters),
            "min_cluster_size": int(args.min_cluster_size),
        },
    )

    # The dense-vocabulary gate's options, written only when they differ from the default
    # (see scripts/run_blocking.py's _resolve_strategy for why).
    if args.force:
        resolved_backend_options["force"] = True
    if int(args.max_rows) != DEFAULT_DENSE_VOCABULARY_MAX_ROWS:
        resolved_backend_options["max_rows"] = int(args.max_rows)
    backend_options: dict[str, object] | None = resolved_backend_options or None

    sbert_model_name = (
        str(args.sbert_model_name).strip()
        if representation == "sbert" and args.sbert_model_name
        else None
    )
    return BlockingStrategyConfig(
        representation=representation,
        top_k=args.top_k,
        min_similarity=args.min_similarity,
        max_candidates_per_source=None,
        similarity_backend=similarity_backend,
        backend_options=backend_options,
        tokenizer=_tokenizer_for(representation),
        tokenizer_profile=str(args.tokenizer_profile).strip(),
        sbert_model_name=sbert_model_name,
        name_transform=str(args.name_transform).strip().lower(),
        preprocess_profile=str(args.preprocess_profile).strip(),
        cleanse_profile=str(args.cleanse_profile).strip(),
    )


def _build_cell_argv(
    *,
    representation: str,
    similarity_backend: str,
    source_system: str,
    target_system: str,
    country: str,
    args: argparse.Namespace,
) -> list[str]:
    """Build the `run_blocking.py` subprocess invocation for one cell.

    Mirrors `_resolve_strategy_config()`'s option resolution, translated into
    that script's CLI surface (`--additional-args STAGE.KEY=VALUE` for the
    tokenizer and backend options; direct `--force`/`--max-rows`
    flags, matching how `run_blocking.py`'s own `_resolve_strategy()` reads
    them). No roots: the cell takes this invocation's through
    `_cell_environment()`, and files its run under the shared tree by its own
    identity, beside every other run of the same pairing, so the comparison
    this script writes is one more read of that tree rather than of a corner
    this script owns.
    """

    tokenizer = _tokenizer_for(representation)
    argv = [
        sys.executable,
        str(_bootstrap.REPO_ROOT / "scripts" / "run_blocking.py"),
        *source_flags(source_system),
        "--target",
        target_system,
        "--countries",
        country,
        "--representation",
        representation,
        "--similarity-backend",
        similarity_backend,
        "--top-k",
        str(args.top_k),
        "--min-similarity",
        str(args.min_similarity),
        "--source-chunk-size",
        str(args.source_chunk_size),
        "--max-rows",
        str(int(args.max_rows)),
        "--match-col",
        str(args.match_col),
        "--name-transform",
        str(args.name_transform),
        "--cleanse-profile",
        str(args.cleanse_profile),
        "--preprocess-profile",
        str(args.preprocess_profile),
        "--additional-args",
        f"tokenizer.profile={args.tokenizer_profile}",
    ]
    if args.force:
        argv += ["--force", "true"]
    if tokenizer != "wordpiece":
        argv += ["--additional-args", f"tokenizer.name={tokenizer}"]
    if representation == "sbert" and args.sbert_model_name:
        argv += ["--sbert-model", str(args.sbert_model_name)]
    if (
        similarity_backend == "sklearn"
        and representation in PREFIX_FILTER_REPRESENTATIONS
    ):
        argv += [
            "--additional-args",
            f"backend.prefix_filter={str(_resolve_prefix_filter(args)).lower()}",
        ]
    elif similarity_backend == "kmeans" and setting_was_given(args, "kmeans_clusters"):
        argv += [
            "--additional-args",
            f"backend.kmeans_clusters={args.kmeans_clusters}",
        ]
    elif similarity_backend == "hdbscan" and setting_was_given(
        args, "min_cluster_size"
    ):
        argv += [
            "--additional-args",
            f"backend.min_cluster_size={args.min_cluster_size}",
        ]
    return argv


def _cell_environment(roots: WorkspaceRoots) -> dict[str, str]:
    """This process's environment with its resolved roots set as the
    variables a cell's `run_blocking.py` resolves its own from, so every cell
    reads and writes under the roots this invocation was given."""
    return {
        **os.environ,
        ENV_DATA_DIR: str(roots.data),
        ENV_OUTPUT_DIR: str(roots.artifacts),
        ENV_TEMP_DIR: str(roots.temp),
    }


def _execute_cell_subprocess(
    argv: list[str],
    *,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
) -> tuple[CellStatus, float, str | None]:
    """Run one cell's subprocess, bounded by `timeout_seconds`.

    Isolated from `_build_cell_argv()`/result-loading on purpose: this is the
    one function that owns `subprocess.run(..., timeout=...)` and the
    `TimeoutExpired`/non-zero-exit handling, so it can be exercised directly
    against a fake/fast `argv` (a `python -c` one-liner) rather than a real
    multi-minute `run_blocking.py` invocation.
    """

    started = time.perf_counter()
    try:
        # A hardcoded argv built by `_build_cell_argv()` (sys.executable + an in-repo
        # script path); not shell=True, no untrusted input reaches the command line.
        completed_process = subprocess.run(  # nosec B603
            argv,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        return "timeout", elapsed, f"exceeded {timeout_seconds:.0f}s runtime budget"

    elapsed = time.perf_counter() - started
    if completed_process.returncode != 0:
        message = (completed_process.stderr or completed_process.stdout or "").strip()
        return (
            "error",
            elapsed,
            message or f"exit code {completed_process.returncode}",
        )

    return "completed", elapsed, None


def _failure_row(
    *,
    label: str,
    similarity_backend: str,
    country: str,
    target_system: str,
    cleanse_profile: str,
    preprocess_profile: str,
    status: CellStatus,
    elapsed_seconds: float,
    error_message: str,
) -> dict[str, object]:
    return {
        "label": label,
        # MANDATORY_SUITE/REFERENCE_SUITE both pair one representation
        # with more than one backend (tfidf/wordpiece/sentencepiece carry both
        # kmeans and lsh), so a failure row needs its own backend to be
        # readable as a specific cell rather than just a representation.
        "similarity_backend": similarity_backend,
        "country": country,
        "target_system": target_system,
        "cleanse_profile": cleanse_profile,
        "preprocess_profile": preprocess_profile,
        "status": status,
        "elapsed_seconds": elapsed_seconds,
        "error_message": error_message,
    }


def _build_cell_config(
    *,
    representation: str,
    similarity_backend: str,
    country: str,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    source_descriptor: BlockingDatasetDescriptor,
    target_descriptor: BlockingDatasetDescriptor,
) -> BlockingRunConfig:
    """The configuration one cell runs under: the one its `run_blocking.py`
    subprocess builds from `_build_cell_argv()`'s argv, and so the identity
    that run is filed under."""

    config = BlockingRunConfig(
        roots=roots,
        prepared_base_dir=None,
        source=source_descriptor,
        target=target_descriptor,
        countries=(country,),
        strategy=_resolve_strategy_config(representation, similarity_backend, args),
        truth=source_truth_for_column(str(args.match_col)),
    )
    validate_blocking_run_config(config)
    return config


def _run_one_cell(
    *,
    representation: str,
    similarity_backend: str,
    source_system: str,
    target_system: str,
    country: str,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    source_descriptor: BlockingDatasetDescriptor,
    target_descriptor: BlockingDatasetDescriptor,
    runtime_budget_seconds: float,
) -> tuple[StrategyRunEntry | None, dict[str, object] | None]:
    """Run, or reuse, and load back one `(representation, similarity_backend,
    country)` cell under `args.cleanse_profile`.

    Returns `(entry, None)` for a completed or reused cell, or
    `(None, failure_row)` for one that timed out or errored -- never both,
    mirroring `CellStatus`'s partition of outcomes. A reused cell's entry
    carries no `runtime_seconds`, since this invocation measured nothing for it.
    """

    try:
        config = _build_cell_config(
            representation=representation,
            similarity_backend=similarity_backend,
            country=country,
            args=args,
            roots=roots,
            source_descriptor=source_descriptor,
            target_descriptor=target_descriptor,
        )
    except ValueError as exc:
        # A cell whose (representation, similarity_backend) pairing is not
        # (yet) valid -- MANDATORY_SUITE can name a cell (`lsh`)
        # ahead of it being wired into `blocking.contracts`' valid-backend
        # set -- fails at config-validation time, in-process, before any
        # subprocess is spawned. Recorded as an ordinary failed cell rather
        # than letting it escape and abort the whole sweep, the same
        # continue-on-failure guarantee every other cell already gets.
        return None, _failure_row(
            label=representation,
            similarity_backend=similarity_backend,
            country=country,
            target_system=target_system,
            cleanse_profile=str(args.cleanse_profile).strip(),
            preprocess_profile=str(args.preprocess_profile).strip(),
            status="error",
            elapsed_seconds=0.0,
            error_message=str(exc),
        )
    cleanse_profile = config.strategy.cleanse_profile
    preprocess_profile = config.strategy.preprocess_profile

    argv = _build_cell_argv(
        representation=representation,
        similarity_backend=similarity_backend,
        source_system=source_system,
        target_system=target_system,
        country=country,
        args=args,
    )

    print(
        f"[compare-blocking] {representation}: starting country={country} "
        f"{source_system} -> {target_system} (backend={similarity_backend}, "
        f"cleanse_profile={cleanse_profile}, preprocess_profile={preprocess_profile}, "
        f"budget={runtime_budget_seconds:.0f}s)",
        flush=True,
    )

    with tempfile.TemporaryDirectory(dir=_temp_parent(roots)) as record_dir:
        record_path = Path(record_dir) / "run_record.json"
        status, elapsed, message = _execute_cell_subprocess(
            [*argv, "--run-record", str(record_path)],
            timeout_seconds=runtime_budget_seconds,
            env=_cell_environment(roots),
        )
        record = (
            json.loads(record_path.read_text(encoding="utf-8"))
            if status == "completed" and record_path.exists()
            else None
        )

    if status != "completed":
        print(
            f"[compare-blocking] {representation}: country={country} {status} "
            f"after {elapsed:.1f}s",
            flush=True,
        )
        return None, _failure_row(
            label=representation,
            similarity_backend=similarity_backend,
            country=country,
            target_system=target_system,
            cleanse_profile=cleanse_profile,
            preprocess_profile=preprocess_profile,
            status=status,
            elapsed_seconds=elapsed,
            error_message=message or status,
        )

    if record is None:
        return None, _failure_row(
            label=representation,
            similarity_backend=similarity_backend,
            country=country,
            target_system=target_system,
            cleanse_profile=cleanse_profile,
            preprocess_profile=preprocess_profile,
            status="error",
            elapsed_seconds=elapsed,
            error_message="cell reported success but recorded no run",
        )
    run_dir = Path(record["run_dir"])
    reused = bool(record["reused"])
    print(
        f"[compare-blocking] {representation}: country={country} "
        + (
            f"reused the finished run at {run_dir}"
            if reused
            else f"completed in {elapsed:.1f}s"
        ),
        flush=True,
    )
    entry = StrategyRunEntry(
        label=representation,
        config=config,
        result=_load_blocking_run_result(config.roots, run_dir),
        # A reused cell measured nothing, so it carries no runtime.
        runtime_seconds=None if reused else elapsed,
    )
    return entry, None


def _temp_parent(roots: WorkspaceRoots) -> Path:
    """The workspace scratch root, created if absent, for a cell's run record."""
    roots.temp.mkdir(parents=True, exist_ok=True)
    return roots.temp


def _refused_reference_cells(
    failures: list[dict[str, object]],
) -> list[tuple[str, str, str]]:
    """`REFERENCE_SUITE` cells this run's own failures show the dense-vocabulary
    gate refused, rather than merely not having run.

    Read from the failures this run itself recorded: a gate refusal surfaces
    as an ordinary subprocess error (`run_blocking.py`'s own handler prints
    the gate's `ValueError` message verbatim to stderr), with no distinct
    `CellStatus` of its own, so this recognizes it by
    `dense_vocabulary_gate.REFUSAL_MESSAGE_MARKER` rather than re-deriving the
    gate's scale check here (which needs the real target row count this
    script never loads directly). Only `wordpiece`/`sentencepiece` on
    `sklearn` can ever be refused this way
    (`comparison_protocol.GATED_REFERENCE_CELLS`); any other cell's error is
    read as an ordinary failure, never a refusal.
    """
    return [
        (
            str(failure["label"]),
            str(failure["similarity_backend"]),
            str(failure["country"]),
        )
        for failure in failures
        if failure.get("status") == "error"
        and str(failure["label"]) in DENSE_VOCABULARY_REPRESENTATIONS
        and REFUSAL_MESSAGE_MARKER in str(failure.get("error_message") or "")
    ]


def _write_baseline_suite_status(
    *,
    comparison: pl.DataFrame | None,
    countries: tuple[str, ...],
    location: BlockingComparisonLocation,
    failures: list[dict[str, object]],
) -> None:
    """Write `baseline_suite_status.json`, the durable "was the suite run"
    artifact `company_vectorize.comparison_protocol.require_baseline_suite_complete()`
    is meant to check before a promotion decision treats this slice's
    comparison as evidence.

    Reports `MANDATORY_SUITE` and `REFERENCE_SUITE` separately:
    `complete` reflects the mandatory set alone, matching
    `require_baseline_suite_complete()`'s own gate, so a slice can be
    complete here with its reference column partly refused or entirely
    missing. Always written, even when every cell failed (`comparison=None`,
    treated the same as an empty frame -- every cell is then missing) -- a
    promotion decision needs to see an incomplete suite reported, not merely
    absent output.
    """

    empty = pl.DataFrame(
        schema={
            "representation": pl.Utf8,
            "similarity_backend": pl.Utf8,
            "country": pl.Utf8,
            "stage": pl.Utf8,
        }
    )
    status_result = missing_baseline_suite_cells(
        comparison if comparison is not None else empty,
        countries=countries,
        refused_reference_cells=_refused_reference_cells(failures),
    )

    def _cells(cells: tuple[BaselineSuiteCell, ...]) -> list[dict[str, object]]:
        return [
            {
                "representation": cell.representation,
                "similarity_backend": cell.similarity_backend,
            }
            for cell in cells
        ]

    def _triples(
        triples: tuple[tuple[str, str, str], ...],
    ) -> list[dict[str, object]]:
        return [
            {
                "representation": representation,
                "similarity_backend": similarity_backend,
                "country": country,
            }
            for representation, similarity_backend, country in triples
        ]

    status = {
        "complete": not status_result.missing_mandatory_cells,
        "countries": list(countries),
        "mandatory_suite": _cells(MANDATORY_SUITE),
        "missing_cells": _triples(status_result.missing_mandatory_cells),
        "reference_suite": _cells(REFERENCE_SUITE),
        "missing_reference_cells": _triples(status_result.missing_reference_cells),
        "refused_reference_cells": _triples(status_result.refused_reference_cells),
    }
    status_path = location.path(ComparisonArtefact.SUITE_STATUS)
    status_path.write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[compare-blocking] mandatory baseline suite "
        f"{'complete' if status['complete'] else 'INCOMPLETE'} "
        f"({len(status_result.missing_mandatory_cells)} cell(s) missing; "
        f"reference column {len(status_result.missing_reference_cells)} missing, "
        f"{len(status_result.refused_reference_cells)} refused) -- wrote {status_path}",
        flush=True,
    )


def _write_name_equality_attribution(
    entries: list[StrategyRunEntry],
    *,
    cleanse_profiles: tuple[str, ...],
    location: BlockingComparisonLocation,
) -> None:
    """Attribute the truth pairs whose name-equality level the cross profile
    moved, for every representation and country completed under both
    profiles, and write the attribution and its tally beside the comparison.

    A cell completed under only one profile has nothing to attribute against
    and is left out; its failure is already in the failures sidecar.
    """

    baseline_profile, cross_profile = cleanse_profiles
    # Keyed by backend as well: a representation is measured on more than one,
    # and a cell keyed without it is overwritten by whichever backend ran last.
    # And by preprocess profile, so each baseline is paired with the cross
    # profile's run under the same preprocessing.
    by_cell = {
        (
            (
                entry.label,
                entry.config.strategy.similarity_backend,
                entry.config.countries,
                entry.config.strategy.preprocess_profile,
            ),
            entry.config.strategy.cleanse_profile,
        ): entry
        for entry in entries
    }
    frames = [
        diff_name_equality_levels(baseline, by_cell[(cell, cross_profile)])
        for (cell, profile), baseline in by_cell.items()
        if profile == baseline_profile and (cell, cross_profile) in by_cell
    ]
    if not frames:
        return
    attribution = pl.concat(frames, how="vertical")
    for path in write_name_equality_attribution_report(
        location,
        attribution=attribution,
        transitions=tally_name_equality_transitions(attribution),
    ):
        print(f"[compare-blocking] wrote {path}", flush=True)


def _compare_one_slice(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    source_system: str,
    target_system: str,
    cleanse_profiles: tuple[str, ...],
) -> int:
    """Run every representation/country cell for one pair, once under each of
    `cleanse_profiles`, and write its report.

    Returns the number of cells that completed. A slice where every cell
    failed still writes its `strategy_comparison_failures.jsonl` sidecar, but
    no `strategy_comparison` report -- there is nothing completed to compare.
    """

    source_descriptor = load_dataset_descriptor(
        roots=roots,
        system=source_system,
        require_ground_truth=True,
        truth=source_truth_for_column(str(args.match_col)),
    )
    target_descriptor = load_dataset_descriptor(
        roots=roots, system=target_system, require_ground_truth=False
    )
    countries = _resolve_countries(
        roots=roots, target_system=target_system, explicit_countries=args.countries
    )

    comparison_location = resolve_comparison_location(
        roots, source_system=source_system, target_system=target_system
    )
    comparison_location.directory.mkdir(parents=True, exist_ok=True)

    entries: list[StrategyRunEntry] = []
    failures: list[dict[str, object]] = []
    overall_started = time.perf_counter()
    for cell in _resolve_sweep_cells(args):
        for country in countries:
            for cleanse_profile in cleanse_profiles:
                for preprocess_profile in _resolve_preprocess_profiles(args):
                    entry, failure = _run_one_cell(
                        representation=cell.representation,
                        similarity_backend=cell.similarity_backend,
                        source_system=source_system,
                        target_system=target_system,
                        country=country,
                        args=_with_preprocess_profile(
                            _with_cleanse_profile(args, cleanse_profile),
                            preprocess_profile,
                        ),
                        roots=roots,
                        source_descriptor=source_descriptor,
                        target_descriptor=target_descriptor,
                        runtime_budget_seconds=args.runtime_budget_seconds,
                    )
                    if entry is not None:
                        entries.append(entry)
                    if failure is not None:
                        failures.append(failure)
    overall_elapsed = time.perf_counter() - overall_started
    print(
        f"[compare-blocking] {source_system} -> {target_system}: sweep completed "
        f"in {overall_elapsed:.1f}s ({len(entries)} cell(s) completed, "
        f"{len(failures)} failed)",
        flush=True,
    )

    if failures:
        failures_path = comparison_location.path(ComparisonArtefact.FAILURES)
        with failures_path.open("w", encoding="utf-8") as fh:
            for failure in failures:
                fh.write(json.dumps(failure, sort_keys=True) + "\n")
        print(
            f"[compare-blocking] wrote {len(failures)} failure(s) to {failures_path}",
            flush=True,
        )

    if not entries:
        print(
            f"[compare-blocking] {source_system} -> {target_system}: no cell "
            "completed -- skipping strategy_comparison report for this slice",
            file=sys.stderr,
        )
        _write_baseline_suite_status(
            comparison=None,
            countries=countries,
            location=comparison_location,
            failures=failures,
        )
        return 0

    # `--cross-cleanser` is the caller that asks for the mixed-population
    # view: comparing runs made under two cleanse profiles side by
    # side is the whole point of that flag, so a `cleanse`/`acronym`/
    # `short_name` transform run under it legitimately produces two
    # different population keys for the same country and must not be
    # refused.
    comparison = build_strategy_comparison(
        entries, allow_mixed_population=len(cleanse_profiles) > 1
    )
    comparison_path = write_strategy_comparison_report(comparison_location, comparison)
    print(f"[compare-blocking] wrote comparison report: {comparison_path}", flush=True)

    csv_path = comparison_location.path(ComparisonArtefact.REPORT_CSV)
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(comparison.write_csv())
    print(
        f"[compare-blocking] wrote human-readable csv: {csv_path}",
        flush=True,
    )

    if len(cleanse_profiles) > 1:
        _write_name_equality_attribution(
            entries, cleanse_profiles=cleanse_profiles, location=comparison_location
        )

    _write_baseline_suite_status(
        comparison=comparison,
        countries=countries,
        location=comparison_location,
        failures=failures,
    )

    return len(entries)


def _planned_outputs(
    roots: WorkspaceRoots,
    slices: list[tuple[str, str]],
    cleanse_profiles: tuple[str, ...],
) -> list[PlannedOutput]:
    """What a sweep of `slices` would write: each pairing's runs, extended, and
    its comparison report's files, replaced; and the cells' run records in the
    temp root."""
    artefacts = [
        ComparisonArtefact.REPORT,
        ComparisonArtefact.REPORT_CSV,
        ComparisonArtefact.FAILURES,
        ComparisonArtefact.SUITE_STATUS,
    ]
    if len(cleanse_profiles) > 1:
        artefacts += [
            ComparisonArtefact.NAME_EQUALITY_ATTRIBUTION,
            ComparisonArtefact.NAME_EQUALITY_TRANSITIONS,
        ]
    outputs: list[PlannedOutput] = []
    for source_system, target_system in slices:
        outputs.append(
            PlannedOutput(
                resolve_pairing_dir(
                    roots, source_system=source_system, target_system=target_system
                ),
                OUTPUT_EXTEND,
                note="each cell's run, or the finished run with its keys reused",
            )
        )
        location = resolve_comparison_location(
            roots, source_system=source_system, target_system=target_system
        )
        outputs += [
            PlannedOutput(location.path(artefact), OUTPUT_CLEAR)
            for artefact in artefacts
        ]
    outputs.append(
        PlannedOutput(
            roots.temp,
            OUTPUT_EXTEND,
            note="each cell's run record, removed once read",
        )
    )
    return outputs


def _compare(args: argparse.Namespace) -> int:
    roots = resolve_workspace_roots_from_args(args)
    slices = _resolve_slices(args, roots)
    cleanse_profiles = _resolve_cleanse_profiles(args)

    if args.dry_run:
        # A sweep of several representations narrows nothing by representation,
        # so every setting any of them takes is reported.
        context: dict[str, object] = {"backend": args.similarity_backend}
        if len(args.representations) == 1:
            context["representation"] = args.representations[0]
        report_resolved_settings(
            "compare_blocking_strategies",
            resolve_declared_settings(args, _DECLARATIONS, _SURFACE, context=context),
            data_dir=roots.data,
            slices=slices,
            representations=list(args.representations),
            # The cells this invocation actually sweeps -- distinct
            # from `representations` above once a representation carries more
            # than one mandatory/reference backend (every classical
            # representation carries both kmeans and lsh).
            cells=[
                f"{cell.representation}/{cell.similarity_backend}"
                for cell in _resolve_sweep_cells(args)
            ],
            runtime_budget_seconds=args.runtime_budget_seconds,
            **(
                {"cleanse_profiles": list(cleanse_profiles)}
                if args.cross_cleanser is not None
                else {}
            ),
            preprocess_profiles=list(_resolve_preprocess_profiles(args)),
        )
        report_output_plan(
            "[dry-run]  ", _planned_outputs(roots, slices, cleanse_profiles)
        )
        return 0

    total_completed = 0
    for source_system, target_system in slices:
        total_completed += _compare_one_slice(
            args=args,
            roots=roots,
            source_system=source_system,
            target_system=target_system,
            cleanse_profiles=cleanse_profiles,
        )

    if total_completed == 0:
        print(
            "[compare-blocking] error: every cell across every slice failed or "
            "timed out -- nothing to compare",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _compare(args)
    except (ValueError, FileNotFoundError) as exc:
        # Mirrors scripts/run_blocking.py's handler. The dense-vocabulary
        # gate's refusal is a ValueError and is meant to be read, not decoded from a traceback.
        print(f"[compare-blocking] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
