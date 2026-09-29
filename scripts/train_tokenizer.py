"""Train or optimize tokenizers for one or more systems, or the pooled `global` scope.

`--mode train` trains one tokenizer per target. `--mode optimize` searches vocabulary sizes and
minimum frequencies over several seeds and scores each candidate; `--dry-run` previews the
schedule, and `--finalize` retrains the winner. `--tokenizer` (`wordpiece` by default, or
`sentencepiece` with `--encoding unigram|bpe`) applies to the whole invocation, so producing both
kinds means two runs. `--systems` takes codes, `global`, a mix, or `all`, which trains every
system and `global` in one pass.

Two settings share the word profile. `--tokenizer-profile` names a global tokenizer profile and
applies only to the `global` scope. A noise-words profile (`none`, `strict`, `balanced`,
`aggressive`) is chosen later, at tokenize time, from the one `noise_words.json`
`generate_noise_words.py` computes all three into.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401
from cli_common import (
    add_optimize_mode_args,
    add_tokenizer_training_core_args,
    run_reporting_argument_errors,
)
from company_tokenize import (
    SUPPORTED_TRAINERS,
    TRAINING_CORPUS_FILENAME,
    resolve_tokenizer_path_for_trainer,
    resolve_tokenizer_paths,
    train_tokenizer_with_trainer,
    validate_trainer_options,
)

from acquisition.pipeline import resolve_system_selection
from acquisition.pipeline_selection import all_target_system_codes
from acquisition.tokenizer_ops import (
    sample_training_corpus,
    sample_training_corpus_for_global,
)
from training import optimize_execution
from training.workloads import (
    OptimizeWorkloadServices,
    TrainWorkloadServices,
    run_optimize_workload,
    run_train_workload,
)
from workspace.roots import default_workspace_roots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tokenizer from cleansed parquet shards."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "optimize"],
        default="train",
        help="Execution mode: regular train or monte-carlo optimize.",
    )
    add_tokenizer_training_core_args(
        parser,
        trainer_choices=list(SUPPORTED_TRAINERS),
        default_corpus_filename=TRAINING_CORPUS_FILENAME,
        systems_detail=(
            "Use explicit system codes (for example 'fr gb ie') for per-system country training. "
            "Use '--systems global' to train a single global tokenizer across every system in '--systems all' with cleansed data. "
            "Use mixed values (for example 'fr ie global') to train country tokenizers and then a global tokenizer in one pass."
        ),
    )

    add_optimize_mode_args(
        parser,
        base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
        default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
        default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
        expansion_strategy_choices=list(optimize_execution.EXPANSION_STRATEGIES),
        default_expansion_strategy=optimize_execution.HEURISTIC_STRATEGY_NAME,
        optimize_preset_choices=list(optimize_execution.OPTIMIZE_PRESETS),
    )

    return parser


def _build_optimize_services() -> OptimizeWorkloadServices:
    return OptimizeWorkloadServices(
        apply_optimize_preset=optimize_execution.build_apply_optimize_preset(
            presets=optimize_execution.OPTIMIZE_PRESETS,
            base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
            default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
            default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
        ),
        validate_optimize_args=optimize_execution.validate_optimize_args,
        build_candidate_policy=optimize_execution.build_optimize_candidate_policy,
        resolve_search_space=optimize_execution.resolve_optimize_search_space,
        run_target=optimize_execution.run_optimize_target,
        validate_trainer_options=validate_trainer_options,
    )


def _build_train_services() -> TrainWorkloadServices:
    return TrainWorkloadServices(
        run_global=optimize_execution.build_train_mode_global_runner(
            all_rows_target=optimize_execution.ALL_ROWS_TARGET,
            resolve_tokenizer_paths_func=resolve_tokenizer_paths,
            resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path_for_trainer,
            sample_training_corpus_for_global_func=sample_training_corpus_for_global,
            train_tokenizer_with_trainer_func=train_tokenizer_with_trainer,
        ),
        run_system=optimize_execution.build_train_mode_system_runner(
            resolve_tokenizer_paths_func=resolve_tokenizer_paths,
            sample_training_corpus_func=sample_training_corpus,
            resolve_tokenizer_path_for_trainer_func=resolve_tokenizer_path_for_trainer,
            train_tokenizer_with_trainer_func=train_tokenizer_with_trainer,
        ),
        validate_trainer_options=validate_trainer_options,
        apply_optimize_preset=optimize_execution.build_apply_optimize_preset(
            presets=optimize_execution.OPTIMIZE_PRESETS,
            base_defaults=optimize_execution.OPTIMIZE_BASE_DEFAULTS,
            default_vocab_grid=optimize_execution.DEFAULT_VOCAB_GRID,
            default_min_freq_grid=optimize_execution.DEFAULT_MIN_FREQ_GRID,
        ),
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    project_root = Path(args.root).resolve()
    roots = default_workspace_roots(project_root)
    # The same `all` the acquisition CLIs expand, read from each plan's
    # `all_systems_target` flag. Resolving it here from statuses instead would
    # let `--systems all` mean one set of systems when acquiring and another
    # when training.
    supported_system_codes_func = all_target_system_codes

    try:
        targets = optimize_execution.resolve_targets(
            args=args,
            roots=roots,
            resolve_system_selection_func=resolve_system_selection,
            supported_system_codes_func=supported_system_codes_func,
            has_cleansed_parquet_func=optimize_execution.has_cleansed_parquet,
        )
    except ValueError as exc:
        parser.error(str(exc))

    target_desc = ", ".join(
        f"{target.scope}:{len(target.systems)}" for target in targets
    )
    print(f"[info] Starting tokenizer training targets -> {target_desc}")

    try:
        if args.mode == "optimize":
            result = run_optimize_workload(
                args=args,
                roots=roots,
                targets=targets,
                services=_build_optimize_services(),
            )
        else:
            result = run_train_workload(
                args=args,
                parser=parser,
                roots=roots,
                targets=targets,
                services=_build_train_services(),
            )
    except (ValueError, RuntimeError) as exc:
        # RuntimeError here is specifically this codebase's own convention for
        # a deliberate, actionable refusal (corpus-resample-orphans-a-sweep)
        # -- a decision for the user to make, not an unexpected bug, so it
        # gets the same clean parser.error() treatment as a plain
        # argument-validation ValueError instead of a raw traceback. A
        # genuine unexpected bug raises some other exception type and still
        # surfaces normally.
        parser.error(str(exc))

    print("[info] Tokenizer training complete")
    return result


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
