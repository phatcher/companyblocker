"""Train and optimize workloads over every target a run names, global or per system.

`run_train_workload` and `run_optimize_workload` loop the targets and apply profiles and policy; the work each target does is injected through `TrainWorkloadServices` and `OptimizeWorkloadServices`, so this module holds the orchestration and none of the training.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from company_tokenize import TrainerOptions
from company_tokenize.contracts import OptimizeCandidatePolicy, TrainingTarget

from workspace.roots import WorkspaceRoots


class TrainGlobalRunner(Protocol):
    def __call__(
        self,
        *,
        args: argparse.Namespace,
        roots: WorkspaceRoots,
        systems: list[str],
        trainer_options: TrainerOptions,
    ) -> None: ...


class TrainSystemRunner(Protocol):
    def __call__(
        self,
        *,
        args: argparse.Namespace,
        roots: WorkspaceRoots,
        system_code: str,
        system_index: int,
        total_systems: int,
        trainer_options: TrainerOptions,
    ) -> int: ...


TrainerOptionsValidator = Callable[..., None]
# Resolves and applies a named --optimize-preset's threshold/grid defaults to
# `args` in place, returning (resolved_preset_name, overrides) for
# both modes' finalize step to record in the run manifest.
OptimizePresetApplier = Callable[[argparse.Namespace], tuple[str, dict[str, Any]]]
OptimizeArgsValidator = Callable[[argparse.Namespace], None]
OptimizeCandidatePolicyBuilder = Callable[[argparse.Namespace], OptimizeCandidatePolicy]
OptimizeSearchSpaceResolver = Callable[
    [argparse.Namespace], tuple[list[int], list[int], list[int]]
]
OptimizeTargetRunner = Callable[..., None]


@dataclass(frozen=True)
class TrainWorkloadServices:
    run_global: TrainGlobalRunner
    run_system: TrainSystemRunner
    validate_trainer_options: TrainerOptionsValidator
    apply_optimize_preset: OptimizePresetApplier


@dataclass(frozen=True)
class OptimizeWorkloadServices:
    apply_optimize_preset: OptimizePresetApplier
    validate_optimize_args: OptimizeArgsValidator
    build_candidate_policy: OptimizeCandidatePolicyBuilder
    resolve_search_space: OptimizeSearchSpaceResolver
    run_target: OptimizeTargetRunner
    validate_trainer_options: TrainerOptionsValidator


def run_train_workload(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    roots: WorkspaceRoots,
    targets: list[TrainingTarget],
    services: TrainWorkloadServices,
) -> int:
    # Plain --mode train has no sweep to size a threshold/grid preset for,
    # but resolving one anyway keeps the run manifest's
    # optimize_preset/optimize_preset_overrides fields populated the same
    # way for both modes, rather than only ever appearing for --mode
    # optimize.
    preset_name, _preset_overrides = services.apply_optimize_preset(args)
    print(f"[info] optimize_preset={preset_name}")

    trainer_options = TrainerOptions(
        tokenizer_encoding=args.tokenizer_encoding,
        sp_character_coverage=args.sp_character_coverage,
        sp_byte_fallback=args.sp_byte_fallback,
    )
    services.validate_trainer_options(trainer=args.tokenizer, options=trainer_options)

    for target in targets:
        systems = target.systems
        if target.scope == "global":
            services.run_global(
                args=args,
                roots=roots,
                systems=systems,
                trainer_options=trainer_options,
            )
            continue

        total_lines = 0
        trained_systems = 0
        for index, system_code in enumerate(systems, start=1):
            line_count = services.run_system(
                args=args,
                roots=roots,
                system_code=system_code,
                system_index=index,
                total_systems=len(systems),
                trainer_options=trainer_options,
            )
            if line_count <= 0:
                continue
            total_lines += line_count
            trained_systems += 1

        print(
            "[info] System tokenizer training complete (all eligible rows). "
            + f"Systems trained: {trained_systems}/{len(systems)}. "
            + f"Total corpus names: {total_lines:,}"
        )

    return 0


def run_optimize_workload(
    *,
    args: argparse.Namespace,
    roots: WorkspaceRoots,
    targets: list[TrainingTarget],
    services: OptimizeWorkloadServices,
) -> int:
    preset_name, preset_overrides = services.apply_optimize_preset(args)
    effective_elbow_extra_steps = args.elbow_extra_steps
    print(f"[optimize] preset={preset_name} overrides={sorted(preset_overrides)}")
    print(
        "[optimize] policy=standard "
        + f"elbow_min_points={args.elbow_min_points} "
        + f"elbow_min_improvement={args.elbow_min_improvement:.6f} "
        + f"elbow_extra_steps={effective_elbow_extra_steps}"
    )

    trainer_options = TrainerOptions(
        tokenizer_encoding=args.tokenizer_encoding,
        sp_character_coverage=args.sp_character_coverage,
        sp_byte_fallback=args.sp_byte_fallback,
    )
    services.validate_trainer_options(trainer=args.tokenizer, options=trainer_options)
    services.validate_optimize_args(args)
    candidate_policy = services.build_candidate_policy(args)
    seeds, vocab_schedule, min_frequencies = services.resolve_search_space(args)
    session_id = f"opt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"

    for target in targets:
        services.run_target(
            args=args,
            roots=roots,
            target=target,
            trainer_options=trainer_options,
            candidate_policy=candidate_policy,
            seeds=seeds,
            vocab_schedule=vocab_schedule,
            min_frequencies=min_frequencies,
            session_id=session_id,
            effective_elbow_extra_steps=effective_elbow_extra_steps,
        )

    return 0
