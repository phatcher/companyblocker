from __future__ import annotations

import argparse

from company_tokenize import TrainingTarget

from training.workloads import (
    OptimizeWorkloadServices,
    TrainWorkloadServices,
    run_optimize_workload,
    run_train_workload,
)
from workspace.roots import WorkspaceRoots


def _train_args(**overrides):
    base = {
        "tokenizer": "wordpiece",
        "tokenizer_encoding": "unigram",
        "sp_character_coverage": 0.9995,
        "sp_byte_fallback": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _apply_optimize_preset_stub(passed_args):
    passed_args.optimize_preset = "wordpiece"
    passed_args.optimize_preset_overrides = {}
    return "wordpiece", {}


def _optimize_args(**overrides):
    base = {
        "tokenizer": "wordpiece",
        "tokenizer_encoding": "unigram",
        "sp_character_coverage": 0.9995,
        "sp_byte_fallback": False,
        "elbow_extra_steps": 2,
        "elbow_min_points": 2,
        "elbow_min_improvement": 0.0001,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_train_workload_runs_country_then_global_targets(
    workspace_roots: WorkspaceRoots,
):
    args = _train_args()
    parser = argparse.ArgumentParser()
    targets = [
        TrainingTarget(scope="country", systems=["fr", "ie"]),
        TrainingTarget(scope="global", systems=["fr", "ie"]),
    ]

    calls: list[tuple[str, str]] = []

    def run_system(**kwargs):
        calls.append(("system", kwargs["system_code"]))
        return 10

    def run_global(**kwargs):
        calls.append(("global", ",".join(kwargs["systems"])))

    services = TrainWorkloadServices(
        run_global=run_global,
        run_system=run_system,
        validate_trainer_options=lambda **kwargs: None,
        apply_optimize_preset=_apply_optimize_preset_stub,
    )

    result = run_train_workload(
        args=args,
        parser=parser,
        roots=workspace_roots,
        targets=targets,
        services=services,
    )

    assert result == 0
    assert calls == [
        ("system", "fr"),
        ("system", "ie"),
        ("global", "fr,ie"),
    ]


def test_run_train_workload_applies_optimize_preset_before_targets(
    workspace_roots: WorkspaceRoots,
):
    # Plain --mode train has no sweep, but the preset's run-manifest fields are
    # populated the same way for both modes, so the preset applier must run
    # here too, not just inside run_optimize_workload.
    args = _train_args()
    parser = argparse.ArgumentParser()
    targets = [TrainingTarget(scope="global", systems=["fr"])]
    apply_calls: list[argparse.Namespace] = []

    def apply_optimize_preset(passed_args):
        apply_calls.append(passed_args)
        passed_args.optimize_preset = "wordpiece"
        passed_args.optimize_preset_overrides = {}
        return "wordpiece", {}

    services = TrainWorkloadServices(
        run_global=lambda **kwargs: None,
        run_system=lambda **kwargs: 1,
        validate_trainer_options=lambda **kwargs: None,
        apply_optimize_preset=apply_optimize_preset,
    )

    result = run_train_workload(
        args=args,
        parser=parser,
        roots=workspace_roots,
        targets=targets,
        services=services,
    )

    assert result == 0
    assert apply_calls == [args]
    assert args.optimize_preset == "wordpiece"


def test_run_optimize_workload_invokes_services_for_each_target(
    workspace_roots: WorkspaceRoots,
):
    args = _optimize_args()
    targets = [
        TrainingTarget(scope="country", systems=["fr"]),
        TrainingTarget(scope="global", systems=["fr", "ie"]),
    ]

    called = {
        "apply": 0,
        "validate_optimize": 0,
        "build_policy": 0,
        "resolve_search_space": 0,
        "run_target": 0,
        "validate_trainer": 0,
    }

    policy = object()

    def _record(key: str, value):
        called[key] = called[key] + 1
        return value

    services = OptimizeWorkloadServices(
        apply_optimize_preset=lambda passed_args: _record("apply", ("wordpiece", {})),
        validate_optimize_args=lambda passed_args: _record("validate_optimize", None),
        build_candidate_policy=lambda passed_args: _record("build_policy", policy),
        resolve_search_space=lambda passed_args: _record(
            "resolve_search_space", ([1], [10000], [1])
        ),
        run_target=lambda **kwargs: _record("run_target", None),
        validate_trainer_options=lambda **kwargs: _record("validate_trainer", None),
    )

    result = run_optimize_workload(
        args=args,
        roots=workspace_roots,
        targets=targets,
        services=services,
    )

    assert result == 0
    assert called["apply"] == 1
    assert called["validate_optimize"] == 1
    assert called["build_policy"] == 1
    assert called["resolve_search_space"] == 1
    assert called["validate_trainer"] == 1
    assert called["run_target"] == 2
