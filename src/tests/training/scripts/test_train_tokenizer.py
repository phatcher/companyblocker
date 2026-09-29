from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from scripts import train_tokenizer
from workspace.data_layout import CLEANSED_LAYER_NAME, layer_system
from workspace.layer_layout import layer_partition_dir
from workspace.roots import WorkspaceRoots


def test_main_trains_country_then_global_when_mixed(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    cleansed_dir = layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME)
    partition_dir = layer_partition_dir(cleansed_dir, value="fr")
    partition_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name_cleansed": ["alpha ltd"]}).write_parquet(
        partition_dir / "part-001.parquet"
    )

    calls: list[tuple[str, object]] = []

    def fake_resolve_tokenizer_paths(
        *,
        tokenizer_root: Path,
        scope: str,
        system: str | None = None,
        profile: str = "default",
        corpus_filename: str = "wordpiece_training_corpus.txt",
    ):
        base = tokenizer_root / scope / (system or profile)
        return SimpleNamespace(
            corpus_path=base / "wordpiece_training_corpus.txt",
            directory=base,
        )

    def fake_sample_training_corpus(*, cleansed_dir: Path, corpus_path: Path, **kwargs):
        calls.append(("country-sample", layer_system(cleansed_dir)))
        return 1

    def fake_sample_training_corpus_for_systems(
        *, cleansed_dirs: list[Path], corpus_path: Path, **kwargs
    ):
        calls.append(("global-sample", [layer_system(path) for path in cleansed_dirs]))
        return 1

    def fake_train_tokenizer_with_trainer(
        *,
        trainer: str,
        corpus_path: Path,
        tokenizer_path: Path,
        vocab_size: int | None,
        min_frequency: int,
        options,
        show_progress: bool = True,
    ):
        calls.append(("train", tokenizer_path.parent.parent.name))
        return 1000

    def fake_evaluate_tokenizer_metrics(**kwargs):
        calls.append(("measure", kwargs["trainer"]))
        return {"fertility_distance": 0.01}, [], [], []

    def fake_publish_naive_candidate(**kwargs):
        calls.append(("publish", kwargs["system"]))
        return SimpleNamespace(uri="tokenizer://fr/wordpiece/k"), None

    monkeypatch.setattr(
        train_tokenizer, "resolve_tokenizer_paths", fake_resolve_tokenizer_paths
    )
    monkeypatch.setattr(
        train_tokenizer, "sample_training_corpus", fake_sample_training_corpus
    )
    monkeypatch.setattr(
        train_tokenizer,
        "sample_training_corpus_for_global",
        fake_sample_training_corpus_for_systems,
    )
    monkeypatch.setattr(
        train_tokenizer,
        "train_tokenizer_with_trainer",
        fake_train_tokenizer_with_trainer,
    )
    monkeypatch.setattr(
        train_tokenizer.optimize_execution,
        "evaluate_tokenizer_metrics",
        fake_evaluate_tokenizer_metrics,
    )
    monkeypatch.setattr(
        train_tokenizer.optimize_execution,
        "publish_naive_candidate",
        fake_publish_naive_candidate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--systems",
            "fr",
            "global",
            "--root",
            str(tmp_path),
        ],
    )

    exit_code = train_tokenizer.main()

    assert exit_code == 0
    call_kinds = [call[0] for call in calls]
    assert call_kinds == [
        "country-sample",
        "train",
        "measure",
        "publish",
        "global-sample",
        "train",
        "measure",
        "publish",
    ]


def test_main_reports_resolve_target_errors(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--systems",
            "global",
            "--root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_tokenizer.main()

    assert exc_info.value.code == 2


def test_main_skips_missing_cleansed_country_systems(
    tmp_path: Path, layer_fixture_dir, monkeypatch, capsys
):
    fr_cleansed = layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME)
    fr_partition = layer_partition_dir(fr_cleansed, value="fr")
    fr_partition.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name_cleansed": ["alpha ltd"]}).write_parquet(
        fr_partition / "part-001.parquet"
    )

    calls: list[tuple[str, str]] = []

    def fake_resolve_system_selection(
        systems, add_global_for_all: bool = False, global_expands_to_all: bool = False
    ):
        return SimpleNamespace(
            requested=["dk", "fr"], concrete=["dk", "fr"], include_global_target=False
        )

    def fake_resolve_tokenizer_paths(
        *,
        tokenizer_root: Path,
        scope: str,
        system: str | None = None,
        profile: str = "default",
        corpus_filename: str = "wordpiece_training_corpus.txt",
    ):
        base = tokenizer_root / scope / (system or profile)
        return SimpleNamespace(
            corpus_path=base / "wordpiece_training_corpus.txt",
            directory=base,
        )

    def fake_sample_training_corpus(*, cleansed_dir: Path, corpus_path: Path, **kwargs):
        calls.append(("sample", layer_system(cleansed_dir)))
        return 1

    def fake_train_tokenizer_with_trainer(
        *,
        trainer: str,
        corpus_path: Path,
        tokenizer_path: Path,
        vocab_size: int | None,
        min_frequency: int,
        options,
        show_progress: bool = True,
    ):
        calls.append(("train", tokenizer_path.parent.parent.name))
        return 1000

    def fake_evaluate_tokenizer_metrics(**kwargs):
        calls.append(("measure", kwargs["trainer"]))
        return {"fertility_distance": 0.01}, [], [], []

    def fake_publish_naive_candidate(**kwargs):
        calls.append(("publish", kwargs["system"]))
        return SimpleNamespace(uri="tokenizer://fr/wordpiece/k"), None

    monkeypatch.setattr(
        train_tokenizer, "resolve_system_selection", fake_resolve_system_selection
    )
    monkeypatch.setattr(
        train_tokenizer, "resolve_tokenizer_paths", fake_resolve_tokenizer_paths
    )
    monkeypatch.setattr(
        train_tokenizer, "sample_training_corpus", fake_sample_training_corpus
    )
    monkeypatch.setattr(
        train_tokenizer,
        "train_tokenizer_with_trainer",
        fake_train_tokenizer_with_trainer,
    )
    monkeypatch.setattr(
        train_tokenizer.optimize_execution,
        "evaluate_tokenizer_metrics",
        fake_evaluate_tokenizer_metrics,
    )
    monkeypatch.setattr(
        train_tokenizer.optimize_execution,
        "publish_naive_candidate",
        fake_publish_naive_candidate,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--systems",
            "dk",
            "fr",
            "--root",
            str(tmp_path),
        ],
    )

    exit_code = train_tokenizer.main()

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "[warn] Skipping 'dk' system training" in out
    call_kinds = [call[0] for call in calls]
    assert call_kinds == ["sample", "train", "measure", "publish"]
    assert calls[0] == ("sample", "fr")
    assert calls[1] == ("train", "fr")
    assert calls[2] == ("measure", "wordpiece")
    assert calls[3] == ("publish", "fr")


def test_main_wires_target_resolution_dependencies(
    tmp_path: Path, workspace_roots: WorkspaceRoots, monkeypatch
):
    """`--systems all` expands here through the same catalog-flag resolver the
    acquisition CLIs use, so this only checks main wires it through --
    which systems it names is the catalog's business, tested against the
    `all_systems_target` flag in the acquisition tests."""
    captured: dict[str, object] = {}

    def fake_resolve_targets(
        *,
        args,
        roots,
        resolve_system_selection_func,
        supported_system_codes_func,
        has_cleansed_parquet_func,
    ):
        captured["roots"] = roots
        captured["selection"] = resolve_system_selection_func(
            ["fr"], add_global_for_all=True
        )
        captured["supported"] = supported_system_codes_func()
        captured["has_cleansed"] = has_cleansed_parquet_func
        return [SimpleNamespace(scope="country", systems=["fr"])]

    monkeypatch.setattr(
        train_tokenizer, "all_target_system_codes", lambda: ["fr", "ie"]
    )
    monkeypatch.setattr(
        train_tokenizer.optimize_execution, "resolve_targets", fake_resolve_targets
    )
    monkeypatch.setattr(train_tokenizer, "run_train_workload", lambda **kwargs: 0)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--systems",
            "fr",
            "--mode",
            "train",
            "--root",
            str(tmp_path),
        ],
    )

    exit_code = train_tokenizer.main()

    assert exit_code == 0
    assert captured["roots"] == workspace_roots
    assert captured["supported"] == ["fr", "ie"]
    assert (
        captured["has_cleansed"]
        is train_tokenizer.optimize_execution.has_cleansed_parquet
    )


def test_main_reports_optimize_guard_refusals_cleanly_not_as_a_traceback(
    tmp_path: Path, monkeypatch
):
    # Regression: a deliberate refusal (canary-mismatch-purge guard,
    # corpus-resample-orphans-a-sweep guard -- both raise RuntimeError as
    # this codebase's own convention for "the user needs to decide, not a
    # bug") used to have nothing catching it at the CLI boundary, so it
    # surfaced as a raw Python traceback instead of a clean, actionable
    # message. It must come out through parser.error() (SystemExit(2)) like
    # any other user-facing argument/validation error, not crash.
    def fake_resolve_targets(**kwargs):
        return [SimpleNamespace(scope="country", systems=["fr"])]

    def raising_run_optimize_workload(**kwargs):
        raise RuntimeError(
            "This sweep's configuration doesn't match what's cached for "
            "trainer=wordpiece: proceeding would purge 3 candidate model(s) "
            "and 12 run-log row(s). Re-run with --optimize-wipe if you intend "
            "this."
        )

    monkeypatch.setattr(
        train_tokenizer.optimize_execution, "resolve_targets", fake_resolve_targets
    )
    monkeypatch.setattr(
        train_tokenizer, "run_optimize_workload", raising_run_optimize_workload
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--systems",
            "fr",
            "--mode",
            "optimize",
            "--root",
            str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        train_tokenizer.main()

    assert exc_info.value.code == 2


def test_parser_rejects_removed_optimize_threshold_profile_flag():
    parser = train_tokenizer.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--systems",
                "ie",
                "--mode",
                "optimize",
                "--optimize-threshold-profile",
                "auto",
            ]
        )


def test_parser_supports_sp_byte_fallback_boolean_optional_forms():
    parser = train_tokenizer.build_parser()

    enabled = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "train",
            "--sp-byte-fallback",
        ]
    )
    disabled = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "train",
            "--no-sp-byte-fallback",
        ]
    )

    assert enabled.sp_byte_fallback is True
    assert disabled.sp_byte_fallback is False


def test_parser_supports_finalize_enable_flag():
    parser = train_tokenizer.build_parser()

    enabled = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
            "--finalize",
        ]
    )

    assert enabled.finalize is True


def test_parser_defaults_finalize_to_false():
    parser = train_tokenizer.build_parser()

    args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
        ]
    )

    assert args.finalize is False


def test_parser_reoptimize_defaults_to_false_and_takes_true():
    parser = train_tokenizer.build_parser()
    base = ["--systems", "ie", "--mode", "optimize"]

    assert parser.parse_args(base).reoptimize is False
    assert parser.parse_args([*base, "--reoptimize", "true"]).reoptimize is True


def test_parser_refining_phase_flags_default_to_auto_derived_and_enabled():
    parser = train_tokenizer.build_parser()

    default_args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
        ]
    )
    assert default_args.refining_phase is True
    assert default_args.pilot_early_stop is True
    assert default_args.skip_equivalent_candidates is True
    assert default_args.refine_vocab_step is None
    assert default_args.refine_vocab_step_minimum == 100

    overridden_args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
            "--refining-phase",
            "false",
            "--pilot-early-stop",
            "false",
            "--skip-equivalent-candidates",
            "false",
            "--refine-vocab-step",
            "500",
            "--refine-vocab-step-minimum",
            "250",
        ]
    )
    assert overridden_args.refining_phase is False
    assert overridden_args.pilot_early_stop is False
    assert overridden_args.skip_equivalent_candidates is False
    assert overridden_args.refine_vocab_step == 500
    assert overridden_args.refine_vocab_step_minimum == 250


def test_parser_delete_rejected_models_defaults_true_and_accepts_false_literal():
    parser = train_tokenizer.build_parser()

    default_args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
        ]
    )
    disabled_args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
            "--delete-rejected-models",
            "false",
        ]
    )

    assert default_args.delete_rejected_models is True
    assert disabled_args.delete_rejected_models is False


def test_parser_optimize_preset_defaults_to_none_and_rejects_unknown_name():
    parser = train_tokenizer.build_parser()

    default_args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
        ]
    )
    assert default_args.optimize_preset is None

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--systems",
                "ie",
                "--mode",
                "optimize",
                "--optimize-preset",
                "does-not-exist",
            ]
        )
    assert exc_info.value.code == 2


def test_parser_optimize_preset_is_distinct_from_profile_flag():
    # --profile is the pre-existing global tokenizer profile name
    # (scripts/cli_common.py); --optimize-preset must be a separate flag
    # rather than shadowing it.
    parser = train_tokenizer.build_parser()

    args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
            "--profile",
            "my-global-profile",
            "--optimize-preset",
            "sentencepiece",
        ]
    )

    assert args.profile == "my-global-profile"
    assert args.optimize_preset == "sentencepiece"


def test_main_optimize_records_preset_and_overrides_in_manifest(
    tmp_path: Path, layer_fixture_dir, monkeypatch
):
    cleansed_dir = layer_fixture_dir("fr", layer=CLEANSED_LAYER_NAME)
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name_cleansed": ["alpha ltd"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    captured_manifest_calls: list[dict] = []

    def fake_run_target(**kwargs):
        args = kwargs["args"]
        # Confirms apply_optimize_preset ran (via workloads.run_optimize_workload)
        # before this target runs, and that an explicit flag beat the
        # trainer-selected preset's own value.
        captured_manifest_calls.append(
            {
                "optimize_preset": args.optimize_preset,
                "optimize_preset_overrides": args.optimize_preset_overrides,
                "fertility_target": args.fertility_target,
            }
        )

    def fake_resolve_targets(**kwargs):
        return [SimpleNamespace(scope="country", systems=["fr"])]

    monkeypatch.setattr(
        train_tokenizer.optimize_execution, "resolve_targets", fake_resolve_targets
    )
    monkeypatch.setattr(
        train_tokenizer.optimize_execution, "run_optimize_target", fake_run_target
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tokenizer.py",
            "--systems",
            "fr",
            "--mode",
            "optimize",
            "--root",
            str(tmp_path),
            "--optimize-preset",
            "sentencepiece",
            "--fertility-target",
            "1.99",
        ],
    )

    exit_code = train_tokenizer.main()

    assert exit_code == 0
    assert captured_manifest_calls == [
        {
            "optimize_preset": "sentencepiece",
            "optimize_preset_overrides": {"fertility_target": 1.99},
            "fertility_target": 1.99,
        }
    ]


def test_parser_expansion_strategy_defaults_to_heuristic_and_rejects_unknown():
    parser = train_tokenizer.build_parser()

    default_args = parser.parse_args(
        [
            "--systems",
            "ie",
            "--mode",
            "optimize",
        ]
    )
    assert default_args.expansion_strategy == "heuristic"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--systems",
                "ie",
                "--mode",
                "optimize",
                "--expansion-strategy",
                "does-not-exist",
            ]
        )
