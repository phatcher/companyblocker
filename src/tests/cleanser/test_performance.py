import cProfile
import glob
import hashlib
import json
import os
import pstats
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

import polars as pl
import pytest
from baseline import BASELINE_DIR
from company_cleanse.pipeline import (
    CANONICAL_COMPANY_TYPE_STEP,
    DERIVE_ACRONYM_FIELD_STEP,
    ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
    ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
    GENERATE_CLEANSED_COMPANY_NAME_STEP,
    STEP_ENGINE_NOOP,
    STEP_ENGINE_POLARS,
    STEP_ENGINE_UDF,
)
from company_cleanse.rules import get_company_type_rules_for_country

from acquisition.cleanser_orchestrate import name_cleanse
from acquisition.registry import get_system_company_type_mapping, get_system_plan
from workspace.artifact_layout import perf_artifact_root

pytestmark = pytest.mark.performance


def _build_perf_input(rows: int) -> pl.DataFrame:
    patterns = [
        "ALPHA CLEANING LIMITED",
        "BETA SERVICES LTD",
        "GAMMA HOLDINGS PLC",
        "(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD",
        '"TRIPLE D" "PROPERTIES LIMITED"',
        "OMEGA JSC",
        "DELTA SPOLKA JAWNA",
        "EPSILON SP Z O O",
    ]
    values = [patterns[i % len(patterns)] for i in range(rows)]
    return pl.DataFrame({"CompanyName": values})


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def _artifacts_dir(repo_root: Path) -> Path:
    return perf_artifact_root(repo_root)


def _resolve_real_company_col(system_plan) -> str:
    canonical_input_columns = (
        getattr(system_plan, "canonical_input_columns", None) or ()
    )
    if canonical_input_columns:
        first = str(canonical_input_columns[0]).strip()
        if first:
            return first
    return "name"


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def _benchmark_mode() -> str:
    mode = os.environ.get("PERF_BENCHMARK_MODE", "regression").strip().lower()
    if mode not in {"regression", "profile"}:
        raise ValueError("PERF_BENCHMARK_MODE must be 'regression' or 'profile'.")
    return mode


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _validate_run_policy(
    *,
    runs: int,
    update_baseline: bool,
    benchmark_mode: str,
    profile_once: bool,
) -> None:
    if runs < 1:
        raise ValueError("PERF_RUNS must be >= 1.")
    if runs % 2 == 0:
        raise ValueError(
            "PERF_RUNS must be odd for stable median behavior (recommended: 3 or 5)."
        )

    if benchmark_mode == "profile" and not profile_once:
        raise ValueError("PERF_BENCHMARK_MODE=profile requires PERF_PROFILE=1.")
    if benchmark_mode == "regression" and profile_once:
        raise ValueError(
            "PERF_PROFILE=1 is only allowed with PERF_BENCHMARK_MODE=profile."
        )

    if benchmark_mode == "profile" and update_baseline:
        raise ValueError("PERF_UPDATE_BASELINE=1 is not allowed in profile mode.")

    allow_single_run_baseline = (
        os.environ.get("PERF_ALLOW_SINGLE_RUN_BASELINE", "0") == "1"
    )
    if update_baseline and runs < 3 and not allow_single_run_baseline:
        raise ValueError(
            "Baseline promotion requires PERF_RUNS>=3. "
            "Set PERF_ALLOW_SINGLE_RUN_BASELINE=1 to override explicitly."
        )


def _stable_path_for_hash(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _file_list_sha256(paths: list[Path], repo_root: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(paths):
        stable_path = _stable_path_for_hash(path, repo_root)
        size = path.stat().st_size
        hasher.update(f"{stable_path}|{size}\n".encode())
    return hasher.hexdigest()


def _run_benchmark(
    *,
    repo_root: Path,
    test_name: str,
    input_dir: Path,
    output_dir: Path,
    expected_rows: int,
    system: str,
    runs: int,
    max_regression_pct: float,
    update_baseline: bool,
    profile_once: bool,
    benchmark_mode: str,
    dataset_identity: dict[str, object],
    company_type_matcher: str,
    company_col: str,
    and_token: str,
    source_company_type_col: str | None,
    source_company_type_mapping: dict[str, str] | None,
    step_engines: dict[str, str] | None,
    write_output: bool,
) -> None:
    is_profile_mode = benchmark_mode == "profile"
    company_type_regex, company_type_mapping = get_company_type_rules_for_country(
        system
    )

    elapsed_seconds: list[float] = []
    profile_path: Path | None = None
    profile_summary_path: Path | None = None

    for run_index in range(runs):
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        if profile_once and run_index == 0:
            profile = cProfile.Profile()
            processed_rows = profile.runcall(
                name_cleanse,
                input_dir=input_dir,
                output_dir=output_dir,
                company_type_regex=company_type_regex,
                company_col=company_col,
                and_tokens=[and_token],
                char_whitelist=r"[^a-z0-9\s!]",
                company_type_mapping=company_type_mapping,
                company_type_matcher=company_type_matcher,
                source_company_type_col=source_company_type_col,
                source_company_type_mapping=source_company_type_mapping,
                step_engines=step_engines,
                write_output=write_output,
            )
            artifacts_dir = _artifacts_dir(repo_root)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            profile_path = artifacts_dir / f"{test_name}_profile.pstats"
            profile_summary_path = artifacts_dir / f"{test_name}_profile_top30.txt"
            profile.dump_stats(str(profile_path))
            with profile_summary_path.open("w", encoding="utf-8") as handle:
                stats = pstats.Stats(profile, stream=handle).sort_stats("cumulative")
                stats.print_stats(30)
        else:
            processed_rows = name_cleanse(
                input_dir=input_dir,
                output_dir=output_dir,
                company_type_regex=company_type_regex,
                company_col=company_col,
                and_tokens=[and_token],
                char_whitelist=r"[^a-z0-9\s!]",
                company_type_mapping=company_type_mapping,
                company_type_matcher=company_type_matcher,
                source_company_type_col=source_company_type_col,
                source_company_type_mapping=source_company_type_mapping,
                step_engines=step_engines,
                write_output=write_output,
            )

        elapsed = max(time.perf_counter() - t0, 1e-9)
        elapsed_seconds.append(elapsed)
        assert processed_rows == expected_rows

    median_elapsed = _median(elapsed_seconds)
    throughput_rows_per_sec = expected_rows / median_elapsed

    now_iso = datetime.now(UTC).isoformat()
    metrics = {
        "test": test_name,
        "benchmark_mode": benchmark_mode,
        "system": system,
        "timestamp_utc": now_iso,
        "rows": expected_rows,
        "runs": runs,
        "elapsed_seconds": elapsed_seconds,
        "median_elapsed_seconds": median_elapsed,
        "throughput_rows_per_sec": throughput_rows_per_sec,
        "dataset": dataset_identity,
        "company_type_matcher": company_type_matcher,
    }

    artifacts_dir = _artifacts_dir(repo_root)
    latest_path = artifacts_dir / f"{test_name}_latest.json"
    history_path = artifacts_dir / f"{test_name}_history.jsonl"
    # The baseline is tracked input beside the repo's other ratchet baselines; the
    # latest, history and profile files are generated output under `artifacts/perf`.
    baseline_path = BASELINE_DIR / f"{test_name}_baseline.json"
    profile_metrics_path = artifacts_dir / f"{test_name}_profile_metrics.json"
    profile_history_path = artifacts_dir / f"{test_name}_profile_history.jsonl"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    if is_profile_mode:
        profile_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _append_jsonl(profile_history_path, metrics)
    else:
        latest_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _append_jsonl(history_path, metrics)

    print(
        f"{test_name} benchmark: "
        f"mode={benchmark_mode}, "
        f"system={system}, rows={expected_rows:,}, runs={runs}, "
        f"median_elapsed={median_elapsed:.3f}s, "
        f"throughput={throughput_rows_per_sec:,.0f} rows/s"
    )
    if is_profile_mode:
        print(f"profile metrics: {profile_metrics_path}")
        print(f"profile history: {profile_history_path}")
    else:
        print(f"latest: {latest_path}")
        print(f"history: {history_path}")
    if profile_path and profile_summary_path:
        print(f"profile: {profile_path}")
        print(f"profile summary: {profile_summary_path}")

    if is_profile_mode:
        print("Regression and baseline updates are skipped in profile mode.")
        return

    if update_baseline:
        baseline_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"baseline updated: {baseline_path}")
        return

    if not baseline_path.exists():
        print(
            "No baseline found. Set PERF_UPDATE_BASELINE=1 once to create "
            f"{baseline_path}."
        )
        return

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_tput = float(baseline.get("throughput_rows_per_sec", 0.0))
    assert baseline_tput > 0.0, "Baseline throughput must be > 0."

    lower_bound = baseline_tput * (1.0 - max_regression_pct)
    assert throughput_rows_per_sec >= lower_bound, (
        "Cleanse throughput regression detected. "
        f"current={throughput_rows_per_sec:,.0f} rows/s, "
        f"baseline={baseline_tput:,.0f} rows/s, "
        f"allowed_regression={max_regression_pct:.0%}"
    )


def _resolve_real_input_files(repo_root: Path, system: str) -> list[Path]:
    files_csv = os.environ.get("PERF_REAL_INPUT_FILES", "").strip()
    if files_csv:
        files: list[Path] = []
        for part in files_csv.split(";"):
            p = Path(part.strip())
            if not p:
                continue
            resolved = p if p.is_absolute() else (repo_root / p)
            if resolved.exists() and resolved.suffix.lower() == ".parquet":
                files.append(resolved)
        return files

    layer = os.environ.get("PERF_REAL_INPUT_LAYER", "canonical").strip().lower()
    default_glob = f"data/{system}/{layer}/**/*.parquet"
    glob_pattern = (
        os.environ.get("PERF_REAL_INPUT_GLOB", default_glob).strip() or default_glob
    )

    resolved_glob = (
        glob_pattern
        if Path(glob_pattern).is_absolute()
        else str(repo_root / glob_pattern)
    )
    matched = [Path(p) for p in glob.glob(resolved_glob, recursive=True)]
    matched = sorted(matched)

    max_files = int(os.environ.get("PERF_REAL_MAX_FILES", "0"))
    if max_files > 0:
        matched = matched[:max_files]

    return [p for p in matched if p.exists() and p.suffix.lower() == ".parquet"]


def _select_short_mode_real_files(
    *,
    files: list[Path],
    repo_root: Path,
    update_baseline: bool,
) -> tuple[list[Path], dict[str, object]]:
    """Select a deterministic short-mode subset of real files for fast iteration.

    Baseline updates always use full corpus selection.
    """
    short_mode_enabled = _env_flag("PERF_SHORT_MODE", True)
    if update_baseline:
        short_mode_enabled = False

    if not short_mode_enabled:
        reason = "disabled_for_baseline" if update_baseline else "disabled"
        return files, {
            "short_mode_enabled": False,
            "short_mode_reason": reason,
            "short_mode_fraction": 1.0,
            "short_mode_seed": None,
            "short_mode_selected_file_count": len(files),
            "short_mode_total_file_count": len(files),
        }

    fraction = float(os.environ.get("PERF_SHORT_MODE_FRACTION", "0.20"))
    if not (0.0 < fraction <= 1.0):
        raise ValueError("PERF_SHORT_MODE_FRACTION must be > 0 and <= 1.")

    if fraction >= 1.0 or len(files) <= 1:
        return files, {
            "short_mode_enabled": True,
            "short_mode_reason": "fraction_ge_1_or_single_file",
            "short_mode_fraction": fraction,
            "short_mode_seed": os.environ.get("PERF_SHORT_MODE_SEED", "20260621"),
            "short_mode_selected_file_count": len(files),
            "short_mode_total_file_count": len(files),
        }

    seed = os.environ.get("PERF_SHORT_MODE_SEED", "20260621")
    keep_count = max(1, round(len(files) * fraction))

    ranked = sorted(
        files,
        key=lambda path: hashlib.sha256(
            f"{seed}:{_stable_path_for_hash(path, repo_root)}".encode()
        ).hexdigest(),
    )
    selected = sorted(ranked[:keep_count])

    return selected, {
        "short_mode_enabled": True,
        "short_mode_reason": "deterministic_file_subset",
        "short_mode_fraction": fraction,
        "short_mode_seed": seed,
        "short_mode_selected_file_count": len(selected),
        "short_mode_total_file_count": len(files),
    }


def _select_hypothesis_mode_real_files(
    *,
    files: list[Path],
    repo_root: Path,
    update_baseline: bool,
) -> tuple[list[Path], dict[str, object]]:
    """Select a single deterministic random file per system for quick hypothesis tests.

    Baseline updates always use full corpus selection.
    """
    hypothesis_mode_enabled = _env_flag("PERF_HYPOTHESIS_MODE", False)
    if update_baseline:
        hypothesis_mode_enabled = False

    if not hypothesis_mode_enabled:
        reason = "disabled_for_baseline" if update_baseline else "disabled"
        return files, {
            "hypothesis_mode_enabled": False,
            "hypothesis_mode_reason": reason,
            "hypothesis_mode_seed": None,
            "hypothesis_mode_selected_file_count": len(files),
            "hypothesis_mode_total_file_count": len(files),
        }

    if not files:
        return files, {
            "hypothesis_mode_enabled": True,
            "hypothesis_mode_reason": "no_files",
            "hypothesis_mode_seed": os.environ.get("PERF_HYPOTHESIS_SEED", "20260621"),
            "hypothesis_mode_selected_file_count": 0,
            "hypothesis_mode_total_file_count": 0,
        }

    seed = os.environ.get("PERF_HYPOTHESIS_SEED", "20260621")
    ranked = sorted(
        files,
        key=lambda path: hashlib.sha256(
            f"{seed}:{_stable_path_for_hash(path, repo_root)}".encode()
        ).hexdigest(),
    )
    selected = sorted(ranked[:1])

    return selected, {
        "hypothesis_mode_enabled": True,
        "hypothesis_mode_reason": "deterministic_single_file",
        "hypothesis_mode_seed": seed,
        "hypothesis_mode_selected_file_count": len(selected),
        "hypothesis_mode_total_file_count": len(files),
    }


def _dataset_identity_synthetic(rows: int) -> dict[str, object]:
    return {
        "kind": "synthetic",
        "rows": rows,
        "pattern_version": "v1",
    }


def _dataset_identity_real(
    repo_root: Path, system: str, selected_files: list[Path]
) -> dict[str, object]:
    files_csv = os.environ.get("PERF_REAL_INPUT_FILES", "").strip()
    layer = os.environ.get("PERF_REAL_INPUT_LAYER", "canonical").strip().lower()
    glob_pattern = os.environ.get("PERF_REAL_INPUT_GLOB", "").strip()

    total_bytes = sum(path.stat().st_size for path in selected_files)
    identity: dict[str, object] = {
        "kind": "real_files",
        "system": system,
        "layer": layer,
        "selection_mode": "explicit_files" if files_csv else "glob",
        "glob": glob_pattern or None,
        "selected_file_count": len(selected_files),
        "selected_total_bytes": total_bytes,
        "selected_file_list_sha256": _file_list_sha256(selected_files, repo_root),
    }
    return identity


def _count_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        total += pl.scan_parquet(str(path)).select(pl.len()).collect().item()
    return total


def _resolve_perf_step_engines_override() -> tuple[
    dict[str, str] | None, dict[str, object]
]:
    raw = os.environ.get("PERF_NOOP_STEPS", "").strip()
    if not raw:
        return None, {
            "perf_noop_steps_enabled": False,
            "perf_noop_steps": [],
        }

    all_steps = [
        CANONICAL_COMPANY_TYPE_STEP,
        GENERATE_CLEANSED_COMPANY_NAME_STEP,
        DERIVE_ACRONYM_FIELD_STEP,
        ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
        ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
    ]
    aliases = {
        "canonical": CANONICAL_COMPANY_TYPE_STEP,
        "generate": GENERATE_CLEANSED_COMPANY_NAME_STEP,
        "acronym": DERIVE_ACRONYM_FIELD_STEP,
        "non_acronym": ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
        "quoted": ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
    }

    requested = [part.strip().lower() for part in raw.split(",") if part.strip()]
    selected: list[str] = []
    for token in requested:
        if token in {"all", "*"}:
            selected = list(all_steps)
            break
        if token in aliases:
            selected.append(aliases[token])
            continue
        if token in all_steps:
            selected.append(token)
            continue
        raise ValueError(
            f"Unknown PERF_NOOP_STEPS value '{token}'. "
            "Use step names, aliases (canonical, generate, acronym, non_acronym, quoted), or 'all'."
        )

    unique_selected = sorted(set(selected))
    if not unique_selected:
        return None, {
            "perf_noop_steps_enabled": False,
            "perf_noop_steps": [],
        }

    return (
        {step_name: STEP_ENGINE_NOOP for step_name in unique_selected},
        {
            "perf_noop_steps_enabled": True,
            "perf_noop_steps": unique_selected,
        },
    )


def _all_switchable_steps() -> list[str]:
    return [
        CANONICAL_COMPANY_TYPE_STEP,
        GENERATE_CLEANSED_COMPANY_NAME_STEP,
        DERIVE_ACRONYM_FIELD_STEP,
        ENSURE_NON_ACRONYM_SHORT_NAME_STEP,
        ENSURE_QUOTED_NAME_IN_CLEANSED_STEP,
    ]


class EngineMatrixResult(TypedDict):
    engine: str
    runs: int
    elapsed_seconds: list[float]
    median_elapsed_seconds: float
    throughput_rows_per_sec: float


class EngineProfileResult(TypedDict):
    elapsed_seconds: float
    rows: int
    throughput_rows_per_sec: float
    profile_path: str
    profile_summary_path: str


def _run_engine_mode_once(
    *,
    input_dir: Path,
    output_dir: Path,
    system: str,
    company_type_matcher: str,
    company_col: str,
    and_token: str,
    source_company_type_col: str | None,
    source_company_type_mapping: dict[str, str] | None,
    step_engines: dict[str, str],
    expected_rows: int,
    write_output: bool,
) -> float:
    company_type_regex, company_type_mapping = get_company_type_rules_for_country(
        system
    )

    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    processed_rows = name_cleanse(
        input_dir=input_dir,
        output_dir=output_dir,
        company_type_regex=company_type_regex,
        company_col=company_col,
        and_tokens=[and_token],
        char_whitelist=r"[^a-z0-9\s!]",
        company_type_mapping=company_type_mapping,
        company_type_matcher=company_type_matcher,
        source_company_type_col=source_company_type_col,
        source_company_type_mapping=source_company_type_mapping,
        step_engines=step_engines,
        write_output=write_output,
    )
    elapsed = max(time.perf_counter() - t0, 1e-9)
    assert processed_rows == expected_rows
    return elapsed


def _profile_engine_mode_once(
    *,
    repo_root: Path,
    input_dir: Path,
    output_dir: Path,
    system: str,
    company_type_matcher: str,
    company_col: str,
    and_token: str,
    source_company_type_col: str | None,
    source_company_type_mapping: dict[str, str] | None,
    step_engines: dict[str, str],
    expected_rows: int,
    test_name: str,
    mode_name: str,
) -> EngineProfileResult:
    company_type_regex, company_type_mapping = get_company_type_rules_for_country(
        system
    )

    shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = cProfile.Profile()
    t0 = time.perf_counter()
    processed_rows = profile.runcall(
        name_cleanse,
        input_dir=input_dir,
        output_dir=output_dir,
        company_type_regex=company_type_regex,
        company_col=company_col,
        and_tokens=[and_token],
        char_whitelist=r"[^a-z0-9\s!]",
        company_type_mapping=company_type_mapping,
        company_type_matcher=company_type_matcher,
        source_company_type_col=source_company_type_col,
        source_company_type_mapping=source_company_type_mapping,
        step_engines=step_engines,
        write_output=True,
    )
    elapsed = max(time.perf_counter() - t0, 1e-9)
    assert processed_rows == expected_rows

    artifacts_dir = _artifacts_dir(repo_root)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    profile_path = artifacts_dir / f"{test_name}_{mode_name}.pstats"
    summary_path = artifacts_dir / f"{test_name}_{mode_name}_top30.txt"
    profile.dump_stats(str(profile_path))

    with summary_path.open("w", encoding="utf-8") as handle:
        stats = pstats.Stats(profile, stream=handle).sort_stats("cumulative")
        stats.print_stats(30)

    return {
        "elapsed_seconds": elapsed,
        "rows": expected_rows,
        "throughput_rows_per_sec": expected_rows / elapsed,
        "profile_path": str(profile_path),
        "profile_summary_path": str(summary_path),
    }


@pytest.mark.skipif(
    os.environ.get("PERF_BENCHMARK", "0") != "1",
    reason="Set PERF_BENCHMARK=1 to run performance benchmarks.",
)
def test_name_cleanse_benchmark_synthetic(repo_root: Path) -> None:
    system = os.environ.get("PERF_SYSTEM", "gb").strip().lower()
    rows = int(os.environ.get("PERF_ROWS", "120000"))
    runs = int(os.environ.get("PERF_RUNS", "3"))
    max_regression_pct = float(os.environ.get("PERF_MAX_REGRESSION_PCT", "0.15"))
    update_baseline = os.environ.get("PERF_UPDATE_BASELINE", "0") == "1"
    profile_once = os.environ.get("PERF_PROFILE", "0") == "1"
    benchmark_mode = _benchmark_mode()
    company_type_matcher = (
        os.environ.get("PERF_COMPANY_TYPE_MATCHER", "trie").strip().lower() or "trie"
    )
    compute_only = _env_flag("PERF_COMPUTE_ONLY", False)
    step_engines, noop_step_meta = _resolve_perf_step_engines_override()
    _validate_run_policy(
        runs=runs,
        update_baseline=update_baseline,
        benchmark_mode=benchmark_mode,
        profile_once=profile_once,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        _build_perf_input(rows).write_parquet(input_dir / "perf.parquet")

        _run_benchmark(
            repo_root=repo_root,
            test_name="cleanse_synthetic",
            input_dir=input_dir,
            output_dir=output_dir,
            expected_rows=rows,
            system=system,
            runs=runs,
            max_regression_pct=max_regression_pct,
            update_baseline=update_baseline,
            profile_once=profile_once,
            benchmark_mode=benchmark_mode,
            dataset_identity={
                **_dataset_identity_synthetic(rows),
                **noop_step_meta,
                "compute_only": compute_only,
            },
            company_type_matcher=company_type_matcher,
            company_col="CompanyName",
            and_token="AND",
            source_company_type_col=None,
            source_company_type_mapping=None,
            step_engines=step_engines,
            write_output=not compute_only,
        )


@pytest.mark.skipif(
    os.environ.get("PERF_BENCHMARK", "0") != "1",
    reason="Set PERF_BENCHMARK=1 to run performance benchmarks.",
)
def test_name_cleanse_benchmark_real_files(repo_root: Path) -> None:
    system = os.environ.get("PERF_SYSTEM", "gb").strip().lower()
    runs = int(os.environ.get("PERF_RUNS", "1"))
    max_regression_pct = float(os.environ.get("PERF_MAX_REGRESSION_PCT", "0.15"))
    update_baseline = os.environ.get("PERF_UPDATE_BASELINE", "0") == "1"
    profile_once = os.environ.get("PERF_PROFILE", "0") == "1"
    benchmark_mode = _benchmark_mode()
    company_type_matcher = (
        os.environ.get("PERF_COMPANY_TYPE_MATCHER", "trie").strip().lower() or "trie"
    )
    compute_only = _env_flag("PERF_COMPUTE_ONLY", False)
    include_source_mapping = _env_flag("PERF_REAL_INCLUDE_SOURCE_MAPPING", True)
    step_engines, noop_step_meta = _resolve_perf_step_engines_override()
    _validate_run_policy(
        runs=runs,
        update_baseline=update_baseline,
        benchmark_mode=benchmark_mode,
        profile_once=profile_once,
    )

    system_plan = get_system_plan(system)
    company_col = _resolve_real_company_col(system_plan)
    configured_and_tokens = getattr(system_plan, "and_tokens", None) or ()
    effective_and_token = configured_and_tokens[0] if configured_and_tokens else "AND"
    source_company_type_col = (
        system_plan.company_type_column if include_source_mapping else None
    )
    source_company_type_mapping = (
        get_system_company_type_mapping(system_plan.code)
        if include_source_mapping and source_company_type_col
        else None
    )
    selected_files = _resolve_real_input_files(repo_root, system)
    selected_files, hypothesis_mode_meta = _select_hypothesis_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=update_baseline,
    )
    selected_files, short_mode_meta = _select_short_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=update_baseline,
    )
    if not selected_files:
        pytest.skip(
            "No real input parquet files matched. "
            "Set PERF_REAL_INPUT_FILES or PERF_REAL_INPUT_GLOB (and optionally PERF_SYSTEM/PERF_REAL_INPUT_LAYER)."
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy once outside timed section so benchmark measures cleanse work only.
        for src in selected_files:
            shutil.copy2(src, input_dir / src.name)

        input_files = sorted(input_dir.glob("*.parquet"))
        expected_rows = _count_rows(input_files)

        print(f"real files selected: {len(input_files)}")
        print(f"real rows selected: {expected_rows:,}")
        print(
            "hypothesis mode: "
            f"enabled={hypothesis_mode_meta['hypothesis_mode_enabled']}, "
            f"reason={hypothesis_mode_meta['hypothesis_mode_reason']}, "
            f"selected={hypothesis_mode_meta['hypothesis_mode_selected_file_count']}/"
            f"{hypothesis_mode_meta['hypothesis_mode_total_file_count']}"
        )
        print(
            "short mode: "
            f"enabled={short_mode_meta['short_mode_enabled']}, "
            f"reason={short_mode_meta['short_mode_reason']}, "
            f"fraction={short_mode_meta['short_mode_fraction']}, "
            f"selected={short_mode_meta['short_mode_selected_file_count']}/"
            f"{short_mode_meta['short_mode_total_file_count']}"
        )

        dataset_identity = _dataset_identity_real(repo_root, system, selected_files)
        dataset_identity.update(hypothesis_mode_meta)
        dataset_identity.update(short_mode_meta)
        dataset_identity.update(
            {
                "script_parity_enabled": include_source_mapping,
                "source_company_type_col": source_company_type_col,
                "source_company_type_mapping_size": (
                    len(source_company_type_mapping)
                    if source_company_type_mapping
                    else 0
                ),
                "and_token": effective_and_token,
                "compute_only": compute_only,
            }
        )
        dataset_identity.update(noop_step_meta)

        _run_benchmark(
            repo_root=repo_root,
            test_name="cleanse_real_files",
            input_dir=input_dir,
            output_dir=output_dir,
            expected_rows=expected_rows,
            system=system,
            runs=runs,
            max_regression_pct=max_regression_pct,
            update_baseline=update_baseline,
            profile_once=profile_once,
            benchmark_mode=benchmark_mode,
            dataset_identity=dataset_identity,
            company_type_matcher=company_type_matcher,
            company_col=company_col,
            and_token=effective_and_token,
            source_company_type_col=source_company_type_col,
            source_company_type_mapping=source_company_type_mapping,
            step_engines=step_engines,
            write_output=not compute_only,
        )


@pytest.mark.skipif(
    os.environ.get("PERF_BENCHMARK", "0") != "1",
    reason="Set PERF_BENCHMARK=1 to run performance benchmarks.",
)
def test_name_cleanse_query_plan_snapshot_real_files(repo_root: Path) -> None:
    """Snapshot real-files query plans into perf artifacts for plan drift inspection."""
    system = os.environ.get("PERF_SYSTEM", "gb").strip().lower()
    company_type_matcher = (
        os.environ.get("PERF_COMPANY_TYPE_MATCHER", "trie").strip().lower() or "trie"
    )
    include_source_mapping = _env_flag("PERF_REAL_INCLUDE_SOURCE_MAPPING", True)
    step_engines, noop_step_meta = _resolve_perf_step_engines_override()
    system_plan = get_system_plan(system)
    company_col = _resolve_real_company_col(system_plan)
    configured_and_tokens = getattr(system_plan, "and_tokens", None) or ()
    effective_and_token = configured_and_tokens[0] if configured_and_tokens else "AND"
    source_company_type_col = (
        system_plan.company_type_column if include_source_mapping else None
    )
    source_company_type_mapping = (
        get_system_company_type_mapping(system_plan.code)
        if include_source_mapping and source_company_type_col
        else None
    )
    update_baseline = os.environ.get("PERF_UPDATE_BASELINE", "0") == "1"
    selected_files = _resolve_real_input_files(repo_root, system)
    selected_files, hypothesis_mode_meta = _select_hypothesis_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=update_baseline,
    )
    selected_files, short_mode_meta = _select_short_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=update_baseline,
    )
    if not selected_files:
        pytest.skip(
            "No real input parquet files matched. "
            "Set PERF_REAL_INPUT_FILES or PERF_REAL_INPUT_GLOB (and optionally PERF_SYSTEM/PERF_REAL_INPUT_LAYER)."
        )

    artifacts_dir = _artifacts_dir(repo_root)
    plans_root = artifacts_dir / "query_plans" / "cleanse_real_files"
    plans_dir = plans_root / f"{system}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy once outside plan generation to keep snapshot generation deterministic.
        for src in selected_files:
            shutil.copy2(src, input_dir / src.name)

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            system
        )
        rows = name_cleanse(
            input_dir=input_dir,
            output_dir=output_dir,
            company_type_regex=company_type_regex,
            company_col=company_col,
            and_tokens=[effective_and_token],
            char_whitelist=r"[^a-z0-9\s!]",
            company_type_mapping=company_type_mapping,
            company_type_matcher=company_type_matcher,
            source_company_type_col=source_company_type_col,
            source_company_type_mapping=source_company_type_mapping,
            query_plan_output_dir=plans_dir,
            step_engines=step_engines,
        )

        input_files = sorted(input_dir.glob("*.parquet"))
        expected_rows = _count_rows(input_files)
        assert rows == expected_rows

    dataset_identity = _dataset_identity_real(repo_root, system, selected_files)
    dataset_identity.update(hypothesis_mode_meta)
    dataset_identity.update(short_mode_meta)
    dataset_identity.update(
        {
            "script_parity_enabled": include_source_mapping,
            "source_company_type_col": source_company_type_col,
            "source_company_type_mapping_size": (
                len(source_company_type_mapping) if source_company_type_mapping else 0
            ),
            "and_token": effective_and_token,
        }
    )
    dataset_identity.update(noop_step_meta)
    manifest = {
        "test": "cleanse_real_files_query_plan_snapshot",
        "system": system,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "rows": rows,
        "dataset": dataset_identity,
        "plans_dir": str(plans_dir),
        "plan_files": sorted(path.name for path in plans_dir.glob("*.plan.txt")),
        "company_type_matcher": company_type_matcher,
    }

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts_dir / "cleanse_real_files_query_plan_latest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _append_jsonl(
        artifacts_dir / "cleanse_real_files_query_plan_history.jsonl", manifest
    )

    assert manifest["plan_files"], "Expected query plan files to be generated."
    print(f"query plan snapshot latest: {manifest_path}")
    print(f"query plan snapshot dir: {plans_dir}")


@pytest.mark.skipif(
    os.environ.get("PERF_BENCHMARK", "0") != "1"
    or os.environ.get("PERF_ENGINE_MATRIX", "0") != "1",
    reason="Set PERF_BENCHMARK=1 and PERF_ENGINE_MATRIX=1 to run engine matrix benchmarks.",
)
def test_name_cleanse_benchmark_real_files_engine_matrix_all_steps(
    repo_root: Path,
) -> None:
    """Benchmark all switchable steps forced to noop/udf/polars on the same real-file corpus."""
    system = os.environ.get("PERF_SYSTEM", "gb").strip().lower()
    runs = int(os.environ.get("PERF_RUNS", "1"))
    company_type_matcher = (
        os.environ.get("PERF_COMPANY_TYPE_MATCHER", "trie").strip().lower() or "trie"
    )
    compute_only = _env_flag("PERF_COMPUTE_ONLY", False)
    include_source_mapping = _env_flag("PERF_REAL_INCLUDE_SOURCE_MAPPING", True)

    if _env_flag("PERF_PROFILE", False):
        raise ValueError(
            "PERF_PROFILE is not supported in matrix benchmark mode. "
            "Use test_name_cleanse_profile_real_files_engine_matrix_all_steps with PERF_ENGINE_MATRIX_PROFILE=1."
        )

    _validate_run_policy(
        runs=runs,
        update_baseline=False,
        benchmark_mode="regression",
        profile_once=False,
    )

    selected_files = _resolve_real_input_files(repo_root, system)
    selected_files, hypothesis_mode_meta = _select_hypothesis_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=False,
    )
    selected_files, short_mode_meta = _select_short_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=False,
    )

    if not selected_files:
        pytest.skip(
            "No real input parquet files matched. "
            "Set PERF_REAL_INPUT_FILES or PERF_REAL_INPUT_GLOB (and optionally PERF_SYSTEM/PERF_REAL_INPUT_LAYER)."
        )

    system_plan = get_system_plan(system)
    company_col = _resolve_real_company_col(system_plan)
    configured_and_tokens = getattr(system_plan, "and_tokens", None) or ()
    effective_and_token = configured_and_tokens[0] if configured_and_tokens else "AND"
    source_company_type_col = (
        system_plan.company_type_column if include_source_mapping else None
    )
    source_company_type_mapping = (
        get_system_company_type_mapping(system_plan.code)
        if include_source_mapping and source_company_type_col
        else None
    )

    engine_modes = {
        "noop": STEP_ENGINE_NOOP,
        "udf": STEP_ENGINE_UDF,
        "polars": STEP_ENGINE_POLARS,
    }
    all_steps = _all_switchable_steps()

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for src in selected_files:
            shutil.copy2(src, input_dir / src.name)

        input_files = sorted(input_dir.glob("*.parquet"))
        expected_rows = _count_rows(input_files)

        matrix_results: dict[str, EngineMatrixResult] = {}
        for mode_name, engine_name in engine_modes.items():
            step_engines = {step_name: engine_name for step_name in all_steps}
            elapsed_seconds: list[float] = []
            for _ in range(runs):
                elapsed_seconds.append(
                    _run_engine_mode_once(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        system=system,
                        company_type_matcher=company_type_matcher,
                        company_col=company_col,
                        and_token=effective_and_token,
                        source_company_type_col=source_company_type_col,
                        source_company_type_mapping=source_company_type_mapping,
                        step_engines=step_engines,
                        expected_rows=expected_rows,
                        write_output=not compute_only,
                    )
                )

            median_elapsed = _median(elapsed_seconds)
            matrix_results[mode_name] = {
                "engine": engine_name,
                "runs": runs,
                "elapsed_seconds": elapsed_seconds,
                "median_elapsed_seconds": median_elapsed,
                "throughput_rows_per_sec": expected_rows / median_elapsed,
            }

        dataset_identity = _dataset_identity_real(repo_root, system, selected_files)
        dataset_identity.update(hypothesis_mode_meta)
        dataset_identity.update(short_mode_meta)
        dataset_identity.update(
            {
                "script_parity_enabled": include_source_mapping,
                "source_company_type_col": source_company_type_col,
                "source_company_type_mapping_size": (
                    len(source_company_type_mapping)
                    if source_company_type_mapping
                    else 0
                ),
                "and_token": effective_and_token,
                "compute_only": compute_only,
                "all_steps": all_steps,
                "engine_modes": list(engine_modes.keys()),
            }
        )

        relative = {
            "udf_vs_noop_rows_per_s": (
                matrix_results["udf"]["throughput_rows_per_sec"]
                / matrix_results["noop"]["throughput_rows_per_sec"]
            ),
            "polars_vs_noop_rows_per_s": (
                matrix_results["polars"]["throughput_rows_per_sec"]
                / matrix_results["noop"]["throughput_rows_per_sec"]
            ),
            "polars_vs_udf_rows_per_s": (
                matrix_results["polars"]["throughput_rows_per_sec"]
                / matrix_results["udf"]["throughput_rows_per_sec"]
            ),
        }

        payload = {
            "test": "cleanse_real_files_engine_matrix_all_steps",
            "benchmark_mode": "matrix",
            "system": system,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "rows": expected_rows,
            "dataset": dataset_identity,
            "company_type_matcher": company_type_matcher,
            "results": matrix_results,
            "relative": relative,
        }

    artifacts_dir = _artifacts_dir(repo_root)
    latest_path = (
        artifacts_dir / "cleanse_real_files_engine_matrix_all_steps_latest.json"
    )
    history_path = (
        artifacts_dir / "cleanse_real_files_engine_matrix_all_steps_history.jsonl"
    )
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_jsonl(history_path, payload)

    print(
        "engine matrix benchmark: "
        f"system={system}, rows={expected_rows:,}, runs={runs}, "
        f"noop={matrix_results['noop']['throughput_rows_per_sec']:,.0f} rows/s, "
        f"udf={matrix_results['udf']['throughput_rows_per_sec']:,.0f} rows/s, "
        f"polars={matrix_results['polars']['throughput_rows_per_sec']:,.0f} rows/s"
    )
    print(
        "relative rows/s: "
        f"udf_vs_noop={relative['udf_vs_noop_rows_per_s']:.3f}x, "
        f"polars_vs_noop={relative['polars_vs_noop_rows_per_s']:.3f}x, "
        f"polars_vs_udf={relative['polars_vs_udf_rows_per_s']:.3f}x"
    )
    print(f"latest: {latest_path}")
    print(f"history: {history_path}")


@pytest.mark.skipif(
    os.environ.get("PERF_ENGINE_MATRIX_PROFILE", "0") != "1",
    reason="Set PERF_ENGINE_MATRIX_PROFILE=1 to run engine matrix profiling.",
)
def test_name_cleanse_profile_real_files_engine_matrix_all_steps(
    repo_root: Path,
) -> None:
    """Profile noop/udf/polars with all switchable steps forced per mode (separate from throughput benchmarks)."""
    system = os.environ.get("PERF_SYSTEM", "gb").strip().lower()
    company_type_matcher = (
        os.environ.get("PERF_COMPANY_TYPE_MATCHER", "trie").strip().lower() or "trie"
    )
    include_source_mapping = _env_flag("PERF_REAL_INCLUDE_SOURCE_MAPPING", True)

    selected_files = _resolve_real_input_files(repo_root, system)
    selected_files, hypothesis_mode_meta = _select_hypothesis_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=False,
    )
    selected_files, short_mode_meta = _select_short_mode_real_files(
        files=selected_files,
        repo_root=repo_root,
        update_baseline=False,
    )

    if not selected_files:
        pytest.skip(
            "No real input parquet files matched. "
            "Set PERF_REAL_INPUT_FILES or PERF_REAL_INPUT_GLOB (and optionally PERF_SYSTEM/PERF_REAL_INPUT_LAYER)."
        )

    system_plan = get_system_plan(system)
    company_col = _resolve_real_company_col(system_plan)
    configured_and_tokens = getattr(system_plan, "and_tokens", None) or ()
    effective_and_token = configured_and_tokens[0] if configured_and_tokens else "AND"
    source_company_type_col = (
        system_plan.company_type_column if include_source_mapping else None
    )
    source_company_type_mapping = (
        get_system_company_type_mapping(system_plan.code)
        if include_source_mapping and source_company_type_col
        else None
    )

    engine_modes = {
        "noop": STEP_ENGINE_NOOP,
        "udf": STEP_ENGINE_UDF,
        "polars": STEP_ENGINE_POLARS,
    }
    all_steps = _all_switchable_steps()
    test_name = "cleanse_real_files_engine_matrix_profile_all_steps"

    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        for src in selected_files:
            shutil.copy2(src, input_dir / src.name)

        input_files = sorted(input_dir.glob("*.parquet"))
        expected_rows = _count_rows(input_files)

        profile_results: dict[str, EngineProfileResult] = {}
        for mode_name, engine_name in engine_modes.items():
            step_engines = {step_name: engine_name for step_name in all_steps}
            profile_results[mode_name] = _profile_engine_mode_once(
                repo_root=repo_root,
                input_dir=input_dir,
                output_dir=output_dir,
                system=system,
                company_type_matcher=company_type_matcher,
                company_col=company_col,
                and_token=effective_and_token,
                source_company_type_col=source_company_type_col,
                source_company_type_mapping=source_company_type_mapping,
                step_engines=step_engines,
                expected_rows=expected_rows,
                test_name=test_name,
                mode_name=mode_name,
            )

        dataset_identity = _dataset_identity_real(repo_root, system, selected_files)
        dataset_identity.update(hypothesis_mode_meta)
        dataset_identity.update(short_mode_meta)
        dataset_identity.update(
            {
                "script_parity_enabled": include_source_mapping,
                "source_company_type_col": source_company_type_col,
                "source_company_type_mapping_size": (
                    len(source_company_type_mapping)
                    if source_company_type_mapping
                    else 0
                ),
                "and_token": effective_and_token,
                "all_steps": all_steps,
                "engine_modes": list(engine_modes.keys()),
            }
        )

        payload = {
            "test": test_name,
            "benchmark_mode": "profile_matrix",
            "system": system,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "rows": expected_rows,
            "dataset": dataset_identity,
            "company_type_matcher": company_type_matcher,
            "results": profile_results,
        }

    artifacts_dir = _artifacts_dir(repo_root)
    latest_path = artifacts_dir / f"{test_name}_latest.json"
    history_path = artifacts_dir / f"{test_name}_history.jsonl"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _append_jsonl(history_path, payload)

    print(
        "engine matrix profiling complete: "
        f"system={system}, rows={expected_rows:,}, "
        f"noop={profile_results['noop']['throughput_rows_per_sec']:,.0f} rows/s, "
        f"udf={profile_results['udf']['throughput_rows_per_sec']:,.0f} rows/s, "
        f"polars={profile_results['polars']['throughput_rows_per_sec']:,.0f} rows/s"
    )
    print(f"latest: {latest_path}")
    print(f"history: {history_path}")
