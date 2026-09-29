"""Write training and comparison metrics as JSON."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .comparison import ComparisonReport
from .pair_train import TrainedPairModelResult
from .train import TrainedModelResult


def save_run_summary_json(
    results: dict[str, TrainedModelResult], output_path: Path
) -> None:
    payload: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        payload[name] = {
            "validation_metrics": asdict(result.validation_metrics),
            "test_metrics": asdict(result.test_metrics),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _category_payload(
    results: dict[str, TrainedPairModelResult],
) -> dict[str, dict[str, object]]:
    return {
        name: {
            "validation_metrics": asdict(result.validation_metrics),
            "test_metrics": asdict(result.test_metrics),
        }
        for name, result in results.items()
    }


def save_comparison_report_json(report: ComparisonReport, output_path: Path) -> None:
    """Write a `ComparisonReport`'s per-model metrics as one side-by-side JSON payload.

    Same per-model shape `save_run_summary_json()` writes (`validation_metrics`/`test_metrics`),
    grouped under top-level `"supervised"`/`"self_supervised"` keys instead of one flat dict, so a
    reader sees which family a name belongs to without cross-referencing the
    `compare_pair_models()` call site that produced `report`.
    """
    payload = {
        "supervised": _category_payload(report.supervised),
        "self_supervised": _category_payload(report.self_supervised),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
