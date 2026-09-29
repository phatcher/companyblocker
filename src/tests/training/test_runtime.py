from __future__ import annotations

import os

import pytest

from training.runtime import format_eta_hms, percentile_75, resolve_optimize_max_workers


def test_resolve_optimize_max_workers_uses_requested_within_job_count():
    assert resolve_optimize_max_workers(3, 10) == 3


def test_resolve_optimize_max_workers_caps_requested_to_job_count():
    assert resolve_optimize_max_workers(8, 2) == 2


def test_resolve_optimize_max_workers_rejects_non_positive_requested():
    with pytest.raises(ValueError, match="--max-workers"):
        resolve_optimize_max_workers(0, 4)


def test_resolve_optimize_max_workers_defaults_to_cpu_minus_one(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert resolve_optimize_max_workers(None, 20) == 7


def test_resolve_optimize_max_workers_returns_one_for_no_jobs():
    assert resolve_optimize_max_workers(None, 0) == 1


def test_format_eta_hms_handles_unknown_and_rounding():
    assert format_eta_hms(None) == "unknown"
    assert format_eta_hms(-1.0) == "unknown"
    assert format_eta_hms(3661.4) == "01:01:01"


def test_percentile_75_handles_empty_and_values():
    assert percentile_75([]) is None
    assert percentile_75([1.0, 2.0, 3.0, 4.0]) == 3.0
