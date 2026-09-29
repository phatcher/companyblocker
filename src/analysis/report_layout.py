"""Where one report run keeps its own `metrics/`, `viz/` and `reports/`
subdirectories.

`workspace.artifact_layout.analysis_report_run_dir` resolves the run
directory itself and says the shape beneath it is each report's own
decision -- every report here makes the same one, so this module is that
decision, made once: every caller that used to spell `run_root / "metrics"`,
`run_root / "viz"` or `run_root / "reports"` inline now calls the function
here instead, whether it is building its own run's paths or reading another
report's already-written run (`noise_layers.py` and `vocab_shrinkage.py`
both read a `token_zipf` run's `metrics/` this way).
"""

from __future__ import annotations

from pathlib import Path

METRICS_DIR_NAME = "metrics"

VIZ_DIR_NAME = "viz"

REPORTS_DIR_NAME = "reports"


def metrics_dir(run_root: Path) -> Path:
    """Where one report run keeps its own tabular output."""
    return run_root / METRICS_DIR_NAME


def viz_dir(run_root: Path) -> Path:
    """Where one report run keeps its own pictures."""
    return run_root / VIZ_DIR_NAME


def reports_dir(run_root: Path) -> Path:
    """Where one report run keeps its own written-up (markdown) reports."""
    return run_root / REPORTS_DIR_NAME
