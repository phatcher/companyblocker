"""Two-dataset blocking workflow orchestration."""

from .comparison import (
    StrategyRunEntry,
    build_strategy_comparison,
    combine_strategy_comparisons,
    diff_pair_recovery,
    summarize_runtime_scaling,
)
from .contracts import BlockingDatasetDescriptor, BlockingRunConfig, BlockingRunResult
from .loader import load_dataset_descriptor, load_run_config
from .reporting import (
    build_blocking_summary,
    write_blocking_report,
    write_pair_recovery_report,
    write_strategy_comparison_aggregate,
    write_strategy_comparison_report,
)
from .workflow import execute_blocking_run, execute_name_variant_recovery_run

__all__ = [
    "BlockingDatasetDescriptor",
    "BlockingRunConfig",
    "BlockingRunResult",
    "StrategyRunEntry",
    "build_blocking_summary",
    "build_strategy_comparison",
    "combine_strategy_comparisons",
    "diff_pair_recovery",
    "execute_blocking_run",
    "execute_name_variant_recovery_run",
    "load_dataset_descriptor",
    "load_run_config",
    "summarize_runtime_scaling",
    "write_blocking_report",
    "write_pair_recovery_report",
    "write_strategy_comparison_aggregate",
    "write_strategy_comparison_report",
]
