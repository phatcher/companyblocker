"""Command-line settings this area's scripts read, declared as plain data.

A script reads these through `scripts/cli_common.py` to build its flags, name
each default in its help, and report which settings a run took and where
each value came from. Nothing here builds a parser; the declaration shape is
the convention `cli_common` documents, and a script decides whether a
setting is a flag or a prefixed key and whether it overrides the owner's
default with one of its own.

A purely structural flag -- which systems, which run date, which input or
output path -- is not declared here: it names *which* run, not a knob the
work itself reads, and stays an ordinary `parser.add_argument` in the script
that owns it. What is declared is every flag whose value reaches an
analysis function's own parameter with a default that function (or the
package it calls into) already carries -- so a script's help names the same
default the code actually applies, rather than a second number authored only
in the script's `argparse` call.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from company_tokenize.tfidf import select_rare_token_candidates as _select_rare_tokens

from analysis.match_metrics import run_match_analysis as _run_match_analysis
from analysis.noise_layers import run_noise_layer_analysis as _run_noise_layer_analysis
from analysis.token_metrics import (
    run_phase1_token_analysis as _run_phase1_token_analysis,
)
from analysis.token_zipf import run_token_zipf_analysis as _run_token_zipf_analysis
from analysis.vocab_shrinkage import (
    DEFAULT_BUDGET_POINTS,
    DEFAULT_V_MIN,
)


def _default(function: Callable[..., object], name: str) -> object:
    """The default `function`'s own parameter `name` carries -- the single
    source of truth a declared setting's default is checked against."""
    return inspect.signature(function).parameters[name].default


SETTINGS: tuple[dict[str, object], ...] = (
    # analyze_noise_layers.py
    {
        "name": "cleanse_tier",
        "type": "str",
        "default": _default(_run_noise_layer_analysis, "cleanse_tier"),
        "help": (
            "The non-raw name-column tier that stands in for the cleanse "
            "stage's output; the cleanse layer's removed set is raw-tier "
            "vocabulary minus this tier's vocabulary."
        ),
    },
    # analyze_token_rarity.py
    {
        "name": "max_document_frequency",
        "type": "int",
        "default": _default(_select_rare_tokens, "max_document_frequency"),
        "help": (
            "Maximum corpus document-frequency (count) for a token to count as rare."
        ),
    },
    {
        "name": "max_document_frequency_pct",
        "type": "float",
        "default": _default(_select_rare_tokens, "max_document_frequency_pct"),
        "help": "Maximum document-frequency percentage for a token to count as rare.",
    },
    {
        "name": "min_token_length",
        "type": "int",
        "default": _default(_select_rare_tokens, "min_token_length"),
        "help": "Minimum token length to include in the rare-token extraction.",
    },
    # analyze_token_zipf.py
    {
        "name": "wordfreq_top_n",
        "type": "int",
        "default": _default(_run_token_zipf_analysis, "wordfreq_top_n"),
        "help": "Number of top wordfreq words used to build the general-language reference curve.",
    },
    {
        "name": "highlight_regressions",
        "type": "bool",
        "default": _default(_run_token_zipf_analysis, "highlight_regressions"),
        "help": (
            "On the OOV/hapax bar charts, highlight a system's cleansed-tier "
            "bar when it's worse than that system's raw-tier bar."
        ),
    },
    {
        "name": "compare_power_law",
        "type": "bool",
        "default": _default(_run_token_zipf_analysis, "compare_power_law"),
        "help": "Also fit and compare power-law vs log-normal per system/tier.",
    },
    {
        "name": "force",
        "type": "bool",
        "default": _default(_run_token_zipf_analysis, "force"),
        "help": "Replace a run that already exists for the run date.",
    },
    # analyze_vocab_shrinkage.py
    {
        "name": "v_min",
        "type": "int",
        "default": DEFAULT_V_MIN,
        "help": "Smallest budget in the generated log grid.",
    },
    {
        "name": "budget_points",
        "type": "int",
        "default": DEFAULT_BUDGET_POINTS,
        "help": "Number of points in the generated log grid.",
    },
    # analyze_matches.py
    {
        "name": "max_rows",
        "type": "int",
        "default": _default(_run_match_analysis, "max_rows"),
        "help": "Optional row cap per input dataset, for fast iteration.",
    },
    # measure_prefix_suffix_divergence.py. No analysis-module function to
    # read this default from: the measurement lives entirely in the script.
    # Mirrors that script's own DEFAULT_MAX_DIVERGENCE constant, which the
    # script's surface passes as this setting's default.
    {
        "name": "max_divergence",
        "type": "int",
        "default": 2,
        "help": (
            "Maximum token-count difference between the two names still "
            "counted as a bounded prefix/suffix run."
        ),
    },
    # analyze_tokens.py
    {
        "name": "top_n",
        "type": "int",
        "default": _default(_run_phase1_token_analysis, "top_n"),
        "help": "Top N tokens per country for analysis outputs (report and country plot).",
    },
    {
        "name": "stoplist_min_coverage_ratio",
        "type": "float",
        "default": _default(_run_phase1_token_analysis, "stoplist_min_coverage_ratio"),
        "help": "Minimum cross-system coverage ratio for stoplist candidates (0..1).",
    },
    {
        "name": "stoplist_min_global_df_pct",
        "type": "float",
        "default": _default(_run_phase1_token_analysis, "stoplist_min_global_df_pct"),
        "help": "Minimum global document-frequency percentage for stoplist candidates (0..1).",
    },
    {
        "name": "engine",
        "type": "str",
        "default": _default(_run_phase1_token_analysis, "engine"),
        "choices": ("polars", "duckdb"),
        "help": "Execution backend for token aggregation.",
    },
    {
        "name": "threads",
        "type": "int",
        "default": _default(_run_phase1_token_analysis, "threads"),
        "help": "DuckDB worker threads (0 or unset means auto/use all available cores).",
    },
    {
        "name": "verbose_progress",
        "type": "bool",
        "default": _default(_run_phase1_token_analysis, "verbose_progress"),
        "help": "Emit per-system heartbeat progress lines during aggregation.",
    },
)
"""Every setting this area's scripts read that a script maps onto its own
surface (a flag or a prefixed key), each carrying the default the analysis
function it reaches already applies."""
