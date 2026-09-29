from __future__ import annotations

import inspect

from company_tokenize.tfidf import select_rare_token_candidates

from analysis._cli_helper import SETTINGS
from analysis.match_metrics import run_match_analysis
from analysis.noise_layers import run_noise_layer_analysis
from analysis.token_metrics import run_phase1_token_analysis
from analysis.token_zipf import run_token_zipf_analysis
from analysis.vocab_shrinkage import DEFAULT_BUDGET_POINTS, DEFAULT_V_MIN
from scripts.measure_prefix_suffix_divergence import DEFAULT_MAX_DIVERGENCE

_TYPES = {"str", "int", "float", "bool"}


def _by_name() -> dict[str, dict[str, object]]:
    return {str(setting["name"]): setting for setting in SETTINGS}


def test_every_setting_is_a_well_formed_declaration_with_a_unique_name() -> None:
    names = [setting["name"] for setting in SETTINGS]

    assert len(names) == len(set(names))
    for setting in SETTINGS:
        assert {"name", "type", "default", "help"} <= set(setting)
        assert setting["type"] in _TYPES


def test_declared_defaults_are_the_ones_the_analysis_functions_themselves_use() -> None:
    settings = _by_name()

    assert settings["cleanse_tier"]["default"] == (
        inspect.signature(run_noise_layer_analysis).parameters["cleanse_tier"].default
    )
    assert settings["max_document_frequency"]["default"] == (
        inspect.signature(select_rare_token_candidates)
        .parameters["max_document_frequency"]
        .default
    )
    assert settings["max_document_frequency_pct"]["default"] == (
        inspect.signature(select_rare_token_candidates)
        .parameters["max_document_frequency_pct"]
        .default
    )
    assert settings["min_token_length"]["default"] == (
        inspect.signature(select_rare_token_candidates)
        .parameters["min_token_length"]
        .default
    )
    assert settings["wordfreq_top_n"]["default"] == (
        inspect.signature(run_token_zipf_analysis).parameters["wordfreq_top_n"].default
    )
    assert settings["highlight_regressions"]["default"] == (
        inspect.signature(run_token_zipf_analysis)
        .parameters["highlight_regressions"]
        .default
    )
    assert settings["compare_power_law"]["default"] == (
        inspect.signature(run_token_zipf_analysis)
        .parameters["compare_power_law"]
        .default
    )
    assert settings["force"]["default"] == (
        inspect.signature(run_token_zipf_analysis).parameters["force"].default
    )
    assert settings["v_min"]["default"] == DEFAULT_V_MIN
    assert settings["budget_points"]["default"] == DEFAULT_BUDGET_POINTS
    assert settings["max_divergence"]["default"] == DEFAULT_MAX_DIVERGENCE
    assert settings["max_rows"]["default"] == (
        inspect.signature(run_match_analysis).parameters["max_rows"].default
    )
    assert settings["top_n"]["default"] == (
        inspect.signature(run_phase1_token_analysis).parameters["top_n"].default
    )
    assert settings["stoplist_min_coverage_ratio"]["default"] == (
        inspect.signature(run_phase1_token_analysis)
        .parameters["stoplist_min_coverage_ratio"]
        .default
    )
    assert settings["stoplist_min_global_df_pct"]["default"] == (
        inspect.signature(run_phase1_token_analysis)
        .parameters["stoplist_min_global_df_pct"]
        .default
    )
    assert settings["engine"]["default"] == (
        inspect.signature(run_phase1_token_analysis).parameters["engine"].default
    )
    assert settings["threads"]["default"] == (
        inspect.signature(run_phase1_token_analysis).parameters["threads"].default
    )
    assert settings["verbose_progress"]["default"] == (
        inspect.signature(run_phase1_token_analysis)
        .parameters["verbose_progress"]
        .default
    )
