from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis.token_zipf import (
    NAME_TIER_COLUMNS,
    build_trim_settings,
    compare_power_law_and_lognormal,
    compute_name_level_incidence,
    compute_word_token_stats,
    discover_cleansed_files,
    draw_zipf_curve,
    find_latest_zipf_summary_for_system,
    fit_zipf_slope,
    load_persisted_power_law_fit,
    load_persisted_word_token_stats,
    render_head_words_plot,
    render_zipf_plot,
    run_token_zipf_analysis,
    run_trim_profile_sweep,
    trim_text_expr,
)
from workspace.artifact_layout import analysis_report_run_dir, tokenizer_scope_dir
from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir
from workspace.roots import WorkspaceRoots


def _cleansed_dir(roots: WorkspaceRoots, system: str) -> Path:
    """This file's one expression for a system's cleansed layer under a test
    root, composed through the workspace layout rather than restated as
    `"data" / <system> / "cleansed"` at each of the twenty-odd call sites that
    need one."""
    return system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)


def _write_persisted_zipf_run(
    roots: WorkspaceRoots,
    system: str,
    run_date: str,
    tiers: tuple[str, ...] = ("raw", "basic", "cleansed"),
) -> None:
    """The artefacts a finished `run_token_zipf_analysis` leaves behind, written
    directly.

    The resolvers below read a run's summary and its per-tier stats tables; producing
    those by running the analysis costs seconds per call and asserts nothing about the
    resolution rule under test, which is which run a directory of them resolves to.
    """
    stats_dir = analysis_report_run_dir(roots, "token_zipf", run_date) / "metrics"
    stats_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system": [system] * len(tiers),
            "tier": list(tiers),
            "total_docs": [4] * len(tiers),
        }
    ).write_parquet(stats_dir / "token_zipf_summary.parquet")
    for tier in tiers:
        pl.DataFrame(
            {"token": ["acme", "ltd"], "document_frequency": [2, 3]}
        ).write_parquet(stats_dir / f"{system}_{tier}_token_stats.parquet")


def _write_cleansed_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "name": ["I B M", "Acme Systems Ltd", "Acme Holdings Ltd", "Beta Ltd"],
            "name_cleansed_basic": [
                "ibm",
                "acme systems ltd",
                "acme holdings ltd",
                "beta ltd",
            ],
            "name_cleansed": ["ibm", "acme systems", "acme holdings", "beta"],
        }
    ).write_parquet(path)


def test_fit_zipf_slope_recovers_known_power_law():
    ranks = [1.0, 2.0, 4.0, 8.0, 16.0]
    frequencies = [1000.0, 500.0, 250.0, 125.0, 62.5]  # exact slope -1 power law

    slope = fit_zipf_slope(ranks, frequencies)

    assert slope == pytest.approx(-1.0)


def test_fit_zipf_slope_handles_degenerate_input():
    assert fit_zipf_slope([], []) == 0.0
    assert fit_zipf_slope([1.0], [10.0]) == 0.0


def test_discover_cleansed_files_matches_system_prefixed_parquet(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-002.parquet")
    _cleansed_dir(workspace_roots, "ie").mkdir(parents=True, exist_ok=True)

    files = discover_cleansed_files(workspace_roots, "ie")

    assert len(files) == 2
    assert all(f.name.startswith("ie-") for f in files)


def test_discover_cleansed_files_excludes_name_variant_companion_files(
    workspace_roots: WorkspaceRoots,
):
    # A names-corpus companion file is one row per name-variant, not one row
    # per entity -- must never be treated as a regular cleansed shard.
    _write_cleansed_fixture(
        _cleansed_dir(workspace_roots, "gleif") / "gleif-001.parquet"
    )
    _write_cleansed_fixture(
        _cleansed_dir(workspace_roots, "gleif") / "gleif-names-001.parquet"
    )

    files = discover_cleansed_files(workspace_roots, "gleif")

    assert [f.name for f in files] == ["gleif-001.parquet"]


def test_discover_cleansed_files_uses_jurisdiction_partitions_when_merged(
    workspace_roots: WorkspaceRoots,
):
    # Once a system's cleansed data is consolidated into jurisdiction_code=*/
    # partitions (the cleansed/cleansed_merge merge), that partitioned
    # data is the only valid source -- a same-named chunks/ directory can
    # coexist as raw, not-yet-merged staging output and must be ignored, not
    # picked up by a blind recursive glob. Regression test for the bug found
    # running this for real against gb/fr/ie/gleif/offeneregister's current
    # on-disk layout.
    cleansed_dir = _cleansed_dir(workspace_roots, "ie")
    _write_cleansed_fixture(
        cleansed_dir / "jurisdiction_code=ie" / "part-00001.parquet"
    )
    _write_cleansed_fixture(cleansed_dir / "chunks" / "ie-001.parquet")

    files = discover_cleansed_files(workspace_roots, "ie")

    assert len(files) == 1
    assert files[0].name == "part-00001.parquet"
    assert "chunks" not in files[0].parts


def test_compute_word_token_stats_counts_document_frequency_not_occurrence(
    tmp_path: Path,
):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"name_cleansed": ["acme systems", "acme holdings", "beta ltd"]}
    ).write_parquet(corpus_path)

    stats, total_docs = compute_word_token_stats([corpus_path], "name_cleansed")

    assert total_docs == 3
    acme_row = stats.filter(pl.col("token") == "acme")
    assert acme_row.get_column("document_frequency").item() == 2
    beta_row = stats.filter(pl.col("token") == "beta")
    assert beta_row.get_column("document_frequency").item() == 1


def test_compute_word_token_stats_is_unaffected_by_case_or_missing_rows(tmp_path: Path):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"name_cleansed": ["Acme Ltd", None, "", "acme ltd"]}).write_parquet(
        corpus_path
    )

    stats, total_docs = compute_word_token_stats([corpus_path], "name_cleansed")

    assert total_docs == 2  # null/empty rows excluded
    assert (
        stats.filter(pl.col("token") == "acme").get_column("document_frequency").item()
        == 2
    )


def test_compute_word_token_stats_keeps_accented_words_intact(tmp_path: Path):
    """Regression: found running this for real on offeneregister's raw tier
    -- an ASCII-only token pattern fragments any diacritic-bearing word at
    the accented character (e.g. 'beschraenkter' -> 'BESCHR'/'NKTER',
    'societe' -> 'SOCI'/'T'/'RALE'), corrupting raw-tier divergence numbers
    for any system with accented source text (fr, offeneregister/de)."""
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame({"name": ["beschränkter haftung", "société générale"]}).write_parquet(
        corpus_path
    )

    stats, _ = compute_word_token_stats([corpus_path], "name")

    tokens = set(stats.get_column("token").to_list())
    assert "beschränkter" in tokens
    assert "société" in tokens
    assert "générale" in tokens
    assert "generale" not in tokens  # sanity: not silently ASCII-folding either
    assert "beschr" not in tokens
    assert "nkter" not in tokens


def test_raw_tier_fragments_single_char_acronym_that_basic_tier_collapses(
    tmp_path: Path,
):
    """Grounds the confound the company_cleanse README records under
    "`name_cleansed_basic` already canonicalizes": raw text carries 'I B M'
    as three separate single-character tokens, while the basic tier already
    sees the collapsed 'IBM' as one token."""
    corpus_path = tmp_path / "corpus.parquet"
    _write_cleansed_fixture(corpus_path)

    raw_stats, _ = compute_word_token_stats([corpus_path], NAME_TIER_COLUMNS["raw"])
    basic_stats, _ = compute_word_token_stats([corpus_path], NAME_TIER_COLUMNS["basic"])

    raw_tokens = set(raw_stats.get_column("token").to_list())
    basic_tokens = set(basic_stats.get_column("token").to_list())

    assert {"i", "b", "m"}.issubset(raw_tokens)
    assert "ibm" not in raw_tokens
    assert "ibm" in basic_tokens
    assert not {"i", "b", "m"}.issubset(basic_tokens)


@pytest.mark.graphics
def test_render_zipf_plot_writes_png_and_returns_fitted_slope(tmp_path: Path):
    stats = pl.DataFrame(
        {
            "token": ["A", "B", "C", "D"],
            "document_frequency": [1200, 600, 400, 300],  # exact freq = 1200/rank
        }
    )
    output_path = tmp_path / "zipf.png"

    result = render_zipf_plot(stats, None, output_path, title="test")

    assert output_path.exists()
    assert result["corpus_slope"] == pytest.approx(-1.0)
    assert result["reference_slope"] == 0.0  # no wordfreq curve supplied


def test_draw_zipf_curve_returns_only_the_whole_curve_slopes():
    """The head/tail split and its slopes are gone: a share threshold marked
    how many legal-form words a tier carried, not where the slope changed,
    and a head of one rank had no slope. The head is `draw_head_words`'."""
    stats = pl.DataFrame(
        {
            "token": ["gmbh", "ev", "kg", "co", "der", "und"],
            "document_frequency": [3000, 700, 600, 550, 100, 90],
            "document_frequency_pct": [0.30, 0.04, 0.03, 0.02, 0.01, 0.009],
        }
    )
    fig, ax = plt.subplots()
    try:
        result = draw_zipf_curve(ax, stats, None, title="test")
    finally:
        plt.close(fig)

    assert set(result) == {"corpus_slope", "reference_slope"}


def test_render_head_words_plot_returns_the_words_in_rank_order_without_rendering(
    tmp_path: Path,
):
    stats = pl.DataFrame(
        {
            "token": ["services", "ltd", "consulting", "management"],
            "document_frequency": [4450, 95290, 1470, 2860],
            "document_frequency_pct": [0.0445, 0.9529, 0.0147, 0.0286],
        }
    )

    words = render_head_words_plot(
        stats, tmp_path / "head_words.png", title="test", top_n=3, render=False
    )

    assert words == ["ltd", "services", "management"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.graphics
def test_render_head_words_plot_writes_png(tmp_path: Path):
    stats = pl.DataFrame(
        {"token": ["ltd", "services"], "document_frequency": [9529, 445]}
    )
    output_path = tmp_path / "head_words.png"

    words = render_head_words_plot(stats, output_path, title="test")

    assert output_path.exists()
    assert words == ["ltd", "services"]


def test_compare_power_law_and_lognormal_returns_fit_metrics():
    ranks = list(range(1, 200))
    document_frequency = [max(1, round(2000 / r)) for r in ranks]
    stats = pl.DataFrame(
        {
            "token": [f"t{r}" for r in ranks],
            "document_frequency": document_frequency,
        }
    )

    result = compare_power_law_and_lognormal(stats)

    assert result["n_tokens"] == 199
    assert result["alpha"] > 1.0
    assert 0.0 <= result["ks_statistic"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0


def test_compare_power_law_and_lognormal_respects_skip_top_n():
    ranks = list(range(1, 200))
    document_frequency = [max(1, round(2000 / r)) for r in ranks]
    stats = pl.DataFrame(
        {
            "token": [f"t{r}" for r in ranks],
            "document_frequency": document_frequency,
        }
    )

    result = compare_power_law_and_lognormal(stats, skip_top_n=10)

    assert result["n_tokens"] == 189


def test_compare_power_law_and_lognormal_handles_degenerate_input():
    stats = pl.DataFrame({"token": ["a"], "document_frequency": [5]})

    result = compare_power_law_and_lognormal(stats)

    assert result["n_tokens"] == 1.0
    assert result["p_value"] == 1.0


def test_compute_name_level_incidence_flags_names_touched_by_hapax_or_oov_tokens(
    tmp_path: Path,
):
    corpus_path = tmp_path / "corpus.parquet"
    pl.DataFrame(
        {"name_cleansed": ["acme systems", "acme holdings", "zyx unique corp"]}
    ).write_parquet(corpus_path)

    rarity_lookup = pl.DataFrame(
        {
            "token": ["acme", "systems", "holdings", "zyx", "unique", "corp"],
            "document_frequency": [2, 1, 1, 1, 1, 1],
            "unseen_in_wordfreq": [False, False, False, True, False, False],
        }
    )

    incidence = compute_name_level_incidence(
        [corpus_path], "name_cleansed", rarity_lookup
    )

    assert incidence["total_names"] == 3
    # every name has at least one hapax token (SYSTEMS/HOLDINGS/ZYX-UNIQUE-CORP)
    assert incidence["names_with_hapax_pct"] == pytest.approx(1.0)
    # only the third name contains the wordfreq-unseen token ZYX
    assert incidence["names_with_unseen_in_wordfreq_pct"] == pytest.approx(1 / 3)
    # 1 of 7 total token occurrences is unseen (ZYX)
    assert incidence["occurrence_weighted_unseen_pct"] == pytest.approx(1 / 7)


@pytest.mark.integration
@pytest.mark.graphics
def test_run_token_zipf_analysis_skips_tiers_missing_from_a_systems_schema(
    workspace_roots: WorkspaceRoots,
):
    """Not every source has migrated to the standardized name-tier contract
    (dbpedia's real cleansed data has `companyname_cleansed`, no
    `name_cleansed_basic` -- found running this analysis for real). A system
    missing a tier's column should be skipped, not crash the whole sweep."""
    cleansed_path = _cleansed_dir(workspace_roots, "dbpedia") / "dbpedia-001.parquet"
    cleansed_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"name": ["Acme Systems"], "companyname_cleansed": ["acme systems"]}
    ).write_parquet(cleansed_path)

    paths = run_token_zipf_analysis(
        workspace_roots, systems=["dbpedia"], run_date="2026-08-26", wordfreq_top_n=50
    )

    summary = pl.read_parquet(paths.summary_path)
    assert summary.get_column("tier").to_list() == ["raw"]  # basic/cleansed skipped


@pytest.mark.integration
@pytest.mark.graphics
def test_run_token_zipf_analysis_produces_report_and_plots_end_to_end(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")

    paths = run_token_zipf_analysis(
        workspace_roots, systems=["ie"], run_date="2026-08-26", wordfreq_top_n=200
    )

    assert paths.report_path.exists()
    assert paths.summary_path.exists()

    summary = pl.read_parquet(paths.summary_path)
    assert set(summary.get_column("tier").to_list()) == {"raw", "basic", "cleansed"}
    for row in summary.to_dicts():
        # Regression guard: plot_path must be written repo-relative, not
        # absolute, so the summary is portable across machines/checkouts.
        assert not Path(row["plot_path"]).is_absolute()
        assert (tmp_path / row["plot_path"]).exists()
        # The head-words picture is the curve's pair, written for every row
        # the same way, since every distribution has a head to name.
        assert not Path(row["head_words_plot_path"]).is_absolute()
        assert (tmp_path / row["head_words_plot_path"]).exists()

    report_text = paths.report_path.read_text(encoding="utf-8")
    assert "raw" in report_text
    assert "basic" in report_text
    assert "cleansed" in report_text


@pytest.mark.integration
def test_run_token_zipf_analysis_auto_discovery_excludes_a_stopped_system(
    workspace_roots: WorkspaceRoots,
):
    """Auto-discovery (`systems=None`) walks `data/` for any system with a
    `cleansed` subdirectory, with no catalog-status awareness of its own --
    dbpedia was stopped 2026-08-29 but still had leftover cleansed parquet on
    disk, so it kept showing up in every zipf sweep run without `systems`
    named explicitly. Naming it explicitly (see
    `test_run_token_zipf_analysis_skips_tiers_missing_from_a_systems_schema`)
    must still resolve it -- only auto-discovery is filtered."""
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")
    cleansed_path = _cleansed_dir(workspace_roots, "dbpedia") / "dbpedia-001.parquet"
    cleansed_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name": ["Acme Systems"]}).write_parquet(cleansed_path)

    # Discovery is the subject; the plots are not read, so they are not drawn.
    paths = run_token_zipf_analysis(
        workspace_roots,
        systems=None,
        run_date="2026-08-26",
        wordfreq_top_n=50,
        render_plots=False,
    )

    summary = pl.read_parquet(paths.summary_path)
    assert "dbpedia" not in summary.get_column("system").to_list()
    assert "ie" in summary.get_column("system").to_list()


@pytest.mark.integration
def test_run_token_zipf_analysis_skips_power_law_comparison_by_default(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")

    paths = run_token_zipf_analysis(
        workspace_roots,
        systems=["ie"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
    )

    summary = pl.read_parquet(paths.summary_path)
    assert set(summary.get_column("power_law_alpha").to_list()) == {None}


@pytest.mark.integration
def test_run_token_zipf_analysis_power_law_comparison_runs_for_a_single_language_system(
    workspace_roots: WorkspaceRoots,
):
    # "ie" resolves to "en" in analysis.token_rarity.SYSTEM_LANGUAGES.
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")

    paths = run_token_zipf_analysis(
        workspace_roots,
        systems=["ie"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
        compare_power_law=True,
    )

    summary = pl.read_parquet(paths.summary_path)
    assert all(value is not None for value in summary.get_column("power_law_alpha"))


@pytest.mark.integration
def test_run_token_zipf_analysis_power_law_comparison_fits_a_no_language_system(
    workspace_roots: WorkspaceRoots,
):
    # "dbpedia" has no entry in SYSTEM_LANGUAGES; the fit reads only the
    # corpus's own document frequencies, so it needs none.
    cleansed_path = _cleansed_dir(workspace_roots, "dbpedia") / "dbpedia-001.parquet"
    cleansed_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"name": ["Acme Systems", "Beta Holdings"]}).write_parquet(
        cleansed_path
    )

    paths = run_token_zipf_analysis(
        workspace_roots,
        systems=["dbpedia"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
        compare_power_law=True,
    )

    summary = pl.read_parquet(paths.summary_path)
    assert all(value is not None for value in summary.get_column("power_law_alpha"))


@pytest.mark.integration
def test_load_persisted_power_law_fit_returns_the_persisted_comparison(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")
    run_token_zipf_analysis(
        workspace_roots,
        systems=["ie"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
        compare_power_law=True,
    )

    result = load_persisted_power_law_fit(workspace_roots, "ie", "cleansed")

    assert result is not None
    assert set(result) == {
        "alpha",
        "xmin",
        "n_fitted",
        "ks_statistic",
        "loglikelihood_ratio",
        "mean_loglikelihood_diff",
        "p_value",
    }


@pytest.mark.integration
def test_load_persisted_power_law_fit_returns_none_when_not_computed(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")
    # compare_power_law defaults to False.
    run_token_zipf_analysis(
        workspace_roots,
        systems=["ie"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
    )

    assert load_persisted_power_law_fit(workspace_roots, "ie", "cleansed") is None


def test_load_persisted_power_law_fit_returns_none_when_never_run(
    workspace_roots: WorkspaceRoots,
):
    assert load_persisted_power_law_fit(workspace_roots, "ie", "cleansed") is None


@pytest.mark.integration
def test_run_token_zipf_analysis_refuses_a_run_date_already_taken_unless_forced(
    workspace_roots: WorkspaceRoots,
):
    """A run's summary is written whole, keyed on its date, so a second run on
    the same date would leave the first's per-system stats on disk with no
    summary row naming them -- found for real when same-day `ie` and `gb`
    invocations left only `gb` visible to the notebook."""
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "gb") / "gb-001.parquet")
    run_token_zipf_analysis(
        workspace_roots,
        systems=["ie"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
    )

    with pytest.raises(FileExistsError):
        run_token_zipf_analysis(
            workspace_roots,
            systems=["gb"],
            run_date="2026-08-26",
            wordfreq_top_n=200,
            render_plots=False,
        )
    assert find_latest_zipf_summary_for_system(workspace_roots, "ie") is not None

    paths = run_token_zipf_analysis(
        workspace_roots,
        systems=["gb"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
        force=True,
    )

    summary = pl.read_parquet(paths.summary_path)
    assert set(summary.get_column("system").to_list()) == {"gb"}


@pytest.mark.integration
@pytest.mark.graphics
def test_run_token_zipf_analysis_report_embeds_its_own_viz_pngs_by_reference(
    workspace_roots: WorkspaceRoots,
):
    """The markdown report must reference (embed) its own viz/ PNGs by a
    working relative link, not leave them as unlinked "drop them in the
    directory" artefacts.
    """
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")

    paths = run_token_zipf_analysis(
        workspace_roots, systems=["ie"], run_date="2026-08-26", wordfreq_top_n=200
    )

    report_text = paths.report_path.read_text(encoding="utf-8")
    assert "## Rank-Frequency Plots" in report_text

    # Every image link the report embeds must resolve to a real file relative
    # to the report's own directory -- this is the "works from wherever the
    # report is archived" contract, checked directly rather than assumed.
    image_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report_text)
    assert image_links, "expected at least one embedded image link"
    for link in image_links:
        assert not Path(link).is_absolute()
        resolved = (paths.report_path.parent / link).resolve()
        assert resolved.exists(), f"embedded image link does not resolve: {link}"
        assert resolved.is_relative_to(paths.viz_dir.resolve())


@pytest.mark.integration
@pytest.mark.graphics
def test_run_trim_profile_sweep_report_embeds_its_own_viz_pngs_by_reference(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")

    paths = run_trim_profile_sweep(
        workspace_roots, systems=["ie"], run_date="2026-08-26", wordfreq_top_n=200
    )

    report_text = paths.report_path.read_text(encoding="utf-8")
    image_links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report_text)
    assert image_links, "expected at least one embedded image link"
    for link in image_links:
        resolved = (paths.report_path.parent / link).resolve()
        assert resolved.exists(), f"embedded image link does not resolve: {link}"


def test_find_latest_zipf_summary_for_system_returns_most_recent_matching_run(
    workspace_roots: WorkspaceRoots,
):
    _write_persisted_zipf_run(workspace_roots, "ie", "2026-08-01")
    _write_persisted_zipf_run(workspace_roots, "ie", "2026-08-20")

    result = find_latest_zipf_summary_for_system(workspace_roots, "ie")

    assert result is not None
    run_date, rows_df = result
    assert run_date == "2026-08-20"
    assert set(rows_df.get_column("system").to_list()) == {"ie"}
    assert set(rows_df.get_column("tier").to_list()) == {"raw", "basic", "cleansed"}


@pytest.mark.integration
def test_find_latest_zipf_summary_for_system_skips_trim_sweep_runs(
    workspace_roots: WorkspaceRoots,
):
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")

    run_trim_profile_sweep(
        workspace_roots,
        systems=["ie"],
        run_date="2026-08-25",
        wordfreq_top_n=200,
        render_plots=False,
    )

    assert find_latest_zipf_summary_for_system(workspace_roots, "ie") is None


def test_find_latest_zipf_summary_for_system_returns_none_when_never_run(
    workspace_roots: WorkspaceRoots,
):
    assert find_latest_zipf_summary_for_system(workspace_roots, "ie") is None


@pytest.mark.integration
def test_run_token_zipf_analysis_persists_a_token_stats_table_per_tier(
    workspace_roots: WorkspaceRoots,
):
    """The per-tier token/document-frequency table this scan produces is
    worth reusing (it's the expensive part of a full-corpus run) -- callers
    like the exploratory notebook load it back via
    `load_persisted_word_token_stats` instead of repeating the scan."""
    _write_cleansed_fixture(_cleansed_dir(workspace_roots, "ie") / "ie-001.parquet")

    paths = run_token_zipf_analysis(
        workspace_roots,
        systems=["ie"],
        run_date="2026-08-26",
        wordfreq_top_n=200,
        render_plots=False,
    )

    for tier in ("raw", "basic", "cleansed"):
        stats_path = paths.stats_dir / f"ie_{tier}_token_stats.parquet"
        assert stats_path.exists()
        stats = pl.read_parquet(stats_path)
        assert {"token", "document_frequency", "document_frequency_pct"}.issubset(
            stats.columns
        )


def test_load_persisted_word_token_stats_returns_most_recent_run_tables(
    workspace_roots: WorkspaceRoots,
):
    _write_persisted_zipf_run(workspace_roots, "ie", "2026-08-01")
    _write_persisted_zipf_run(workspace_roots, "ie", "2026-08-20")

    result = load_persisted_word_token_stats(workspace_roots, "ie")

    assert result is not None
    assert set(result) == {"raw", "basic", "cleansed"}
    stats, total_docs = result["cleansed"]
    assert total_docs == 4
    assert "acme" in stats.get_column("token").to_list()


def test_load_persisted_word_token_stats_returns_none_when_never_run(
    workspace_roots: WorkspaceRoots,
):
    assert load_persisted_word_token_stats(workspace_roots, "ie") is None


def test_trim_text_expr_removes_whole_words_case_insensitively():
    df = pl.DataFrame({"text": ["Acme Systems Ltd", "Beta Ltd", "Gamma Corp"]})

    trimmed = df.select(trim_text_expr(pl.col("text"), frozenset({"ltd"})))

    assert trimmed.to_series().to_list() == ["Acme Systems", "Beta", "Gamma Corp"]


def test_trim_text_expr_is_noop_for_empty_noise_words():
    df = pl.DataFrame({"text": ["Acme Systems Ltd"]})

    trimmed = df.select(trim_text_expr(pl.col("text"), frozenset()))

    assert trimmed.to_series().to_list() == ["Acme Systems Ltd"]


def test_build_trim_settings_is_none_and_packaged_default_only_without_per_corpus_file(
    workspace_roots: WorkspaceRoots,
):
    settings = build_trim_settings("ie", workspace_roots)

    labels = [label for label, _ in settings]
    assert labels == ["none", "packaged_default"]
    none_tokens = dict(settings)["none"]
    packaged_tokens = dict(settings)["packaged_default"]
    assert none_tokens == frozenset()
    assert "ltd" in packaged_tokens  # packaged default is company_cleanse's profile


def test_build_trim_settings_adds_per_corpus_profiles_when_noise_words_json_exists(
    workspace_roots: WorkspaceRoots,
):
    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    )
    noise_words_path.parent.mkdir(parents=True, exist_ok=True)
    noise_words_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "strict": {"tokens": ["acme"]},
                    "balanced": {"tokens": []},
                    "aggressive": {"tokens": []},
                }
            }
        ),
        encoding="utf-8",
    )

    settings = build_trim_settings("ie", workspace_roots)

    labels = [label for label, _ in settings]
    assert labels == [
        "none",
        "packaged_default",
        "per_corpus_strict",
        "per_corpus_balanced",
        "per_corpus_aggressive",
    ]
    per_corpus_strict = dict(settings)["per_corpus_strict"]
    assert "acme" in per_corpus_strict  # corpus-specific, not in the packaged default
    packaged_default = dict(settings)["packaged_default"]
    assert "acme" not in packaged_default


@pytest.mark.integration
@pytest.mark.graphics
def test_run_trim_profile_sweep_shows_per_corpus_trim_changing_the_corpus(
    tmp_path: Path,
    workspace_roots: WorkspaceRoots,
):
    cleansed_path = _cleansed_dir(workspace_roots, "ie") / "ie-001.parquet"
    cleansed_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "name_cleansed": [
                "acme systems",
                "acme holdings",
                "acme trading",
                "beta group",
            ]
        }
    ).write_parquet(cleansed_path)

    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    )
    noise_words_path.parent.mkdir(parents=True, exist_ok=True)
    noise_words_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "strict": {"tokens": ["acme"]},
                    "balanced": {"tokens": []},
                    "aggressive": {"tokens": []},
                }
            }
        ),
        encoding="utf-8",
    )

    paths = run_trim_profile_sweep(
        workspace_roots,
        systems=["ie"],
        name_col="name_cleansed",
        run_date="2026-08-26",
        wordfreq_top_n=50,
    )

    summary = pl.read_parquet(paths.summary_path)
    rows = {row["trim_setting"]: row for row in summary.to_dicts()}
    assert set(rows) == {
        "none",
        "packaged_default",
        "per_corpus_strict",
        "per_corpus_balanced",
        "per_corpus_aggressive",
    }
    # "acme" appears in 3/4 names -- trimming it away under per_corpus_strict
    # should reduce unique-token count versus the untrimmed setting.
    assert rows["per_corpus_strict"]["unique_tokens"] < rows["none"]["unique_tokens"]
    for row in rows.values():
        assert not Path(row["plot_path"]).is_absolute()
        assert (tmp_path / row["plot_path"]).exists()

    report_text = paths.report_path.read_text(encoding="utf-8")
    assert "per_corpus_strict" in report_text
