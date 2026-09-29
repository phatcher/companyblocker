from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis.noise_layers import CorpusTierStats, project_cross_corpus_tokens
from analysis.vocab_shrinkage import (
    IN_LANGUAGE_LAYER,
    OOV_LAYER,
    UNSPLIT_LAYER,
    build_budget_grid,
    build_pooled_ranking,
    build_top_v_overlap_membership,
    classify_tokens_in_language,
    compute_dropped_mass_curve,
    draw_dropped_mass_curve,
    render_dropped_mass_curves,
    resolve_vocab_shrinkage_paths,
    run_vocab_shrinkage_analysis,
)
from workspace.artifact_layout import analysis_report_run_dir
from workspace.roots import WorkspaceRoots


def _tier_stats(
    system: str, tokens: dict[str, tuple[int, float]], *, total_docs: int
) -> CorpusTierStats:
    return CorpusTierStats(
        system=system,
        tier="raw",
        stats=pl.DataFrame(
            {
                "token": list(tokens.keys()),
                "document_frequency": [v[0] for v in tokens.values()],
                "document_frequency_pct": [v[1] for v in tokens.values()],
            }
        ),
        total_docs=total_docs,
    )


# Two synthetic corpora, "small" and "big", with document-frequency counts
# and per-token occurrence shares chosen so the dropped mass at each budget
# V is known by hand:
#
#   small: ltd=(100, 0.5), acme=(60, 0.3), eire=(40, 0.2)
#   big:   sarl=(1000, 0.55), acme=(500, 0.30), sas=(300, 0.15)
#
# count-weighted pool score = summed document_frequency:
#   sarl=1000, acme=560, sas=300, ltd=100, eire=40
#   -> ranking: sarl, acme, sas, ltd, eire
#
# equal-weighted pool score = summed document_frequency_pct:
#   acme=0.60, sarl=0.55, ltd=0.5, eire=0.2, sas=0.15
#   -> ranking: acme, sarl, ltd, eire, sas
#
# Each corpus's own ranking (by its own document_frequency) is unaffected by
# pooling: small = ltd, acme, eire; big = sarl, acme, sas.
def _two_corpus_projected() -> pl.DataFrame:
    tier_stats = [
        _tier_stats(
            "small",
            {"ltd": (100, 0.5), "acme": (60, 0.3), "eire": (40, 0.2)},
            total_docs=200,
        ),
        _tier_stats(
            "big",
            {"sarl": (1000, 0.55), "acme": (500, 0.30), "sas": (300, 0.15)},
            total_docs=2000,
        ),
    ]
    return project_cross_corpus_tokens(tier_stats)


def test_build_budget_grid_is_sorted_unique_and_spans_bounds():
    grid = build_budget_grid(10, 1000, 5)

    assert grid == sorted(set(grid))
    assert grid[0] == 10
    assert grid[-1] == 1000
    assert len(grid) <= 5


def test_build_budget_grid_single_point_returns_max():
    assert build_budget_grid(10, 1000, 1) == [1000]


def test_build_budget_grid_rejects_v_min_above_v_max():
    with pytest.raises(ValueError, match="v_min must be <= v_max"):
        build_budget_grid(100, 10, 5)


def test_build_budget_grid_rejects_non_positive_bounds():
    with pytest.raises(ValueError, match="positive"):
        build_budget_grid(0, 10, 5)


def test_build_pooled_ranking_count_weight_ranks_by_summed_document_frequency():
    projected = _two_corpus_projected()

    ranking = build_pooled_ranking(projected, ["small", "big"], weight="count")

    ordered_tokens = ranking.sort("rank").get_column("token").to_list()
    assert ordered_tokens == ["sarl", "acme", "sas", "ltd", "eire"]


def test_build_pooled_ranking_equal_weight_ranks_by_summed_document_frequency_pct():
    projected = _two_corpus_projected()

    ranking = build_pooled_ranking(projected, ["small", "big"], weight="equal")

    ordered_tokens = ranking.sort("rank").get_column("token").to_list()
    # "acme" (0.3 + 0.30 = 0.60) outranks "sarl" (0.55) under equal weighting
    # even though sarl's raw count (1000) dwarfs acme's combined count
    # (560) -- the point of the equal-weighted pool.
    assert ordered_tokens == ["acme", "sarl", "ltd", "eire", "sas"]


def test_build_pooled_ranking_rejects_unknown_weight():
    projected = _two_corpus_projected()

    with pytest.raises(ValueError, match="weight"):
        build_pooled_ranking(projected, ["small", "big"], weight="bogus")


def test_build_pooled_ranking_returns_empty_frame_for_no_matching_rows():
    projected = _two_corpus_projected()

    ranking = build_pooled_ranking(projected, ["missing"], weight="count")

    assert ranking.height == 0
    assert ranking.columns == ["token", "pooled_score", "rank"]


@pytest.mark.parametrize(
    ("v", "expected_small", "expected_big"),
    [
        (1, 0.5, 0.0),
        (2, 0.5, 0.0),
        (3, 0.7, 0.0),
    ],
)
def test_compute_dropped_mass_curve_count_weight_matches_hand_computed_values(
    v, expected_small, expected_big
):
    projected = _two_corpus_projected()

    curve = compute_dropped_mass_curve(projected, systems=["small", "big"], budgets=[v])

    by_system = {
        row["system"]: row["dropped_mass"]
        for row in curve.filter(pl.col("pool_weight") == "count").iter_rows(named=True)
    }
    assert by_system["small"] == pytest.approx(expected_small)
    assert by_system["big"] == pytest.approx(expected_big)


@pytest.mark.parametrize(
    ("v", "expected_small", "expected_big"),
    [
        (1, 0.5, 0.55),
        (2, 0.5, 0.0),
        (3, 0.2, 0.15),
    ],
)
def test_compute_dropped_mass_curve_equal_weight_matches_hand_computed_values(
    v, expected_small, expected_big
):
    projected = _two_corpus_projected()

    curve = compute_dropped_mass_curve(projected, systems=["small", "big"], budgets=[v])

    by_system = {
        row["system"]: row["dropped_mass"]
        for row in curve.filter(pl.col("pool_weight") == "equal").iter_rows(named=True)
    }
    assert by_system["small"] == pytest.approx(expected_small)
    assert by_system["big"] == pytest.approx(expected_big)


def test_compute_dropped_mass_curve_reports_corpus_top_v_size_and_dropped_count():
    projected = _two_corpus_projected()

    curve = compute_dropped_mass_curve(projected, systems=["small", "big"], budgets=[3])

    row = curve.filter(
        (pl.col("system") == "small") & (pl.col("pool_weight") == "count")
    ).row(0, named=True)
    assert row["corpus_top_v_size"] == 3
    assert row["dropped_token_count"] == 2


def test_compute_dropped_mass_curve_empty_for_no_budgets_or_systems():
    projected = _two_corpus_projected()

    assert (
        compute_dropped_mass_curve(projected, systems=["small"], budgets=[]).height == 0
    )
    assert compute_dropped_mass_curve(projected, systems=[], budgets=[1]).height == 0


def test_classify_tokens_in_language_recognizes_any_member_language():
    # "haftung" ("liability", part of the German "GmbH" legal form) scores
    # zero in English but is a real German word; "beschraenkter"
    # ("limited") scores zero in both -- the same real-word/gibberish
    # contrast `token_rarity`'s own tests use, confirmed against the
    # installed wordfreq data rather than assumed.
    in_language = classify_tokens_in_language(
        ["haftung", "beschraenkter"], ["en", "de"]
    )

    assert "haftung" in in_language
    assert "beschraenkter" not in in_language


def test_classify_tokens_in_language_requires_the_recognizing_language_to_be_a_member():
    # Same token as above, but "de" is not offered as a member language this
    # time -- confirms the recognition came from "de" being present, not
    # leakage from "en" alone.
    in_language = classify_tokens_in_language(["haftung"], ["en"])

    assert "haftung" not in in_language


def test_classify_tokens_in_language_empty_languages_returns_empty_set():
    assert classify_tokens_in_language(["the"], []) == set()


def test_build_top_v_overlap_membership_splits_by_any_member_language():
    tier_stats = [
        _tier_stats(
            "offeneregister",
            {"haftung": (100, 0.5), "beschraenkter": (50, 0.3)},
            total_docs=200,
        ),
        _tier_stats(
            "gb", {"the": (200, 0.6), "zzqxplkxxnope": (80, 0.2)}, total_docs=300
        ),
        _tier_stats("gleif", {"randomtoken": (10, 1.0)}, total_docs=10),
    ]
    projected = project_cross_corpus_tokens(tier_stats)

    membership = build_top_v_overlap_membership(
        projected, systems=["offeneregister", "gb", "gleif"], v=2
    )

    all_layer = membership.filter(pl.col("layer") == UNSPLIT_LAYER)
    assert set(all_layer.get_column("system").unique().to_list()) == {
        "offeneregister",
        "gb",
        "gleif",
    }

    in_language = membership.filter(pl.col("layer") == IN_LANGUAGE_LAYER)
    assert set(
        zip(
            in_language.get_column("system").to_list(),
            in_language.get_column("token").to_list(),
            strict=False,
        )
    ) == {("offeneregister", "haftung"), ("gb", "the")}
    assert "gleif" not in in_language.get_column("system").to_list()

    oov = membership.filter(pl.col("layer") == OOV_LAYER)
    assert set(
        zip(
            oov.get_column("system").to_list(),
            oov.get_column("token").to_list(),
            strict=False,
        )
    ) == {("offeneregister", "beschraenkter"), ("gb", "zzqxplkxxnope")}
    assert "gleif" not in oov.get_column("system").to_list()


def test_build_top_v_overlap_membership_skips_split_layers_with_fewer_than_two_mapped_systems():
    tier_stats = [
        _tier_stats("gb", {"the": (200, 0.6)}, total_docs=300),
        _tier_stats("gleif", {"randomtoken": (10, 1.0)}, total_docs=10),
    ]
    projected = project_cross_corpus_tokens(tier_stats)

    membership = build_top_v_overlap_membership(projected, systems=["gb", "gleif"], v=1)

    assert set(membership.get_column("layer").unique().to_list()) == {UNSPLIT_LAYER}


def test_build_top_v_overlap_membership_attaches_region_for_shared_tokens():
    tier_stats = [
        _tier_stats("gb", {"the": (200, 0.6), "unique_gb": (10, 0.1)}, total_docs=300),
        _tier_stats("ie", {"the": (150, 0.5), "unique_ie": (5, 0.05)}, total_docs=200),
    ]
    projected = project_cross_corpus_tokens(tier_stats)

    membership = build_top_v_overlap_membership(projected, systems=["gb", "ie"], v=2)

    all_layer = membership.filter(pl.col("layer") == UNSPLIT_LAYER)
    region_by_token = dict(
        zip(
            all_layer.get_column("token").to_list(),
            all_layer.get_column("region").to_list(),
            strict=False,
        )
    )
    assert region_by_token["the"] == "gb+ie"
    assert region_by_token["unique_gb"] == "gb"


def test_build_top_v_overlap_membership_returns_empty_frame_for_no_input():
    projected = _two_corpus_projected()

    membership = build_top_v_overlap_membership(projected, systems=["missing"], v=5)

    assert membership.height == 0
    assert membership.columns == [
        "layer",
        "system",
        "token",
        "document_frequency",
        "document_frequency_pct",
        "region",
    ]


def test_draw_dropped_mass_curve_draws_one_line_per_system():
    curve = pl.DataFrame(
        {
            "system": ["a", "a", "b", "b"],
            "pool_weight": ["count", "count", "count", "count"],
            "v": [10, 100, 10, 100],
            "corpus_top_v_size": [10, 100, 10, 100],
            "dropped_token_count": [1, 2, 0, 1],
            "dropped_mass": [0.1, 0.2, 0.0, 0.05],
        }
    )
    fig, ax = plt.subplots()

    draw_dropped_mass_curve(ax, curve, pool_weight="count", title="t")
    plt.close(fig)

    assert len(ax.lines) == 2


@pytest.mark.graphics
def test_render_dropped_mass_curves_writes_one_file_per_weight_present(tmp_path: Path):
    curve = pl.DataFrame(
        {
            "system": ["a", "a"],
            "pool_weight": ["count", "equal"],
            "v": [10, 10],
            "corpus_top_v_size": [10, 10],
            "dropped_token_count": [1, 1],
            "dropped_mass": [0.1, 0.2],
        }
    )

    outputs = render_dropped_mass_curves(curve, tmp_path)

    assert set(outputs) == {"count", "equal"}
    for path in outputs.values():
        assert path.exists()


def test_render_dropped_mass_curves_handles_empty_curve(tmp_path: Path):
    empty = compute_dropped_mass_curve(
        _two_corpus_projected(), systems=[], budgets=[10]
    )

    outputs = render_dropped_mass_curves(empty, tmp_path)

    assert outputs == {}


def test_resolve_vocab_shrinkage_paths_creates_metrics_and_viz_dirs(
    workspace_roots: WorkspaceRoots,
):
    paths = resolve_vocab_shrinkage_paths(workspace_roots, "2026-09-05")

    # Where `run_root` lands is `workspace.artifact_layout`'s contract and is
    # asserted there; this test's own subject is that the resolver creates
    # the two directories a run then writes into.
    assert paths.metrics_dir.is_dir()
    assert paths.viz_dir.is_dir()


@pytest.mark.integration
@pytest.mark.graphics
def test_run_vocab_shrinkage_analysis_writes_expected_artifacts(
    workspace_roots: WorkspaceRoots,
):
    zipf_metrics_dir = (
        analysis_report_run_dir(workspace_roots, "token_zipf", "2026-09-05") / "metrics"
    )
    zipf_metrics_dir.mkdir(parents=True)
    # "the"/"acme" are real English words wordfreq recognizes (nonzero
    # zipf_frequency for "en"); "zzqxplkxxnope"/"qqxzznopeworld" are
    # gibberish it recognizes in no language -- confirmed against the
    # installed wordfreq data, not assumed. Each system contributes its own
    # gibberish token so both the in-language and OOV overlap diagrams have
    # at least two systems to draw (an overlap diagram needs at least two).
    systems_tokens = {
        "ie": {"the": (100, 0.5), "zzqxplkxxnope": (60, 0.3), "eire": (40, 0.2)},
        "gb": {"the": (900, 0.6), "acme": (500, 0.3), "qqxzznopeworld": (150, 0.1)},
    }
    for system, tokens in systems_tokens.items():
        pl.DataFrame(
            {
                "token": list(tokens.keys()),
                "document_frequency": [v[0] for v in tokens.values()],
                "document_frequency_pct": [v[1] for v in tokens.values()],
            }
        ).write_parquet(zipf_metrics_dir / f"{system}_raw_token_stats.parquet")
    pl.DataFrame(
        {
            "system": ["ie", "gb"],
            "tier": ["raw", "raw"],
            "total_docs": [200, 1500],
        }
    ).write_parquet(zipf_metrics_dir / "token_zipf_summary.parquet")

    paths = run_vocab_shrinkage_analysis(
        workspace_roots,
        systems=["ie", "gb"],
        run_date="2026-09-05",
        budgets=[1, 2, 3],
    )

    assert (paths.metrics_dir / "dropped_mass_curve.parquet").exists()
    assert (paths.metrics_dir / "top_v_membership.parquet").exists()
    assert (paths.metrics_dir / "top_v_region_summary.parquet").exists()
    assert (paths.viz_dir / "_dropped_mass_count.png").exists()
    assert (paths.viz_dir / "_dropped_mass_equal.png").exists()
    assert (paths.viz_dir / f"_overlap_{UNSPLIT_LAYER}.png").exists()
    assert (paths.viz_dir / f"_overlap_{IN_LANGUAGE_LAYER}.png").exists()
    assert (paths.viz_dir / f"_overlap_{OOV_LAYER}.png").exists()

    curve = pl.read_parquet(paths.metrics_dir / "dropped_mass_curve.parquet")
    assert set(curve.get_column("pool_weight").unique().to_list()) == {"count", "equal"}
    assert set(curve.get_column("v").unique().to_list()) == {1, 2, 3}
