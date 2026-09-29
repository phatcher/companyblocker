from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis.noise_layers import (
    CLEANSE_LAYER,
    CorpusTierStats,
    NoiseLayerSpec,
    TokenizerIdfStats,
    compute_corpus_layer_metrics,
    compute_layer_removals,
    compute_pairwise_jaccard,
    draw_layer_bar_chart,
    draw_overlap_diagram,
    load_tokenizer_idf_stats,
    load_zipf_tier_stats,
    project_cross_corpus_tokens,
    render_layer_bar_charts,
    render_overlap_diagrams,
    resolve_noise_layer_paths,
    resolve_noise_layers,
    run_noise_layer_analysis,
    summarize_layer_regions,
)
from workspace.artifact_layout import analysis_report_run_dir, tokenizer_scope_dir
from workspace.roots import WorkspaceRoots


def _tier_stats(
    system: str, tier: str, tokens: dict[str, tuple[int, float]], total_docs: int
) -> CorpusTierStats:
    return CorpusTierStats(
        system=system,
        tier=tier,
        stats=pl.DataFrame(
            {
                "token": list(tokens.keys()),
                "document_frequency": [v[0] for v in tokens.values()],
                "document_frequency_pct": [v[1] for v in tokens.values()],
            }
        ),
        total_docs=total_docs,
    )


# A small synthetic three-corpus fixture with a known removed-set overlap
# once the "packaged_default" layer (noise_words={"the", "acme"}) is applied:
# "the" is removed from all three ("a+b+c"), "acme" only from a/b ("a+b"),
# and each corpus keeps one unique token the layer never touches.
def _three_corpus_projected() -> pl.DataFrame:
    tier_stats = [
        _tier_stats(
            "a", "raw", {"the": (10, 1.0), "acme": (5, 0.5), "alpha": (2, 0.2)}, 10
        ),
        _tier_stats("a", "cleansed", {"acme": (5, 0.5), "alpha": (2, 0.2)}, 10),
        _tier_stats(
            "b", "raw", {"the": (8, 0.8), "acme": (4, 0.4), "beta": (3, 0.3)}, 10
        ),
        _tier_stats("b", "cleansed", {"acme": (4, 0.4), "beta": (3, 0.3)}, 10),
        _tier_stats(
            "c", "raw", {"the": (9, 0.9), "gmbh": (6, 0.6), "gamma": (1, 0.1)}, 10
        ),
        _tier_stats("c", "cleansed", {"gamma": (1, 0.1)}, 10),
    ]
    return project_cross_corpus_tokens(tier_stats)


def _three_corpus_layers() -> dict[str, list[NoiseLayerSpec]]:
    packaged = NoiseLayerSpec("packaged_default", frozenset({"the", "acme"}))
    return {"a": [packaged], "b": [packaged], "c": [packaged]}


def test_project_cross_corpus_tokens_returns_empty_frame_for_no_input():
    result = project_cross_corpus_tokens([])

    assert result.height == 0
    assert result.columns == [
        "token",
        "system",
        "tier",
        "language",
        "document_frequency",
        "document_frequency_pct",
        "idf",
    ]


def test_project_cross_corpus_tokens_resolves_language_and_nulls_idf_by_default():
    tier_stats = [_tier_stats("ie", "raw", {"ltd": (5, 0.5)}, 10)]

    projected = project_cross_corpus_tokens(tier_stats)

    row = projected.row(0, named=True)
    assert row["language"] == "en"
    assert row["idf"] is None


def test_project_cross_corpus_tokens_leaves_language_null_for_unmapped_system():
    tier_stats = [_tier_stats("gleif", "raw", {"ltd": (5, 0.5)}, 10)]

    projected = project_cross_corpus_tokens(tier_stats)

    assert projected.row(0, named=True)["language"] is None


def test_project_cross_corpus_tokens_attaches_idf_from_tokenizer_stats():
    tier_stats = [
        _tier_stats("ie", "cleansed", {"ltd": (5, 0.5), "acme": (3, 0.3)}, 10)
    ]
    idf_stats = [
        TokenizerIdfStats(
            system="ie",
            tier="cleansed",
            stats=pl.DataFrame({"token": ["ltd"], "idf": [1.2]}),
        )
    ]

    projected = project_cross_corpus_tokens(tier_stats, idf_stats)

    by_token = {row["token"]: row["idf"] for row in projected.iter_rows(named=True)}
    assert by_token["ltd"] == pytest.approx(1.2)
    assert by_token["acme"] is None


def test_load_zipf_tier_stats_reads_persisted_run_and_skips_missing_tier(
    workspace_roots: WorkspaceRoots,
):
    run_root = analysis_report_run_dir(workspace_roots, "token_zipf", "2026-09-05")
    metrics_dir = run_root / "metrics"
    metrics_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "token": ["ltd", "acme"],
            "document_frequency": [5, 2],
            "document_frequency_pct": [0.5, 0.2],
        }
    ).write_parquet(metrics_dir / "ie_raw_token_stats.parquet")
    pl.DataFrame(
        {
            "system": ["ie"],
            "tier": ["raw"],
            "total_docs": [10],
        }
    ).write_parquet(metrics_dir / "token_zipf_summary.parquet")

    entries = load_zipf_tier_stats(
        workspace_roots, "2026-09-05", ["ie"], tiers=("raw", "cleansed")
    )

    assert len(entries) == 1
    assert entries[0].system == "ie"
    assert entries[0].tier == "raw"
    assert entries[0].total_docs == 10


def test_load_tokenizer_idf_stats_resolves_tier_from_metadata_name_col(
    workspace_roots: WorkspaceRoots,
):
    system_dir = tokenizer_scope_dir(workspace_roots, system="ie")
    system_dir.mkdir(parents=True)
    (system_dir / "metadata.json").write_text(
        json.dumps({"name_col": "name_cleansed"}), encoding="utf-8"
    )
    pl.DataFrame(
        {
            "token": ["ltd"],
            "document_frequency": [5],
            "document_frequency_pct": [0.5],
            "idf": [1.1],
        }
    ).write_parquet(system_dir / "token_tfidf_stats.parquet")

    entries = load_tokenizer_idf_stats(workspace_roots, ["ie"])

    assert len(entries) == 1
    assert entries[0].tier == "cleansed"


def test_load_tokenizer_idf_stats_skips_system_with_no_stats_file(
    workspace_roots: WorkspaceRoots,
):
    entries = load_tokenizer_idf_stats(workspace_roots, ["missing_system"])

    assert entries == []


def test_resolve_noise_layers_always_includes_packaged_default(
    workspace_roots: WorkspaceRoots,
):
    layers = resolve_noise_layers("ie", workspace_roots)

    assert [layer.name for layer in layers] == ["packaged_default"]
    assert "ltd" in layers[0].noise_words


def test_resolve_noise_layers_adds_per_corpus_profiles_when_noise_words_json_exists(
    workspace_roots: WorkspaceRoots,
):
    noise_words_path = (
        tokenizer_scope_dir(workspace_roots, system="ie") / "noise_words.json"
    )
    noise_words_path.parent.mkdir(parents=True)
    noise_words_path.write_text(
        json.dumps(
            {
                "profiles": {
                    "strict": {"tokens": ["acme"]},
                    "balanced": {"tokens": ["acme", "systems"]},
                    "aggressive": {"tokens": ["acme", "systems", "trading"]},
                }
            }
        ),
        encoding="utf-8",
    )

    layers = resolve_noise_layers("ie", workspace_roots)

    assert [layer.name for layer in layers] == [
        "packaged_default",
        "per_corpus_strict",
        "per_corpus_balanced",
        "per_corpus_aggressive",
    ]
    strict_layer = next(layer for layer in layers if layer.name == "per_corpus_strict")
    assert strict_layer.noise_words == {"acme"}


def test_resolve_noise_layers_adds_one_layer_per_extra_noise_word_path(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    extra_path = tmp_path / "pooled_candidates.txt"
    extra_path.write_text("foo\nbar\n", encoding="utf-8")

    layers = resolve_noise_layers(
        "ie", workspace_roots, extra_noise_word_paths=[("pooled", extra_path)]
    )

    assert layers[-1].name == "extra_pooled"
    assert layers[-1].noise_words == {"foo", "bar"}


def test_compute_layer_removals_cleanse_layer_is_raw_minus_cleansed_tier_diff():
    projected = _three_corpus_projected()

    membership = compute_layer_removals(
        projected, systems=["a", "b", "c"], cleanse_tier="cleansed", layers_by_system={}
    )

    cleanse_rows = membership.filter(pl.col("layer") == CLEANSE_LAYER)
    removed_by_system = {
        system: set(group.get_column("token"))
        for (system,), group in cleanse_rows.group_by("system")
    }
    assert removed_by_system == {"a": {"the"}, "b": {"the"}, "c": {"the", "gmbh"}}


def test_compute_layer_removals_noise_word_layer_intersects_raw_vocab():
    projected = _three_corpus_projected()

    membership = compute_layer_removals(
        projected,
        systems=["a", "b", "c"],
        cleanse_tier="cleansed",
        layers_by_system=_three_corpus_layers(),
    )

    layer_rows = membership.filter(pl.col("layer") == "packaged_default")
    removed_by_system = {
        system: set(group.get_column("token"))
        for (system,), group in layer_rows.group_by("system")
    }
    # "acme" is in the noise-word set and in a/b's raw vocab, but "c" never had
    # "acme" in its raw vocab at all -- it must not appear for "c".
    assert removed_by_system == {
        "a": {"the", "acme"},
        "b": {"the", "acme"},
        "c": {"the"},
    }


def test_compute_layer_removals_attaches_the_exact_matching_region():
    projected = _three_corpus_projected()

    membership = compute_layer_removals(
        projected,
        systems=["a", "b", "c"],
        cleanse_tier="cleansed",
        layers_by_system=_three_corpus_layers(),
    )

    layer_rows = membership.filter(pl.col("layer") == "packaged_default")
    region_by_token = dict(
        zip(
            layer_rows.get_column("token").to_list(),
            layer_rows.get_column("region").to_list(),
            strict=False,
        )
    )
    assert set(region_by_token["acme"].split("+")) == {"a", "b"}
    assert set(region_by_token["the"].split("+")) == {"a", "b", "c"}


def test_compute_corpus_layer_metrics_gives_zero_row_for_a_layer_that_removed_nothing():
    projected = _three_corpus_projected()
    layers_by_system = {"a": [], "b": [], "c": []}
    membership = compute_layer_removals(
        projected,
        systems=["a", "b", "c"],
        cleanse_tier="cleansed",
        layers_by_system=layers_by_system,
    )

    metrics = compute_corpus_layer_metrics(
        projected,
        membership,
        layer_names_by_system={"a": [CLEANSE_LAYER, "packaged_default"]},
    )

    row = metrics.filter(
        (pl.col("system") == "a") & (pl.col("layer") == "packaged_default")
    ).row(0, named=True)
    assert row["removed_count"] == 0
    assert row["removed_mass"] == 0.0
    assert row["remaining_count"] == row["raw_vocab_size"]


def test_compute_corpus_layer_metrics_reports_removed_and_remaining_counts():
    projected = _three_corpus_projected()
    membership = compute_layer_removals(
        projected,
        systems=["a", "b", "c"],
        cleanse_tier="cleansed",
        layers_by_system=_three_corpus_layers(),
    )

    metrics = compute_corpus_layer_metrics(
        projected,
        membership,
        layer_names_by_system={
            "a": [CLEANSE_LAYER, "packaged_default"],
            "b": [CLEANSE_LAYER, "packaged_default"],
            "c": [CLEANSE_LAYER, "packaged_default"],
        },
    )

    row = metrics.filter(
        (pl.col("system") == "a") & (pl.col("layer") == "packaged_default")
    ).row(0, named=True)
    assert row["raw_vocab_size"] == 3
    assert row["removed_count"] == 2
    assert row["remaining_count"] == 1
    assert row["removed_mass"] == pytest.approx(1.5)


def test_summarize_layer_regions_reports_token_count_and_mass_per_region():
    projected = _three_corpus_projected()
    membership = compute_layer_removals(
        projected,
        systems=["a", "b", "c"],
        cleanse_tier="cleansed",
        layers_by_system=_three_corpus_layers(),
    )

    regions = summarize_layer_regions(membership, layer="packaged_default")

    region_ab = regions.filter(pl.col("region") == "a+b")
    assert set(region_ab.get_column("system").to_list()) == {"a", "b"}
    assert region_ab.get_column("token_count").to_list() == [1, 1]


def test_compute_pairwise_jaccard_matches_known_overlap():
    projected = _three_corpus_projected()
    membership = compute_layer_removals(
        projected,
        systems=["a", "b", "c"],
        cleanse_tier="cleansed",
        layers_by_system=_three_corpus_layers(),
    )

    jaccard = compute_pairwise_jaccard(membership, layer="packaged_default")

    by_pair = {
        (row["system_a"], row["system_b"]): row["jaccard"]
        for row in jaccard.iter_rows(named=True)
    }
    # a={the,acme}, b={the,acme}, c={the}: a/b identical (1.0), a/c and b/c
    # share exactly "the" out of a 2-token union (0.5).
    assert by_pair[("a", "b")] == pytest.approx(1.0)
    assert by_pair[("a", "c")] == pytest.approx(0.5)
    assert by_pair[("b", "c")] == pytest.approx(0.5)


def test_draw_layer_bar_chart_draws_one_bar_container_per_layer():
    metrics = pl.DataFrame(
        {
            "system": ["a", "a", "b", "b"],
            "layer": ["cleanse", "packaged_default", "cleanse", "packaged_default"],
            "remaining_count": [3, 1, 2, 2],
        }
    )
    fig, ax = plt.subplots()

    draw_layer_bar_chart(
        ax, metrics, value_col="remaining_count", ylabel="y", title="t"
    )
    plt.close(fig)

    assert len(ax.containers) == 2


@pytest.mark.graphics
def test_render_layer_bar_charts_writes_both_files(tmp_path: Path):
    metrics = pl.DataFrame(
        {
            "system": ["a", "b"],
            "layer": ["cleanse", "cleanse"],
            "raw_vocab_size": [3, 3],
            "removed_count": [1, 1],
            "remaining_count": [2, 2],
            "removed_mass": [0.5, 0.4],
        }
    )

    outputs = render_layer_bar_charts(metrics, tmp_path)

    assert set(outputs) == {"remaining_count", "removed_mass"}
    for path in outputs.values():
        assert path.exists()


def test_draw_overlap_diagram_requires_at_least_two_corpora():
    fig = plt.figure()

    with pytest.raises(ValueError, match="at least two"):
        draw_overlap_diagram(fig, {"a": frozenset({"x"})}, {}, title="t")
    plt.close(fig)


def test_draw_overlap_diagram_draws_venn_for_two_corpora():
    fig = plt.figure()

    draw_overlap_diagram(
        fig,
        {"a": frozenset({"x", "y"}), "b": frozenset({"y", "z"})},
        {"a": (10, 5), "b": (10, 5)},
        title="t",
    )

    assert len(fig.axes) == 1
    plt.close(fig)


def test_draw_overlap_diagram_draws_upset_for_four_corpora():
    fig = plt.figure()
    removed_sets = {
        "a": frozenset({"x"}),
        "b": frozenset({"x", "y"}),
        "c": frozenset({"y"}),
        "d": frozenset({"z"}),
    }

    draw_overlap_diagram(
        fig, removed_sets, {k: (10, 5) for k in removed_sets}, title="t"
    )

    assert len(fig.axes) > 1
    plt.close(fig)


def test_render_overlap_diagrams_skips_a_layer_with_only_one_corpus(tmp_path: Path):
    membership = pl.DataFrame(
        {
            "layer": ["solo_layer"],
            "system": ["a"],
            "token": ["x"],
            "document_frequency": [1],
            "document_frequency_pct": [0.1],
            "region": ["a"],
        }
    )

    outputs = render_overlap_diagrams(membership, {"a": (10, 5)}, tmp_path)

    assert outputs == {}


@pytest.mark.graphics
def test_render_overlap_diagrams_writes_one_file_per_layer(tmp_path: Path):
    projected = _three_corpus_projected()
    membership = compute_layer_removals(
        projected,
        systems=["a", "b", "c"],
        cleanse_tier="cleansed",
        layers_by_system=_three_corpus_layers(),
    )
    corpus_totals = {"a": (3, 10), "b": (3, 10), "c": (3, 10)}

    outputs = render_overlap_diagrams(membership, corpus_totals, tmp_path)

    assert set(outputs) == {CLEANSE_LAYER, "packaged_default"}
    for path in outputs.values():
        assert path.exists()


def test_resolve_noise_layer_paths_creates_metrics_and_viz_dirs(
    workspace_roots: WorkspaceRoots,
):
    paths = resolve_noise_layer_paths(workspace_roots, "2026-09-05")

    # Where `run_root` lands is `workspace.artifact_layout`'s contract and is
    # asserted there; this test's own subject is that the resolver creates the
    # two directories a run then writes into.
    assert paths.metrics_dir.is_dir()
    assert paths.viz_dir.is_dir()


@pytest.mark.integration
@pytest.mark.graphics
def test_run_noise_layer_analysis_writes_expected_artifacts(
    workspace_roots: WorkspaceRoots,
):
    zipf_metrics_dir = (
        analysis_report_run_dir(workspace_roots, "token_zipf", "2026-09-05") / "metrics"
    )
    zipf_metrics_dir.mkdir(parents=True)
    for system, raw_tokens, cleansed_tokens in [
        ("ie", {"ltd": (10, 1.0), "acme": (5, 0.5)}, {"acme": (5, 0.5)}),
        ("gb", {"ltd": (8, 0.8), "acme": (3, 0.3)}, {"acme": (3, 0.3)}),
    ]:
        pl.DataFrame(
            {
                "token": list(raw_tokens.keys()),
                "document_frequency": [v[0] for v in raw_tokens.values()],
                "document_frequency_pct": [v[1] for v in raw_tokens.values()],
            }
        ).write_parquet(zipf_metrics_dir / f"{system}_raw_token_stats.parquet")
        pl.DataFrame(
            {
                "token": list(cleansed_tokens.keys()),
                "document_frequency": [v[0] for v in cleansed_tokens.values()],
                "document_frequency_pct": [v[1] for v in cleansed_tokens.values()],
            }
        ).write_parquet(zipf_metrics_dir / f"{system}_cleansed_token_stats.parquet")
    pl.DataFrame(
        {
            "system": ["ie", "ie", "gb", "gb"],
            "tier": ["raw", "cleansed", "raw", "cleansed"],
            "total_docs": [10, 10, 10, 10],
        }
    ).write_parquet(zipf_metrics_dir / "token_zipf_summary.parquet")

    paths = run_noise_layer_analysis(
        workspace_roots, systems=["ie", "gb"], run_date="2026-09-05"
    )

    assert (paths.metrics_dir / "projected_tokens.parquet").exists()
    assert (paths.metrics_dir / "membership.parquet").exists()
    assert (paths.metrics_dir / "layer_metrics.parquet").exists()
    assert (paths.metrics_dir / "region_summary.parquet").exists()
    assert (paths.metrics_dir / "region_jaccard.parquet").exists()
    assert (paths.viz_dir / "_remaining_tokens_by_layer.png").exists()
    assert (paths.viz_dir / "_removed_mass_by_layer.png").exists()
    assert (paths.viz_dir / f"_overlap_{CLEANSE_LAYER}.png").exists()

    membership = pl.read_parquet(paths.metrics_dir / "membership.parquet")
    assert (
        "ltd"
        in membership.filter(pl.col("layer") == "packaged_default")
        .get_column("token")
        .to_list()
    )
