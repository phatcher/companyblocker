from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from company_classify import (
    EntitySplitter,
    NameVariant,
    NegativeSamplingConfig,
    PairSource,
    RealAliasPairProducer,
)

from scripts.measure_pair_classifier_ceiling import (
    build_ceiling_report,
    load_real_alias_variants,
    main,
)
from workspace.data_layout import CANONICAL_LAYER_NAME
from workspace.derived_uri import Perturbation, name_variant_uri, perturbed_uri
from workspace.layer_layout import names_family_dir
from workspace.match_resolution import UnknownUriSchemeError
from workspace.roots import WorkspaceRoots

_BASE_NAMES = [
    "acme systems",
    "blue river technologies",
    "northshore trading",
    "westfield logistics",
    "meridian group",
    "summit capital partners",
    "cedar valley holdings",
    "ironbridge freight",
    "lighthouse consulting",
    "harbor point ventures",
    "silverline manufacturing",
    "brightwater energy",
    "stonegate insurance",
    "pinecrest logistics",
    "oakridge financial",
    "riverside analytics",
    "crestview partners",
    "amberfield industries",
    "goldenrod capital",
    "eastbrook holdings",
]
_LEGAL_SUFFIXES = ["limited", "ltd", "gmbh", "llc", "plc", "inc"]


def _variants(count: int = 40) -> list[NameVariant]:
    variants: list[NameVariant] = []
    for index in range(count):
        base = _BASE_NAMES[index % len(_BASE_NAMES)]
        uri = f"gleif://LEI{index:06d}"
        primary = f"{base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}"
        previous = f"{base} {_LEGAL_SUFFIXES[(index + 1) % len(_LEGAL_SUFFIXES)]}"
        variants.append(NameVariant(uri, primary, "primary"))
        variants.append(NameVariant(uri, previous, "previous"))
    return variants


def test_load_real_alias_variants_filters_by_jurisdiction_and_reads_names_sidecar(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    names_dir = names_family_dir(
        layer_fixture_dir("gleif", layer=CANONICAL_LAYER_NAME) / "2026-06-21"
    )
    names_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["gleif://LEI000001", "gleif://LEI000002"],
            "name": ["ACME SYSTEMS LIMITED", "BOREALIS SHIPPING AS"],
            "name_type": ["primary", "primary"],
            "jurisdiction_code": ["GB", "FR"],
        }
    ).write_parquet(names_dir / "gleif-names-001.parquet")

    variants = load_real_alias_variants(
        workspace_roots,
        system="gleif",
        canonical_date="2026-06-21",
        jurisdiction_code="GB",
    )

    assert [variant.system_uri for variant in variants] == ["gleif://LEI000001"]
    assert variants[0].name == "ACME SYSTEMS LIMITED"
    assert variants[0].name_type == "primary"


def test_load_real_alias_variants_groups_name_and_perturbed_rows_by_entity(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A name variant (one hop) and a perturbed name variant (two hops) of the same
    entity resolve to that entity's own identity, so `RealAliasPairProducer` puts them
    in one group and one split rather than the singleton-per-row groups a raw,
    unresolved `system_uri` would produce."""
    entity_uri = "gb://123"
    primary_uri = name_variant_uri(
        source_uri=entity_uri, name_type="primary", value="Acme Ltd"
    )
    previous_uri = name_variant_uri(
        source_uri=entity_uri, name_type="previous", value="Acme Group"
    )
    perturbed_previous_uri = perturbed_uri(
        source_uri=previous_uri,
        perturbation=Perturbation(profile="en-lite", version="v1", seed=42),
    )

    names_dir = names_family_dir(
        layer_fixture_dir("gb", layer=CANONICAL_LAYER_NAME) / "2026-06-21"
    )
    names_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": [primary_uri, previous_uri, perturbed_previous_uri],
            "name": ["Acme Ltd", "Acme Group", "Acme Grp"],
            "name_type": ["primary", "previous", "previous"],
            "jurisdiction_code": ["GB", "GB", "GB"],
        }
    ).write_parquet(names_dir / "gb-names-001.parquet")

    variants = load_real_alias_variants(
        workspace_roots,
        system="gb",
        canonical_date="2026-06-21",
        jurisdiction_code="GB",
    )

    assert {variant.system_uri for variant in variants} == {entity_uri}

    splitter = EntitySplitter(seed=1)
    negatives = NegativeSamplingConfig(seed=1)
    producer = RealAliasPairProducer(
        variants=variants, splitter=splitter, negatives=negatives
    )
    pairs = producer.generate()

    real_positives = [
        pair for pair in pairs if pair.source is PairSource.REAL_ALIAS and pair.is_match
    ]
    assert len(real_positives) > 0
    assert {pair.split for pair in real_positives} == {splitter.assign(entity_uri)}


def test_load_real_alias_variants_refuses_an_old_form_name_uri(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
) -> None:
    """A `system_uri` in the pre-`name_variant_uri` form, `name://<hash>` with no system
    or entity segment, fails at resolution rather than silently reading as its own
    entity."""
    names_dir = names_family_dir(
        layer_fixture_dir("gb", layer=CANONICAL_LAYER_NAME) / "2026-06-21"
    )
    names_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "system_uri": ["name://0123456789abcdef"],
            "name": ["Acme Ltd"],
            "name_type": ["primary"],
            "jurisdiction_code": ["GB"],
        }
    ).write_parquet(names_dir / "gb-names-001.parquet")

    with pytest.raises(UnknownUriSchemeError):
        load_real_alias_variants(
            workspace_roots,
            system="gb",
            canonical_date="2026-06-21",
            jurisdiction_code="GB",
        )


def test_load_real_alias_variants_raises_when_no_sidecar_found(
    workspace_roots: WorkspaceRoots,
) -> None:
    with pytest.raises(FileNotFoundError):
        load_real_alias_variants(
            workspace_roots,
            system="gleif",
            canonical_date="2026-06-21",
            jurisdiction_code="GB",
        )


@pytest.mark.integration
def test_build_ceiling_report_records_classifier_and_baseline_side_by_side() -> None:
    splitter = EntitySplitter(seed=42)
    negatives = NegativeSamplingConfig(seed=42)

    report = build_ceiling_report(
        system="gleif",
        jurisdiction_code="GB",
        canonical_date="2026-06-21",
        splitter=splitter,
        negatives=negatives,
        variants=_variants(),
    )

    assert report["run"]["system"] == "gleif"
    assert report["run"]["jurisdiction_code"] == "GB"
    assert report["run"]["split_seed"] == 42
    assert report["run"]["pair_counts"]["total"] > 0
    assert (
        report["run"]["pair_counts"]["train"]
        + report["run"]["pair_counts"]["validation"]
        + report["run"]["pair_counts"]["test"]
        == report["run"]["pair_counts"]["total"]
    )

    assert set(report["baseline"]) == {"tfidf_logreg", "tfidf_mlp"}
    for metrics in (
        report["pair_classifier"]["test_metrics"],
        *(bundle["test_metrics"] for bundle in report["baseline"].values()),
    ):
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    ceiling = report["ceiling"]
    assert ceiling["best_baseline"] in {"tfidf_logreg", "tfidf_mlp"}
    assert ceiling["f1_delta"] == pytest.approx(
        ceiling["classifier_f1"] - ceiling["baseline_f1"]
    )
    assert isinstance(ceiling["classifier_beats_baseline"], bool)


def test_dry_run_reports_the_plan_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_dir = tmp_path / "data"
    json_out = tmp_path / "reports" / "ceiling.json"

    exit_code = main(
        [
            "--dry-run",
            "--data-dir",
            str(data_dir),
            "--jurisdiction",
            "FR",
            "--split-seed",
            "7",
            "--negative-strategy",
            "hard",
            "--json-out",
            str(json_out),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "split_seed=7 (given)" in out
    assert "negative_strategy='hard' (given)" in out
    assert "negatives_per_positive=1.0 (default)" in out
    assert (data_dir / "wikidata" / "canonical" / "2026-07-16").as_posix() in out
    assert "jurisdiction='FR'" in out
    assert f"would clear {json_out}" in out

    assert not json_out.exists()
    assert not data_dir.exists()


def test_dry_run_reports_nothing_would_be_written_without_json_out(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["--dry-run", "--data-dir", str(tmp_path / "data")])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "json_out=None" in out
    assert "--json-out not given; nothing would be written" in out
