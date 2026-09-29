"""Direct tests for `validation.perturbation_materializer`.

Two rows are enough for everything here. The frame builder's contract is one row per
source record, the columns it carries, and the identity it composes -- none of which
needs a corpus, a cleanse pass or anything on disk. What genuinely needs those is
marked `integration` and kept to one path.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from company_perturbation.chain import ChainStep
from company_perturbation.generation import SourceRecord
from company_perturbation.profile_schema import (
    ExclusionRules,
    PerturbationProfile,
    Scenario,
)

from validation.perturbation_dataset_contracts import (
    PERTURBATION_DATASET_REQUIRED_COLUMNS,
)
from validation.perturbation_materializer import (
    PerturbationCollisionError,
    SystemUriParseError,
    _check_no_cross_batch_collision,
    _SourceLineage,
    build_perturbed_frame,
    parse_system_uri,
    relative_materialization_manifest_path,
    resolve_materialization_manifest_path,
)
from workspace.roots import WorkspaceRoots


def _profile(*steps: ChainStep, exclusions: ExclusionRules | None = None):
    return PerturbationProfile(
        profile_id="robustness-v1",
        scenarios=(
            Scenario(
                scenario_id="suffix-swap",
                chain=steps or (ChainStep("legal_suffix.drop", 1.0, 1),),
                exclusions=exclusions,
            ),
        ),
    )


def _two_records():
    return [
        SourceRecord(local_id="1", name="acme systems ltd", system="ie", country="ie"),
        SourceRecord(local_id="2", name="widget corp ltd", system="ie", country="ie"),
    ]


def _pairs(match_uri: str | None = None):
    """Records paired with their lineage, the shape the frame builder consumes."""
    return [
        (
            record,
            _SourceLineage(
                source_uri=f"ie://{record.local_id}",
                match_uri=match_uri if record.local_id == "1" else None,
                system="ie",
                country="ie",
            ),
        )
        for record in _two_records()
    ]


# -- system_uri ------------------------------------------------------------------


def test_parse_system_uri_valid():
    assert parse_system_uri("gb://12345678") == ("gb", "12345678")


@pytest.mark.parametrize(
    ("system_uri", "message"),
    [
        ("gb-12345678", "://"),
        ("gb://", "names nothing"),
        ("GB://12345678", "no system code"),
    ],
)
def test_parse_system_uri_rejects_malformed(system_uri, message):
    with pytest.raises(SystemUriParseError, match=message):
        parse_system_uri(system_uri)


def test_a_derived_row_is_no_entity_to_perturb_from():
    with pytest.raises(SystemUriParseError, match="no entity"):
        parse_system_uri("name://gb/123/0123456789abcdef")


# -- build_perturbed_frame -------------------------------------------------------


def test_every_source_record_produces_exactly_one_row():
    frame = build_perturbed_frame(_profile(), _pairs(), profile_version="v1", seed=42)

    assert frame.height == 2
    assert frame.get_column("system_uri").n_unique() == 2


def test_the_frame_carries_every_required_column():
    frame = build_perturbed_frame(_profile(), _pairs(), profile_version="v1", seed=42)
    produced = set(frame.columns)

    # name_cleansed/name_cleansed_basic arrive later, from the real cleanse pass.
    missing = (
        set(PERTURBATION_DATASET_REQUIRED_COLUMNS)
        - produced
        - {
            "name_cleansed",
            "name_cleansed_basic",
        }
    )
    assert not missing


def test_an_operator_that_fires_is_reported_as_changed():
    """Also the cheapest guard against an unpopulated operator registry.

    A registry with nothing in it raises here rather than surviving to a real run,
    which is exactly what happened once when only the package's `__init__` registered
    the old families.
    """
    frame = build_perturbed_frame(
        _profile(ChainStep("legal_suffix.drop", 1.0, 1)),
        _pairs(),
        profile_version="v1",
        seed=42,
    )

    assert frame.get_column("changed").all()
    assert frame.get_column("name").to_list() == ["acme systems", "widget corp"]
    assert frame.get_column("original_name").to_list() == [
        "acme systems ltd",
        "widget corp ltd",
    ]


def test_a_record_with_no_applicable_scenario_still_emits_unchanged():
    """The one-to-one guarantee holds even when nothing applies."""
    frame = build_perturbed_frame(
        _profile(exclusions=ExclusionRules(exclude_countries=("ie",))),
        _pairs(),
        profile_version="v1",
        seed=42,
    )

    assert frame.height == 2
    assert not frame.get_column("changed").any()
    assert frame.get_column("scenario_id").null_count() == 2


def test_every_row_records_the_version_of_the_profile_reference():
    """The profile carries no version; the reference it was loaded from does."""
    frame = build_perturbed_frame(_profile(), _pairs(), profile_version="v3", seed=42)

    assert frame.get_column("profile_version").unique().to_list() == ["v3"]


def test_lineage_is_carried_through_unchanged():
    frame = build_perturbed_frame(
        _profile(), _pairs("gleif://LEI1"), profile_version="v1", seed=42
    )
    row = frame.filter(pl.col("source_uri") == "ie://1")

    assert row.get_column("match_uri").item() == "gleif://LEI1"


def test_two_records_sharing_an_identity_in_one_batch_hard_fail():
    """The merge this output is read back through would silently drop the second row.

    Two records with the same `(system, local_id)` compose the same fingerprint and so
    the same `system_uri`, which is a corrupted dataset rather than a duplicate row.
    """
    same_identity = [
        (
            SourceRecord(local_id="1", name=name, system="ie", country="ie"),
            _SourceLineage(
                source_uri="ie://1", match_uri=None, system="ie", country="ie"
            ),
        )
        for name in ("acme systems ltd", "widget corp ltd")
    ]

    with pytest.raises(PerturbationCollisionError, match="more than one record"):
        build_perturbed_frame(_profile(), same_identity, profile_version="v1", seed=42)


def test_the_collision_error_names_the_colliding_identity():
    """So the report says which record to look at, not merely that one collided."""
    same_identity = [
        (
            SourceRecord(local_id="1", name="acme systems ltd", system="ie"),
            _SourceLineage(
                source_uri="ie://1", match_uri=None, system="ie", country="ie"
            ),
        )
    ] * 2

    with pytest.raises(
        PerturbationCollisionError, match="perturbed://ie/robustness-v1/v1/42/1"
    ):
        build_perturbed_frame(_profile(), same_identity, profile_version="v1", seed=42)


def test_distinct_records_do_not_trip_the_collision_check():
    frame = build_perturbed_frame(_profile(), _pairs(), profile_version="v1", seed=42)

    assert frame.get_column("system_uri").n_unique() == frame.height


def test_a_collision_between_batches_of_one_run_hard_fails():
    """The within-batch check cannot see it: the two rows are never in memory together."""
    seen: set[str] = set()
    batch = build_perturbed_frame(_profile(), _pairs(), profile_version="v1", seed=42)

    _check_no_cross_batch_collision(frame=batch, seen=seen)

    with pytest.raises(PerturbationCollisionError, match="more than once"):
        _check_no_cross_batch_collision(frame=batch, seen=seen)


def test_successive_batches_of_distinct_records_pass():
    seen: set[str] = set()

    for local_id in ("1", "2", "3"):
        record = SourceRecord(local_id=local_id, name="acme ltd", system="ie")
        lineage = _SourceLineage(
            source_uri=f"ie://{local_id}", match_uri=None, system="ie", country="ie"
        )
        frame = build_perturbed_frame(
            _profile(), [(record, lineage)], profile_version="v1", seed=42
        )
        _check_no_cross_batch_collision(frame=frame, seen=seen)

    assert len(seen) == 3


def test_the_same_seed_reproduces_the_frame():
    first = build_perturbed_frame(
        _profile(ChainStep("typo.keyboard_substitution", 0.5, 2)),
        _pairs(),
        profile_version="v1",
        seed=42,
    )
    second = build_perturbed_frame(
        _profile(ChainStep("typo.keyboard_substitution", 0.5, 2)),
        _pairs(),
        profile_version="v1",
        seed=42,
    )

    assert first.equals(second)


def test_a_different_seed_gives_different_names():
    def names_at(seed: int) -> list[str]:
        return (
            build_perturbed_frame(
                _profile(ChainStep("typo.keyboard_substitution", 1.0, 2)),
                _pairs(),
                profile_version="v1",
                seed=seed,
            )
            .get_column("name")
            .to_list()
        )

    assert names_at(42) != names_at(43)


# -- manifest paths --------------------------------------------------------------


def test_the_relative_manifest_path_is_repo_relative_and_posix(
    workspace_roots: WorkspaceRoots,
):
    relative = relative_materialization_manifest_path(
        roots=workspace_roots,
        source_system="ie",
        profile_id="robustness-v1",
        version="v1",
        seed=42,
    )

    assert not relative.is_absolute()
    assert str(relative) == (
        "artifacts/perturbed/data/ie/robustness-v1/v1/42/_materialization_manifest_ie.json"
    )


def test_the_absolute_manifest_path_composes_the_relative_one(
    tmp_path: Path, workspace_roots: WorkspaceRoots
):
    assert resolve_materialization_manifest_path(
        roots=workspace_roots,
        source_system="ie",
        profile_id="robustness-v1",
        version="v1",
        seed=42,
    ) == tmp_path / relative_materialization_manifest_path(
        roots=workspace_roots,
        source_system="ie",
        profile_id="robustness-v1",
        version="v1",
        seed=42,
    )


def test_two_seeds_do_not_share_a_manifest(workspace_roots: WorkspaceRoots):
    assert relative_materialization_manifest_path(
        roots=workspace_roots,
        source_system="ie",
        profile_id="robustness-v1",
        version="v1",
        seed=42,
    ) != relative_materialization_manifest_path(
        roots=workspace_roots,
        source_system="ie",
        profile_id="robustness-v1",
        version="v1",
        seed=43,
    )


# -- end to end, in a temp directory ----------------------------------------------


def _write_source_cleansed_system(
    roots: WorkspaceRoots, system: str, rows: list[dict]
) -> None:
    from acquisition.cleanser_orchestrate import materialize_cleansed_merge
    from workspace.data_layout import CLEANSED_LAYER_NAME, system_layer_dir

    cleansed_dir = system_layer_dir(roots, system, layer=CLEANSED_LAYER_NAME)
    chunks_dir = cleansed_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(chunks_dir / f"{system}-001.parquet")
    materialize_cleansed_merge(
        source_dir=chunks_dir,
        output_dir=cleansed_dir,
        system=system,
        name_col="name_cleansed",
        rows_per_file=1_000_000,
        force_rebuild=True,
    )


@pytest.mark.integration
def test_materialize_perturbations_writes_a_dataset(workspace_roots: WorkspaceRoots):
    """The whole path over two rows, entirely inside `tmp_path`.

    Staging, the real cleanse pass, the merge and the schema and quality checks. Two
    rows is enough: what this proves is that the stages hand off to each other, not
    anything about scale.
    """
    from validation.perturbation_materializer import (
        MaterializationConfig,
        materialize_perturbations,
    )

    _write_source_cleansed_system(
        workspace_roots,
        "ie",
        [
            {
                "system_uri": "ie://1",
                "name": "acme systems ltd",
                "name_cleansed": "acme systems",
                "name_cleansed_basic": "acme systems ltd",
                "jurisdiction_code": "ie",
            },
            {
                "system_uri": "ie://2",
                "name": "widget corp ltd",
                "name_cleansed": "widget corp",
                "name_cleansed_basic": "widget corp ltd",
                "jurisdiction_code": "ie",
            },
        ],
    )

    result = materialize_perturbations(
        MaterializationConfig(
            roots=workspace_roots,
            source_system="ie",
            profile=_profile(),
            profile_version="v1",
            seed=42,
            countries=("ie",),
        )
    )

    assert result.rows_emitted == 2
    assert result.output_dir.is_dir()
    assert result.manifest_path.is_file()

    # `chunks/` holds the pre-merge copies of the same rows; the merged view is
    # everything else under the layer, so counting both would double every row.
    merged = [
        path
        for path in result.output_dir.glob("**/*.parquet")
        if "chunks" not in path.parts
    ]
    written = pl.read_parquet(merged)
    assert written.height == 2
    assert set(PERTURBATION_DATASET_REQUIRED_COLUMNS) <= set(written.columns)
