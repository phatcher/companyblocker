import polars as pl
import pytest

from blocking.truth import (
    ColumnTruth,
    MatchedLayerTruth,
    SourceTruthResolver,
    source_truth_for_column,
)
from workspace.roots import WorkspaceRoots


def test_source_truth_for_column_maps_the_default_to_the_matched_layer_rule() -> None:
    assert source_truth_for_column("match_uri") == MatchedLayerTruth()
    assert source_truth_for_column("source_uri") == ColumnTruth("source_uri")
    assert isinstance(source_truth_for_column(" source_uri "), SourceTruthResolver)


def test_source_truth_for_column_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        source_truth_for_column("  ")


def test_truth_column_names_the_column_each_rule_reads() -> None:
    assert MatchedLayerTruth().truth_column == "match_uri"
    assert ColumnTruth("source_uri").truth_column == "source_uri"


def test_column_truth_reads_the_named_column_as_each_rows_target(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A perturbed row's truth is the identity it was derived from, read off
    the row itself: no walk, no matched/ layer on disk."""
    source_frame = pl.DataFrame(
        {
            "system_uri": ["perturbed://ie/p/1/a", "perturbed://ie/p/1/b"],
            "source_uri": ["ie://1", None],
            "match_uri": [None, None],
        }
    )

    result = ColumnTruth("source_uri").resolve(
        source_frame, roots=workspace_roots, source_system="ie", target_system="ie"
    )

    assert result.sort("source_id").to_dicts() == [
        {"source_id": "perturbed://ie/p/1/a", "source_match_uri": "ie://1"},
        {"source_id": "perturbed://ie/p/1/b", "source_match_uri": None},
    ]


def test_column_truth_refuses_a_column_naming_rows_of_another_system(
    workspace_roots: WorkspaceRoots,
) -> None:
    """A populated value whose scheme is not the target's names a row in some
    other system, so scoring would count every truth pair as missed."""
    source_frame = pl.DataFrame(
        {"system_uri": ["perturbed://ie/p/1/a"], "source_uri": ["ie://1"]}
    )

    with pytest.raises(ValueError, match="not in target 'gb'"):
        ColumnTruth("source_uri").resolve(
            source_frame, roots=workspace_roots, source_system="ie", target_system="gb"
        )


def test_column_truth_refuses_a_frame_without_the_column(
    workspace_roots: WorkspaceRoots,
) -> None:
    source_frame = pl.DataFrame({"system_uri": ["ie://1"], "name": ["acme"]})

    with pytest.raises(ValueError, match="source_uri"):
        ColumnTruth("source_uri").resolve(
            source_frame, roots=workspace_roots, source_system="ie", target_system="ie"
        )


def test_column_truth_evaluates_on_the_frame_not_the_layer() -> None:
    """Whether a frame can be scored is whether it carries the column, so a
    name sidecar handed in as an override scores whatever layer its
    descriptor resolved to."""
    with_column = pl.DataFrame({"system_uri": ["name://a"], "source_uri": ["gb:1"]})
    without = pl.DataFrame({"system_uri": ["gb:1"]})

    truth = ColumnTruth("source_uri")

    assert truth.can_evaluate(with_column, source_has_ground_truth=False)
    assert not truth.can_evaluate(without, source_has_ground_truth=True)
    assert MatchedLayerTruth().can_evaluate(without, source_has_ground_truth=True)
    assert not MatchedLayerTruth().can_evaluate(
        with_column, source_has_ground_truth=False
    )


def test_matched_layer_truth_pairing_needs_the_target_in_the_match_metadata() -> None:
    truth = MatchedLayerTruth()

    with pytest.raises(ValueError, match="unrelated match_uri"):
        truth.validate_pairing(
            source_system="gleif",
            target_system="ie",
            source_has_ground_truth=True,
            matched_target_systems=("gb",),
        )
    # A column rule has no metadata to check: the scheme is checked when the
    # frame is resolved instead.
    ColumnTruth("source_uri").validate_pairing(
        source_system="ie",
        target_system="ie",
        source_has_ground_truth=True,
        matched_target_systems=(),
    )
