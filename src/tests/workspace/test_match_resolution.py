"""Direct tests for `workspace.match_resolution`.

What is pinned is that a row's match is its entity's, reached by decomposing the
row's own URI: the only file read is the entity's system's `matched/` layer, so
no names or perturbed dataset exists in any of these fixtures.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest

from workspace.data_layout import MATCHED_LAYER_NAME
from workspace.derived_uri import Perturbation, name_variant_uri, perturbed_uri
from workspace.match_resolution import (
    MatchLayerNotFoundError,
    UnknownUriSchemeError,
    entity_of,
    resolve_cross_system_match,
)
from workspace.roots import WorkspaceRoots

_EN_LITE = Perturbation(profile="en-lite", version="v1", seed=42)
_NAME = name_variant_uri(source_uri="gb://1", name_type="alias", value="Acme")


@pytest.fixture
def matched_gb(layer_fixture_dir: Callable[..., Path]) -> None:
    matched_dir = layer_fixture_dir("gb", layer=MATCHED_LAYER_NAME)
    matched_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"system_uri": ["gb://1", "gb://2"], "match_uri": ["gleif://A", None]}
    ).write_parquet(matched_dir / "gb-001.parquet")


@pytest.mark.parametrize(
    "uri",
    [
        "gb://1",
        _NAME,
        perturbed_uri(source_uri="gb://1", perturbation=_EN_LITE),
        perturbed_uri(source_uri=_NAME, perturbation=_EN_LITE),
    ],
)
def test_a_row_resolves_to_its_entity_s_match_with_no_file_read_but_matched(
    workspace_roots: WorkspaceRoots, matched_gb: None, uri: str
):
    assert entity_of(uri).uri == "gb://1"
    assert resolve_cross_system_match(workspace_roots, uri=uri) == "gleif://A"


def test_an_entity_with_no_recorded_match_is_none(
    workspace_roots: WorkspaceRoots, matched_gb: None
):
    assert resolve_cross_system_match(workspace_roots, uri="gb://2") is None
    assert resolve_cross_system_match(workspace_roots, uri="gb://3") is None


def test_a_system_with_no_matched_layer_is_detected_rather_than_assumed(
    workspace_roots: WorkspaceRoots,
):
    with pytest.raises(MatchLayerNotFoundError):
        resolve_cross_system_match(workspace_roots, uri="gb://1")


@pytest.mark.parametrize(
    "uri",
    ["not-a-uri", "name://abc123", "perturbed://ie/en-lite/v1/42", "perturbed://x/y/z"],
)
def test_a_uri_naming_no_row_is_refused_rather_than_read_as_no_match(
    workspace_roots: WorkspaceRoots, uri: str
):
    with pytest.raises(UnknownUriSchemeError):
        resolve_cross_system_match(workspace_roots, uri=uri)
