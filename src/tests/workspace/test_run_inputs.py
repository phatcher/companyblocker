"""Direct tests for `workspace.run_inputs`.

What is pinned is that each natural way of naming a run's input gives the
reference it means, that what cannot be meant is refused, and that a request
leaving out a version or a seed resolves against what is on disk.
"""

from __future__ import annotations

import pytest

from workspace.reference import (
    AmbiguousReferenceError,
    InvalidReferenceError,
    locate,
    resolve_latest,
)
from workspace.roots import WorkspaceRoots
from workspace.run_inputs import (
    implied_truth_column,
    profile_request,
    source_request,
    target_request,
)


@pytest.mark.parametrize(
    ("facts", "uri", "truth"),
    [
        ({"source": "gleif"}, "data://gleif/matched", "match_uri"),
        ({"source": "gb", "dataset": "names"}, "data://gb/names", "source_uri"),
        (
            {"source": "gb", "dataset": "names", "date": "2026-06-01"},
            "data://gb/names/2026-06-01",
            "source_uri",
        ),
        (
            {"source": "ie", "perturbed": "en-lite"},
            "perturbed://ie/en-lite",
            "source_uri",
        ),
        (
            {"source": "ie", "perturbed": "en-lite", "version": "v1", "seed": 42},
            "perturbed://ie/en-lite/v1/42",
            "source_uri",
        ),
        (
            {"source": "gb", "dataset": "names", "perturbed": "en-lite", "seed": 42},
            "perturbed://gb/names/en-lite/*/42",
            "source_uri",
        ),
    ],
)
def test_a_source_is_named_by_natural_facts(facts: dict, uri: str, truth: str):
    request = source_request(**facts)

    assert request.uri == uri
    assert implied_truth_column(request) == truth


def test_a_target_is_a_system_s_canonical_snapshot_the_latest_unless_dated():
    assert target_request(target="ie").uri == "data://ie/canonical"
    assert target_request(target="ie", date="2026-06-01").uri == (
        "data://ie/canonical/2026-06-01"
    )


def test_a_profile_is_the_latest_promoted_version_unless_named_or_draft():
    assert profile_request(perturb="en-lite").uri == "perturbation://en-lite"
    assert profile_request(perturb="en-lite", version="v2", draft=True).uri == (
        "perturbation://en-lite/v2/draft"
    )


@pytest.mark.parametrize(
    "facts",
    [
        {},
        {"source": "gb", "dataset": "canonical"},
        {"source": "gb", "seed": 42},
        {"source": "gb", "version": "v1"},
        {"source": "gb", "date": "2026-06-01"},
        {"source": "ie", "perturbed": "en-lite", "date": "2026-06-01"},
        {"benchmark": "amazon-google", "source": "gb"},
        {"benchmark": "amazon-google", "perturbed": "en-lite"},
        {"benchmark": "amazon-google"},
    ],
)
def test_facts_that_name_no_source_are_refused(facts: dict):
    with pytest.raises(InvalidReferenceError):
        source_request(**facts)


def test_a_version_left_out_is_the_latest_and_a_seed_left_out_must_be_the_only_one(
    workspace_roots: WorkspaceRoots,
):
    def made(version: str, seed: int) -> None:
        dataset = source_request(
            source="ie", perturbed="en-lite", version=version, seed=seed
        )
        locate(workspace_roots, dataset).mkdir(parents=True)

    made("v9", 42)
    made("v10", 42)
    request = source_request(source="ie", perturbed="en-lite")

    assert (
        resolve_latest(workspace_roots, request).uri == "perturbed://ie/en-lite/v10/42"
    )

    made("v10", 7)

    with pytest.raises(AmbiguousReferenceError, match="v10/42") as refused:
        resolve_latest(workspace_roots, request)
    assert "v10/7" in str(refused.value)
    assert (
        resolve_latest(
            workspace_roots, source_request(source="ie", perturbed="en-lite", seed=7)
        ).uri
        == "perturbed://ie/en-lite/v10/7"
    )
