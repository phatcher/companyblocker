"""Direct tests for `validation.perturbation_profiles`.

The module is the join between a package that holds no location and a workspace
that owns one, so what it has to prove is that it resolves against the anchor
rather than against a path a caller composed: given a root, a caller names a
profile and gets it. Everything about the profile's own shape is
`company_perturbation`'s and is asserted there.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from company_perturbation import (
    ChainStep,
    PerturbationProfile,
    Scenario,
)

from validation.perturbation_profiles import (
    PROFILE_FILENAME,
    SeedNotResolvedError,
    available_perturbation_profiles,
    load_perturbation_profile,
    resolve_seed,
)
from workspace.kind_layout import Kind
from workspace.reference import (
    AmbiguousReferenceError,
    Reference,
    ReferenceNotFoundError,
    Side,
    locate,
    reference,
)
from workspace.repository import repository_root
from workspace.roots import WorkspaceRoots, default_workspace_roots


def _profile(**kwargs) -> PerturbationProfile:
    return PerturbationProfile(
        profile_id="light-noise",
        scenarios=(
            Scenario(
                scenario_id="one-typo",
                chain=(ChainStep("typo.keyboard_substitution", 1.0, 1),),
            ),
        ),
        **kwargs,
    )


def _ref(name: str, version: str | None = None) -> Reference:
    values = {"name": name} if version is None else {"name": name, "version": version}
    return reference(Kind.PERTURBATION, Side.PROFILE, **values)


def _author(roots: WorkspaceRoots, name: str, version: str = "v1", **extra) -> None:
    directory = locate(roots, _ref(name, version))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / PROFILE_FILENAME).write_text(
        json.dumps(
            {
                "profile_id": name,
                "scenarios": [
                    {
                        "scenario_id": "one-typo",
                        "chain": [{"operator_id": "typo.keyboard_substitution"}],
                    }
                ],
                **extra,
            }
        ),
        encoding="utf-8",
    )


def test_a_profile_is_loaded_by_reference_with_the_reference_it_resolved_to(
    workspace_roots,
):
    _author(workspace_roots, "light-noise")

    loaded = load_perturbation_profile(workspace_roots, _ref("light-noise"))

    assert loaded.profile.profile_id == "light-noise"
    assert loaded.reference == _ref("light-noise", "v1")
    assert loaded.version == "v1"


def test_the_profiles_available_are_the_ones_authored_under_the_config_root(
    workspace_roots,
):
    _author(workspace_roots, "light-noise")
    _author(workspace_roots, "heavy-noise", "v2")

    assert available_perturbation_profiles(workspace_roots) == [
        _ref("heavy-noise", "v2"),
        _ref("light-noise", "v1"),
    ]


def test_a_name_no_profile_was_authored_under_names_what_exists(workspace_roots):
    _author(workspace_roots, "light-noise")

    with pytest.raises(ReferenceNotFoundError, match="perturbation://light-noise/v1"):
        load_perturbation_profile(workspace_roots, _ref("lite-noise"))


def test_a_name_with_several_versions_must_say_which(workspace_roots):
    _author(workspace_roots, "light-noise", "v1")
    _author(workspace_roots, "light-noise", "v2")

    with pytest.raises(AmbiguousReferenceError):
        load_perturbation_profile(workspace_roots, _ref("light-noise"))


def test_an_authored_default_seed_is_loaded_with_the_profile(workspace_roots):
    _author(workspace_roots, "light-noise", default_seed=20260910)

    loaded = load_perturbation_profile(workspace_roots, _ref("light-noise"))

    assert loaded.profile.default_seed == 20260910


def test_the_repositorys_own_profiles_load():
    """The two tracked profiles sit where their references locate them."""
    roots = default_workspace_roots(repository_root(Path(__file__)))

    for name in ("en-lite", "en-robustness"):
        assert (
            load_perturbation_profile(roots, _ref(name, "v1")).profile.profile_id
            == name
        )


# --- which sample a run draws ----------------------------------------------


def test_a_caller_that_names_a_seed_gets_it():
    profile = _profile(default_seed=20260910)

    assert resolve_seed(profile, requested=7) == 7


def test_a_caller_that_names_no_seed_gets_the_profiles_canonical_sample():
    """So two runs of one battery agree without anyone typing the same number twice."""
    profile = _profile(default_seed=20260910)

    assert resolve_seed(profile, requested=None) == 20260910


def test_a_seed_of_zero_from_the_caller_is_a_choice_rather_than_an_absence():
    profile = _profile(default_seed=20260910)

    assert resolve_seed(profile, requested=0) == 0


def test_a_profile_may_bless_zero_as_its_canonical_sample():
    assert resolve_seed(_profile(default_seed=0), requested=None) == 0


def test_neither_a_caller_nor_a_profile_naming_a_seed_is_refused():
    """Defaulting to a literal would make one arbitrary sample look canonical."""
    with pytest.raises(SeedNotResolvedError):
        resolve_seed(_profile(), requested=None)


def test_the_refusal_names_both_remedies():
    with pytest.raises(SeedNotResolvedError, match="Pass one for this run, or author"):
        resolve_seed(_profile(), requested=None)
