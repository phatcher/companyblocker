"""Where this repository keeps its perturbation profiles, and how a caller gets one.

`company_perturbation` deliberately holds no location: it reads one profile
from a path it is handed. `workspace` owns where profiles live, as the
`perturbation://<name>/<version>[/draft]` reference layout. This module
is the join between them, so a caller names a profile by reference and never
composes a path: the seam is persistence rather than paths, a caller asking for
the artefact rather than for somewhere to open it.

Named `perturbation_profiles` rather than `profiles` because the bare word
already means several unrelated things here: a cleanse profile, a dataset
profile, a threshold profile, a performance profile.
"""

from __future__ import annotations

from dataclasses import dataclass

from company_perturbation import PerturbationProfile, read_profile

from workspace.kind_layout import Kind
from workspace.reference import (
    Reference,
    Side,
    locate,
    reference,
    require_reference,
    select_references,
)
from workspace.roots import WorkspaceRoots

PROFILE_FILENAME = "profile.json"
"""The file inside a profile reference's directory that holds the profile."""


class SeedNotResolvedError(ValueError):
    """Neither the caller nor the profile said which sample to draw."""


@dataclass(frozen=True)
class AuthoredProfile:
    """A profile and the one reference it was loaded from, whose version and
    stage are what a dataset built from it records."""

    reference: Reference
    profile: PerturbationProfile

    @property
    def version(self) -> str:
        return self.reference.fields["version"]


def load_perturbation_profile(
    roots: WorkspaceRoots, selection: Reference
) -> AuthoredProfile:
    """The one profile `selection` names, checked to exist before it is read.

    A selection matching nothing, or several versions where it names none,
    raises `workspace.reference`'s error naming what does exist.
    """
    found = require_reference(roots, selection)
    profile = read_profile(
        locate(roots, found) / PROFILE_FILENAME, profile_id=found.fields["name"]
    )
    return AuthoredProfile(reference=found, profile=profile)


def available_perturbation_profiles(roots: WorkspaceRoots) -> list[Reference]:
    """Every promoted profile authored in this repository, in URI order."""
    return select_references(roots, reference(Kind.PERTURBATION, Side.PROFILE))


def resolve_seed(profile: PerturbationProfile, *, requested: int | None) -> int:
    """Which sample to draw: the caller's choice, or the profile's canonical one.

    `company_perturbation` takes a seed as an argument and holds no policy about
    where one comes from, which leaves the choice to whoever runs a profile. The
    policy is here: a caller who names a seed gets it, and one who does not gets
    the sample the profile blessed. Two runs of the same battery then agree by
    default rather than by whoever typed the same number twice, and resampling
    stays one deliberate argument.

    A profile with no `default_seed` and a caller with no seed is refused rather
    than defaulted to a literal. A silent `0` would make one arbitrary sample look
    canonical without an author ever having said so, which is the thing the field
    exists to settle.
    """
    if requested is not None:
        return requested
    if profile.default_seed is not None:
        return profile.default_seed
    raise SeedNotResolvedError(
        f"profile {profile.profile_id!r} declares no default_seed, so a seed must be "
        "given. Pass one for this run, or author a 'default_seed' in the profile to "
        "name its canonical sample."
    )
