"""The authored profile: named, versioned scenarios over the operator contract.

A **profile** is the governed unit a run is named by: its `profile_id`, and a
version that is where the repository keeps it rather than a field of its own. It
holds **scenarios**, each an ordered **chain** of steps, and each step names one
operator with two numbers that both have units:

- `probability`: The chance any one change fires.
- `attempts`: How many are tried, and so the ceiling on how many may land.

Neither scales with the length of a name, so a forty-character name is not corrupted
twice as hard as a twenty-character one at the same setting, and a single-site
operator such as `legal_suffix.drop` is a step with one attempt rather than a
threshold.

**No scenario-level default.** Every step states both its own numbers, and nothing
is inherited from the scenario holding it.

**Order is the author's, and it matters.** An operator that must recognize something
-- a legal suffix, a noise word -- has to run before one that perturbs characters,
or it finds nothing left to recognize. `chain.ChainResult.steps_that_landed_nothing`
is what surfaces a chain ordered the wrong way round; nothing here enforces it.

**One scenario per record, drawn by `weight`.** A profile is a set of corruption
kinds a record *might* have suffered, not a list of things that all happen to it. A
record draws exactly one of the scenarios that apply to it, so a corpus of 5,000
records produces 5,000 perturbed rows and adding a scenario changes the mix rather
than multiplying the output. It also matches how data actually goes wrong: a record
has one history, not one per kind of corruption.

The cost is that two scenarios can never be compared on the *same* record, since
each record only ever draws one. Comparing corruption kinds is a between-groups
question across the corpus, or a second run under a different seed.

**An optional canonical seed.** `default_seed` names the sample a caller gets when it
does not choose one. Without it a profile has no standard sample: two runs of the same
battery under two invented seeds are both valid and neither is the one to cite, which a
profile is the only thing able to settle. It is a convention, not part of the profile's
identity -- the profile's id and version describe the corruption model, and a
materialized dataset is named `perturbed:<source>:<profile_id>:<version>:<seed>`, so the seed a
run actually used is recoverable from its own output whether it was authored or passed.
Changing `default_seed` therefore re-blesses which sample is standard and leaves every
existing dataset holding its own name: it is not a model change and needs no new
version.

Nothing in this package reads the field. `perturb_records` still takes its seed as an
argument, because the seed being an input is what lets one profile yield independent
samples; resolving an authored default against a caller's own choice is the caller's,
and this package only carries the number the author wrote down.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .chain import ChainStep
from .families import Family
from .sited_operator import SitedOperatorRegistry, sited_registry

_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class ExclusionRules:
    """Which records a profile or scenario does not apply to."""

    exclude_systems: tuple[str, ...] = ()
    exclude_countries: tuple[str, ...] = ()
    exclude_families: tuple[str, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """One named chain: a kind of corruption a record may have suffered.

    `weight` is its share of the draw. A record picks exactly one scenario from
    those that apply to it, so a corpus produces as many rows as it started with
    rather than one per scenario. Weights are relative, not probabilities: they are
    normalised across whichever scenarios survive exclusions for a given record.
    """

    scenario_id: str
    chain: tuple[ChainStep, ...]
    weight: float = 1.0
    exclusions: ExclusionRules | None = None
    description: str | None = None


@dataclass(frozen=True)
class PerturbationProfile:
    profile_id: str
    scenarios: tuple[Scenario, ...]
    exclusions: ExclusionRules = field(default_factory=ExclusionRules)
    description: str | None = None
    default_seed: int | None = None
    """The sample to draw when a caller names no seed. Advisory, and outside identity."""


def _parse_exclusions(raw: Mapping[str, Any] | None) -> ExclusionRules | None:
    if raw is None:
        return None
    return ExclusionRules(
        exclude_systems=tuple(str(value) for value in raw.get("exclude_systems", ())),
        exclude_countries=tuple(
            str(value) for value in raw.get("exclude_countries", ())
        ),
        exclude_families=tuple(str(value) for value in raw.get("exclude_families", ())),
    )


def _parse_step(raw: Mapping[str, Any]) -> ChainStep:
    return ChainStep(
        operator_id=str(raw["operator_id"]),
        probability=float(raw.get("probability", 1.0)),
        attempts=int(raw.get("attempts", 1)),
    )


def parse_profile(raw: Mapping[str, Any]) -> PerturbationProfile:
    declared_seed = raw.get("default_seed")
    return PerturbationProfile(
        profile_id=str(raw["profile_id"]),
        description=raw.get("description"),
        default_seed=None if declared_seed is None else int(declared_seed),
        exclusions=_parse_exclusions(raw.get("exclusions")) or ExclusionRules(),
        scenarios=tuple(
            Scenario(
                scenario_id=str(scenario["scenario_id"]),
                weight=float(scenario.get("weight", 1.0)),
                description=scenario.get("description"),
                exclusions=_parse_exclusions(scenario.get("exclusions")),
                chain=tuple(_parse_step(step) for step in scenario["chain"]),
            )
            for scenario in raw["scenarios"]
        ),
    )


def serialize_profile(profile: PerturbationProfile) -> dict[str, object]:
    def _exclusions(rules: ExclusionRules) -> dict[str, object]:
        return {
            "exclude_systems": list(rules.exclude_systems),
            "exclude_countries": list(rules.exclude_countries),
            "exclude_families": list(rules.exclude_families),
        }

    return {
        "profile_id": profile.profile_id,
        "description": profile.description,
        "default_seed": profile.default_seed,
        "exclusions": _exclusions(profile.exclusions),
        "scenarios": [
            {
                "scenario_id": scenario.scenario_id,
                "weight": scenario.weight,
                "description": scenario.description,
                **(
                    {"exclusions": _exclusions(scenario.exclusions)}
                    if scenario.exclusions is not None
                    else {}
                ),
                "chain": [
                    {
                        "operator_id": step.operator_id,
                        "probability": step.probability,
                        "attempts": step.attempts,
                    }
                    for step in scenario.chain
                ],
            }
            for scenario in profile.scenarios
        ],
    }


def effective_exclusions(
    scenario: Scenario, *, profile: PerturbationProfile
) -> ExclusionRules:
    """A scenario's own rules replace the profile's entirely, never merge with them."""
    return (
        scenario.exclusions if scenario.exclusions is not None else profile.exclusions
    )


def validate_profile(
    profile: PerturbationProfile, *, registry: SitedOperatorRegistry = sited_registry
) -> None:
    """Raise on anything a run could not act on."""
    if not _ID_PATTERN.match(profile.profile_id):
        raise ValueError(
            f"profile_id {profile.profile_id!r} must be lowercase kebab-case."
        )
    if not profile.scenarios:
        raise ValueError(f"profile {profile.profile_id!r} declares no scenarios.")

    seen: set[str] = set()
    known_families = {member.value for member in Family}

    for scenario in profile.scenarios:
        if not _ID_PATTERN.match(scenario.scenario_id):
            raise ValueError(
                f"scenario_id {scenario.scenario_id!r} must be lowercase kebab-case."
            )
        if scenario.scenario_id in seen:
            raise ValueError(f"scenario_id {scenario.scenario_id!r} is declared twice.")
        seen.add(scenario.scenario_id)

        if not scenario.chain:
            raise ValueError(f"scenario {scenario.scenario_id!r} has an empty chain.")
        if scenario.weight < 0:
            raise ValueError(
                f"scenario {scenario.scenario_id!r} weight {scenario.weight!r} must be >= 0."
            )

        for step in scenario.chain:
            if step.operator_id not in registry:
                raise ValueError(
                    f"scenario {scenario.scenario_id!r} names unknown operator "
                    f"{step.operator_id!r}."
                )
            if not 0.0 <= step.probability <= 1.0:
                raise ValueError(
                    f"{step.operator_id!r} probability {step.probability!r} is outside "
                    "[0.0, 1.0]."
                )
            if step.attempts < 0:
                raise ValueError(
                    f"{step.operator_id!r} attempts {step.attempts!r} must be >= 0."
                )

    for rules in (profile.exclusions, *(s.exclusions for s in profile.scenarios)):
        for family in rules.exclude_families if rules else ():
            if family not in known_families:
                raise ValueError(f"unknown family {family!r} in exclusions.")
