"""The operator contract: operators own the change, the runner owns selection.

- An **operator** declares the sites in a name where it *could* act, and knows
  how to make one change at one site. It has no notion of how many.
- The **runner** decides how many changes to attempt, whether each fires, and
  where. One implementation, identical for every family.

**The two knobs.** A step asks for `attempts` independent trials, each firing with
`probability`. So `probability=0.3, attempts=3` means three independent draws at
30%: expected 0.9 edits, never more than 3, and never a function of how long the
name is. Both are absolute quantities a profile author can reason about.

`attempts` is a ceiling on how many changes a step may make, not a count it will
reach. What actually landed is the length of the applied-change tuple this returns.

**Sites are found once, from the input string.** That single call gives both the
pool to draw from and the ceiling: `attempts` is capped at the number of viable
sites, so a three-character name cannot absorb five substitutions however large a
profile's request. The bound comes from the name's own structure rather than from
its length, so a long name is not damaged harder than a short one at the same
setting, and a short name is never asked for more than it can hold.

**Sites are drawn without replacement, and a chosen site consumes every site it
overlaps.** Two edits never land on the same place, and for the families whose
sites overlap by construction -- adjacent character pairs for a transposition,
adjacent word pairs for a swap -- taking one neighbour removes the others, so the
same characters are not swapped twice over. That rule lives here rather than in
each operator because it is about selection, and selection is the runner's job:
for a family whose sites are single characters it is simply a no-op.

**Changes are applied right to left, highest offset first.** An edit then never
shifts the position of one not yet applied, so offsets taken from the input string
stay valid without recomputing anything.

**Fixed draw count for a given name.** Each trial consumes exactly two values from
the generator whether or not it fires, so a step landing no edits and a step
landing several advance the stream identically. That is what makes one generator
safe to carry through a chain. A name offering no sites runs no trials and takes
nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from random import Random

from .families import Family


@dataclass(frozen=True)
class Site:
    """One place in a name where an operator could act.

    A half-open character span. What it delimits is the operator's business: a
    single character for a typo family, a token for a word-drop family, the
    trailing match for a legal-suffix family. The runner only ever counts sites
    and picks one, and never interprets what a site means.
    """

    start: int
    end: int


SiteFinder = Callable[[str, str | None], Sequence[Site]]
"""`(name, country) -> sites`. Country is passed because eligibility can be
jurisdiction-dependent, the same reason the existing contract carries it."""

SiteChanger = Callable[[str, Site, Random, str | None], str]
"""`(name, site, rng, country) -> mutated name`. Makes exactly one change at one
site and returns the whole name. Draws from `rng` only for choices genuinely
internal to the change, such as which adjacent key to substitute."""


@dataclass(frozen=True)
class SitedOperator:
    """An operator under the inverted contract: what it changes, and where it could."""

    operator_id: str
    family: Family
    sites: SiteFinder
    change: SiteChanger


class SitedOperatorRegistry:
    """Operator id to `SitedOperator` lookup."""

    def __init__(self) -> None:
        self._operators: dict[str, SitedOperator] = {}

    def register(self, operator: SitedOperator) -> None:
        if operator.operator_id in self._operators:
            raise ValueError(
                f"operator_id {operator.operator_id!r} is already registered."
            )
        self._operators[operator.operator_id] = operator

    def get(self, operator_id: str) -> SitedOperator:
        try:
            return self._operators[operator_id]
        except KeyError:
            raise KeyError(
                f"no operator registered under id {operator_id!r}."
            ) from None

    def __contains__(self, operator_id: str) -> bool:
        return operator_id in self._operators

    def list_operators(self) -> tuple[str, ...]:
        return tuple(sorted(self._operators))


sited_registry = SitedOperatorRegistry()


@dataclass(frozen=True)
class AppliedChange:
    """One change that actually landed, for provenance."""

    operator_id: str
    family: Family
    site: Site


def apply_operator(
    name: str,
    operator: SitedOperator,
    *,
    rng: Random,
    probability: float,
    attempts: int,
    country: str | None = None,
) -> tuple[str, tuple[AppliedChange, ...]]:
    """Run up to `attempts` independent trials of `operator` over `name`.

    Sites are found once from `name`; `attempts` is capped at how many there are.
    Each trial draws exactly twice, whether it fires and which site, so the
    generator advances the same amount however many land. A site that fires is
    removed along with every site overlapping it, and the chosen changes are then
    applied right to left so the offsets taken from `name` stay valid.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be within [0.0, 1.0], got {probability!r}.")
    if attempts < 0:
        raise ValueError(f"attempts must be >= 0, got {attempts!r}.")

    remaining = list(operator.sites(name, country))
    chosen: list[Site] = []

    for _ in range(min(attempts, len(remaining))):
        fires = rng.random() < probability
        position = rng.random()
        if not fires or not remaining:
            continue

        site = remaining[min(int(position * len(remaining)), len(remaining) - 1)]
        chosen.append(site)
        remaining = [
            other
            for other in remaining
            if other.end <= site.start or other.start >= site.end
        ]

    for site in sorted(chosen, key=lambda site: site.start, reverse=True):
        name = operator.change(name, site, rng, country)

    return name, tuple(
        AppliedChange(
            operator_id=operator.operator_id, family=operator.family, site=site
        )
        for site in sorted(chosen, key=lambda site: site.start)
    )
