"""Running an ordered chain of operators over one name, carrying one generator.

This is the unit the whole design rests on: an input string and a seed go in, a
series of operators run over the value in order, an output string comes out. The
generator is created by the caller from its own seed and threaded through every
step, so the result is deterministic *in the seed* rather than in the identity of
the record being perturbed. Two seeds give two independent, individually
reproducible samples of the same name -- which is what makes a robustness
measurement able to carry an error bar rather than being a single fixed draw.

**Order matters, and the chain does not reorder itself.** An operator that has to
*recognise* something needs intact input: a legal suffix, a noise word, a word
boundary. An operator that perturbs characters destroys what the first kind reads.
So `legal_suffix.variant_substitution` then `typo.keyboard_substitution` models
something real -- a source recorded `Ltd` as `Limited`, and someone later mistyped
it -- while the reverse asks an operator to find a suffix in a name whose suffix
has already been mangled.

That failure is silent by construction: a mangled suffix simply yields no sites, so
the step makes no change and nothing raises. `StepOutcome.requested` against
`len(StepOutcome.changes)` is what surfaces it, so a caller can see that a step
asked for changes and landed none rather than discovering it in the output.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from random import Random

from .sited_operator import (
    AppliedChange,
    SitedOperatorRegistry,
    apply_operator,
    sited_registry,
)


@dataclass(frozen=True)
class ChainStep:
    """One operator in a chain, with how hard to push it.

    `probability` is the chance any one change fires; `attempts` is how many are
    tried, and so the ceiling on how many may land. Neither scales with the length
    of the name.
    """

    operator_id: str
    probability: float = 1.0
    attempts: int = 1


@dataclass(frozen=True)
class StepOutcome:
    """What one step of a chain actually did."""

    operator_id: str
    requested: int
    changes: tuple[AppliedChange, ...]

    @property
    def landed_nothing(self) -> bool:
        """Asked for changes and made none.

        Not an error: a name may simply offer the operator nowhere to act. It is
        worth surfacing because the commonest cause is a chain ordered so that an
        earlier step destroyed what this one needed to recognise.
        """
        return self.requested > 0 and not self.changes


@dataclass(frozen=True)
class ChainResult:
    name: str
    outcomes: tuple[StepOutcome, ...]

    @property
    def changes(self) -> tuple[AppliedChange, ...]:
        return tuple(change for outcome in self.outcomes for change in outcome.changes)

    @property
    def steps_that_landed_nothing(self) -> tuple[str, ...]:
        return tuple(
            outcome.operator_id for outcome in self.outcomes if outcome.landed_nothing
        )


def run_chain(
    name: str,
    steps: Iterable[ChainStep],
    *,
    rng: Random,
    registry: SitedOperatorRegistry = sited_registry,
    country: str | None = None,
) -> ChainResult:
    """Run `steps` over `name` in order, threading `rng` through all of them.

    Each step receives what the previous one returned, so the chain composes as
    written. The generator is never reset between steps: one seed governs the whole
    chain, and a step's draws follow on from the last.
    """
    outcomes: list[StepOutcome] = []

    for step in steps:
        operator = registry.get(step.operator_id)
        name, changes = apply_operator(
            name,
            operator,
            rng=rng,
            probability=step.probability,
            attempts=step.attempts,
            country=country,
        )
        outcomes.append(
            StepOutcome(
                operator_id=step.operator_id,
                requested=step.attempts,
                changes=changes,
            )
        )

    return ChainResult(name=name, outcomes=tuple(outcomes))


def perturb(
    name: str,
    steps: Sequence[ChainStep],
    *,
    seed: int,
    registry: SitedOperatorRegistry = sited_registry,
    country: str | None = None,
) -> ChainResult:
    """`run_chain` from a seed rather than a generator.

    The seed is the caller's, and the only thing that varies the outcome: the same
    name and the same seed always give the same result, and a different seed gives
    an independent one. Nothing is derived from the record's own identity.
    """
    return run_chain(name, steps, rng=Random(seed), registry=registry, country=country)  # nosec B311 - seeded for reproducibility, not security
