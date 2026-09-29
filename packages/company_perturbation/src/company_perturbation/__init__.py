"""Name perturbation: an input string and a seed in, a mutated string out.

An operator declares where in a name it *could* act and how to make one change at
one of those sites; the runner decides how many changes to attempt, whether each
fires and where. Nothing here reads a dataframe, composes a URI or knows what a
`system_uri` is -- the repository using this package owns all of that, which is what
keeps the package publishable on its own.

Nothing is drawn from entropy, and nothing is derived from the identity of the thing
being perturbed: randomness comes from a generator the caller supplies, so the same
name under two seeds gives two independent, individually reproducible results.

    from company_perturbation import ChainStep, perturb

    perturb("acme holdings ltd", [ChainStep("typo.keyboard_substitution", 0.4, 2)],
            seed=42, country="ie")
"""

from importlib.metadata import PackageNotFoundError, version

from . import operators  # noqa: F401  (importing a family registers it)
from .chain import (
    ChainResult,
    ChainStep,
    StepOutcome,
    perturb,
    run_chain,
)
from .families import Family
from .generation import (
    PerturbedRecord,
    SourceRecord,
    applicable_scenarios,
    perturb_record,
    perturb_records,
    select_scenario,
)
from .profile_schema import (
    ExclusionRules,
    PerturbationProfile,
    Scenario,
    effective_exclusions,
    parse_profile,
    serialize_profile,
    validate_profile,
)
from .profile_store import ProfileIdMismatchError, read_profile
from .sited_operator import (
    AppliedChange,
    Site,
    SitedOperator,
    SitedOperatorRegistry,
    apply_operator,
    sited_registry,
)

try:
    __version__ = version("company-perturbation")
except PackageNotFoundError:  # pragma: no cover - only when not installed
    __version__ = "0.0.0"

__all__ = [
    "AppliedChange",
    "ChainResult",
    "ChainStep",
    "ExclusionRules",
    "Family",
    "PerturbationProfile",
    "PerturbedRecord",
    "ProfileIdMismatchError",
    "Scenario",
    "Site",
    "SitedOperator",
    "SitedOperatorRegistry",
    "SourceRecord",
    "StepOutcome",
    "__version__",
    "applicable_scenarios",
    "apply_operator",
    "effective_exclusions",
    "parse_profile",
    "perturb",
    "perturb_record",
    "perturb_records",
    "read_profile",
    "run_chain",
    "select_scenario",
    "serialize_profile",
    "sited_registry",
    "validate_profile",
]
