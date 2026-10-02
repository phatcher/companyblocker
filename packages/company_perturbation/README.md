# company_perturbation

Deterministic name-mutation operators for generating perturbed company names, used to measure how robust blocking and matching are to real-world variation. The package reads no dataframes and knows nothing of the repository using it, so a standalone install can load a profile and perturb a name end to end.

## Perturbing a name

An input string and a seed go in, a chain of operators runs over the value in order, and an output string comes out. The same name under the same seed always gives the same result; a different seed gives an independent one.

```python
from company_perturbation import ChainStep, perturb

result = perturb(
    "acme holdings ltd",
    [ChainStep("legal_suffix.variant_substitution", probability=1.0, attempts=1),
     ChainStep("typo.keyboard_substitution", probability=0.4, attempts=2)],
    seed=42,
    country="ie",
)
result.name  # 'acme noldings public limitwd company'
```

Each step fires up to `attempts` changes, each with chance `probability`. Neither scales with the length of a name. Order matters and nothing enforces it: an operator that recognises something, such as a legal suffix, must run before one that mangles characters, and `ChainResult.steps_that_landed_nothing` shows a step that found nothing to act on.

## Operator families

Thirteen operators in seven families, following [TAXONOMY.md](TAXONOMY.md). Importing `company_perturbation.operators` registers them all in `sited_registry`.

| family | operators | a site is |
| --- | --- | --- |
| `typo` | `keyboard_substitution`, `transposition`, `insertion`, `deletion`, `duplication` | a character, or an adjacent pair |
| `diacritic_punct` | `diacritic_substitution`, `separator_substitution` | a character |
| `transliteration` | `homoglyph_substitution` | a character |
| `phonetic` | `soundex_swap` | a word with a Soundex-preserving respelling |
| `word_order` | `word_shuffle` | an adjacent word pair |
| `low_salience` | `token_drop` | a corpus-derived noise word |
| `legal_suffix` | `drop`, `variant_substitution` | the trailing entity-form match, or nothing |

## Profiles and corpora

A profile names a set of weighted scenarios, each an ordered chain of steps:

```json
{
  "profile_id": "robustness",
  "default_seed": 20260910,
  "scenarios": [
    {
      "scenario_id": "suffix-then-typo",
      "weight": 1.0,
      "chain": [
        {"operator_id": "legal_suffix.variant_substitution", "probability": 1.0, "attempts": 1},
        {"operator_id": "typo.keyboard_substitution", "probability": 0.3, "attempts": 2}
      ]
    }
  ]
}
```

A profile may carry `exclusions`, naming the records it does not apply to by `exclude_systems`, `exclude_countries` and `exclude_families`. A scenario's own `exclusions` replace the profile's entirely.

Each record draws one scenario, so a corpus yields one perturbed row per source record, in order, joinable one-to-one against its source:

```python
from pathlib import Path

from company_perturbation import SourceRecord, perturb_records, read_profile

profile = read_profile(Path("profiles/robustness.json"), profile_id="robustness")
rows = perturb_records(
    [SourceRecord(local_id="gb-1", name="acme holdings ltd", system="gb", country="gb")],
    profile,
    seed=42,
)
```

## API

Each module's docstring gives its rules.

- `sited_operator.py`: `Site`, `SitedOperator`, `SitedOperatorRegistry`, `sited_registry`, `apply_operator`, `AppliedChange`. The operator contract, and the runner's selection rules.
- `families.py`: `Family`.
- `chain.py`: `ChainStep`, `StepOutcome`, `ChainResult`, `run_chain`, `perturb`. Running a chain over one name.
- `profile_schema.py`: `PerturbationProfile`, `Scenario`, `ExclusionRules`, `parse_profile`, `serialize_profile`, `validate_profile`, `effective_exclusions`. The authored profile.
- `profile_store.py`: `read_profile`, `ProfileIdMismatchError`. Reading one profile from a file the caller supplies.
- `generation.py`: `SourceRecord`, `PerturbedRecord`, `applicable_scenarios`, `select_scenario`, `perturb_record`, `perturb_records`. One perturbed row per source record.
- `__version__`: The installed package's version.
