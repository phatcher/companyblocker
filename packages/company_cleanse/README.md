# company_cleanse

Deterministic company-name cleansing: normalization, company-type extraction and canonicalisation, and short-name derivation, over a Polars `LazyFrame` or a single string. Reading and writing parquet, and tokenization, are the caller's.

## Use

```python
import polars as pl
from company_cleanse import CleanseConfig, cleanse_lazyframe, strip_company_suffix

lf = pl.LazyFrame({"company_name": ["Acme Systems Plc", '"ABC" Ltd (Associated Business Consultants)']})
cleansed = cleanse_lazyframe(lf, CleanseConfig()).collect()

strip_company_suffix("Acme Systems Plc", company_type="Plc")  # "acme"
strip_company_suffix("Acme Systems Plc", company_type="Plc", normalization_profile="default|-noise_words")  # "acme systems"
```

`cleanse_lazyframe` keeps every input column and appends `name_cleansed`, `name_cleansed_basic`, `short_name`, `quoted_name`, `acronym`, `company_type` and its provenance; its docstring defines each. Behaviour is chosen through `CleanseConfig` and two `|`-separated profiles: `normalization_profile` for the cleansed names, and `short_name_profile` for `short_name`. The profile syntax is `normalize`'s module docstring.

## API

Everything below is importable from `company_cleanse`; each function's docstring gives its contract.

- Cleansing: `cleanse_lazyframe`, `CleanseConfig`, and the single-string equivalents `strip_company_suffix` and `extract_quoted_name`.
- Company types: `get_company_type_rules` recognises a legal-form suffix from any country; `get_company_type_rules_for_country` restricts to one and raises `UnknownCompanyTypeCountryError` for a country with no rule data.
- Noise words: `get_manual_noise_words`, `get_profiled_noise_words`, `get_effective_noise_words` (the one to start from when customising), and the jurisdiction-scoped `get_corpus_noise_words`.
- Short-name candidates: `derive_short_name_candidate` returns a `ShortNameCandidate` from four layered patterns, or `None` rather than a low-confidence guess.
- Reference data: `get_geographic_terms` (the terms the opt-in `geographic_terms` operation strips) and `get_ascii_homoglyphs` (ASCII letters to visually similar characters).

Low-level helpers are not root exports; a caller building its own adapter imports them from `company_cleanse.normalize` and `company_cleanse.extract`.

## Resources

- `resources/company_type_rules.json`: the company-type rules; `rules.py`'s module docstring gives the entry format, and `tests/test_rules.py` checks it.
- `resources/noise_words.json`, `resources/short_name_noise_word_candidates.json`: noise words promoted from `company_tokenize`'s corpus-derived workflow.
- `resources/geographic_terms.json`, `resources/entity_legal_forms_iso20275.json`: the geographic terms and the ISO 20275 legal forms they are partly derived from.

## Diagrams

In the monorepo's `docs/plantuml/`: `cleanse-sequence.puml` (processing flow), `cleanse-company-type-decision-flow.puml` (company-type resolution), `cleanse-short-name-derivation-activity.puml` (short-name derivation) and `pipeline-handoff-canonical-cleanse-tokenize-sequence.puml` (the handoff between stages).
