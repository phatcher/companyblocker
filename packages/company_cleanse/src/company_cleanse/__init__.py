"""Deterministic company-name cleansing: normalization, company-type extraction and short names.

`cleanse_lazyframe` is the batch entry point over a Polars `LazyFrame`; `strip_company_suffix`
and `extract_quoted_name` are single-string equivalents of two of its output columns. The root
exports are the supported surface. Low-level helpers live in `normalize` and `extract` and are
imported from there by a caller building its own adapter. Reading and writing parquet, and
tokenization, belong to the caller.
"""

from .api import cleanse_lazyframe, extract_quoted_name, strip_company_suffix
from .config import (
    CleanseConfig,
    get_effective_noise_words,
    get_manual_noise_words,
    get_profiled_noise_words,
)
from .corpus_noise_words import get_corpus_noise_words
from .geographic_terms import get_geographic_terms
from .normalize import get_ascii_homoglyphs
from .rules import (
    UnknownCompanyTypeCountryError,
    get_company_type_rules,
    get_company_type_rules_for_country,
)
from .short_name_candidates import ShortNameCandidate, derive_short_name_candidate

# This is the package's whole supported public surface (see README's "Imports"
# section for the classification behind each entry). Everything else --
# normalize.py, extract.py, pipeline.py internals, and any name starting with
# "_" -- is an internal helper: reachable by importing the submodule directly,
# but not part of the contract this package holds itself to. Adding a name
# here without adding it to __all__ below is caught by
# test_api.py::test_package_root_has_no_untracked_public_names.
__all__ = [
    "CleanseConfig",
    "get_manual_noise_words",
    "get_profiled_noise_words",
    "get_effective_noise_words",
    "get_corpus_noise_words",
    "cleanse_lazyframe",
    "strip_company_suffix",
    "extract_quoted_name",
    "get_company_type_rules",
    "get_company_type_rules_for_country",
    "UnknownCompanyTypeCountryError",
    "get_ascii_homoglyphs",
    "get_geographic_terms",
    "ShortNameCandidate",
    "derive_short_name_candidate",
]
