import polars as pl
import pytest
from company_cleanse.pipeline import (
    _step_build_name_body,
    _step_derive_lead_token_fields,
    _step_extract_special_parts,
    _step_prepare_stripped_name,
)
from company_cleanse.rules import (
    COMPANY_TYPE_MAPPING,
    _build_company_type_suffix_trie,
    get_company_type_rules,
)


@pytest.fixture(scope="function")
def benchmark_dataset():
    return pl.DataFrame(
        {
            "CompanyName": [
                "Acme Limited",
                "Smith & Jones LLC",
                "The Global Corporation",
                "(IBIS) International Business Inc",
                '"Microsoft" Software Ltd',
            ]
            * 20,
        }
    )


@pytest.fixture(scope="function")
def initialized_dataset(benchmark_dataset):
    lf = benchmark_dataset.lazy()
    lf = _step_prepare_stripped_name(lf, "CompanyName")
    lf = _step_extract_special_parts(lf, COMPANY_TYPE_MAPPING)
    lf = _step_derive_lead_token_fields(lf)
    lf = _step_build_name_body(lf)
    return lf.collect()


@pytest.fixture(scope="function")
def matcher_setup():
    company_type_regex, company_type_mapping = get_company_type_rules()
    suffix_trie, suffix_trie_max_tokens = _build_company_type_suffix_trie(
        company_type_mapping
    )
    return {
        "regex": company_type_regex,
        "trie": suffix_trie,
        "trie_max_tokens": suffix_trie_max_tokens,
        "mapping": company_type_mapping,
    }
