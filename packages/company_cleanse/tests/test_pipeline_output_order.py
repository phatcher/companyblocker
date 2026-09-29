import polars as pl
from company_cleanse.pipeline import _process_cleanse_lazyframe
from company_cleanse.rules import get_company_type_rules


def test_process_cleanse_lazyframe_preserves_input_columns_and_drops_temp_work_columns():
    source = pl.DataFrame(
        {
            "CompanyName": ["Acme Ltd"],
            "source_record_uri": ["ie://4"],
            "custom_payload": ["keep-me"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = _process_cleanse_lazyframe(
        source.lazy(),
        company_col="CompanyName",
        company_type_regex=company_type_regex,
        company_type_mapping=company_type_mapping,
        company_type_matcher="regex",
        and_tokens=("AND",),
        char_whitelist=r"[^a-z0-9\s!&]",
        source_company_type_col=None,
        source_company_type_mapping=None,
        use_source_company_type=False,
    ).collect()

    assert "source_record_uri" in actual.columns
    assert actual["source_record_uri"].to_list() == ["ie://4"]
    assert "custom_payload" in actual.columns
    assert actual["custom_payload"].to_list() == ["keep-me"]
    assert "_lower_name" not in actual.columns
    assert "_name_body" not in actual.columns


def test_process_cleanse_lazyframe_keeps_input_order_then_appends_derived_columns():
    source = pl.DataFrame(
        {
            "jurisdiction_code": ["ie"],
            "company_number": ["123456"],
            "name": ["Acme Limited"],
            "custom_payload": ["keep-me"],
        }
    )

    company_type_regex, company_type_mapping = get_company_type_rules()
    actual = _process_cleanse_lazyframe(
        source.lazy(),
        company_col="name",
        company_type_regex=company_type_regex,
        company_type_mapping=company_type_mapping,
        company_type_matcher="regex",
        and_tokens=("AND",),
        char_whitelist=r"[^a-z0-9\s!&]",
        source_company_type_col=None,
        source_company_type_mapping=None,
        use_source_company_type=False,
    ).collect()

    input_columns = ["jurisdiction_code", "company_number", "name", "custom_payload"]
    assert actual.columns[: len(input_columns)] == input_columns

    derived_tail = [
        "short_name",
        "quoted_name",
        "acronym",
        "name_cleansed_basic",
        "name_cleansed",
        "personal_owner",
        "company_type_source",
        "company_type_missing",
    ]
    for column_name in derived_tail:
        assert column_name in actual.columns

    tail_positions = [actual.columns.index(column_name) for column_name in derived_tail]
    assert tail_positions == sorted(tail_positions)
