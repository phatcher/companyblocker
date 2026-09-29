from __future__ import annotations

from pathlib import Path

import polars as pl
from company_cleanse.rules import get_company_type_rules_for_country

from acquisition.cleanser_orchestrate import name_cleanse


def test_name_cleanse_collates_chunk_numbers(tmp_path: Path):
    input_dir = tmp_path / "data" / "gb"
    output_dir = tmp_path / "data" / "gb" / "cleansed"
    input_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"CompanyName": ["Example Limited"]}).write_parquet(
        input_dir / "gb-1.parquet"
    )

    company_type_regex, company_type_mapping = get_company_type_rules_for_country("gb")
    rows = name_cleanse(
        input_dir=input_dir,
        output_dir=output_dir,
        company_type_regex=company_type_regex,
        company_col="CompanyName",
        and_tokens=["AND"],
        char_whitelist=r"[^a-z0-9\s!]",
        company_type_mapping=company_type_mapping,
    )

    assert rows == 1
    assert (output_dir / "chunks" / "gb-001.parquet").exists()
