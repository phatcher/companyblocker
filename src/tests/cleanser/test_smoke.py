import tempfile
from pathlib import Path

import polars as pl
from company_cleanse.rules import get_company_type_rules_for_country

from acquisition.cleanser_orchestrate import name_cleanse


def test_name_cleanse_smoke_single_row_end_to_end():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": ["Acme Ltd"],
            }
        )
        source.write_parquet(input_dir / "smoke.parquet")

        company_type_regex, company_type_mapping = get_company_type_rules_for_country(
            "gb"
        )
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

        actual = pl.read_parquet(output_dir / "chunks" / "smoke.parquet")
        assert actual.height == 1

        row = actual.row(0, named=True)
        assert row["company_type"] == "ltd"
        assert row["name_cleansed"] == "acme ltd"
        assert row["short_name"] == "acme"
        assert row["quoted_name"] is None
