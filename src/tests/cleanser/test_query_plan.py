import tempfile
from pathlib import Path

import polars as pl
from company_cleanse.rules import get_company_type_rules_for_country

from acquisition.cleanser_orchestrate import name_cleanse


def test_name_cleanse_can_dump_polars_query_plan_files():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        input_dir = root / "input"
        output_dir = root / "output"
        plan_dir = root / "plans"
        input_dir.mkdir(parents=True, exist_ok=True)

        source = pl.DataFrame(
            {
                "CompanyName": [
                    "(MISG) MEDICAL INDUSTRY SUPPORT GROUP LTD",
                    '"TRIPLE D" "PROPERTIES LIMITED"',
                    "OMEGA JSC",
                ]
            }
        )
        source.write_parquet(input_dir / "cases.parquet")

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
            query_plan_output_dir=plan_dir,
        )

        assert rows == 3

        optimized_plan = plan_dir / "cases.optimized.plan.txt"
        unoptimized_plan = plan_dir / "cases.unoptimized.plan.txt"
        assert optimized_plan.exists()
        assert unoptimized_plan.exists()

        optimized_text = optimized_plan.read_text(encoding="utf-8")
        unoptimized_text = unoptimized_plan.read_text(encoding="utf-8")

        # Plan text formats differ across Polars versions; assert only stable semantic markers.
        assert "name_cleansed" in optimized_text
        assert "name_cleansed" in unoptimized_text
        assert "WITH_COLUMNS" in optimized_text.upper()
        assert "WITH_COLUMNS" in unoptimized_text.upper()
