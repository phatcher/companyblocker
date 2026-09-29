from pathlib import Path

import polars as pl
import pytest

from acquisition.cleanser_ops import analyze_private_last_tokens_by_country


def test_analyze_private_last_tokens_by_country_raises_when_no_files(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No cleansed files"):
        analyze_private_last_tokens_by_country(tmp_path)


def test_analyze_private_last_tokens_by_country_raises_when_no_eligible_files(
    tmp_path: Path,
):
    cleansed_dir = tmp_path / "cleansed"
    cleansed_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"Other": ["x"]}).write_parquet(cleansed_dir / "part.parquet")

    with pytest.raises(ValueError, match="No eligible parquet files"):
        analyze_private_last_tokens_by_country(cleansed_dir)


def test_analyze_private_last_tokens_by_country_writes_csv_and_applies_top_n(
    tmp_path: Path,
):
    cleansed_dir = tmp_path / "cleansed"
    output_path = tmp_path / "analysis" / "private_last_tokens.csv"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(
        {
            "CountryOfOrigin": ["Belgium", "Belgium", "Belgium", "Belgium"],
            "company_type": ["PRIVATE", "PRIVATE", "PRIVATE", "PRIVATE"],
            "name_cleansed": ["A NV", "B NV", "C BV", "D NV"],
        }
    ).write_parquet(cleansed_dir / "part.parquet")

    result = analyze_private_last_tokens_by_country(
        cleansed_dir,
        output_path=output_path,
        top_n_per_country=1,
    )

    assert output_path.exists()
    assert result.to_dicts() == [
        {"country": "Belgium", "last_token": "nv", "quantity": 3}
    ]
