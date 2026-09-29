from __future__ import annotations

import glob
from pathlib import Path

import polars as pl


def analyze_private_last_tokens_by_country(
    cleansed_dir: Path,
    output_path: Path | None = None,
    country_col: str = "CountryOfOrigin",
    company_type_col: str = "company_type",
    companyname_col: str = "name_cleansed",
    private_label: str = "private",
    top_n_per_country: int | None = None,
) -> pl.DataFrame:
    cleansed_files = sorted(glob.glob(str(cleansed_dir / "*.parquet")))
    if not cleansed_files:
        raise FileNotFoundError(
            f"No cleansed files in {cleansed_dir}. Run Pass 1 first."
        )
    required_cols = {country_col, company_type_col, companyname_col}

    eligible_files: list[str] = []
    skipped_files = 0
    for file_path in cleansed_files:
        schema_names = set(pl.read_parquet_schema(file_path).keys())
        if required_cols.issubset(schema_names):
            eligible_files.append(file_path)
        else:
            skipped_files += 1

    if not eligible_files:
        missing = ", ".join(sorted(required_cols))
        raise ValueError(
            "No eligible parquet files contain all required analysis columns: "
            f"{missing}"
        )

    lf_parts = [
        pl.scan_parquet(file_path, extra_columns="ignore").select(
            [
                pl.col(country_col),
                pl.col(company_type_col),
                pl.col(companyname_col),
            ]
        )
        for file_path in eligible_files
    ]
    lf = pl.concat(lf_parts, how="vertical_relaxed")

    private_rows = (
        lf.filter(
            pl.col(company_type_col).cast(pl.Utf8).str.to_lowercase()
            == private_label.lower()
        )
        .with_columns(
            [
                pl.col(country_col).fill_null("UNKNOWN").cast(pl.Utf8).alias("country"),
                pl.col(companyname_col)
                .fill_null("")
                .cast(pl.Utf8)
                .str.strip_chars()
                .str.extract(r"(\S+)$", 1)
                .str.to_lowercase()
                .alias("last_token"),
            ]
        )
        .filter(pl.col("last_token").is_not_null() & (pl.col("last_token") != ""))
        .group_by(["country", "last_token"])
        .agg(pl.len().alias("quantity"))
        .sort(["country", "quantity", "last_token"], descending=[False, True, False])
    )

    if top_n_per_country is not None:
        private_rows = (
            private_rows.with_columns(
                pl.col("quantity")
                .rank(method="dense", descending=True)
                .over("country")
                .alias("_rank")
            )
            .filter(pl.col("_rank") <= top_n_per_country)
            .drop("_rank")
        )

    result = private_rows.collect()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".csv":
            result.write_csv(output_path)
        else:
            result.write_parquet(output_path, compression="snappy")
        print(f"Private token analysis written -> {output_path}")

    print(
        "Private token analysis ready "
        f"({len(result):,} country/token rows across "
        f"{result.select(pl.col('country').n_unique()).item():,} countries; "
        f"eligible_files={len(eligible_files)}, skipped_files={skipped_files})."
    )
    return result
