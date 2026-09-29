from pathlib import Path

import polars as pl
import pytest
from company_tokenize import (
    TokenizationRequest,
    TokenizeFilesRequest,
    TokenizerCalculationSpec,
    compile_tokenization_request,
    tokenize_name_dataframe_with_request,
    tokenize_name_with_request,
)


def test_compile_tokenization_request_rejects_duplicate_token_cols():
    request = TokenizationRequest(
        tokenizer_calculations=(
            TokenizerCalculationSpec(token_col="tokens"),
            TokenizerCalculationSpec(token_col="tokens"),
        )
    )

    with pytest.raises(ValueError, match="Duplicate token column requested"):
        compile_tokenization_request(request)


def test_tokenize_name_dataframe_with_request_supports_multiple_calculations():
    df = pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]})
    request = TokenizationRequest(
        noise_words_profile="aggressive",
        tokenizer_calculations=(
            TokenizerCalculationSpec(token_col="tokens_a"),
            TokenizerCalculationSpec(token_col="tokens_b"),
        ),
    )

    actual = tokenize_name_dataframe_with_request(df, request=request)

    assert "tokens_a" in actual.columns
    assert "tokens_b" in actual.columns
    assert actual["tokens_a"].to_list()[0] == ["alpha"]
    assert actual["tokens_b"].to_list()[0] == ["alpha"]


def test_tokenize_name_with_request_writes_output(tmp_path: Path):
    cleansed_dir = tmp_path / "data" / "fr" / "cleansed"
    tokenized_dir = tmp_path / "data" / "fr" / "tokenized"
    cleansed_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({"name_cleansed": ["alpha ltd", "betacorp"]}).write_parquet(
        cleansed_dir / "fr-001.parquet"
    )

    request = TokenizeFilesRequest(
        cleansed_dir=cleansed_dir,
        tokenized_dir=tokenized_dir,
        tokenization=TokenizationRequest(
            tokenizer_calculations=(TokenizerCalculationSpec(token_col="name_tokens"),),
        ),
        source_files=(cleansed_dir / "fr-001.parquet",),
    )

    rows = tokenize_name_with_request(request)

    assert rows == 2
    written = pl.read_parquet(tokenized_dir / "fr-001.parquet")
    assert "name_tokens" in written.columns
