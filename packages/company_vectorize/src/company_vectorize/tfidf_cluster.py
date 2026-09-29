from __future__ import annotations

import polars as pl


def build_clustering_text_view(
    frame: pl.DataFrame,
    *,
    name_col: str = "name",
    token_col: str | None = None,
    text_view: str = "name",
    include_cluster_text: bool = True,
) -> pl.DataFrame:
    if text_view not in {"name", "tokens"}:
        raise ValueError("text_view must be one of: name, tokens")

    if name_col not in frame.columns:
        raise ValueError(f"DataFrame does not contain required column '{name_col}'.")

    with_text = frame.with_columns(
        pl.col(name_col)
        .cast(pl.Utf8, strict=False)
        .fill_null("")
        .str.strip_chars()
        .alias("_name_text")
    )
    if text_view == "name":
        return with_text.with_columns(pl.col("_name_text").alias("cluster_text")).drop(
            "_name_text"
        )

    if token_col is None or token_col not in frame.columns:
        raise ValueError(
            "token_col must be provided and present when text_view is 'tokens'."
        )

    with_tokens = with_text.with_columns(
        pl.col(token_col)
        .cast(pl.List(pl.Utf8), strict=False)
        .fill_null([])
        .alias("_cluster_tokens")
    )
    token_output = with_tokens.with_columns(
        pl.col("_cluster_tokens").alias("cluster_tokens"),
    )
    if include_cluster_text:
        token_output = token_output.with_columns(
            pl.col("_cluster_tokens").list.join(" ").alias("cluster_text"),
        )
    return token_output.drop(["_name_text", "_cluster_tokens"])
