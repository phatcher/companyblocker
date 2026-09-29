"""Polars-native multi-column composition into a single blocking-text column.

Every strategy in this package (`tfidf_strategy.py`, `sbert_strategy.py`,
`token_list_strategy.py`, ...) scores against one `text_col`. When the source
frame carries the signal across several columns (name, address, registered
number, ...), something upstream has to compose them into that one column
before a strategy sees it. `compose_blocking_text()` is that composition
step, kept deterministic on purpose: it never looks at row order (every row
is transformed independently) and never reaches for global or cached state
(unlike `tfidf_cluster.py`'s noise-word lookup), so the same frame composed
twice, in any process, produces byte-identical text every time.

Column order is the caller's `columns` argument, not the frame's own column
order or a set (which Python does not guarantee an iteration order for) --
composing `["name", "city"]` never equals composing `["city", "name"]`, and
that is deliberate: the caller's order is part of what makes the output
reproducible against a spec rather than against whatever order a dict or a
`DataFrame.columns` happened to list them in.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

DEFAULT_BLOCKING_TEXT_SEPARATOR = " "


def compose_blocking_text(
    frame: pl.DataFrame,
    *,
    columns: Sequence[str],
    output_col: str = "blocking_text",
    separator: str = DEFAULT_BLOCKING_TEXT_SEPARATOR,
    lowercase: bool = True,
) -> pl.DataFrame:
    """Compose `columns` into one deterministic text column named `output_col`.

    Each source column is cast to `Utf8` (non-strict, so a numeric or other
    non-string column composes rather than raising), a null becomes `""`,
    and each value is whitespace-stripped before joining -- the same
    per-value treatment `tfidf_strategy.py`'s single-column path already
    applies, extended across columns rather than repeated ad hoc by every
    caller that needs more than one. Values are joined with `separator`, the
    empty contributions from missing/blank columns collapse rather than
    leaving stray runs of separators, and the composed result is
    whitespace-stripped once more so a row where every column was blank
    composes to `""`, not to a string of bare separators.

    `lowercase=True` (the default) lowercases the composed text, matching
    the case-folding every strategy that consumes `text_col` already does
    internally (`TfidfVectorizer(lowercase=True)`, `sbert`'s normalized
    embeddings); set it `False` to keep the source casing when a caller
    needs it, e.g. to preserve acronyms for a downstream display column
    rather than for scoring.

    Raises `ValueError` if `columns` is empty or any name in it is not a
    column of `frame`; raises before touching `frame` so a typo never
    silently composes fewer columns than the caller asked for.
    """
    if not columns:
        raise ValueError("columns must be non-empty to compose blocking text")

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required column(s): {', '.join(missing)}")

    parts: list[pl.Expr] = []
    for index, column in enumerate(columns):
        part = (
            pl.col(column).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
        )
        if index > 0:
            part = (
                pl.when(part == "")
                .then(pl.lit(""))
                .otherwise(pl.concat_str(pl.lit(separator), part))
            )
        parts.append(part)

    composed = pl.concat_str(parts).str.strip_chars()
    if lowercase:
        composed = composed.str.to_lowercase()

    return frame.with_columns(composed.alias(output_col))
