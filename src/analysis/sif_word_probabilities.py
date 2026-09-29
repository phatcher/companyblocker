"""Word probabilities SIF weights a name by, counted over a stated list.

SIF weights each word of a name by how common that word is, and which names
that frequency is counted over -- a checkpoint's general-language counts, one
system's own names, or a `global` list across systems -- weights the same
name differently. This module counts word document frequency over a stated
list, one system's names or `global` across every runnable system,
and converts it to a token probability explicitly rather than treating
document frequency as one unchanged: a token probability divides a word's
occurrences by all tokens, where document frequency divides the names
containing it by the number of names, and the two denominators differ by the
list's mean tokens per name (`attach_token_probabilities`).

The result is written through `workspace.artifact_archive`'s content-addressed
store, keyed by the list, its preparation and a digest of the exact rows
counted, so a later request for the same list and preparation over unchanged
data reuses the earlier artefact and a changed one lands at a new key rather
than silently overwriting it (see that module's docstring). The artefact
records the list, its preparation, its number of names and its mean tokens
per name, and is addressable by the file path this module resolves -- a SIF
run reads it by that path, not through this module.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from analysis.system_discovery import filter_runnable_systems
from analysis.token_zipf import NAME_TIER_COLUMNS, discover_cleansed_files
from workspace.artifact_archive import (
    artifact_is_complete,
    begin_artifact_write,
    commit_artifact_write,
    resolve_candidate_dir,
)
from workspace.artifact_layout import artifact_store_root
from workspace.data_layout import CLEANSED_LAYER_NAME, data_root, system_layer_dir
from workspace.identity import rows_digest
from workspace.roots import WorkspaceRoots

WORD_PROBABILITIES_STORE_FACET = "sif-word-probabilities"
WORD_PROBABILITIES_FILENAME = "word_probabilities.parquet"
DERIVER_VERSION = "1"
GLOBAL_LIST_NAME = "global"
DEFAULT_PREPARATION = "cleansed"
"""The `NAME_TIER_COLUMNS` key counted by default: the cleansed name, the
tier the tokenizer corpus and the run's default name transform both use."""

# Unicode letter/number, matching `analysis.token_zipf`'s own word-token
# pattern for the same reason: an ASCII-only class fragments an accented
# word at each diacritic.
_WORD_TOKEN_PATTERN = r"[\p{L}\p{N}]+"

_ID_COL = "system_uri"


def _preparation_column(preparation: str) -> str:
    """The name column `preparation` selects, one of `NAME_TIER_COLUMNS`."""
    try:
        return NAME_TIER_COLUMNS[preparation]
    except KeyError:
        allowed = ", ".join(sorted(NAME_TIER_COLUMNS))
        raise ValueError(f"preparation must be one of: {allowed}") from None


def _facet(value: str) -> str:
    """`value` as one directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "_"


def count_word_document_frequencies(
    frame: pl.DataFrame, *, name_col: str
) -> tuple[pl.DataFrame, int, int]:
    """Word-level document frequency and total token occurrences over
    `frame`'s non-null, non-blank `name_col`.

    A token is counted once per name for document frequency (how many names
    contain it at least once) and every time it appears for the total, the
    second figure `attach_token_probabilities` needs alongside
    `document_frequency` to convert it to a token probability: the two
    numerators agree in practice, since a company name rarely repeats a
    word, but a token probability's denominator is all tokens, not all
    names, and the two differ by exactly this list's mean tokens per name.

    Returns `(stats, num_names, total_tokens)`: `stats` carries one row per
    distinct token with its `document_frequency`; `num_names` is how many
    rows contributed; `total_tokens` is every word occurrence counted,
    `total_tokens / num_names` being the list's mean tokens per name.
    """
    non_empty = frame.filter(
        pl.col(name_col).is_not_null() & (pl.col(name_col).str.strip_chars() != "")
    )
    num_names = non_empty.height
    if num_names == 0:
        empty = pl.DataFrame(
            {"token": [], "document_frequency": []},
            schema={"token": pl.Utf8, "document_frequency": pl.UInt32},
        )
        return empty, 0, 0

    tokens = non_empty.select(
        pl.col(name_col)
        .str.to_lowercase()
        .str.extract_all(_WORD_TOKEN_PATTERN)
        .alias("_tokens")
    )
    total_tokens = int(tokens.get_column("_tokens").list.len().sum())
    stats = (
        tokens.select(pl.col("_tokens").list.unique().explode().alias("token"))
        .drop_nulls("token")
        .group_by("token")
        .len()
        .rename({"len": "document_frequency"})
    )
    return stats, num_names, total_tokens


def attach_token_probabilities(
    stats: pl.DataFrame, *, num_names: int, mean_tokens_per_name: float
) -> pl.DataFrame:
    """`stats` (`token`, `document_frequency`) with a `probability` column.

    `p(w) = document_frequency(w) / (num_names * mean_tokens_per_name)`: a
    document-frequency count is not substitutable for a token probability
    unchanged, since its denominator is the number of names rather than all
    tokens; `num_names * mean_tokens_per_name` is that same denominator
    restated in the units document frequency's numerator (occurrences, in
    practice, since a name rarely repeats a word) needs to divide by.

    Raises `ValueError` when `num_names` or `mean_tokens_per_name` is not
    positive, since the conversion divides by their product.
    """
    if num_names <= 0 or mean_tokens_per_name <= 0:
        raise ValueError("num_names and mean_tokens_per_name must be positive")
    denominator = num_names * mean_tokens_per_name
    return stats.with_columns(
        (pl.col("document_frequency") / denominator).alias("probability")
    )


def _system_name_frame(files: list[Path], name_col: str) -> pl.DataFrame:
    """One system's `(system_uri, name_col)` rows, read from whichever of its
    cleansed shards carry both columns -- not every source has migrated to
    the standardized name-tier columns (`analysis.token_zipf`)."""
    required = {_ID_COL, name_col}
    eligible = [
        str(path)
        for path in files
        if required.issubset(pl.read_parquet_schema(path).keys())
    ]
    if not eligible:
        return pl.DataFrame(schema={_ID_COL: pl.Utf8, name_col: pl.Utf8})
    return pl.scan_parquet(eligible).select(_ID_COL, name_col).collect()


@dataclass(frozen=True)
class _SystemCount:
    system: str
    stats: pl.DataFrame
    num_names: int
    total_tokens: int
    content_digest: str


def _count_system(
    roots: WorkspaceRoots, system: str, *, name_col: str
) -> _SystemCount | None:
    """One system's word counts under `name_col`, or `None` when it has no
    cleansed names to count."""
    files = discover_cleansed_files(roots, system)
    if not files:
        return None
    frame = _system_name_frame(files, name_col)
    if frame.height == 0:
        return None
    stats, num_names, total_tokens = count_word_document_frequencies(
        frame, name_col=name_col
    )
    if num_names == 0:
        return None
    digest = rows_digest(frame, id_col=_ID_COL, value_col=name_col)
    return _SystemCount(
        system=system,
        stats=stats,
        num_names=num_names,
        total_tokens=total_tokens,
        content_digest=digest,
    )


def _combine_counts(counts: list[_SystemCount]) -> tuple[pl.DataFrame, int, int, str]:
    """Every system's counts pooled: summed document frequencies, summed
    names and tokens, and one content digest over the exact per-system rows
    counted -- a `global` list is therefore keyed by what it actually
    counted, not by the set of system codes alone, so a system's cleansed
    data changing moves the key."""
    stats = (
        pl.concat([count.stats for count in counts])
        .group_by("token")
        .agg(pl.col("document_frequency").sum())
    )
    num_names = sum(count.num_names for count in counts)
    total_tokens = sum(count.total_tokens for count in counts)
    digest_source = "|".join(
        f"{count.system}:{count.content_digest}"
        for count in sorted(counts, key=lambda count: count.system)
    )
    content_digest = hashlib.blake2b(
        digest_source.encode("utf-8"), digest_size=16
    ).hexdigest()
    return stats, num_names, total_tokens, content_digest


def _discover_runnable_systems_with_cleansed(roots: WorkspaceRoots) -> list[str]:
    """Every system directory under `data/` holding a cleansed layer, filtered
    to catalog-runnable ones -- the same auto-discovery
    `analysis.token_zipf.run_token_zipf_analysis` applies when no systems are
    named, so the `global` list means the same population that analysis
    already treats as "every system"."""
    data_root_dir = data_root(roots)
    if not data_root_dir.exists():
        return []
    discovered = sorted(
        child.name
        for child in data_root_dir.iterdir()
        if child.is_dir()
        and system_layer_dir(roots, child.name, layer=CLEANSED_LAYER_NAME).exists()
    )
    return filter_runnable_systems(discovered)


def resolve_word_probabilities(
    roots: WorkspaceRoots,
    *,
    system: str | None,
    preparation: str = DEFAULT_PREPARATION,
) -> Path:
    """The word-probabilities artefact for one system's names, or the
    `global` list (every runnable system with a cleansed layer) when
    `system` is `None`, at word level under `preparation` (one of
    `NAME_TIER_COLUMNS`).

    Content-addressed under `artifacts/store/`
    (`workspace.artifact_archive`): a candidate already complete for this
    list, preparation and the exact rows counted is returned unchanged,
    never rewritten; once a system's cleansed data changes and is recounted,
    the new count lands at a new key rather than silently overwriting the
    old one's manifest. The manifest records the list, the systems it drew
    from, the preparation, the number of names counted and their mean
    tokens per name.

    Raises `ValueError` when `preparation` is not one of `NAME_TIER_COLUMNS`,
    or when no named system (or, for the `global` list, no runnable system)
    has names under it.
    """
    name_col = _preparation_column(preparation)
    if system is not None:
        list_name = system.strip().lower()
        candidate_systems = [list_name]
    else:
        list_name = GLOBAL_LIST_NAME
        candidate_systems = _discover_runnable_systems_with_cleansed(roots)

    counts = [
        found
        for found in (
            _count_system(roots, candidate, name_col=name_col)
            for candidate in candidate_systems
        )
        if found is not None
    ]
    if not counts:
        raise ValueError(
            f"no names found for {list_name!r} under preparation {preparation!r}"
        )

    stats, num_names, total_tokens, content_digest = _combine_counts(counts)
    mean_tokens_per_name = total_tokens / num_names

    settings = {
        "list": list_name,
        "systems": sorted(count.system for count in counts),
        "preparation": preparation,
        "deriver_version": DERIVER_VERSION,
        "content_digest": content_digest,
    }
    entry = resolve_candidate_dir(
        artifact_store_root(roots),
        WORD_PROBABILITIES_STORE_FACET,
        _facet(list_name),
        _facet(preparation),
        settings=settings,
    )
    if artifact_is_complete(entry):
        return entry

    probabilities = attach_token_probabilities(
        stats, num_names=num_names, mean_tokens_per_name=mean_tokens_per_name
    )
    manifest = {
        **settings,
        "num_names": num_names,
        "mean_tokens_per_name": mean_tokens_per_name,
    }
    staging = begin_artifact_write(entry)
    probabilities.write_parquet(staging / WORD_PROBABILITIES_FILENAME)
    commit_artifact_write(entry, staging, manifest=manifest)
    return entry


__all__ = [
    "DEFAULT_PREPARATION",
    "DERIVER_VERSION",
    "GLOBAL_LIST_NAME",
    "WORD_PROBABILITIES_FILENAME",
    "WORD_PROBABILITIES_STORE_FACET",
    "attach_token_probabilities",
    "count_word_document_frequencies",
    "resolve_word_probabilities",
]
