"""Internal name-variant clustering validation over the GLEIF population.

GLEIF's own `previous_names`/alternative-name chain gives a same-entity
self-join population: every entity with 2+ name-variant rows is a
data-model-asserted known-true cluster. A regenerated sidecar gives each row
-- alias or primary -- its own unique `system_uri`, so which rows share an
entity is no longer readable from raw `system_uri` equality; it is resolved
through the shared cross-system walk instead
(`workspace.match_resolution.resolve_cross_system_match`, via
`split_alias_and_primary_rows`), scoping this module's population to entities
whose identity resolves that way. This module treats that population as
self-ground-truth for candidate-representation quality (`short_name`, for
example), complementary to and not a replacement for the cross-system
`gleif -> gb` pair-truth numbers computed elsewhere in `runner.py`.

GLEIF is the only population handled here; nothing in this module
generalizes to another system's name-variant rows without new work.

Everything here is live-compute rather than materialize: both the `cleanse`
knockout baseline and the representation under test are computed in-memory,
once per measurement pass, directly from each row's raw name text. Neither is
read from a cached `name_cleansed` column, since sidecar rows may not have
one and a cached column risks measuring cleanse-ruleset-version skew rather
than real name divergence. Nothing is written out: this module computes a
metric and returns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
from company_cleanse import CleanseConfig, cleanse_lazyframe, get_company_type_rules

from acquisition.company_type_registry import get_system_company_type_mapping
from acquisition.plan_registry import get_system_plan
from workspace.data_file_naming import COMPANION_NAMES, companion_data_file_glob
from workspace.data_layout import CANONICAL_LAYER_NAME, system_layer_dir
from workspace.layer_layout import resolve_name_files
from workspace.match_resolution import resolve_cross_system_match
from workspace.roots import WorkspaceRoots

from .contracts import POPULATION_UNIVERSE
from .runner import compute_pair_truth_eval, pair_truth_eval_row

_GLEIF_SYSTEM_CODE = "gleif"
_GLEIF_NAME_FILE_GLOB = companion_data_file_glob(
    system_code=_GLEIF_SYSTEM_CODE, family=COMPANION_NAMES
)
_GLEIF_NAME_COL = "name"
_GLEIF_CHAR_WHITELIST = r"[^a-z0-9\s!&]"
_PRIMARY_NAME_TYPE = "primary"

_STRATUM_EXCLUDED_SHARED_SUBSTRING = "excluded_shared_substring"
_STRATUM_EXCLUDED_KNOCKOUT = "excluded_knockout"
_STRATUM_TESTABLE = "testable"


def _resolve_effective_and_tokens(system_plan: object) -> tuple[str, ...] | None:
    """Mirror `scripts/process_companies.py`'s `_resolve_effective_and_tokens`.

    Duplicated rather than imported: `tach.toml` has `scripts` depend on
    `src/validation`, not the reverse, so importing it here would be a
    backward cross-module dependency. The logic itself is a few lines of
    normalization with no independent versioning risk.
    """
    raw = getattr(system_plan, "and_tokens", None)
    if raw:
        cleaned = tuple(
            token.strip().upper() for token in raw if token and token.strip()
        )
        if cleaned:
            return cleaned
    return None


def resolve_gleif_cleanse_config() -> CleanseConfig:
    """Build the same `CleanseConfig` GLEIF's production cleanse stage
    resolves (mirrors `scripts/pipeline_runner.py`'s `execute_cleanse`
    parameter resolution for `system_code="gleif"`), so this item's live
    `cleanse` baseline matches the accepted canonical-form baseline rather
    than an arbitrary default. That resolution no longer narrows company-type
    rules by system code -- `get_company_type_rules()`'s full multi-jurisdiction
    default is what recognizes a suffix regardless of which country it's from;
    `"gleif"` is a system code, not a jurisdiction, and was never a valid
    argument to the jurisdiction-scoped selector.
    """
    company_type_regex, company_type_mapping = get_company_type_rules()
    system_plan = get_system_plan(_GLEIF_SYSTEM_CODE)
    and_tokens = _resolve_effective_and_tokens(system_plan)
    source_company_type_mapping = get_system_company_type_mapping(system_plan.code)

    config_kwargs: dict[str, Any] = {
        "company_col": _GLEIF_NAME_COL,
        "and_tokens": and_tokens,
        "char_whitelist": _GLEIF_CHAR_WHITELIST,
        "company_type_regex": company_type_regex,
        "company_type_mapping": company_type_mapping,
        "source_company_type_col": system_plan.company_type_column,
        "source_company_type_mapping": source_company_type_mapping,
    }
    personal_owner_markers = system_plan.personal_owner_markers
    if personal_owner_markers is not None:
        config_kwargs["personal_owner_markers"] = tuple(personal_owner_markers)

    return CleanseConfig(**config_kwargs)


def _latest_gleif_canonical_dir(*, roots: WorkspaceRoots) -> Path:
    canonical_root = system_layer_dir(
        roots, _GLEIF_SYSTEM_CODE, layer=CANONICAL_LAYER_NAME
    )
    if not canonical_root.exists():
        raise FileNotFoundError(
            f"No GLEIF canonical directory found at {canonical_root}"
        )
    snapshot_dirs = sorted(p for p in canonical_root.iterdir() if p.is_dir())
    if not snapshot_dirs:
        raise FileNotFoundError(
            f"No GLEIF canonical snapshot directories found under {canonical_root}"
        )
    return snapshot_dirs[-1]


def load_gleif_name_variant_rows(*, roots: WorkspaceRoots) -> pl.DataFrame:
    """Read every `gleif-names-*.parquet` companion file from the
    latest GLEIF canonical snapshot -- one row per name-variant (including
    the `primary` legal-name row), not per entity. These rows are excluded
    from tokenizer training corpora already (`sample_training_corpus`'s
    `is_primary_data_file` filter), so they are safe to use as held-out
    validation data as-is.
    """
    snapshot_dir = _latest_gleif_canonical_dir(roots=roots)
    files = resolve_name_files(snapshot_dir, system_code=_GLEIF_SYSTEM_CODE)
    if not files:
        raise FileNotFoundError(
            f"No {_GLEIF_NAME_FILE_GLOB} files found under {snapshot_dir}"
        )
    frames = [
        pl.read_parquet(file_path, columns=["system_uri", "name", "name_type"])
        for file_path in files
    ]
    return pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]


def resolve_gleif_entity_identities(
    system_uris: pl.Series, *, roots: WorkspaceRoots
) -> pl.DataFrame:
    """Resolve every distinct `system_uri` to its entity identity through
    the shared cross-system walk (`workspace.match_resolution
    .resolve_cross_system_match`), rather than assuming an alias row and its
    primary share one raw `system_uri`. A regenerated sidecar gives every
    row -- alias or primary -- its own unique `system_uri`, so that raw
    equality no longer links a variant to its entity; the walk does, via the
    alias row's `source_uri` back-reference.

    Deduplicates `system_uris` before the walk, since each
    `resolve_cross_system_match` call re-reads the relevant sidecar/`matched/`
    layer(s) from disk rather than being backed by an in-memory frame here.
    A row whose entity carries no recorded cross-system match resolves to
    `None`; the caller drops such rows before pairing, since two different
    unmatched entities would otherwise collide on that same `None` value.
    """
    distinct_uris = system_uris.unique().to_list()
    identity_by_uri = {
        uri: resolve_cross_system_match(roots, uri=uri) for uri in distinct_uris
    }
    return pl.DataFrame(
        {
            "system_uri": list(identity_by_uri.keys()),
            "entity_identity": list(identity_by_uri.values()),
        },
        schema={"system_uri": pl.Utf8, "entity_identity": pl.Utf8},
    )


def split_alias_and_primary_rows(
    name_rows: pl.DataFrame, *, roots: WorkspaceRoots
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split raw name-variant rows into alias rows (`source`, one per
    variant) and primary rows (`target`, one per entity), scoped to the
    self-contained population this module validates against: entities with at
    least one recorded alias/variant, and whose identity resolves through the
    shared walk (`resolve_gleif_entity_identities`) -- rows whose entity
    carries no recorded cross-system match are dropped rather than linked by
    the raw `system_uri` a regenerated sidecar no longer shares between an
    alias and its primary. `variant_id` is each alias row's own real,
    already-unique `system_uri`, not a synthetic identifier built here.
    """
    identity_lookup = resolve_gleif_entity_identities(
        name_rows.get_column("system_uri"), roots=roots
    )
    name_rows = name_rows.join(identity_lookup, on="system_uri", how="left").filter(
        pl.col("entity_identity").is_not_null()
    )

    primary_rows = (
        name_rows.filter(pl.col("name_type") == _PRIMARY_NAME_TYPE)
        .unique(subset=["entity_identity"], keep="first")
        .rename({"entity_identity": "target_id", "name": "target_name_raw"})
        .select("target_id", "target_name_raw")
    )

    alias_rows = (
        name_rows.filter(pl.col("name_type") != _PRIMARY_NAME_TYPE)
        .rename(
            {
                "system_uri": "variant_id",
                "entity_identity": "primary_system_uri",
                "name": "alias_name_raw",
            }
        )
        .select("variant_id", "primary_system_uri", "name_type", "alias_name_raw")
    )

    # Scope the primary/target population to entities actually referenced by
    # at least one alias row -- entities with no recorded variant at all
    # aren't part of this self-join population and would otherwise dominate
    # the target frame for no benefit (GLEIF's real population is mostly
    # single-name entities).
    referenced_primary_ids = alias_rows.select(
        pl.col("primary_system_uri").alias("target_id")
    ).unique()
    primary_rows = primary_rows.join(
        referenced_primary_ids, on="target_id", how="inner"
    )

    return alias_rows, primary_rows


def compute_live_name_representations(
    *,
    alias_rows: pl.DataFrame,
    primary_rows: pl.DataFrame,
    cleanse_config: CleanseConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Live-compute `name_cleansed`/`short_name` for both alias and primary
    raw names in a single `cleanse_lazyframe` pass, then split the result
    back onto the alias/primary frames. Running both sides through one call
    rules out ever comparing
    two differently-configured or differently-timestamped cleanse outputs,
    on top of never reading a cached `name_cleansed` column at all.
    """
    # cleanse_lazyframe drops any leading-underscore input column as an
    # internal work column, so the bookkeeping id column below must not
    # start with "_" or it would be silently dropped from the output.
    row_id_col = "gnv_row_id"
    alias_input = alias_rows.select(
        pl.col("variant_id").alias(row_id_col),
        pl.col("alias_name_raw").alias(_GLEIF_NAME_COL),
    )
    primary_input = primary_rows.select(
        pl.col("target_id").alias(row_id_col),
        pl.col("target_name_raw").alias(_GLEIF_NAME_COL),
    )
    combined = pl.concat([alias_input, primary_input], how="vertical_relaxed")

    cleansed = cleanse_lazyframe(
        combined.lazy(), cleanse_config, file_columns=combined.columns
    ).collect()
    representations = cleansed.select(row_id_col, "name_cleansed", "short_name")

    alias_reps = alias_rows.join(
        representations.rename({row_id_col: "variant_id"}), on="variant_id", how="left"
    )
    primary_reps = primary_rows.join(
        representations.rename({row_id_col: "target_id"}), on="target_id", how="left"
    )
    return alias_reps, primary_reps


def _tokenize_for_overlap(name: pl.Expr) -> pl.Expr:
    """Lowercased whitespace-run tokenization for the shared-substring
    (token-overlap) exclusion pass, mirroring
    `scripts/measure_initialism_recall.py`'s `_tokenize`/zero-overlap
    precedent (duplicated rather than imported for the same `tach.toml`
    directionality reason as `_resolve_effective_and_tokens` above).
    """
    return name.fill_null("").str.to_lowercase().str.extract_all(r"\S+")


def build_variant_pair_frame(
    alias_reps: pl.DataFrame, primary_reps: pl.DataFrame
) -> pl.DataFrame:
    """Join alias rows to their known-true primary row and apply the two
    independent stratification/exclusion passes, in order: (1) shared-token
    presence -- a pair with no shared token at all isn't solvable by any
    string-based representation and would otherwise drag the reported
    collapse rate down for reasons unrelated to representation quality; (2)
    knockout on live-computed `cleanse` -- pairs already byte-identical
    after canonicalization have no real variation left to test.
    """
    pairs = alias_reps.join(
        primary_reps.rename(
            {
                "target_id": "primary_system_uri",
                "target_name_raw": "primary_name_raw",
                "name_cleansed": "primary_name_cleansed",
                "short_name": "primary_short_name",
            }
        ),
        on="primary_system_uri",
        how="inner",
    ).rename({"name_cleansed": "alias_name_cleansed", "short_name": "alias_short_name"})

    pairs = pairs.with_columns(
        _tokenize_for_overlap(pl.col("alias_name_raw")).alias("_alias_tokens"),
        _tokenize_for_overlap(pl.col("primary_name_raw")).alias("_primary_tokens"),
    ).with_columns(
        (
            pl.col("_alias_tokens")
            .list.set_intersection(pl.col("_primary_tokens"))
            .list.len()
            > 0
        ).alias("_has_shared_token"),
        (
            pl.col("alias_name_cleansed").fill_null("")
            == pl.col("primary_name_cleansed").fill_null("")
        ).alias("_is_knockout_identical"),
    )

    pairs = pairs.with_columns(
        pl.when(~pl.col("_has_shared_token"))
        .then(pl.lit(_STRATUM_EXCLUDED_SHARED_SUBSTRING))
        .when(pl.col("_is_knockout_identical"))
        .then(pl.lit(_STRATUM_EXCLUDED_KNOCKOUT))
        .otherwise(pl.lit(_STRATUM_TESTABLE))
        .alias("stratum")
    ).drop(
        "_alias_tokens",
        "_primary_tokens",
        "_has_shared_token",
        "_is_knockout_identical",
    )

    return pairs


@dataclass(frozen=True)
class GleifNameVariantCollapseResult:
    """Stratified collapse-rate result for one representation.

    Measured against the GLEIF name-variant self-join population. `recall` on
    `pair_truth_eval`'s universe row is the collapse rate: the fraction of
    testable variant pairs whose representation value collapses the alias into
    a candidate block containing its own known-true primary.
    `precision`/`reduction_ratio` describe how over-broad those blocks are
    within the scoped self-join population rather than the full GLEIF corpus
    (see `split_alias_and_primary_rows`), a secondary signal alongside the
    collapse rate itself.
    """

    representation: str
    total_variant_pairs: int
    excluded_shared_substring: int
    excluded_knockout: int
    testable_pairs: int
    target_rows: int
    pair_truth_eval: pl.DataFrame

    @property
    def collapse_rate(self) -> float | None:
        if self.pair_truth_eval.height == 0:
            return None
        value = pair_truth_eval_row(self.pair_truth_eval, POPULATION_UNIVERSE)["recall"]
        return float(value) if isinstance(value, (int, float)) else None


def compute_gleif_name_variant_collapse(
    *,
    roots: WorkspaceRoots,
    representation: str = "short_name",
) -> GleifNameVariantCollapseResult:
    """Compute the stratified collapse-rate metric for `representation`.

    `representation` is a column produced by `compute_live_name_representations`.

    Reuses `compute_pair_truth_eval` (`runner.py`) rather than a bespoke
    scoring harness: `source_truth_map` is the testable-stratum alias rows
    (`source_id` = the alias row's own `variant_id`, `source_match_uri` = the
    alias's own known-true primary `target_id`, both resolved through the
    shared walk by `split_alias_and_primary_rows`); `matched_edges` is a
    self-join of
    testable alias rows against the full (referenced) primary population on
    the representation value -- a real candidate-block equality check, not
    a same-pair-only lookup, so `compute_pair_truth_eval`'s precision/recall
    arithmetic reflects real block over-collision, not just per-pair recall.
    """
    if representation not in {"short_name", "name_cleansed"}:
        raise ValueError(
            f"Unsupported representation column '{representation}'; expected one of "
            "'short_name', 'name_cleansed' (the columns compute_live_name_representations "
            "produces)."
        )

    name_rows = load_gleif_name_variant_rows(roots=roots)
    alias_rows, primary_rows = split_alias_and_primary_rows(name_rows, roots=roots)
    cleanse_config = resolve_gleif_cleanse_config()
    alias_reps, primary_reps = compute_live_name_representations(
        alias_rows=alias_rows, primary_rows=primary_rows, cleanse_config=cleanse_config
    )

    pairs = build_variant_pair_frame(alias_reps, primary_reps)
    total_variant_pairs = int(pairs.height)
    excluded_shared_substring = int(
        pairs.filter(pl.col("stratum") == _STRATUM_EXCLUDED_SHARED_SUBSTRING).height
    )
    excluded_knockout = int(
        pairs.filter(pl.col("stratum") == _STRATUM_EXCLUDED_KNOCKOUT).height
    )
    testable_pairs_frame = pairs.filter(pl.col("stratum") == _STRATUM_TESTABLE)
    testable_pairs = int(testable_pairs_frame.height)

    testable_variant_ids = testable_pairs_frame.select("variant_id")
    source_truth_map = testable_pairs_frame.select(
        pl.col("variant_id").alias("source_id"),
        pl.col("primary_system_uri").alias("source_match_uri"),
    )

    alias_repr_lookup = (
        alias_reps.join(testable_variant_ids, on="variant_id", how="inner")
        .select(
            pl.col("variant_id").alias("source_id"),
            pl.col(representation).alias("_repr_value"),
        )
        .filter(
            pl.col("_repr_value").is_not_null()
            & (pl.col("_repr_value").str.strip_chars() != "")
        )
    )
    target_repr_lookup = primary_reps.select(
        "target_id",
        pl.col(representation).alias("_repr_value"),
    ).filter(
        pl.col("_repr_value").is_not_null()
        & (pl.col("_repr_value").str.strip_chars() != "")
    )

    matched_edges = alias_repr_lookup.join(
        target_repr_lookup, on="_repr_value", how="inner"
    ).select("source_id", "target_id")

    pair_truth_eval = compute_pair_truth_eval(
        source_truth_map=source_truth_map,
        matched_edges=matched_edges,
        source_system="gleif_name_variant_alias",
        target_system="gleif_name_variant_primary",
        country="__all__",
        target_rows=int(primary_reps.height),
    )

    return GleifNameVariantCollapseResult(
        representation=representation,
        total_variant_pairs=total_variant_pairs,
        excluded_shared_substring=excluded_shared_substring,
        excluded_knockout=excluded_knockout,
        testable_pairs=testable_pairs,
        target_rows=int(primary_reps.height),
        pair_truth_eval=pair_truth_eval,
    )
