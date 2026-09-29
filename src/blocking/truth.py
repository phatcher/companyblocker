"""Source-side ground truth: which target row a source row is equal to.

`validation.runner.compute_pair_truth_eval` scores candidates against one
map, `source_id` to `source_match_uri`. Where that map comes from depends on
what the source is. An entity row loaded from `data/<system>/matched/`
carries its cross-system match as `match_uri`, written by the Match stage,
and a derived row (`name://`, `perturbed://`) reaches that same match by
walking its URI chain back to the entity it stands in for. A perturbed row
validated against the very system it was made from, or a recorded name
variant scored against its own system's primary records, has no
cross-system match to find at all: its truth is a column on the row itself,
`source_uri`, the identity it was derived from.

Each resolver here is one such rule, and a run carries exactly one of them
(`contracts.BlockingRunConfig.truth`). `MatchedLayerTruth` is the default
and the cached-column-plus-walk behaviour every run had before the rule was
named. `ColumnTruth` reads a named column directly. A resolver answers every
question the run asks about ground truth -- which layers can supply it,
whether a resolved layer carries it, whether the source/target pairing is
valid, whether a loaded frame can be evaluated, and the map itself -- so a
new rule is one more class rather than a branch at each of those sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import polars as pl

from workspace.derived_uri import parse_uri
from workspace.match_resolution import resolve_cross_system_match
from workspace.reference import InvalidReferenceError
from workspace.roots import WorkspaceRoots

MATCH_URI_COLUMN = "match_uri"

SOURCE_TRUTH_MAP_SCHEMA: dict[str, type[pl.DataType]] = {
    "source_id": pl.Utf8,
    "source_match_uri": pl.Utf8,
}

_MATCHED_LAYER = "matched"
_CANONICAL_LAYER = "canonical"


@runtime_checkable
class SourceTruthResolver(Protocol):
    """One rule for reading a source row's ground truth.

    Implemented as a frozen dataclass so `contracts.build_blocking_run_identity`
    can walk its fields into the run key: two runs differing only in how they
    read truth are not the same run.
    """

    @property
    def kind(self) -> str:
        """Stable name for the rule, part of the run identity."""
        ...

    @property
    def truth_column(self) -> str:
        """The column a resolved layer must carry for this rule to read
        truth from it, named in a refusal so a reader knows what was
        missing without decoding the rule's own repr."""
        ...

    def layers(self, *, require_ground_truth: bool) -> tuple[str, ...]:
        """Which layers, in preference order, `loader.load_dataset_descriptor`
        may resolve a source from under this rule."""
        ...

    def layer_has_ground_truth(self, *, layer: str, layer_dir: Path) -> bool:
        """Whether a resolved layer can supply truth under this rule."""
        ...

    def validate_pairing(
        self,
        *,
        source_system: str,
        target_system: str,
        source_has_ground_truth: bool,
        matched_target_systems: tuple[str, ...],
    ) -> None:
        """Refuse a source/target pairing this rule cannot score correctly."""
        ...

    def can_evaluate(
        self, source_frame: pl.DataFrame, *, source_has_ground_truth: bool
    ) -> bool:
        """Whether one loaded source frame can be scored under this rule."""
        ...

    def resolve(
        self,
        source_frame: pl.DataFrame,
        *,
        roots: WorkspaceRoots,
        source_system: str,
        target_system: str,
    ) -> pl.DataFrame:
        """The `source_id`/`source_match_uri` map for one loaded source frame."""
        ...


def _empty_truth_map() -> pl.DataFrame:
    return pl.DataFrame(schema=SOURCE_TRUTH_MAP_SCHEMA)


def _system_of(uri: str) -> str | None:
    """The system a row's URI belongs to, or None where it is no URI."""
    try:
        return parse_uri(uri).system
    except InvalidReferenceError:
        return None


def _is_derived_row(uri: str) -> bool:
    """Whether a source id is a name variant or a perturbed row, whose match is
    its entity's; anything else, an id with no scheme included, is looked up as
    it is."""
    try:
        return parse_uri(uri).derived
    except InvalidReferenceError:
        return False


@dataclass(slots=True, frozen=True)
class MatchedLayerTruth:
    """Ground truth is the source system's own recorded cross-system match.

    An entity row is answered straight off the loaded frame's own
    `match_uri` column: `data/<system>/matched/` holds one row per entity
    keyed by that entity's own `system_uri`, so the column already carries
    the answer `workspace.match_resolution.resolve_cross_system_match` would
    return for it, and re-reading the layer per row would cost one full
    parquet read per source row for no different an answer. A row whose
    scheme is not the system's own -- a name-variant or perturbed row -- is
    walked back to its entity one hop at a time instead, so a derived source
    still scores through the shared mechanism rather than reading `None`.

    Only `matched/` carries this truth, and the pairing is valid only when
    the target is one the layer's `_match_metadata.json` says its
    `match_uri` values were joined against.
    """

    @property
    def kind(self) -> str:
        return "matched_layer"

    @property
    def truth_column(self) -> str:
        return MATCH_URI_COLUMN

    def layers(self, *, require_ground_truth: bool) -> tuple[str, ...]:
        # A side that needs truth is a source and reads the Match stage's
        # layer; a side that does not is a target, which is never matched,
        # so it reads its canonical snapshot. Neither reads a cleansed
        # layer: the run derives every name form from the raw `name`
        # itself, and a carried form is one it could pick up by mistake.
        if require_ground_truth:
            return (_MATCHED_LAYER,)
        return (_CANONICAL_LAYER,)

    def layer_has_ground_truth(self, *, layer: str, layer_dir: Path) -> bool:
        return layer == _MATCHED_LAYER

    def validate_pairing(
        self,
        *,
        source_system: str,
        target_system: str,
        source_has_ground_truth: bool,
        matched_target_systems: tuple[str, ...],
    ) -> None:
        if not source_has_ground_truth:
            return
        if target_system not in matched_target_systems:
            raise ValueError(
                f"source '{source_system}' matched/ was joined against "
                f"{matched_target_systems!r}, not target '{target_system}' -- "
                "pair_truth_eval would score against unrelated match_uri values"
            )

    def can_evaluate(
        self, source_frame: pl.DataFrame, *, source_has_ground_truth: bool
    ) -> bool:
        return source_has_ground_truth

    def resolve(
        self,
        source_frame: pl.DataFrame,
        *,
        roots: WorkspaceRoots,
        source_system: str,
        target_system: str,
    ) -> pl.DataFrame:
        if source_frame.height == 0:
            return _empty_truth_map()

        source_ids = source_frame.get_column("system_uri").cast(pl.Utf8).to_list()
        cached_match_uris: dict[str, str | None] | None = None
        if MATCH_URI_COLUMN in source_frame.columns:
            cached_match_uris = dict(
                zip(
                    source_ids,
                    source_frame.get_column(MATCH_URI_COLUMN)
                    .cast(pl.Utf8, strict=False)
                    .to_list(),
                    strict=True,
                )
            )

        resolved: list[str | None] = []
        for source_id in source_ids:
            # Only a derived row takes the walk; an entity's own id, or one
            # with no scheme at all, takes the cached path.
            if _is_derived_row(source_id):
                resolved.append(resolve_cross_system_match(roots, uri=source_id))
            else:
                resolved.append(
                    cached_match_uris.get(source_id)
                    if cached_match_uris is not None
                    else None
                )

        return pl.DataFrame(
            {"source_id": source_ids, "source_match_uri": resolved},
            schema=SOURCE_TRUTH_MAP_SCHEMA,
        )


@dataclass(slots=True, frozen=True)
class ColumnTruth:
    """Ground truth is a named column on the source row itself.

    The case for a perturbed dataset validated against the system it was
    made from, and for a recorded name variant scored against its own
    system's primary records: each row's `source_uri` is the identity it was
    derived from, which is the target it should recover. No walk, no
    `matched/` layer, no match metadata: the column is read as it is.

    A source resolves from `matched/` or canonical, whichever carries the
    column. The pairing is checked when the frame is resolved rather than
    from metadata: a populated value whose scheme is not the target system's
    names a row in some other system, so a frame where no populated value
    carries the target's scheme is refused rather than scored to zero.
    """

    column: str

    @property
    def kind(self) -> str:
        return "column"

    @property
    def truth_column(self) -> str:
        return self.column

    def layers(self, *, require_ground_truth: bool) -> tuple[str, ...]:
        # Either layer can carry the column; a source is looked for in the
        # Match stage's layer first, and a target reads canonical.
        if require_ground_truth:
            return (_MATCHED_LAYER, _CANONICAL_LAYER)
        return (_CANONICAL_LAYER,)

    def layer_has_ground_truth(self, *, layer: str, layer_dir: Path) -> bool:
        sample = next(layer_dir.rglob("*.parquet"), None)
        if sample is None:
            return False
        return self.column in pl.read_parquet_schema(sample)

    def validate_pairing(
        self,
        *,
        source_system: str,
        target_system: str,
        source_has_ground_truth: bool,
        matched_target_systems: tuple[str, ...],
    ) -> None:
        return None

    def can_evaluate(
        self, source_frame: pl.DataFrame, *, source_has_ground_truth: bool
    ) -> bool:
        return self.column in source_frame.columns

    def resolve(
        self,
        source_frame: pl.DataFrame,
        *,
        roots: WorkspaceRoots,
        source_system: str,
        target_system: str,
    ) -> pl.DataFrame:
        if source_frame.height == 0:
            return _empty_truth_map()
        if self.column not in source_frame.columns:
            raise ValueError(
                f"source frame carries no {self.column!r} column to read ground "
                f"truth from; columns are {source_frame.columns!r}"
            )

        truth_map = source_frame.select(
            pl.col("system_uri").cast(pl.Utf8).alias("source_id"),
            pl.col(self.column).cast(pl.Utf8, strict=False).alias("source_match_uri"),
        )

        schemes = {
            scheme
            for value in truth_map.get_column("source_match_uri").drop_nulls().to_list()
            if (scheme := _system_of(value)) is not None
        }
        if schemes and target_system not in schemes:
            raise ValueError(
                f"source column {self.column!r} names rows in "
                f"{sorted(schemes)!r}, not in target '{target_system}' -- "
                "pair_truth_eval would score against rows of another system"
            )
        return truth_map


def source_truth_for_column(column: str) -> SourceTruthResolver:
    """The resolver a `--match-col` value selects.

    `match_uri`, the default, is the matched layer's own cross-system match
    and takes the walk for derived rows; any other name is read as a plain
    column on the source row.
    """
    normalized = column.strip()
    if not normalized:
        raise ValueError("match column must be non-empty")
    if normalized == MATCH_URI_COLUMN:
        return MatchedLayerTruth()
    return ColumnTruth(normalized)


__all__ = [
    "MATCH_URI_COLUMN",
    "SOURCE_TRUTH_MAP_SCHEMA",
    "ColumnTruth",
    "MatchedLayerTruth",
    "SourceTruthResolver",
    "source_truth_for_column",
]
