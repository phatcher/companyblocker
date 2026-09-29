"""Perturbation materialization contract and ownership boundary.



Owns turning `company_perturbation`'s in-memory perturbation generation

(`company_perturbation.generation`'s `perturb_records()`) into a schema-validated,

loader-readable dataset on disk.

Reuses existing machinery end to end rather than reimplementing it:



- Source rows come from `validation.loader.load_validation_snapshot` (the same function

  `runner.py`'s real validation matrix uses) -- no new I/O path for reading canonical/cleansed

  input.

- Perturbed rows are routed through the *real* cleanse pipeline

  (`company_cleanse.cleanse_lazyframe`), so

  `name_cleansed`/`name_cleansed_basic` are genuinely recomputed from the mutated name (the

  actual point of robustness testing) and extra provenance columns survive unchanged because

  `company_cleanse`'s own column-ordering step already preserves arbitrary input columns by

  construction.

- Output lands under `workspace.data_layout.perturbed_dataset_dir()`, which resolves

  through `workspace.kind_layout` -- that module owns which tree perturbed data sits in and what

  the segment is called there, so neither this module nor any reader composes the join.

  Source system, `profile_id` and seed are sub-keys beneath it, and the `cleansed/` leaf below

  that is this module's own. Both

  `validation.prepared_dataset.resolve_prepared_system_dir` and

  `scripts/validate_clustering.py`'s registry gate resolve the

  `perturbed://<system>/<profile>/<version>/<seed>` URI against the same function, so

  writer and readers cannot drift apart.



**Perturbed `system_uri` composition**: `workspace.derived_uri.perturbed_uri` composes it from the

row that was perturbed and the profile, version and seed it was perturbed with,

`perturbed://<source>/<profile_id>/<version>/<seed>/<local_id>`, so it decomposes to the value of

its own `source_uri` with no lookup and this module spells no identity of its own. A name

variant's perturbed copy carries the `names` dataset word and its hash the same way.



**`source_uri` vs `match_uri` -- two distinct columns, not one overloaded field.** An earlier

draft of this design used the host to encode the record's true origin and a single `match_uri`

column for "which original row this came from." Both were wrong once a genuinely different input

shape was considered: a *matched-set* row (e.g. a gleif row with its own real `match_uri`

pointing at its true gb counterpart) already uses `match_uri` for cross-system ground truth --

overwriting it with the perturbation lineage link would destroy that ground truth. So:



- `source_uri` (new, always non-null): the record's own real `system_uri` before perturbation --

  "this perturbed row is a variant of that row." Populated unconditionally.

- `match_uri` (passed through unchanged, null unless the source row already had one): whatever

  cross-system ground truth the original row carried, untouched by perturbation.



This also happens to generalize to name-variant/alias data: a regenerated sidecar now gives every

variant row its own unique `system_uri` and a `source_uri` back-reference to its owner entity

(`workspace.match_resolution`'s `name://` scheme), rather than sharing one `system_uri`

many-to-one across an entity's rows as it once did. At that point a name-variant row is

structurally identical to a matched-set row from this module's perspective, needing no

special-casing here -- `source_uri`/`match_uri` already mean exactly what this module expects.

Perturbing name-variant/alias data is still not wired up (nothing here calls

`load_source_canonical_records` against a names sidecar), but that is no longer blocked on the

identity shape; it is a separate, not-yet-started undertaking.



**One dataset per source system, profile version and seed.** A dataset is identified by all of them

together, since one profile over one source under two seeds is two datasets of the same size,

and it lives under `perturbed_dataset_dir()`'s `<source_system>/<profile_id>/<version>/<seed>/`. Two source

systems therefore never write into one directory, and re-materializing a dataset either reuses

its own directory or, with `force_rebuild`, replaces it.



Identity collisions are caught in two places, both before any disk write:



1. **Within a batch** (`build_perturbed_frame`): two records in the same batch composing one

   `system_uri`. The URI holds the source's own id, so this can only be a source repeating

   a record, and is defense in depth rather than the primary safeguard.

2. **Across a run's batches** (`_check_no_cross_batch_collision`): the same, for two records that

   fell in different batches. A run is written a batch at a time, so the within-batch check alone

   would let a collision through whenever the two rows were never in memory together, and the

   guarantee has to hold over the dataset rather than over a slice of it.



**Load-bearing ordering note**: the merged view this output is read back through deduplicates on

`system_uri`, silently dropping a collision rather than failing. The "hard-fail on any identity

collision" contract is satisfied *only* because both checks above run first. A refactor that

skips either, or that streams without carrying the run-level set, reintroduces the gap.



**`seed` storage note**: the seed materialized is the run's own, the caller's integer, carried

in the manifest and in every row's `system_uri`. A record's own generator seed is derived from

it inside `company_perturbation.generation` and never stored: a row is replayed from

`(profile_id, seed, original_local_id)`, which the `system_uri` already holds.



**Deviation from the original design sketch**: `MaterializationConfig` has no `prepared_base_dir`

field. `validation.loader.load_validation_snapshot` (reused here for zero new I/O code) resolves

its input directory via a path that hardcodes `prepared_base_dir=None` regardless of what's

passed to higher-level callers -- the same is true of every other real caller of that function

(`runner.py`'s `build_directional_frames`). Adding a config field that couldn't actually change

behavior would be misleading, so it was left out rather than wired to nothing.

"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import polars as pl
from company_cleanse import CleanseConfig, cleanse_lazyframe
from company_perturbation.generation import SourceRecord, perturb_records
from company_perturbation.profile_schema import (
    PerturbationProfile,
    serialize_profile,
)
from company_perturbation.sited_operator import SitedOperatorRegistry, sited_registry

from workspace.artifact_layout import validation_artifact_root
from workspace.data_layout import (
    CLEANSED_LAYER_NAME,
    layer_directory,
    perturbed_dataset_dir,
)
from workspace.derived_uri import Perturbation, parse_uri, perturbed_uri
from workspace.layer_layout import layer_partition_dir, resolve_primary_files
from workspace.reference import InvalidReferenceError
from workspace.roots import WorkspaceRoots
from workspace.run_inputs import perturbed_source_uri

from .config import resolve_prepared_rows_per_file
from .loader import load_validation_snapshot
from .perturbation_dataset_contracts import validate_perturbation_dataset_schema
from .prepared_dataset import resolve_prepared_system_dir

# polars accepts a dtype class (e.g. pl.Utf8) or an instance (e.g. pl.Utf8()) interchangeably
# in schema dicts; pl.DataType alone only covers the latter (matches contracts.py's own alias).

_PolarsDType = type[pl.DataType] | pl.DataType


_MAX_REPORTED_PARSE_ERRORS = 20

_MAX_REPORTED_COLLISIONS = 20

_CHUNK_INDEX_MANIFEST_NAME = "_chunk_index_by_source_system.json"


class SystemUriParseError(ValueError):
    pass


class PerturbationCollisionError(ValueError):
    pass


@dataclass(frozen=True)
class MaterializationConfig:
    roots: WorkspaceRoots

    source_system: str

    profile: PerturbationProfile

    profile_version: str
    """The version of the profile reference `profile` was loaded from, which
    every perturbed row records; the profile itself carries no version."""

    seed: int

    countries: tuple[str, ...] | None = None

    name_col: str = "name"

    rows_per_file: int | None = None

    force_rebuild: bool = False


@dataclass(frozen=True)
class MaterializationResult:
    profile_id: str

    system_code: str

    output_dir: Path

    manifest_path: Path

    scenario_tally: pl.DataFrame

    rows_emitted: int

    dataset_snapshot_fingerprint_by_country: dict[str, str]


@dataclass(frozen=True)
class MaterializationManifest:
    profile: dict[str, object]

    seed: int

    source_system: str

    source_dataset_snapshot_fingerprint_by_country: dict[str, str]

    output_dir: str

    scenario_tally_path: str

    rows_emitted: int

    generated_at: str


def parse_system_uri(system_uri: str) -> tuple[str, str]:
    """An entity row's system and local id, through `workspace`'s parser. Hard-fails
    for anything that is not an entity's own URI, a derived row's included."""
    try:
        parsed = parse_uri(system_uri)
    except InvalidReferenceError as error:
        raise SystemUriParseError(
            f"system_uri {system_uri!r} is no entity's URI: {error}"
        ) from error
    if parsed.derived or parsed.entity_id is None:
        raise SystemUriParseError(f"system_uri {system_uri!r} is no entity's URI.")
    return parsed.system, parsed.entity_id


@dataclass(frozen=True)
class _SourceLineage:
    source_uri: str

    match_uri: str | None

    system: str

    country: str | None


def load_source_canonical_records(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    countries: tuple[str, ...] | None,
    name_col: str,
) -> tuple[list[SourceRecord], dict[str, _SourceLineage]]:
    """Reuses `load_validation_snapshot` -- zero new I/O code. Strict-parses every `system_uri`

    via `parse_system_uri` (including a scheme mismatch against `source_system`), AGGREGATING

    all parse failures into one `SystemUriParseError` rather than failing on the first bad row.

    Returns `(records, {local_id: _SourceLineage})`: `source_uri` is the record's own real

    `system_uri` (always present); `match_uri` is whatever cross-system ground truth

    `load_validation_snapshot` surfaced for that row, if any (null for plain, unmatched

    cleansed data)."""

    frame = load_validation_snapshot(
        roots=roots, system=source_system, run_date=None, name_col=name_col
    )

    if countries:
        allowed = set(countries)

        frame = frame.filter(pl.col("country").is_in(sorted(allowed)))

    records: list[SourceRecord] = []

    lineage_by_local_id: dict[str, _SourceLineage] = {}

    parse_errors: list[str] = []

    for row in frame.select("system_uri", "match_uri", "name", "country").iter_rows(
        named=True
    ):
        system_uri = row["system_uri"]

        try:
            scheme, local_id = parse_system_uri(system_uri)

            if scheme != source_system:
                raise SystemUriParseError(
                    f"system_uri {system_uri!r} has scheme {scheme!r}, expected "
                    f"{source_system!r}"
                )

        except SystemUriParseError as exc:
            parse_errors.append(str(exc))

            continue

        lineage_by_local_id[local_id] = _SourceLineage(
            source_uri=system_uri,
            match_uri=row["match_uri"],
            system=source_system,
            country=row["country"],
        )

        records.append(
            SourceRecord(
                local_id=local_id,
                name=row["name"],
                system=source_system,
                country=row["country"],
            )
        )

    if parse_errors:
        preview = parse_errors[:_MAX_REPORTED_PARSE_ERRORS]

        suffix = (
            ""
            if len(parse_errors) <= _MAX_REPORTED_PARSE_ERRORS
            else f" (+{len(parse_errors) - _MAX_REPORTED_PARSE_ERRORS} more)"
        )

        raise SystemUriParseError(
            f"{len(parse_errors)} system_uri value(s) for source system {source_system!r} "
            "failed strict parsing:\n" + "\n".join(preview) + suffix
        )

    return records, lineage_by_local_id


_RAW_FRAME_SCHEMA: dict[str, _PolarsDType] = {
    "system_uri": pl.Utf8,
    "source_uri": pl.Utf8,
    "match_uri": pl.Utf8,
    "name": pl.Utf8,
    "jurisdiction_code": pl.Utf8,
    "profile_id": pl.Utf8,
    "profile_version": pl.Utf8,
    "scenario_id": pl.Utf8,
    "original_name": pl.Utf8,
    "changed": pl.Boolean,
}


def _records_with_lineage(
    batch: pl.DataFrame, *, source_system: str
) -> Iterator[tuple[SourceRecord, _SourceLineage]]:
    """Turn one batch of source rows into records paired with their lineage.

    Paired, not two structures rejoined by `local_id`. The dict that used to hold
    every lineage existed only because the two were split, and it silently dropped a
    record whenever a source repeated a `system_uri` -- last write won. Travelling
    together, there is nothing to key on and nothing to lose.
    """
    for row in batch.select("system_uri", "match_uri", "name", "country").iter_rows(
        named=True
    ):
        scheme, local_id = parse_system_uri(row["system_uri"])
        if scheme != source_system:
            raise SystemUriParseError(
                f"system_uri {row['system_uri']!r} has scheme {scheme!r}, expected "
                f"{source_system!r}"
            )
        yield (
            SourceRecord(
                local_id=local_id,
                name=row["name"],
                system=source_system,
                country=row["country"],
            ),
            _SourceLineage(
                source_uri=row["system_uri"],
                match_uri=row["match_uri"],
                system=source_system,
                country=row["country"],
            ),
        )


def build_perturbed_frame(
    profile: PerturbationProfile,
    records_with_lineage: Iterable[tuple[SourceRecord, _SourceLineage]],
    *,
    profile_version: str,
    seed: int,
    registry: SitedOperatorRegistry = sited_registry,
) -> pl.DataFrame:
    """One raw pre-cleanse row per source record, for one batch.

    Every record emits, including one whose drawn scenario landed no change: the
    perturbed set is joinable one-to-one against its source, and `changed` says which
    case a row is without comparing strings.
    """
    system_uris: list[str] = []
    source_uris: list[str | None] = []
    match_uris: list[str | None] = []
    countries: list[str | None] = []
    names: list[str] = []
    original_names: list[str] = []
    scenario_ids: list[str | None] = []
    changed_flags: list[bool] = []

    perturbation = Perturbation(
        profile=profile.profile_id, version=profile_version, seed=seed
    )
    pairs = list(records_with_lineage)
    lineage_by_local_id = {record.local_id: lineage for record, lineage in pairs}

    for perturbed in perturb_records(
        (record for record, _ in pairs), profile, seed=seed, registry=registry
    ):
        lineage = lineage_by_local_id[perturbed.local_id]
        system_uris.append(
            perturbed_uri(source_uri=lineage.source_uri, perturbation=perturbation)
        )
        source_uris.append(lineage.source_uri)
        match_uris.append(lineage.match_uri)
        countries.append(lineage.country)
        names.append(perturbed.name)
        original_names.append(perturbed.original_name)
        scenario_ids.append(perturbed.scenario_id)
        changed_flags.append(perturbed.changed)

    # Typed columns, not a list of per-row dicts: a dict per row costs the
    # construction plus a schema inference pass over every one of them, and a batch
    # pays it again. The values are already grouped by column here.
    frame = pl.DataFrame(
        [
            pl.Series("system_uri", system_uris, dtype=pl.Utf8),
            pl.Series("source_uri", source_uris, dtype=pl.Utf8),
            pl.Series("match_uri", match_uris, dtype=pl.Utf8),
            pl.Series("name", names, dtype=pl.Utf8),
            pl.Series("jurisdiction_code", countries, dtype=pl.Utf8),
            pl.Series("profile_id", [profile.profile_id] * len(names), dtype=pl.Utf8),
            pl.Series("profile_version", [profile_version] * len(names), dtype=pl.Utf8),
            pl.Series("scenario_id", scenario_ids, dtype=pl.Utf8),
            pl.Series("original_name", original_names, dtype=pl.Utf8),
            pl.Series("changed", changed_flags, dtype=pl.Boolean),
        ]
    )

    if frame.height:
        duplicated = frame.filter(pl.col("system_uri").is_duplicated())
        if duplicated.height:
            colliding = sorted(duplicated.get_column("system_uri").unique().to_list())
            raise PerturbationCollisionError(
                "Identity collision in materialized perturbation output: the following "
                "system_uri value(s) were produced by more than one record: "
                f"{colliding[:_MAX_REPORTED_COLLISIONS]}"
            )

    return frame


def _partition_writer(
    *, cleansed_dir: Path, rows_per_file: int
) -> Callable[[pl.DataFrame], None]:
    """Write batches into `jurisdiction_code=<c>/part-NNNN.parquet`, flushing at size.

    Deliberately not `materialize_cleansed_merge`. That reads a directory of
    per-input-file chunks and rejoins them, which needs the chunks to exist, which
    needs a staging round trip -- and it brings acquisition's own layout with it
    (`primary/`, `chunks/`, a per-source-system chunk-index sidecar so filenames do
    not collide). None of that is work: it is bookkeeping for a round trip we no
    longer make.

    `layer_layout.resolve_primary_files` falls back to a layer's own top level when
    there is no `primary/`, so a partitioned view written straight here is readable
    by every existing resolver without the family directory.
    """
    buffers: dict[str, list[pl.DataFrame]] = {}
    buffered_rows: dict[str, int] = {}
    part_index: dict[str, int] = {}

    def _flush(country: str) -> None:
        frames = buffers.get(country)
        if not frames:
            return
        partition_dir = layer_partition_dir(cleansed_dir, value=country)
        partition_dir.mkdir(parents=True, exist_ok=True)
        index = part_index.get(country, 0)
        pl.concat(frames, how="vertical_relaxed").write_parquet(
            partition_dir / f"part-{index:05d}.parquet"
        )
        part_index[country] = index + 1
        buffers[country] = []
        buffered_rows[country] = 0

    def write(frame: pl.DataFrame) -> None:
        for country, group in frame.partition_by(
            "jurisdiction_code", as_dict=True
        ).items():
            key = country[0] if isinstance(country, tuple) else country
            buffers.setdefault(key, []).append(group)
            buffered_rows[key] = buffered_rows.get(key, 0) + group.height
            if buffered_rows[key] >= rows_per_file:
                _flush(key)

    def close() -> None:
        for country in list(buffers):
            _flush(country)

    write.close = close  # type: ignore[attr-defined]
    return write


def _check_no_cross_batch_collision(*, frame: pl.DataFrame, seen: set[str]) -> None:
    """Fail if this batch repeats a `system_uri` an earlier batch of the run emitted.

    `build_perturbed_frame` guarantees uniqueness only inside the batch it was handed,
    and a run is written a batch at a time, so two records sharing an identity land
    unnoticed whenever they fall in different batches. `seen` accumulates across the
    run and is what closes that, since the guarantee has to hold over the dataset
    rather than over whichever slice of it was in memory at the time.

    Checked before the write, because the merge this output is read back through
    deduplicates on `system_uri` and would silently drop the second row rather than
    report it.
    """
    uris = frame.get_column("system_uri").to_list()
    colliding = sorted(seen.intersection(uris))
    seen.update(uris)

    if colliding:
        preview = colliding[:_MAX_REPORTED_COLLISIONS]

        raise PerturbationCollisionError(
            "Identity collision across batches of one materialization run: the "
            "following system_uri value(s) were emitted more than once: "
            f"{preview}"
        )


def _run_quality_checks(
    *,
    merged_frame: pl.DataFrame,
    source_system: str,
    valid_source_uris: set[str],
    expected_rows_emitted: int,
    profile_id: str,
    skip_source_comparison_checks: bool,
) -> None:

    null_source_uri_count = int(
        merged_frame.filter(
            pl.col("source_uri").is_null() | (pl.col("source_uri") == "")
        ).height
    )

    if null_source_uri_count:
        raise ValueError(
            f"{null_source_uri_count} materialized row(s) for profile {profile_id!r} have a "
            "null/empty source_uri"
        )

    if skip_source_comparison_checks:
        # force_rebuild=False reused stale output (a known, accepted behavior -- see the
        # module docstring's hard-fail ordering note): the merged view predates this call's
        # freshly-loaded source records, so comparing the two would reject perfectly valid,
        # already-once-validated output. Only the null-check above (an intrinsic property of
        # the output itself, independent of any particular source load) still applies.

        return

    # Scope the remaining checks to THIS call's own contribution: under multi-source-system
    # accumulation (see module docstring), merged_frame may also carry rows from OTHER source
    # systems materialized in a prior call -- those were already validated then, and comparing
    # them against this call's freshly-loaded (single-source-system) valid_source_uris would
    # reject perfectly valid, already-once-validated rows that simply belong to someone else.

    own_rows = merged_frame.filter(
        pl.col("source_uri").str.starts_with(f"{source_system}://")
    )

    actual_source_uris = set(own_rows.get_column("source_uri").to_list())

    unknown_source_uris = actual_source_uris - valid_source_uris

    if unknown_source_uris:
        preview = sorted(unknown_source_uris)[:_MAX_REPORTED_COLLISIONS]

        raise ValueError(
            f"Materialized output for profile {profile_id!r} contains "
            f"{len(unknown_source_uris)} source_uri value(s) that don't reference the source "
            f"system's own records: {preview}"
        )

    actual_rows = int(own_rows.height)

    if actual_rows != expected_rows_emitted:
        raise ValueError(
            f"Materialized row count for source system {source_system!r} ({actual_rows}) in "
            f"profile {profile_id!r} does not match this call's generation report summed "
            f"records_emitted ({expected_rows_emitted}) -- materialize_cleansed_merge's own "
            "dedupe may have silently dropped rows"
        )


def _source_snapshot_fingerprint(*, system_dir: Path, country: str) -> str:
    """Mirrors `validation.runner._prepared_country_fingerprint`'s pattern (sha256 over

    `_prepared_metadata.json` bytes + per-file name/size/mtime_ns) -- duplicated locally since

    that function is private/module-scoped in `runner.py`."""

    digest = hashlib.sha256()

    metadata_path = system_dir / "_prepared_metadata.json"

    if metadata_path.exists():
        digest.update(metadata_path.read_bytes())

    partition_dir = layer_partition_dir(system_dir, value=country)

    digest.update(country.encode("utf-8"))

    if partition_dir.exists():
        for path in sorted(partition_dir.glob("*.parquet")):
            stat = path.stat()

            digest.update(path.name.encode("utf-8"))

            digest.update(str(stat.st_size).encode("utf-8"))

            digest.update(str(stat.st_mtime_ns).encode("utf-8"))

    return digest.hexdigest()[:16]


def _source_snapshot_fingerprint_by_country(
    *, roots: WorkspaceRoots, source_system: str, countries: Iterable[str]
) -> dict[str, str]:

    system_dir = resolve_prepared_system_dir(
        roots=roots, system=source_system, prepared_base_dir=None
    )

    return {
        country: _source_snapshot_fingerprint(system_dir=system_dir, country=country)
        for country in sorted(set(countries))
    }


def _repo_relative(path: Path, *, roots: WorkspaceRoots) -> str:
    """A path as this repository stores it: relative to the checkout, POSIX-separated.

    Every worktree reaches one shared store through its own checkout, so an absolute
    path names a checkout no other has, and a backslash-separated one written
    on Windows does not read on Linux. An absolute path that is not under the checkout
    raises rather than being persisted: a manifest that resolves only on the
    machine that wrote it is worse than one that refuses to be written.
    """
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(roots.checkout).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"{path} is outside the checkout {roots.checkout} and cannot be recorded "
            "in a manifest"
        ) from exc


def build_materialization_manifest(
    *,
    roots: WorkspaceRoots,
    profile: PerturbationProfile,
    seed: int,
    source_system: str,
    source_dataset_snapshot_fingerprint_by_country: dict[str, str],
    output_dir: Path,
    scenario_tally_path: Path,
    rows_emitted: int,
    generated_at: str | None = None,
) -> MaterializationManifest:

    return MaterializationManifest(
        profile=serialize_profile(profile),
        seed=seed,
        source_system=source_system,
        source_dataset_snapshot_fingerprint_by_country=(
            source_dataset_snapshot_fingerprint_by_country
        ),
        output_dir=_repo_relative(output_dir, roots=roots),
        scenario_tally_path=_repo_relative(scenario_tally_path, roots=roots),
        rows_emitted=rows_emitted,
        generated_at=generated_at or datetime.now(UTC).isoformat(),
    )


def serialize_materialization_manifest(
    manifest: MaterializationManifest,
) -> dict[str, object]:

    return {
        "profile": manifest.profile,
        "seed": manifest.seed,
        "source_system": manifest.source_system,
        "source_dataset_snapshot_fingerprint_by_country": dict(
            sorted(manifest.source_dataset_snapshot_fingerprint_by_country.items())
        ),
        "output_dir": manifest.output_dir,
        "scenario_tally_path": manifest.scenario_tally_path,
        "rows_emitted": manifest.rows_emitted,
        "generated_at": manifest.generated_at,
    }


def write_materialization_manifest(
    manifest: MaterializationManifest, path: Path
) -> Path:
    """Atomic write, matching `run_manifest.write_run_manifest`'s own tmp+replace convention."""

    payload = serialize_materialization_manifest(manifest)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    tmp_path.replace(path)

    return path


def relative_materialization_manifest_path(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    profile_id: str,
    version: str,
    seed: int,
) -> PurePosixPath:
    """Repo-relative location of one profile's per-source-system `MaterializationManifest`.



    The single place this naming convention lives; both the absolute

    resolver below and anything persisting a pointer to the manifest go

    through here, so they cannot drift apart.



    Repo-relative and POSIX-separated because this is the form that gets

    written into an artifact. An absolute path is wrong to persist here

    twice over: every worktree has its own root while `data/` is one shared

    physical store behind a junction, so a path recorded from a worktree

    names a root no other checkout has, and a backslash-separated path

    written on Windows does not read on Linux. Compose it against a real

    checkout with `resolve_materialization_manifest_path` at the moment the

    filesystem is actually touched, never before.

    """

    profile_dir = perturbed_dataset_dir(
        roots=roots,
        source_system=source_system,
        profile_id=profile_id,
        version=version,
        seed=seed,
    )

    return PurePosixPath(profile_dir.relative_to(roots.checkout).as_posix()) / (
        f"_materialization_manifest_{source_system}.json"
    )


def resolve_materialization_manifest_path(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    profile_id: str,
    version: str,
    seed: int,
) -> Path:
    """Absolute path to one profile's per-source-system `MaterializationManifest`.



    The checkout composed with `relative_materialization_manifest_path`. Both

    `materialize_perturbations`' write path and `load_materialization_

    manifest`'s read path resolve through here, so they can never drift

    apart from each other. Use this to touch the file; use the relative

    form to record where it is.

    """

    return roots.checkout / relative_materialization_manifest_path(
        roots=roots,
        source_system=source_system,
        profile_id=profile_id,
        version=version,
        seed=seed,
    )


def resolve_scenario_tally_path(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    profile_id: str,
    version: str,
    seed: int,
) -> Path:
    """Absolute path to one profile's per-source-system generation report.



    Resolved through `workspace.artifact_layout.validation_artifact_root` so

    the `artifacts/validation/` root stays this repository's one shared

    convention; `perturbed/<profile_id>/` beneath it is this module's own

    subtree. `materialize_perturbations`'s write path goes through here.

    """

    return (
        validation_artifact_root(roots)
        / "perturbed"
        / profile_id
        / version
        / str(seed)
        / f"scenario_tally_{source_system}.parquet"
    )


def load_materialization_manifest(
    *,
    roots: WorkspaceRoots,
    source_system: str,
    profile_id: str,
    version: str,
    seed: int,
) -> dict[str, object] | None:
    """Load a profile's per-source-system `MaterializationManifest`, or `None`.



    Returns the raw dict `serialize_materialization_manifest` wrote (a

    resolved `RunManifest` under `"run_manifest"` plus

    `source_dataset_snapshot_fingerprint_by_country`), not a reconstructed

    `MaterializationManifest` dataclass -- a caller wiring this into a

    `robustness_eval` output's `dataset_snapshot_manifest_path` pointer

    needs the on-disk shape to inspect, not a round-tripped object.



    Absence is a normal caller state (this `(profile_id, source_system)`

    hasn't been materialized yet), not an error -- mirrors `src.analysis.

    run_manifest.load_run_manifest`'s own "absence is normal" contract.

    """

    path = resolve_materialization_manifest_path(
        roots=roots,
        profile_id=profile_id,
        source_system=source_system,
        version=version,
        seed=seed,
    )

    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


ProgressCallback = Callable[[dict[str, object]], None]


def materialize_perturbations(
    config: MaterializationConfig,
    *,
    progress_callback: ProgressCallback | None = None,
    batch_size: int = 50_000,
) -> MaterializationResult:
    """One pass: read a batch of source rows, perturb, cleanse in flight, write.

    Nothing holds the corpus. The previous shape materialised a list of every
    source record, a dict of every lineage, a list of every output row and then a
    frame, staged that frame to disk, read it back through `name_cleanse`, and
    merged the chunks -- four copies and three passes to produce one dataset. Here
    a batch is the largest thing alive at once, the first partition lands seconds
    in, and progress reports rows actually written rather than a tick against a
    total.

    Uniqueness needs no whole-corpus structure either: `canonical` already dedupes
    on `system_uri` upstream, and there is exactly one perturbed row per source
    record, so the output is unique by construction rather than by a check.
    """
    profile = config.profile
    profile_id = profile.profile_id

    started = time.perf_counter()
    last = started

    def _emit(phase: str, **fields: object) -> None:
        nonlocal last
        if progress_callback is None:
            return
        now = time.perf_counter()
        progress_callback(
            {
                "phase": phase,
                **fields,
                "elapsed_s": round(now - started, 1),
                "step_s": round(now - last, 1),
            }
        )
        last = now

    _emit(
        "start",
        profile_id=profile_id,
        source_system=config.source_system,
        seed=config.seed,
    )

    dataset_dir = perturbed_dataset_dir(
        roots=config.roots,
        source_system=config.source_system,
        profile_id=profile_id,
        version=config.profile_version,
        seed=config.seed,
    )
    cleansed_dir = layer_directory(dataset_dir, CLEANSED_LAYER_NAME)
    if config.force_rebuild and cleansed_dir.exists():
        shutil.rmtree(cleansed_dir)

    source_frame = load_validation_snapshot(
        roots=config.roots,
        system=config.source_system,
        run_date=None,
        name_col=config.name_col,
    )
    if config.countries:
        source_frame = source_frame.filter(
            pl.col("country").is_in(sorted(set(config.countries)))
        )
    _emit("source_opened", rows=int(source_frame.height))

    write = _partition_writer(
        cleansed_dir=cleansed_dir,
        rows_per_file=resolve_prepared_rows_per_file(config.rows_per_file),
    )
    cleanse_config = CleanseConfig(company_col="name")

    rows_written = 0
    rows_changed = 0
    countries_seen: set[str] = set()
    scenario_counts: dict[tuple[str | None, bool], int] = {}
    emitted_uris: set[str] = set()

    for batch in source_frame.iter_slices(batch_size):
        raw = build_perturbed_frame(
            profile,
            _records_with_lineage(batch, source_system=config.source_system),
            profile_version=config.profile_version,
            seed=config.seed,
        )
        if raw.height == 0:
            continue

        _check_no_cross_batch_collision(frame=raw, seen=emitted_uris)

        cleansed = cleanse_lazyframe(raw.lazy(), cleanse_config).collect()
        write(cleansed)

        rows_written += cleansed.height
        rows_changed += int(raw.get_column("changed").sum())
        countries_seen.update(
            value
            for value in raw.get_column("jurisdiction_code").unique().to_list()
            if value
        )
        for scenario_id, changed, count in (
            raw.group_by("scenario_id", "changed").len(name="n").iter_rows()
        ):
            key = (scenario_id, changed)
            scenario_counts[key] = scenario_counts.get(key, 0) + count

        _emit("written", rows=rows_written, of=int(source_frame.height))

    write.close()  # type: ignore[attr-defined]

    if rows_written == 0:
        raise ValueError(
            f"No perturbed rows were emitted for profile {profile_id!r} against source "
            f"system {config.source_system!r} -- nothing to materialize"
        )

    _emit("generated", rows=rows_written, changed=rows_changed)

    merged_frame = pl.read_parquet(
        resolve_primary_files(cleansed_dir, system_code=profile_id)
    )
    validate_perturbation_dataset_schema(merged_frame, profile_id=profile_id)
    _emit("validated", rows=int(merged_frame.height))

    scenario_tally = pl.DataFrame(
        [
            {"scenario_id": scenario_id, "changed": changed, "records": count}
            for (scenario_id, changed), count in sorted(
                scenario_counts.items(), key=lambda item: (item[0][0] or "", item[0][1])
            )
        ]
    )
    scenario_tally_path = resolve_scenario_tally_path(
        roots=config.roots,
        source_system=config.source_system,
        profile_id=profile_id,
        version=config.profile_version,
        seed=config.seed,
    )
    scenario_tally_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_tally.write_parquet(scenario_tally_path)

    dataset_snapshot_fingerprint_by_country = _source_snapshot_fingerprint_by_country(
        roots=config.roots,
        source_system=config.source_system,
        countries=sorted(countries_seen),
    )

    manifest = build_materialization_manifest(
        roots=config.roots,
        profile=profile,
        seed=config.seed,
        source_system=config.source_system,
        source_dataset_snapshot_fingerprint_by_country=dataset_snapshot_fingerprint_by_country,
        output_dir=cleansed_dir,
        scenario_tally_path=scenario_tally_path,
        rows_emitted=rows_written,
    )
    manifest_path = resolve_materialization_manifest_path(
        roots=config.roots,
        source_system=config.source_system,
        profile_id=profile_id,
        version=config.profile_version,
        seed=config.seed,
    )
    write_materialization_manifest(manifest, manifest_path)
    _emit("complete", manifest=str(manifest_path), output_dir=str(cleansed_dir))

    return MaterializationResult(
        profile_id=profile_id,
        system_code=perturbed_source_uri(
            source_system=config.source_system,
            profile_id=profile_id,
            version=config.profile_version,
            seed=config.seed,
        ),
        output_dir=cleansed_dir,
        manifest_path=manifest_path,
        scenario_tally=scenario_tally,
        rows_emitted=rows_written,
        dataset_snapshot_fingerprint_by_country=dataset_snapshot_fingerprint_by_country,
    )
