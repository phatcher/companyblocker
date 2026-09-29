"""Sole owner of the on-disk shape of every data layer.

`src/acquisition` writes `canonical/` and `cleansed/`, and the shape it
writes is the shape `matched/` and `tokenized/` inherit, so no stage
downstream should be deciding it again. Each one used to:
Cleanse chose whether to partition, Tokenize sniffed for
`jurisdiction_code=*` and mirrored what it found, Match built
`jurisdiction_code=<country>` by hand, and a dozen readers each re-derived
"which files here are the real data". Every one of those is a place the rule
can drift, and it has -- a blind glob once picked up `chunks/` staging as
though it were output, and the fix landed in one resolver while its twin had
to be corrected separately.

**The shape.** A layer directory (dated for `canonical/`, undated for the
rest) holds its data in one subdirectory per *family*:

    <layer>/[<snapshot-date>/]
        primary/
            jurisdiction_code=<value>/part-{NNNNN}.parquet   partitioned, or
            {code}-{NNN}.parquet                             flat
        names/
            {code}-names-{NNN}.parquet
        sample.parquet                  scoped artifact, not part of a family
        _dedupe_report.json             canonical only
        {code}-duplicates-{NNN}.parquet canonical only

The `primary` family holds one row per entity; every other family holds a
different row shape keyed off the same `system_uri` (see
`data_file_naming`). They sit in separate subdirectories so that separation
is structural: a `scan_parquet(<layer>/**/*.parquet, hive_partitioning=True)`
over a mixed directory either fails on the schema mismatch or silently
coerces it, and telling the families apart by filename alone means every
reader carries the rule.

**Downstream layers mirror, they do not decide.** A stage that maps a layer
to the next one reproduces its input's relative path under its own root
(`mirror_relative_path`). That is why `cleansed/` has a `primary/`: not
because Cleanse chose to partition or to split families, but because
`canonical/` does and Cleanse reflects it. A stage that never constructs a
path cannot disagree about the shape.

A partition directory's value is lowercased and a null is written as the Hive
null sentinel (`acquisition.output_chunking.NULL_PARTITION_SENTINEL`), since
DuckDB reads a bare `jurisdiction_code=` as an empty string. The rows keep the
source's own value, so a `jurisdiction_code` of `AE-DU` lives under
`jurisdiction_code=ae-du`.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .data_file_naming import (
    COMPANION_NAMES,
    companion_data_file_glob,
    is_primary_data_file,
    primary_data_file_glob,
)

SAMPLE_FILE_NAME = "sample.parquet"

PARTITION_COLUMN = "jurisdiction_code"

STAGING_DIR_NAME = "chunks"

PRIMARY_FAMILY_DIR_NAME = "primary"

NAMES_FAMILY_DIR_NAME = "names"

PARTITION_ROWS_PER_FILE = 1_000_000
"""Rows per file in a partitioned family, and the consolidation size for a
flat one.

Deliberately not `chunk_size` (100,000, which still governs every
Shard-stage write). At 100,000 rows a canonical file is 2-7MB, below one
parquet row group, so a consumer pays small-file overhead on every scan.
1,000,000 is the value Cleanse's merge already used for the view it built
from these same rows, producing 50-74MB files on real `fr`, `gb` and
`offeneregister` data.
"""

LAYER_STAGING_SUFFIX = ".__writing__"
"""Suffix of the directory a stage builds a layer in before swapping it in."""

_LAYER_SUPERSEDED_SUFFIX = ".__superseded__"

_PARTITION_DIR_GLOB = f"{PARTITION_COLUMN}=*"


def primary_family_dir(layer_dir: Path) -> Path:
    """Where a layer keeps its per-entity rows."""
    return layer_dir / PRIMARY_FAMILY_DIR_NAME


def names_family_dir(layer_dir: Path) -> Path:
    """Where a layer keeps its name-variant rows."""
    return layer_dir / NAMES_FAMILY_DIR_NAME


def is_partitioned_layer(layer_dir: Path) -> bool:
    """True when a layer's primary family is Hive-partitioned.

    Checks the pre-family-split location too -- the layer's own top level --
    so a layer that was partitioned but not yet split into families still
    reads as partitioned rather than falling through to the flat branch.
    """
    return any(
        path.is_dir()
        for root in (primary_family_dir(layer_dir), layer_dir)
        for path in root.glob(_PARTITION_DIR_GLOB)
    )


def resolve_primary_files(layer_dir: Path, *, system_code: str) -> list[Path]:
    """Every parquet file holding a layer's per-entity rows.

    Prefers `primary/` and falls back to the layer's own top level, so a
    layer written before the family split still resolves; within either,
    prefers the partitioned view over the flat `{code}-{NNN}.parquet` one.
    Never returns a companion family's files, the scoped `sample.parquet`,
    or anything under `chunks/`. Empty for a layer with none of these
    layouts -- callers decide whether that is an error.
    """
    if not layer_dir.exists():
        return []
    for root in (primary_family_dir(layer_dir), layer_dir):
        if not root.exists():
            continue
        partitioned = sorted(root.glob(f"{_PARTITION_DIR_GLOB}/*.parquet"))
        if partitioned:
            return partitioned
        flat = sorted(
            path
            for path in root.glob(primary_data_file_glob(system_code))
            if is_primary_data_file(file_name=path.name, system_code=system_code)
        )
        if flat:
            return flat
    return []


def resolve_name_files(layer_dir: Path, *, system_code: str) -> list[Path]:
    """Every parquet file holding a layer's name-variant rows.

    Prefers `names/`, falling back to the layer's own top level. Empty when
    the system has no sidecar, which is the ordinary case for several
    systems rather than an error -- see
    `blocking.loader.load_name_variant_frame`.
    """
    if not layer_dir.exists():
        return []
    names_glob = companion_data_file_glob(
        system_code=system_code, family=COMPANION_NAMES
    )
    for root in (names_family_dir(layer_dir), layer_dir):
        if not root.exists():
            continue
        found = sorted(root.glob(names_glob))
        if found:
            return found
    return []


def layer_partition_dir(layer_dir: Path, *, value: str) -> Path:
    """Where a stage should *write* one partition of a layer's primary
    family. Always the current shape -- a writer has no legacy layout to be
    compatible with, only readers do."""
    return primary_family_dir(layer_dir) / f"{PARTITION_COLUMN}={value}"


def resolve_partition_dir(layer_dir: Path, *, value: str) -> Path | None:
    """Where one partition of a layer's primary family actually *is*, or
    None when that partition does not exist.

    Prefers `primary/` and falls back to the layer's own top level, so a
    layer written before the family split still reads.
    """
    for candidate in (
        layer_partition_dir(layer_dir, value=value),
        layer_dir / f"{PARTITION_COLUMN}={value}",
    ):
        if candidate.is_dir():
            return candidate
    return None


def partition_values(layer_dir: Path) -> list[str]:
    """Every partition value present in a layer's primary family, lowercased
    as the directory names are, in sorted order."""
    prefix = f"{PARTITION_COLUMN}="
    for root in (primary_family_dir(layer_dir), layer_dir):
        if not root.exists():
            continue
        values = sorted(
            path.name[len(prefix) :].strip().lower()
            for path in root.glob(_PARTITION_DIR_GLOB)
            if path.is_dir() and path.name[len(prefix) :].strip()
        )
        if values:
            return values
    return []


@dataclass(frozen=True)
class CountryRows:
    """The files holding one country's rows of a dataset, and the column to
    filter them on where they hold other countries' rows too."""

    files: tuple[Path, ...]
    filter_column: str | None


def resolve_country_rows(dataset_dir: Path, *, country: str) -> CountryRows:
    """One country's rows of a dataset, whichever way the dataset stores them.

    `dataset_dir` is a dataset's own location, the family directory its
    reference locates. `canonical` is Hive-partitioned by `PARTITION_COLUMN`,
    so a country is its partition's files and needs no filter; `names` is
    chunked files carrying the column, so a country is every file filtered on
    it. A partitioned dataset with no partition for the country holds no rows
    of it, which is an empty result and no error.
    """
    value = country.strip().lower()
    dataset_dir = Path(dataset_dir)
    if any(path.is_dir() for path in dataset_dir.glob(_PARTITION_DIR_GLOB)):
        partition = dataset_dir / f"{PARTITION_COLUMN}={value}"
        return CountryRows(tuple(sorted(partition.glob("*.parquet"))), None)
    return CountryRows(tuple(sorted(dataset_dir.glob("*.parquet"))), PARTITION_COLUMN)


def mirror_relative_path(input_file: Path, *, input_layer_dir: Path) -> Path:
    """The path a mapping stage should write `input_file`'s output to,
    relative to its own layer root.

    The whole of a stage's shape decision, in one place: reproduce the input
    layer's own relative path. A partitioned
    `primary/jurisdiction_code=gb/part-00001.parquet` maps to the same
    relative path in the next layer, a flat `primary/ie-001.parquet` to a
    flat one, and a family subdirectory carries through because it is part
    of that path -- so Cleanse does not choose whether `cleansed/` has a
    `primary/`, it inherits the answer from `canonical/`.
    """
    return input_file.relative_to(input_layer_dir)


def partition_value_of(relative_path: Path) -> str | None:
    """The partition value in a mirrored relative path, e.g. "gb" for
    `primary/jurisdiction_code=gb/part-00001.parquet`. None for a flat
    layout, which has no partition directory to read one from.
    """
    prefix = f"{PARTITION_COLUMN}="
    for part in relative_path.parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def fresh_layer_staging_dir(live_dir: Path) -> Path:
    """An empty sibling directory to build a layer in before swapping it in."""
    staging_dir = live_dir.parent / f"{live_dir.name}{LAYER_STAGING_SUFFIX}"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    return staging_dir


class LayerSwapError(OSError):
    """A finished layer could not be swapped in over the live one.

    Distinct from a failure *producing* the layer: the staged output is
    complete and valid, so a caller must keep it rather than treat this like
    a failed run and discard the work.
    """


SWAP_RETRY_DELAYS_SECONDS: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
"""Backoff between attempts to rename a layer directory into place.

Windows refuses to rename a directory while any handle inside it is open,
and a just-written layer reliably has one for a moment: a real `gb`
regeneration was blocked on its first attempt and succeeded 0.5s later, on
both runs. Whatever holds it is outside this process -- the handle survives
dropping every frame and a `gc.collect()`, and a small synthetic directory
cannot reproduce it, which points at something scanning the ~570MB of
freshly-written parquet rather than at polars. `ie`, writing a single
smaller file, was never blocked. Concurrent readers are a second possible
source: `data/` is shared by junction across every worktree, so another
session merely reading a layer would block the swap too.

Either way the block is brief, so a short backoff clears it; anything that
outlasts the backoff is real contention an operator should see rather than
something to spin on.
"""


def swap_layer_into_place(*, staging: Path, live: Path) -> None:
    """Replace `live` with `staging` via renames, then drop the old copy.

    Two renames rather than a write-in-place, so the window in which the
    live layer is absent is milliseconds instead of the length of the run.
    `Path.rename` will not overwrite an existing directory on Windows, hence
    moving the old one aside first rather than renaming straight over it.

    Retries on `OSError` (see `SWAP_RETRY_DELAYS_SECONDS`) and, if the swap
    still cannot be made, raises `LayerSwapError` leaving the staged layer
    on disk -- it is finished work, and the previous layer is left in place
    and readable meanwhile, so neither side is lost.
    """
    live.parent.mkdir(parents=True, exist_ok=True)
    superseded = live.parent / f"{live.name}{_LAYER_SUPERSEDED_SUFFIX}"

    last_error: OSError | None = None
    for attempt, delay in enumerate((*SWAP_RETRY_DELAYS_SECONDS, None)):
        try:
            shutil.rmtree(superseded, ignore_errors=True)
            if live.exists():
                live.rename(superseded)
            try:
                staging.rename(live)
            except OSError:
                # Put the previous layer back rather than leaving nothing
                # behind while we wait to try again.
                if superseded.exists() and not live.exists():
                    superseded.rename(live)
                raise
        except OSError as error:
            last_error = error
            if delay is None:
                break
            print(
                f"[warn] Could not swap {live.name} into place "
                f"(attempt {attempt + 1}): {error}. Retrying in {delay:g}s -- "
                "another process is most likely reading it."
            )
            time.sleep(delay)
            continue
        shutil.rmtree(superseded, ignore_errors=True)
        return

    raise LayerSwapError(
        f"Wrote {live.name} successfully but could not swap it in over {live}: "
        f"{last_error}. The finished output is kept at {staging} and the previous "
        f"{live.name} is untouched; move it into place once whatever holds {live} "
        "has released it, rather than re-running the whole stage."
    )
