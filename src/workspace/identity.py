"""One input-identity contract: what a run is identified by, and the one digest.

A run is identified by four keys, each hashed from what it consumed rather than
the files that held it, so rewriting a layer with identical rows moves nothing:

- the **settings key**, from the configured settings;
- the **population key** per side, from the id and text rows consumed;
- the **truth key**, from the truth rows;
- the **index key**, from the target population key, the settings the index
  is built under and the content of the tokenizer it read.

A target's neighbour-edges key is the index key plus the edge settings.

A run consumes its populations a partition at a time, so each side's key is
built from one digest per partition (`rows_digest`) combined by
`combine_part_keys`, and an index key likewise per partition. Every key is a
`blake2b` digest `DIGEST_SIZE` bytes wide.

No code fingerprint joins any key: a run records its commit as provenance
(`current_commit`), and whether a code change moved results is a known-answer
test's to say.

**Identity from references.** A run named through `workspace.reference` is
identified by the references it consumed and its parameters (`reference_key`).
Each reference contributes its parsed fields, never its rendered URI, so
renaming a scheme or reordering a URI moves no stored key; only the segments
its layout marks as identity count, and roots never do, since a reference holds
none. An immutable reference names one content and contributes itself alone. A
mutable one, a draft or a layer regenerated under one name, must bring a digest
of its content (`digest_directory` for small authored content, a row digest for
a layer), so an edit moves the key. A parameter at its declared default is
dropped, so adding a parameter whose default reproduces past behaviour moves no
existing key.
"""

from __future__ import annotations

import hashlib
import struct
import subprocess  # nosec B404 - reads the checkout's own commit, fixed argv
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from workspace.artifact_archive import compute_artifact_signature
from workspace.reference import Reference

DIGEST_SIZE = 16
"""Bytes of every identity digest (32 hex characters)."""


def digest_settings(settings: Mapping[str, object]) -> str:
    """Digest a settings mapping, independent of key order."""
    return compute_artifact_signature(settings, digest_size=DIGEST_SIZE)


def digest_file(path: Path) -> str:
    """Digest a file's bytes, independent of its timestamps."""
    digest = hashlib.blake2b(digest_size=DIGEST_SIZE)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_directory(directory: Path) -> str:
    """Digest every file beneath `directory` by relative path and bytes,
    independent of timestamps and of the order the filesystem lists them in."""
    digest = hashlib.blake2b(digest_size=DIGEST_SIZE)
    root = Path(directory)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        for field in (path.relative_to(root).as_posix(), digest_file(path)):
            encoded = field.encode("utf-8")
            digest.update(struct.pack("<q", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()


class ContentDigestError(ValueError):
    """A reference given the wrong content digest for its mutability."""


@dataclass(frozen=True)
class ConsumedReference:
    """One reference a run consumed, with its content digest when it is mutable."""

    reference: Reference
    content_digest: str | None = None

    def facet(self) -> dict[str, object]:
        """What this input contributes to a key: kind, side, identity segments
        by name, and the content digest a mutable reference must carry."""
        ref = self.reference
        if ref.immutable and self.content_digest is not None:
            raise ContentDigestError(
                f"{ref.uri} is immutable and names its content; it takes no digest."
            )
        if not ref.immutable and self.content_digest is None:
            raise ContentDigestError(
                f"{ref.uri} is mutable, so its key needs a digest of its content."
            )
        layout = ref.layout
        facet: dict[str, object] = {
            "kind": ref.kind.value,
            "side": ref.side.value,
            "segments": {
                name: value
                for name, value in ref.values
                if layout.segment(name).identity
            },
        }
        if self.content_digest is not None:
            facet["content"] = self.content_digest
        return facet


def reference_key(
    inputs: Mapping[str, ConsumedReference],
    parameters: Mapping[str, object],
    *,
    defaults: Mapping[str, object] | None = None,
    digest_size: int = DIGEST_SIZE,
) -> str:
    """A run's key from the references it consumed, by role, and its parameters.

    Every input must be a complete reference. The key is what an output's own
    reference ends in, cut to that segment's width by the caller.
    """
    for role, consumed in inputs.items():
        if not consumed.reference.complete:
            raise ValueError(
                f"input {role!r} is the selection {consumed.reference.uri}, not one reference."
            )
    effective_defaults = defaults or {}
    return compute_artifact_signature(
        {
            "inputs": {role: inputs[role].facet() for role in sorted(inputs)},
            "parameters": {
                name: value
                for name, value in parameters.items()
                if name not in effective_defaults or effective_defaults[name] != value
            },
        },
        digest_size=digest_size,
    )


def rows_digest(frame: pl.DataFrame, *, id_col: str, value_col: str) -> str:
    """One partition's digest of `frame`'s `(id_col, value_col)` rows, ordered
    by `id_col`, so the rows' order on disk does not matter.

    A population digests its id and scored text, a truth set its source id
    and match.

    Each column is hashed whole, not row by row: a null mask, the value
    lengths and the value bytes, after nulls are filled so a null slot
    contributes no bytes. The mask keeps a missing value apart from an empty
    string, and the lengths keep values apart when their bytes run together.

    Why whole columns: a run computes these keys before it scores, to find a
    finished run to reuse, and every run is meant to fit on one machine. A
    per-row Python loop measured about 2 microseconds a row and turned every
    value into a Python string, so a 200-million-row target would spend
    minutes on keys alone. Polars' own `hash` is not used because its output
    is not guaranteed stable across Polars versions, and a key is compared
    against keys written by earlier runs.

    Measured on 2026-09-15, a blocking key pass over both sides of gleif -> fr
    (13.1 million rows) spends 5.1 seconds here, about 0.4 microseconds a row,
    against about 2 before. What remains is the sort and the one copy of each
    column's text `_update_with_string_column` makes; that copy is the next
    thing to remove if a larger target needs it.
    """
    ordered = frame.select(
        pl.col(id_col).cast(pl.Utf8).alias(id_col),
        pl.col(value_col).cast(pl.Utf8, strict=False).alias(value_col),
    ).sort(id_col)
    digest = hashlib.blake2b(digest_size=DIGEST_SIZE)
    digest.update(struct.pack("<q", ordered.height))
    for column in (id_col, value_col):
        _update_with_string_column(digest, ordered.get_column(column))
    return digest.hexdigest()


def _update_with_string_column(digest: hashlib.blake2b, series: pl.Series) -> None:
    """Feed one string column to `digest`: its null mask, then its value
    lengths in bytes, then its values' bytes run together, the mask and lengths
    fixed-width little-endian so the result does not depend on the platform or
    on how the column is chunked.

    Built from Polars and numpy alone. The joined bytes are one copy of the
    column's text held at once, so peak memory is about twice the largest
    partition's text; reading Arrow's buffers directly would avoid that copy
    but needs `pyarrow` declared as a dependency.
    """
    digest.update(series.is_null().to_numpy().astype(np.uint8).tobytes())
    filled = series.fill_null("")
    lengths = filled.str.len_bytes().cast(pl.Int64).to_numpy()
    digest.update(lengths.astype("<i8").tobytes())
    if filled.len():
        digest.update(filled.str.join("").item().encode("utf-8"))


def combine_part_keys(parts: Mapping[str, str]) -> str:
    """One key from per-partition keys, named by partition."""
    return digest_settings({"parts": dict(sorted(parts.items()))})


def index_key(
    *,
    population_key: str,
    build_settings: Mapping[str, object],
    tokenizer_digest: str | None,
) -> str:
    """A target index's key: the population it is built over, the settings it
    is built under, and the tokenizer content it read, `None` for an index
    built from text rather than tokens."""
    facets: dict[str, object] = {
        "population_key": population_key,
        "build_settings": dict(build_settings),
    }
    if tokenizer_digest is not None:
        facets["tokenizer_digest"] = tokenizer_digest
    return digest_settings(facets)


def neighbour_edges_key(*, index_key: str, edge_settings: Mapping[str, object]) -> str:
    """A target's neighbour-edges key: its index key plus the edge settings."""
    return digest_settings(
        {"index_key": index_key, "edge_settings": dict(edge_settings)}
    )


@dataclass(frozen=True)
class RunKeys:
    """The four keys one run is identified by.

    `truth` is `None` for a run with no ground truth. Two runs are the same
    run exactly when all four agree.
    """

    settings: str
    source_population: str
    target_population: str
    truth: str | None
    index: str

    def as_identity(self) -> dict[str, object]:
        """The mapping a run's directory key is digested from."""
        return {
            "settings_key": self.settings,
            "source_population_key": self.source_population,
            "target_population_key": self.target_population,
            "truth_key": self.truth,
            "index_key": self.index,
        }

    @classmethod
    def from_identity(cls, identity: Mapping[str, object]) -> RunKeys:
        """The keys a recorded identity holds, for a run read back from disk."""
        truth = identity.get("truth_key")
        return cls(
            settings=str(identity["settings_key"]),
            source_population=str(identity["source_population_key"]),
            target_population=str(identity["target_population_key"]),
            truth=None if truth is None else str(truth),
            index=str(identity["index_key"]),
        )


def current_commit(checkout: Path) -> str | None:
    """The commit `checkout` has checked out, for a run's provenance; `None`
    outside a git checkout."""
    try:
        completed = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=checkout,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None
