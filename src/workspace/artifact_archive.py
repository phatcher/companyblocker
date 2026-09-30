"""One shared convention for a content-addressed artifact archive.

Every area that keeps a durable, content-addressed candidate archive used to
invent its own key: `company_tokenize`'s optimize sweep already canonicalizes
a settings payload to JSON and hashes it with `blake2b` to key the innermost
directory beneath a corpus-hash/trainer/model-variant path, and a second area
needing the same guarantee (a trained-model archive) would otherwise write a
second canonicalization that agrees with the first only by convention, until
the day one of them changes shape and they stop agreeing.

**The shape.** A candidate directory is `root / facet / facet / .../
signature`. The facets are the navigational part, chosen by the caller so a
candidate tree stays browsable by hand -- a corpus-hash prefix, a trainer
name, a model family -- rather than becoming a run of opaque digests, and
`signature` is `compute_artifact_signature`'s digest of everything else that
determines the artifact: both the settings that configured the run and a
content hash of whatever it trained against, folded into the same mapping. A
signature covering only the settings would serve a stale artifact silently
the first time identical settings ran against a different input, rather than
missing outright.

**The consuming area hands over structured values and receives a resolved
directory back.** It derives no key of its own: a package that instead
canonicalized and hashed its own settings before calling this module would
have already done the work this module exists to centralize. `resolve_candidate_dir`
is the one entry point that turns facets-plus-settings into a directory.

**Durability is free by construction.** A different signature is a different
directory, so a later and differently-configured run can never collide with
an earlier one's candidates, and therefore never has to clear them to
proceed. Nothing here enforces that as a runtime check -- there is nothing to
enforce, since two runs that disagree on settings never write to the same
place to begin with.

**Surviving a new setting gaining a default.** Hashing a settings mapping
naively invalidates every existing key the day a new field is added, even
when its default reproduces every past run's behaviour exactly: the new key
appears in every future call's mapping and never in the one that produced an
old digest, so the canonical JSON payload changes and the digest changes with
it, whether or not what it describes actually did. `compute_artifact_signature`'s
`defaults` parameter fixes this by dropping any key from the hashed payload
whose value equals its declared default before hashing, so a caller that adds
a field with a default matching prior behaviour keeps producing the digests
it always has, and only a value that actually deviates from its default
changes anything.

**What this does not resolve.** The promoted/operational location for an
archived artifact is deliberately not part of this shape -- it stays
wherever it already resolves for that area (`tokenizer_scope_dir` for a
tokenizer) -- this module only resolves a *candidate* subtree.

**The package boundary, settled.** A package cannot depend on this module at
all (see this package's README's Boundaries section), so a derivation
reached only from `src`-tier orchestration belongs here, and the key of a
tree a package lays out itself cannot: `company_tokenize` builds the payload
its optimize sweep is keyed by, lays out the sweep path and reads the stored
key back, so its derivation cannot move here without making a package depend
on `src/workspace`.
`company_tokenize.optimize.compute_optimize_grid_hash` stays the tokenizer
archive's one key derivation, permanently, called unchanged by
`src/training/optimize_execution.py`. It is
never migrated here, not for lack of attention, but because doing so would
either break the boundary or split one tree's key from its layout, two
halves that only agree by convention -- the exact drift this
module exists to close. This module is what a *new* area with no such
package-owned tree reaches for instead of writing a second derivation
of its own; an existing package-owned one that already agrees with itself
is a second, equally valid resting place, not a migration backlog. Nothing
about this decision changes `compute_optimize_grid_hash`'s digest, so every
directory already keyed by it stays valid with no migration triggered.

**The keyed store** (`artifact_store_root`) is append-only storage shared by
worktrees and remote sessions. A writer stages a key in a temporary sibling
(`begin_artifact_write`), writes its manifest last and publishes it with one
directory rename (`commit_artifact_write`); a second writer of the same key
keeps the directory already published. Row batches are immutable parquet
files merged by row key on read (`read_artifact_batches`), and the advisory
lock is bounded rather than a reason to wait indefinitely. Only
`prune_artifact` and `prune_stale_temporary_dirs`, through
`scripts/prune_artifacts.py`, delete anything.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_SIGNATURE_DIGEST_SIZE = 16
"""Bytes of the `blake2b` digest `compute_artifact_signature` returns (32 hex
characters) unless a caller overrides it. Full-length by default: shortening
a signature for a shorter directory name is a caller's own display choice
(the same way `company_tokenize`'s existing corpus-hash-prefix facet is
truncated by the caller, not by the function that computed it), never this
module's, since a `digest_size` chosen to make a readable directory name is
not the same guarantee as a `digest_size` chosen to decide whether two
signatures are allowed to collide.
"""

MANIFEST_FILENAME = "manifest.json"
BATCHES_DIRNAME = "batches"
LOCK_FILENAME = ".lock"
LOCK_WAIT_SECONDS = 5.0
LOCK_STALE_SECONDS = 15 * 60.0


def compute_artifact_signature(
    settings: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any] | None = None,
    digest_size: int = DEFAULT_SIGNATURE_DIGEST_SIZE,
) -> str:
    """Stable, order-independent digest of everything that determines one artifact.

    `settings` is expected to carry both the knobs that configured the run and a content
    hash of whatever it trained against (see the module docstring's shape section); this
    function does not distinguish the two, it hashes whatever mapping it is given.

    Two mappings that agree once `defaults` is applied produce the same digest regardless
    of key order (`json.dumps(..., sort_keys=True)`) or process (`hashlib.blake2b`, never
    Python's per-process-salted `hash()`), and two mappings that disagree anywhere in the
    hashed payload produce different digests.

    `defaults`, when given, drops any key from the hashed payload whose value in `settings`
    equals its declared default, so a caller that adds a new field with a default
    reproducing prior behaviour keeps producing every digest it already has. A key absent
    from `defaults` is always hashed; a key present in `defaults` but absent from `settings`
    contributes nothing either way.
    """
    effective_defaults = defaults or {}
    hashed = {
        key: value
        for key, value in settings.items()
        if key not in effective_defaults or effective_defaults[key] != value
    }
    canonical = json.dumps(hashed, sort_keys=True, default=str)
    return hashlib.blake2b(
        canonical.encode("utf-8"), digest_size=digest_size
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ResolvedCandidate:
    """A candidate directory and the digest that named it.

    Both come back together so a caller that records the digest -- in a run
    manifest, or beside a report -- states the one this resolver used rather
    than computing a second digest of the same settings and trusting the two
    to agree.
    """

    directory: Path
    signature: str


def resolve_candidate(
    root: Path,
    *facets: str,
    settings: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
    digest_size: int = DEFAULT_SIGNATURE_DIGEST_SIZE,
) -> ResolvedCandidate:
    """One artifact's candidate directory and its digest: `root` beneath its navigational
    `facets`, keyed by `compute_artifact_signature`'s digest of `settings` as the innermost
    segment.

    The consuming area hands over the facets and the settings and receives the resolved
    directory back rather than deriving a key of its own. What is promoted from a resolved
    candidate directory to an operational/current location is deliberately not this
    function's concern -- see the module docstring's "What this does not resolve".
    """
    signature = compute_artifact_signature(
        settings, defaults=defaults, digest_size=digest_size
    )
    return ResolvedCandidate(
        directory=Path(root).joinpath(*facets, signature), signature=signature
    )


def resolve_candidate_dir(
    root: Path,
    *facets: str,
    settings: Mapping[str, Any],
    defaults: Mapping[str, Any] | None = None,
    digest_size: int = DEFAULT_SIGNATURE_DIGEST_SIZE,
) -> Path:
    """Where `resolve_candidate` puts one artifact's candidate, for a caller with no
    use for the digest that named it."""
    return resolve_candidate(
        root, *facets, settings=settings, defaults=defaults, digest_size=digest_size
    ).directory


def artifact_is_complete(candidate_dir: Path) -> bool:
    """Return whether a candidate has a valid completion manifest."""
    manifest_path = Path(candidate_dir) / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("complete") is True


def begin_artifact_write(candidate_dir: Path) -> Path:
    """Create a temporary sibling directory for one candidate."""
    candidate_dir = Path(candidate_dir)
    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(prefix=f".{candidate_dir.name}.", dir=candidate_dir.parent)
    )


def commit_artifact_write(
    candidate_dir: Path,
    temporary_dir: Path,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Publish a completed temporary candidate without replacing an existing one."""
    candidate_dir = Path(candidate_dir)
    temporary_dir = Path(temporary_dir)
    payload = {**dict(manifest or {}), "complete": True}
    (temporary_dir / MANIFEST_FILENAME).write_text(
        json.dumps(payload, sort_keys=True, indent=2, default=str), encoding="utf-8"
    )
    try:
        temporary_dir.rename(candidate_dir)
    except FileExistsError:
        shutil.rmtree(temporary_dir)
    return candidate_dir


def read_artifact_manifest(candidate_dir: Path) -> dict[str, Any] | None:
    """Read a candidate manifest, or return ``None`` for an incomplete candidate."""
    if not artifact_is_complete(candidate_dir):
        return None
    return json.loads(
        (Path(candidate_dir) / MANIFEST_FILENAME).read_text(encoding="utf-8")
    )


def write_artifact_batch(
    candidate_dir: Path,
    frame: pl.DataFrame,
    *,
    batch_key: str,
) -> Path:
    """Write one immutable parquet batch, making a repeated push harmless."""
    batches_dir = Path(candidate_dir) / BATCHES_DIRNAME
    batches_dir.mkdir(parents=True, exist_ok=True)
    destination = batches_dir / f"{batch_key}.parquet"
    if destination.exists():
        return destination
    temporary_path = batches_dir / f".{batch_key}.{os.getpid()}.tmp"
    frame.write_parquet(temporary_path)
    try:
        temporary_path.rename(destination)
    except FileExistsError:
        temporary_path.unlink(missing_ok=True)
    return destination


def read_artifact_batches(candidate_dir: Path, *, row_key: str) -> pl.DataFrame:
    """Read batches as one frame, keeping the last row for each row key."""
    batch_paths = sorted((Path(candidate_dir) / BATCHES_DIRNAME).glob("*.parquet"))
    if not batch_paths:
        return pl.DataFrame()
    return pl.concat(
        [pl.read_parquet(path) for path in batch_paths], how="diagonal"
    ).unique(subset=[row_key], keep="last", maintain_order=True)


def _lock_is_stale(lock_path: Path, *, stale_after: float) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        started_at = float(payload["started_at"])
        if time.time() - started_at > stale_after:
            return True
        if payload.get("host") != socket.gethostname():
            return False
        os.kill(int(payload["pid"]), 0)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return True
    return False


@contextmanager
def advisory_artifact_lock(
    lock_path: Path,
    *,
    wait_seconds: float = LOCK_WAIT_SECONDS,
    stale_after: float = LOCK_STALE_SECONDS,
) -> Iterator[bool]:
    """Take a bounded advisory lock and yield whether this process acquired it."""
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + wait_seconds
    acquired = False
    while not acquired:
        try:
            with lock_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "host": socket.gethostname(),
                        "pid": os.getpid(),
                        "started_at": time.time(),
                    },
                    handle,
                )
            acquired = True
        except FileExistsError:
            if _lock_is_stale(lock_path, stale_after=stale_after):
                lock_path.unlink(missing_ok=True)
            elif time.monotonic() >= deadline:
                break
            else:
                time.sleep(0.05)
    try:
        yield acquired
    finally:
        if acquired:
            lock_path.unlink(missing_ok=True)


def prune_artifact(candidate_dir: Path) -> bool:
    """Remove one named candidate; this is the store's explicit delete operation."""
    candidate_dir = Path(candidate_dir)
    if not candidate_dir.is_dir():
        return False
    shutil.rmtree(candidate_dir)
    return True


def prune_stale_temporary_dirs(
    root: Path,
    *,
    older_than: float = LOCK_STALE_SECONDS,
) -> list[Path]:
    """Remove abandoned temporary candidate directories beneath a store root."""
    now = time.time()
    removed: list[Path] = []
    for path in Path(root).rglob(".*"):
        if not path.is_dir() or path.name.startswith(".git"):
            continue
        try:
            is_stale = now - path.stat().st_mtime > older_than
        except OSError:
            continue
        if is_stale:
            shutil.rmtree(path)
            removed.append(path)
    return removed
