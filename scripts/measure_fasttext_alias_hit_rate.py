"""Download a registered pretrained fastText checkpoint and measure its alias-pair hit rate.

`company_classify`'s package code (`fasttext_registry.py`, `token_vector_lookup.py`,
`alias_probe.py`) resolves no artifact path and downloads nothing itself, matching this
package's existing convention that a caller supplies the destination
(`persistence.py`'s module docstring). This script is that caller: it resolves one registry
entry (by slug, or from `--jurisdiction` through the same selection rule a run would use),
downloads its checkpoint into `workspace.artifact_layout.pretrained_vector_artifact_root()`
keyed by the checkpoint's own sha256 checksum (never re-downloading an already-present
checksum directory), loads it through `load_pretrained_fasttext_vectors`, and runs
`nearest_neighbour_alias_hit_rate()` over one jurisdiction's real observed name-variant rows.
`--jurisdiction` is the one argument for both: it picks the checkpoint through the registry's
own jurisdiction list (`gb`/`ie` resolve to English) and filters the names sidecar, both
compared case-insensitively. Two separate, differently-cased arguments used to pick the
checkpoint and filter the names is a defect this script once had: a non-English run picked
the right checkpoint and still filtered on the English default, loading zero rows and
reporting a 0.0 hit rate as if that were a real measurement.

A checkpoint archive is several gigabytes; downloading it is the one real-data step this
script performs, gated behind `--download` (default off) so a dry or CI invocation never
triggers a multi-gigabyte fetch by accident. The full report, including the checksum actually
downloaded, is written to `artifacts/analysis/fasttext_alias_hit_rate/runs/<date>/metrics/`
(never a caller-named path -- this script takes only the workspace roots and plain
parameters, no location flag of its own) for the caller to fold into the registry's own
`checksum` field.

The download is decompressed to a plain `.bin` sibling before loading (streamed through
Python's own `gzip` module, the archive kept beside the decompressed `.bin`), never handed to
`gensim` compressed: `cc.en.300.bin.gz` decompresses past 4 GiB, and
`gensim.models.fasttext.load_facebook_vectors`'s advertised transparent gzip reading
(`smart_open`) reads that real checkpoint short partway through its vectors matrix (confirmed
on this item's dispatch; the downloaded archive's own gzip integrity, `gzip -t`, passes, so
this is a large-stream read defect in the compressed-loading path, not a corrupt download).
Decompressing through the `gzip` module rather than shelling out needs nothing on `PATH`: a
Windows shell without Git's `usr\\bin` on its `PATH` does not find the system `gzip` binary a
subprocess-based decompression would call.

Usage:
    .venv/Scripts/python.exe scripts/measure_fasttext_alias_hit_rate.py --download
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import polars as pl
from cli_common import (
    add_run_date_arg,
    add_workspace_roots_args,
    resolve_workspace_roots_from_args,
    run_reporting_argument_errors,
)
from company_classify import (
    TokenVectorProvenance,
    load_pretrained_fasttext_vectors,
    mean_pool_name_vector,
    name_variants_from_rows,
    nearest_neighbour_alias_hit_rate,
    resolve_fasttext_checkpoint_entry,
    resolve_fasttext_slug_for_jurisdictions,
)

from acquisition.downloader_common import build_remote_request, stream_download_to_file
from analysis.report_layout import metrics_dir
from workspace.artifact_layout import (
    analysis_report_run_dir,
    pretrained_vector_artifact_root,
)
from workspace.data_layout import canonical_snapshot_dir
from workspace.layer_layout import resolve_name_files
from workspace.match_resolution import entity_of
from workspace.roots import WorkspaceRoots

_SYSTEM = "wikidata"
_CANONICAL_DATE = "2026-07-16"
_JURISDICTION = "GB"
_USER_AGENT = "blocking-fasttext-download/0.1"
REPORT_NAME = "fasttext_alias_hit_rate"


def report_path(roots: WorkspaceRoots, run_date: str, slug: str) -> Path:
    """Where one checkpoint's full report is written for one run date."""
    run_dir = analysis_report_run_dir(roots, REPORT_NAME, run_date)
    return metrics_dir(run_dir) / f"{slug}_alias_hit_rate.json"


def resolve_checkpoint_slug(*, slug: str | None, jurisdiction: str | None) -> str:
    """`--slug` if given, otherwise the checkpoint `--jurisdiction` alone would resolve."""
    if slug:
        return slug
    return resolve_fasttext_slug_for_jurisdictions(
        (jurisdiction,) if jurisdiction else None
    )


def ensure_checkpoint_downloaded(
    roots: WorkspaceRoots, *, slug: str, allow_download: bool
) -> tuple[Path, str]:
    """Resolve `slug`'s checkpoint file on disk, downloading it if `allow_download` and absent.

    Returns `(local_path, checksum)`. If the registry already records a checksum and a file
    already sits at `<root>/<checksum>/<filename>`, nothing is downloaded. A freshly-downloaded
    file is verified against the registry's checksum when one is recorded, and written beneath
    the checksum it actually hashed to otherwise -- never over an existing checksum directory.

    Raises:
        FileNotFoundError: The checkpoint is not present locally and `allow_download` is
            `False`.
    """
    entry = resolve_fasttext_checkpoint_entry(slug)
    filename = entry.source_url.rsplit("/", maxsplit=1)[-1]
    root = pretrained_vector_artifact_root(roots)

    if entry.checksum:
        destination = root / entry.checksum / filename
        if destination.is_file():
            return destination, entry.checksum

    if not allow_download:
        raise FileNotFoundError(
            f"No local checkpoint for '{slug}' and --download was not given "
            f"(would fetch {entry.source_url})"
        )

    temporary_dir = root / f".downloading-{slug}"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_dir / filename
    request = build_remote_request(
        entry.source_url, headers={"User-Agent": _USER_AGENT}
    )
    print(f"Downloading {entry.source_url} -> {temporary_path}")
    started = time.perf_counter()
    checksum = stream_download_to_file(request, temporary_path)
    elapsed = time.perf_counter() - started
    print(
        f"Downloaded {temporary_path.stat().st_size:,} bytes in {elapsed:.1f}s, sha256={checksum}"
    )

    if entry.checksum and entry.checksum != checksum:
        raise ValueError(
            f"Downloaded checksum {checksum} does not match registered checksum "
            f"{entry.checksum} for '{slug}'"
        )

    destination_dir = root / checksum
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / filename
    if destination.is_file():
        temporary_path.unlink(missing_ok=True)
    else:
        temporary_path.replace(destination)
    temporary_dir.rmdir()
    return destination, checksum


def ensure_checkpoint_decompressed(archive_path: Path) -> Path:
    """The plain, decompressed sibling of a downloaded `.gz` checkpoint, decompressing once.

    `archive_path` must end in `.gz`; the sibling with that suffix stripped is reused if it
    already exists (never re-decompressed), and produced by streaming through Python's own
    `gzip` module (keeping the archive) otherwise -- see this module's docstring for why
    `gensim` never reads the `.gz` directly here, and for why this streams rather than
    shelling out to the system `gzip` binary.
    """
    if archive_path.suffix != ".gz":
        return archive_path
    decompressed_path = archive_path.with_suffix("")
    if decompressed_path.is_file():
        return decompressed_path
    print(f"Decompressing {archive_path} -> {decompressed_path}")
    started = time.perf_counter()
    with gzip.open(archive_path, "rb") as source, open(decompressed_path, "wb") as dest:
        shutil.copyfileobj(source, dest)
    print(f"Decompressed in {time.perf_counter() - started:.1f}s")
    return decompressed_path


def load_real_alias_variants(
    roots: WorkspaceRoots, *, system: str, canonical_date: str, jurisdiction_code: str
):
    """Real observed name-variant rows for one jurisdiction, grouped by entity.

    Resolves each row's own `system_uri` back to its entity via
    `workspace.match_resolution.entity_of` before adapting rows through
    `company_classify.pairs.name_variants_from_rows()` -- the same resolution the two
    ceiling scripts (`measure_pair_classifier_ceiling.py`,
    `measure_pooled_subword_encoder_ceiling.py`) apply: a names sidecar's own `system_uri`
    column is the row's own derived identity, one per name variant
    (`workspace.derived_uri.name_variant_uri`'s per-row content hash), not the entity it
    belongs to (see `pairs.NameVariant.system_uri`'s own docstring), so grouping on it
    unresolved puts every variant in its own singleton group and yields zero alias pairs for
    a probe to find.

    `jurisdiction_code` is matched against the sidecar's own `jurisdiction_code` column
    case-insensitively: the sidecar stores it upper-case, but this filters the same argument
    `resolve_checkpoint_slug()` uses to pick the checkpoint, which is compared lower-case
    there (`checkpoint_selection.resolve_checkpoint_for_jurisdictions`).

    Raises:
        FileNotFoundError: No names sidecar parquet exists under the resolved canonical
            snapshot directory.
        ValueError: `jurisdiction_code` matched no name rows -- a filter that silently loads
            nothing must not be read as a valid, if empty, measurement.
    """
    canonical_dir = canonical_snapshot_dir(
        roots, system=system, run_date=canonical_date
    )
    files = resolve_name_files(canonical_dir, system_code=system)
    if not files:
        raise FileNotFoundError(f"No names sidecar parquet under {canonical_dir}")

    wanted = jurisdiction_code.strip().lower()
    frame = (
        pl.scan_parquet([str(path) for path in files])
        .filter(pl.col("jurisdiction_code").str.to_lowercase() == wanted)
        .select(["system_uri", "name", "name_type"])
        .collect()
    )
    rows = (
        {**row, "system_uri": entity_of(str(row["system_uri"])).uri}
        for row in frame.iter_rows(named=True)
    )
    variants = name_variants_from_rows(rows)
    if not variants:
        raise ValueError(
            f"No name rows matched jurisdiction {jurisdiction_code!r} in "
            f"{system}/{canonical_date}"
        )
    return variants


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workspace_roots_args(parser)
    parser.add_argument("--system", default=_SYSTEM)
    parser.add_argument("--canonical-date", default=_CANONICAL_DATE)
    parser.add_argument(
        "--jurisdiction",
        default=_JURISDICTION,
        help="Jurisdiction to probe (e.g. 'gb', 'IE', 'fr'). Picks the checkpoint through "
        "the registry's jurisdiction list when --slug is not given, and filters the names "
        "sidecar to this jurisdiction; both comparisons are case-insensitive.",
    )
    parser.add_argument(
        "--slug",
        default=None,
        help="Registry slug to measure. Default: resolved from --jurisdiction "
        "(or the pretrained default if unset).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the checkpoint if it is not already present locally.",
    )
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Anchors scored against the pool per batch (see "
        "`nearest_neighbour_alias_hit_rate`'s own default when omitted). Peak memory is "
        "roughly `chunk_size x pool_size` floats: shrink this to fit a larger jurisdiction's "
        "pool into less memory.",
    )
    add_run_date_arg(
        parser,
        detail="Names the analysis run directory the report is written under.",
        default_behavior="today",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    roots = resolve_workspace_roots_from_args(args)

    slug = resolve_checkpoint_slug(slug=args.slug, jurisdiction=args.jurisdiction)
    entry = resolve_fasttext_checkpoint_entry(slug)
    print(f"Resolved checkpoint slug: {slug} ({entry.source_url})")

    archive_path, checksum = ensure_checkpoint_downloaded(
        roots, slug=slug, allow_download=args.download
    )
    checkpoint_path = ensure_checkpoint_decompressed(archive_path)

    provenance = TokenVectorProvenance(
        slug=slug, source=entry.source_url, checksum=checksum
    )
    print(f"Loading checkpoint from {checkpoint_path} ...")
    load_started = time.perf_counter()
    lookup = load_pretrained_fasttext_vectors(checkpoint_path, provenance=provenance)
    load_elapsed = time.perf_counter() - load_started
    print(
        f"Loaded in {load_elapsed:.1f}s: dimension={lookup.dimension}, "
        f"vocab_total_count={lookup.total}"
    )

    variants = load_real_alias_variants(
        roots,
        system=args.system,
        canonical_date=args.canonical_date,
        jurisdiction_code=args.jurisdiction,
    )
    print(
        f"{len(variants)} real name-variant rows loaded for {args.system}/{args.jurisdiction}"
    )

    probe_kwargs: dict[str, Any] = {"top_k": args.top_k}
    if args.chunk_size is not None:
        probe_kwargs["chunk_size"] = args.chunk_size

    probe_started = time.perf_counter()
    result = nearest_neighbour_alias_hit_rate(
        variants,
        embed=lambda name: mean_pool_name_vector(name, lookup),
        **probe_kwargs,
    )
    probe_elapsed = time.perf_counter() - probe_started

    report: dict[str, Any] = {
        "slug": slug,
        "source_url": entry.source_url,
        "checksum": checksum,
        "dimension": lookup.dimension,
        "vocab_total_count": lookup.total,
        "system": args.system,
        "jurisdiction": args.jurisdiction,
        "canonical_date": args.canonical_date,
        "variant_rows": len(variants),
        "probe": {
            "hit_rate": result.hit_rate,
            "hits": result.hits,
            "total": result.total,
            "top_k": result.top_k,
            "pool_size": result.pool_size,
        },
        "timing_seconds": {
            "load_checkpoint": load_elapsed,
            "probe": probe_elapsed,
        },
    }
    print(json.dumps(report, indent=2))

    json_out = report_path(
        roots, args.run_date or datetime.now(UTC).date().isoformat(), slug
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote full report to {json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run_reporting_argument_errors(main))
