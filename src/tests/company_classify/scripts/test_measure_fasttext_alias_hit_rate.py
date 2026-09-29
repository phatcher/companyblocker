"""Tests for `scripts.measure_fasttext_alias_hit_rate`'s own logic: checkpoint selection,
the download/decompression cache paths, and reading real alias variants -- never a real
download or a real gensim load, which has been run once for real by hand.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl
import pytest
from company_classify import DEFAULT_FASTTEXT_CHECKPOINT_SLUG

from scripts.measure_fasttext_alias_hit_rate import (
    ensure_checkpoint_decompressed,
    ensure_checkpoint_downloaded,
    load_real_alias_variants,
    report_path,
    resolve_checkpoint_slug,
)
from workspace.artifact_layout import (
    analysis_report_run_dir,
    pretrained_vector_artifact_root,
)
from workspace.data_layout import CANONICAL_LAYER_NAME
from workspace.derived_uri import name_variant_uri
from workspace.layer_layout import names_family_dir
from workspace.roots import WorkspaceRoots


def test_report_path_is_under_the_dated_analysis_run(workspace_roots: WorkspaceRoots):
    run_dir = analysis_report_run_dir(
        workspace_roots, "fasttext_alias_hit_rate", "2026-09-29"
    )
    assert report_path(workspace_roots, "2026-09-29", "en-cc-300") == (
        run_dir / "metrics" / "en-cc-300_alias_hit_rate.json"
    )


def test_resolve_checkpoint_slug_prefers_an_explicit_slug():
    assert resolve_checkpoint_slug(slug="fr-cc-300", jurisdiction="de") == "fr-cc-300"


@pytest.mark.parametrize(
    ("jurisdiction", "expected_slug"),
    [
        ("gb", DEFAULT_FASTTEXT_CHECKPOINT_SLUG),
        ("IE", DEFAULT_FASTTEXT_CHECKPOINT_SLUG),
        ("fr", "fr-cc-300"),
    ],
)
def test_resolve_checkpoint_slug_falls_back_to_jurisdiction_resolution(
    jurisdiction: str, expected_slug: str
):
    # The registry's own jurisdiction lists are lower-case ("gb", "ie"); a mixed-case
    # --jurisdiction (e.g. "IE") must still resolve the checkpoint a lower-case one would.
    assert (
        resolve_checkpoint_slug(slug=None, jurisdiction=jurisdiction) == expected_slug
    )


def test_resolve_checkpoint_slug_defaults_with_no_jurisdiction():
    assert (
        resolve_checkpoint_slug(slug=None, jurisdiction=None)
        == DEFAULT_FASTTEXT_CHECKPOINT_SLUG
    )


def test_ensure_checkpoint_downloaded_returns_the_already_present_file_with_no_download(
    workspace_roots: WorkspaceRoots,
):
    # en-cc-300's registry checksum is recorded (this item's dispatch downloaded and
    # checksummed it for real); pre-creating the file it names must short-circuit
    # ensure_checkpoint_downloaded() without needing allow_download=True.
    from company_classify import resolve_fasttext_checkpoint_entry

    entry = resolve_fasttext_checkpoint_entry(DEFAULT_FASTTEXT_CHECKPOINT_SLUG)
    assert entry.checksum, "expected en-cc-300 to carry a recorded checksum"
    filename = entry.source_url.rsplit("/", maxsplit=1)[-1]
    destination = (
        pretrained_vector_artifact_root(workspace_roots) / entry.checksum / filename
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"fake checkpoint bytes")

    resolved_path, checksum = ensure_checkpoint_downloaded(
        workspace_roots, slug=DEFAULT_FASTTEXT_CHECKPOINT_SLUG, allow_download=False
    )

    assert resolved_path == destination
    assert checksum == entry.checksum


def test_ensure_checkpoint_downloaded_raises_when_absent_and_download_not_allowed(
    workspace_roots: WorkspaceRoots,
):
    # fr-cc-300 has never been downloaded on this dispatch, so its registry checksum is
    # still unset and nothing can short-circuit the download.
    with pytest.raises(FileNotFoundError):
        ensure_checkpoint_downloaded(
            workspace_roots, slug="fr-cc-300", allow_download=False
        )


def test_ensure_checkpoint_decompressed_passes_through_a_non_gz_path(tmp_path: Path):
    plain_path = tmp_path / "already-plain.bin"
    plain_path.write_bytes(b"plain")

    assert ensure_checkpoint_decompressed(plain_path) == plain_path


def test_ensure_checkpoint_decompressed_decompresses_a_gz_archive(tmp_path: Path):
    archive_path = tmp_path / "checkpoint.bin.gz"
    with gzip.open(archive_path, "wb") as handle:
        handle.write(b"decompressed checkpoint content")

    decompressed_path = ensure_checkpoint_decompressed(archive_path)

    assert decompressed_path == tmp_path / "checkpoint.bin"
    assert decompressed_path.read_bytes() == b"decompressed checkpoint content"
    assert archive_path.exists()  # gzip -dk keeps the archive


def test_ensure_checkpoint_decompressed_reuses_an_already_decompressed_sibling(
    tmp_path: Path,
):
    archive_path = tmp_path / "checkpoint.bin.gz"
    with gzip.open(archive_path, "wb") as handle:
        handle.write(b"this would be wrong if re-decompressed over the sentinel")
    decompressed_path = tmp_path / "checkpoint.bin"
    decompressed_path.write_bytes(b"sentinel: already decompressed")

    result = ensure_checkpoint_decompressed(archive_path)

    assert result == decompressed_path
    assert result.read_bytes() == b"sentinel: already decompressed"


def _write_wikidata_names_fixture(layer_fixture_dir) -> None:
    """One entity per jurisdiction (GB, IE, FR), each with two recorded name forms, and its
    `system_uri` column holding real per-row name-variant URIs (`name://<system>/<id>/<hash>`)
    resolvable to the entity through `workspace.match_resolution.entity_of` -- the shape the
    real wikidata canonical sidecar carries.
    """
    names_dir = names_family_dir(
        layer_fixture_dir("wikidata", layer=CANONICAL_LAYER_NAME) / "2026-07-16"
    )
    names_dir.mkdir(parents=True, exist_ok=True)
    rows = {
        "wikidata://Q1": (
            "GB",
            [("label", "Acme Systems Limited"), ("alias", "Acme Systems Ltd")],
        ),
        "wikidata://Q2": (
            "IE",
            [("label", "Blue River Group"), ("alias", "Blue River Grp")],
        ),
        "wikidata://Q3": ("FR", [("label", "Rive Bleue SA"), ("alias", "Rive Bleue")]),
    }
    system_uris: list[str] = []
    names: list[str] = []
    name_types: list[str] = []
    jurisdictions: list[str] = []
    for entity_uri, (jurisdiction, forms) in rows.items():
        for name_type, value in forms:
            system_uris.append(
                name_variant_uri(
                    source_uri=entity_uri, name_type=name_type, value=value
                )
            )
            names.append(value)
            name_types.append(name_type)
            jurisdictions.append(jurisdiction)
    pl.DataFrame(
        {
            "system_uri": system_uris,
            "name": names,
            "name_type": name_types,
            "jurisdiction_code": jurisdictions,
        }
    ).write_parquet(names_dir / "wikidata-names-001.parquet")


def test_load_real_alias_variants_groups_name_rows_by_entity_through_entity_of(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    # resolves through `workspace.match_resolution.entity_of`, like the two ceiling
    # scripts, rather than reading the sidecar's own `source_uri` column.
    _write_wikidata_names_fixture(layer_fixture_dir)

    variants = load_real_alias_variants(
        workspace_roots,
        system="wikidata",
        canonical_date="2026-07-16",
        jurisdiction_code="GB",
    )

    assert {variant.system_uri for variant in variants} == {"wikidata://Q1"}
    assert sorted(variant.name for variant in variants) == [
        "Acme Systems Limited",
        "Acme Systems Ltd",
    ]


@pytest.mark.parametrize(
    ("argument", "expected_entity"),
    [("gb", "wikidata://Q1"), ("IE", "wikidata://Q2"), ("fr", "wikidata://Q3")],
)
def test_load_real_alias_variants_filters_case_insensitively(
    workspace_roots: WorkspaceRoots,
    layer_fixture_dir,
    argument: str,
    expected_entity: str,
):
    # the sidecar's own `jurisdiction_code` is upper-case; a mixed-case
    # `--jurisdiction` argument must still match it.
    _write_wikidata_names_fixture(layer_fixture_dir)

    variants = load_real_alias_variants(
        workspace_roots,
        system="wikidata",
        canonical_date="2026-07-16",
        jurisdiction_code=argument,
    )

    assert {variant.system_uri for variant in variants} == {expected_entity}


def test_load_real_alias_variants_raises_when_the_filter_matches_no_rows(
    workspace_roots: WorkspaceRoots, layer_fixture_dir
):
    # a filter that silently loads nothing must not be read as a valid, if empty,
    # measurement -- the bug fixed here reported a 0.0 hit rate exactly this way.
    _write_wikidata_names_fixture(layer_fixture_dir)

    with pytest.raises(ValueError, match="No name rows matched"):
        load_real_alias_variants(
            workspace_roots,
            system="wikidata",
            canonical_date="2026-07-16",
            jurisdiction_code="de",
        )
