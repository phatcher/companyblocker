"""Language-keyed registry of pretrained fastText checkpoints.

Same shape and purpose as `company_vectorize.sbert_model_registry`: a stable slug mapped to a
downloadable checkpoint with its languages and jurisdictions, packaged as JSON
(`resources/fasttext_checkpoints.json`) rather than hand-typed at each call site. It differs
from that registry in two ways a sentence-embedding checkpoint does not need: a checkpoint
here is never resolved to something a library downloads for the caller (`sentence_transformers`
manages the sbert registry's hub cache; nothing manages this one's), so an entry carries a
`source_url` a caller downloads from and a `checksum` that names what was actually downloaded,
not an opaque hub identifier a second party already vouches for. `checksum` is `None` until an
entry's checkpoint has actually been downloaded once -- the registry records what has been
verified, not what is merely intended -- and is populated in this packaged file the first time
that happens.

A run resolves one entry from its own countries through
`checkpoint_selection.resolve_checkpoint_for_jurisdictions`, the same rule
`company_vectorize.clustering_policy.resolve_sbert_model_for_jurisdictions` applies to the
sbert registry, generalised to take the registry as a parameter instead of being copied here
with fastText's field names substituted in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

from .checkpoint_selection import resolve_checkpoint_for_jurisdictions

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VALID_COVERAGE = {"monolingual", "multilingual"}

DEFAULT_FASTTEXT_CHECKPOINT_SLUG = "en-cc-300"
"""The pretrained ingredient's default checkpoint: English, the same jurisdictions
(`gb`, `ie`) `company_vectorize`'s sbert registry's English entry declares."""


@dataclass(frozen=True)
class FastTextCheckpointEntry:
    """One registered pretrained fastText checkpoint.

    Attributes:
        slug: Stable short name callers select the checkpoint by. Lowercase alphanumerics
            and hyphens only, matching `SbertModelEntry.slug`'s own pattern.
        source_url: Where the checkpoint archive downloads from. Facebook's official
            Common Crawl + Wikipedia release for every entry registered so far
            (`https://fasttext.cc/docs/en/crawl-vectors.html`), gzip-compressed. A downloader
            decompresses this to a plain `.bin` before handing it to
            `token_vector_lookup.load_pretrained_fasttext_vectors()` -- see that module's
            docstring for why a checkpoint this size cannot be loaded straight from the `.gz`.
        languages: ISO 639-1 codes the checkpoint is intended for.
        jurisdictions: `jurisdiction_code` partition values (ISO 3166-1 alpha-2) this
            checkpoint is intended for -- the same values
            `company_vectorize.sbert_model_registry.SbertModelEntry.jurisdictions` is keyed
            on, so a run's countries resolve a language the same way for either
            representation.
        coverage: `"monolingual"` or `"multilingual"`.
        dimension: Width of the checkpoint's vectors.
        checksum: sha256 hex digest of the downloaded `source_url` payload, or `None` if this
            entry has never been downloaded. A downloaded checkpoint's identity is this value,
            not `source_url` alone -- the URL can start serving a different file without
            changing, the checksum cannot change without naming a different file.
        notes: Free text; what the checkpoint is and any caveat worth carrying.
    """

    slug: str
    source_url: str
    languages: tuple[str, ...]
    jurisdictions: tuple[str, ...]
    coverage: str
    dimension: int
    checksum: str | None
    notes: str


def _parse_entry(raw: dict, *, index: int) -> FastTextCheckpointEntry:
    try:
        slug = str(raw["slug"])
        source_url = str(raw["source_url"])
        languages = tuple(str(value) for value in raw["languages"])
        jurisdictions = tuple(str(value) for value in raw["jurisdictions"])
        coverage = str(raw["coverage"])
        dimension = int(raw["dimension"])
    except KeyError as exc:
        raise ValueError(
            f"fasttext_checkpoints.json entry #{index} is missing required key {exc}"
        ) from None

    if not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError(
            f"fasttext_checkpoints.json entry #{index} has invalid slug '{slug}': slugs "
            f"must match {_SLUG_PATTERN.pattern}."
        )
    if not source_url.strip():
        raise ValueError(
            f"fasttext_checkpoints.json entry #{index} has an empty source_url"
        )
    if not languages:
        raise ValueError(
            f"fasttext_checkpoints.json entry #{index} ('{slug}') declares no languages"
        )
    if coverage not in _VALID_COVERAGE:
        allowed = ", ".join(sorted(_VALID_COVERAGE))
        raise ValueError(
            f"fasttext_checkpoints.json entry #{index} ('{slug}') has coverage "
            f"'{coverage}'; must be one of: {allowed}"
        )

    checksum = raw.get("checksum")
    return FastTextCheckpointEntry(
        slug=slug,
        source_url=source_url,
        languages=languages,
        jurisdictions=jurisdictions,
        coverage=coverage,
        dimension=dimension,
        checksum=str(checksum) if checksum else None,
        notes=str(raw.get("notes", "")),
    )


@cache
def load_fasttext_checkpoint_registry() -> dict[str, FastTextCheckpointEntry]:
    """Load the packaged registry, keyed by slug.

    Raises:
        TypeError: The packaged JSON is not a list of entries.
        ValueError: An entry is malformed -- a missing key, an invalid slug, or a duplicate
            slug.
    """
    resource = files("company_classify.resources").joinpath("fasttext_checkpoints.json")
    raw_entries = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise TypeError(
            "fasttext_checkpoints.json must contain a list of checkpoint entries"
        )

    registry: dict[str, FastTextCheckpointEntry] = {}
    for index, raw in enumerate(raw_entries, start=1):
        entry = _parse_entry(raw, index=index)
        if entry.slug in registry:
            raise ValueError(
                f"fasttext_checkpoints.json has a duplicate slug: '{entry.slug}'"
            )
        registry[entry.slug] = entry
    return registry


def list_fasttext_checkpoints() -> tuple[FastTextCheckpointEntry, ...]:
    """Every registered checkpoint, in packaged-file order."""
    return tuple(load_fasttext_checkpoint_registry().values())


def fasttext_checkpoints_for_language(
    language: str,
) -> tuple[FastTextCheckpointEntry, ...]:
    """Registered checkpoints declaring `language` (an ISO 639-1 code)."""
    wanted = str(language).strip().lower()
    return tuple(
        entry
        for entry in list_fasttext_checkpoints()
        if wanted in tuple(entry.languages)
    )


def resolve_fasttext_checkpoint_entry(slug: str) -> FastTextCheckpointEntry:
    """The registered entry for `slug`.

    Raises:
        KeyError: `slug` is not a registered slug.
    """
    registry = load_fasttext_checkpoint_registry()
    try:
        return registry[str(slug).strip()]
    except KeyError:
        raise KeyError(
            f"'{slug}' is not a registered fastText checkpoint slug"
        ) from None


def resolve_fasttext_slug_for_jurisdictions(
    jurisdictions: tuple[str, ...] | None,
) -> str:
    """Pick a registered checkpoint slug for a run scoped to `jurisdictions`.

    Delegates to `checkpoint_selection.resolve_checkpoint_for_jurisdictions` -- the same rule
    `company_vectorize.clustering_policy.resolve_sbert_model_for_jurisdictions` applies to the
    sbert registry -- over this registry's own entries, falling back to
    `DEFAULT_FASTTEXT_CHECKPOINT_SLUG` whenever `jurisdictions` is `None`/empty, spans more
    than one monolingual entry's coverage, or matches no registered monolingual entry.
    """
    return resolve_checkpoint_for_jurisdictions(
        jurisdictions,
        list_fasttext_checkpoints(),
        identifier=lambda entry: entry.slug,
        default=DEFAULT_FASTTEXT_CHECKPOINT_SLUG,
    )
