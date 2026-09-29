"""Language-keyed registry of sentence-embedding checkpoints.

The `"sbert"` representation is parameterised by one string --
`SbertTargetIndexBuildSettings.model_name` -- which `sentence_transformers`
resolves as either a hub checkpoint identifier or a local directory. That
string stays the contract; this module only adds a *slug* namespace in front
of it, so a per-language model can be named as `"fr-sentence-camembert-base"`
instead of a full checkpoint path, and so the set of checkpoints that have
actually been checked is written down in one packaged file rather than in
whichever caller happened to type one.

Purely additive, in both directions:

- An unregistered string passes through `resolve_sbert_model_name()`
  unchanged, so any hub identifier or local path keeps working exactly as it
  did before this module existed, without needing an entry here.
- Slugs and checkpoint identifiers cannot collide: a checkpoint identifier
  always contains `"/"` (`"org/model"`) or is a filesystem path, and a slug
  is validated to contain neither.

The registry itself is a packaged JSON resource
(`resources/sbert_models.json`), following `company_cleanse`'s
`company_type_rules.json` precedent rather than introducing a second
reference-data mechanism.

Being listed here is *not* evidence that a model blocks better than the
English default on any corpus; registration implies no comparative
evaluation. It is evidence of the narrower fact that the checkpoint is a
published sentence-embedding model rather than a NER or masked-LM checkpoint
that `sentence_transformers` would silently wrap in untrained pooling (see
`sbert_pooling_gate.py`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from importlib.resources import files

_SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VALID_COVERAGE = {"monolingual", "multilingual"}


@dataclass(frozen=True)
class SbertModelEntry:
    """One registered sentence-embedding checkpoint.

    Attributes:
        slug: Stable short name callers select the model by. Lowercase
            alphanumerics and hyphens only, so it can never be mistaken for a
            checkpoint identifier (which contains `"/"`) or a filesystem path.
        checkpoint: The `sentence_transformers` checkpoint identifier this
            slug resolves to, as passed to
            `SbertTargetIndexBuildSettings.model_name`.
        languages: ISO 639-1 codes the model is intended for. For a
            `"multilingual"` entry this is the subset relevant to this repo's
            corpora, not the model's full documented coverage -- see `notes`.
        jurisdictions: `jurisdiction_code` partition values (ISO 3166-1
            alpha-2, matching the `jurisdiction_code=` partitions under
            `data/`) this model is intended for.
        coverage: `"monolingual"` or `"multilingual"` -- whether `languages`
            is the model's whole scope or a repo-relevant slice of it.
        notes: Free text; what the model is and any caveat worth carrying.
    """

    slug: str
    checkpoint: str
    languages: tuple[str, ...]
    jurisdictions: tuple[str, ...]
    coverage: str
    notes: str


def _parse_entry(raw: dict, *, index: int) -> SbertModelEntry:
    try:
        slug = str(raw["slug"])
        checkpoint = str(raw["checkpoint"])
        languages = tuple(str(value) for value in raw["languages"])
        jurisdictions = tuple(str(value) for value in raw["jurisdictions"])
        coverage = str(raw["coverage"])
    except KeyError as exc:
        raise ValueError(
            f"sbert_models.json entry #{index} is missing required key {exc}"
        ) from None

    if not _SLUG_PATTERN.fullmatch(slug):
        raise ValueError(
            f"sbert_models.json entry #{index} has invalid slug '{slug}': slugs must "
            f"match {_SLUG_PATTERN.pattern} so they cannot collide with a checkpoint "
            "identifier or a filesystem path."
        )
    if not checkpoint.strip():
        raise ValueError(f"sbert_models.json entry #{index} has an empty checkpoint")
    if not languages:
        raise ValueError(
            f"sbert_models.json entry #{index} ('{slug}') declares no languages"
        )
    if coverage not in _VALID_COVERAGE:
        allowed = ", ".join(sorted(_VALID_COVERAGE))
        raise ValueError(
            f"sbert_models.json entry #{index} ('{slug}') has coverage '{coverage}'; "
            f"must be one of: {allowed}"
        )

    return SbertModelEntry(
        slug=slug,
        checkpoint=checkpoint,
        languages=languages,
        jurisdictions=jurisdictions,
        coverage=coverage,
        notes=str(raw.get("notes", "")),
    )


@cache
def load_sbert_model_registry() -> dict[str, SbertModelEntry]:
    """Load the packaged registry, keyed by slug.

    Raises:
        TypeError: The packaged JSON is not a list of entries.
        ValueError: An entry is malformed -- a missing key, an invalid slug,
            or a duplicate slug.
    """
    resource = files("company_vectorize.resources").joinpath("sbert_models.json")
    raw_entries = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(raw_entries, list):
        raise TypeError("sbert_models.json must contain a list of model entries")

    registry: dict[str, SbertModelEntry] = {}
    for index, raw in enumerate(raw_entries, start=1):
        entry = _parse_entry(raw, index=index)
        if entry.slug in registry:
            raise ValueError(f"sbert_models.json has a duplicate slug: '{entry.slug}'")
        registry[entry.slug] = entry
    return registry


def list_sbert_models() -> tuple[SbertModelEntry, ...]:
    """Every registered model, in packaged-file order."""
    return tuple(load_sbert_model_registry().values())


def sbert_models_for_language(language: str) -> tuple[SbertModelEntry, ...]:
    """Registered models declaring `language` (an ISO 639-1 code)."""
    wanted = str(language).strip().lower()
    return tuple(
        entry for entry in list_sbert_models() if wanted in tuple(entry.languages)
    )


def resolve_sbert_model_name(model: str) -> str:
    """Resolve a registry slug to its checkpoint identifier.

    Idempotent, and safe to apply to any caller-supplied value: anything that
    is not a registered slug -- a hub identifier, a local checkpoint path -- is
    returned unchanged, so the registry never becomes a gate on which models
    are reachable.
    """
    entry = load_sbert_model_registry().get(str(model).strip())
    if entry is None:
        return model
    return entry.checkpoint


def is_registered_checkpoint(model: str) -> bool:
    """Whether `model` is a registered slug or a registered checkpoint.

    Used by `sbert_pooling_gate.py` to skip re-probing a checkpoint whose
    sentence-embedding provenance was already established when it was added
    to the registry.
    """
    candidate = str(model).strip()
    registry = load_sbert_model_registry()
    if candidate in registry:
        return True
    return any(entry.checkpoint == candidate for entry in registry.values())
