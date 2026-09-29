"""Representation -> strategy/build-settings selection policy.

`clustering_factory.py`'s two public functions used to mix two concerns in
one `if`/`elif` ladder each: *which* concrete type a representation name
maps to (policy), and *actually instantiating* that type (construction/
wiring). This module owns only the former: given a representation name (and,
for build settings, the raw config values a caller supplied), it returns
*what* to build -- a class object and, for build settings, the keyword
arguments to build it with -- without ever calling the constructor itself.
`clustering_factory.py` calls into this module and performs the actual
construction, so the two concerns are separately testable: a policy test can
assert "sbert maps to `SbertClusteringStrategy`" without instantiating
anything, and a factory test can assert the returned object is wired
correctly without re-deriving the representation-name mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .clustering_contract import (
    ClusteringStrategy,
    EncoderHandle,
    EncoderTargetIndexBuildSettings,
    SbertTargetIndexBuildSettings,
    SentencepieceTargetIndexBuildSettings,
    TargetIndexBuildSettings,
    TfidfTargetIndexBuildSettings,
    WordpieceTargetIndexBuildSettings,
)
from .encoder_strategy import EncoderClusteringStrategy
from .sbert_model_registry import list_sbert_models, resolve_sbert_model_name
from .sbert_strategy import DEFAULT_SBERT_MODEL_NAME, SbertClusteringStrategy
from .sentencepiece_strategy import SentencepieceClusteringStrategy
from .tfidf_strategy import TfidfClusteringStrategy
from .wordpiece_cluster import WordpieceClusteringStrategy

_STRATEGY_CLASSES: dict[str, type[ClusteringStrategy]] = {
    "tfidf": TfidfClusteringStrategy,
    "wordpiece": WordpieceClusteringStrategy,
    "sentencepiece": SentencepieceClusteringStrategy,
    "sbert": SbertClusteringStrategy,
    "encoder": EncoderClusteringStrategy,
}


def resolve_strategy_class(representation: str) -> type[ClusteringStrategy]:
    """Pick which `ClusteringStrategy` subclass a representation name maps to.

    Pure lookup/validation: returns the class object itself, never an
    instance -- callers that need an instance should go through
    `clustering_factory.resolve_clustering_strategy()`, which performs the
    construction step this function deliberately stays out of.

    Raises:
        ValueError: `representation` is not a known representation name.
    """
    try:
        return _STRATEGY_CLASSES[representation]
    except KeyError:
        raise ValueError(f"Unsupported representation: {representation}") from None


def resolve_sbert_model_for_jurisdictions(jurisdictions: tuple[str, ...] | None) -> str:
    """Pick a registered checkpoint for an `sbert` run scoped to `jurisdictions`.

    Selection is per run rather than per record: the caller
    supplies the jurisdictions its run is already scoped to --
    `blocking.contracts.BlockingRunConfig.countries` for a two-dataset
    blocking run -- and this picks the one registered *monolingual* entry
    whose own `jurisdictions` covers all of them, on the view that a run
    scoped to a single jurisdiction supplies real language signal for free.
    `jurisdiction_code` values are exactly what `sbert_model_registry.json`'s
    `jurisdictions` field is keyed on, so no separate country-to-language
    mapping is needed.

    Falls back to `DEFAULT_SBERT_MODEL_NAME` -- the pretrained English
    default -- whenever `jurisdictions` is `None`/empty, spans more than one
    monolingual entry's coverage (ambiguous), or matches no registered
    monolingual entry at all, since most runs are not language-specific and
    an unmatched jurisdiction should resolve to something usable rather than
    fail the run. A multilingual entry is never returned here: it exists for
    an explicit `sbert_model_name`, not as an inferred per-jurisdiction
    default.
    """
    if jurisdictions:
        wanted = {str(value).strip().lower() for value in jurisdictions}
        candidates = [
            entry
            for entry in list_sbert_models()
            if entry.coverage == "monolingual" and wanted <= set(entry.jurisdictions)
        ]
        if len(candidates) == 1:
            return candidates[0].checkpoint
    return DEFAULT_SBERT_MODEL_NAME


@dataclass(frozen=True)
class BuildSettingsPolicy:
    """Which `TargetIndexBuildSettings` subclass to build, and with what.

    Attributes:
        settings_type: The `TargetIndexBuildSettings` subclass matching the
            resolved representation.
        kwargs: Keyword arguments to pass to `settings_type(...)`. Carried
            as data rather than applied here so a caller/test can inspect
            the resolved construction plan without instantiating it.
    """

    settings_type: type[TargetIndexBuildSettings]
    kwargs: dict[str, object] = field(default_factory=dict)


def resolve_build_settings_policy(
    representation: str,
    *,
    tfidf_ngram_min: int,
    tfidf_ngram_max: int,
    tfidf_analyzer: str = "char_wb",
    sbert_model_name: str = DEFAULT_SBERT_MODEL_NAME,
    sbert_force_untrained_pooling: bool = False,
    encoder: EncoderHandle | None = None,
    encoder_name: str = "",
) -> BuildSettingsPolicy:
    """Pick which `TargetIndexBuildSettings` subclass and kwargs to build.

    Pure selection: resolves the representation-specific subclass and
    normalizes/forwards the relevant subset of the raw config values into
    its constructor kwargs, but never calls the constructor -- see
    `clustering_factory.resolve_target_index_build_settings()` for the
    construction step.

    Args:
        representation: The representation name to build settings for.
        tfidf_ngram_min: `"tfidf"` only; minimum n-gram length.
        tfidf_ngram_max: `"tfidf"` only; maximum n-gram length.
        tfidf_analyzer: `"tfidf"` only; what an n-gram is a run of, one of
            `TFIDF_ANALYZERS`.
        sbert_model_name: `"sbert"` only; a registry slug, a hub
            checkpoint identifier, or a local checkpoint path. A slug is
            resolved to its checkpoint here, so the resolved
            `SbertTargetIndexBuildSettings.model_name` is always something
            `sentence_transformers` can load directly.
        sbert_force_untrained_pooling: `"sbert"` only; load `sbert_model_name`
            even though it declares no sentence-embedding configuration (see
            `sbert_pooling_gate.py`).
        encoder: `"encoder"` only; the already-loaded `EncoderHandle` to
            build the target index over. Required for that representation --
            unlike `sbert_model_name`, this strategy never resolves one
            itself.
        encoder_name: `"encoder"` only; free-text identifier for `encoder`,
            carried through unchanged onto
            `EncoderTargetIndexBuildSettings.encoder_name`.

    Raises:
        ValueError: `representation` is not a known representation name, or
            is `"encoder"` with no `encoder` supplied.
    """
    if representation == "tfidf":
        return BuildSettingsPolicy(
            TfidfTargetIndexBuildSettings,
            {
                "ngram_min": int(tfidf_ngram_min),
                "ngram_max": int(tfidf_ngram_max),
                "analyzer": str(tfidf_analyzer),
            },
        )
    if representation == "wordpiece":
        return BuildSettingsPolicy(WordpieceTargetIndexBuildSettings)
    if representation == "sentencepiece":
        return BuildSettingsPolicy(SentencepieceTargetIndexBuildSettings)
    if representation == "sbert":
        return BuildSettingsPolicy(
            SbertTargetIndexBuildSettings,
            {
                "model_name": resolve_sbert_model_name(sbert_model_name),
                "force_untrained_pooling": bool(sbert_force_untrained_pooling),
            },
        )
    if representation == "encoder":
        if encoder is None:
            raise ValueError(
                "the 'encoder' representation requires encoder=<a loaded EncoderHandle>"
            )
        return BuildSettingsPolicy(
            EncoderTargetIndexBuildSettings,
            {"encoder": encoder, "encoder_name": str(encoder_name)},
        )
    raise ValueError(f"Unsupported representation: {representation}")
