"""Persist a fitted classifier to disk with its provenance, and load it back for scoring.

Nothing elsewhere in this package outlives the process that fitted it: `train.py` and
`pair_train.py` mutate a model in place and hand back the same in-memory instance,
`inference.py`'s `predict_batch()` takes an already-fitted model as an argument. This module is
the one place a fitted model crosses a process boundary.

It attaches to the two model protocols this package already defines (`models.ClassifierModel`,
`pair_models.PairClassifierModel`) rather than to any one concrete classifier, so every
classifier here, single-text and pair alike, gains persistence from the same contract, and a
classifier added later inherits it with no second save/load path to write. The four
`save_*`/`load_*` functions below are thin, protocol-typed wrappers over one private
implementation that only ever pickles whatever object it is given through `joblib`
(`scikit-learn`'s own recommended serialization for its estimators, already resolved
transitively through this package's `scikit-learn` dependency) -- it inspects no concrete
attribute of the model, so it needs no change for a future classifier.

This module resolves no workspace layout and derives no directory structure of its own: the
caller supplies the exact destination (a directory) to save into, or the exact source (the same
kind of directory) to load from, and the two fixed filenames (`MODEL_FILENAME`,
`MANIFEST_FILENAME`) are the only path segments this module ever appends beneath it. Deciding
where that directory lives, and any archival/promotion/reuse policy above one saved model, is
out of scope here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import joblib

from .models import ClassifierModel
from .pair_models import PairClassifierModel

MODEL_FILENAME = "model.joblib"
MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class ModelProvenance:
    """Caller-supplied facts about how a persisted model was fitted.

    Nothing here is derived from the model itself: the model only knows its own
    hyperparameters (already captured by pickling it whole), not what it was trained on or by
    which run, so a caller that just ran `fit()` supplies all four.

    Attributes:
        training_source: What produced the training examples/pairs (for example the
            `PairSource` values mixed into the training split, or a description of a
            single-text `LabeledExample` dataset). Free text; this module does not interpret
            it.
        split: The split identity for the examples/pairs `model.fit()` last ran over (a split
            name, its seed, or an encoded ratio description). Free text for the same reason.
        fitted_params: The hyperparameters that produced this fit -- an already-fitted
            classifier's own configuration (for example `dataclasses.asdict()` of a dataclass
            classifier before `fit()` mutated its internal state) -- not the trained weights,
            which `save_classifier_model()`/`save_pair_classifier_model()` pickle alongside it.
        run_id: Caller-supplied identifier for the training run that produced this fit.
    """

    training_source: str
    split: str
    fitted_params: Mapping[str, object]
    run_id: str


@dataclass(frozen=True)
class PersistedModel[ModelT]:
    """A model loaded back from disk, together with the provenance manifest saved beside it.

    Attributes:
        model: The unpickled, already-fitted model. Ready to score with; not refitted.
        provenance: The `ModelProvenance` supplied to the matching `save_*` call.
        saved_at: UTC timestamp, ISO 8601 format, `save_*` recorded at write time.
        model_class: `f"{module}.{qualname}"` of the saved model's type, recorded at write time
            for a human reading the manifest; loading does not need it, since `joblib` recovers
            the class from the pickle itself.
    """

    model: ModelT
    provenance: ModelProvenance
    saved_at: str
    model_class: str


def _save(model: object, destination: Path, provenance: ModelProvenance) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination / MODEL_FILENAME)

    manifest = {
        "training_source": provenance.training_source,
        "split": provenance.split,
        "fitted_params": dict(provenance.fitted_params),
        "run_id": provenance.run_id,
        "saved_at": datetime.now(UTC).isoformat(),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    (destination / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _load(source: Path) -> PersistedModel[object]:
    model_path = source / MODEL_FILENAME
    manifest_path = source / MANIFEST_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(f"No persisted model at {model_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"No provenance manifest at {manifest_path}")

    model = joblib.load(model_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    provenance = ModelProvenance(
        training_source=manifest["training_source"],
        split=manifest["split"],
        fitted_params=manifest["fitted_params"],
        run_id=manifest["run_id"],
    )
    return PersistedModel(
        model=model,
        provenance=provenance,
        saved_at=manifest["saved_at"],
        model_class=manifest["model_class"],
    )


def save_classifier_model(
    model: ClassifierModel, destination: Path, provenance: ModelProvenance
) -> None:
    """Persist a fitted `ClassifierModel` and its `provenance` beneath `destination`.

    Creates `destination` (and any missing parent) if it does not already exist, then writes
    `MODEL_FILENAME` and `MANIFEST_FILENAME` directly beneath it -- no other path is composed.
    """
    _save(model, destination, provenance)


def load_classifier_model(source: Path) -> PersistedModel[ClassifierModel]:
    """Load a `ClassifierModel` persisted by `save_classifier_model()` from `source`.

    `source` is the same directory `save_classifier_model()` was given as `destination`. The
    returned model is ready to score with via `predict_scores()`; it is not refitted.
    """
    return cast(PersistedModel[ClassifierModel], _load(source))


def save_pair_classifier_model(
    model: PairClassifierModel, destination: Path, provenance: ModelProvenance
) -> None:
    """Persist a fitted `PairClassifierModel` and its `provenance` beneath `destination`.

    Creates `destination` (and any missing parent) if it does not already exist, then writes
    `MODEL_FILENAME` and `MANIFEST_FILENAME` directly beneath it -- no other path is composed.
    """
    _save(model, destination, provenance)


def load_pair_classifier_model(source: Path) -> PersistedModel[PairClassifierModel]:
    """Load a `PairClassifierModel` persisted by `save_pair_classifier_model()` from `source`.

    `source` is the same directory `save_pair_classifier_model()` was given as `destination`.
    The returned model is ready to score with via `predict_scores()`; it is not refitted.
    """
    return cast(PersistedModel[PairClassifierModel], _load(source))
