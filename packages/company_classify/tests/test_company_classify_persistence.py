import json
from pathlib import Path

import pytest
from company_classify import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    LabeledExample,
    ModelProvenance,
    PairRecord,
    PairSource,
    TfidfLogRegClassifier,
    TfidfPairMlpClassifier,
    load_classifier_model,
    load_pair_classifier_model,
    save_classifier_model,
    save_pair_classifier_model,
)
from company_classify.entity_split import SplitName
from company_classify.persistence import MANIFEST_FILENAME, MODEL_FILENAME

_TEXTS = [
    LabeledExample(text="Acme Holdings Ltd", label=1),
    LabeledExample(text="Blue River Technologies GmbH", label=1),
    LabeledExample(text="Northshore Trading LLC", label=1),
    LabeledExample(text="hello world", label=0),
    LabeledExample(text="random sentence only", label=0),
    LabeledExample(text="totally unrelated note", label=0),
]

_PAIRS = [
    PairRecord(
        left_name="acme systems limited",
        right_name="acme systems ltd",
        label=MATCH_LABEL,
        source=PairSource.SYNTHETIC,
        split=SplitName.TRAIN,
        left_system_uri="co://0000",
        right_system_uri="co://0000",
    ),
    PairRecord(
        left_name="acme systems limited",
        right_name="eastbrook holdings inc",
        label=NON_MATCH_LABEL,
        source=PairSource.SYNTHETIC,
        split=SplitName.TRAIN,
        left_system_uri="co://0000",
        right_system_uri="co://0001",
    ),
    PairRecord(
        left_name="blue river technologies",
        right_name="blue river technologies gmbh",
        label=MATCH_LABEL,
        source=PairSource.SYNTHETIC,
        split=SplitName.TRAIN,
        left_system_uri="co://0002",
        right_system_uri="co://0002",
    ),
    PairRecord(
        left_name="blue river technologies",
        right_name="northshore trading plc",
        label=NON_MATCH_LABEL,
        source=PairSource.SYNTHETIC,
        split=SplitName.TRAIN,
        left_system_uri="co://0002",
        right_system_uri="co://0003",
    ),
]

_PROVENANCE = ModelProvenance(
    training_source="unit-test fixture",
    split="train",
    fitted_params={"max_features": 128},
    run_id="test-run-1",
)


def test_save_and_load_classifier_model_scores_identically(tmp_path: Path):
    model = TfidfLogRegClassifier(max_features=128)
    model.fit(_TEXTS)
    texts = ["Acme Holdings", "a plain sentence"]
    expected = model.predict_scores(texts)

    destination = tmp_path / "model"
    save_classifier_model(model, destination, _PROVENANCE)
    loaded = load_classifier_model(destination)

    assert loaded.model.predict_scores(texts) == expected
    assert loaded.provenance == _PROVENANCE
    assert loaded.model_class.endswith("TfidfLogRegClassifier")


@pytest.mark.integration
def test_save_and_load_pair_classifier_model_scores_identically(tmp_path: Path):
    model = TfidfPairMlpClassifier(
        max_features=256, hidden_layer_sizes=(16,), random_state=7
    )
    model.fit(_PAIRS)
    pairs = [
        ("acme systems limited", "acme systems ltd"),
        ("acme systems limited", "eastbrook holdings inc"),
    ]
    expected = model.predict_scores(pairs)

    destination = tmp_path / "pair_model"
    save_pair_classifier_model(model, destination, _PROVENANCE)
    loaded = load_pair_classifier_model(destination)

    assert loaded.model.predict_scores(pairs) == expected
    assert loaded.provenance == _PROVENANCE


def test_save_classifier_model_writes_manifest_beneath_destination_only(
    tmp_path: Path,
):
    model = TfidfLogRegClassifier(max_features=64)
    model.fit(_TEXTS)

    destination = tmp_path / "nested" / "model"
    save_classifier_model(model, destination, _PROVENANCE)

    assert (destination / MODEL_FILENAME).exists()
    assert (destination / MANIFEST_FILENAME).exists()
    assert not (tmp_path / MODEL_FILENAME).exists()

    manifest = json.loads((destination / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["training_source"] == _PROVENANCE.training_source
    assert manifest["split"] == _PROVENANCE.split
    assert manifest["fitted_params"] == dict(_PROVENANCE.fitted_params)
    assert manifest["run_id"] == _PROVENANCE.run_id
    assert "saved_at" in manifest


def test_load_classifier_model_missing_source_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_classifier_model(tmp_path / "does_not_exist")


def test_load_classifier_model_missing_manifest_raises(tmp_path: Path):
    model = TfidfLogRegClassifier(max_features=64)
    model.fit(_TEXTS)

    destination = tmp_path / "model"
    save_classifier_model(model, destination, _PROVENANCE)
    (destination / MANIFEST_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="manifest"):
        load_classifier_model(destination)
