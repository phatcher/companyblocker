import pytest
from company_classify import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    EntitySplitter,
    PairRecord,
    PairSource,
    TfidfLogRegClassifier,
    TfidfMlpClassifier,
    TfidfPairMlpClassifier,
    cross_train_pair_baseline,
    cross_train_pairs,
    pair_split_to_labeled_split,
    pairs_to_labeled_examples,
    split_pairs,
)

SPLITTER = EntitySplitter(seed=42)

_BASE_NAMES = [
    "acme systems",
    "blue river technologies",
    "northshore trading",
    "westfield logistics",
    "meridian group",
    "summit capital partners",
    "cedar valley holdings",
    "ironbridge freight",
    "lighthouse consulting",
    "harbor point ventures",
    "silverline manufacturing",
    "brightwater energy",
    "stonegate insurance",
    "pinecrest logistics",
    "oakridge financial",
    "riverside analytics",
    "crestview partners",
    "amberfield industries",
    "goldenrod capital",
    "eastbrook holdings",
]
_LEGAL_SUFFIXES = ["limited", "ltd", "gmbh", "llc", "plc", "inc"]


def _pairs(count: int = 60) -> list[PairRecord]:
    pairs: list[PairRecord] = []
    for index in range(count):
        base = _BASE_NAMES[index % len(_BASE_NAMES)]
        other_base = _BASE_NAMES[(index + 1) % len(_BASE_NAMES)]
        uri = f"co://{index:04d}"
        other_uri = f"co://{(index + 1) % count:04d}"
        anchor = f"{base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}"
        variant = f"{base} {_LEGAL_SUFFIXES[(index + 1) % len(_LEGAL_SUFFIXES)]}"
        split = SPLITTER.assign(uri)

        pairs.append(
            PairRecord(
                left_name=anchor,
                right_name=variant,
                label=MATCH_LABEL,
                source=PairSource.SYNTHETIC,
                split=split,
                left_system_uri=uri,
                right_system_uri=uri,
            )
        )
        pairs.append(
            PairRecord(
                left_name=anchor,
                right_name=f"{other_base} {_LEGAL_SUFFIXES[index % len(_LEGAL_SUFFIXES)]}",
                label=NON_MATCH_LABEL,
                source=PairSource.SYNTHETIC,
                split=split,
                left_system_uri=uri,
                right_system_uri=other_uri,
            )
        )
    return pairs


def test_split_pairs_partitions_by_the_existing_split_tag() -> None:
    pairs = _pairs()

    split = split_pairs(pairs)

    assert len(split.train) + len(split.validation) + len(split.test) == len(pairs)
    for pair in split.train:
        assert SPLITTER.assign(pair.left_system_uri).value == "train"
    for pair in split.validation:
        assert SPLITTER.assign(pair.left_system_uri).value == "validation"
    for pair in split.test:
        assert SPLITTER.assign(pair.left_system_uri).value == "test"


@pytest.mark.integration
def test_cross_train_pairs_returns_metrics_for_registered_models() -> None:
    split = split_pairs(_pairs())

    results = cross_train_pairs(
        split,
        {
            "pair_mlp": TfidfPairMlpClassifier(
                max_features=512, hidden_layer_sizes=(16,)
            )
        },
    )

    assert set(results.keys()) == {"pair_mlp"}
    for result in results.values():
        assert 0.0 <= result.validation_metrics.f1 <= 1.0
        assert 0.0 <= result.test_metrics.pr_auc <= 1.0
        assert 0.0 <= result.validation_metrics.recall_at_k <= 1.0
        assert 0.0 <= result.test_metrics.candidate_set_size_ratio <= 1.0
        assert 0.0 <= result.validation_metrics.precision_at_k <= 1.0
        assert 0.0 <= result.test_metrics.duplication_rate <= 1.0
        assert result.validation_metrics.latency_per_1k_rows >= 0.0
        assert result.test_metrics.latency_per_1k_rows >= 0.0


def test_cross_train_pairs_rejects_empty_models() -> None:
    split = split_pairs(_pairs())

    with pytest.raises(ValueError):
        cross_train_pairs(split, {})


def test_pairs_to_labeled_examples_concatenates_text_and_carries_the_label() -> None:
    pair = PairRecord(
        left_name="acme systems limited",
        right_name="acme systems ltd",
        label=MATCH_LABEL,
        source=PairSource.SYNTHETIC,
        split=SPLITTER.assign("co://0000"),
        left_system_uri="co://0000",
        right_system_uri="co://0000",
    )

    examples = pairs_to_labeled_examples([pair])

    assert len(examples) == 1
    assert examples[0].label == MATCH_LABEL
    assert examples[0].text == "acme systems limited ||| acme systems ltd"


def test_pair_split_to_labeled_split_preserves_slice_sizes() -> None:
    pair_split = split_pairs(_pairs())

    labeled_split = pair_split_to_labeled_split(pair_split)

    assert len(labeled_split.train) == len(pair_split.train)
    assert len(labeled_split.validation) == len(pair_split.validation)
    assert len(labeled_split.test) == len(pair_split.test)


@pytest.mark.integration
def test_cross_train_pair_baseline_runs_rec00_models_on_the_same_pairs() -> None:
    split = split_pairs(_pairs())

    results = cross_train_pair_baseline(
        split,
        {
            "logreg": TfidfLogRegClassifier(max_features=512),
            "mlp": TfidfMlpClassifier(max_features=512, hidden_layer_sizes=(16,)),
        },
    )

    assert set(results.keys()) == {"logreg", "mlp"}
    for result in results.values():
        assert 0.0 <= result.validation_metrics.f1 <= 1.0
        assert 0.0 <= result.test_metrics.pr_auc <= 1.0
