import pytest
from company_classify import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    EntitySplitter,
    PairFeatureMode,
    PairRecord,
    PairSource,
    TfidfPairMlpClassifier,
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


def _pairs() -> list[PairRecord]:
    """Positives: one base name against a different legal-suffix variant of itself. Negatives:
    that same base name against a different company's name entirely -- easily separable by a
    character-n-gram vectorizer, so the classifier has real signal to learn."""
    pairs: list[PairRecord] = []
    for index, base in enumerate(_BASE_NAMES):
        uri = f"co://{index:04d}"
        other_uri = f"co://{(index + 1) % len(_BASE_NAMES):04d}"
        other_base = _BASE_NAMES[(index + 1) % len(_BASE_NAMES)]
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


@pytest.mark.integration
def test_fit_and_predict_scores_round_trip_with_concat_features() -> None:
    pairs = _pairs()
    model = TfidfPairMlpClassifier(
        max_features=256,
        hidden_layer_sizes=(16,),
        random_state=7,
        feature_mode=PairFeatureMode.CONCAT,
    )
    model.fit(pairs)

    scores = model.predict_scores(
        [
            ("acme systems limited", "acme systems ltd"),
            ("acme systems limited", "eastbrook holdings inc"),
        ]
    )

    assert len(scores) == 2
    for score in scores:
        assert 0.0 <= score <= 1.0


@pytest.mark.integration
def test_fit_and_predict_scores_round_trip_with_difference_features() -> None:
    pairs = _pairs()
    model = TfidfPairMlpClassifier(
        max_features=256,
        hidden_layer_sizes=(16,),
        random_state=7,
        feature_mode=PairFeatureMode.DIFFERENCE,
    )
    model.fit(pairs)

    scores = model.predict_scores(
        [
            ("acme systems limited", "acme systems ltd"),
            ("acme systems limited", "eastbrook holdings inc"),
        ]
    )

    assert len(scores) == 2
    for score in scores:
        assert 0.0 <= score <= 1.0


def test_default_feature_mode_is_concat() -> None:
    assert TfidfPairMlpClassifier().feature_mode is PairFeatureMode.CONCAT


@pytest.mark.integration
def test_classifier_separates_match_from_non_match_pairs() -> None:
    """Not a benchmark -- just confirms the pipeline learns *something* from clearly
    separable training data, catching a broken feature-construction wiring."""
    pairs = _pairs()
    model = TfidfPairMlpClassifier(
        max_features=512, hidden_layer_sizes=(32,), random_state=1
    )
    model.fit(pairs)

    match_score, non_match_score = model.predict_scores(
        [
            ("acme systems limited", "acme systems ltd"),
            ("acme systems limited", "eastbrook holdings inc"),
        ]
    )

    assert match_score > non_match_score
