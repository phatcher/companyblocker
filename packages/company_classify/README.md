# company_classify

Company-name classifiers: whether a text is a valid company name, and whether two names refer to the same entity. It covers the self-supervised pairs they train on, their training, evaluation, comparison and persistence, and pretrained fastText vectors as a baseline ingredient.

## Use

A validity classifier, trained and compared on one split:

```python
from company_classify import LabeledExample, TfidfLogRegClassifier, TfidfMlpClassifier, cross_train, split_examples

examples = [
    LabeledExample(text="Acme Holdings Ltd", label=1),
    LabeledExample(text="this is not a legal entity", label=0),
]
split = split_examples(examples, train_ratio=0.7, validation_ratio=0.15, seed=42)
result = cross_train(split, {"logreg": TfidfLogRegClassifier(), "mlp": TfidfMlpClassifier()})
```

Self-supervised pairs from several producers, all tagged against one shared entity split:

```python
from company_classify import (
    EntityRecord, EntitySplitter, NameShortNamePairProducer, NameVariant,
    RealAliasPairProducer, SplitName, collect_pairs, validate_pairs,
)

splitter = EntitySplitter(seed=42)
records = [EntityRecord(system_uri="gb://00445790", name="ACME SYSTEMS LIMITED", system="gb", country="gb", short_name="ACME")]
variants = [
    NameVariant("gb://00445790", "ACME SYSTEMS LIMITED", "primary"),
    NameVariant("gb://00445790", "ACME SYSTEMS LTD", "previous"),
]
pairs = collect_pairs([
    NameShortNamePairProducer(records=records, splitter=splitter),
    RealAliasPairProducer(variants=variants, splitter=splitter),
])
validate_pairs(pairs, splitter=splitter)
held_out = [pair for pair in pairs if pair.split == SplitName.TEST]
```

## API

Everything is importable from `company_classify`. Each module's docstring gives its rules.

- `schema.py`: `LabeledExample`, `Prediction`, `MetricBundle`, `DatasetSplit`.
- `split.py`: `split_examples`.
- `entity_split.py`: `EntitySplitter`, `SplitName`, `SplitRatios`, `DEFAULT_SPLIT_SEED`.
- `pairs.py`: `PairRecord`, `PairSource`, `EntityRecord`, `NameVariant`, `MATCH_LABEL`, `NON_MATCH_LABEL`, `PairContractError`, `validate_pairs`, `find_contract_violations`, `find_split_leakage`, `find_ambiguous_pairs`, `name_variants_from_rows`.
- `negatives.py`: `NegativeSamplingConfig`, `NegativeStrategy`, `sample_negatives`, `compose_negative_seed`, `lexical_similarity`.
- `pair_producers.py`: `PairProducer`, `SyntheticPairProducer`, `RealAliasPairProducer`, `NameShortNamePairProducer`, `collect_pairs`.
- `models.py`: `ClassifierModel`, `TfidfLogRegClassifier`, `TfidfMlpClassifier`.
- `train.py`: `cross_train`, `TrainedModelResult`.
- `evaluate.py`: `compute_binary_metrics`.
- `inference.py`: `predict_batch`.
- `artifacts.py`: `save_run_summary_json`, `save_comparison_report_json`.
- `pair_models.py`: `PairFeatureMode`, `PairClassifierModel`, `TfidfPairMlpClassifier`.
- `pair_train.py`: `PairDatasetSplit`, `TrainedPairModelResult`, `split_pairs`, `cross_train_pairs`, `pairs_to_labeled_examples`, `pair_split_to_labeled_split`, `cross_train_pair_baseline`.
- `encoder_models.py`: `PooledSubwordContrastiveEncoder`.
- `comparison.py`: `ModelCategory`, `ComparisonReport`, `RepeatabilityResult`, `compare_pair_models`, `check_repeatability`.
- `persistence.py`: `ModelProvenance`, `PersistedModel`, `MODEL_FILENAME`, `MANIFEST_FILENAME`, `save_classifier_model`, `load_classifier_model`, `save_pair_classifier_model`, `load_pair_classifier_model`.
- `checkpoint_selection.py`: `resolve_checkpoint_for_jurisdictions`.
- `fasttext_registry.py`: `FastTextCheckpointEntry`, `DEFAULT_FASTTEXT_CHECKPOINT_SLUG`, `load_fasttext_checkpoint_registry`, `list_fasttext_checkpoints`, `fasttext_checkpoints_for_language`, `resolve_fasttext_checkpoint_entry`, `resolve_fasttext_slug_for_jurisdictions`.
- `token_vector_lookup.py`: `TokenVectorLookup`, `TokenVectorProvenance`, `PretrainedFastTextVectors`, `load_pretrained_fasttext_vectors`, `default_name_tokenizer`, `mean_pool_name_vector`.
- `alias_probe.py`: `AliasProbeResult`, `nearest_neighbour_alias_hit_rate`.

The package resolves no location and downloads nothing: a caller supplies every path, including where a model is saved and which fastText checkpoint file is loaded.
