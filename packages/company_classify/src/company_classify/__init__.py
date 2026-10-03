"""Company-name classifiers: whether a text is a company name, and whether two names are one entity.

Single-text validity classifiers (`models`), pair classifiers and a pooled-subword encoder
(`pair_models`, `encoder_models`), the self-supervised pairs they train on (`pairs`,
`pair_producers`, `negatives`, `entity_split`), their training, evaluation, comparison and
persistence, and pretrained fastText vectors as a baseline ingredient.
"""

from .alias_probe import AliasProbeResult, nearest_neighbour_alias_hit_rate
from .artifacts import save_comparison_report_json, save_run_summary_json
from .checkpoint_selection import resolve_checkpoint_for_jurisdictions
from .comparison import (
    ComparisonReport,
    ModelCategory,
    RepeatabilityResult,
    check_repeatability,
    compare_pair_models,
)
from .encoder_models import PooledSubwordContrastiveEncoder
from .entity_split import (
    DEFAULT_SPLIT_SEED,
    EntitySplitter,
    SplitName,
    SplitRatios,
)
from .fasttext_registry import (
    DEFAULT_FASTTEXT_CHECKPOINT_SLUG,
    FastTextCheckpointEntry,
    fasttext_checkpoints_for_language,
    list_fasttext_checkpoints,
    load_fasttext_checkpoint_registry,
    resolve_fasttext_checkpoint_entry,
    resolve_fasttext_slug_for_jurisdictions,
)
from .inference import predict_batch
from .models import ClassifierModel, TfidfLogRegClassifier, TfidfMlpClassifier
from .negatives import (
    NegativeSamplingConfig,
    NegativeStrategy,
    compose_negative_seed,
    lexical_similarity,
    sample_negatives,
)
from .pair_models import (
    PairClassifierModel,
    PairFeatureMode,
    TfidfPairMlpClassifier,
)
from .pair_producers import (
    NameShortNamePairProducer,
    PairProducer,
    RealAliasPairProducer,
    SyntheticPairProducer,
    collect_pairs,
)
from .pair_train import (
    PairDatasetSplit,
    TrainedPairModelResult,
    cross_train_pair_baseline,
    cross_train_pairs,
    pair_split_to_labeled_split,
    pairs_to_labeled_examples,
    split_pairs,
)
from .pairs import (
    MATCH_LABEL,
    NON_MATCH_LABEL,
    EntityRecord,
    NameVariant,
    PairContractError,
    PairRecord,
    PairSource,
    find_ambiguous_pairs,
    find_contract_violations,
    find_split_leakage,
    name_variants_from_rows,
    validate_pairs,
)
from .persistence import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    ModelProvenance,
    PersistedModel,
    load_classifier_model,
    load_pair_classifier_model,
    save_classifier_model,
    save_pair_classifier_model,
)
from .schema import DatasetSplit, LabeledExample, MetricBundle, Prediction
from .split import split_examples
from .token_vector_lookup import (
    MeanPooledTokenVectorEncoder,
    PretrainedFastTextVectors,
    TokenVectorLookup,
    TokenVectorProvenance,
    default_name_tokenizer,
    load_pretrained_fasttext_vectors,
    mean_pool_name_vector,
)
from .train import TrainedModelResult, cross_train

__all__ = [
    "DEFAULT_FASTTEXT_CHECKPOINT_SLUG",
    "DEFAULT_SPLIT_SEED",
    "MANIFEST_FILENAME",
    "MATCH_LABEL",
    "MODEL_FILENAME",
    "NON_MATCH_LABEL",
    "AliasProbeResult",
    "ComparisonReport",
    "DatasetSplit",
    "EntityRecord",
    "EntitySplitter",
    "FastTextCheckpointEntry",
    "LabeledExample",
    "MetricBundle",
    "ModelCategory",
    "ModelProvenance",
    "NameShortNamePairProducer",
    "NameVariant",
    "NegativeSamplingConfig",
    "NegativeStrategy",
    "PairContractError",
    "PairDatasetSplit",
    "PairFeatureMode",
    "PairProducer",
    "PairClassifierModel",
    "PairRecord",
    "PairSource",
    "PersistedModel",
    "PooledSubwordContrastiveEncoder",
    "Prediction",
    "MeanPooledTokenVectorEncoder",
    "PretrainedFastTextVectors",
    "RealAliasPairProducer",
    "RepeatabilityResult",
    "SplitName",
    "SplitRatios",
    "SyntheticPairProducer",
    "ClassifierModel",
    "TfidfLogRegClassifier",
    "TfidfMlpClassifier",
    "TfidfPairMlpClassifier",
    "TokenVectorLookup",
    "TokenVectorProvenance",
    "TrainedModelResult",
    "TrainedPairModelResult",
    "check_repeatability",
    "collect_pairs",
    "compare_pair_models",
    "compose_negative_seed",
    "cross_train",
    "cross_train_pair_baseline",
    "cross_train_pairs",
    "default_name_tokenizer",
    "fasttext_checkpoints_for_language",
    "find_ambiguous_pairs",
    "find_contract_violations",
    "find_split_leakage",
    "list_fasttext_checkpoints",
    "load_classifier_model",
    "load_fasttext_checkpoint_registry",
    "load_pair_classifier_model",
    "load_pretrained_fasttext_vectors",
    "mean_pool_name_vector",
    "name_variants_from_rows",
    "nearest_neighbour_alias_hit_rate",
    "lexical_similarity",
    "pair_split_to_labeled_split",
    "pairs_to_labeled_examples",
    "predict_batch",
    "resolve_checkpoint_for_jurisdictions",
    "resolve_fasttext_checkpoint_entry",
    "resolve_fasttext_slug_for_jurisdictions",
    "sample_negatives",
    "save_classifier_model",
    "save_comparison_report_json",
    "save_pair_classifier_model",
    "save_run_summary_json",
    "split_examples",
    "split_pairs",
    "validate_pairs",
]
