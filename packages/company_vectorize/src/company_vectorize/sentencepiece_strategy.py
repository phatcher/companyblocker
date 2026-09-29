from __future__ import annotations

from .clustering_contract import SentencepieceTargetIndexBuildSettings
from .token_list_strategy import TokenListClusteringStrategy


class SentencepieceClusteringStrategy(TokenListClusteringStrategy):
    representation = "sentencepiece"
    settings_type = SentencepieceTargetIndexBuildSettings
