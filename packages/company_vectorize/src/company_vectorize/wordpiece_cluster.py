from __future__ import annotations

from .clustering_contract import WordpieceTargetIndexBuildSettings
from .token_list_strategy import TokenListClusteringStrategy


class WordpieceClusteringStrategy(TokenListClusteringStrategy):
    representation = "wordpiece"
    settings_type = WordpieceTargetIndexBuildSettings
