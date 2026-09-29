"""Resolver compositions: a name to a decided entity through a candidate source, a scorer and a decision policy."""

from __future__ import annotations

from .contract import (
    CandidateSource,
    DecisionPolicy,
    PairScorer,
    ResolutionResult,
    Resolver,
    RetrievedCandidate,
    ScoredCandidate,
)

__all__ = [
    "CandidateSource",
    "DecisionPolicy",
    "PairScorer",
    "Resolver",
    "ResolutionResult",
    "RetrievedCandidate",
    "ScoredCandidate",
]
