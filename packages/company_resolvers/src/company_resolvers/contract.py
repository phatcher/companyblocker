"""The `Resolver` contract: the `CandidateSource`, `PairScorer` and `DecisionPolicy` seams it composes.

A resolver takes a name to a decided match through three seams it consumes rather than
implements: `CandidateSource` retrieves ranked candidate ids for a name, `PairScorer` scores a
name against one candidate, and `DecisionPolicy` turns those scores into a decision. None of the
three seams' operating points, thresholds or retrieval mechanics belong here: this package fixes
only the shape they compose through, so a resolver can be built over any selectable strategy's
index, any pair classifier and any threshold profile without this package importing any of them.
Where those operating points come from -- what precision-first or recall-first mean against a
dataset profile -- belongs with the area holding the measurements; `DecisionPolicy` here is a
seam and nothing more, and this module supplies no default policy or threshold.

`ResolutionResult` carries what a resolution was retrieved at, not only what it decided: each
candidate's retrieval score and match score, and the number of comparisons the resolution spent.
A resolver that returned only the decided match and a fixed candidate count would record one
retrieval depth and nothing about what reaching it cost, which is the shape that makes two
strategies incomparable when they retrieve at different depths. Carrying both scores per
candidate and the comparison count lets a cost-normalised recall figure be derived from a
completed resolution rather than by re-running it at other depths.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedCandidate:
    """One candidate id a `CandidateSource` retrieved for a name, with its retrieval score.

    Attributes:
        candidate_id: The candidate entity's identifier.
        retrieval_score: The score the candidate was retrieved at. Higher-is-better is a
            convention every `CandidateSource` implementation is expected to hold, not
            something this contract enforces.
    """

    candidate_id: str
    retrieval_score: float


@dataclass(frozen=True)
class ScoredCandidate:
    """One candidate a resolution considered, carrying both scores it was judged on.

    Attributes:
        candidate_id: The candidate entity's identifier, as returned by a `CandidateSource`.
        retrieval_score: The score the candidate was retrieved at (see `RetrievedCandidate`).
        match_score: The score a `PairScorer` gave this name against this candidate.
    """

    candidate_id: str
    retrieval_score: float
    match_score: float


@dataclass(frozen=True)
class ResolutionResult:
    """What resolving one name against a candidate source, scorer and decision policy spent.

    Attributes:
        name: The name that was resolved.
        candidates: Every candidate considered, ranked by retrieval, each carrying both its
            retrieval score and its match score. A cost-normalised recall figure at any
            retrieval depth up to the `top_k` the resolution ran at is derivable from this
            list alone, without re-running the resolution.
        decision: The candidate id `DecisionPolicy` decided is a match, or `None` if it
            decided no candidate matches.
        comparisons: The number of name/candidate comparisons this resolution spent, i.e. the
            number of `PairScorer` calls it made. Distinct from `len(candidates)`, the number
            of ids a `CandidateSource` retrieved: a resolver or policy that short-circuits
            scoring may spend fewer comparisons than candidates retrieved.
    """

    name: str
    candidates: tuple[ScoredCandidate, ...]
    decision: str | None
    comparisons: int


@runtime_checkable
class CandidateSource(Protocol):
    """Retrieves ranked candidate ids for a name, with the score each was retrieved at.

    Supplies no scoring or decision logic of its own; a resolver is free to draw candidates
    from any index this protocol is satisfied over, such as a blocking strategy's target index.
    """

    def candidates(self, name: str, *, top_k: int) -> Sequence[RetrievedCandidate]:
        """Return up to `top_k` candidates for `name`, each with its retrieval score."""
        ...


@runtime_checkable
class PairScorer(Protocol):
    """Scores one name against one candidate id; carries no notion of ranking or retrieval."""

    def score(self, name: str, candidate_id: str) -> float:
        """Return the match score for `name` against `candidate_id`."""
        ...


@runtime_checkable
class DecisionPolicy(Protocol):
    """Turns a name's ranked, scored candidates into a decision; supplies no thresholds itself.

    Where an operating point comes from, and what precision-first or recall-first mean against
    a dataset profile, is out of scope for this protocol: an implementation supplies that, this
    contract only fixes the shape it is called through.
    """

    def decide(self, name: str, candidates: Sequence[ScoredCandidate]) -> str | None:
        """Return the decided candidate id from `candidates`, or `None` if none matches."""
        ...


@runtime_checkable
class Resolver(Protocol):
    """One call from names to decided matches, composed from the three seams above."""

    def resolve(
        self, names: Sequence[str], *, top_k: int
    ) -> Sequence[ResolutionResult]:
        """Resolve each of `names` to a `ResolutionResult` retrieved at depth `top_k`."""
        ...
