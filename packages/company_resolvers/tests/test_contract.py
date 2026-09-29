"""Direct test of the `Resolver` contract: drives a resolver end to end through fake seams.

`FakeResolver` is a test-local composition, not a shipped implementation: a real composed
resolver over a real candidate source, scorer and decision policy ships separately. This test
exists to prove the contract types line up when all three seams are satisfied, and that a
completed `ResolutionResult` carries what it cost to reach, not only what it decided.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from company_resolvers import (
    CandidateSource,
    DecisionPolicy,
    PairScorer,
    ResolutionResult,
    RetrievedCandidate,
    ScoredCandidate,
)


@dataclass
class FakeCandidateSource:
    """Returns a fixed candidate list per name, ignoring `top_k` beyond truncation."""

    by_name: dict[str, list[RetrievedCandidate]]

    def candidates(self, name: str, *, top_k: int) -> Sequence[RetrievedCandidate]:
        return self.by_name.get(name, [])[:top_k]


@dataclass
class FakePairScorer:
    """Returns a fixed match score per (name, candidate_id) pair, counting how many it scored."""

    by_pair: dict[tuple[str, str], float]
    calls: int = 0

    def score(self, name: str, candidate_id: str) -> float:
        self.calls += 1
        return self.by_pair[(name, candidate_id)]


@dataclass
class FakeDecisionPolicy:
    """Decides the highest-scoring candidate a match if it clears a fixed cut."""

    cut: float

    def decide(self, name: str, candidates: Sequence[ScoredCandidate]) -> str | None:
        if not candidates:
            return None
        best = max(candidates, key=lambda c: c.match_score)
        return best.candidate_id if best.match_score >= self.cut else None


@dataclass
class FakeResolver:
    """A minimal `Resolver` composing the three seams, for exercising the contract only."""

    source: CandidateSource
    scorer: PairScorer
    policy: DecisionPolicy

    def resolve(
        self, names: Sequence[str], *, top_k: int
    ) -> Sequence[ResolutionResult]:
        results = []
        for name in names:
            retrieved = self.source.candidates(name, top_k=top_k)
            scored = tuple(
                ScoredCandidate(
                    candidate_id=r.candidate_id,
                    retrieval_score=r.retrieval_score,
                    match_score=self.scorer.score(name, r.candidate_id),
                )
                for r in retrieved
            )
            decision = self.policy.decide(name, scored)
            results.append(
                ResolutionResult(
                    name=name,
                    candidates=scored,
                    decision=decision,
                    comparisons=len(scored),
                )
            )
        return results


def test_resolver_composes_the_three_seams_end_to_end() -> None:
    source = FakeCandidateSource(
        by_name={
            "acme corp": [
                RetrievedCandidate(candidate_id="e1", retrieval_score=0.9),
                RetrievedCandidate(candidate_id="e2", retrieval_score=0.4),
            ],
            "no match co": [
                RetrievedCandidate(candidate_id="e3", retrieval_score=0.2),
            ],
        }
    )
    scorer = FakePairScorer(
        by_pair={
            ("acme corp", "e1"): 0.95,
            ("acme corp", "e2"): 0.3,
            ("no match co", "e3"): 0.1,
        }
    )
    policy = FakeDecisionPolicy(cut=0.5)
    resolver = FakeResolver(source=source, scorer=scorer, policy=policy)

    results = resolver.resolve(["acme corp", "no match co"], top_k=2)

    assert len(results) == 2

    matched = results[0]
    assert matched.name == "acme corp"
    assert matched.decision == "e1"
    assert matched.comparisons == 2
    assert [c.candidate_id for c in matched.candidates] == ["e1", "e2"]
    assert [c.retrieval_score for c in matched.candidates] == [0.9, 0.4]
    assert [c.match_score for c in matched.candidates] == [0.95, 0.3]

    unmatched = results[1]
    assert unmatched.name == "no match co"
    assert unmatched.decision is None
    assert unmatched.comparisons == 1

    # The comparisons carried on each result equal exactly the scorer calls it drove: a
    # cost-normalised figure is derivable per result without re-running at another depth.
    assert scorer.calls == matched.comparisons + unmatched.comparisons


def test_resolution_result_recoverable_at_a_shallower_depth_without_rerunning() -> None:
    """Carrying both scores per candidate means a shallower-depth recall is derivable after the fact."""
    result = ResolutionResult(
        name="acme corp",
        candidates=(
            ScoredCandidate(candidate_id="e1", retrieval_score=0.9, match_score=0.95),
            ScoredCandidate(candidate_id="e2", retrieval_score=0.4, match_score=0.3),
        ),
        decision="e1",
        comparisons=2,
    )

    # Re-deriving "would top_k=1 have found the decided match" needs no re-run: it reads off
    # the already-carried candidates.
    top_1_ids = [c.candidate_id for c in result.candidates[:1]]
    assert result.decision in top_1_ids
