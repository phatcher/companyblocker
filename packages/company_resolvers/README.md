# company_resolvers

Contracts for resolving a company name to a decided entity: a candidate source retrieves candidates, a pair scorer scores each one, and a decision policy decides. The package ships the contracts only. A caller supplies every implementation, and the package supplies no thresholds, embeddings or retrieval of its own.

## Use

```python
from company_resolvers import (
    CandidateSource,
    DecisionPolicy,
    PairScorer,
    ResolutionResult,
    Resolver,
    RetrievedCandidate,
    ScoredCandidate,
)
```

- `CandidateSource`, `PairScorer` and `DecisionPolicy`: The three protocols a resolver composes.
- `Resolver`: The composition, taking names to one `ResolutionResult` each.
- `RetrievedCandidate`, `ScoredCandidate` and `ResolutionResult`: The records passed between them. A `ResolutionResult` keeps both scores for every candidate and the number of scorer calls spent, so recall at any depth up to the `top_k` used can be derived without running the resolution again.

Each is documented in [src/company_resolvers/contract.py](src/company_resolvers/contract.py). [tests/test_contract.py](tests/test_contract.py) drives a resolver end to end through fakes of all three protocols.
