"""Records an optimize run passes around: its policy, its targets and each candidate's result."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OptimizeCandidateResult:
    trainer: str
    seed: int
    vocab_requested: int
    min_frequency: int
    train_rows: int
    validation_rows: int
    resolved_vocab_size: int
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    fertility_generalization_delta: float
    unk_rate_generalization_delta: float
    model_path: str
    rejected: bool
    rejection_reason: str
    selection_score: float
    elapsed_seconds: float
    # The vocabulary the trainer actually produced. `resolved_vocab_size` is
    # what was asked of it, and a `min_frequency` that binds leaves the
    # trained vocabulary short of that. None on a result rebuilt from a run
    # log written before this was recorded.
    trained_vocab_size: int | None = None
    # "v<vocab>_mf<min_frequency>" of the trained candidate this result was
    # carried from when no training was run for it, else None.
    carried_from: str | None = None


@dataclass(frozen=True)
class OptimizeCandidatePolicy:
    """Evaluation policy for `--mode optimize`'s safety-gate-then-score model.

    See the `optimize` module's docstring.

    Attributes:
        fertility_target: Target tokens-per-word fertility that selection
            scoring measures distance from. Not itself a gate.
        unk_rate_threshold: Safety-gate ceiling on unknown-token rate;
            candidates above this are rejected outright.
        fertility_tolerance: Scaling divisor for fertility distance (and for
            fertility train/validation drift) in the selection score. A
            scoring scale, not a gate.
        fertility_min: Safety-gate lower bound on fertility; candidates below
            this are rejected.
        fertility_max: Safety-gate upper bound on fertility; candidates above
            this are rejected.
        token_count_median_max: Not a safety gate: `token_count_median` is
            tokens per name, so its floor is the corpus's words per name
            rather than any property of the candidate. It is still reported
            per candidate, and this field is kept for call-site compatibility.
        token_count_p95_max: Not a safety gate: `token_count_p95` is a step
            function over the vocabulary parameter it would gate, not a
            continuous quality signal, so what a cap on it costs depends on
            where its step falls relative to the fertility elbow rather than
            on any candidate property, and it neither rejects nor scores
            candidates. `token_count_p95` is still reported per candidate;
            this field is retained only so a real threshold can be set from
            that data later.
        eligibility_pass_rate: Minimum fraction of seed runs for a
            ``(vocab_size_requested, min_frequency)`` pair that must pass the
            safety gates for that pair to be eligible for selection.
        diagnostic_top_n: Number of top high-fertility and UNK-containing
            example rows to retain for diagnostics/logging per evaluation.
        delete_rejected_models: Whether to delete a rejected candidate's
            on-disk model artifact immediately after evaluation, rather than
            leaving it in the optimize scratch directory.
    """

    fertility_target: float
    unk_rate_threshold: float
    fertility_tolerance: float
    fertility_min: float
    fertility_max: float
    token_count_median_max: float
    token_count_p95_max: float
    eligibility_pass_rate: float
    diagnostic_top_n: int
    delete_rejected_models: bool


@dataclass(frozen=True)
class TrainingTarget:
    """One scope/systems pairing to train or optimize a tokenizer for.

    Attributes:
        scope: ``"country"`` (one tokenizer per system in ``systems``) or
            ``"global"`` (one combined tokenizer trained across ``systems``).
        systems: System codes involved in this target (for example
            ``["ie", "gb"]``). For ``scope="country"`` these are trained
            independently; for ``scope="global"`` they are the systems
            combined into the single global corpus.
    """

    scope: str
    systems: list[str]
