from pathlib import Path

import pytest
from company_tokenize.tokenization_params import (
    TokenizationRequest,
    TokenizerCalculationSpec,
    coerce_tokenization_request,
    compile_tokenization_request,
)


def test_coerce_tokenization_request_passes_through_dataclass_specs_unchanged():
    spec = TokenizerCalculationSpec(token_col="tokens", trainer="sentencepiece")

    request = coerce_tokenization_request(tokenizer_specs=[spec])

    assert request.tokenizer_calculations == (spec,)
    assert request.noise_words_profile == "none"
    assert request.noise_words_set_kind == "combined"


def test_coerce_tokenization_request_builds_specs_from_mappings_with_request_defaults():
    request = coerce_tokenization_request(
        tokenizer_specs=[{"token_col": "tokens"}],
        noise_words_profile="strict",
        noise_words_set_kind="tokens",
    )

    (calc,) = request.tokenizer_calculations
    assert calc.token_col == "tokens"
    assert calc.trainer == "wordpiece"
    assert calc.noise_words_profile == "strict"
    assert calc.noise_words_set_kind == "tokens"


def test_coerce_tokenization_request_mapping_spec_can_override_request_defaults():
    request = coerce_tokenization_request(
        tokenizer_specs=[
            {"token_col": "tokens", "noise_words_profile": "balanced"},
        ],
        noise_words_profile="strict",
    )

    (calc,) = request.tokenizer_calculations
    assert calc.noise_words_profile == "balanced"


def _spec(
    *,
    token_col: str = "tokens",
    tokenizer_path: str | Path | None = None,
    trainer: str = "wordpiece",
    label: str | None = None,
    noise_words: set[str] | list[str] | tuple[str, ...] | None = None,
    noise_words_profile: str | None = None,
    noise_words_set_kind: str | None = None,
) -> TokenizerCalculationSpec:
    return TokenizerCalculationSpec(
        token_col=token_col,
        tokenizer_path=tokenizer_path,
        trainer=trainer,
        label=label,
        noise_words=noise_words,
        noise_words_profile=noise_words_profile,
        noise_words_set_kind=noise_words_set_kind,
    )


def test_compile_tokenization_request_rejects_blank_name_col():
    request = TokenizationRequest(tokenizer_calculations=(_spec(),), name_col="   ")

    with pytest.raises(ValueError, match="name_col must be non-empty"):
        compile_tokenization_request(request)


def test_compile_tokenization_request_rejects_no_calculations():
    request = TokenizationRequest(tokenizer_calculations=())

    with pytest.raises(ValueError, match="at least one tokenizer specification"):
        compile_tokenization_request(request)


def test_compile_tokenization_request_rejects_blank_token_col():
    request = TokenizationRequest(tokenizer_calculations=(_spec(token_col="  "),))

    with pytest.raises(ValueError, match="non-empty 'token_col'"):
        compile_tokenization_request(request)


def test_compile_tokenization_request_rejects_duplicate_token_cols():
    request = TokenizationRequest(
        tokenizer_calculations=(_spec(token_col="tokens"), _spec(token_col="tokens"))
    )

    with pytest.raises(ValueError, match="Duplicate token column requested"):
        compile_tokenization_request(request)


def test_compile_tokenization_request_rejects_non_sequence_noise_words():
    request = TokenizationRequest(tokenizer_calculations=(_spec(),), noise_words="ltd")

    with pytest.raises(ValueError, match="noise_words must be a set, list, or tuple"):
        compile_tokenization_request(request)


def test_compile_tokenization_request_rejects_non_sequence_calc_noise_words():
    request = TokenizationRequest(tokenizer_calculations=(_spec(noise_words="ltd"),))

    with pytest.raises(
        ValueError, match="Tokenizer spec 'noise_words' must be a set, list, or tuple"
    ):
        compile_tokenization_request(request)


def test_compile_tokenization_request_falls_back_to_request_defaults():
    request = TokenizationRequest(
        tokenizer_calculations=(_spec(),),
        noise_words_profile="strict",
        noise_words_set_kind="tokens",
    )

    compiled = compile_tokenization_request(request)

    (calc,) = compiled.tokenizer_calculations
    assert calc.label == "tokens"
    assert calc.trainer == "wordpiece"
    assert calc.noise_words_profile == "strict"
    assert calc.noise_words_set_kind == "tokens"
    assert compiled.noise_words_profile == "strict"
    assert compiled.noise_words_set_kind == "tokens"


def test_compile_tokenization_request_lets_a_calculation_override_the_default():
    request = TokenizationRequest(
        tokenizer_calculations=(
            _spec(
                token_col="tokens_a",
                trainer=" SentencePiece ",
                label="  custom  ",
                noise_words_profile="balanced",
                noise_words_set_kind="tokens",
                tokenizer_path="artifacts/tok.model",
            ),
            _spec(token_col="tokens_b"),
        ),
        noise_words_profile="strict",
        noise_words_set_kind="combined",
    )

    compiled = compile_tokenization_request(request)

    overridden, defaulted = compiled.tokenizer_calculations
    assert overridden.trainer == "sentencepiece"
    assert overridden.label == "custom"
    assert overridden.noise_words_profile == "balanced"
    assert overridden.noise_words_set_kind == "tokens"
    assert overridden.tokenizer_path == Path("artifacts/tok.model")
    assert defaulted.noise_words_profile == "strict"
    assert defaulted.noise_words_set_kind == "combined"
    assert defaulted.tokenizer_path is None
