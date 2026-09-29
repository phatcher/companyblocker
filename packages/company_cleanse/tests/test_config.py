from __future__ import annotations

import pytest
from company_cleanse import config


class _FakeResourceFile:
    def __init__(self, text: str) -> None:
        self._text = text

    def read_text(self, encoding: str = "utf-8") -> str:
        _ = encoding
        return self._text


class _FakeResourcePackage:
    def __init__(self, text: str) -> None:
        self._text = text

    def joinpath(self, name: str) -> _FakeResourceFile:
        assert name in {"manual_noise_words.json", "noise_words.json"}
        return _FakeResourceFile(self._text)


def _patch_noise_words_json(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    monkeypatch.setattr(config, "files", lambda _: _FakeResourcePackage(payload))


def test_get_manual_noise_words_returns_expected_scopes():
    tokens = config.get_manual_noise_words()
    assert set(tokens.keys()) == {"suffix", "anywhere"}
    assert isinstance(tokens["suffix"], tuple)
    assert isinstance(tokens["anywhere"], tuple)


def test_load_manual_noise_words_by_scope_rejects_non_object(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(monkeypatch, '["x"]')
    with pytest.raises(TypeError, match="must be an object"):
        config._load_manual_noise_words_by_scope()


def test_load_manual_noise_words_by_scope_rejects_unknown_keys(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(monkeypatch, '{"suffix": [], "bad": []}')
    with pytest.raises(ValueError, match="only supports 'suffix', 'anywhere'"):
        config._load_manual_noise_words_by_scope()


def test_load_manual_noise_words_by_scope_allows_optional_provenance_key(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(
        monkeypatch,
        '{"suffix": ["LTD"], "anywhere": [], "provenance": {"kind": "manual"}}',
    )
    scoped = config._load_manual_noise_words_by_scope()
    assert scoped == {"suffix": ("LTD",), "anywhere": ()}


def test_load_manual_noise_words_by_scope_rejects_non_list_values(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(monkeypatch, '{"suffix": "x", "anywhere": []}')
    with pytest.raises(TypeError, match="values must be lists"):
        config._load_manual_noise_words_by_scope()


def test_load_manual_noise_words_by_scope_rejects_non_string_suffix(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(monkeypatch, '{"suffix": [1], "anywhere": []}')
    with pytest.raises(ValueError, match="'suffix' list entries must be strings"):
        config._load_manual_noise_words_by_scope()


def test_load_manual_noise_words_by_scope_rejects_non_string_anywhere(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(monkeypatch, '{"suffix": [], "anywhere": [1]}')
    with pytest.raises(ValueError, match="'anywhere' list entries must be strings"):
        config._load_manual_noise_words_by_scope()


def test_load_manual_noise_words_rejects_invalid_scope(
    monkeypatch: pytest.MonkeyPatch,
):
    original = config._MANUAL_NOISE_WORDS_BY_SCOPE
    try:
        monkeypatch.setattr(
            config,
            "_MANUAL_NOISE_WORDS_BY_SCOPE",
            {"suffix": ({"token": "A", "scope": "invalid"},), "anywhere": ()},
        )
        with pytest.raises(ValueError, match="scope must be 'suffix' or 'anywhere'"):
            config._load_manual_noise_words()
    finally:
        monkeypatch.setattr(config, "_MANUAL_NOISE_WORDS_BY_SCOPE", original)


def test_load_manual_noise_words_rejects_invalid_entry_type(
    monkeypatch: pytest.MonkeyPatch,
):
    original = config._MANUAL_NOISE_WORDS_BY_SCOPE
    try:
        monkeypatch.setattr(
            config,
            "_MANUAL_NOISE_WORDS_BY_SCOPE",
            {"suffix": (123,), "anywhere": ()},
        )
        with pytest.raises(ValueError, match="entries must be strings or objects"):
            config._load_manual_noise_words()
    finally:
        monkeypatch.setattr(config, "_MANUAL_NOISE_WORDS_BY_SCOPE", original)


def test_cleanse_config_defaults_are_stable():
    cfg = config.CleanseConfig()
    assert cfg.company_col == "company_name"
    assert cfg.company_type_matcher == "trie"
    assert cfg.source_company_type_col is None
    assert cfg.noise_words_profile == "aggressive"
    assert cfg.noise_words_set_kind == "combined"
    assert cfg.short_name_profile == "default"


def test_get_profiled_noise_words_returns_combined_values():
    tokens = config.get_profiled_noise_words()
    assert isinstance(tokens, tuple)
    assert "ltd" in {token.lower() for token in tokens}


def test_get_profiled_noise_words_returns_specific_set_kind():
    tokens = config.get_profiled_noise_words(
        noise_words_profile="balanced",
        noise_words_set_kind="tokens",
    )
    assert isinstance(tokens, tuple)
    assert "services" in {token.lower() for token in tokens}


def test_get_effective_noise_words_combines_scoped_and_profiled_sets():
    tokens = config.get_effective_noise_words(
        noise_words_profile="balanced",
        noise_words_set_kind="combined",
    )
    normalized = {
        str(token).lower()
        if isinstance(token, str)
        else str(token.get("token", "")).lower()
        for token in tokens
    }
    assert "systems" in normalized
    assert "consulting" in normalized


def test_load_profiled_noise_words_payload_rejects_non_object(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(monkeypatch, '["x"]')
    with pytest.raises(TypeError, match="noise_words.json must be an object"):
        config._load_profiled_noise_words_payload()


def test_load_profiled_noise_words_payload_requires_profiles_object(
    monkeypatch: pytest.MonkeyPatch,
):
    _patch_noise_words_json(monkeypatch, '{"profiles": []}')
    with pytest.raises(TypeError, match="must include an object field 'profiles'"):
        config._load_profiled_noise_words_payload()


def test_resolve_profile_cumulative_tokens_handles_custom_and_skips_invalid_profiles():
    profiles_obj = {
        "strict": {"seed_legal": ["LTD"], "tokens": ["GROUP"]},
        "balanced": "skip-me",
        "custom": {"seed_legal": ["LLC"], "tfidf_descriptor": ["HOLDINGS"]},
    }

    assert config._resolve_profile_cumulative_tokens(
        profiles_obj=profiles_obj, profile_key="strict"
    ) == ("LTD", "GROUP")
    assert config._resolve_profile_cumulative_tokens(
        profiles_obj=profiles_obj, profile_key="custom"
    ) == ("LLC", "HOLDINGS")


def test_get_profiled_noise_words_rejects_unknown_profile():
    with pytest.raises(TypeError, match="profile 'missing' not found"):
        config.get_profiled_noise_words(noise_words_profile="missing")


def test_get_profiled_noise_words_rejects_unknown_set_kind(
    monkeypatch: pytest.MonkeyPatch,
):
    original = config._DEFAULT_PROFILED_NOISE_WORDS_PAYLOAD
    monkeypatch.setattr(
        config,
        "_DEFAULT_PROFILED_NOISE_WORDS_PAYLOAD",
        {"profiles": {"strict": {"seed_legal": ["LTD"]}}},
    )
    try:
        with pytest.raises(TypeError, match="set kind 'unknown' not found"):
            config.get_profiled_noise_words(
                noise_words_profile="strict", noise_words_set_kind="unknown"
            )
    finally:
        monkeypatch.setattr(config, "_DEFAULT_PROFILED_NOISE_WORDS_PAYLOAD", original)


def test_validate_profiled_noise_words_payload_accepts_well_formed_payload():
    payload = {
        "profiles": {
            "strict": {"seed_legal": ["LTD"], "tokens": ["GROUP"]},
            "balanced": {"tfidf_descriptor": ["SERVICES"]},
        }
    }
    assert config.validate_profiled_noise_words_payload(payload) is payload


def test_validate_profiled_noise_words_payload_rejects_non_object():
    with pytest.raises(TypeError, match="candidate.json must be an object"):
        config.validate_profiled_noise_words_payload(
            ["x"], source_label="candidate.json"
        )


def test_validate_profiled_noise_words_payload_requires_profiles_object():
    with pytest.raises(TypeError, match="must include an object field 'profiles'"):
        config.validate_profiled_noise_words_payload({"profiles": []})


def test_validate_profiled_noise_words_payload_requires_a_known_profile_key():
    with pytest.raises(ValueError, match="must include at least one of"):
        config.validate_profiled_noise_words_payload(
            {"profiles": {"custom": {"tokens": ["X"]}}}
        )


def test_validate_profiled_noise_words_payload_rejects_non_object_profile():
    with pytest.raises(TypeError, match="profile 'strict' must be an object"):
        config.validate_profiled_noise_words_payload(
            {"profiles": {"strict": "not-an-object"}}
        )


def test_validate_profiled_noise_words_payload_rejects_non_string_tokens():
    with pytest.raises(ValueError, match="'tokens'/'tfidf_descriptor' entries"):
        config.validate_profiled_noise_words_payload(
            {"profiles": {"strict": {"tokens": [1, 2]}}}
        )


def test_validate_profiled_noise_words_payload_rejects_non_list_seed_legal():
    with pytest.raises(ValueError, match="'seed_legal' must be a list of strings"):
        config.validate_profiled_noise_words_payload(
            {"profiles": {"strict": {"tokens": ["X"], "seed_legal": "LTD"}}}
        )


def test_summarize_profiled_noise_word_counts_counts_tokens_and_seed_legal():
    payload = {
        "profiles": {
            "strict": {"seed_legal": ["LTD"], "tokens": ["GROUP", "HOLDING"]},
            "balanced": {"tfidf_descriptor": ["SERVICES"]},
            "empty": "not-an-object",
        }
    }
    assert config.summarize_profiled_noise_word_counts(payload) == {
        "strict": 3,
        "balanced": 1,
    }


def test_summarize_profiled_noise_word_counts_handles_missing_profiles_key():
    assert config.summarize_profiled_noise_word_counts({}) == {}


def test_get_effective_noise_words_deduplicates_profiled_suffix_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        config,
        "MANUAL_NOISE_WORDS",
        ("systems", {"token": "alpha", "scope": "anywhere"}),
    )
    monkeypatch.setattr(
        config, "get_profiled_noise_words", lambda **_: ("systems", "holdings")
    )

    tokens = config.get_effective_noise_words()
    assert tokens.count("systems") == 1
    assert "holdings" in tokens
