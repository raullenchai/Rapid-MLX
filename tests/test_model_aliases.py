# SPDX-License-Identifier: Apache-2.0
"""Tests for the model alias registry."""

import os

from vllm_mlx.model_aliases import list_aliases, resolve_model, suggest_similar


def test_known_alias_resolves():
    assert resolve_model("qwen3.5-9b") == "mlx-community/Qwen3.5-9B-4bit"
    assert resolve_model("llama3-3b") == "mlx-community/Llama-3.2-3B-Instruct-4bit"


def test_full_path_passes_through():
    assert resolve_model("mlx-community/Foo-Bar") == "mlx-community/Foo-Bar"
    assert resolve_model("/Users/me/local-model") == "/Users/me/local-model"


def test_unknown_name_passes_through():
    assert resolve_model("nonexistent-model") == "nonexistent-model"


def test_local_path_takes_priority_over_alias(tmp_path):
    """A local directory matching an alias name should win."""
    local_dir = tmp_path / "qwen3.5-9b"
    local_dir.mkdir()
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert resolve_model("qwen3.5-9b") == "qwen3.5-9b"
    finally:
        os.chdir(old_cwd)


def test_list_aliases_nonempty():
    aliases = list_aliases()
    assert len(aliases) >= 15
    assert "qwen3.5-9b" in aliases


def test_hermes_alias_not_llama():
    """Hermes-3 should be under its own name, not llama3-8b."""
    aliases = list_aliases()
    assert "llama3-8b" not in aliases
    assert "hermes3-8b" in aliases


def test_suggest_similar_stays_within_family():
    """Real reproduction: typing ``deepseek-v4-27b`` (non-existent variant)
    must suggest the deepseek-v4 family — NOT mix in deepseek-r1 which is
    a different model. Generic edit-distance ranking failed this; the
    family-aware filter exists to fix it."""
    suggestions = suggest_similar("deepseek-v4-27b")
    assert suggestions, "expected at least one suggestion"
    # Every suggestion must share the deepseek-v4 family — no deepseek-r1
    # bait-and-switch.
    for s in suggestions:
        assert s.startswith("deepseek-v4"), s


def test_suggest_similar_correctly_typo_for_close_size():
    """Typing ``qwen3.5-30b`` (typo for ``qwen3.5-35b``) should rank the
    correct alias first."""
    suggestions = suggest_similar("qwen3.5-30b")
    assert suggestions, "expected at least one suggestion"
    assert suggestions[0] == "qwen3.5-35b", suggestions


def test_suggest_similar_empty_for_nonsense():
    assert suggest_similar("xyzabc12345") == []


def test_suggest_similar_lets_legitimate_hf_ids_through():
    """Bare HF IDs must NOT match — otherwise the CLI fast-fail in
    ``main()`` would block legitimate single-segment HuggingFace
    repositories like ``gpt2`` and ``bert-base-uncased``."""
    assert suggest_similar("gpt2") == []
    assert suggest_similar("bert-base-uncased") == []


def test_suggest_similar_one_letter_no_match():
    """Single-character inputs must not match anything (would otherwise
    spuriously suggest with cutoff=0.5)."""
    assert suggest_similar("q") == []
    assert suggest_similar("g") == []


def test_suggest_similar_matches_partial_family_token():
    """A bare family name like ``hermes`` should suggest aliases that
    share that prefix (``hermes3-8b``), not return [] just because there's
    no exact ``hermes-foo`` separator pattern."""
    suggestions = suggest_similar("hermes")
    assert "hermes3-8b" in suggestions, suggestions


# --- Letter-only fallback (separator-mismatched names) ----------------


def test_suggest_similar_letter_fallback_handles_separator_mismatch():
    """Real bug from the field: ``rapid-mlx chat gemma4-27b`` returned
    zero suggestions because the strict family parser sees ``gemma4`` and
    no alias starts with ``gemma4`` (we have ``gemma-4-26b`` and
    ``gemma3-27b``). The letter-only fallback must catch this — extract
    ``gemma`` and match the whole gemma family."""
    suggestions = suggest_similar("gemma4-27b")
    assert suggestions, "letter-only fallback must produce gemma family suggestions"
    # All suggestions must be in the gemma family — no llama / qwen leakage.
    for s in suggestions:
        assert s.startswith("gemma"), s


def test_suggest_similar_letter_fallback_collapsed_separator():
    """User collapses our hyphen — ``mistral24b`` should still suggest
    ``mistral-24b``, not return []."""
    assert "mistral-24b" in suggest_similar("mistral24b")


def test_suggest_similar_letter_fallback_skips_legit_looking_names():
    """When the input has no size/quant suffix tokens (i.e., looks
    structurally like a legit single-segment HF repo ID), suggest_similar
    must return [] — not bait-and-switch ``gpt2`` to ``gpt-oss-20b`` or
    ``qwen-coder`` to ``qwen3-coder``. The CLI layer's POPULAR_ALIASES
    fallback handles those cases at presentation time."""
    # ``gpt2`` has been pinned by test_suggest_similar_lets_legitimate_hf_ids_through;
    # this case adds the partial-family equivalent.
    assert suggest_similar("qwen-coder") == []


def test_popular_aliases_curated_list_resolves():
    """Every entry in POPULAR_ALIASES (used as the user-facing
    'try one of these' fallback when zero fuzzy matches) must be a real
    alias in aliases.json — otherwise the error message is a lie."""
    from vllm_mlx.model_aliases import POPULAR_ALIASES

    aliases = list_aliases()
    missing = [a for a in POPULAR_ALIASES if a not in aliases]
    assert not missing, (
        f"POPULAR_ALIASES references non-existent aliases: {missing}. "
        f"Update vllm_mlx/model_aliases.py POPULAR_ALIASES tuple after "
        f"removing or renaming aliases."
    )
