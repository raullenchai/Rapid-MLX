# SPDX-License-Identifier: Apache-2.0
"""Tests for the model alias registry."""

import os

from vllm_mlx.model_aliases import list_aliases, resolve_model


def test_known_alias_resolves():
    assert resolve_model("qwen3.5-9b") == "mlx-community/Qwen3.5-9B-4bit"
    assert (
        resolve_model("qwen3.6-27b")
        == "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed"
    )
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
