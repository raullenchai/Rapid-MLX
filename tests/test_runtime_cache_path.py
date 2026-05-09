# SPDX-License-Identifier: Apache-2.0
"""Tests for prefix-cache directory sanitization (issue #194).

The model name flows from ``--model`` / ``--served-model-name`` (arbitrary
user input) into a filesystem path under ``~/.cache/vllm-mlx/prefix_cache/``.
A name containing ``..`` previously resolved to a path *outside* the
prefix-cache root, which is a defense-in-depth gap even if HF repo names
don't permit ``..``.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from vllm_mlx.runtime.cache import get_cache_dir


def _patched_cfg(name: str):
    """Stub config object with model_path/model_name set for the test."""

    class _Cfg:
        model_path = None
        model_name = name
        engine = None

    return _Cfg()


def _resolve(name: str) -> str:
    with patch("vllm_mlx.runtime.cache.get_config", return_value=_patched_cfg(name)):
        return os.path.realpath(get_cache_dir())


def _root() -> str:
    return os.path.realpath(
        os.path.join(os.path.expanduser("~"), ".cache", "vllm-mlx", "prefix_cache")
    )


def test_normal_hf_name_resolves_under_prefix_cache_root():
    """The 99% case: a normal HF org/repo name resolves cleanly with a
    stable hash suffix for collision protection."""
    p = _resolve("mlx-community/Qwen3-0.6B-8bit")
    leaf = os.path.basename(p)
    # ``mlx-community--Qwen3-0.6B-8bit--<8 hex>``
    assert leaf.startswith("mlx-community--Qwen3-0.6B-8bit--")
    assert p.startswith(_root() + os.sep)
    # Same input → same output (deterministic hash).
    assert _resolve("mlx-community/Qwen3-0.6B-8bit") == p


def test_distinct_models_get_distinct_dirs_even_when_sanitized_clash():
    """Different model names whose sanitization collapses to the same
    leaf must still get distinct cache dirs. ``a/b`` and ``a--b`` both
    sanitize to ``a--b`` — only the hash suffix keeps them apart."""
    p_slash = _resolve("a/b")
    p_dash = _resolve("a--b")
    assert p_slash != p_dash, f"sanitized-leaf collision: a/b and a--b both → {p_slash}"
    # Both leaves should still START with the same sanitized prefix.
    assert os.path.basename(p_slash).startswith("a--b--")
    assert os.path.basename(p_dash).startswith("a--b--")


def test_traversal_double_dot_does_not_escape_root():
    """A name with ``..`` must NOT escape the prefix-cache root."""
    p = _resolve("../evil")
    assert p.startswith(_root() + os.sep), (
        f"path traversal escaped prefix-cache root: {p}"
    )


def test_traversal_chained_does_not_escape_root():
    """Multiple ``../`` segments still must stay rooted."""
    p = _resolve("../../../etc/passwd")
    assert p.startswith(_root() + os.sep), (
        f"chained traversal escaped prefix-cache root: {p}"
    )


def test_traversal_mixed_separators_does_not_escape_root():
    """Backslash + forward-slash + ``..`` mixes are sanitized too."""
    p = _resolve("..\\..\\evil")
    assert p.startswith(_root() + os.sep)


def test_leading_dots_stripped():
    """A name beginning with ``.`` must not produce a hidden directory
    that some tools (find, du) silently skip."""
    p = _resolve(".hidden-model")
    leaf = os.path.basename(p)
    assert not leaf.startswith("."), f"hidden leaf would be skipped by tools: {leaf}"


def test_empty_after_sanitization_falls_back_to_default():
    """Pathological input that sanitizes to empty must NOT collapse the
    cache path to the prefix-cache root itself (which would mix entries
    across all models). Fall back to a placeholder leaf, and let the
    hash suffix keep distinct empty-sanitization inputs apart."""
    # ``.`` after lstrip(".") is empty → fallback to "default".
    p_dot = _resolve(".")
    leaf_dot = os.path.basename(p_dot)
    assert leaf_dot.startswith("default--"), (
        f"empty-sanitization must hit 'default' fallback: {leaf_dot!r}"
    )
    assert p_dot.startswith(_root() + os.sep)
    # ``...`` and ``.`` both fall back to "default" prefix but get
    # different hashes from the original raw name → distinct dirs.
    p_three = _resolve("...")
    assert p_three != p_dot, (
        "raw 'default' fallbacks for distinct inputs must keep distinct dirs"
    )


def test_normal_name_unchanged_aside_from_separator_swap_and_hash():
    """Confirm the sanitization does NOT mangle benign characters
    (dots inside a name, hyphens, digits) — only the dangerous patterns.
    The leaf has the sanitized prefix + ``--<hash>``."""
    p = _resolve("mlx-community/gemma-4-26b-a4b-it-4bit")
    leaf = os.path.basename(p)
    assert leaf.startswith("mlx-community--gemma-4-26b-a4b-it-4bit--")
