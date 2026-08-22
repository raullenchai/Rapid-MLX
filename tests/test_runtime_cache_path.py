# SPDX-License-Identifier: Apache-2.0
"""Tests for prefix-cache directory sanitization (issue #194).

The model name flows from ``--model`` / ``--served-model-name`` (arbitrary
user input) into a filesystem path under ``~/.cache/rapid-mlx/prefix_cache/``.
A name containing ``..`` previously resolved to a path *outside* the
prefix-cache root, which is a defense-in-depth gap even if HF repo names
don't permit ``..``.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from vllm_mlx.runtime.cache import (
    _cached_model_revision,
    _resolved_model_source,
    get_cache_dir,
    pin_prefix_cache_identity,
)


def _patched_cfg(name: str, kv_cache_dtype: str = "bf16"):
    """Stub config object with model_path/model_name set for the test."""

    return SimpleNamespace(
        model_path=None,
        model_name=name,
        engine=None,
        kv_cache_dtype=kv_cache_dtype,
    )


def _resolve(name: str) -> str:
    with patch("vllm_mlx.runtime.cache.get_config", return_value=_patched_cfg(name)):
        return os.path.realpath(get_cache_dir())


def _resolve_with_dtype(name: str, dtype: str) -> str:
    with (
        patch(
            "vllm_mlx.runtime.cache.get_config",
            return_value=_patched_cfg(name, dtype),
        ),
        patch("vllm_mlx.runtime.cache._cached_model_revision", return_value="rev-a"),
    ):
        return os.path.realpath(get_cache_dir())


def _root() -> str:
    return os.path.realpath(
        os.path.join(os.path.expanduser("~"), ".cache", "rapid-mlx", "prefix_cache")
    )


def test_normal_hf_name_resolves_under_prefix_cache_root():
    """The 99% case: a normal HF org/repo name resolves cleanly with a
    stable hash suffix for collision protection."""
    p = _resolve("mlx-community/Qwen3-0.6B-8bit")
    leaf = os.path.basename(p)
    # ``mlx-community--Qwen3-0.6B-8bit--<16 hex>``
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


def test_distinct_kv_dtypes_get_distinct_cache_dirs():
    assert _resolve_with_dtype("org/model", "bf16") != _resolve_with_dtype(
        "org/model", "int8"
    )


def test_live_scheduler_dtype_wins_over_server_config_fallback():
    cfg = _patched_cfg("org/model", "bf16")
    cfg.engine = SimpleNamespace(
        scheduler=SimpleNamespace(config=SimpleNamespace(kv_cache_dtype="int8"))
    )
    with (
        patch("vllm_mlx.runtime.cache.get_config", return_value=cfg),
        patch("vllm_mlx.runtime.cache._cached_model_revision", return_value="rev-a"),
    ):
        programmatic = get_cache_dir()
    assert os.path.basename(programmatic) == os.path.basename(
        _resolve_with_dtype("org/model", "int8")
    )
    assert os.path.basename(programmatic) != os.path.basename(
        _resolve_with_dtype("org/model", "bf16")
    )


def test_distinct_model_revisions_get_distinct_cache_dirs():
    cfg = _patched_cfg("org/model", "int8")
    with patch("vllm_mlx.runtime.cache.get_config", return_value=cfg):
        with patch(
            "vllm_mlx.runtime.cache._cached_model_revision", return_value="rev-a"
        ):
            first = get_cache_dir()
        with patch(
            "vllm_mlx.runtime.cache._cached_model_revision", return_value="rev-b"
        ):
            second = get_cache_dir()
    assert first != second


def test_loaded_engine_pins_revision_identity_for_load_and_save():
    cfg = _patched_cfg("org/model", "int8")
    cfg.engine = SimpleNamespace(
        scheduler=SimpleNamespace(config=SimpleNamespace(kv_cache_dtype="int8"))
    )
    with (
        patch("vllm_mlx.runtime.cache.get_config", return_value=cfg),
        patch(
            "vllm_mlx.runtime.cache._cached_model_revision",
            side_effect=["revision-at-load", "revision-after-external-update"],
        ) as revision,
    ):
        load_path = get_cache_dir()
        save_path = get_cache_dir()
    assert load_path == save_path
    revision.assert_called_once_with("org/model")


def test_explicit_engine_pin_never_rereads_mutable_remote_ref():
    engine = SimpleNamespace()
    pin_prefix_cache_identity(
        engine,
        raw_model_name="org/model",
        checkpoint_source="/cache/snapshots/immutable-revision",
        kv_dtype="int8",
    )
    cfg = _patched_cfg("org/model", "int8")
    cfg.engine = engine
    with (
        patch("vllm_mlx.runtime.cache.get_config", return_value=cfg),
        patch(
            "vllm_mlx.runtime.cache._cached_model_revision",
            side_effect=AssertionError("must use the pre-pinned identity"),
        ),
    ):
        assert get_cache_dir() == get_cache_dir()


def test_new_engine_recaptures_updated_revision_identity():
    cfg = _patched_cfg("org/model", "int8")
    cfg.engine = SimpleNamespace(
        scheduler=SimpleNamespace(config=SimpleNamespace(kv_cache_dtype="int8"))
    )
    with (
        patch("vllm_mlx.runtime.cache.get_config", return_value=cfg),
        patch(
            "vllm_mlx.runtime.cache._cached_model_revision",
            side_effect=["revision-a", "revision-b"],
        ),
    ):
        first = get_cache_dir()
        cfg.engine = SimpleNamespace(
            scheduler=SimpleNamespace(config=SimpleNamespace(kv_cache_dtype="int8"))
        )
        second = get_cache_dir()
    assert first != second


def test_builtin_alias_resolves_to_underlying_hf_repo():
    assert (
        _resolved_model_source("deepseek-r1-32b-4bit")
        == "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit"
    )


def test_local_checkpoint_fingerprint_changes_when_weights_change(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type":"test"}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first")
    first = _cached_model_revision(str(tmp_path))

    weights.write_bytes(b"replacement-weights")
    second = _cached_model_revision(str(tmp_path))
    assert first != second


def test_local_checkpoint_same_size_preserved_mtime_still_invalidates(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"first-weights")
    original = weights.stat()
    first = _cached_model_revision(str(tmp_path))

    weights.write_bytes(b"other-weights")  # same byte length
    os.utime(weights, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert weights.stat().st_size == original.st_size
    assert weights.stat().st_mtime_ns == original.st_mtime_ns
    second = _cached_model_revision(str(tmp_path))
    assert first != second


def test_local_indexed_nested_shard_change_invalidates(tmp_path):
    shard_dir = tmp_path / "weights"
    shard_dir.mkdir()
    shard = shard_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"first-shard")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": str(shard.relative_to(tmp_path))}})
    )
    first = _cached_model_revision(str(tmp_path))

    shard.write_bytes(b"replacement-shard")
    second = _cached_model_revision(str(tmp_path))
    assert first != second


def test_local_custom_model_code_change_invalidates(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "custom", "model_file": "code/model.py"})
    )
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    model_file = code_dir / "model.py"
    model_file.write_text("MODEL_VERSION = 1\n")
    (tmp_path / "model.safetensors").write_bytes(b"unchanged-weights")
    first = _cached_model_revision(str(tmp_path))

    model_file.write_text("MODEL_VERSION = 2\n")
    second = _cached_model_revision(str(tmp_path))
    assert first != second


def test_local_custom_model_arbitrary_asset_change_invalidates(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "custom", "model_file": "model.py"})
    )
    (tmp_path / "model.py").write_text("# reads params.table\n")
    asset = tmp_path / "params.table"
    asset.write_bytes(b"semantic-version-one")
    first = _cached_model_revision(str(tmp_path))

    asset.write_bytes(b"semantic-version-two")
    second = _cached_model_revision(str(tmp_path))
    assert first != second


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
