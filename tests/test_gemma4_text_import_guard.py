# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the Gemma 4 text loader fallback."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vllm_mlx.models.gemma4_text import load_gemma4_text


def _write_minimal_gemma4_config(tmp_path: Path) -> Path:
    """Write a tiny config the vendored loader can instantiate cheaply."""
    cfg = {
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 32,
            "vocab_size_per_layer_input": 32,
            "hidden_size_per_layer_input": 0,
            "num_kv_shared_layers": 0,
            "sliding_window_pattern": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_double_wide_mlp": False,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_missing_mlx_vlm_uses_vendored_text_loader(tmp_path, monkeypatch):
    """A bare install without ``mlx_vlm`` must still use vendored Gemma 4
    text classes.

    The current fresh-install contract is stronger than the original
    actionable ImportError: Gemma 4 text-only inference should boot without
    the ``[vision]`` extra. This test forces the upstream import branch to
    fail and asserts the loader gets past class construction to the expected
    local-weight check.
    """
    model_dir = _write_minimal_gemma4_config(tmp_path)

    # Force ``import mlx_vlm`` (and any submodule) to fail even if the
    # package is actually installed in the test environment.
    for mod_name in list(sys.modules):
        if mod_name == "mlx_vlm" or mod_name.startswith("mlx_vlm."):
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def blocking_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mlx_vlm" or name.startswith("mlx_vlm."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", blocking_import)

    with pytest.raises(FileNotFoundError, match="No .safetensors files"):
        load_gemma4_text(model_dir, None)


def test_is_gemma4_model_uses_hf_hub_download_not_snapshot(monkeypatch) -> None:
    """Regression: ``is_gemma4_model`` must fetch only ``config.json`` via
    ``hf_hub_download``, never call ``snapshot_download`` (which would
    pull the entire multi-GB model on every cold ``rapid-mlx serve``
    start to peek at one ~5 KB JSON file).

    Root cause behind PR #600 stress_e2e_bench server-boot timeouts on
    35B / 27B models — the old code paid a ~35 GB Xet revalidation tax
    every time it asked "is this a gemma4 model?".
    """
    from huggingface_hub import hf_hub_download as _real_hf_hub_download  # noqa: F401

    import vllm_mlx.models.gemma4_text as gemma_mod

    called: dict[str, object] = {}

    def fake_hf_hub_download(repo_id: str, filename: str, **kwargs) -> str:
        """Pretend to fetch one file; return a path that doesn't exist
        so the caller falls through to its ``not config_path.exists()``
        branch — we don't care about the read, just the call shape."""
        called["repo_id"] = repo_id
        called["filename"] = filename
        return "/tmp/nonexistent-gemma4-config-test.json"

    def fake_snapshot_download(*args, **kwargs) -> str:
        raise AssertionError(
            "snapshot_download must NOT be called from is_gemma4_model — "
            "it would download the entire model tree just to read config.json. "
            "Use hf_hub_download(repo_id=..., filename='config.json') instead."
        )

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    # Hand in a HF repo id (not a local path) so the cache-miss branch fires.
    gemma_mod.is_gemma4_model("mlx-community/Qwen3.5-35B-A3B-8bit")

    assert called.get("filename") == "config.json", (
        f"Expected hf_hub_download with filename='config.json'; got called={called}"
    )
    assert called.get("repo_id") == "mlx-community/Qwen3.5-35B-A3B-8bit"
