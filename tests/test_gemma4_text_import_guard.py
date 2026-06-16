# SPDX-License-Identifier: Apache-2.0
"""Regression test for the Gemma 4 loader import guard.

User report: ``rapid-mlx chat gemma-4-12b-qat-8bit`` on a default brew
install (no ``[vision]`` extra) crashed with a raw
``ModuleNotFoundError: No module named 'mlx_vlm'`` because
``load_gemma4_text`` did ``from mlx_vlm.models.gemma4...`` without
guarding. The fix wraps the import in ``try/except ImportError`` and
re-raises with an actionable message pointing at the cheapest install
path (``pip install --no-deps mlx-vlm``, +16 MB).

The brew tap formula now installs mlx-vlm with ``--no-deps``
automatically, so brew users no longer hit this path. PyPI users on the
bare install still do — this test pins the actionable error so the
guard isn't accidentally removed by a future refactor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from vllm_mlx.models.gemma4_text import load_gemma4_text


def _write_minimal_gemma4_config(tmp_path: Path) -> Path:
    """Write a config.json the loader will accept up to the mlx-vlm import."""
    cfg = {
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 256,
            "num_hidden_layers": 1,
            "intermediate_size": 512,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": 32,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tmp_path


def test_missing_mlx_vlm_raises_actionable_error(tmp_path, monkeypatch):
    """Importing ``mlx_vlm`` is the only mlx-vlm contact point during
    load. If the package is absent, the user must see a message that
    names the fix command — NOT a raw ``ModuleNotFoundError`` traceback
    from deep inside the loader."""
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

    with pytest.raises(ImportError) as excinfo:
        load_gemma4_text(model_dir, None)

    msg = str(excinfo.value)
    # Must name BOTH install options so users on a slim brew install
    # can pick the cheap one (--no-deps) and PyPI users can pick the
    # extras one.
    assert "mlx-vlm" in msg
    assert "--no-deps" in msg
    assert "rapid-mlx[vision]" in msg
    # Original ModuleNotFoundError must be chained via ``from e`` so the
    # underlying cause stays visible in the traceback.
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ImportError)


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
