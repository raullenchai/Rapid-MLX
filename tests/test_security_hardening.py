# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for PR #2011 security boundaries."""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_remote_code_policy_preserves_historical_unset_default(monkeypatch):
    from vllm_mlx.utils.tokenizer import apply_remote_code_policy

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    config, enabled = apply_remote_code_policy(None)
    assert config is None
    assert enabled is True


@pytest.mark.parametrize("value", ["0", "false", "NO", " off "])
def test_remote_code_opt_out_is_authoritative_and_non_mutating(monkeypatch, value):
    from vllm_mlx.utils.tokenizer import apply_remote_code_policy

    original = {"trust_remote_code": True, "eos_token": "<eos>"}
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", value)
    config, enabled = apply_remote_code_policy(original)

    assert enabled is False
    assert config == {"trust_remote_code": False, "eos_token": "<eos>"}
    assert original["trust_remote_code"] is True


def test_mllm_remote_code_optout_reaches_model_and_config_loaders(monkeypatch):
    """The process-wide opt-out must cover every MLLM repository read."""
    import types

    from vllm_mlx.models import mllm as mllm_mod

    calls = {}
    model = SimpleNamespace(config=SimpleNamespace())
    processor = SimpleNamespace(tokenizer=SimpleNamespace())

    def fake_load(model_name, **kwargs):
        calls["model"] = (model_name, kwargs)
        return model, processor

    def fake_load_config(model_name, **kwargs):
        calls["config"] = (model_name, kwargs)
        return {}

    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", "0")
    monkeypatch.setattr(mllm_mod, "_require_mlx_vlm", lambda: None)
    monkeypatch.setitem(sys.modules, "mlx_vlm", types.SimpleNamespace(load=fake_load))
    monkeypatch.setitem(
        sys.modules,
        "mlx_vlm.utils",
        types.SimpleNamespace(load_config=fake_load_config),
    )
    monkeypatch.setattr(
        "vllm_mlx.utils.tokenizer.augment_eos_token_ids_from_generation_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_mlx.utils.tokenizer.repair_byte_level_decoder",
        lambda *_args, **_kwargs: None,
    )

    instance = mllm_mod.MLXMultimodalLM("org/model")
    instance.load()

    assert calls["model"] == ("org/model", {"trust_remote_code": False})
    assert calls["config"] == ("org/model", {"trust_remote_code": False})


def test_trusted_hosts_default_off_and_cli_comma_normalization(monkeypatch):
    import vllm_mlx.server as server

    monkeypatch.delenv("RAPID_MLX_TRUSTED_HOSTS", raising=False)
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    monkeypatch.setattr(server, "app", app)
    assert server.configure_trusted_hosts(None) == []
    assert (
        TestClient(app).get("/health", headers={"host": "anything.test"}).status_code
        == 200
    )

    app = FastAPI()

    @app.get("/health")
    async def protected_health():
        return {"ok": True}

    monkeypatch.setattr(server, "app", app)
    assert server.configure_trusted_hosts(["localhost,127.0.0.1"]) == [
        "localhost",
        "127.0.0.1",
    ]
    client = TestClient(app)
    assert client.get("/health", headers={"host": "localhost"}).status_code == 200
    assert client.get("/health", headers={"host": "evil.test"}).status_code == 400


def test_trusted_hosts_env_is_used_when_cli_absent(monkeypatch):
    import vllm_mlx.server as server

    app = FastAPI()
    monkeypatch.setattr(server, "app", app)
    monkeypatch.setenv("RAPID_MLX_TRUSTED_HOSTS", "localhost, 127.0.0.1")
    assert server.configure_trusted_hosts(None) == ["localhost", "127.0.0.1"]


def test_sa3_checkpoint_refuses_automatic_unsafe_pickle(monkeypatch):
    from vllm_mlx.audio.sa3.models.defs.checkpoint_security import (
        load_torch_checkpoint,
    )

    calls: list[bool] = []

    def fake_load(_path, *, map_location, weights_only):
        assert map_location == "cpu"
        calls.append(weights_only)
        raise pickle.UnpicklingError("legacy pickle")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))
    monkeypatch.delenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", raising=False)
    with pytest.raises(RuntimeError, match="Refusing unsafe pickle fallback"):
        load_torch_checkpoint("model.ckpt")
    assert calls == [True]


def test_sa3_checkpoint_unsafe_fallback_requires_explicit_opt_in(monkeypatch):
    from vllm_mlx.audio.sa3.models.defs.checkpoint_security import (
        load_torch_checkpoint,
    )

    calls: list[bool] = []

    def fake_load(_path, *, map_location, weights_only):
        calls.append(weights_only)
        if weights_only:
            raise pickle.UnpicklingError("legacy pickle")
        return {"state_dict": {}}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))
    monkeypatch.setenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", "1")
    assert load_torch_checkpoint("model.ckpt") == {"state_dict": {}}
    assert calls == [True, False]


def test_installer_pins_reviewed_python_artifact_digest():
    installer = (Path(__file__).resolve().parents[1] / "install.sh").read_text()
    assert 'PY_BUILD="20260408"' in installer
    assert (
        'PY_SHA256="6000d09545602d3704bdff943f37663b3148b7c1a3a8a1fcc6c1ebd505a3cfc3"'
        in installer
    )
    assert (
        'download "https://github.com/astral-sh/python-build-standalone/releases/download/${PY_BUILD}/SHA256SUMS"'
        not in installer
    )
