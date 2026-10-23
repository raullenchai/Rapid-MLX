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


@pytest.fixture(autouse=True)
def isolated_host_snapshot(monkeypatch):
    from rapid_mlx.middleware import browser_security

    monkeypatch.setattr(browser_security, "_configured_hosts", None)


def test_remote_code_policy_requires_opt_in(monkeypatch):
    from rapid_mlx.utils.tokenizer import apply_remote_code_policy

    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    config, enabled = apply_remote_code_policy(None)
    assert config == {"trust_remote_code": False}
    assert enabled is False


@pytest.mark.parametrize("value", ["0", "false", "NO", " off "])
def test_remote_code_opt_out_is_authoritative_and_non_mutating(monkeypatch, value):
    from rapid_mlx.utils.tokenizer import apply_remote_code_policy

    original = {"trust_remote_code": True, "eos_token": "<eos>"}
    monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", value)
    config, enabled = apply_remote_code_policy(original)

    assert enabled is False
    assert config == {"trust_remote_code": False, "eos_token": "<eos>"}
    assert original["trust_remote_code"] is True


def test_mllm_remote_code_optout_reaches_model_and_config_loaders(monkeypatch):
    """The process-wide opt-out must cover every MLLM repository read."""
    import types

    from rapid_mlx.models import mllm as mllm_mod

    monkeypatch.setattr(mllm_mod, "_resolve_checkpoint", lambda name: name)
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
    # This test owns only the remote-code forwarding contract. Keep the three
    # independently-covered GLM compatibility installers out of its synthetic
    # ``mlx_vlm`` module graph so the fake remains intentionally minimal.
    for module_name, installer_name in (
        (
            "rapid_mlx.patches.glm5_next_runtime",
            "install_glm5_next_runtime_fix",
        ),
        (
            "rapid_mlx.patches.glm5_next_processor",
            "install_glm5_next_processor_patch",
        ),
        (
            "rapid_mlx.patches.glm5_next_forget_gate_quant",
            "install_glm5_next_forget_gate_quant_fix",
        ),
    ):
        monkeypatch.setitem(
            sys.modules,
            module_name,
            types.SimpleNamespace(**{installer_name: lambda: None}),
        )
    monkeypatch.setattr(
        "rapid_mlx.utils.tokenizer.augment_eos_token_ids_from_generation_config",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "rapid_mlx.utils.tokenizer.repair_byte_level_decoder",
        lambda *_args, **_kwargs: None,
    )

    instance = mllm_mod.MLXMultimodalLM("org/model")
    instance.load()

    assert calls["model"] == ("org/model", {"trust_remote_code": False})
    assert calls["config"] == ("org/model", {"trust_remote_code": False})


def test_trusted_hosts_default_on_and_cli_comma_normalization(monkeypatch):
    import rapid_mlx.server as server

    monkeypatch.delenv("RAPID_MLX_TRUSTED_HOSTS", raising=False)
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"ok": True}

    monkeypatch.setattr(server, "app", app)
    assert server.configure_trusted_hosts(None) == [
        "localhost",
        "127.0.0.1",
        "[::1]",
        "*.localhost",
    ]
    assert (
        TestClient(app).get("/health", headers={"host": "anything.test"}).status_code
        == 400
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
    import rapid_mlx.server as server

    app = FastAPI()
    monkeypatch.setattr(server, "app", app)
    monkeypatch.setenv("RAPID_MLX_TRUSTED_HOSTS", "localhost, 127.0.0.1")
    assert server.configure_trusted_hosts(None) == ["localhost", "127.0.0.1"]


def test_sa3_checkpoint_refuses_automatic_unsafe_pickle(monkeypatch):
    from rapid_mlx.audio.sa3.models.defs.checkpoint_security import (
        load_torch_checkpoint,
    )

    calls: list[bool] = []

    def fake_load(_path, *, map_location, weights_only):
        assert map_location == "cpu"
        calls.append(weights_only)
        raise pickle.UnpicklingError("legacy pickle")

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(load=fake_load, __version__="2.10.0")
    )
    monkeypatch.delenv("RAPID_MLX_ALLOW_UNSAFE_SA3_PICKLE", raising=False)
    with pytest.raises(RuntimeError, match="Refusing unsafe pickle fallback"):
        load_torch_checkpoint("model.ckpt")
    assert calls == [True]


def test_sa3_checkpoint_unsafe_fallback_requires_explicit_opt_in(monkeypatch):
    from rapid_mlx.audio.sa3.models.defs.checkpoint_security import (
        load_torch_checkpoint,
    )

    calls: list[bool] = []

    def fake_load(_path, *, map_location, weights_only):
        calls.append(weights_only)
        if weights_only:
            raise pickle.UnpicklingError("legacy pickle")
        return {"state_dict": {}}

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(load=fake_load, __version__="2.10.0")
    )
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


@pytest.mark.parametrize(
    "value,requested,expected",
    [
        (None, None, False),
        ("1", None, True),
        (None, True, True),
        ("1", False, False),
        ("0", True, False),
        ("garbage", None, False),
        (None, "false", False),
    ],
)
def test_remote_code_explicit_consent(monkeypatch, value, requested, expected):
    from rapid_mlx.utils.tokenizer import apply_remote_code_policy

    if value is None:
        monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    else:
        monkeypatch.setenv("RAPID_MLX_TRUST_REMOTE_CODE", value)
    config, enabled = apply_remote_code_policy({"trust_remote_code": requested})
    assert enabled is expected
    assert config["trust_remote_code"] is expected


def test_tokenizer_auto_map_cannot_execute_by_default(tmp_path, monkeypatch):
    import json

    from transformers import AutoTokenizer

    from rapid_mlx.utils.tokenizer import apply_remote_code_policy

    marker = tmp_path / "executed"
    (tmp_path / "tokenization_custom.py").write_text(
        f"open({str(marker)!r}, 'w').write('executed')\n"
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "CustomTokenizer",
                "auto_map": {
                    "AutoTokenizer": ["tokenization_custom.CustomTokenizer", None]
                },
            }
        )
    )
    monkeypatch.delenv("RAPID_MLX_TRUST_REMOTE_CODE", raising=False)
    config, _ = apply_remote_code_policy(None)
    with pytest.raises(ValueError):
        AutoTokenizer.from_pretrained(tmp_path, local_files_only=True, **config)
    assert not marker.exists()
