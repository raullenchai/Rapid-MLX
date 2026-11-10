# SPDX-License-Identifier: Apache-2.0
"""Always-reasoning chat budgets, without MLX, network, or model weights."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from rapid_mlx import cli
from rapid_mlx.config import get_config
from rapid_mlx.routes import models


@pytest.fixture(autouse=True)
def isolate_optional_server_stack(monkeypatch):
    # Exercise the model payload without importing the MLX-backed server.
    monkeypatch.setattr(models, "_tools_capable", lambda *_: False)
    monkeypatch.setattr(models, "_audio_lane_snapshot", lambda: None)
    monkeypatch.setattr(models, "_locked_embedding_id", lambda: None)
    monkeypatch.setattr(models, "_locked_rerank_id", lambda: None)
    monkeypatch.setattr(
        models, "effective_parsers_for", lambda *_: (None, "deepseek_r1")
    )


MODEL_EMITTED_THINK = "{% if add_generation_prompt %}{{ '<｜Assistant｜>' }}{% endif %}"
ALWAYS = "{% if add_generation_prompt %}{{ '<think>' }}{% endif %}"
SWITCHABLE = (
    "{% if add_generation_prompt and enable_thinking %}{{ '<think>' }}{% endif %}"
)


@pytest.mark.parametrize(
    "template, parser, expected",
    [
        (MODEL_EMITTED_THINK, "deepseek_r1", True),
        (ALWAYS, "deepseek_r1", True),
        (ALWAYS + "{# enable_thinking #}", "deepseek_r1", True),
        (SWITCHABLE, "qwen3", False),
        (MODEL_EMITTED_THINK, None, False),
        (ALWAYS, None, False),
        ("{{ '<think>old</think>' }}", "deepseek_r1", True),
        ("{% if reasoning %}<think>{% endif %}", "deepseek_r1", False),
        (
            "{% if reasoning %}<think>optional</think>{% endif %}" + ALWAYS,
            "deepseek_r1",
            True,
        ),
        (
            "{% set reasoning = reasoning if reasoning is defined else true %}"
            "{% if add_generation_prompt %}<think>"
            "{% if not reasoning %}</think>{% endif %}{% endif %}",
            "deepseek_r1",
            False,
        ),
        ("{% if enable_thinking", "qwen3", False),
        (None, "deepseek_r1", None),
        ({"default": MODEL_EMITTED_THINK, "tool_use": SWITCHABLE}, "deepseek_r1", True),
        ({"default": SWITCHABLE, "tool_use": ALWAYS}, "qwen3", False),
        ({"only": MODEL_EMITTED_THINK}, "deepseek_r1", True),
        ({"first": ALWAYS, "second": ALWAYS}, "deepseek_r1", None),
        ({}, "deepseek_r1", None),
        (123, "deepseek_r1", None),
    ],
)
def test_server_signal_from_live_template(monkeypatch, template, parser, expected):
    monkeypatch.setattr(models, "effective_parsers_for", lambda *_: (None, parser))
    cfg = get_config()
    monkeypatch.setattr(cfg, "model_registry", None)
    monkeypatch.setattr(cfg, "model_name", "custom-model")
    monkeypatch.setattr(cfg, "model_alias", "custom-alias")
    monkeypatch.setattr(
        cfg,
        "engine",
        SimpleNamespace(tokenizer=SimpleNamespace(chat_template=template)),
    )
    for model_id in ("custom-model", "custom-alias"):
        assert (
            models._build_model_info(model_id).model_dump()["always_reasoning"]
            is expected
        )
    assert models._build_model_info("unloaded-model").always_reasoning is None


def test_server_signal_probe_failure(monkeypatch):
    class BrokenEngine:
        @property
        def tokenizer(self):
            raise RuntimeError("not ready")

    monkeypatch.setattr(models, "_engine_for", lambda _: BrokenEngine())
    assert models._build_model_info("custom-model").always_reasoning is None


@pytest.mark.parametrize(
    "metadata, explicit, think, expected, notice",
    [
        (True, None, False, 4096, "this model always reasons"),
        (True, 2048, False, 2048, None),
        (True, 512, False, 512, None),
        (False, None, False, 2048, None),
        (None, None, False, 2048, None),
        ("unavailable", None, False, 2048, None),
        ("malformed", None, False, 2048, None),
        ("non-object", None, False, 2048, None),
        (True, None, True, 4096, "--think on"),
    ],
)
def test_chat_budget(monkeypatch, capsys, metadata, explicit, think, expected, notice):
    response = Mock()
    response.json.return_value = {
        "data": [
            {"id": "other-model", "always_reasoning": True},
            {"id": "served-alias", "always_reasoning": metadata},
        ]
    }
    if metadata == "unavailable":
        response.raise_for_status.side_effect = requests.ConnectionError("offline")
    elif metadata == "malformed":
        response.json.return_value = {"data": None}
    elif metadata == "non-object":
        response.json.return_value = []
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    inputs = iter(["hello", "exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    monkeypatch.setattr("atexit.register", lambda _: None)
    monkeypatch.setattr("signal.signal", lambda *_: None)
    monkeypatch.setattr(cli, "_stream_chat_response", Mock(return_value=""))
    args = SimpleNamespace(
        base_url="http://unused",
        port=None,
        model="custom-model",
        _original_alias="served-alias",
        system=None,
        think=think,
        max_tokens=explicit,
        temperature=0.0,
        response_timeout=5,
    )
    cli.chat_command(args)
    payload = cli._stream_chat_response.call_args.args[1]
    assert payload["max_tokens"] == expected
    if not think:
        assert payload["enable_thinking"] is False
    out = capsys.readouterr().out
    if notice:
        assert notice in out
        assert "raised --max-tokens to 4096" in out
        assert "pass --max-tokens to override" in out
    else:
        assert "raised --max-tokens" not in out


@pytest.mark.parametrize("path", ["/v1/models", "/v1/models/custom-alias"])
def test_always_reasoning_on_models_wire(monkeypatch, path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    cfg = get_config()
    monkeypatch.setattr(cfg, "model_registry", None)
    monkeypatch.setattr(cfg, "model_name", "custom-model")
    monkeypatch.setattr(cfg, "model_alias", "custom-alias")
    monkeypatch.setattr(cfg, "api_key", None)
    monkeypatch.setattr(
        cfg, "engine", SimpleNamespace(tokenizer=SimpleNamespace(chat_template=ALWAYS))
    )
    app = FastAPI()
    app.include_router(models.router)
    with TestClient(app) as client:
        response = client.get(path)
    assert response.status_code == 200
    payload = response.json()
    entries = payload["data"] if path == "/v1/models" else [payload]
    assert entries
    assert all(entry["always_reasoning"] is True for entry in entries)
