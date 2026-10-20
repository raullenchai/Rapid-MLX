# SPDX-License-Identifier: Apache-2.0
"""Claude Desktop tier discovery stays opt-in and preserves local cards."""

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request


def test_tier_mapping_is_opt_in_and_resolves_documented_aliases(tmp_path, monkeypatch):
    from rapid_mlx.launch import claude_desktop

    monkeypatch.setattr(claude_desktop, "connection_dir", lambda: tmp_path)
    monkeypatch.setattr(
        claude_desktop,
        "_private_write",
        lambda path, content: path.write_text(content, encoding="utf-8"),
    )
    assert claude_desktop.mappings() == {}
    assert claude_desktop.resolve_model("sonnet") == "sonnet"
    monkeypatch.setattr(claude_desktop, "stable_key", lambda: "a" * 64)
    claude_desktop.configure("local-model", opus="other-model")
    assert claude_desktop.mappings() == {
        "opus": "other-model",
        "sonnet": "local-model",
        "haiku": "local-model",
    }
    assert claude_desktop.resolve_model("sonnet") == "local-model"
    assert claude_desktop.resolve_model("claude-opus-rapid-mlx") == "other-model"
    assert claude_desktop.resolve_model("claude-unrelated") == "claude-unrelated"


@pytest.mark.parametrize(
    "content",
    [
        "{invalid",
        "[]",
        '{"opus":"local"}',
        '{"opus":"local","sonnet":1,"haiku":"local"}',
    ],
)
def test_invalid_tier_mapping_is_disabled(tmp_path, monkeypatch, content):
    from rapid_mlx.launch import claude_desktop

    monkeypatch.setattr(claude_desktop, "connection_dir", lambda: tmp_path)
    claude_desktop.mapping_path().write_text(content, encoding="utf-8")
    assert claude_desktop.mappings() == {}


def test_tier_mapping_saves_explicit_gateway_port(tmp_path, monkeypatch):
    from rapid_mlx.launch import claude_desktop

    saved_ports = []
    monkeypatch.setattr(claude_desktop, "connection_dir", lambda: tmp_path)
    monkeypatch.setattr(claude_desktop, "stable_key", lambda: "a" * 64)
    monkeypatch.setattr(claude_desktop, "save_port", saved_ports.append)
    monkeypatch.setattr(
        claude_desktop,
        "_private_write",
        lambda path, content: path.write_text(content, encoding="utf-8"),
    )
    url, _key = claude_desktop.configure("local-model", port=7659)
    assert url == "http://127.0.0.1:7659"
    assert saved_ports == [7659]


def test_discovery_adds_tiers_without_changing_standard_cards(monkeypatch):
    from rapid_mlx.config import get_config
    from rapid_mlx.launch import claude_desktop
    from rapid_mlx.routes import models

    cfg = get_config()
    saved = (
        cfg.model_name,
        cfg.model_alias,
        cfg.model_registry,
        cfg.api_key,
        cfg.tool_call_parser,
    )
    cfg.model_name = "qwen3.5-4b-4bit"
    cfg.model_alias = None
    cfg.model_registry = None
    cfg.api_key = None
    cfg.tool_call_parser = "qwen3_xml"
    app = FastAPI()
    app.include_router(models.router)
    client = TestClient(app)
    try:
        monkeypatch.setattr(claude_desktop, "mappings", lambda: {})
        original = client.get("/v1/models")
        assert original.status_code == 200
        monkeypatch.setattr(
            claude_desktop,
            "mappings",
            lambda: {tier: cfg.model_name for tier in claude_desktop.TIERS},
        )
        enabled = client.get("/v1/models")
        assert enabled.status_code == 200
        original_cards = original.json()["data"]
        enabled_cards = enabled.json()["data"][: len(original_cards)]
        for before, after in zip(original_cards, enabled_cards, strict=True):
            # `created` is generated for every discovery call.
            assert {k: v for k, v in after.items() if k != "created"} == {
                k: v for k, v in before.items() if k != "created"
            }
        tiers = enabled.json()["data"][len(original.json()["data"]) :]
        assert {entry["anthropic_family_tier"] for entry in tiers} == set(
            claude_desktop.TIERS
        )
        assert all(entry["is_family_default"] is True for entry in tiers)
        assert all(entry["id"].startswith("claude-") for entry in tiers)
        monkeypatch.setattr(claude_desktop, "mappings", lambda: {"sonnet": "absent"})
        assert [entry["id"] for entry in client.get("/v1/models").json()["data"]] == [
            entry["id"] for entry in original_cards
        ]
        build_info = models._build_model_info
        monkeypatch.setattr(
            models,
            "_build_model_info",
            lambda model_id: build_info(model_id).model_copy(
                update={"capabilities": ["text"]}
            ),
        )
        monkeypatch.setattr(
            claude_desktop, "mappings", lambda: {"sonnet": cfg.model_name}
        )
        assert [entry["id"] for entry in client.get("/v1/models").json()["data"]] == [
            entry["id"] for entry in original_cards
        ]
    finally:
        (
            cfg.model_name,
            cfg.model_alias,
            cfg.model_registry,
            cfg.api_key,
            cfg.tool_call_parser,
        ) = saved


def test_chat_alias_reaches_model_validation_as_local_model(monkeypatch):
    from rapid_mlx.launch import claude_desktop
    from rapid_mlx.routes import chat

    monkeypatch.setattr(claude_desktop, "mappings", lambda: {"sonnet": "local-model"})
    selected = []

    def stop_after_resolution(model):
        selected.append(model)
        raise RuntimeError("resolved")

    monkeypatch.setattr(chat, "_validate_model_name", stop_after_resolution)
    request = chat.ChatCompletionRequest(
        model="sonnet", messages=[{"role": "user", "content": "hello"}]
    )
    with pytest.raises(RuntimeError, match="resolved"):
        asyncio.run(chat.create_chat_completion(request, None))
    assert selected == ["local-model"]


def test_responses_alias_reaches_model_validation_as_local_model(monkeypatch):
    from rapid_mlx.launch import claude_desktop
    from rapid_mlx.routes import responses

    monkeypatch.setattr(claude_desktop, "mappings", lambda: {"sonnet": "local-model"})
    selected = []

    def stop_after_resolution(model):
        selected.append(model)
        raise RuntimeError("resolved")

    monkeypatch.setattr(responses, "_validate_model_name", stop_after_resolution)

    async def receive():
        return {"type": "http.request", "body": b'{"model":"sonnet","input":"hello"}'}

    request = Request({"type": "http", "headers": []}, receive)
    with pytest.raises(RuntimeError, match="resolved"):
        asyncio.run(responses.create_response(request))
    assert selected == ["local-model"]
