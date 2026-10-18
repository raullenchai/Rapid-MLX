"""Output-free prefix and draft warming contracts."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rapid_mlx.routes import prefill as api
from rapid_mlx.service.helpers import _TOOL_USE_SYSTEM_SUFFIX


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = 0

    async def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        try:
            await self.release.wait()
            yield SimpleNamespace(
                finished=True,
                prompt_tokens=14000,
                cached_tokens=13900,
                text="secret generated text",
            )
        finally:
            self.closed += 1


class ConnectedRequest:
    async def is_disconnected(self):
        return False


def test_prefill_rejects_media_and_disk_drafts():
    with pytest.raises(ValidationError):
        api.PrefillRequest(
            kind="draft",
            messages=[{"role": "user", "content": "hi"}],
            draft_key="a",
            revision=1,
            persist=True,
        )
    with pytest.raises(ValidationError):
        api.PrefillRequest(kind="prefix", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(ValidationError):
        api.PrefillRequest(
            kind="draft",
            messages=[{"role": "user", "content": [{"type": "image_url"}]}],
            draft_key="a",
            revision=1,
        )


@pytest.mark.asyncio
async def test_prefix_prefill_hides_generated_text_and_checkpoints(monkeypatch):
    engine = FakeEngine()
    engine.release.set()
    saved = []
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    monkeypatch.setattr(
        api, "save_prefix_cache_to_disk", lambda budget: saved.append(budget) or True
    )
    body = api.PrefillRequest(
        kind="prefix",
        messages=[{"role": "system", "content": "shared instructions"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        persist=True,
    )
    response = await api._warm(engine, body)
    assert response == {
        "status": "prefilled",
        "prompt_tokens": 14000,
        "cached_tokens": 13900,
        "cache_saved": True,
    }
    assert "secret" not in str(response)
    assert engine.calls[0]["max_tokens"] == 1
    assert engine.calls[0]["enable_thinking"] is False
    assert engine.closed == 1
    assert saved == [30]


def test_prefill_matches_chat_tool_prompt_and_schema(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(
            tool_call_parser="qwen3",
            engine=SimpleNamespace(is_mllm=False, preserve_native_tool_format=False),
        ),
    )
    body = api.PrefillRequest(
        kind="prefix",
        messages=[{"role": "system", "content": "agent instructions"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    )
    messages, tools, thinking = api._render_inputs(body)
    assert messages[0]["content"] == "agent instructions" + _TOOL_USE_SYSTEM_SUFFIX
    assert body.messages[0]["content"] == "agent instructions"
    assert tools == api.convert_tools_for_template(body.tools)
    assert thinking is False


def test_prefill_rejects_unsupported_tool_choice():
    with pytest.raises(ValidationError):
        api.PrefillRequest(
            kind="draft",
            messages=[{"role": "user", "content": "hi"}],
            draft_key="choice",
            revision=1,
            tool_choice="none",
        )


def test_draft_normalizes_developer_and_tool_history(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(
            engine=SimpleNamespace(is_mllm=False, preserve_native_tool_format=False),
            tool_call_parser=None,
            reasoning_parser_name="qwen3",
            no_thinking=False,
        ),
    )
    body = api.PrefillRequest(
        kind="draft",
        messages=[
            {"role": "developer", "content": "Use tools carefully"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "value"},
            {"role": "user", "content": "continue"},
        ],
        draft_key="history",
        revision=1,
    )
    messages, _, thinking = api._render_inputs(body)
    assert messages[0] == {"role": "system", "content": "Use tools carefully"}
    assert all(message["role"] != "tool" for message in messages)
    assert any("Tool Result" in message["content"] for message in messages)
    assert thinking is False


def test_mllm_draft_drops_null_tool_content_like_chat(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(
            engine=SimpleNamespace(is_mllm=True, preserve_native_tool_format=True),
            tool_call_parser=None,
        ),
    )
    body = api.PrefillRequest(
        kind="draft",
        messages=[
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "value"},
            {"role": "user", "content": "continue"},
        ],
        draft_key="mllm-history",
        revision=1,
    )
    messages, _, _ = api._render_inputs(body)
    assert "content" not in messages[0]
    assert messages[0]["tool_calls"][0]["function"]["arguments"] == {}


@pytest.mark.asyncio
async def test_new_draft_revision_cancels_previous_and_keeps_disk_clean(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(
        api, "save_prefix_cache_to_disk", lambda _: pytest.fail("draft wrote to disk")
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 100
    )
    one = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "partial"}],
        draft_key="chat",
        revision=1,
    )
    two = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "partial text"}],
        draft_key="chat",
        revision=2,
    )
    first = asyncio.create_task(api.prefill(one, ConnectedRequest()))
    await engine.started.wait()
    second = asyncio.create_task(api.prefill(two, ConnectedRequest()))
    with pytest.raises(asyncio.CancelledError):
        await first
    for _ in range(100):
        if len(engine.calls) == 2:
            break
        await asyncio.sleep(0.01)
    assert len(engine.calls) == 2
    engine.release.set()
    result = await second
    assert result["cache_saved"] is False
    assert engine.closed == 2
    with pytest.raises(HTTPException) as stale:
        await api.prefill(one, ConnectedRequest())
    assert stale.value.status_code == 409


@pytest.mark.asyncio
async def test_draft_requires_disk_consent_before_engine_work(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "unsent"}],
        draft_key="consent-test",
        revision=1,
    )
    with pytest.raises(HTTPException) as error:
        await api.prefill(body, ConnectedRequest())
    assert error.value.status_code == 409
    assert engine.calls == []


@pytest.mark.asyncio
async def test_context_rejection_happens_before_prefill(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)

    def too_long(*args, **kwargs):
        raise HTTPException(status_code=400, detail="context_length_exceeded")

    monkeypatch.setattr(api, "enforce_context_length_for_messages", too_long)
    body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "large draft"}],
        draft_key="context-test",
        revision=1,
    )
    with pytest.raises(HTTPException) as error:
        await api.prefill(body, ConnectedRequest())
    assert error.value.status_code == 400
    assert engine.calls == []


@pytest.mark.asyncio
async def test_handler_cancellation_waits_for_engine_cleanup(monkeypatch):
    class SlowCleanupEngine(FakeEngine):
        async def stream_chat(self, **kwargs):
            self.calls.append(kwargs)
            self.started.set()
            try:
                await self.release.wait()
                yield SimpleNamespace(finished=True, prompt_tokens=1, cached_tokens=0)
            finally:
                await asyncio.sleep(0.05)
                self.closed += 1

    engine = SlowCleanupEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 1
    )
    body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "partial"}],
        draft_key="handler-cancel-test",
        revision=1,
    )
    handler = asyncio.create_task(api.prefill(body, ConnectedRequest()))
    await engine.started.wait()
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    assert engine.closed == 1
    assert api._active is None
