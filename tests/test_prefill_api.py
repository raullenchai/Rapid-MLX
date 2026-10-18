"""Output-free prefix and draft warming contracts."""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rapid_mlx.routes import prefill as api


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
        "persisted": True,
    }
    assert "secret" not in str(response)
    assert engine.calls[0]["max_tokens"] == 1
    assert engine.calls[0]["enable_thinking"] is False
    assert engine.closed == 1
    assert saved == [0]


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
    assert result["persisted"] is False
    assert engine.closed == 2
