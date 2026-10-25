"""Output-free prefix and draft warming contracts."""

import asyncio
import threading
from collections import OrderedDict
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from rapid_mlx.routes import prefill as api
from rapid_mlx.runtime import cache as runtime_cache
from rapid_mlx.service.helpers import _TOOL_USE_SYSTEM_SUFFIX


@pytest.fixture(autouse=True)
def isolate_prefill_guard(monkeypatch):
    # Contended locks bind to a loop; each async test uses a fresh loop.
    monkeypatch.setattr(api, "_guard", asyncio.Lock())


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


@pytest.mark.parametrize("missing_engine", [True, False])
def test_unavailable_cache_reports_no_disk_checkpoint(monkeypatch, missing_engine):
    engine = None if missing_engine else SimpleNamespace()
    if engine is not None:
        setattr(engine, runtime_cache._AUTO_LOAD_SKIPPED_ATTR, True)
    monkeypatch.setattr(
        runtime_cache, "get_config", lambda: SimpleNamespace(engine=engine)
    )
    monkeypatch.setattr(
        runtime_cache,
        "get_cache_dir",
        lambda: pytest.fail("skipped restore must preserve the disk snapshot"),
    )

    assert runtime_cache.save_prefix_cache_to_disk(budget_sec=0) is False


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


@pytest.mark.parametrize(
    "changes",
    [
        {"messages": [{"role": "alien", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "x" * 500_001}]},
        {
            "kind": "prefix",
            "messages": [{"role": "system", "content": "x"}],
            "draft_key": "draft",
        },
        {"messages": [{"role": "assistant", "content": "done"}]},
        {"draft_key": None},
    ],
)
def test_prefill_rejects_unsafe_or_incomplete_shapes(changes):
    data = {
        "kind": "draft",
        "messages": [{"role": "user", "content": "hi"}],
        "draft_key": "shape",
        "revision": 1,
    }
    data.update(changes)
    with pytest.raises(ValidationError):
        api.PrefillRequest(**data)


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


@pytest.mark.asyncio
async def test_cancelled_stream_that_returns_normally_cannot_checkpoint(monkeypatch):
    started = asyncio.Event()
    saved = []

    class SuppressingEngine:
        async def stream_chat(self, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return
            yield SimpleNamespace(finished=True)

    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    monkeypatch.setattr(
        api, "save_prefix_cache_to_disk", lambda budget: saved.append(budget)
    )
    body = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}], persist=True
    )
    task = asyncio.create_task(api._warm(SuppressingEngine(), body))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert saved == []


@pytest.mark.asyncio
async def test_repeated_warm_cancellation_waits_for_stream_cleanup(monkeypatch):
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release_cleanup = asyncio.Event()
    closed = asyncio.Event()

    class SuppressingEngine:
        async def stream_chat(self, **kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release_cleanup.wait()
            try:
                yield SimpleNamespace(finished=True)
            finally:
                closed.set()

    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    monkeypatch.setattr(
        api,
        "save_prefix_cache_to_disk",
        lambda budget: pytest.fail("cancelled stream wrote a checkpoint"),
    )
    body = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}], persist=True
    )
    task = asyncio.create_task(api._warm(SuppressingEngine(), body))
    try:
        await started.wait()
        task.cancel()
        await cleaning.wait()
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not closed.is_set()
    finally:
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure", [RuntimeError("stream failed"), asyncio.CancelledError()]
)
async def test_stream_failure_propagates_and_closes_without_checkpoint(
    monkeypatch, failure
):
    closed = asyncio.Event()

    class FailingEngine:
        async def stream_chat(self, **kwargs):
            try:
                raise failure
                yield
            finally:
                closed.set()

    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    monkeypatch.setattr(
        api,
        "save_prefix_cache_to_disk",
        lambda budget: pytest.fail("failed stream wrote a checkpoint"),
    )
    body = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}], persist=True
    )
    with pytest.raises(type(failure)):
        await api._warm(FailingEngine(), body)
    assert closed.is_set()


@pytest.mark.asyncio
async def test_empty_stream_does_not_report_success_or_checkpoint(monkeypatch):
    class EmptyEngine:
        async def stream_chat(self, **kwargs):
            if False:
                yield None

    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    monkeypatch.setattr(
        api,
        "save_prefix_cache_to_disk",
        lambda budget: pytest.fail("saved empty stream"),
    )
    body = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}], persist=True
    )
    with pytest.raises(HTTPException) as error:
        await api._warm(EmptyEngine(), body)
    assert error.value.status_code == 503


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
    assert messages[0]["content"] == "agent instructions"
    assert body.messages[0]["content"] == "agent instructions"
    assert tools == api.convert_tools_for_template(body.tools)
    assert thinking is False


def test_tool_free_system_prefix_is_accepted():
    body = api.PrefillRequest(
        kind="prefix",
        messages=[{"role": "system", "content": "shared instructions"}],
    )
    messages, tools, _ = api._render_inputs(body)
    assert messages == body.messages
    assert tools is None


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


def test_tool_prompt_without_system_and_no_thinking_override(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(
            engine=SimpleNamespace(is_mllm=False, preserve_native_tool_format=False),
            tool_call_parser="qwen3",
            no_thinking=True,
        ),
    )
    body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "draft"}],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        draft_key="no-system",
        revision=1,
        enable_thinking=True,
    )
    messages, _, thinking = api._render_inputs(body)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == _TOOL_USE_SYSTEM_SUFFIX.strip()
    assert thinking is False


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
@pytest.mark.parametrize("suppress_predecessor", [False, True])
async def test_cancelled_replacement_does_not_launch_or_consume_revision(
    monkeypatch, suppress_predecessor
):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 1
    )
    started = asyncio.Event()
    cleaning = asyncio.Event()
    release_cleanup = asyncio.Event()
    launched = []

    async def warm(_engine, body):
        launched.append(body.revision)
        if body.revision == 1:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                while True:
                    try:
                        await release_cleanup.wait()
                        break
                    except asyncio.CancelledError:
                        if not suppress_predecessor:
                            raise
                if not suppress_predecessor:
                    raise
        return {"status": "prefilled"}

    monkeypatch.setattr(api, "_warm", warm)
    draft_key = f"cancel-replace-{suppress_predecessor}"
    first_body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "partial"}],
        draft_key=draft_key,
        revision=1,
    )
    second_body = first_body.model_copy(update={"revision": 2})
    first = asyncio.create_task(api.prefill(first_body, ConnectedRequest()))
    await started.wait()
    second = asyncio.create_task(api.prefill(second_body, ConnectedRequest()))
    await cleaning.wait()
    second.cancel()
    release_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert launched == [1]
    assert api._draft_revisions[draft_key][0] == 1
    if suppress_predecessor:
        assert await first == {"status": "prefilled"}
    else:
        with pytest.raises(asyncio.CancelledError):
            await first


@pytest.mark.asyncio
async def test_repeated_replacement_cancellation_waits_for_predecessor(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=FakeEngine()),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 1
    )
    started = asyncio.Event()
    cleaning = asyncio.Event()
    cancelled_again = asyncio.Event()
    release_cleanup = asyncio.Event()
    launched = []

    async def warm(_engine, body):
        launched.append(body.revision)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleaning.set()
            while not release_cleanup.is_set():
                try:
                    await release_cleanup.wait()
                except asyncio.CancelledError:
                    cancelled_again.set()
            raise

    monkeypatch.setattr(api, "_warm", warm)
    body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "partial"}],
        draft_key="repeated-replacement-cancel",
        revision=1,
    )
    first = asyncio.create_task(api.prefill(body, ConnectedRequest()))
    await started.wait()
    predecessor = api._active
    second = asyncio.create_task(
        api.prefill(body.model_copy(update={"revision": 2}), ConnectedRequest())
    )
    try:
        await cleaning.wait()
        second.cancel()
        await cancelled_again.wait()
        second.cancel()
        await asyncio.sleep(0)
        assert not second.done()
        assert not predecessor.done()
        assert api._guard.locked()
        assert launched == [1]
        assert api._draft_revisions[body.draft_key][0] == 1
    finally:
        release_cleanup.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    assert api._active is None


@pytest.mark.asyncio
async def test_replacement_propagates_predecessor_failure(monkeypatch):
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=FakeEngine()),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 1
    )
    started = asyncio.Event()
    launched = []
    failure = RuntimeError("predecessor cleanup failed")

    async def warm(_engine, body):
        launched.append(body.revision)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise failure

    monkeypatch.setattr(api, "_warm", warm)
    body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "partial"}],
        draft_key="replacement-predecessor-failure",
        revision=1,
    )
    first = asyncio.create_task(api.prefill(body, ConnectedRequest()))
    await started.wait()
    second = asyncio.create_task(
        api.prefill(body.model_copy(update={"revision": 2}), ConnectedRequest())
    )
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert results == [failure, failure]
    assert launched == [1]
    assert api._draft_revisions[body.draft_key][0] == 1
    assert api._active is None


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
@pytest.mark.parametrize("persist", [False, True])
async def test_prefix_requires_disk_consent_before_engine_work(monkeypatch, persist):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    body = api.PrefillRequest(
        kind="prefix",
        messages=[{"role": "system", "content": "private instructions"}],
        persist=persist,
    )
    with pytest.raises(HTTPException) as error:
        await api.prefill(body, ConnectedRequest())
    assert error.value.status_code == 409
    assert engine.calls == []
    assert api.PrefillRequest(
        kind="prefix",
        messages=body.messages,
        persist=persist,
        allow_disk_cache=True,
    ).allow_disk_cache


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
async def test_prefill_rejects_unready_and_unmatched_prompt_settings(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    body = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}]
    )
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=False, draining=False, engine=engine),
    )
    with pytest.raises(HTTPException) as unavailable:
        await api.prefill(body, ConnectedRequest())
    assert unavailable.value.status_code == 503

    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(
            ready=True,
            draining=False,
            engine=engine,
            default_reasoning_effort="low",
        ),
    )
    with pytest.raises(HTTPException) as unmatched:
        await api.prefill(body, ConnectedRequest())
    assert unmatched.value.status_code == 409
    assert engine.calls == []


@pytest.mark.asyncio
async def test_prefill_rejects_unrenderable_prefix_and_uncountable_draft(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    prefix = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}]
    )
    with pytest.raises(HTTPException) as unrenderable:
        await api.prefill(prefix, ConnectedRequest())
    assert unrenderable.value.status_code == 400

    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: None
    )
    draft = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "draft"}],
        draft_key="uncountable",
        revision=1,
    )
    with pytest.raises(HTTPException) as uncountable:
        await api.prefill(draft, ConnectedRequest())
    assert uncountable.value.status_code == 400
    assert engine.calls == []


@pytest.mark.asyncio
async def test_prefill_limits_overlapping_clients(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 1
    )
    one = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "first"}],
        draft_key="client-one",
        revision=1,
    )
    two = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "second"}],
        draft_key="client-two",
        revision=1,
    )
    first = asyncio.create_task(api.prefill(one, ConnectedRequest()))
    await engine.started.wait()
    with pytest.raises(HTTPException) as busy:
        await api.prefill(two, ConnectedRequest())
    assert busy.value.status_code == 429
    engine.release.set()
    await first


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


@pytest.mark.asyncio
@pytest.mark.parametrize("save_raises", [False, True])
async def test_cancelled_checkpoint_waits_for_thread(monkeypatch, save_raises):
    engine = FakeEngine()
    engine.release.set()
    started = threading.Event()
    finish = threading.Event()
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)

    def save(_budget):
        started.set()
        assert finish.wait(timeout=2)
        if save_raises:
            raise RuntimeError("checkpoint failed")
        return True

    monkeypatch.setattr(api, "save_prefix_cache_to_disk", save)
    body = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}], persist=True
    )
    task = asyncio.create_task(api._warm(engine, body))
    assert await asyncio.to_thread(started.wait, 2)
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    task.cancel()
    await asyncio.sleep(0.01)
    assert not task.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_handler_cancellation_keeps_checkpoint_slot_until_thread_exits(
    monkeypatch,
):
    class DisconnectRequest:
        disconnected = False
        disconnect_observed = False

        async def is_disconnected(self):
            if self.disconnected:
                self.disconnect_observed = True
            return self.disconnected

    engine = FakeEngine()
    engine.release.set()
    engine._shared_system_tools_prompt = lambda *args: ([1, 2, 3], 2)
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: False)
    monkeypatch.setattr(api, "enforce_context_length", lambda *args, **kwargs: None)
    started = threading.Event()
    finish = threading.Event()

    def save(_budget):
        started.set()
        assert finish.wait(timeout=3)
        return True

    monkeypatch.setattr(api, "save_prefix_cache_to_disk", save)
    body = api.PrefillRequest(
        kind="prefix",
        messages=[{"role": "system", "content": "private instructions"}],
        persist=True,
        allow_disk_cache=True,
    )
    request = DisconnectRequest()
    handler = asyncio.create_task(api.prefill(body, request))
    assert await asyncio.to_thread(started.wait, 2)
    request.disconnected = True
    for _ in range(30):
        if request.disconnect_observed:
            break
        await asyncio.sleep(0.01)
    assert request.disconnect_observed
    handler.cancel()
    await asyncio.sleep(0.01)
    assert api._active is not None
    assert not api._active.done()
    with pytest.raises(HTTPException) as busy:
        await api.prefill(body, ConnectedRequest())
    assert busy.value.status_code == 429
    checkpoint = api._active
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await handler
    with pytest.raises(asyncio.CancelledError):
        await checkpoint


@pytest.mark.asyncio
async def test_successful_prefix_preflight_and_invalid_normalization(monkeypatch):
    engine = FakeEngine()
    engine.release.set()
    engine._shared_system_tools_prompt = lambda *args: ([1, 2, 3], 2)
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(api, "enforce_context_length", lambda *args, **kwargs: None)
    body = api.PrefillRequest(
        kind="prefix", messages=[{"role": "system", "content": "shared"}]
    )
    result = await api.prefill(body, ConnectedRequest())
    assert result["status"] == "prefilled"
    assert len(engine.calls) == 1

    monkeypatch.setattr(
        api,
        "_render_inputs",
        lambda _body: (_ for _ in ()).throw(ValueError("bad history")),
    )
    with pytest.raises(HTTPException) as invalid:
        await api.prefill(body, ConnectedRequest())
    assert invalid.value.status_code == 400


@pytest.mark.asyncio
async def test_revision_window_expires_and_evicts_old_keys(monkeypatch):
    engine = FakeEngine()
    engine.release.set()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 1
    )
    monkeypatch.setattr(api, "_draft_revisions", OrderedDict({"expired": (1, -1000.0)}))
    monkeypatch.setattr(api, "_MAX_REVISION_KEYS", 1)
    for key in ("fresh-one", "fresh-two"):
        body = api.PrefillRequest(
            kind="draft",
            messages=[{"role": "user", "content": key}],
            draft_key=key,
            revision=1,
        )
        await api.prefill(body, ConnectedRequest())
    assert list(api._draft_revisions) == ["fresh-two"]


@pytest.mark.asyncio
async def test_disconnect_cancels_draft_work(monkeypatch):
    class DisconnectedRequest:
        async def is_disconnected(self):
            return True

    engine = FakeEngine()
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
        messages=[{"role": "user", "content": "unsent"}],
        draft_key="disconnect",
        revision=1,
    )
    with pytest.raises(asyncio.CancelledError):
        await api.prefill(body, DisconnectedRequest())
    assert api._active is None


@pytest.mark.asyncio
async def test_active_revision_remains_stale_if_watermark_expires(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(
        api,
        "get_config",
        lambda: SimpleNamespace(ready=True, draining=False, engine=engine),
    )
    monkeypatch.setattr(api.disk_caches, "disabled", lambda: True)
    monkeypatch.setattr(
        api, "enforce_context_length_for_messages", lambda *args, **kwargs: 1
    )
    monkeypatch.setattr(api, "_draft_revisions", OrderedDict())
    body = api.PrefillRequest(
        kind="draft",
        messages=[{"role": "user", "content": "partial"}],
        draft_key="active-stale",
        revision=1,
    )
    first = asyncio.create_task(api.prefill(body, ConnectedRequest()))
    await engine.started.wait()
    api._draft_revisions.clear()
    with pytest.raises(HTTPException) as stale:
        await api.prefill(body, ConnectedRequest())
    assert stale.value.status_code == 409
    engine.release.set()
    await first
