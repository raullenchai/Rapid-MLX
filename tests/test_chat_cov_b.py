"""Focused coverage of chat runtime boundary behavior."""

import asyncio
import json
import os
import sys
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic_ai.messages import ModelRequest, SystemPromptPart, UserPromptPart

from rapid_mlx.chat_runtime import owner
from rapid_mlx.chat_runtime.context_limits import context_limit
from rapid_mlx.chat_runtime.history import _project_messages, compact_tool_history
from rapid_mlx.chat_runtime.path_policy import workspace_path
from rapid_mlx.chat_runtime.presentation import result_presentation, tool_label
from rapid_mlx.chat_runtime.token_budget import TokenCounter, count_payload


def test_presentation_edges():
    assert tool_label("web_fetch", {"url": "http://[invalid"})["target"] == "web page"
    assert tool_label("grep", {"pattern": "x", "path": "src"})["target"] == "x src"
    assert result_presentation("weather", "")["summary"] == "No weather data"
    assert (
        "timed out; effects unknown"
        in result_presentation("terminal", "timed out")["summary"]
    )
    assert (
        result_presentation("glob", '{"matches":[1],"truncated":true}')["summary"]
        == "1 result (more available)"
    )
    assert (
        result_presentation("delegate_task", "plain")["summary"] == "Helper completed"
    )
    assert result_presentation("other", "plain\nmore")["summary"] == "plain"


def test_limits_and_workspace(tmp_path):
    with pytest.raises(ValueError, match="too small"):
        context_limit(100)
    file = tmp_path / "file"
    file.touch()
    with pytest.raises(ValueError, match="directory"):
        workspace_path(str(file))


def test_history_invalid_and_projection():
    def row(role, parts, ident="1"):
        return {
            "id": ident,
            "role": role,
            "parts": parts,
            "created_at": "2026-01-01T00:00:00+00:00",
        }

    for role, part, error in [
        ("tool", {"type": "text", "text": "x"}, "text in"),
        ("assistant", {"type": "attachment"}, "attachments belong"),
        ("user", {"type": "tool_call", "call_id": "c"}, "invalid or duplicate"),
        ("user", {"type": "unknown"}, "unsupported"),
    ]:
        with pytest.raises(ValueError, match=error):
            _project_messages([row(role, [part])])
    call = lambda i: {"type": "tool_call", "call_id": i, "name": "x"}
    ret = lambda i: {"type": "tool_result", "call_id": i, "name": "x"}
    with pytest.raises(ValueError, match="duplicate"):
        _project_messages([row("assistant", [call("a"), call("a")])])
    projected = _project_messages(
        [
            row("assistant", [call("a"), call("b")]),
            row("tool", [ret("a")], "2"),
            row("tool", [ret("b")], "3"),
            row("assistant", [{"type": "thinking"}], "4"),
        ]
    )
    assert len(projected) == 2 and len(projected[-1].parts) == 2
    assert compact_tool_history(projected, 10000) == projected


async def test_token_counter_invalid_and_eviction():
    assert (
        count_payload([ModelRequest(parts=[SystemPromptPart("rules")])])["system"]
        == "rules"
    )

    class Client:
        value = True

        async def post(self, *args, **kwargs):
            return httpx.Response(
                200,
                json={"input_tokens": self.value},
                request=httpx.Request("POST", "http://test"),
            )

    client = Client()
    request = SimpleNamespace(
        model="m", inference_url="http://test", inference_api_key=None
    )
    counter = TokenCounter(SimpleNamespace(_http=client), request)
    assert (await counter.measure([]))["estimated"]
    assert not counter.available
    client.value = 42
    counter = TokenCounter(SimpleNamespace(_http=client), request)
    counter.cache = {str(i): i for i in range(256)}
    assert (await counter.measure([]))["used_tokens"] == 42
    assert len(counter.cache) == 256 and "0" not in counter.cache


def test_owner_directory_and_endpoint(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir(mode=0o755)
    bad.chmod(0o755)
    with pytest.raises(RuntimeError, match="permissions"):
        owner._private_directory(bad)
    link = tmp_path / "link"
    link.symlink_to(bad, target_is_directory=True)
    with pytest.raises(RuntimeError, match="owned"):
        owner._private_directory(link)
    with pytest.raises(RuntimeError, match="endpoint"):
        owner._endpoint("http://example.com:80")


def test_discover_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_HOME", str(tmp_path))
    assert owner.discover("p") is None
    run = tmp_path / "p" / "run"
    owner._private_directory(run)
    epoch = str(uuid.uuid4())
    record = {
        "format_version": owner.FORMAT_VERSION,
        "wire_version": owner.WIRE_VERSION,
        "schema_version": owner.SCHEMA_VERSION,
        "owner_epoch": epoch,
        "credential_file": "bearer-" + epoch,
        "endpoint": "http://127.0.0.1:1234",
        "pid": os.getpid(),
        "process_start_time": owner.psutil.Process().create_time(),
        "store_uuid": "store",
    }
    path = run / "discovery.json"
    credential = run / record["credential_file"]

    def save():
        path.write_text(json.dumps(record))
        path.chmod(0o600)

    save()
    credential.write_text("")
    credential.chmod(0o600)
    assert owner.discover("p") is None
    credential.write_text("token")
    for key, value, message in [
        ("format_version", 2, "discovery"),
        ("wire_version", 2, "protocol"),
        ("credential_file", "bad", "credential"),
    ]:
        old = record[key]
        record[key] = value
        save()
        with pytest.raises(RuntimeError, match=message):
            owner.discover("p")
        record[key] = old
    record["process_start_time"] -= 1
    save()
    assert owner.discover("p") is None
    record["process_start_time"] += 1
    save()
    response = SimpleNamespace(status_code=500, json=lambda: {})

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return response

    monkeypatch.setattr(owner.httpx, "Client", Client)
    assert owner.discover("p") is None
    response.status_code = 200
    response.json = lambda: {"owner_epoch": epoch, "store_uuid": "wrong"}
    assert owner.discover("p") is None
    response.json = lambda: {
        "owner_epoch": epoch,
        "store_uuid": "store",
        "wire_version": 2,
    }
    with pytest.raises(RuntimeError, match="protocol"):
        owner.discover("p")
    response.json = lambda: {
        "owner_epoch": epoch,
        "store_uuid": "store",
        "wire_version": owner.WIRE_VERSION,
    }
    assert owner.discover("p")["token"] == "token"


async def test_run_owner_lifecycle(tmp_path, monkeypatch):
    import uvicorn

    from rapid_mlx.chat_runtime import runtime, store

    monkeypatch.setenv("RAPID_MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setenv("RAPID_MLX_CHAT_IDLE_SECONDS", "0")
    fake_store = SimpleNamespace(
        store_uuid="s", recover_interrupted=AsyncMock(), close=AsyncMock()
    )
    monkeypatch.setattr(store.Store, "open", AsyncMock(return_value=fake_store))
    fake_runtime = SimpleNamespace(active_turns={}, close=AsyncMock())
    monkeypatch.setattr(runtime, "Runtime", lambda s: fake_runtime)

    class Server:
        def __init__(self, config):
            self.should_exit = False

        async def serve(self, sockets):
            assert (tmp_path / "p/run/discovery.json").exists()
            await asyncio.sleep(0.06)
            assert self.should_exit

    monkeypatch.setattr(uvicorn, "Server", Server)
    await owner.run_owner("p")
    assert not (tmp_path / "p/run/discovery.json").exists()
    fake_store.close.assert_awaited_once()
    fake_runtime.close.assert_awaited_once()


def test_owner_main_error(monkeypatch, capsys):
    async def fail(profile):
        raise ValueError("private")

    monkeypatch.setattr(owner, "run_owner", fail)
    monkeypatch.setattr(sys, "argv", ["owner", "--profile", "p"])
    with pytest.raises(SystemExit) as error:
        owner.main()
    assert error.value.code == 1
    assert "ValueError" in capsys.readouterr().err


async def test_synthesis_minimum_failure_and_short_instruction():
    from pydantic_ai.models import ModelRequestParameters

    from rapid_mlx.chat_runtime.context import (
        ContextSynthesisUnavailableError,
        fit_synthesis,
    )

    class Counter:
        def __init__(self, allow_last):
            self.calls = 0
            self.allow_last = allow_last

        async def measure(self, messages, parameters):
            self.calls += 1
            used = 1 if self.allow_last and parameters.instruction_parts else 10000
            return {"used_tokens": used, "system_tools_tokens": 100}

    messages = [ModelRequest(parts=[UserPromptPart("question")])]
    with pytest.raises(ContextSynthesisUnavailableError):
        await fit_synthesis(messages, ModelRequestParameters(), Counter(False), 100)
    counter = Counter(True)
    _, params, measured = await fit_synthesis(
        messages, ModelRequestParameters(), counter, 100, work_summary=True
    )
    assert measured["used_tokens"] <= 100
    assert "Summarize completed work" in params.instruction_parts[0].content


async def test_owner_start_failure_and_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(owner, "discover", lambda profile: None)
    monkeypatch.setattr(
        owner.subprocess,
        "Popen",
        lambda *args, **kwargs: SimpleNamespace(poll=lambda: 1),
    )
    with pytest.raises(RuntimeError, match="could not start"):
        owner.discover_or_start("fail")
    real_flock = owner.fcntl.flock

    def locked(fd, operation):
        if operation & owner.fcntl.LOCK_NB:
            raise BlockingIOError
        return real_flock(fd, operation)

    monkeypatch.setattr(owner.fcntl, "flock", locked)
    with pytest.raises(RuntimeError, match="Timed out"):
        owner.discover_or_start("locked", timeout=0)
    await owner.run_owner("locked")
    assert not (tmp_path / "locked/run/discovery.json").exists()


def test_owner_existing_lifetime_lock_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPID_MLX_CHAT_HOME", str(tmp_path))
    monkeypatch.setattr(owner, "discover", lambda profile: None)
    real_flock = owner.fcntl.flock
    calls = 0

    def flock(fd, operation):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BlockingIOError
        return real_flock(fd, operation)

    monkeypatch.setattr(owner.fcntl, "flock", flock)
    with pytest.raises(RuntimeError, match="lifetime lock"):
        owner.discover_or_start("p", timeout=0.001)


def test_owner_module_entrypoint(monkeypatch):
    import runpy

    called = []

    def run(coroutine):
        called.append(coroutine)
        coroutine.close()

    monkeypatch.setattr(asyncio, "run", run)
    monkeypatch.setattr(sys, "argv", ["owner", "--profile", "entry"])
    runpy.run_module("rapid_mlx.chat_runtime.owner", run_name="__main__")
    assert len(called) == 1


async def test_summary_and_manual_checkpoint():
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import FunctionModel

    from rapid_mlx.chat_runtime.context import ContextManager

    rows = [
        {
            "id": "a",
            "role": "user",
            "created_at": "2026-01-01T00:00:00+00:00",
            "parts": [{"type": "text", "text": "remember this"}],
        }
    ]
    store = SimpleNamespace(save_summary=AsyncMock())
    runtime = SimpleNamespace(model_override=True, store=store)
    request = SimpleNamespace(
        context_tokens=1024,
        model="test",
        branch_id="b",
        expected_revision=3,
        turn_id="t",
    )
    deps = SimpleNamespace(
        child_usage={"input_tokens": 0, "output_tokens": 0, "requests": 0}
    )
    model = FunctionModel(
        lambda messages, info: ModelResponse(parts=[TextPart("checkpoint")])
    )
    manager = ContextManager(runtime, request, deps, rows, model)
    assert await manager.summarize(rows, instructions="focus") == "checkpoint"
    assert await manager.checkpoint_older(manual=True, instructions="focus")
    store.save_summary.assert_awaited_once_with(
        "b", "checkpoint", "a", expected_revision=3, turn_id=None
    )
    assert manager.covered_ids == {"a"} and manager.rows == []


async def test_api_routes_and_errors(tmp_path):
    from rapid_mlx.chat_runtime.api import PREFIX, create_app
    from rapid_mlx.chat_runtime.runtime import Runtime
    from rapid_mlx.chat_runtime.store import Store

    store = await Store.open(tmp_path)
    runtime = Runtime(store)
    app = create_app(store, runtime, "token", metadata={"owner_epoch": "epoch"})
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": "Bearer token"},
        ) as client:
            assert (await client.get(PREFIX + "/capabilities")).json()[
                "owner_epoch"
            ] == "epoch"
            assert (
                await client.get(PREFIX + "/conversations/missing")
            ).status_code == 404
            conversation = (
                await client.post(PREFIX + "/conversations", json={})
            ).json()
            cid = conversation["id"]
            body = {
                "turn_id": "t",
                "conversation_id": "wrong",
                "branch_id": conversation["branch_id"],
                "expected_revision": 0,
                "text": "hello",
                "model": "m",
                "inference_url": "http://127.0.0.1:1",
            }
            assert (
                await client.post(PREFIX + f"/conversations/{cid}/context", json=body)
            ).status_code == 400
            assert (
                await client.get(PREFIX + "/turns/missing/subruns")
            ).status_code == 404
            assert (
                await client.get(PREFIX + f"/conversations/{cid}/operations")
            ).json() == []
    finally:
        await runtime.close()
        await store.close()
