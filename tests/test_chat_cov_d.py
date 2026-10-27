# SPDX-License-Identifier: Apache-2.0
"""Additional error and boundary contracts for chat tools."""

import asyncio
import io
import json
import runpy
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from rapid_mlx.chat_runtime import file_extraction as extraction
from rapid_mlx.chat_runtime import file_operations as ops
from rapid_mlx.chat_runtime import file_worker
from rapid_mlx.chat_runtime.tools import action_policy as policy
from rapid_mlx.chat_runtime.tools import action_tools, page, web
from tests.test_chat_harness_extraction import workbook


def test_extraction_boundaries(monkeypatch):
    with pytest.raises(extraction.DocumentError, match="Binary"):
        extraction.extract_document(b"a\x00b", "x.csv")
    data = workbook(
        '<row><c r="A1" t="b"><v>1</v></c><c r="B1" t="b"><v>0</v></c></row>'
    )
    assert "A=TRUE\tB=FALSE" in extraction.extract_document(data, "x.xlsx")
    monkeypatch.setattr(extraction.extraction, "MAX_DOCUMENT_PAGES", 0)
    with pytest.raises(extraction.DocumentError) as exc:
        extraction.extract_document(data, "x.xlsx")
    assert exc.value.code == "file_too_large"


@pytest.mark.parametrize(
    ("payload", "name", "key"), [(b"hello", "a.txt", "text"), (b"\0", "a.csv", "error")]
)
def test_extraction_main(monkeypatch, capsys, payload, name, key):
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(payload)))
    monkeypatch.setattr(sys, "argv", ["file_extraction", name])
    runpy.run_module("rapid_mlx.chat_runtime.file_extraction", run_name="__main__")
    assert key in json.loads(capsys.readouterr().out)


def test_file_operations_boundaries(tmp_path, monkeypatch):
    (tmp_path / ".gitignore").write_text("# comment\n\n*.tmp\n")
    assert len(ops._ignore_rules(tmp_path, tmp_path)) == 1
    (tmp_path / "pyvenv.cfg").touch()
    assert list(ops.walk_files(tmp_path, tmp_path, 0)) == []
    assert ops.search(tmp_path, "x")["matches"] == []
    (tmp_path / "pyvenv.cfg").unlink()
    (tmp_path / "a.txt").write_text("x")
    with pytest.raises(ops.FileError):
        ops.search(tmp_path, "")
    monkeypatch.setattr(ops, "MAX_FILES", 0)
    assert ops.search(tmp_path, "x")["truncated"]
    with pytest.raises(ops.FileError, match="limit"):
        list(ops.walk_files(tmp_path, tmp_path, -1))
    values = iter([0, 2])
    monkeypatch.setattr(ops.time, "monotonic", lambda: next(values))
    with pytest.raises(ops.FileError, match="timed out"):
        list(ops.walk_files(tmp_path, tmp_path, 1))


def test_dispatch_errors(monkeypatch):
    assert (
        ops.dispatch({"args": {}, "output_cap": 1})["error"]["code"] == "context_limit"
    )
    for error, code in [(FileNotFoundError(), "not_found"), (OSError(), "unreadable")]:

        def fail(*args, _error=error, **kwargs):
            raise _error

        monkeypatch.setattr(ops, "read_file", fail)
        assert (
            ops.dispatch({"args": {}, "operation": "read_file"})["error"]["code"]
            == code
        )


def test_operations_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"args": {}, "output_cap": 1}'))
    runpy.run_module("rapid_mlx.chat_runtime.file_operations", run_name="__main__")
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "context_limit"


async def test_worker_failures(monkeypatch):
    process = SimpleNamespace(
        returncode=1, communicate=AsyncMock(return_value=(b"", b""))
    )
    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    assert (await file_worker.run_worker("x", b""))["error"]["code"] == "worker_failed"
    process.returncode = None
    process.pid = 123
    process.communicate = AsyncMock(side_effect=TimeoutError)
    process.wait = AsyncMock()
    monkeypatch.setattr(
        file_worker.os,
        "killpg",
        lambda *args: (_ for _ in ()).throw(ProcessLookupError()),
    )
    assert (await file_worker.run_worker("x", b""))["error"]["code"] == "timeout"
    process.wait.assert_awaited_once()


def test_policy_boundaries(tmp_path, monkeypatch):
    assert policy.file_policy(tmp_path, ".git/config").risk == "dangerous"
    assert (
        policy.scan_code("x" * 65537, tmp_path) == "Code exceeds the inspection limit"
    )
    assert (
        policy.scan_code("(", tmp_path, python=True) == "Python could not be inspected"
    )
    assert (
        policy.scan_code("echo '", tmp_path) == "Shell quoting could not be inspected"
    )
    assert (
        policy.scan_code("cd folder", tmp_path)
        == "Command changes the workspace directory"
    )
    assert (
        policy.scan_code("touch .git/config", tmp_path)
        == "Command may change a protected path"
    )
    script = tmp_path / "script.py"
    script.write_text("print(1)")
    assert policy.scan_code("python script.py", tmp_path) is None
    script.write_text("x" * 65537)
    assert (
        policy.scan_code("python script.py", tmp_path)
        == "Python script exceeds the inspection limit"
    )
    script.unlink()
    assert (
        policy.scan_code("python script.py", tmp_path)
        == "Python script could not be inspected"
    )
    assert (
        policy.scan_code("python ../outside.py", tmp_path)
        == "Python script is outside the workspace or protected"
    )
    monkeypatch.setattr(
        policy, "resolve_path", lambda *args: (_ for _ in ()).throw(ValueError())
    )
    assert (
        policy.scan_code('"hello"', tmp_path, python=True)
        == "Code path could not be inspected"
    )
    assert (
        policy.scan_code("echo hello", tmp_path)
        == "Command path could not be inspected"
    )


def test_images_and_preparation(tmp_path, monkeypatch):
    path = tmp_path / "file"
    path.write_bytes(b"\0")
    with pytest.raises(ValueError, match="Binary"):
        action_tools.read_image(path)
    path.write_bytes(b"x" * (action_tools.MAX_FILE_BYTES + 1))
    with pytest.raises(ValueError, match="regular"):
        action_tools.read_image(path)
    path.write_text("before")
    with pytest.raises(ValueError, match="changed"):
        action_tools.replace_image(path, b"after", None, b"wrong")
    actions = action_tools.ActionTools(SimpleNamespace(directory=tmp_path / "profile"))
    deps = SimpleNamespace(workspace=str(tmp_path))
    with pytest.raises(ValueError, match="profile"):
        actions.prepare("write_file", {"path": "profile/a", "content": "x"}, deps)
    with pytest.raises(ValueError, match="exact"):
        actions.prepare(
            "patch", {"path": "file", "old_text": "missing", "new_text": "x"}, deps
        )
    with pytest.raises(ValueError, match="UTF-8"):
        actions.prepare("write_file", {"path": "file", "content": "\0"}, deps)
    monkeypatch.setattr(
        action_tools, "file_policy", lambda *args: policy.Policy(risk="secret")
    )
    secret = actions.prepare(
        "write_file", {"path": "file", "content": "password"}, deps
    )
    assert "password" not in secret.preview
    with pytest.raises(ValueError, match="nonempty"):
        actions.prepare("terminal", {"command": " ", "timeout": 1}, deps)


def test_page_parser_and_web_validation():
    assert not page.public_ip("bad ip")
    parser = page._PageParser("https://example.com")
    parser.feed(
        "<script><script>x</script></script><main><main><li>item</li><br/></main></main>"
    )
    assert "- item" in parser.text()
    assert not web._safe_url("http://[invalid")
    with pytest.raises(ValueError, match="object"):
        web._rows([1], "snippet")
    with pytest.raises(ValueError, match="text"):
        web._rows([{"url": 1}], "snippet")
    assert web._match(None) == ""
    with pytest.raises(ValueError):
        web._match(1)
    with pytest.raises(ValueError):
        web._select_place("x", [], "", "", {})
    with pytest.raises(ValueError):
        web._number(float("nan"))


async def test_page_errors(monkeypatch):
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "getaddrinfo", AsyncMock(side_effect=OSError()))
    with pytest.raises(web.ProviderError) as exc:
        await page.resolve_public("https://example.com")
    assert exc.value.error.code == "unresolvable"
    monkeypatch.setattr(
        page,
        "resolve_public",
        AsyncMock(return_value=(httpx.URL("https://example.com"), "8.8.8.8")),
    )
    with pytest.raises(web.ProviderError) as exc:
        await page.direct_fetch(
            "https://example.com", httpx.MockTransport(lambda r: httpx.Response(404))
        )
    assert exc.value.error.code == "fetch_error"
    result = await page.direct_fetch(
        "https://example.com",
        httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                headers={"content-type": "text/plain; charset=not-a-codec"},
                content=b"hello",
            )
        ),
    )
    assert "hello" in result.content


async def test_search_validation():
    client = web.WebTools(
        httpx.MockTransport(lambda r: httpx.Response(500)),
        {"RAPID_MLX_WEB_SEARCH_PROVIDER": "invalid"},
    )
    assert (await client.search("")).error.code == "invalid_arguments"
    assert (await client.search("test")).error.code == "configuration"


def test_forecast_skips_bad_days_and_limits_count():
    current = {
        "weather_code": 0,
        "temperature_2m": 1,
        "relative_humidity_2m": 2,
        "wind_speed_10m": 3,
    }
    daily = {
        "time": ["bad"] + ["2026-01-01"] * 8,
        "weather_code": [0] * 9,
        "temperature_2m_min": [1] * 9,
        "temperature_2m_max": [2] * 9,
    }
    result = web._forecast_report(
        {"name": "Town"}, {"current": current, "daily": daily}, False
    )
    assert result.count("2026-01-01") == 6
    assert "bad" not in result


async def test_tool_adapters(monkeypatch):
    from rapid_mlx.chat_runtime import tools
    from rapid_mlx.chat_runtime.tools import files

    deps = SimpleNamespace(
        emit=AsyncMock(),
        web_calls=0,
        web_bytes=0,
        web=None,
        sources={},
        read_budget_bytes=3000,
        read_lock=asyncio.Lock(),
        workspace="/tmp",
        attachment_ids={"a"},
    )
    ctx = SimpleNamespace(deps=deps, tool_call_id="call")
    result = await tools._web_call(ctx, "weather", {})
    assert json.loads(result)["error"]["code"] == "disabled"
    deps.web = SimpleNamespace(weather=AsyncMock(return_value=web.WebResult("Sunny")))
    assert await tools._web_call(ctx, "weather", {}) == "Sunny"
    assert "nonnegative" in await tools.read_attachment(ctx, "a", offset=-1)
    monkeypatch.setattr(
        files, "run_worker", AsyncMock(return_value={"content": "x" * 5000})
    )
    result = await files._execute(ctx, "read_file", {})
    assert json.loads(result)["error"]["code"] == "context_limit"


async def test_action_failure_boundaries(tmp_path, monkeypatch):
    store = SimpleNamespace(directory=tmp_path / "profile")
    actions = action_tools.ActionTools(store)
    deps = SimpleNamespace(
        action_tools=False,
        workspace=str(tmp_path),
        emit=AsyncMock(),
        conversation_id="c",
        turn_id="t",
    )
    ctx = SimpleNamespace(deps=deps, tool_call_id="call")
    assert "disabled" in await actions.call(ctx, "terminal", {})
    deps.action_tools = True
    args = {"command": "", "timeout": 1}
    actions.prepared[("t", "call")] = (
        SimpleNamespace(name="terminal", args=args),
        "op",
    )
    store.get_operation = AsyncMock(side_effect=ValueError("invalid"))
    assert "invalid" in await actions.call(ctx, "terminal", args)
    store.begin_undo = AsyncMock(
        return_value=(
            {"id": "op", "revision": 1},
            {"path": str(tmp_path / "f"), "post_hash": "x"},
        )
    )
    store.transition_operation = AsyncMock()
    monkeypatch.setattr(
        action_tools, "read_image", lambda path: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(OSError):
        await actions._undo("c", "op")
    assert store.transition_operation.await_args.args[2] == "outcome_unknown"


async def test_search_transport_and_duplicates(monkeypatch):
    client = web.WebTools(None, {})
    monkeypatch.setattr(
        client, "_search_provider", AsyncMock(side_effect=httpx.ConnectError("offline"))
    )
    with pytest.raises(httpx.ConnectError):
        await client._search(None, "test")
    row = web.SearchResult("Title", "https://example.com", "snippet")
    monkeypatch.setattr(client, "_search_provider", AsyncMock(return_value=[row, row]))
    result = await client._search(None, "test")
    assert len(result.sources) == 1


def test_policy_resolution_errors_and_symlink(tmp_path, monkeypatch):
    original = policy.resolve_path
    calls = 0

    def fail_second(workspace, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("unresolvable")
        return original(workspace, value)

    monkeypatch.setattr(policy, "resolve_path", fail_second)
    assert (
        policy.scan_code('"folder/file"', tmp_path, python=True)
        == "Code path could not be inspected"
    )
    monkeypatch.setattr(policy, "resolve_path", original)
    (tmp_path / "link").symlink_to(tmp_path.parent, target_is_directory=True)
    assert (
        policy.scan_code("touch link/file", tmp_path)
        == "Command may write through a path outside the workspace"
    )


async def test_approval_timeout_conflict():
    from rapid_mlx.chat_runtime.store import Conflict

    store = SimpleNamespace(
        get_operation=AsyncMock(
            side_effect=[
                {"state": "intent", "expires_at": 0, "revision": 1},
                {"state": "denied"},
            ]
        ),
        decide_operation=AsyncMock(side_effect=Conflict("changed")),
    )
    assert (await action_tools.ActionTools(store).wait("op"))["state"] == "denied"


def test_nonregular_image(tmp_path):
    import os

    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular"):
        action_tools.read_image(fifo)
    with pytest.raises(ValueError, match="regular"):
        action_tools.replace_image(fifo, b"x", None, b"")


async def test_invalid_weather_coordinates():
    client = web.WebTools(
        httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={"results": [{"name": "Town", "latitude": 91, "longitude": 0}]},
            )
        ),
        {},
    )
    result = await client.weather("Town")
    assert result.error is not None


async def test_page_unknown_encoding(monkeypatch):
    monkeypatch.setattr(
        page,
        "resolve_public",
        AsyncMock(return_value=(httpx.URL("https://example.com"), "8.8.8.8")),
    )
    monkeypatch.setattr(
        httpx.Response, "encoding", property(lambda self: "invalid-codec")
    )
    result = await page.direct_fetch(
        "https://example.com",
        httpx.MockTransport(
            lambda req: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"hello"
            )
        ),
    )
    assert "hello" in result.content
    parser = page._PageParser("https://example.com")
    parser.handle_starttag("script", [])
    parser.handle_starttag("script", [])
    assert parser.hidden == ["script", "script"]


async def test_cancel_during_process_spawn(monkeypatch, tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()
    process = SimpleNamespace(pid=123, wait=AsyncMock())

    async def spawn(*args, **kwargs):
        entered.set()
        await release.wait()
        return process

    killed = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(action_tools, "kill_group", killed)
    task = asyncio.create_task(
        action_tools.run_process(["ignored"], str(tmp_path), 1, AsyncMock(), "c")
    )
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    killed.assert_awaited_once_with(process)


async def test_action_dispatch_guards(tmp_path, monkeypatch):
    args = {"command": "echo ok", "timeout": 1}
    deps = SimpleNamespace(
        action_tools=True,
        workspace=str(tmp_path / "."),
        emit=AsyncMock(),
        conversation_id="c",
        turn_id="t",
    )
    ctx = SimpleNamespace(deps=deps, tool_call_id="call")
    prepared = action_tools.Prepared(
        "terminal", args, "echo ok", "echo ok", policy.Policy()
    )
    store = SimpleNamespace(
        get_operation=AsyncMock(return_value={"state": "approved", "revision": 1}),
        transition_operation=AsyncMock(),
        dispatch_operation=AsyncMock(return_value={"revision": 2}),
    )
    actions = action_tools.ActionTools(store)
    actions.prepared[("t", "call")] = (prepared, "op")
    assert "Action could not inspect" in await actions.call(
        ctx, "terminal", {"command": "changed", "timeout": 1}
    )
    deps.workspace = str(tmp_path) + "/."
    assert "Workspace changed" in await actions.call(ctx, "terminal", args)
    assert store.transition_operation.await_args.args[2] == "cancelled_before_effect"
    deps.workspace = str(tmp_path)
    monkeypatch.setattr(
        action_tools, "run_process", AsyncMock(side_effect=OSError("failure"))
    )
    assert "effects unknown" in await actions.call(ctx, "terminal", args)
    assert store.transition_operation.await_args.args[2] == "outcome_unknown"


def test_edit_rejects_file_that_grew_after_stat(tmp_path, monkeypatch):
    path = tmp_path / "growing"
    path.write_bytes(b"x" * (action_tools.MAX_FILE_BYTES + 1))
    original = action_tools.os.fstat

    def stale_stat(fd):
        info = original(fd)
        return SimpleNamespace(st_mode=info.st_mode, st_size=0)

    monkeypatch.setattr(action_tools.os, "fstat", stale_stat)
    with pytest.raises(ValueError, match="grew"):
        action_tools.read_image(path)
