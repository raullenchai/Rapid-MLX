# SPDX-License-Identifier: Apache-2.0
"""Real SDK lifecycle regression for #4466; isolate asyncio.run cleanup too."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_real_manager_stdio_lifecycle(tmp_path):
    pytest.importorskip("mcp")
    child = tmp_path / "tools.py"
    child.write_text(
        textwrap.dedent("""
        import asyncio, os, sys
        from pathlib import Path
        with Path(sys.argv[1]).open("a") as file:
            file.write(str(os.getpid()) + "\\n")
        try:
            from mcp.server.fastmcp import FastMCP as Server
        except ImportError:
            from mcp.server.mcpserver import MCPServer as Server
        mcp = Server("lifecycle")
        @mcp.tool()
        async def echo(text: str) -> str:
            if text == "wait":
                await asyncio.sleep(60)
            return text
        @mcp.tool()
        def pid() -> int:
            return os.getpid()
        mcp.run(transport="stdio")
    """)
    )
    parent = textwrap.dedent("""
        import asyncio, gc, logging, os, sys
        from rapid_mlx.mcp.manager import MCPClientManager
        from rapid_mlx.mcp.types import MCPConfig
        logging.basicConfig(level=logging.INFO)
        async def main():
            errors = []
            asyncio.get_running_loop().set_exception_handler(lambda loop, ctx: errors.append(ctx))
            config = MCPConfig.from_dict({"mcpServers": {name: {
                "command": sys.executable, "args": [sys.argv[1], sys.argv[2]], "skip_security_validation": True
            } for name in ("one", "two")}})
            manager = MCPClientManager(config)
            pids = []
            for cycle in range(2):
                await manager.start()
                assert len(manager.get_all_tools()) == 4
                for name in ("one", "two"):
                    result = await manager.execute_tool(name + "__echo", {"text": "dogfood"})
                    assert not result.is_error and result.content == "dogfood", result
                    result = await manager.execute_tool(name + "__pid", {})
                    assert not result.is_error, result
                    pids.append(int(result.content))
                await manager.reconnect("one")
                await manager.reconnect()
                await manager.refresh_tools()
                assert len(manager.get_all_tools()) == 4
                await manager.stop()
                await manager.stop()
                assert not manager.is_started
                assert not manager.get_all_tools()
            pending = asyncio.create_task(manager.start())
            await pending
            call = asyncio.create_task(manager.execute_tool("one__echo", {"text": "wait"}))
            await asyncio.sleep(0.05)
            call.cancel()
            try:
                await call
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("tool cancellation was swallowed")
            result = await manager.execute_tool("one__echo", {"text": "after cancel"})
            assert not result.is_error and result.content == "after cancel"
            await manager.stop()

            # Cancel startup with a real transport/session already entered.
            entered = asyncio.Event()
            client = manager.get_client("one")
            original = client._initialize_session
            async def blocked_initialize():
                await original()
                entered.set()
                await asyncio.Event().wait()
            client._initialize_session = blocked_initialize
            startup = asyncio.create_task(manager.start())
            await entered.wait()
            startup.cancel()
            try:
                await startup
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("startup cancellation was swallowed")
            assert not manager.is_started
            assert not manager.get_all_tools()
            gc.collect()
            await asyncio.sleep(0.05)
            assert not errors, errors
            from pathlib import Path
            pids.extend(int(pid) for pid in Path(sys.argv[2]).read_text().splitlines())
            for pid in pids:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    continue
                raise AssertionError(f"MCP child survived shutdown: {pid}")
        asyncio.run(main())
        print("DOGFOOD_OK")
    """)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    result = subprocess.run(
        [sys.executable, "-c", parent, str(child), str(tmp_path / "pids")],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "DOGFOOD_OK" in result.stdout
    assert "cancel scope" not in result.stderr, result.stderr
    assert "Error disconnecting" not in result.stderr, result.stderr
    assert "ERROR:" not in result.stderr, result.stderr


@pytest.fixture
def owned_client(monkeypatch):
    """Use actual AnyIO scopes, with deterministic startup/teardown barriers."""
    import asyncio
    from contextlib import asynccontextmanager

    import anyio

    from rapid_mlx.mcp.client import MCPClient
    from rapid_mlx.mcp.types import MCPServerConfig, MCPTool

    client = MCPClient(MCPServerConfig(name="owned", command="python3"))
    events = []
    initializing = asyncio.Event()
    initialize_release = asyncio.Event()
    closing = asyncio.Event()
    close_release = asyncio.Event()
    close_release.set()

    @asynccontextmanager
    async def context(kind):
        owner = asyncio.current_task()
        async with anyio.create_task_group():
            events.append((kind, "enter"))
            try:
                yield
            finally:
                assert asyncio.current_task() is owner
                closing.set()
                await close_release.wait()
                events.append((kind, "exit"))

    async def transport(self):
        await self._exit_stack.enter_async_context(context("transport"))
        await self._exit_stack.enter_async_context(context("session"))

    async def initialize(self):
        initializing.set()
        await initialize_release.wait()

    async def discover(self):
        self._tools = [MCPTool("owned", "echo", "", {})]

    monkeypatch.setattr(MCPClient, "_connect_stdio", transport)
    monkeypatch.setattr(MCPClient, "_connect_sse", transport)
    monkeypatch.setattr(MCPClient, "_initialize_session", initialize)
    monkeypatch.setattr(MCPClient, "_discover_tools", discover)
    return client, events, initializing, initialize_release, closing, close_release


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "sse"])
async def test_contexts_exit_in_owner_after_reconnect(owned_client, transport):
    import asyncio

    from rapid_mlx.mcp.types import MCPTransport

    client, events, _, release, _, _ = owned_client
    client.config.transport = MCPTransport(transport)
    release.set()
    for _ in range(2):
        assert await asyncio.create_task(client.connect())
        assert await client.connect()  # idempotent, one owner
        await asyncio.create_task(client.disconnect())
        assert client._session is None
        assert client._lifecycle_task is None
    assert events == [
        (k, action)
        for _ in range(2)
        for k, action in (
            ("transport", "enter"),
            ("session", "enter"),
            ("session", "exit"),
            ("transport", "exit"),
        )
    ]


@pytest.mark.asyncio
async def test_cancelled_start_drains_successful_siblings(owned_client):
    import asyncio

    from rapid_mlx.mcp.client import MCPClient
    from rapid_mlx.mcp.manager import MCPClientManager
    from rapid_mlx.mcp.types import MCPConfig, MCPServerConfig

    client, events, initializing, _, _, _ = owned_client
    peer = MCPClient(MCPServerConfig(name="peer", command="python3"))

    async def initialized():
        pass

    peer._initialize_session = initialized
    manager = MCPClientManager(MCPConfig())
    manager._clients = {"owned": client, "peer": peer}
    task = asyncio.create_task(manager.start())
    await initializing.wait()
    while not peer.is_connected:
        await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not manager.is_started
    assert not manager.get_all_tools()
    assert all(c._lifecycle_task is None for c in manager._clients.values())
    assert events.count(("transport", "enter")) == 2
    assert events.count(("transport", "exit")) == 2


@pytest.mark.asyncio
async def test_repeated_cancelled_stop_waits_for_owner_cleanup(owned_client):
    import asyncio

    from rapid_mlx.mcp.manager import MCPClientManager
    from rapid_mlx.mcp.types import MCPConfig

    client, events, _, release, closing, close_release = owned_client
    release.set()
    manager = MCPClientManager(MCPConfig())
    manager._clients = {"owned": client}
    await manager.start()
    close_release.clear()
    task = asyncio.create_task(manager.stop())
    await closing.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not manager.is_started
    assert client._lifecycle_task is None
    assert events[-2:] == [("session", "exit"), ("transport", "exit")]
    await manager.start()
    await manager.stop()


@pytest.mark.asyncio
async def test_failed_handshake_unwinds_and_can_retry(owned_client, monkeypatch):
    from rapid_mlx.mcp.types import MCPServerState

    client, events, _, release, _, _ = owned_client
    initialize = client._initialize_session

    async def fail():
        raise RuntimeError("handshake failed")

    monkeypatch.setattr(client, "_initialize_session", fail)
    assert not await client.connect()
    assert client.state == MCPServerState.ERROR
    assert "handshake failed" in client.get_status().error
    assert events[-2:] == [("session", "exit"), ("transport", "exit")]
    monkeypatch.setattr(client, "_initialize_session", initialize)
    release.set()
    assert await client.connect()
    await client.disconnect()


@pytest.mark.asyncio
async def test_unexpected_owner_cancellation_is_reported(owned_client):
    import asyncio

    from rapid_mlx.mcp.types import MCPServerState

    client, _, _, release, _, _ = owned_client
    release.set()
    assert await client.connect()
    client._lifecycle_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await client.disconnect()
    assert client.state == MCPServerState.ERROR
    assert "cancelled" in client.get_status().error
    assert client._lifecycle_task is None
