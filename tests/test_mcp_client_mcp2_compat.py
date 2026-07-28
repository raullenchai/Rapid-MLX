# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the mcp SDK camelCase→snake_case rename (rapid-desktop#604).

mcp 1.x exposed model fields in camelCase (``protocolVersion``, ``inputSchema``,
``isError``); mcp 2.0 renamed the Python attributes to snake_case. The dev/CI
env resolves mcp 1.x, but a fresh sidecar build resolved ``mcp>=1.9.3`` to
mcp 2.0 — so ``vllm_mlx/mcp/client.py`` reading camelCase raised
``AttributeError`` mid-handshake and every configured MCP server reported
**0 tools**. The pre-existing MCP tests all fake the SDK, so the skew slipped
through.

The first three tests are **version-independent**: they simulate mcp-2.0
snake_case-only SDK objects directly, so they fail against the old camelCase
code regardless of which mcp version is installed (i.e. they catch the bug on
CI's mcp 1.x too). The last is a real end-to-end stdio round-trip against the
actually-installed SDK.
"""

from __future__ import annotations

import sys
import textwrap
from types import SimpleNamespace

import pytest

from vllm_mlx.mcp.client import MCPClient, _sdk_attr
from vllm_mlx.mcp.types import MCPServerConfig, MCPTransport

# --- helper compat shim -----------------------------------------------------


def test_sdk_attr_prefers_snake_then_camel_then_default():
    snake_only = SimpleNamespace(protocol_version="2025-11-25")  # mcp>=2.0 shape
    camel_only = SimpleNamespace(protocolVersion="2024-11-05")  # mcp<2.0 shape
    both = SimpleNamespace(protocol_version="new", protocolVersion="old")
    neither = SimpleNamespace()

    assert _sdk_attr(snake_only, "protocol_version", "protocolVersion") == "2025-11-25"
    assert _sdk_attr(camel_only, "protocol_version", "protocolVersion") == "2024-11-05"
    # snake_case (mcp 2.0) wins when both are present
    assert _sdk_attr(both, "protocol_version", "protocolVersion") == "new"
    assert _sdk_attr(neither, "protocol_version", "protocolVersion") is None
    assert _sdk_attr(neither, "input_schema", "inputSchema", {}) == {}
    # falsy-but-present values are returned as-is, not swallowed by the default
    assert (
        _sdk_attr(SimpleNamespace(is_error=False), "is_error", "isError", True) is False
    )


# --- version-independent regression guards (simulate mcp 2.0 snake_case) ----


class _FakeSession:
    """Minimal stand-in for mcp.ClientSession returning mcp-2.0-shaped objects."""

    def __init__(self, *, init_result=None, tools_result=None):
        self._init_result = init_result
        self._tools_result = tools_result

    async def initialize(self):
        return self._init_result

    async def list_tools(self):
        return self._tools_result


@pytest.mark.asyncio
async def test_initialize_session_reads_mcp2_snake_case():
    """Old code did ``result.protocolVersion`` — AttributeError under mcp 2.0."""
    client = MCPClient(MCPServerConfig(name="t", command="python3"))
    client._session = _FakeSession(
        init_result=SimpleNamespace(
            protocol_version="2025-11-25",  # snake only (mcp>=2.0)
            server_info=SimpleNamespace(name="dogfood"),
        )
    )
    # Must not raise (old camelCase access raised AttributeError here).
    await client._initialize_session()


@pytest.mark.asyncio
async def test_discover_tools_reads_mcp2_snake_case_input_schema():
    """Old code fell back to ``{}`` for a snake_case-only ``input_schema``."""
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    client = MCPClient(MCPServerConfig(name="t", command="python3"))
    client._session = _FakeSession(
        tools_result=SimpleNamespace(
            tools=[
                SimpleNamespace(
                    name="get_weather",
                    description="weather",
                    input_schema=schema,  # snake only (mcp>=2.0); no inputSchema
                )
            ]
        )
    )
    await client._discover_tools()
    assert len(client.tools) == 1
    # The whole point: the schema survives the SDK rename, not dropped to {}.
    assert client.tools[0].input_schema == schema


# --- real end-to-end stdio round-trip against the installed SDK -------------

_STDIO_SERVER = textwrap.dedent(
    """
    import sys, json
    TOOLS = [{"name": "get_secret_word",
              "description": "Return the secret word.",
              "inputSchema": {"type": "object", "properties": {}, "required": []}},
             {"name": "add",
              "description": "Add a and b.",
              "inputSchema": {"type": "object",
                              "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                              "required": ["a", "b"]}}]
    def dispatch(name, args):
        if name == "get_secret_word": return "PLATYPUS-42"
        if name == "add": return str(int(args.get("a", 0)) + int(args.get("b", 0)))
        raise ValueError(name)
    def send(o): sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        m = json.loads(line); method = m.get("method"); mid = m.get("id")
        if method == "initialize":
            pv = (m.get("params") or {}).get("protocolVersion", "2025-06-18")
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": pv, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "dogfood-raw", "version": "1.0.0"}}})
        elif method in ("notifications/initialized", "initialized"):
            pass
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            p = m.get("params") or {}
            try:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": dispatch(p.get("name"), p.get("arguments") or {})}],
                    "isError": False}})
            except Exception as e:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": str(e)}], "isError": True}})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": mid, "result": {}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "method not found"}})
    """
)


@pytest.mark.asyncio
async def test_real_stdio_server_discovers_and_calls_tools(tmp_path):
    """End-to-end: real subprocess stdio server + real mcp SDK.

    Exercises connect → initialize → tools/list → tools/call through the whole
    client against whatever mcp version is installed. Bypasses the command
    allowlist via ``skip_security_validation`` because the test runner's
    interpreter basename (e.g. ``python3.12``) is not on the allowlist.
    """
    pytest.importorskip("mcp")
    server = tmp_path / "stdio_server.py"
    server.write_text(_STDIO_SERVER)

    client = MCPClient(
        MCPServerConfig(
            name="dogfood",
            transport=MCPTransport.STDIO,
            command=sys.executable,
            args=[str(server)],
            skip_security_validation=True,
        )
    )
    try:
        connected = await client.connect()
        assert connected is True, f"connect failed: {client.get_status()}"
        names = sorted(t.name for t in client.tools)
        assert names == ["add", "get_secret_word"]
        # tool schema survives the discovery path
        add = next(t for t in client.tools if t.name == "add")
        assert add.input_schema.get("properties", {}).keys() == {"a", "b"}
        # and a real call round-trips
        result = await client.call_tool("add", {"a": 20, "b": 22})
        assert result.is_error is False
        assert "42" in str(result.content)
    finally:
        await client.disconnect()
