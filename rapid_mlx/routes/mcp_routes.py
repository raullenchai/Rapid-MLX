# SPDX-License-Identifier: Apache-2.0
"""MCP (Model Context Protocol) endpoints."""

from fastapi import APIRouter, Depends, HTTPException

from ..api.models import (
    MCPExecuteRequest,
    MCPExecuteResponse,
    MCPServerInfo,
    MCPServersResponse,
    MCPToolInfo,
    MCPToolsResponse,
)
from ..config import get_config
from ..mcp.security import MCPSecurityError, get_sandbox
from ..middleware.auth import verify_api_key, verify_api_key_or_x_api_key

router = APIRouter()

# Control-plane route (``/v1/mcp/reload``). It re-reads the config file and
# spawns whatever local processes it names, so it sits on the same gate the
# other destructive routes use (``admin_router`` in ``routes/health.py``) —
# Bearer OR ``x-api-key``, per the operator-intent revert of #728. Deliberately
# NOT the internal-header gate that revert removed: reintroducing it here would
# put this one route on a scheme nothing else in the tree uses.
admin_router = APIRouter(dependencies=[Depends(verify_api_key_or_x_api_key)])


@router.get("/v1/mcp/tools", dependencies=[Depends(verify_api_key)])
async def list_mcp_tools() -> MCPToolsResponse:
    """List all available MCP tools."""
    cfg = get_config()

    if cfg.mcp_manager is None:
        return MCPToolsResponse(tools=[], count=0)

    tools = []
    for tool in cfg.mcp_manager.get_all_tools():
        tools.append(
            MCPToolInfo(
                name=tool.full_name,
                description=tool.description,
                server=tool.server_name,
                parameters=tool.input_schema,
            )
        )

    return MCPToolsResponse(tools=tools, count=len(tools))


def _rejected_server_infos() -> list[MCPServerInfo]:
    """Render config entries the tolerant load dropped as error rows.

    Issue #1716: a server rejected by security validation never becomes a
    client, so it has no status to report and would simply be MISSING from
    the list — the user sees the row they added silently disappear. Listing
    it in ``error`` state with the validator's reason is what makes the
    rejection actionable.
    """
    cfg = get_config()
    return [
        MCPServerInfo(
            name=entry.name,
            state="error",
            transport=getattr(entry, "transport", "stdio"),
            tools_count=0,
            error=entry.error,
        )
        for entry in getattr(cfg, "mcp_rejected", []) or []
    ]


@router.get("/v1/mcp/servers", dependencies=[Depends(verify_api_key)])
async def list_mcp_servers() -> MCPServersResponse:
    """Get status of all MCP servers."""
    cfg = get_config()
    configured = getattr(cfg, "mcp_config_path", None) is not None
    init_error = getattr(cfg, "mcp_init_error", None)

    if cfg.mcp_manager is None:
        # Rejected entries still list — when the whole load failed there are
        # none, and ``error`` carries the reason instead.
        return MCPServersResponse(
            servers=_rejected_server_infos(),
            error=init_error,
            configured=configured,
        )

    servers = []
    for status in cfg.mcp_manager.get_server_status():
        servers.append(
            MCPServerInfo(
                name=status.name,
                state=status.state.value,
                transport=status.transport.value,
                tools_count=status.tools_count,
                error=status.error,
            )
        )

    servers.extend(_rejected_server_infos())

    return MCPServersResponse(
        servers=servers,
        error=init_error,
        configured=configured,
    )


@router.get("/v1/mcp/status", dependencies=[Depends(verify_api_key)])
async def get_mcp_status() -> MCPServersResponse:
    """Backward-compatible alias for ``/v1/mcp/servers``.

    The MCP tools guide documents ``/v1/mcp/status`` as the status endpoint,
    so this route prevents 404s for users following the docs.
    """
    return await list_mcp_servers()


@admin_router.post("/v1/mcp/reload")
async def reload_mcp_servers() -> MCPServersResponse:
    """Re-read the MCP config from disk and rebuild every connection.

    Issue #1716: the config was read exactly once, at process start, so the
    desktop app could only apply a connector edit by restarting the server —
    a multi-GB model reload for a one-line JSON change. This tears the
    manager down and brings it back from the same path.

    Failure is reported in the response body (``error``), not as a 5xx: a
    connector that won't start is a normal, user-fixable state, and the
    caller still wants the per-server rows to render alongside the reason.
    """
    from ..server import reload_mcp

    await reload_mcp()
    return await list_mcp_servers()


@router.post("/v1/mcp/execute", dependencies=[Depends(verify_api_key)])
async def execute_mcp_tool(request: MCPExecuteRequest) -> MCPExecuteResponse:
    """Execute an MCP tool."""
    cfg = get_config()

    if cfg.mcp_manager is None:
        raise HTTPException(
            status_code=503, detail="MCP not configured. Start server with --mcp-config"
        )

    # Server-side sandbox gate. The in-process tool loop runs this through
    # ``ToolExecutor``; this route does not, so without an explicit check the
    # default-deny on high-risk tools (shell/exec/eval), the argument-pattern
    # scrub, and the ``allowed_high_risk_tools`` allowlist wired up in
    # ``_start_mcp`` would all be inert here and the UI approval click would be
    # the sole gate (issue #1716). Validate against the SAME (server, tool)
    # split ``execute_tool`` will dispatch on.
    server_name, bare_tool = cfg.mcp_manager.resolve_tool_target(request.tool_name)
    if server_name is not None:
        try:
            get_sandbox().validate_tool_execution(
                bare_tool, server_name, request.arguments
            )
        except MCPSecurityError as exc:
            return MCPExecuteResponse(
                tool_name=request.tool_name,
                content=None,
                is_error=True,
                error_message=str(exc),
            )

    result = await cfg.mcp_manager.execute_tool(
        request.tool_name,
        request.arguments,
    )

    return MCPExecuteResponse(
        tool_name=result.tool_name,
        content=result.content,
        is_error=result.is_error,
        error_message=result.error_message,
    )
