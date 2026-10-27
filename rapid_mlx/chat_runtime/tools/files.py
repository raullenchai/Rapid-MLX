# SPDX-License-Identifier: Apache-2.0
"""Read-only file tool adapters; each operation has a cancellable worker."""

from __future__ import annotations

import json

from pydantic_ai import RunContext

from ..file_worker import run_worker, serialized_read
from . import TurnDeps, emit_result

TOOL_TIMEOUT = 10


@serialized_read
async def _execute(ctx, name, args) -> str:
    call_id = ctx.tool_call_id or ""
    await ctx.deps.emit(
        {"type": "tool.started", "call_id": call_id, "name": name, "args": args}
    )
    cap = min(18000, max(0, ctx.deps.read_budget_bytes - 512))
    payload = json.dumps(
        {
            "operation": name,
            "workspace": ctx.deps.workspace,
            "args": args,
            "output_cap": cap,
        }
    ).encode()
    result = await run_worker(
        "rapid_mlx.chat_runtime.file_operations", payload, timeout=TOOL_TIMEOUT
    )
    encoded = json.dumps(result, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > max(1024, cap):
        encoded = json.dumps(
            {
                "error": {
                    "code": "context_limit",
                    "message": "Result metadata exceeds the context budget; narrow the request or start a new turn.",
                }
            }
        )
        result = json.loads(encoded)
    ctx.deps.read_budget_bytes = max(
        0, ctx.deps.read_budget_bytes - len(encoded.encode("utf-8")) - 256
    )
    return await emit_result(
        ctx.deps, call_id, encoded, "error" if "error" in result else "success"
    )


async def read_file(
    ctx: RunContext[TurnDeps], path: str, offset: int = 1, limit: int = 200
) -> str:
    """Read numbered lines from a workspace text/code, PDF, DOCX, XLSX/CSV, PPTX, JSON/YAML or HTML file. Path may be absolute, relative to the workspace, or ~/path; it must resolve inside the workspace. Output format is LINE_NUM|CONTENT; the prefix is display metadata, not file content. Offset >=1; limit <=2000. Page with next_offset; no images."""
    return await _execute(
        ctx, "read_file", {"path": path, "offset": offset, "limit": limit}
    )


async def grep(
    ctx: RunContext[TurnDeps],
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    case: bool = True,
    max_results: int = 50,
) -> str:
    """Search workspace text by regex. Path may be absolute, relative to the workspace, or ~/path; it must resolve inside the workspace; case=True is case-sensitive, max_results 1..200. Honors .gitignore; skips binaries, secrets, symlinks and dependencies. Narrow path/glob when truncated."""
    return await _execute(
        ctx,
        "grep",
        {
            "pattern": pattern,
            "path": path,
            "glob": glob,
            "case": case,
            "max_results": max_results,
        },
    )


async def glob(ctx: RunContext[TurnDeps], pattern: str, path: str = ".") -> str:
    """Find up to 200 workspace files, newest first. Path may be absolute, relative to the workspace, or ~/path; it must resolve inside the workspace. Use **/*.md for recursion; honors .gitignore and excludes secrets/dependencies. Narrow path/pattern when truncated."""
    return await _execute(
        ctx, "glob", {"pattern": pattern, "path": path, "max_results": 200}
    )
