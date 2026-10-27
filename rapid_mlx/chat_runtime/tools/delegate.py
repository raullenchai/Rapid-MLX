"""Depth-one delegation on the owner's inference endpoint and capability policy."""

from typing import cast

from pydantic_ai import RunContext

from . import TurnDeps


async def delegate_task(
    ctx: RunContext[TurnDeps],
    task: str,
    context: str | None = None,
    tools: list[str] | None = None,
) -> str:
    """Give a helper a focused task and explicit context. Helpers share your workspace and approvals, with only the requested tools (default: yours). Call up to three together for parallel work; combine returned summaries and sources."""
    if ctx.deps.subagent_id or ctx.deps.delegation is None:
        return "Delegation is unavailable: helpers cannot create other helpers."
    return cast(str, await ctx.deps.delegation.run(ctx, task, context, tools))
