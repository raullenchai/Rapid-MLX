# SPDX-License-Identifier: Apache-2.0
"""Request-level tool authorization, independent of parser and tool choice."""

import logging

logger = logging.getLogger(__name__)


def declared_names(request) -> set[str]:
    original: set[str] | None = getattr(request, "_tool_authorization_names", None)
    if original is not None:
        return original
    if hasattr(request, "model_dump"):
        request = request.model_dump()
    names = set()
    for tool in (request or {}).get("tools") or []:
        if hasattr(tool, "model_dump"):
            tool = tool.model_dump()
        if isinstance(tool, dict):
            function = tool.get("function", tool)
            if isinstance(function, dict) and function.get("name"):
                names.add(function["name"])
    return names


def call_name(call):
    if hasattr(call, "model_dump"):
        call = call.model_dump()
    return call.get("function", call).get("name")


def resolve_tool_names(calls, request):
    """Resolve separator variants only when one declared name matches.

    Exact names win, including dotted names. Unknown and ambiguous names
    remain available for client error recovery; argument deltas pass through.
    """
    names = declared_names(request)
    if not names:
        return calls

    def canonical(name):
        return name.replace(".", "_").replace("/", "_")

    candidates: dict[str, set[str]] = {}
    for name in names:
        candidates.setdefault(canonical(name), set()).add(name)

    resolved = []
    for call in calls:
        name = call_name(call)
        matches = candidates.get(canonical(name), set()) if name else set()
        if name not in names and len(matches) == 1:
            target = next(iter(matches))
            logger.debug("Resolved tool name %r to declared tool %r", name, target)
            if hasattr(call, "model_copy"):
                function = call.function.model_copy(update={"name": target})
                call = call.model_copy(update={"function": function})
            elif "function" in call:
                call = {**call, "function": {**call["function"], "name": target}}
            else:
                call = {**call, "name": target}
        resolved.append(call)
    return resolved


def log_drop(name):
    logger.warning("Dropping call to undeclared tool %r", name)
