# SPDX-License-Identifier: Apache-2.0
"""Bounded, human-readable tool presentation owned by the runtime.

The wire retains original evidence separately. Clients render these fields without
parsing tool JSON, inferring action results, or accessing the workspace.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from .progress import tool_progress

MAX_DETAILS = 8
MAX_LINE = 240


def _line(value, limit=MAX_LINE):
    # Keep presentation independent of a particular terminal or markup renderer.
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def tool_label(name, args, workspace=None):
    labels = {
        "read_file": "Read",
        "read_attachment": "Read",
        "write_file": "Edit",
        "patch": "Edit",
        "terminal": "Run",
        "execute_code": "Run Python",
        "weather": "Weather",
        "web_search": "Search",
        "web_fetch": "Fetch",
        "grep": "Search",
        "glob": "Find",
        "delegate_task": "Delegate",
    }
    target = ""
    if name in {"read_file", "write_file", "patch"}:
        target = str(args.get("path") or "")
        if workspace:
            try:
                target = str(Path(target).relative_to(workspace))
            except ValueError:
                pass
    elif name == "read_attachment":
        target = args.get("title") or args.get("attachment_id", "attachment")
    elif name == "terminal":
        target = args.get("command", "")
    elif name == "weather":
        target = args.get("location", "")
    elif name == "web_search":
        target = args.get("query", "")
    elif name == "web_fetch":
        try:
            target = urlsplit(str(args.get("url", ""))).hostname or "web page"
        except ValueError:
            target = "web page"
    elif name in {"grep", "glob"}:
        target = args.get("pattern", "")
        if args.get("path") not in {None, "."}:
            target = f"{target}  {args['path']}"
    elif name == "delegate_task":
        target = args.get("task", "")
    return {
        "label": labels.get(name, _line(name.replace("_", " ")).capitalize()),
        "target": _line(target, 160),
    }


def permission_presentation(event):
    name = event.get("tool", "")
    title, language = {
        "execute_code": ("Run Python code?", "python"),
        "terminal": ("Run this command?", "bash"),
        "write_file": ("Apply this file edit?", "diff"),
        "patch": ("Apply this file edit?", "diff"),
    }.get(name, ("Allow this action?", "text"))
    return {**event, "title": title, "preview_language": language}


def _weather_summary(result):
    temperature = re.search(r"^\s*Temperature:\s*(.+)$", result, re.M)
    conditions = re.search(r"^\s*Conditions:\s*(.+)$", result, re.M)
    daily = re.search(r"\(today\):[^\n]*?,\s*(-?[\d.]+)–(-?[\d.]+)(°[CF])", result)
    if not temperature or not conditions:
        return _line(result.splitlines()[0] if result else "No weather data")
    parts = [temperature[1], conditions[1]]
    if daily:
        parts.append(f"H {daily[2]}° L {daily[1]}°")
    return " · ".join(parts)


def result_presentation(name, result, *, outcome="success", preview="", sources=()):
    """Summarize typed tool evidence, never expose raw JSON as a fallback."""
    details = []
    kind = "text"
    omitted = 0
    try:
        data = json.loads(result)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict) and data.get("error"):
        error = data["error"]
        summary = (
            error.get("message", "Tool could not complete")
            if isinstance(error, dict)
            else "Tool could not complete"
        )
    elif outcome == "error" and name not in {"terminal", "execute_code"}:
        summary = _line(result) if data is None else "Tool could not complete"
    elif name in {"write_file", "patch"}:
        changes = re.search(r"\(\+\d+ [−-]\d+\)", result)
        summary = changes[0] if changes else "File updated"
        lines = [
            line for line in preview.splitlines() if not line.startswith(("---", "+++"))
        ]
        # File headers duplicate the target; preserve hunk markers and context.
        details = lines[:MAX_DETAILS]
        omitted = max(0, len(lines) - len(details))
        kind = "diff"
    elif name in {"terminal", "execute_code"}:
        lines = result.splitlines()
        header = lines.pop(0) if lines else ""
        timing = re.search(r"\(exit [^)]*\)", header)
        output = [line for line in lines if line.strip()]
        excerpt = " … ".join(
            _line(line, 100)
            for line in ([output[0], output[-1]] if len(output) > 1 else output)
        )
        summary = " ".join(
            p for p in (excerpt, timing[0] if timing else _line(header)) if p
        )
        if len(output) > 2:
            details = [output[0], output[-1]]
            omitted = len(output) - 2
        if "timed out" in header:
            summary += " — timed out; effects unknown"
        kind = "output"
    elif name == "read_file" and isinstance(data, dict):
        count = sum(
            bool(re.match(r"^\d+\|", line))
            for line in str(data.get("content", "")).splitlines()
        )
        summary = f"{count} line{'s' if count != 1 else ''}"
        if data.get("truncated"):
            summary += f" (of {data.get('total_lines', '?')}; more available)"
    elif name in {"grep", "glob"} and isinstance(data, dict):
        count = len(data.get("matches", []))
        summary = f"{count} result{'s' if count != 1 else ''}"
        if data.get("truncated"):
            summary += " (more available)"
    elif name == "web_search":
        count = len(re.findall(r"^\[\d+\] ", result, re.M))
        summary = f"{count} result{'s' if count != 1 else ''}"
    elif name == "web_fetch":
        title = re.search(r"^\[\d+\]\s+(.+)$", result, re.M)
        summary = (
            title[1]
            if title
            else next(
                (s.get("title", "Page retrieved") for s in sources), "Page retrieved"
            )
        )
    elif name == "weather":
        summary = _weather_summary(result)
    elif name == "delegate_task":
        if isinstance(data, dict):
            summary = (
                "Helper completed"
                if data.get("status") == "completed"
                else _line(data.get("summary", "Helper finished"))
            )
        else:
            summary = "Helper completed"
    elif name == "read_attachment":
        count = len(result.splitlines())
        summary = f"{count} line{'s' if count != 1 else ''}"
    elif isinstance(data, dict):
        summary = data.get("summary") or "Completed"
    elif data is not None:
        summary = "Completed"
    else:
        summary = result.splitlines()[0] if result else "Completed"
    return {
        "summary": _line(summary),
        "details": [
            _line(line) if kind != "diff" else line[:MAX_LINE] for line in details
        ],
        "details_kind": kind,
        "omitted_lines": omitted,
    }


class ToolPresentation:
    """Per-turn presentation state; call IDs bind concurrent results to starts."""

    def __init__(self, workspace=None):
        self.workspace = workspace
        self.calls = {}
        self.sources = {}

    def event(self, event, operation=None):
        kind = event["type"]
        call_id = event.get("call_id", "")
        if kind == "tool.started":
            operation = operation or {}
            args = dict(event.get("args", {}))
            if operation and event["name"] in {"write_file", "patch"}:
                match = re.fullmatch(
                    r"Edited (.+) \(\+\d+ [−-]\d+\)", operation.get("summary", "")
                )
                if match:
                    args["path"] = match[1]
            elif operation and event["name"] == "terminal":
                args["command"] = operation.get("summary", "")
            labels = tool_label(event["name"], args, self.workspace)
            self.calls[call_id] = {
                "name": event["name"],
                "preview": operation.get("preview", ""),
                **labels,
            }
            return {
                **event,
                **labels,
                "summary": event.get("summary") or tool_progress(event["name"], args),
            }
        if kind == "source":
            self.sources.setdefault(call_id, []).append(event)
        if kind == "permission.request":
            return permission_presentation(event)
        if kind == "tool.result":
            call = self.calls.get(call_id, {})
            presentation = result_presentation(
                call.get("name", ""),
                event.get("result", ""),
                outcome=event.get("outcome", "success"),
                preview=call.get("preview", ""),
                sources=self.sources.get(call_id, ()),
            )
            return {
                **event,
                "label": call.get("label", "Tool"),
                "target": call.get("target", ""),
                **presentation,
            }
        return event
