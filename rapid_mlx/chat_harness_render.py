# SPDX-License-Identifier: Apache-2.0
"""Streaming Markdown and transient progress for the harness terminal."""

from __future__ import annotations

import io
import os
import shutil
import sys
from collections.abc import Iterable
from typing import TextIO
from urllib.parse import quote, urlsplit

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.segment import Segment
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text
from rich.theme import Theme

from rapid_mlx.chat_render import (
    _console,
    _markdown,
    supports_rich_output,
    terminal_safe_text,
)
from rapid_mlx.chat_runtime.progress import tool_progress as owner_tool_progress
from rapid_mlx.chat_state import TurnState


def safe_markdown(text: str) -> str:
    """Keep Markdown whitespace while removing terminal control sequences."""
    return "\n".join(
        terminal_safe_text(line.replace("\t", "    ")) for line in text.split("\n")
    )


def tool_progress(tool: dict) -> str:
    """Render owner labels safely, including optional wire-v1 fields."""
    if tool.get("label"):
        label = terminal_safe_text(str(tool["label"]))
        target = terminal_safe_text(str(tool.get("target", "")))
        return label + ("  " + target if target else "")
    label = terminal_safe_text(
        tool.get("summary") or owner_tool_progress(tool["name"], tool.get("args", {}))
    )[:180].removesuffix("…")
    status = tool.get("status", "running")
    if status == "running":
        return label + "…"
    outcome = "done" if status in {"success", "completed"} else status
    return f"{label} · {terminal_safe_text(outcome)}"


def approval_card(data: dict, *, expanded: bool = False, width: int = 80) -> str:
    """Render owner-provided approval copy; selection stays in the input editor."""
    preview = safe_markdown(str(data.get("preview", "")))
    lines = preview.splitlines()
    visible = lines if expanded else lines[:6]
    content = [Text(terminal_safe_text(str(data.get("reason", ""))), style="dim")]
    if visible:
        content.append(
            Syntax(
                "\n".join(visible),
                data.get("preview_language", "text"),
                theme="ansi_dark",
                background_color="default",
                line_numbers=True,
                word_wrap=True,
            )
        )
    if len(visible) < len(lines):
        content.append(
            Text(
                f"… {len(lines) - len(visible)} more lines  (v to view all)",
                style="dim",
            )
        )
    elif expanded and len(lines) > 6:
        content.append(Text("v to collapse", style="dim"))
    output = io.StringIO()
    console = Console(
        file=output,
        width=max(20, width),
        force_terminal=True,
        color_system=None if "NO_COLOR" in os.environ else "standard",
        highlight=False,
    )
    console.print(
        Panel(
            Group(*content),
            title=Text(
                terminal_safe_text(str(data.get("title") or "Allow this action?")),
                style="bold",
            ),
            title_align="left",
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )
    return output.getvalue()


def subagent_progress(child: dict) -> str:
    """One bounded child row; detailed activity disappears when it finishes."""
    title = terminal_safe_text(child["title"])[:110].removesuffix("…")
    count = child["tool_count"]
    tools = f"{count} tool{'s' if count != 1 else ''}"
    status = child["status"]
    if status in {"queued", "running"}:
        suffix = " · queued" if status == "queued" else ""
        return f"  ↳ {title}… ({tools}){suffix}"
    status = {"completed": "done", "budget_exhausted": "budget reached"}.get(
        status, status
    )
    return f"  ↳ {title} · {terminal_safe_text(status)} ({tools})"


def progress_row(text: str, style: str = "dim") -> Text:
    return Text(text, style=style, no_wrap=True, overflow="ellipsis")


def source_lines(sources: Iterable[dict], *, markdown: bool = False) -> list[str]:
    """Render source URLs without allowing terminal or Markdown injection."""
    lines = []
    for source in sources:
        url = source.get("url")
        if not isinstance(url, str) or any(not c.isprintable() for c in url):
            continue
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            if parsed.username is not None or parsed.password is not None:
                continue
        except ValueError:
            continue
        number = terminal_safe_text(str(source["source_id"]))
        title = terminal_safe_text(str(source["title"]))
        if markdown:
            for char in "\\`*_{}[]<>()!#":
                title = title.replace(char, "\\" + char)
                number = number.replace(char, "\\" + char)
            url = "<" + quote(url, safe=":/?#[]@!$&'()*+,;=%~_-") + ">"
            lines.append(f"\\[{number}\\] {title} — {url}")
        else:
            lines.append(f"[{number}] {title} — {url}")
    return lines


class _Preview:
    """Keep redraws inside the viewport, including Live's final refresh."""

    def __init__(self, renderable):
        self.renderable = renderable

    def __rich_console__(self, console, options):
        lines = console.render_lines(self.renderable, options, pad=False)
        # Leave room for the cursor/newline; no preview row may scroll offscreen
        # before it can be erased. Show the newest text on long answers.
        height = max(1, console.size.height - 2)
        lines = lines[-height:]
        for index, line in enumerate(lines):
            if index:
                yield Segment.line()
            yield from line


class HarnessRenderer:
    def __init__(
        self, state: TurnState, *, stream: TextIO | None = None, expanded: bool = False
    ):
        self.state = state
        self.stream = stream or sys.stdout
        self.expanded = expanded
        self.live = None
        self.terminal = bool(getattr(self.stream, "isatty", lambda: False)())
        self.waiting = Spinner("dots", style="dim")
        self._plain_thinking = False
        self._finished = False
        self._committed_timeline = 0
        self._committed_reasoning = 0
        self._committed_children: set[str] = set()
        self._last_started_id = None
        if supports_rich_output(self.stream):
            console = _console(self.stream)
            console.push_theme(
                Theme(
                    {
                        **{f"markdown.h{level}": "bold" for level in range(1, 7)},
                        "markdown.h1.border": "dim",
                        "markdown.strong": "bold",
                    }
                )
            )
            self.live = Live(
                _Preview(self.view()),
                console=console,
                refresh_per_second=20,
                vertical_overflow="crop",
                transient=True,
            )
            self.live.start()
        elif self.terminal:
            self.stream.write("…\n")
            self.stream.flush()

    def thought_summary(self):
        return f"Thought for {self.state.reasoning_seconds:.1f} s"

    def view(self):
        items = []
        if len(self.state.reasoning) > self._committed_reasoning:
            label = (
                "Thinking…"
                if self.state.reasoning_since is not None
                else self.thought_summary()
            )
            items.append(Text(label, style="dim"))
            if self.expanded:
                items.append(Text(safe_markdown(self.state.reasoning), style="dim"))
        shown_children = set(self._committed_children)
        timeline = self.state.timeline
        last_started = next(
            (
                key
                for kind, key in reversed(timeline[: self._committed_timeline])
                if kind == "tool"
            ),
            None,
        )
        for kind, key in timeline[self._committed_timeline :]:
            if kind == "tool":
                last_started = key
                tool = self.state.tools[key]
                items.append(progress_row("⏺ " + tool_progress(tool)))
                for child in self.state.subagents.values():
                    if child["call_id"] == key and child["id"] not in shown_children:
                        items.extend(self.child_view(child))
                        shown_children.add(child["id"])
            elif kind == "result":
                items.extend(
                    self.result_view(
                        self.state.tools[key], identify=last_started != key
                    )
                )
            elif kind == "text":
                items.append(_markdown(safe_markdown(self.state.parts[key])))
            elif kind == "notice":
                items.append(
                    Text(terminal_safe_text(self.state.notices[key]), style="dim")
                )
        for child in self.state.subagents.values():
            if child["id"] not in shown_children:
                items.extend(self.child_view(child))
        if self.state.terminal or self._finished:
            items.extend(
                Text(line, style="dim")
                for line in source_lines(self.state.sources.values())
            )
        if not items and not self.state.terminal and not self._finished:
            items.append(self.waiting)
        return Group(*items)

    def result_view(self, tool, *, identify=False):
        summary = terminal_safe_text(
            str(tool.get("summary") or tool.get("status", "done"))
        )
        if identify:
            summary = tool_progress(tool) + " — " + summary
        rows = [
            progress_row(
                "  ⎿ " + summary,
                style="red" if tool.get("status") == "error" else "dim",
            )
        ]
        for line in tool.get("details", []):
            line = terminal_safe_text(str(line))
            style = "dim"
            if tool.get("details_kind") == "diff":
                style = (
                    "green"
                    if line.startswith("+")
                    else "red"
                    if line.startswith("-")
                    else "dim"
                )
            rows.append(progress_row("    " + line, style))
        if tool.get("omitted_lines"):
            rows.append(Text(f"    … {tool['omitted_lines']} more lines", style="dim"))
        return rows

    def child_view(self, child):
        rows = [progress_row(subagent_progress(child))]
        if (
            self.expanded
            and child["status"] in {"queued", "running"}
            and child["summary"]
        ):
            rows.append(
                progress_row("    " + terminal_safe_text(child["summary"])[:160])
            )
        return rows

    def plain_row(self, text):
        if not self.terminal:
            return text
        row = Text(text)
        row.truncate(shutil.get_terminal_size().columns, overflow="ellipsis")
        return row.plain

    def pause(self):
        if self.live:
            self.live.update(Text(""), refresh=True)
            self.live.stop()
            self.live.console.print(self.view())
            self._committed_timeline = len(self.state.timeline)
            self._committed_reasoning = len(self.state.reasoning)
            self._committed_children.update(
                child["id"]
                for child in self.state.subagents.values()
                if child["status"] not in {"queued", "running"}
            )

    def resume(self):
        if self.live:
            self.live.start()
            self.live.update(_Preview(self.view()), refresh=True)

    def update(self, event: dict | None = None):
        if self.live:
            self.live.update(_Preview(self.view()), refresh=True)
        elif event:
            data = event["event"]
            kind = data["type"]
            if self.terminal:
                if self._plain_thinking and self.state.reasoning_since is None:
                    self.stream.write(self.thought_summary() + "\n")
                    self._plain_thinking = False
                elif (
                    self.state.reasoning_since is not None and not self._plain_thinking
                ):
                    self.stream.write("Thinking…\n")
                    self._plain_thinking = True
            if kind == "text.delta":
                self.stream.write(safe_markdown(data["text"]))
                self.stream.flush()
            elif kind in {"subagent.started", "subagent.finished"}:
                self.stream.write(self.plain_row(subagent_progress(data)) + "\n")
                self.stream.flush()
            elif kind == "tool.started":
                self._last_started_id = data["call_id"]
                self.stream.write(
                    "\n" + self.plain_row("⏺ " + tool_progress(data)) + "\n"
                )
                self.stream.flush()
            elif kind == "tool.result":
                for row in self.result_view(
                    self.state.tools[data["call_id"]],
                    identify=self._last_started_id != data["call_id"],
                ):
                    self.stream.write(self.plain_row(row.plain) + "\n")
                self.stream.flush()
            elif kind in {"context", "usage"} and data.get("notice"):
                self.stream.write("\n" + terminal_safe_text(data["notice"]) + "\n")
                self.stream.flush()
            elif (
                kind == "reasoning.delta" and self.expanded and not data.get("omitted")
            ):
                self.stream.write(safe_markdown(data.get("text", "")))
                self.stream.flush()
            if self.terminal:
                self.stream.flush()

    def finish(self):
        if self._finished:
            return
        self._finished = True
        self.state.end_reasoning()
        if self.live:
            # Live.stop() switches overflow to "visible" and refreshes once.
            # Clear the bounded preview first, so that final refresh cannot
            # scroll a copy of it above the completed answer.
            self.live.update(Text(""), refresh=True)
            self.live.stop()
            self.live.console.print(self.view())
        else:
            if self._plain_thinking:
                self.stream.write(self.thought_summary() + "\n")
            self.stream.write("\n")
            for line in source_lines(self.state.sources.values()):
                self.stream.write(line + "\n")
            self.stream.flush()
